#!/usr/bin/env bash
# Two-router network testbed inside the Docker VM:
#   qos-client (172.30.1.10) -> qos-rlan (172.30.1.2) -> host: edge server (Ollama)
#                            -> qos-rwan (172.30.1.3) -> Internet: cloud API
# netem on each router interface shapes one direction (towards the client = downlink, towards the core = uplink).
# EDGE_HOST overrides the edge server address (default: host.docker.internal as resolved inside the VM).
# Usage: topology.sh up|down
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
SRC=$HERE/../src
SECRETS=$HOME/.config/qos-llm
IMAGE=qos-net:1

if [[ ${1:-up} == down ]]; then
  docker rm -f qos-client qos-rlan qos-rwan >/dev/null 2>&1 || true
  docker network rm qos-access qos-core >/dev/null 2>&1 || true
  exit 0
fi

docker network create --subnet 172.30.1.0/24 qos-access >/dev/null
docker network create --subnet 172.30.2.0/24 qos-core >/dev/null
for router in rlan:2 rwan:3; do
  name=${router%%:*}; octet=${router##*:}
  docker run -d --name qos-$name --hostname qos-$name --cap-add NET_ADMIN --sysctl net.ipv4.ip_forward=1 \
    --network qos-core --ip 172.30.2.$octet $IMAGE sleep infinity >/dev/null
  docker network connect --ip 172.30.1.$octet qos-access qos-$name
  core_if=$(docker exec qos-$name sh -c "ip -o -4 addr | awk '/172\.30\.2\./{print \$2}'")
  docker exec qos-$name ip route replace default via 172.30.2.1 dev "$core_if"
  docker exec qos-$name iptables -t nat -A POSTROUTING -o "$core_if" -j MASQUERADE
done

edge_host=${EDGE_HOST:-$(docker run --rm $IMAGE getent hosts host.docker.internal | awk '{print $1}')}
if [[ -z $edge_host ]]; then
  echo "cannot resolve host.docker.internal; set EDGE_HOST to the edge server address" >&2
  exit 1
fi

mkdir -p "$SECRETS"
docker run -d --name qos-client --hostname qos-client --cap-add NET_ADMIN --network qos-access --ip 172.30.1.10 \
  -v "$SRC":/app/src:ro -v "$SECRETS":/secrets:ro -w /app \
  -e PYTHONPATH=/app/src -e EDGE_URL="http://$edge_host:11434" \
  $IMAGE python -u -m qos_orchestration.agent >/dev/null
docker exec qos-client ip route replace default via 172.30.1.3
docker exec qos-client ip route add "$edge_host"/32 via 172.30.1.2
if ! docker exec qos-client python -c "import urllib.request; urllib.request.urlopen('http://$edge_host:11434/api/version', timeout=5)"; then
  echo "edge server http://$edge_host:11434 is not reachable from the client container" >&2
  exit 1
fi
echo "testbed up; edge server $edge_host (export EDGE_HOST=$edge_host for the measurement scripts)"
