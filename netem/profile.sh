#!/usr/bin/env bash
# Apply an emulated link profile to one router.
# Usage: profile.sh <rlan|rwan> <added_rtt_ms> <down_mbit> <up_mbit> [loss_pct] [jitter_ms]
# Half of the added RTT is applied in each direction; a rate of 0 means no limit.
set -euo pipefail
router=qos-$1; rtt=$2; down=$3; up=$4; loss=${5:-0}; jitter=${6:-0}
half=$(python3 -c "print($rtt / 2)")
half_jitter=$(python3 -c "print($jitter / 2)")
access_if=$(docker exec "$router" sh -c "ip -o -4 addr | awk '/172\.30\.1\./{print \$2}'")
core_if=$(docker exec "$router" sh -c "ip -o -4 addr | awk '/172\.30\.2\./{print \$2}'")

apply() {
  local dev=$1 rate=$2 args="delay ${half}ms"
  [[ $jitter != 0 ]] && args="$args ${half_jitter}ms distribution normal"
  [[ $loss != 0 ]] && args="$args loss ${loss}%"
  [[ $rate != 0 ]] && args="$args rate ${rate}mbit"
  docker exec "$router" tc qdisc del dev "$dev" root >/dev/null 2>&1 || true
  docker exec "$router" tc qdisc add dev "$dev" root netem limit 100000 $args
}

apply "$access_if" "$down"
apply "$core_if" "$up"
docker exec "$router" sh -c "tc qdisc show | grep netem"
