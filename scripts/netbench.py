"""E1: HTTP transfer times over the emulated LAN path for the validation of the network model.

Grid: added RTT x symmetric rate limit x request/response size x connection reuse, five repetitions each.
Requires scripts/echo_server.py running on the host. Output: results/raw/e1_netbench.jsonl, e1_loadavg.jsonl.
"""

import itertools
import os
import random
import time

from qos_orchestration.experiment import EDGE_HOST, append_jsonl
from qos_orchestration.paths import RAW
from qos_orchestration.testbed import agent, set_profile

ECHO_PORT = 18090
RTTS_MS = [0, 20, 50, 100, 200, 300]
RATES_MBIT = [0, 50, 10, 2, 0.5]
PAYLOADS = [(1_000, 4_000), (16_000, 16_000), (128_000, 32_000), (2_000, 256_000)]
REPETITIONS = 5


def main():
    url = f"http://{EDGE_HOST}:{ECHO_PORT}/"
    out = RAW / "e1_netbench.jsonl"
    cells = list(itertools.product(RTTS_MS, RATES_MBIT))
    random.Random(11).shuffle(cells)
    base = agent({"op": "probe", "host": EDGE_HOST, "port": ECHO_PORT, "n": 10, "gap": 0.1})
    append_jsonl(out, {"kind": "base_probe", "tcp_connect_s": base, "wall": time.time()})
    for rtt, rate in cells:
        qdisc = set_profile("rlan", rtt, rate, rate)
        probe = agent({"op": "probe", "host": EDGE_HOST, "port": ECHO_PORT, "n": 3, "gap": 0.05})
        for (up, down), keepalive in itertools.product(PAYLOADS, (True, False)):
            times = agent({"op": "echo", "url": url, "up": up, "down": down, "n": REPETITIONS, "keepalive": keepalive})
            append_jsonl(
                out,
                {
                    "kind": "echo",
                    "rtt_ms": rtt,
                    "rate_mbit": rate,
                    "s_up": up,
                    "s_down": down,
                    "keepalive": keepalive,
                    "t_s": times,
                    "probe_s": probe,
                    "qdisc": qdisc,
                    "wall": time.time(),
                },
            )
        append_jsonl(
            RAW / "e1_loadavg.jsonl",
            {"rtt_ms": rtt, "rate_mbit": rate, "loadavg": os.getloadavg(), "wall": time.time()},
        )
        print(rtt, rate, flush=True)
    set_profile("rlan")


if __name__ == "__main__":
    main()
