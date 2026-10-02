"""E4: repeat the first five tasks of each class live under emulated impairments.

The measured latency is compared with the trace-driven prediction built from the same task's base-path
measurement (scripts/analyze.py). Output: results/raw/live_<target>_<condition>.jsonl.
"""

import argparse
import time

from qos_orchestration.experiment import append_jsonl, config, host_telemetry, load_tasks, successful_items
from qos_orchestration.paths import RAW
from qos_orchestration.testbed import agent, set_profile

CONDITIONS = {
    ("cloud", "moderate"): None,
    ("cloud", "high_latency"): None,
    ("cloud", "degraded"): None,
    ("edge", "edge_wan100"): {"rtt_ms": 100, "down_mbit": 10, "up_mbit": 10},
    ("cloud", "lowbw_cold"): {"rtt_ms": 250, "down_mbit": 0.032, "up_mbit": 0.016, "new_conn": True},
}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--target", choices=["edge", "cloud"])
    ap.add_argument("--condition", choices=sorted({c for _, c in CONDITIONS}))
    args = ap.parse_args()
    cfg = config()
    tasks = load_tasks(per_class=5)
    for (target, name), params in CONDITIONS.items():
        if (args.target and target != args.target) or (args.condition and name != args.condition):
            continue
        p = params or cfg["profiles_emulated"][name]["cloud"]
        path = "rwan" if target == "cloud" else "rlan"
        qdisc = set_profile(path, p["rtt_ms"], p["down_mbit"], p["up_mbit"])
        agent({"op": "reset"})
        agent({"op": "run", "target": target, "prompt": "Hi", "max_tokens": 4})
        out = RAW / f"live_{target}_{name}.jsonl"
        done = successful_items(out)
        for task in tasks:
            if task["id"] in done:
                continue
            before = host_telemetry()
            rec = agent(
                {
                    "op": "run",
                    "target": target,
                    "prompt": task["prompt"],
                    "max_tokens": cfg["max_tokens"][task["cls"]],
                    "keepalive": not p.get("new_conn", False),
                }
            )
            rec.update(
                {
                    "item_id": task["id"],
                    "cls": task["cls"],
                    "phase": "live",
                    "profile": name,
                    "profile_params": p,
                    "qdisc": qdisc,
                    "telemetry_before": before,
                    "telemetry_after": host_telemetry(),
                    "wall_end": time.time(),
                }
            )
            append_jsonl(out, rec)
            print(target, name, task["id"], round(rec["t_end"], 2), rec.get("error", ""), flush=True)
        set_profile(path)


if __name__ == "__main__":
    main()
