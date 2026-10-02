"""E2: run every task on each execution target over the base (unimpaired) network paths.

Each target runs as one block so that the local and edge models never compete for the GPU. Results are appended
to results/raw/characterize_<target>.jsonl; finished tasks are skipped, failed ones are retried.
"""

import argparse
import os
import random
import time

from qos_orchestration.clients import make_client, run_request
from qos_orchestration.experiment import (
    CLOUD_HOST,
    EDGE_HOST,
    OLLAMA_PORT,
    append_jsonl,
    config,
    host_telemetry,
    load_tasks,
    successful_items,
)
from qos_orchestration.paths import RAW
from qos_orchestration.testbed import agent, set_profile


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--targets", default="local,edge,cloud")
    ap.add_argument("--per-class", type=int, default=25)
    ap.add_argument("--max-load", type=float, default=6.0, help="wait while the 1-min load average is higher")
    args = ap.parse_args()
    max_tokens = config()["max_tokens"]

    set_profile("rlan")
    set_profile("rwan")
    append_jsonl(
        RAW / "base_probes.jsonl",
        {
            "edge": agent({"op": "probe", "host": EDGE_HOST, "port": OLLAMA_PORT, "n": 10, "gap": 0.2}),
            "cloud": agent({"op": "probe", "host": CLOUD_HOST, "port": 443, "n": 10, "gap": 0.2}),
            "wall": time.time(),
        },
    )

    tasks = load_tasks(args.per_class)
    random.Random(7).shuffle(tasks)
    host_client = make_client(True)
    for target in args.targets.split(","):
        out = RAW / f"characterize_{target}.jsonl"
        done = successful_items(out)
        if target == "local":  # load the model before timing
            run_request(host_client, "local", "Hi", 4)
        else:
            agent({"op": "run", "target": target, "prompt": "Hi", "max_tokens": 4})
        for task in tasks:
            if task["id"] in done:
                continue
            while os.getloadavg()[0] > args.max_load:
                print("waiting for load", os.getloadavg(), flush=True)
                time.sleep(30)
            before = host_telemetry()
            n = max_tokens[task["cls"]]
            if target == "local":
                rec = run_request(host_client, "local", task["prompt"], n)
            else:
                rec = agent({"op": "run", "target": target, "prompt": task["prompt"], "max_tokens": n})
            rec.update(
                {
                    "item_id": task["id"],
                    "cls": task["cls"],
                    "phase": "characterize",
                    "telemetry_before": before,
                    "telemetry_after": host_telemetry(),
                }
            )
            append_jsonl(out, rec)
            print(task["id"], target, round(rec["t_end"], 2), rec.get("n_out"), rec.get("error", ""), flush=True)


if __name__ == "__main__":
    main()
