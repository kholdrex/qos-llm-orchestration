"""Re-draw the task sample (seed 20261001) and compare it with data/tasks.jsonl; --overwrite replaces it."""

import argparse

from qos_orchestration.tasks import fetch_tasks

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--overwrite", action="store_true")
    raise SystemExit(0 if fetch_tasks(ap.parse_args().overwrite) else 1)
