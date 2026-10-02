"""Score measurement logs: results/raw/<pattern> -> results/<output>."""

import argparse
import json

from qos_orchestration.paths import DATA, RAW, RESULTS
from qos_orchestration.scoring import score_records


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pattern", default="characterize_*.jsonl")
    ap.add_argument("--output", default="scored_characterize.jsonl")
    args = ap.parse_args()
    items = {it["id"]: it for it in map(json.loads, (DATA / "tasks.jsonl").open())}
    records = [json.loads(line) for f in sorted(RAW.glob(args.pattern)) for line in f.open()]
    score_records(records, items)
    with (RESULTS / args.output).open("w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{len(records)} records -> {args.output}")


if __name__ == "__main__":
    main()
