"""Provider-side statistics (latency, generation time, cost) of logged cloud requests from OpenRouter.

Only complete statistics are stored; requests without them are retried on the next run.
"""

import json
import time
from pathlib import Path

import httpx

from qos_orchestration.experiment import append_jsonl
from qos_orchestration.paths import RAW

KEY_FILE = Path.home() / ".config/qos-llm/openrouter.key"
FIELDS = (
    "latency",
    "generation_time",
    "provider_name",
    "total_cost",
    "native_tokens_prompt",
    "native_tokens_completion",
    "created_at",
)


def fetch(client, gid, headers, tries=5):
    for i in range(tries):
        resp = client.get("https://openrouter.ai/api/v1/generation", params={"id": gid}, headers=headers)
        if resp.status_code == 200:
            return resp.json().get("data")
        time.sleep(2 * (i + 1))
    return None


def main():
    out = RAW / "cloud_genstats.jsonl"
    have = {json.loads(line)["gen_id"] for line in out.open()} if out.exists() else set()
    headers = {"Authorization": f"Bearer {KEY_FILE.read_text().strip()}"}
    files = [RAW / "characterize_cloud.jsonl", *sorted(RAW.glob("live_cloud_*.jsonl"))]
    with httpx.Client(timeout=30) as client:
        for path in files:
            for r in map(json.loads, path.open()):
                gid = r.get("gen_id")
                if not gid or gid in have or r.get("error"):
                    continue
                data = fetch(client, gid, headers)
                if not data or data.get("generation_time") is None:
                    print("no statistics yet for", gid, flush=True)
                    continue
                have.add(gid)
                append_jsonl(
                    out, {"gen_id": gid, "file": path.name, "item_id": r["item_id"], **{k: data.get(k) for k in FIELDS}}
                )


if __name__ == "__main__":
    main()
