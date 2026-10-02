"""Shared settings and helpers of the measurement scripts."""

import json
import os
import urllib.request

from .paths import CONFIGS, DATA

EDGE_HOST = os.environ.get("EDGE_HOST", "192.168.5.2")  # edge server address as seen from the client container
CLOUD_HOST = "openrouter.ai"
OLLAMA_PORT = 11434


def config():
    return json.loads((CONFIGS / "experiment.json").read_text())


def load_tasks(per_class=None):
    """Tasks in their seeded order; with `per_class`, only the first N of each class."""
    items = [json.loads(line) for line in (DATA / "tasks.jsonl").open()]
    if per_class is None:
        return items
    taken, out = {}, []
    for it in items:
        if taken.get(it["cls"], 0) < per_class:
            out.append(it)
            taken[it["cls"]] = taken.get(it["cls"], 0) + 1
    return out


def host_telemetry():
    """Load average and models currently loaded by the host's Ollama, logged with every request."""
    t = {"loadavg": os.getloadavg()}
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{OLLAMA_PORT}/api/ps", timeout=3) as r:
            t["ollama_ps"] = [m["name"] for m in json.load(r).get("models", [])]
    except OSError as e:
        t["ollama_ps"] = f"error: {e}"
    return t


def is_success(rec):
    """A usable measurement: no error and token usage reported."""
    return not rec.get("error") and rec.get("n_out") is not None


def successful_items(path):
    if not path.exists():
        return set()
    return {r["item_id"] for r in map(json.loads, path.open()) if is_success(r)}


def append_jsonl(path, record):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
