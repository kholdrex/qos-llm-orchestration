"""Seeded task subsets from public Hugging Face datasets, fetched through the datasets-server rows API.

Writes data/tasks.jsonl (one task per line) and data/datasets_manifest.json (dataset revisions and row indices).
"""

import hashlib
import json
import random
import time
import urllib.request

from .paths import DATA

SEED = 20261001
N_PER_CLASS = 80
N_CALIB = 30

SOURCES = {
    "qa": ("google-research-datasets/nq_open", "nq_open", "validation"),
    "math": ("openai/gsm8k", "main", "test"),
    "code": ("openai/openai_humaneval", "openai_humaneval", "test"),
    "long": ("google/IFEval", "default", "train"),
}


def get_json(url, tries=5):
    for i in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                return json.load(r)
        except Exception:
            if i == tries - 1:
                raise
            time.sleep(min(60, 5 * 2**i))


def fetch_all(ds, cfg, split, revision):
    """All rows of a split, cached per dataset revision under data/raw/."""
    cache = DATA / "raw" / f"{ds.replace('/', '__')}__{cfg}__{split}__{revision[:12]}.json"
    if cache.exists():
        return json.loads(cache.read_text())
    info = get_json(f"https://datasets-server.huggingface.co/info?dataset={ds}")
    n = info["dataset_info"][cfg]["splits"][split]["num_examples"]
    rows = []
    for off in range(0, n, 100):
        page = get_json(
            f"https://datasets-server.huggingface.co/rows?dataset={ds}&config={cfg}&split={split}&offset={off}&length=100"
        )
        rows += [r["row"] for r in page["rows"]]
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(rows))
    return rows


def is_long_form(row):
    """IFEval prompts whose verifiable constraints require a long response."""
    for iid, kw in zip(row["instruction_id_list"], row["kwargs"], strict=True):
        if (
            iid == "length_constraints:number_words"
            and kw.get("relation") == "at least"
            and (kw.get("num_words") or 0) >= 150
        ):
            return True
        if iid == "length_constraints:number_paragraphs" and (kw.get("num_paragraphs") or 0) >= 3:
            return True
        if iid == "length_constraints:nth_paragraph_first_word" and (kw.get("num_paragraphs") or 0) >= 3:
            return True
        if (
            iid == "length_constraints:number_sentences"
            and kw.get("relation") == "at least"
            and (kw.get("num_sentences") or 0) >= 10
        ):
            return True
        if iid == "detectable_format:multiple_sections" and (kw.get("num_sections") or 0) >= 3:
            return True
    return False


def build_item(cls, row, idx):
    if cls == "qa":
        prompt = f"Answer the question with a short phrase only (no explanation).\nQuestion: {row['question']}?"
        ref = {"answers": row["answer"]}
        src_id = f"nq_open/validation/{idx}"
    elif cls == "math":
        prompt = (
            "Solve the math word problem. Reason step by step briefly, then give the final answer on the last line "
            f"in the form 'Answer: <number>'.\n\nProblem: {row['question']}"
        )
        ref = {"answer": row["answer"].split("####")[-1].strip().replace(",", "")}
        src_id = f"gsm8k/test/{idx}"
    elif cls == "code":
        prompt = (
            "Complete the following Python function. Return the complete function (including the signature and any "
            "needed imports) in a single ```python code block, without tests or explanations.\n\n" + row["prompt"]
        )
        ref = {"prompt": row["prompt"], "test": row["test"], "entry_point": row["entry_point"]}
        src_id = row["task_id"]
    else:
        prompt = row["prompt"]
        ref = {"instruction_id_list": row["instruction_id_list"], "kwargs": row["kwargs"], "key": row["key"]}
        src_id = f"ifeval/{row['key']}"
    return {"id": f"{cls}-{idx:05d}", "cls": cls, "source_id": src_id, "prompt": prompt, "ref": ref}


def sample_tasks():
    """Draw the seeded task sample from the current dataset revisions; returns (task lines, manifest)."""
    rng = random.Random(SEED)
    items, manifest = [], {"seed": SEED, "n_per_class": N_PER_CLASS, "n_calib": N_CALIB, "sources": {}}
    for cls, (ds, cfg, split) in SOURCES.items():
        revision = get_json(f"https://huggingface.co/api/datasets/{ds}").get("sha")
        rows = fetch_all(ds, cfg, split, revision)
        pool = list(range(len(rows)))
        if cls == "long":
            pool = [i for i in pool if is_long_form(rows[i])]
        chosen = rng.sample(pool, N_PER_CLASS)
        manifest["sources"][cls] = {
            "dataset": ds,
            "config": cfg,
            "split": split,
            "revision": revision,
            "rows": len(rows),
            "eligible": len(pool),
            "indices": chosen,
        }
        for k, i in enumerate(chosen):
            it = build_item(cls, rows[i], i)
            it["split"] = "calib" if k < N_CALIB else "test"
            items.append(it)
    lines = "".join(json.dumps(it, ensure_ascii=False) + "\n" for it in items)
    manifest["tasks_sha256"] = hashlib.sha256(lines.encode()).hexdigest()
    return lines, manifest


def fetch_tasks(overwrite=False):
    """Re-draw the sample and compare it with the published data/tasks.jsonl; replace it only with `overwrite`.

    The datasets-server API serves the latest dataset revision, so a changed upstream dataset shows up here as a
    hash or revision mismatch instead of silently replacing the published tasks. Returns True on a match or after a
    completed overwrite.
    """
    lines, manifest = sample_tasks()
    published = json.loads((DATA / "datasets_manifest.json").read_text())
    tasks_file = DATA / "tasks.jsonl"
    file_hash = hashlib.sha256(tasks_file.read_bytes()).hexdigest() if tasks_file.exists() else None
    for cls, src in manifest["sources"].items():
        recorded = published["sources"][cls]["revision"]
        if src["revision"] != recorded:
            print(f"{src['dataset']}: current revision {src['revision']}, published sample used {recorded}")
    if file_hash != published["tasks_sha256"]:
        print("data/tasks.jsonl is missing or does not match datasets_manifest.json")
    same = manifest["tasks_sha256"] == published["tasks_sha256"] == file_hash
    print("sample identical to data/tasks.jsonl" if same else "sample differs from data/tasks.jsonl")
    if overwrite and not same:
        tasks_file.write_text(lines)
        (DATA / "datasets_manifest.json").write_text(json.dumps(manifest, indent=1))
        print("data/tasks.jsonl and datasets_manifest.json replaced")
        return True
    return same


if __name__ == "__main__":
    fetch_tasks()
