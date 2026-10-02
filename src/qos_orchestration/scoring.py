"""Deterministic quality scores in [0, 1].

qa    NQ-open: a normalised reference answer occurs in the answer             {0, 1}
math  GSM8K: the final number equals the reference                            {0, 1}
code  HumanEval: the reference tests pass in a container without network      {0, 1}
long  IFEval long-form subset: share of verifiable instructions satisfied     [0, 1]
"""

import json
import re
import secrets
import string
import subprocess
from pathlib import Path

from langdetect import DetectorFactory

from .ifeval import instructions_registry

SANDBOX_IMAGE = "qos-net:1"
DetectorFactory.seed = 0  # language detection in the IFEval checkers is randomised otherwise
_PUNCT = set(string.punctuation)
_NUMBER = r"-?\$?\d[\d,]*(?:\.\d+)?"


def normalise(s):
    s = "".join(ch for ch in s.lower() if ch not in _PUNCT)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def score_qa(text, ref):
    t = normalise(text)
    return float(any(normalise(a) and normalise(a) in t for a in ref["answers"]))


def score_math(text, ref):
    explicit = re.findall(r"answer\s*[:：]\s*\**\s*(" + _NUMBER + ")", text, flags=re.I)
    candidate = explicit[-1] if explicit else (re.findall(_NUMBER, text) or [None])[-1]
    if candidate is None:
        return 0.0
    try:
        return float(abs(float(candidate.replace(",", "").replace("$", "")) - float(ref["answer"])) < 1e-6)
    except ValueError:
        return 0.0


def extract_code(text, entry_point):
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", text, flags=re.S)
    for b in blocks:
        if f"def {entry_point}" in b:
            return b
    return blocks[0] if blocks else text


def new_marker():
    """Random completion marker; generated code cannot print it to fake a pass."""
    return f"tests-passed-{secrets.token_hex(16)}"


def code_program(text, ref, marker):
    """Test program: the original prompt (helpers, imports) followed by the model's definition and the tests."""
    code = extract_code(text, ref["entry_point"])
    if f"def {ref['entry_point']}" in code:
        code = ref["prompt"] + "\n    pass\n\n" + code
    else:
        code = ref["prompt"] + code
    return code + "\n\n" + ref["test"] + f"\n\ncheck({ref['entry_point']})\nprint({marker!r})\n"


def run_programs(programs, marker):
    """Run programs in a throw-away container without network: 512 MB, 10 s per program. A program passes only if
    it exits normally after printing the completion marker, so exiting early cannot skip the tests."""
    runner = (Path(__file__).parent / "sandbox_runner.py").read_text()
    payload = json.dumps({"runner": runner, "programs": programs, "marker": marker})
    p = subprocess.run(
        [
            "docker",
            "run",
            "--rm",
            "-i",
            "--network",
            "none",
            "--memory",
            "512m",
            "--pids-limit",
            "256",
            SANDBOX_IMAGE,
            "python",
            "-c",
            "import json,sys;d=json.load(sys.stdin);exec(d['runner'])",
        ],
        input=payload.encode(),
        capture_output=True,
        timeout=3600,
    )
    if p.returncode:
        raise RuntimeError(p.stderr.decode()[-2000:])
    return json.loads(p.stdout)


def score_long(text, ref, prompt=""):
    """Returns (share of instructions followed, 1.0 if all were followed)."""
    followed = []
    for iid, kwargs in zip(ref["instruction_id_list"], ref["kwargs"], strict=True):
        inst = instructions_registry.INSTRUCTION_DICT[iid](iid)
        inst.build_description(**{k: v for k, v in (kwargs or {}).items() if v is not None})
        if "prompt" in (inst.get_instruction_args() or {}):
            inst.build_description(prompt=prompt)
        followed.append(bool(text.strip()) and inst.check_following(text))
    return sum(followed) / len(followed), float(all(followed))


def score_records(records, items):
    """Add `quality` (and `q_strict` for long-form, `code_err` for code) to measurement records in place."""
    pending_code = []
    for r in records:
        it, text = items[r["item_id"]], r.get("text") or ""
        r["quality"], r["q_strict"] = None, None
        if r.get("error") or not text:
            r["quality"] = 0.0
        elif it["cls"] == "qa":
            r["quality"] = score_qa(text, it["ref"])
        elif it["cls"] == "math":
            r["quality"] = score_math(text, it["ref"])
        elif it["cls"] == "long":
            r["quality"], r["q_strict"] = score_long(text, it["ref"], it["prompt"])
        else:
            pending_code.append(r)
    if pending_code:
        marker = new_marker()
        programs = [code_program(r["text"], items[r["item_id"]]["ref"], marker) for r in pending_code]
        results = run_programs(programs, marker)
        for r, res in zip(pending_code, results, strict=True):
            r["quality"], r["code_err"] = float(res["passed"]), res["err"]
    return records
