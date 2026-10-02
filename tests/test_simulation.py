import json

from qos_orchestration.simulation import load_measurements


def record(target, error=None, n_out=3):
    rec = {"item_id": "qa-1", "cls": "qa", "target": target, "n_out": n_out, "quality": 1.0}
    if error:
        rec["error"], rec["n_out"] = error, None
    return rec


def test_failed_retry_does_not_replace_success(tmp_path):
    log = tmp_path / "scored.jsonl"
    rows = [record("local"), record("edge"), record("cloud"), record("cloud", error="timeout")]
    log.write_text("".join(json.dumps(r) + "\n" for r in rows))
    data = load_measurements(log)
    assert set(data["qa-1"]) == {"local", "edge", "cloud"}
    assert not data["qa-1"]["cloud"].get("error")


def test_item_without_successful_cloud_record_is_dropped(tmp_path):
    log = tmp_path / "scored.jsonl"
    rows = [record("local"), record("edge"), record("cloud", error="HTTP 429")]
    log.write_text("".join(json.dumps(r) + "\n" for r in rows))
    assert load_measurements(log) == {}
