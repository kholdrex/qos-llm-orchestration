import json

import httpx

from qos_orchestration.clients import run_request


def client_for(status, body):
    return httpx.Client(
        transport=httpx.MockTransport(lambda request: httpx.Response(status, stream=httpx.ByteStream(body)))
    )


def ollama_line(**obj):
    return json.dumps(obj).encode() + b"\n"


def test_http_error_is_recorded():
    rec = run_request(client_for(404, b'{"error": "model not found"}\n'), "local", "hi", 4)
    assert rec["error"].startswith("HTTP 404")


def test_stream_without_final_message_is_a_failure():
    rec = run_request(client_for(200, ollama_line(message={"content": "hello"}, done=False)), "local", "hi", 4)
    assert rec["text"] == "hello"
    assert rec["error"] == "incomplete stream: no final message"


def test_final_message_without_trailing_newline_is_parsed():
    body = (
        ollama_line(message={"content": "hi"}, done=False)
        + json.dumps(
            {"message": {"content": ""}, "done": True, "prompt_eval_count": 7, "eval_count": 2, "done_reason": "stop"}
        ).encode()
    )
    rec = run_request(client_for(200, body), "local", "hi", 4)
    assert not rec.get("error")
    assert (rec["n_in"], rec["n_out"], rec["finish"]) == (7, 2, "stop")


def test_malformed_stream_is_a_failure():
    rec = run_request(client_for(200, b"<html>bad gateway</html>\n"), "local", "hi", 4)
    assert rec["error"].startswith("malformed stream")


def test_server_sent_events_with_usage():
    events = [
        {"id": "gen-1", "choices": [{"delta": {"content": "ok"}}]},
        {
            "id": "gen-1",
            "choices": [{"delta": {}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 1, "cost": 1e-6},
        },
    ]
    body = b"".join(b"data: " + json.dumps(e).encode() + b"\n\n" for e in events) + b"data: [DONE]\n\n"
    rec = run_request(client_for(200, body), "cloud", "hi", 4, api_key="test")
    assert not rec.get("error")
    assert (rec["text"], rec["n_out"], rec["cost_usd"], rec["gen_id"]) == ("ok", 1, 1e-6, "gen-1")
