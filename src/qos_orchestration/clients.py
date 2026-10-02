"""Streaming LLM clients that record client-side timing and byte counts.

All times are `time.perf_counter()` offsets in seconds from the moment the request is issued.
"""

import json
import os
import time

import httpx

TARGETS = {
    # name: (API flavour, base-URL environment variable, default base URL, model)
    "local": ("ollama", "LOCAL_URL", "http://127.0.0.1:11434", "qwen2.5:3b-instruct"),
    "edge": ("ollama", "EDGE_URL", "http://host.docker.internal:11434", "qwen2.5:7b-instruct"),
    "cloud": ("openai", "CLOUD_URL", "https://openrouter.ai/api/v1", "qwen/qwen-2.5-72b-instruct"),
}
CLOUD_PROVIDER = os.environ.get("CLOUD_PROVIDER", "DeepInfra")
SEED = 42


class ConnectionTrace:
    """httpx trace hook collecting connection set-up timestamps (present only for new connections)."""

    def __init__(self, t0):
        self.t0, self.events = t0, {}

    def __call__(self, name, info):
        if name.endswith((".started", ".complete")):
            self.events.setdefault(name, time.perf_counter() - self.t0)


def _ollama_body(model, prompt, max_tokens, options=None):
    opts = {"temperature": 0, "seed": SEED, "num_predict": max_tokens, "num_ctx": 4096, **(options or {})}
    return {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
        "keep_alive": "60m",
        "options": opts,
    }


def _openai_body(model, prompt, max_tokens):
    return {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
        "temperature": 0,
        "seed": SEED,
        "max_tokens": max_tokens,
        "provider": {"order": [CLOUD_PROVIDER], "allow_fallbacks": False},
        "usage": {"include": True},
    }


def _parse_line(api, line, rec):
    """Return (text piece, final object or None) for one line of the response stream."""
    if api == "ollama":
        obj = json.loads(line)
        return obj.get("message", {}).get("content", ""), (obj if obj.get("done") else None)
    if not line.startswith(b"data:"):
        return "", None
    data = line[5:].strip()
    if data == b"[DONE]":
        return "", None
    obj = json.loads(data)
    if "error" in obj:
        rec["error"] = obj["error"]
    rec.setdefault("gen_id", obj.get("id"))
    if obj.get("provider"):
        rec["provider"] = obj["provider"]
    choice = (obj.get("choices") or [{}])[0]
    piece = (choice.get("delta") or {}).get("content") or ""
    return piece, (obj if obj.get("usage") else None)


def run_request(
    client: httpx.Client,
    target: str,
    prompt: str,
    max_tokens: int,
    api_key: str | None = None,
    base_url: str | None = None,
    options: dict | None = None,
) -> dict:
    api, env, default_url, model = TARGETS[target]
    base = base_url or os.environ.get(env, default_url)
    if api == "ollama":
        url, body, headers = base + "/api/chat", _ollama_body(model, prompt, max_tokens, options), {}
    else:
        url, body = base + "/chat/completions", _openai_body(model, prompt, max_tokens)
        headers = {"Authorization": f"Bearer {api_key}", "X-Title": "qos-llm-orchestration"}
    payload = json.dumps(body).encode()
    rec = {"target": target, "model": model, "url": url, "bytes_up_body": len(payload)}
    t0 = time.perf_counter()
    rec["wall_start"] = time.time()
    trace = ConnectionTrace(t0)
    state = {"text": [], "chunks": 0, "t_first": None, "t_last": None, "final": None}
    bytes_down = 0

    def consume(line):
        line = line.strip()
        if not line:
            return
        piece, final = _parse_line(api, line, rec)
        state["final"] = final or state["final"]
        if piece:
            now = time.perf_counter() - t0
            state["t_first"] = now if state["t_first"] is None else state["t_first"]
            state["t_last"] = now
            state["chunks"] += 1
            state["text"].append(piece)

    try:
        with client.stream(
            "POST",
            url,
            content=payload,
            headers={"Content-Type": "application/json", **headers},
            extensions={"trace": trace},
        ) as resp:
            rec["t_headers"] = time.perf_counter() - t0
            rec["status"] = resp.status_code
            rec["resp_headers_bytes"] = sum(len(k) + len(v) + 4 for k, v in resp.headers.raw)
            if resp.status_code != 200:
                body = resp.read()
                bytes_down = len(body)
                rec["error"] = f"HTTP {resp.status_code}: {body.decode(errors='replace')[:500]}"
            else:
                buf = b""
                for raw in resp.iter_raw():
                    bytes_down += len(raw)
                    *lines, buf = (buf + raw).split(b"\n")
                    for line in lines:
                        consume(line)
                consume(buf)
    except httpx.HTTPError as e:
        rec["error"] = f"{type(e).__name__}: {e}"[:500]
    except json.JSONDecodeError as e:
        rec["error"] = f"malformed stream: {e}"[:500]
    final = state["final"]
    if not rec.get("error") and final is None:
        rec["error"] = "incomplete stream: no final message"
    rec["t_end"] = time.perf_counter() - t0
    rec.update(
        {
            "t_first_token": state["t_first"],
            "t_last_token": state["t_last"],
            "n_chunks": state["chunks"],
            "bytes_down_body": bytes_down,
            "conn_events": trace.events,
            "text": "".join(state["text"]),
        }
    )
    if final is None:
        return rec
    if api == "ollama":
        rec.update(
            {
                k: final.get(k)
                for k in (
                    "prompt_eval_count",
                    "eval_count",
                    "total_duration",
                    "load_duration",
                    "prompt_eval_duration",
                    "eval_duration",
                )
            }
        )
        rec["n_in"], rec["n_out"], rec["finish"] = (
            final.get("prompt_eval_count"),
            final.get("eval_count"),
            final.get("done_reason"),
        )
    else:
        usage = final.get("usage", {})
        rec["n_in"], rec["n_out"], rec["cost_usd"] = (
            usage.get("prompt_tokens"),
            usage.get("completion_tokens"),
            usage.get("cost"),
        )
        rec["finish"] = (final.get("choices") or [{}])[0].get("finish_reason")
    if (rec.get("n_in") is None or rec.get("n_out") is None) and not rec.get("error"):
        rec["error"] = "missing token usage in final message"
    return rec


def make_client(keepalive=True, timeout=300.0):
    limits = httpx.Limits(max_keepalive_connections=4 if keepalive else 0, keepalive_expiry=600)
    return httpx.Client(timeout=timeout, limits=limits)
