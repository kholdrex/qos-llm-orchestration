"""Measurement agent inside the client container; timings are taken here, behind the emulated links.

Run as `python -m qos_orchestration.agent`; the host sends JSON commands with `python -m qos_orchestration.rpc`.
"""

import json
import socket
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from .clients import make_client, run_request

KEY_FILE = Path("/secrets/openrouter.key")
_pool = {"client": make_client(True)}


def _api_key():
    return KEY_FILE.read_text().strip() if KEY_FILE.exists() else None


def tcp_connect_times(host, port, n, gap):
    out = []
    for _ in range(n):
        t0 = time.perf_counter()
        try:
            with socket.create_connection((host, port), timeout=10):
                out.append(time.perf_counter() - t0)
        except OSError:
            out.append(None)
        time.sleep(gap)
    return out


def echo_times(url, up, down, n, keepalive):
    """Durations of POSTing `up` bytes and receiving `down` bytes from the echo server."""
    body, res = b"x" * up, []
    warm = make_client(True) if keepalive else None
    if warm:
        warm.get(url + "?down=1")
    for _ in range(n):
        client = warm or make_client(False)
        t0 = time.perf_counter()
        client.post(f"{url}?down={down}", content=body).read()
        res.append(time.perf_counter() - t0)
        if not warm:
            client.close()
    if warm:
        warm.close()
    return res


def handle(cmd):
    op = cmd.pop("op")
    if op == "run":
        keepalive = cmd.pop("keepalive", True)
        client = _pool["client"] if keepalive else make_client(False)
        out = run_request(client, cmd["target"], cmd["prompt"], cmd["max_tokens"], _api_key(), cmd.get("base_url"))
        if not keepalive:
            client.close()
        return out
    if op == "probe":
        return tcp_connect_times(cmd["host"], cmd["port"], cmd.get("n", 5), cmd.get("gap", 0.2))
    if op == "echo":
        return echo_times(cmd["url"], cmd["up"], cmd["down"], cmd.get("n", 5), cmd.get("keepalive", True))
    if op == "reset":
        _pool["client"].close()
        _pool["client"] = make_client(True)
        return "ok"
    return {"error": f"unknown op {op}"}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_POST(self):
        cmd = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        data = json.dumps(handle(cmd)).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


if __name__ == "__main__":
    ThreadingHTTPServer(("0.0.0.0", 8080), Handler).serve_forever()
