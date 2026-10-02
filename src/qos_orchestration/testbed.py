"""Host-side control of the emulated testbed: agent commands and netem profiles."""

import json
import subprocess

from .paths import NETEM

CLIENT_CONTAINER = "qos-client"


def agent(cmd: dict):
    p = subprocess.run(
        ["docker", "exec", "-i", CLIENT_CONTAINER, "python", "-m", "qos_orchestration.rpc"],
        input=json.dumps(cmd).encode(),
        capture_output=True,
        timeout=1200,
    )
    if p.returncode:
        raise RuntimeError(p.stderr.decode()[:500])
    return json.loads(p.stdout)


def set_profile(path: str, rtt_ms=0, down_mbit=0, up_mbit=0, loss_pct=0, jitter_ms=0) -> str:
    """Apply netem delay/rate/loss to the 'rlan' (edge) or 'rwan' (cloud) router; returns the qdisc listing."""
    args = [str(NETEM / "profile.sh"), path, *(str(v) for v in (rtt_ms, down_mbit, up_mbit, loss_pct, jitter_ms))]
    return subprocess.run(args, capture_output=True, text=True, check=True).stdout.strip()
