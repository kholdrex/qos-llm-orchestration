"""Execution-target selection policies.

Policies see only estimates: class-level quality and output length from calibration, service-time telemetry,
network-state estimates (RTT, bandwidth, round trips), the edge queue and prices. The realised outcome of the
routed request is never visible to them.
"""

from dataclasses import dataclass, field

from .netmodel import tx_simple, tx_slowstart

TARGETS = ("local", "edge", "cloud")


@dataclass
class Profile:
    """Estimates for one execution target."""

    q: dict  # class -> mean quality
    n_out: dict  # class -> median number of output tokens
    ttft0: float  # server-side time to first token at zero prompt length, s
    prefill: float  # s per prompt token
    decode: float  # s per output token after the first
    bytes_per_tok: float  # HTTP response bytes per output token
    price_in: float = 0.0  # USD per prompt token
    price_out: float = 0.0  # USD per output token


@dataclass
class NetState:
    rtt: dict  # target -> RTT, s
    b_up: dict  # target -> bit/s, 0 = not limiting
    b_down: dict
    queue: dict = field(default_factory=lambda: {m: 0.0 for m in TARGETS})  # expected queue wait, s
    k: dict = field(default_factory=dict)  # round trips before service: 1 warm, 2 TCP, 3 TCP + TLS


@dataclass
class Request:
    cls: str
    n_in: int
    s_up: int


def predict_latency(p: Profile, req: Request, net: NetState, m: str) -> float:
    """Expected end-to-end latency of `req` on target `m`, eqs. (5)-(6)."""
    n_decode = max(0.0, p.n_out[req.cls] - 1)
    first = p.ttft0 + req.n_in * p.prefill + net.queue.get(m, 0.0)
    generation = n_decode * p.decode
    if m == "local":
        return first + generation
    k = net.k.get(m, 1)
    upload = (tx_slowstart if k > 1 else tx_simple)(req.s_up, net.b_up[m], net.rtt[m])
    b_down = net.b_down[m]
    download = 8.0 * p.n_out[req.cls] * p.bytes_per_tok / b_down if b_down else 0.0
    return k * net.rtt[m] + upload + first + max(generation, download)


def predict_cost(p: Profile, req: Request) -> float:
    return p.price_in * req.n_in + p.price_out * p.n_out[req.cls]


def qos_aware(req, profiles, net, tau_q, tau_l, lam_l=1.0, lam_c=0.0, cost_ref=1e-3):
    """Cheapest target with estimated quality >= tau_Q and predicted latency <= tau_L, eq. (9); ties go to the
    lower latency. Without a feasible target, maximise U = Q - lam_L * L / tau_L - lam_C * C / C_ref, eq. (10)."""
    est = {
        m: (profiles[m].q[req.cls], predict_latency(profiles[m], req, net, m), predict_cost(profiles[m], req))
        for m in TARGETS
    }
    feasible = [m for m, (q, lat, _) in est.items() if q >= tau_q[req.cls] and lat <= tau_l[req.cls]]
    if feasible:
        return min(feasible, key=lambda m: (round(est[m][2], 9), est[m][1]))
    return max(TARGETS, key=lambda m: est[m][0] - lam_l * est[m][1] / tau_l[req.cls] - lam_c * est[m][2] / cost_ref)


def quality_only(req, profiles, net, **_):
    return max(TARGETS, key=lambda m: profiles[m].q[req.cls])


def latency_only(req, profiles, net, **_):
    return min(TARGETS, key=lambda m: predict_latency(profiles[m], req, net, m))


def quality_cost(req, profiles, net, tau_q, **_):
    """Cheapest target whose estimated class quality reaches tau_Q, ignoring latency."""
    feasible = [m for m in TARGETS if profiles[m].q[req.cls] >= tau_q[req.cls]]
    if not feasible:
        return quality_only(req, profiles, net)
    return min(feasible, key=lambda m: (predict_cost(profiles[m], req), TARGETS.index(m)))


def static_rtt(req, profiles, net, rtt_threshold=0.1, **_):
    """CLOUD while the measured cloud RTT is below a fixed threshold, otherwise LOCAL."""
    return "cloud" if net.rtt["cloud"] <= rtt_threshold else "local"


def fixed(target):
    return lambda req, profiles, net, **_: target
