"""Trace-driven evaluation of routing policies under emulated network conditions.

The latency of a request on a target under a network scenario is the measured server-side time of that request
plus the network time from the latency model; quality and API cost are the measured values of that request.
"""

import copy
import json
import statistics as st

import numpy as np

from . import policy as P
from .experiment import is_success
from .netmodel import tx_simple, tx_slowstart
from .paths import CONFIGS, RESULTS
from .policy import TARGETS, NetState, Profile, Request

CLASSES = ("qa", "math", "code", "long")
REQUEST_HEADER_BYTES = {"local": 0, "edge": 180, "cloud": 420}


def load_config():
    return json.loads((CONFIGS / "experiment.json").read_text())


def load_measurements(path=RESULTS / "scored_characterize.jsonl"):
    """Scored base-path records as {item_id: {target: record}}; the last successful record per pair wins."""
    data = {}
    for line in path.open():
        r = json.loads(line)
        if r["target"] in TARGETS and is_success(r):
            data.setdefault(r["item_id"], {})[r["target"]] = r
    return {i: d for i, d in data.items() if all(m in d for m in TARGETS)}


def decompose(r, base_rtt):
    """Split a measured request into network-independent parts, in seconds and bytes."""
    ev = r.get("conn_events", {})
    handshake = 0.0
    for phase in ("connect_tcp", "start_tls"):
        if f"connection.{phase}.complete" in ev:
            handshake += ev[f"connection.{phase}.complete"] - ev.get(f"connection.{phase}.started", 0.0)
    ttft = r["t_first_token"] if r.get("t_first_token") is not None else r["t_end"]
    last = r["t_last_token"] if r.get("t_last_token") is not None else ttft
    return {
        "srv_first": max(0.0, ttft - handshake - base_rtt),
        "gen": max(0.0, last - ttft),
        "tail": max(0.0, r["t_end"] - last),
        "s_up": r.get("bytes_up_body", 0) + REQUEST_HEADER_BYTES[r["target"]],
        "s_down": r.get("bytes_down_body", 0) + r.get("resp_headers_bytes", 0),
    }


def decode_time_per_token(r, base_rtt, min_tokens=20):
    parts = decompose(r, base_rtt)
    n = r["n_out"] or 0
    return parts["gen"] / (n - 1) if n > min_tokens and parts["gen"] > 0 else None


def scenario(cloud, edge=None, edge_queue_s=0.0, round_trips=None):
    """Emulated condition of the WAN (cloud) and LAN (edge) paths; rtt_ms is added to the measured base RTT."""
    sc = {"cloud": cloud, "edge": edge or {"rtt_ms": 0, "down_mbit": 0, "up_mbit": 0}, "edge_queue_s": edge_queue_s}
    if round_trips:
        sc["k"] = round_trips
    return sc


def realised(r, parts, m, sc, base):
    """End-to-end latency and time to first token of measured request `r` on target `m` under scenario `sc`."""
    if m == "local":
        return r["t_end"], (r["t_first_token"] or r["t_end"])
    path = sc["edge"] if m == "edge" else sc["cloud"]
    rtt = base[m] + path["rtt_ms"] / 1000
    k = sc.get("k", {}).get(m, 1)
    upload = (tx_slowstart if k > 1 else tx_simple)(parts["s_up"], path["up_mbit"] * 1e6, rtt)
    first = k * rtt + upload + parts["srv_first"]
    if m == "edge":
        first += sc.get("edge_queue_s", 0.0)
    download = tx_simple(parts["s_down"], path["down_mbit"] * 1e6)
    return first + max(parts["gen"], download) + parts["tail"], first


def build_profiles(data, base, ids, cfg):
    profiles = {}
    for m in TARGETS:
        rs = [data[i][m] for i in ids]
        q = {c: st.mean(r["quality"] for r in rs if r["cls"] == c) for c in CLASSES}
        n_out = {c: st.median(r["n_out"] or 0 for r in rs if r["cls"] == c) for c in CLASSES}
        parts = [decompose(r, base[m]) for r in rs]
        x = np.array([r["n_in"] or 0 for r in rs], float)
        y = np.array([p["srv_first"] for p in parts])
        (a0, a1), *_ = np.linalg.lstsq(np.vstack([np.ones_like(x), x]).T, y, rcond=None)
        long_enough = [(p, r) for p, r in zip(parts, rs, strict=True) if (r["n_out"] or 0) > 5]
        decode = st.median(p["gen"] / max(1, r["n_out"] - 1) for p, r in long_enough)
        bytes_per_tok = st.median(p["s_down"] / r["n_out"] for p, r in long_enough)
        price_in, price_out = cfg["prices_usd_per_mtok"].get(m, [0, 0])
        profiles[m] = Profile(
            q=q,
            n_out=n_out,
            ttft0=max(0.0, a0),
            prefill=max(0.0, a1),
            decode=decode,
            bytes_per_tok=bytes_per_tok,
            price_in=price_in / 1e6,
            price_out=price_out / 1e6,
        )
    return profiles


def loo_profiles(data, base, ids, cfg):
    """Leave-one-out: the profile used to route request i is estimated on all other requests."""
    return {i: build_profiles(data, base, [j for j in ids if j != i], cfg) for i in ids}


def telemetry_profiles(data, base, profiles, window=5, min_history=3):
    """Replace decode time and server-side TTFT by medians over the last `window` requests to the same target
    that finished before the request started; keep calibration values when the history is shorter."""
    history = {}
    for m in TARGETS:
        rs = sorted((data[i][m] for i in data), key=lambda r: r["wall_start"])
        history[m] = [
            (r["wall_start"] + r["t_end"], decode_time_per_token(r, base[m]), decompose(r, base[m])["srv_first"])
            for r in rs
        ]
    out = {}
    for i, prof in profiles.items():
        updated = {m: copy.copy(prof[m]) for m in TARGETS}
        for m in TARGETS:
            start = data[i][m]["wall_start"]
            previous = [h for h in history[m] if h[0] <= start][-window:]
            decode = [h[1] for h in previous if h[1] is not None]
            if len(decode) >= min_history:
                updated[m].decode = st.median(decode)
            if len(previous) >= min_history:
                updated[m].ttft0, updated[m].prefill = st.median(h[2] for h in previous), 0.0
        out[i] = updated
    return out


def net_state(sc, base, rtt_error=1.0, ignore_bandwidth=False, queue_visible=True):
    f = 0.0 if ignore_bandwidth else 1.0
    rtt = {"local": 0.0}
    up, down = {"local": 0}, {"local": 0}
    for m in ("edge", "cloud"):
        rtt[m] = (base[m] + sc[m]["rtt_ms"] / 1000) * rtt_error
        up[m] = sc[m]["up_mbit"] * 1e6 * f
        down[m] = sc[m]["down_mbit"] * 1e6 * f
    queue = {"local": 0.0, "edge": sc.get("edge_queue_s", 0.0) if queue_visible else 0.0, "cloud": 0.0}
    return NetState(rtt=rtt, b_up=up, b_down=down, queue=queue, k=dict(sc.get("k", {})))


NETWORK_BLIND = NetState(rtt={m: 0.0 for m in TARGETS}, b_up={m: 0 for m in TARGETS}, b_down={m: 0 for m in TARGETS})


def policies(tau_l, rtt_threshold, **qos_kw):
    """Compared policies; each maps (request, profiles, net state, tau_Q) to a target."""
    return {
        "always_local": lambda r, p, n, tq: "local",
        "always_edge": lambda r, p, n, tq: "edge",
        "always_cloud": lambda r, p, n, tq: "cloud",
        "quality_only": lambda r, p, n, tq: P.quality_only(r, p, n),
        "latency_only": lambda r, p, n, tq: P.latency_only(r, p, n),
        "static_rtt": lambda r, p, n, tq: P.static_rtt(r, p, n, rtt_threshold=rtt_threshold),
        "quality_cost": lambda r, p, n, tq: P.quality_cost(r, p, n, tau_q=tq),
        "qos_aware": lambda r, p, n, tq: P.qos_aware(r, p, n, tau_q=tq, tau_l=tau_l, **qos_kw),
        "network_blind": lambda r, p, n, tq: P.qos_aware(r, p, NETWORK_BLIND, tau_q=tq, tau_l=tau_l, **qos_kw),
    }


def quality_targets(profiles, factor):
    """tau_Q(c) = factor * best estimated class quality, per routed request."""
    return {i: {c: factor * max(p[m].q[c] for m in TARGETS) for c in CLASSES} for i, p in profiles.items()}


def _clairvoyant(i, data, real, deadline):
    on_time = [m for m in TARGETS if real[i][m][0] <= deadline]
    if not on_time:
        return min(TARGETS, key=lambda m: real[i][m][0])
    return max(on_time, key=lambda m: (data[i][m]["quality"], -(data[i][m].get("cost_usd") or 0), -real[i][m][0]))


def evaluate(data, base, profiles, ids, sc, pols, tau_q, tau_l, net=None, lam_l=1.0, lam_c=0.0, cost_ref=1e-3):
    """Route every request with every policy and summarise realised outcomes; also returns per-request rows.
    A clairvoyant benchmark that knows realised latency and quality is added as "clairvoyant"."""
    net = net or net_state(sc, base)
    parts = {i: {m: decompose(data[i][m], base[m]) for m in TARGETS} for i in ids}
    real = {i: {m: realised(data[i][m], parts[i][m], m, sc, base) for m in TARGETS} for i in ids}
    summary, rows_by_policy = {}, {}
    for name, pol in list(pols.items()) + [("clairvoyant", None)]:
        rows = []
        for i in ids:
            c = data[i]["local"]["cls"]
            if pol is None:
                m = _clairvoyant(i, data, real, tau_l[c])
            else:
                req = Request(cls=c, n_in=data[i]["local"]["n_in"] or 0, s_up=parts[i]["cloud"]["s_up"])
                m = pol(req, profiles[i], net, tau_q[i])
            rec, (lat, ttft), gen = data[i][m], real[i][m], parts[i][m]["gen"]
            rows.append(
                {
                    "id": i,
                    "cls": c,
                    "target": m,
                    "quality": rec["quality"],
                    "latency": lat,
                    "ttft": ttft,
                    "cost": rec.get("cost_usd") or 0.0,
                    "tokens_per_s": (rec["n_out"] or 0) / gen if gen > 0 else None,
                    "sla_violation": lat > tau_l[c],
                    "quality_violation": profiles[i][m].q[c] < tau_q[i][c],
                }
            )
        rows_by_policy[name] = rows
        summary[name] = _summarise(rows, tau_l, lam_l, lam_c, cost_ref)
    return summary, rows_by_policy


def _summarise(rows, tau_l, lam_l, lam_c, cost_ref):
    lat = [x["latency"] for x in rows]
    return {
        "quality": st.mean(x["quality"] for x in rows),
        "lat_mean": st.mean(lat),
        "lat_p50": float(np.percentile(lat, 50)),
        "lat_p95": float(np.percentile(lat, 95)),
        "ttft_p50": float(np.percentile([x["ttft"] for x in rows], 50)),
        "tps_median": float(np.median([x["tokens_per_s"] for x in rows if x["tokens_per_s"]])),
        "cost_per_1k": 1000 * st.mean(x["cost"] for x in rows),
        "share": {m: sum(x["target"] == m for x in rows) / len(rows) for m in TARGETS},
        "sla_viol": st.mean(x["sla_violation"] for x in rows),
        "q_viol": st.mean(x["quality_violation"] for x in rows),
        "utility": st.mean(
            x["quality"] - lam_l * x["latency"] / tau_l[x["cls"]] - lam_c * x["cost"] / cost_ref for x in rows
        ),
        "by_cls": {
            c: {
                "quality": st.mean(x["quality"] for x in rows if x["cls"] == c),
                "lat_p95": float(np.percentile([x["latency"] for x in rows if x["cls"] == c], 95)),
                "sla_viol": st.mean(x["sla_violation"] for x in rows if x["cls"] == c),
                "share": {m: st.mean(x["target"] == m for x in rows if x["cls"] == c) for m in TARGETS},
            }
            for c in CLASSES
        },
    }


def bootstrap_diff(a_rows, b_rows, key, n=2000, seed=1):
    """Mean paired difference of `key` and its 95 % percentile bootstrap interval over requests."""
    rng = np.random.default_rng(seed)
    d = np.array([float(a[key]) - float(b[key]) for a, b in zip(a_rows, b_rows, strict=True)])
    boot = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(n)]
    return float(d.mean()), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))
