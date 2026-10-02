"""Tables, figures and summary numbers from the logged measurements.

Writes results/summary.json, results/{targets,policies,network_awareness,sensitivity}.csv and figures/fig3-fig5.
"""

import copy
import csv
import json
import statistics as st

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from qos_orchestration import simulation as S  # noqa: E402
from qos_orchestration.experiment import is_success  # noqa: E402
from qos_orchestration.netmodel import tx_simple  # noqa: E402
from qos_orchestration.paths import FIGURES, RAW, RESULTS  # noqa: E402
from qos_orchestration.policy import TARGETS  # noqa: E402

CLASSES = S.CLASSES
CLASS_NAMES = {"qa": "Factual QA", "math": "Math reasoning", "code": "Code generation", "long": "Long-form"}
TARGET_NAMES = {"local": "LOCAL_SMALL (3B)", "edge": "EDGE (7B)", "cloud": "REMOTE_CLOUD (72B)"}
POLICY_NAMES = {
    "always_local": "always LOCAL",
    "always_edge": "always EDGE",
    "always_cloud": "always CLOUD",
    "quality_only": "quality-only",
    "latency_only": "latency-only",
    "static_rtt": "static RTT threshold",
    "quality_cost": "quality-cost router",
    "qos_aware": "QoS-aware (proposed)",
    "network_blind": "deadline-aware, network-blind",
    "clairvoyant": "clairvoyant benchmark",
}
TARGET_STYLE = {
    "local": dict(color="0.0", ls="-", marker="o"),
    "edge": dict(color="0.45", ls="--", marker="s"),
    "cloud": dict(color="0.7", ls="-.", marker="^"),
}
NEW_CONNECTIONS = {"edge": 2, "cloud": 3}
RTT_SWEEP_MS = list(range(0, 1001, 25))
BANDWIDTH_SWEEP_MBIT = [10.0, 1.0, 0.256, 0.128, 0.064, 0.032, 0.016]
TAU_Q_FACTORS = [0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]

plt.rcParams.update({"font.family": ["Times New Roman", "DejaVu Serif"], "font.size": 10.5})


class Study:
    """Measurements, estimates and policies shared by all analyses."""

    def __init__(self):
        self.cfg = S.load_config()
        self.base = base_rtts()
        self.data = S.load_measurements()
        self.ids = sorted(self.data)
        self.tau_l = self.cfg["sla_tau_L_s"]
        self.calibration = S.loo_profiles(self.data, self.base, self.ids, self.cfg)
        self.profiles = S.telemetry_profiles(self.data, self.base, self.calibration)
        self.tau_q = S.quality_targets(self.profiles, self.cfg["tau_Q_factor"])
        u = self.cfg["utility"]
        self.weights = {"lam_l": u["lambda_L"], "lam_c": u["lambda_C"], "cost_ref": u["cost_ref_usd"]}
        self.policies = S.policies(self.tau_l, self.cfg["static_rtt_threshold_s"], **self.weights)

    def profile(self, name):
        return S.scenario(self.cfg["profiles_emulated"][name]["cloud"])

    def evaluate(self, sc, policies=None, profiles=None, tau_q=None, net=None):
        pols = self.policies if policies is None else {k: self.policies[k] for k in policies}
        return S.evaluate(
            self.data,
            self.base,
            profiles or self.profiles,
            self.ids,
            sc,
            pols,
            tau_q or self.tau_q,
            self.tau_l,
            net=net,
            **self.weights,
        )

    def class_ids(self, m, c):
        return [i for i in self.ids if self.data[i][m]["cls"] == c]

    def median_latency(self, m, c, sc):
        return st.median(
            S.realised(self.data[i][m], S.decompose(self.data[i][m], self.base[m]), m, sc, self.base)[0]
            for i in self.class_ids(m, c)
        )


def base_rtts():
    probes = [json.loads(line) for line in (RAW / "base_probes.jsonl").open()]
    base = {"local": 0.0, **{m: st.median(x for p in probes for x in p[m] if x) for m in ("edge", "cloud")}}
    (RESULTS / "base_rtt.json").write_text(json.dumps(base, indent=1))
    return base


def write_csv(name, rows):
    with (RESULTS / name).open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def policy_row(v):
    return {
        "quality": round(v["quality"], 3),
        "lat_p50_s": round(v["lat_p50"], 2),
        "lat_p95_s": round(v["lat_p95"], 2),
        "sla_viol_pct": round(100 * v["sla_viol"], 1),
        "q_viol_pct": round(100 * v["q_viol"], 1),
        "cost_per_1k_usd": round(v["cost_per_1k"], 4),
        **{f"{m}_pct": round(100 * v["share"][m]) for m in TARGETS},
    }


def targets_table(s):
    rows = []
    for m in TARGETS:
        for c in CLASSES:
            rs = [s.data[i][m] for i in s.class_ids(m, c)]
            parts = [S.decompose(r, s.base[m]) for r in rs]
            decode = [
                (r["n_out"] - 1) / p["gen"] for r, p in zip(rs, parts, strict=True) if p["gen"] > 0 and r["n_out"] > 5
            ]
            rows.append(
                {
                    "target": m,
                    "cls": c,
                    "n": len(rs),
                    "quality": round(st.mean(r["quality"] for r in rs), 3),
                    "ttft_med_s": round(st.median(r["t_first_token"] for r in rs), 3),
                    "e2e_med_s": round(st.median(r["t_end"] for r in rs), 2),
                    "e2e_p95_s": round(float(np.percentile([r["t_end"] for r in rs], 95)), 2),
                    "n_out_med": st.median(r["n_out"] for r in rs),
                    "decode_tok_s_med": round(st.median(decode), 1) if decode else None,
                    "truncated": sum(r.get("finish") == "length" for r in rs),
                    "cost_usd_per_1k": round(1000 * st.mean((r.get("cost_usd") or 0) for r in rs), 4),
                    "bytes_per_token_med": round(
                        st.median(p["s_down"] / r["n_out"] for r, p in zip(rs, parts, strict=True) if r["n_out"] > 5), 1
                    ),
                }
            )
    write_csv("targets.csv", rows)
    return rows


def crossover_rtt(s):
    """Eq. (7) from class medians with non-limiting bandwidth and k = 1, in ms (negative: local always faster)."""
    out = {}
    for c in CLASSES:
        local = st.median(s.data[i]["local"]["t_end"] for i in s.class_ids("local", c))
        out[c] = {}
        for m in ("edge", "cloud"):
            parts = [S.decompose(s.data[i][m], s.base[m]) for i in s.class_ids(m, c)]
            out[c][m] = round(1000 * (local - st.median(p["srv_first"] + p["gen"] for p in parts)), 1)
    return out


def rtt_sweep(s):
    warm_curves = {c: {m: [] for m in TARGETS} for c in CLASSES}
    shares = {"warm": {m: [] for m in TARGETS}, "new_connections": {m: [] for m in TARGETS}}
    cloud_cold = {c: [] for c in CLASSES}
    for rtt in RTT_SWEEP_MS:
        cloud = {"rtt_ms": rtt, "down_mbit": 0, "up_mbit": 0}
        warm, cold = S.scenario(cloud), S.scenario(cloud, round_trips=NEW_CONNECTIONS)
        for c in CLASSES:
            for m in TARGETS:
                warm_curves[c][m].append(s.median_latency(m, c, warm))
            cloud_cold[c].append(s.median_latency("cloud", c, cold))
        for key, sc in (("warm", warm), ("new_connections", cold)):
            summary, _ = s.evaluate(sc, policies=["qos_aware"])
            for m in TARGETS:
                shares[key][m].append(summary["qos_aware"]["share"][m])
    return {
        "rtt_ms": RTT_SWEEP_MS,
        "median_latency_warm": warm_curves,
        "cloud_median_latency_new_connections": cloud_cold,
        "shares_qos_aware": shares,
    }


def bandwidth_sweep(s):
    out = {
        "down_mbit": BANDWIDTH_SWEEP_MBIT,
        "cloud_median_latency": {c: [] for c in CLASSES},
        "shares_qos_aware": {m: [] for m in TARGETS},
    }
    for b in BANDWIDTH_SWEEP_MBIT:
        sc = S.scenario({"rtt_ms": 50, "down_mbit": b, "up_mbit": b / 2})
        for c in CLASSES:
            out["cloud_median_latency"][c].append(s.median_latency("cloud", c, sc))
        summary, _ = s.evaluate(sc, policies=["qos_aware"])
        for m in TARGETS:
            out["shares_qos_aware"][m].append(summary["qos_aware"]["share"][m])
    return out


def named_profiles(s):
    rows, summaries, bootstrap = [], {}, {}
    for name in s.cfg["profiles_emulated"]:
        summary, per_request = s.evaluate(s.profile(name))
        summaries[name] = summary
        rows += [{"profile": name, "policy": k, **policy_row(v)} for k, v in summary.items()]
        bootstrap[name] = {
            other: {
                "d_quality": S.bootstrap_diff(per_request["qos_aware"], per_request[other], "quality"),
                "d_sla": S.bootstrap_diff(per_request["qos_aware"], per_request[other], "sla_violation"),
            }
            for other in ("always_cloud", "quality_cost", "always_local", "network_blind")
        }
    write_csv("policies.csv", rows)
    return summaries, bootstrap


def quality_target_frontier(s):
    frontier = {}
    for name in ("moderate", "high_latency"):
        frontier[name] = {"qos_aware": [], "quality_cost": []}
        for f in TAU_Q_FACTORS:
            summary, _ = s.evaluate(
                s.profile(name), policies=list(frontier[name]), tau_q=S.quality_targets(s.profiles, f)
            )
            for k in frontier[name]:
                v = summary[k]
                frontier[name][k].append(
                    {
                        "tau_q_factor": f,
                        "quality": v["quality"],
                        "sla_viol": v["sla_viol"],
                        "lat_p95": v["lat_p95"],
                        "cost_per_1k": v["cost_per_1k"],
                        "share": v["share"],
                    }
                )
    return frontier


def network_awareness(s):
    """The proposed rule against the same rule without network information."""
    p = s.cfg["profiles_emulated"]
    cases = {
        "moderate, warm": S.scenario(p["moderate"]["cloud"]),
        "high_latency, warm": S.scenario(p["high_latency"]["cloud"]),
        "degraded, warm": S.scenario(p["degraded"]["cloud"]),
        "+250 ms, new connections": S.scenario(
            {"rtt_ms": 250, "down_mbit": 0, "up_mbit": 0}, round_trips=NEW_CONNECTIONS
        ),
        "+600 ms, new connections": S.scenario(
            {"rtt_ms": 600, "down_mbit": 0, "up_mbit": 0}, round_trips=NEW_CONNECTIONS
        ),
        "+50 ms, 32/16 kbit/s, warm": S.scenario({"rtt_ms": 50, "down_mbit": 0.032, "up_mbit": 0.016}),
        "+250 ms, 32/16 kbit/s, new connections": S.scenario(
            {"rtt_ms": 250, "down_mbit": 0.032, "up_mbit": 0.016}, round_trips=NEW_CONNECTIONS
        ),
    }
    out, rows = {}, []
    for name, sc in cases.items():
        summary, per_request = s.evaluate(sc, policies=["qos_aware", "network_blind", "always_cloud"])
        out[name] = {
            k: {
                "quality": v["quality"],
                "sla_viol": v["sla_viol"],
                "lat_p95": v["lat_p95"],
                "cost_per_1k": v["cost_per_1k"],
                "share": v["share"],
                "q_viol": v["q_viol"],
            }
            for k, v in summary.items()
        }
        out[name]["d_sla_vs_network_blind"] = S.bootstrap_diff(
            per_request["qos_aware"], per_request["network_blind"], "sla_violation"
        )
        out[name]["d_quality_vs_network_blind"] = S.bootstrap_diff(
            per_request["qos_aware"], per_request["network_blind"], "quality"
        )
        rows += [{"condition": name, "policy": k, **policy_row(v)} for k, v in summary.items()]
    write_csv("network_awareness.csv", rows)
    return out


def sensitivity(s):
    rows = []

    def add(label, profile, sc=None, **kw):
        summary, _ = s.evaluate(sc or s.profile(profile), policies=["qos_aware"], **kw)
        rows.append({"variant": label, "profile": profile, **policy_row(summary["qos_aware"])})

    for name in ("moderate", "high_latency"):
        sc = s.profile(name)
        add("reference", name)
        for f in (0.5, 2.0):
            add(f"RTT estimate x{f}", name, net=S.net_state(sc, s.base, rtt_error=f))
        for f in (0.75, 1.33):
            scaled = {i: {m: copy.copy(p[m]) for m in TARGETS} for i, p in s.profiles.items()}
            for p in scaled.values():
                for m in TARGETS:
                    p[m].decode *= f
            add(f"decode time per token x{f}", name, profiles=scaled)
        for f in (0.8, 1.0):
            add(f"tau_Q factor {f}", name, tau_q=S.quality_targets(s.profiles, f))
        add("without service-time telemetry", name, profiles=s.calibration)
    degraded = s.profile("degraded")
    add("reference", "degraded")
    add("without bandwidth term", "degraded", net=S.net_state(degraded, s.base, ignore_bandwidth=True))
    queued = S.scenario(s.cfg["profiles_emulated"]["moderate"]["cloud"], edge_queue_s=5.0)
    add("edge queue 5 s, visible", "moderate + edge queue", sc=queued)
    add(
        "edge queue 5 s, not visible",
        "moderate + edge queue",
        sc=queued,
        net=S.net_state(queued, s.base, queue_visible=False),
    )
    write_csv("sensitivity.csv", rows)
    return rows


def live_checks(s):
    """Compare live requests under emulated impairments with trace-driven predictions."""
    stats_file = RAW / "cloud_genstats.jsonl"
    stats = {r["gen_id"]: r for r in map(json.loads, stats_file.open())} if stats_file.exists() else {}

    def network_part(r):
        g = stats.get(r.get("gen_id"))
        return r["t_end"] - g["generation_time"] / 1000 if g and g.get("generation_time") else None

    out = {}
    for path in sorted(RAW.glob("live_*.jsonl")):
        attempts = [json.loads(line) for line in path.open()]
        recs = [r for r in attempts if is_success(r)]
        out[path.stem] = {"attempts": len(attempts), "n": len(recs)}
        if not recs:
            continue
        p, m = recs[0]["profile_params"], recs[0]["target"]
        k = NEW_CONNECTIONS if p.get("new_conn") else None
        sc = (
            S.scenario(p, round_trips=k)
            if m == "cloud"
            else S.scenario(s.cfg["profiles_emulated"]["base"]["cloud"], edge=p, round_trips=k)
        )
        pairs = []
        for r in recs:
            ref = s.data[r["item_id"]][m]
            pairs.append((S.realised(ref, S.decompose(ref, s.base[m]), m, sc, s.base)[0], r["t_end"]))
        out[path.stem] |= {
            "mape_pct": 100 * st.mean(abs(a - b) / b for a, b in pairs),
            "median_ape_pct": 100 * st.median(abs(a - b) / b for a, b in pairs),
            "mean_measured_s": st.mean(b for _, b in pairs),
            "mean_predicted_s": st.mean(a for a, _ in pairs),
        }

    base_parts = [x for x in (network_part(s.data[i]["cloud"]) for i in s.ids) if x is not None]
    overhead = st.median(base_parts) - s.base["cloud"] if base_parts else None
    for path in sorted(RAW.glob("live_cloud_*.jsonl")):
        recs = [r for r in map(json.loads, path.open()) if is_success(r)]
        if not recs or overhead is None:
            continue
        p = recs[0]["profile_params"]
        rtt = s.base["cloud"] + p["rtt_ms"] / 1000
        k = NEW_CONNECTIONS["cloud"] if p.get("new_conn") else 1
        rows = []
        for r in recs:
            measured = network_part(r)
            if measured is None:
                continue
            parts = S.decompose(r, s.base["cloud"])
            upload = tx_simple(parts["s_up"], p["up_mbit"] * 1e6)
            download = tx_simple(parts["s_down"], p["down_mbit"] * 1e6)
            predicted = k * rtt + overhead + upload + max(0.0, download - parts["gen"])
            generation = stats[r["gen_id"]]["generation_time"] / 1000
            predicted_e2e = k * rtt + overhead + upload + max(generation, download)
            rows.append((predicted, measured, predicted_e2e, r["t_end"]))
        if not rows:
            continue
        out[path.stem].update(
            {
                "network_measured_median_ms": 1000 * st.median(x[1] for x in rows),
                "network_predicted_median_ms": 1000 * st.median(x[0] for x in rows),
                "network_mae_ms": 1000 * st.mean(abs(x[0] - x[1]) for x in rows),
                "e2e_given_provider_time_mape_pct": 100 * st.mean(abs(x[2] - x[3]) / x[3] for x in rows),
                "e2e_given_provider_time_bias_pct": 100 * st.mean((x[2] - x[3]) / x[3] for x in rows),
            }
        )

    attempts = [json.loads(line) for line in (RAW / "characterize_cloud.jsonl").open()]
    decode = [
        (s.data[i]["cloud"]["n_out"] - 1) / S.decompose(s.data[i]["cloud"], s.base["cloud"])["gen"]
        for i in s.ids
        if s.data[i]["cloud"]["n_out"] > 20
    ]
    out["cloud"] = {
        "base_network_part_median_ms": 1000 * st.median(base_parts) if base_parts else None,
        "residual_overhead_ms": 1000 * overhead if overhead is not None else None,
        "attempts": len(attempts),
        "http_errors": sum(1 for r in attempts if r.get("error")),
        "decode_tok_s_p10_p50_p90": [float(np.percentile(decode, q)) for q in (10, 50, 90)],
    }
    return out


def plot_latency_vs_rtt(s, sweep):
    fig, axs = plt.subplots(1, 4, figsize=(7.2, 2.5))
    for ax, c in zip(axs, CLASSES, strict=True):
        for m in TARGETS:
            ax.plot(
                sweep["rtt_ms"],
                sweep["median_latency_warm"][c][m],
                label=TARGET_NAMES[m],
                markevery=8,
                ms=3.5,
                lw=1.1,
                **TARGET_STYLE[m],
            )
        ax.axhline(s.tau_l[c], color="0.3", lw=0.7, ls=":")
        ax.set_title(CLASS_NAMES[c], fontsize=9)
        ax.set_xlabel("Added RTT, ms", fontsize=8.5)
        ax.set_xticks([0, 500, 1000])
        ax.tick_params(labelsize=8)
        ax.grid(alpha=0.3, lw=0.4)
    axs[0].set_ylabel("Median E2E latency, s", fontsize=8.5)
    fig.legend(*axs[0].get_legend_handles_labels(), loc="lower center", ncol=3, fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(FIGURES / "fig3_latency_vs_rtt.png", dpi=300)
    plt.close(fig)


def plot_routing_shares(sweep):
    fig, axs = plt.subplots(1, 2, figsize=(6.6, 3.0), sharey=True)
    titles = {
        "warm": "(a) warm keep-alive connections, k = 1",
        "new_connections": "(b) new connection per request, k = 2 (edge) / 3 (cloud)",
    }
    for ax, (key, title) in zip(axs, titles.items(), strict=True):
        shares = sweep["shares_qos_aware"][key]
        ax.stackplot(
            sweep["rtt_ms"],
            [shares[m] for m in TARGETS],
            labels=[TARGET_NAMES[m] for m in TARGETS],
            colors=["0.15", "0.55", "0.85"],
            edgecolor="black",
            lw=0.3,
        )
        ax.set_xlabel("Added RTT on the WAN path, ms")
        ax.set_ylim(0, 1)
        ax.set_xlim(0, sweep["rtt_ms"][-1])
        ax.set_title(title, fontsize=8)
    axs[0].set_ylabel("Share of requests")
    fig.legend(*axs[0].get_legend_handles_labels(), loc="lower center", ncol=3, fontsize=9, frameon=False)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(FIGURES / "fig4_routing_shares.png", dpi=300)
    plt.close(fig)


def plot_quality_vs_sla(summaries, frontier):
    markers = {
        "always_local": "o",
        "always_edge": "s",
        "always_cloud": "^",
        "quality_only": "v",
        "latency_only": "<",
        "static_rtt": ">",
        "quality_cost": "D",
        "clairvoyant": "P",
    }
    fig, axs = plt.subplots(1, 2, figsize=(6.6, 3.5), sharey=True)
    for ax, name in zip(axs, ("moderate", "high_latency"), strict=True):
        for k, v in summaries[name].items():
            if k == "network_blind":
                ax.scatter(
                    100 * v["sla_viol"],
                    v["quality"],
                    marker="x",
                    s=40,
                    c="black",
                    lw=1.0,
                    label=POLICY_NAMES[k],
                    zorder=4,
                )
            elif k in markers:
                ax.scatter(
                    100 * v["sla_viol"],
                    v["quality"],
                    marker=markers[k],
                    s=30,
                    lw=0.8,
                    edgecolor="black",
                    facecolor="0.6" if k == "clairvoyant" else "white",
                    label=POLICY_NAMES[k],
                    zorder=3,
                )
        for k, style in (
            ("qos_aware", dict(color="black", ls="-", marker="*", ms=7)),
            ("quality_cost", dict(color="0.5", ls="--", marker="D", ms=3.5)),
        ):
            points = frontier[name][k]
            ax.plot(
                [100 * x["sla_viol"] for x in points],
                [x["quality"] for x in points],
                label=POLICY_NAMES[k] + ", τ_Q varied",
                lw=1,
                **style,
            )
        ax.set_title({"moderate": "(a) moderate profile", "high_latency": "(b) high-latency profile"}[name], fontsize=9)
        ax.set_xlabel("SLA violations, %")
        ax.grid(alpha=0.3, lw=0.4)
    axs[0].set_ylabel("Mean task quality")
    fig.legend(*axs[0].get_legend_handles_labels(), loc="lower center", ncol=3, fontsize=9, frameon=False)
    fig.tight_layout(rect=(0, 0.24, 1, 1))
    fig.savefig(FIGURES / "fig5_quality_sla.png", dpi=300)
    plt.close(fig)


def main():
    s = Study()
    sweep = rtt_sweep(s)
    summaries, bootstrap = named_profiles(s)
    frontier = quality_target_frontier(s)
    summary = {
        "n_items": len(s.ids),
        "base_rtt_ms": {k: 1000 * v for k, v in s.base.items()},
        "tau_L": s.tau_l,
        "targets": targets_table(s),
        "crossover_rtt_ms": crossover_rtt(s),
        "rtt_sweep": sweep,
        "bandwidth_sweep": bandwidth_sweep(s),
        "profiles": summaries,
        "bootstrap_qos_aware_vs": bootstrap,
        "quality_target_frontier": frontier,
        "network_awareness": network_awareness(s),
        "sensitivity": sensitivity(s),
        "live_checks": live_checks(s),
        "netbench": json.loads((RESULTS / "netbench_summary.json").read_text()),
    }
    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=1, default=float))
    FIGURES.mkdir(parents=True, exist_ok=True)
    plot_latency_vs_rtt(s, sweep)
    plot_routing_shares(sweep)
    plot_quality_vs_sla(summaries, frontier)
    print(json.dumps({k: summary[k] for k in ("base_rtt_ms", "crossover_rtt_ms")}, indent=1))


if __name__ == "__main__":
    main()
