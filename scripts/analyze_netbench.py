"""E1 analysis: measured HTTP transfer times against the network model, eqs. (2)-(3).

Writes results/netbench_cells.json (median per configuration with predictions) and results/netbench_summary.json.
"""

import json
import statistics as st

from qos_orchestration.netmodel import t_net
from qos_orchestration.paths import RAW, RESULTS

HTTP_HEADER_BYTES = 150


def main():
    rows = [json.loads(line) for line in (RAW / "e1_netbench.jsonl").open()]
    base = st.median(x for r in rows if r["kind"] == "base_probe" for x in r["tcp_connect_s"])
    cells = []
    for r in rows:
        if r["kind"] != "echo":
            continue
        rtt, bw = base + r["rtt_ms"] / 1000, r["rate_mbit"] * 1e6
        up, down, new_conn = r["s_up"] + HTTP_HEADER_BYTES, r["s_down"] + HTTP_HEADER_BYTES, not r["keepalive"]
        cells.append(
            {
                "rtt_ms": r["rtt_ms"],
                "rate_mbit": r["rate_mbit"],
                "s_up": r["s_up"],
                "s_down": r["s_down"],
                "keepalive": r["keepalive"],
                "measured_s": st.median(r["t_s"]),
                "simple_s": t_net(rtt, up, down, bw, bw, new_conn, slowstart=False),
                "slowstart_s": t_net(rtt, up, down, bw, bw, new_conn, slowstart=True),
            }
        )

    def errors(model, keepalive):
        return [(c[model] - c["measured_s"]) / c["measured_s"] for c in cells if c["keepalive"] == keepalive]

    summary = {
        "base_rtt_ms": 1000 * base,
        "cells": len(cells),
        "repetitions": len(rows[1]["t_s"]),
        "mape_pct": {},
        "bias_pct": {},
    }
    for model in ("simple", "slowstart"):
        for keepalive, label in ((True, "warm"), (False, "new")):
            e = errors(f"{model}_s", keepalive)
            summary["mape_pct"][f"{model}_{label}"] = 100 * st.mean(abs(x) for x in e)
            summary["bias_pct"][f"{model}_{label}"] = 100 * st.mean(e)
    combined = [c["simple_s"] if c["keepalive"] else c["slowstart_s"] for c in cells]
    ape = [abs(p - c["measured_s"]) / c["measured_s"] for p, c in zip(combined, cells, strict=True)]
    abs_ms = sorted(abs(p - c["measured_s"]) * 1000 for p, c in zip(combined, cells, strict=True))
    summary.update(
        {
            "combined_mape_pct": 100 * st.mean(ape),
            "combined_median_ape_pct": 100 * st.median(ape),
            "combined_mae_ms": st.mean(abs_ms),
            "combined_p90_ae_ms": abs_ms[int(0.9 * len(abs_ms))],
        }
    )
    (RESULTS / "netbench_cells.json").write_text(json.dumps(cells, indent=1))
    (RESULTS / "netbench_summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
