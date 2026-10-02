"""Fig. 1 (system architecture) and Fig. 2 (decision procedure)."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

from qos_orchestration.paths import FIGURES as OUT  # noqa: E402

plt.rcParams.update({"font.family": ["Times New Roman", "DejaVu Serif"], "font.size": 10})


def box(ax, x, y, w, h, text, fc="white", lw=1.0, fs=9, ls="-"):
    ax.add_patch(
        FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", fc=fc, ec="black", lw=lw, ls=ls)
    )
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, wrap=True)


def arrow(ax, x1, y1, x2, y2, text="", style="-|>", ls="-", off=(0, 0.12), fs=9):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, mutation_scale=10, lw=1, ls=ls, color="black"))
    if text:
        ax.text((x1 + x2) / 2 + off[0], (y1 + y2) / 2 + off[1], text, ha="center", va="bottom", fontsize=fs)


def fig1():
    fig, ax = plt.subplots(figsize=(7.0, 3.4))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis("off")
    # client device
    ax.add_patch(
        FancyBboxPatch(
            (0.2, 0.4), 5.6, 7.2, boxstyle="round,pad=0.02,rounding_size=0.1", fc="#f2f2f2", ec="black", ls="--"
        )
    )
    ax.text(3.0, 7.25, "Client device / application node", ha="center", fontsize=9, style="italic")
    box(ax, 0.6, 5.6, 4.8, 1.0, "Application (request q,\ntask class c, deadline τ_L)")
    box(
        ax,
        0.6,
        3.4,
        4.8,
        1.7,
        "Orchestrator\n(estimate Q̂, L̂, Ĉ for each target;\nconstrained selection, Fig. 2)",
        fc="#dddddd",
        lw=1.4,
    )
    box(ax, 1.5, 1.9, 1.85, 1.0, "Telemetry:\nRTT, B_up,\nB_down, queue,\ntokens/s", fs=8)
    box(ax, 3.55, 1.9, 1.85, 1.0, "Calibration\nprofiles per\nclass and target", fs=8.5)
    box(ax, 0.6, 0.6, 4.8, 0.9, "LOCAL_SMALL: Qwen2.5-3B\n(on-device, no network)", fs=9)
    arrow(ax, 3.0, 5.6, 3.0, 5.1)
    arrow(ax, 2.42, 2.9, 2.42, 3.4)
    arrow(ax, 4.47, 2.9, 4.47, 3.4)
    arrow(ax, 1.0, 3.4, 1.0, 1.5)
    # network
    box(ax, 7.0, 5.2, 2.6, 1.3, "LAN path\n(netem: qos-rlan)\nRTT≈2–3 ms", fs=9)
    box(ax, 7.0, 1.6, 2.6, 1.6, "WAN path\n(netem: qos-rwan)\n+ΔRTT, B_up, B_down\n+ real Internet", fs=9)
    box(ax, 10.6, 5.2, 3.2, 1.3, "LOCAL_LARGE / EDGE:\nQwen2.5-7B (Ollama)", fs=9)
    box(ax, 10.6, 1.6, 3.2, 1.6, "REMOTE_CLOUD:\nQwen2.5-72B via\nOpenRouter API\n(queue + inference)", fs=9)
    arrow(ax, 5.4, 4.6, 7.0, 5.8)
    arrow(ax, 9.6, 5.85, 10.6, 5.85)
    arrow(ax, 5.4, 3.9, 7.0, 2.4)
    arrow(ax, 9.6, 2.4, 10.6, 2.4)
    ax.text(12.2, 4.2, "streamed tokens\n(NDJSON / SSE)", ha="center", fontsize=9, style="italic")
    fig.tight_layout()
    fig.savefig(OUT / "fig1_architecture.png", dpi=300)
    plt.close(fig)


def fig2():
    fig, ax = plt.subplots(figsize=(7.0, 2.9))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 5.8)
    ax.axis("off")
    box(ax, 0.1, 4.2, 4.3, 1.4, "Request q: class c,\nprompt size n_in, S_req;\ndeadline τ_L(c)", fs=9)
    box(
        ax,
        4.85,
        4.2,
        4.3,
        1.4,
        "Telemetry s: RTT, B_up, B_down,\nedge queue W, median speed\nof the last 5 requests",
        fs=9,
    )
    box(ax, 9.6, 4.2, 4.3, 1.4, "For m ∈ {LOCAL, EDGE, CLOUD}:\nQ̂(m,c), L̂(m,q,s) by (5)–(6),\nĈ(m,q)", fs=9)
    arrow(ax, 4.4, 4.9, 4.85, 4.9)
    arrow(ax, 9.15, 4.9, 9.6, 4.9)
    box(ax, 4.85, 2.2, 4.3, 1.4, "Feasible set F:\nQ̂ ≥ τ_Q(c) and L̂ ≤ τ_L(c).\nIs F non-empty?", fs=9, fc="#eeeeee")
    arrow(ax, 11.75, 4.2, 9.15, 3.2)
    box(ax, 0.1, 0.2, 4.3, 1.4, "yes: m* = argmin over F of Ĉ\n(ties: smaller L̂)", fs=9, fc="#dddddd")
    box(ax, 9.6, 0.2, 4.3, 1.4, "no: m* = argmax U =\nQ̂ − λ_L·L̂/τ_L − λ_C·Ĉ/C_ref", fs=9, fc="#dddddd")
    box(ax, 4.85, 0.2, 4.3, 1.4, "Dispatch q to m*; log latency\nand tokens/s; update telemetry", fs=9)
    arrow(ax, 4.85, 2.6, 2.25, 1.6)
    arrow(ax, 9.15, 2.6, 11.75, 1.6)
    arrow(ax, 4.4, 0.9, 4.85, 0.9)
    arrow(ax, 9.6, 0.9, 9.15, 0.9)
    fig.tight_layout()
    fig.savefig(OUT / "fig2_algorithm.png", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    fig1()
    fig2()
