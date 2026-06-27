"""
Small metric charts for Touchstone. Each function returns PNG bytes; the server
wraps them in an MCP Image so they render inline in the chat.

Figures are deliberately small (compact dashboard tiles, not report figures).
"""
from __future__ import annotations
import io
import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt

_FIGSIZE = (5.2, 3.0)
_DPI = 120


def _png(fig) -> bytes:
    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png", dpi=_DPI)
    plt.close(fig)
    return buf.getvalue()


def trend(runs: list, metric: str, threshold: float | None = None,
          show_range: bool = True) -> bytes:
    """Line of a metric's mean score across runs, with a threshold line and an
    optional shaded min-max band (the 'range' of the window)."""
    xs = list(range(len(runs)))
    ys = [r["aggregate"].get(metric, {}).get("mean_score") for r in runs]
    labels = [r["label"] or r["run_id"][2:13] for r in runs]

    fig, ax = plt.subplots(figsize=_FIGSIZE)
    ax.plot(xs, ys, marker="o", linewidth=2, color="#2563eb", label=metric)

    valid = [y for y in ys if y is not None]
    if show_range and len(valid) >= 2:
        lo, hi = min(valid), max(valid)
        ax.axhspan(lo, hi, color="#2563eb", alpha=0.08, label=f"range {lo:.2f}-{hi:.2f}")
    if threshold is not None:
        ax.axhline(threshold, color="#dc2626", linestyle="--", linewidth=1,
                   label=f"threshold {threshold:.2f}")

    ax.set_ylim(0, 1.02)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=7)
    ax.set_ylabel("mean score")
    ax.set_title(f"{metric} over {len(runs)} runs", fontsize=10)
    ax.legend(fontsize=7, loc="lower left")
    ax.grid(True, alpha=0.25)
    return _png(fig)


def run_bars(run: dict, threshold: float | None = None) -> bytes:
    """Bar of every metric's mean score for a single run."""
    agg = run["aggregate"]
    metrics = list(agg.keys())
    ys = [agg[m]["mean_score"] for m in metrics]
    colors = ["#16a34a" if (threshold is None or y >= threshold) else "#dc2626" for y in ys]

    fig, ax = plt.subplots(figsize=_FIGSIZE)
    ax.bar(metrics, ys, color=colors)
    if threshold is not None:
        ax.axhline(threshold, color="#dc2626", linestyle="--", linewidth=1)
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("mean score")
    ax.set_title(f"{run['label'] or run['run_id']}", fontsize=10)
    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels(metrics, rotation=30, ha="right", fontsize=7)
    ax.grid(True, axis="y", alpha=0.25)
    return _png(fig)


def compare(runs: list, threshold: float | None = None) -> bytes:
    """Grouped bars comparing several runs across all shared metrics."""
    metrics = sorted({m for r in runs for m in r["aggregate"]})
    n = len(runs)
    width = 0.8 / max(n, 1)
    palette = ["#2563eb", "#1e40af", "#60a5fa", "#3b82f6", "#1d4ed8", "#93c5fd"]

    fig, ax = plt.subplots(figsize=_FIGSIZE)
    for i, r in enumerate(runs):
        ys = [r["aggregate"].get(m, {}).get("mean_score", 0) for m in metrics]
        xs = [j + i * width for j in range(len(metrics))]
        ax.bar(xs, ys, width=width, color=palette[i % len(palette)],
               label=r["label"] or r["run_id"][2:13])
    if threshold is not None:
        ax.axhline(threshold, color="#dc2626", linestyle="--", linewidth=1)

    ax.set_ylim(0, 1.02)
    ax.set_xticks([j + width * (n - 1) / 2 for j in range(len(metrics))])
    ax.set_xticklabels(metrics, rotation=30, ha="right", fontsize=7)
    ax.set_ylabel("mean score")
    ax.set_title("run comparison", fontsize=10)
    ax.legend(fontsize=7)
    ax.grid(True, axis="y", alpha=0.25)
    return _png(fig)
