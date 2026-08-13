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
    palette = ["#2563eb", "#16a34a", "#f59e0b", "#9333ea", "#ec4899", "#0891b2"]

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


# ---------------------------------------------------------------------------
# Text renderers.
#
# Some MCP clients receive an image content block but never display it, so a
# PNG-only chart is invisible there. These render the same information as plain
# text, which every client can show.
# ---------------------------------------------------------------------------

_BLOCKS = "▁▂▃▄▅▆▇█"


def _bar(value, width: int = 24) -> str:
    """Horizontal bar for a 0..1 score."""
    if value is None:
        return "?" * 3
    filled = max(0, min(width, round(float(value) * width)))
    return "█" * filled + "░" * (width - filled)


def _verdict(value, threshold) -> str:
    if value is None or threshold is None:
        return ""
    return "PASS" if value >= threshold else "FAIL"


def _sparkline(values) -> str:
    vals = [v for v in values if v is not None]
    if not vals:
        return ""
    lo, hi = min(vals), max(vals)
    span = (hi - lo) or 1.0
    out = []
    for v in values:
        if v is None:
            out.append(" ")
        else:
            idx = int((v - lo) / span * (len(_BLOCKS) - 1))
            out.append(_BLOCKS[idx])
    return "".join(out)


def run_bars_text(run: dict, threshold: float | None = None) -> str:
    """Every metric's mean score for one run, as labelled bars."""
    agg = run.get("aggregate", {}) or {}
    thr = threshold if threshold is not None else run.get("threshold", 0.7)
    label = run.get("label") or run.get("run_id", "run")
    gs = run.get("golden_set", "")
    n_cases = len(run.get("per_case", []) or [])

    width = max((len(m) for m in agg), default=6)
    lines = [f"{label}" + (f"  ·  {gs}" if gs else "") +
             (f"  ·  {n_cases} cases" if n_cases else ""),
             "-" * 58]
    for m, a in agg.items():
        score = a.get("mean_score")
        pr = a.get("pass_rate")
        v = _verdict(score, thr)
        score_s = "  n/a" if score is None else f"{score:5.2f}"
        pr_s = "" if pr is None else f"  pass rate {pr*100:3.0f}%"
        lines.append(f"{m:<{width}}  {score_s}  {_bar(score)}  {v}{pr_s}")
    lines.append("-" * 58)
    lines.append(f"threshold {thr:.2f}")
    return "\n".join(lines)


def trend_text(runs: list, metric: str, threshold: float | None = None) -> str:
    """One metric's mean score across runs, as a sparkline plus a value table."""
    ys = [r.get("aggregate", {}).get(metric, {}).get("mean_score") for r in runs]
    thr = threshold if threshold is not None else 0.7
    labels = [(r.get("label") or r.get("run_id", "?")) for r in runs]

    lines = [f"{metric} across {len(runs)} run(s)", "-" * 58]
    spark = _sparkline(ys)
    if spark:
        lines.append(f"  {spark}   (oldest to newest)")
        lines.append("")
    width = max((len(l) for l in labels), default=6)
    for lab, v in zip(labels, ys):
        v_s = "  n/a" if v is None else f"{v:5.2f}"
        lines.append(f"{lab:<{width}}  {v_s}  {_bar(v)}  {_verdict(v, thr)}")
    vals = [v for v in ys if v is not None]
    lines.append("-" * 58)
    if vals:
        delta = vals[-1] - vals[0]
        arrow = "up" if delta > 0.001 else ("down" if delta < -0.001 else "flat")
        lines.append(f"threshold {thr:.2f}   first {vals[0]:.2f} -> latest "
                     f"{vals[-1]:.2f}  ({arrow} {abs(delta):.2f})")
    return "\n".join(lines)


def compare_text(runs: list, threshold: float | None = None) -> str:
    """All metrics across all runs, as a table."""
    metrics = []
    for r in runs:
        for m in (r.get("aggregate", {}) or {}):
            if m not in metrics:
                metrics.append(m)
    thr = threshold if threshold is not None else 0.7
    labels = [(r.get("label") or r.get("run_id", "?")) for r in runs]
    lab_w = max((len(l) for l in labels), default=6)
    col_w = max((len(m) for m in metrics), default=6) + 2

    header = " " * lab_w + "".join(f"  {m:>{col_w}}" for m in metrics)
    lines = [f"{len(runs)} run(s) x {len(metrics)} metric(s)", "-" * len(header),
             header, "-" * len(header)]
    for r, lab in zip(runs, labels):
        agg = r.get("aggregate", {}) or {}
        row = f"{lab:<{lab_w}}"
        for m in metrics:
            v = agg.get(m, {}).get("mean_score")
            row += f"  {'n/a' if v is None else f'{v:.2f}':>{col_w}}"
        lines.append(row)
    lines.append("-" * len(header))
    lines.append(f"threshold {thr:.2f}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Markdown renderers.
#
# Chat clients render markdown but not images, so a markdown table with an
# inline bar column reads far better than plain ASCII: real columns, real
# alignment, and eighth-block characters give eight times the bar resolution.
# ---------------------------------------------------------------------------

_EIGHTHS = "▏▎▍▌▋▊▉█"


def _smooth_bar(value, width: int = 20) -> str:
    """Bar for a 0..1 score with 1/8-cell resolution."""
    if value is None:
        return "·" * width
    units = max(0.0, min(1.0, float(value))) * width
    full = int(units)
    rem = units - full
    bar = "█" * full
    if rem > 0.06 and full < width:
        bar += _EIGHTHS[min(7, int(rem * 8))]
    return bar.ljust(width, "·")


def _case_score(case: dict, metric: str | None = None):
    sc = (case or {}).get("scores", {}) or {}
    if metric and metric in sc:
        return sc[metric].get("score")
    if sc:
        return next(iter(sc.values())).get("score")
    return case.get("min_score")


def run_report_md(run: dict, threshold: float | None = None) -> str:
    """Full markdown report for one run: scores, case strip, distribution."""
    agg = run.get("aggregate", {}) or {}
    thr = threshold if threshold is not None else run.get("threshold", 0.7)
    cases = run.get("per_case", []) or []
    label = run.get("label") or run.get("run_id", "run")

    out = [f"**{label}**"
           + (f" · `{run.get('golden_set')}`" if run.get("golden_set") else "")
           + (f" · {len(cases)} cases" if cases else "")]

    # score table with an inline bar column
    rows = ["| metric | score | | verdict |", "|---|---:|---|---|"]
    for m, a in agg.items():
        s = a.get("mean_score")
        verdict = "**PASS**" if (s is not None and s >= thr) else "**FAIL**"
        pr = a.get("pass_rate")
        pr_s = f" {pr*100:.0f}%" if pr is not None else ""
        rows.append(f"| {m} | `{s:.2f}` | `{_smooth_bar(s)}` | {verdict}{pr_s} |")
    out.append("\n".join(rows))

    if cases:
        metric = next(iter(agg), None)
        marks, fails = [], []
        for i, c in enumerate(cases):
            s = _case_score(c, metric)
            ok = s is not None and s >= thr
            marks.append("●" if ok else "○")
            if not ok:
                fails.append((i, s, (c.get("input") or "")[:60]))
        passed = marks.count("●")
        out.append(f"**Cases** `{''.join(marks)}`  ({passed} pass, "
                   f"{len(marks)-passed} fail · ● pass, ○ fail)")

        # distribution across score bands
        bands = [(0.8, 1.01, "0.8–1.0"), (0.6, 0.8, "0.6–0.8"),
                 (0.4, 0.6, "0.4–0.6"), (0.2, 0.4, "0.2–0.4"), (-0.01, 0.2, "0.0–0.2")]
        scores = [_case_score(c, metric) for c in cases]
        scores = [s for s in scores if s is not None]
        if scores:
            dist = ["| score | cases | |", "|---|---:|---|"]
            for lo, hi, name in bands:
                n = sum(1 for s in scores if lo <= s < hi)
                if n:
                    dist.append(f"| {name} | {n} | `{'█' * n}` |")
            out.append("\n".join(dist))

        if fails:
            fl = ["**Lowest scoring**", "", "| # | score | case |", "|---|---:|---|"]
            for i, s, text in sorted(fails, key=lambda x: (x[1] is None, x[1]))[:5]:
                fl.append(f"| {i} | `{'n/a' if s is None else f'{s:.2f}'}` | {text} |")
            out.append("\n".join(fl))

    out.append(f"_threshold {thr:.2f}_")
    return "\n\n".join(out)


def trend_report_md(runs: list, metric: str, threshold: float | None = None) -> str:
    """Markdown trend for one metric across runs."""
    thr = threshold if threshold is not None else 0.7
    rows = ["| run | score | | verdict |", "|---|---:|---|---|"]
    vals = []
    for r in runs:
        s = r.get("aggregate", {}).get(metric, {}).get("mean_score")
        vals.append(s)
        label = r.get("label") or r.get("run_id", "?")
        verdict = "**PASS**" if (s is not None and s >= thr) else "**FAIL**"
        s_s = "n/a" if s is None else f"{s:.2f}"
        rows.append(f"| {label} | `{s_s}` | `{_smooth_bar(s)}` | {verdict} |")

    out = [f"**{metric}** across {len(runs)} run(s)", "\n".join(rows)]
    real = [v for v in vals if v is not None]
    if len(real) >= 2:
        d = real[-1] - real[0]
        word = "improved" if d > 0.001 else ("regressed" if d < -0.001 else "unchanged")
        out.append(f"`{_sparkline(vals)}`  {real[0]:.2f} → {real[-1]:.2f} "
                   f"({word} {abs(d):.2f})")
    out.append(f"_threshold {thr:.2f}_")
    return "\n\n".join(out)


def compare_report_md(runs: list, threshold: float | None = None) -> str:
    """Markdown matrix of every metric across every run."""
    thr = threshold if threshold is not None else 0.7
    metrics = []
    for r in runs:
        for m in (r.get("aggregate", {}) or {}):
            if m not in metrics:
                metrics.append(m)
    head = "| run | " + " | ".join(metrics) + " |"
    sep = "|---|" + "---:|" * len(metrics)
    rows = [head, sep]
    for r in runs:
        agg = r.get("aggregate", {}) or {}
        label = r.get("label") or r.get("run_id", "?")
        cells = []
        for m in metrics:
            v = agg.get(m, {}).get("mean_score")
            if v is None:
                cells.append("n/a")
            else:
                flag = "" if v >= thr else " ⚠"
                cells.append(f"`{v:.2f}`{flag}")
        rows.append(f"| {label} | " + " | ".join(cells) + " |")
    return "\n".join(rows) + f"\n\n_threshold {thr:.2f} · ⚠ below threshold_"
