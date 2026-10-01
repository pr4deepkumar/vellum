"""Charts from sweep results: one small-multiple panel per metric, one line per config."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter, NullFormatter  # noqa: E402
from matplotlib.transforms import ScaledTranslation  # noqa: E402

# Categorical slots in fixed order (validated palette, light surface).
# Color follows the config label, so fp16 is always slot 1 across every chart.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
PREFERRED_ORDER = ["hf-fp16", "fp16", "vllm-fp16", "awq", "vllm-awq", "fp8", "prefix-off", "prefix-on"]
MODES = (
    ("concurrency", "concurrent requests (log2)", True),
    ("batch_size", "batch size (log2)", True),
    ("request_rate", "offered load (req/s)", False),
)
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"


def load_runs(results_dir: Path) -> dict[str, dict]:
    runs = {}
    for path in sorted(results_dir.glob("*.json")):
        data = json.loads(path.read_text())
        if "levels" in data:
            runs[data["meta"]["label"]] = data
    order = sorted(runs, key=lambda k: (PREFERRED_ORDER.index(k) if k in PREFERRED_ORDER else 99, k))
    return {k: runs[k] for k in order}


def _style(ax, title: str, ylabel: str, xlabel: str) -> None:
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", fontsize=11, color=INK, fontweight="bold")
    ax.set_ylabel(ylabel, color=INK_2, fontsize=9)
    ax.set_xlabel(xlabel, color=INK_2, fontsize=9)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=8)


def _line(ax, xs, ys, color, label) -> tuple | None:
    """Draw one series; return its end point for direct labelling."""
    pts = [(x, y) for x, y in zip(xs, ys) if y is not None]
    if not pts:
        return None
    x, y = zip(*pts)
    ax.plot(x, y, color=color, linewidth=2, marker="o", markersize=5,
            markeredgecolor=SURFACE, markeredgewidth=1.5, label=label, solid_capstyle="round")
    return x[-1], y[-1], label


def _direct_labels(ax, ends: list) -> None:
    """Label line ends, nudging labels apart vertically so they never overlap."""
    nudge_right = ScaledTranslation(6 / 72, 0, ax.figure.dpi_scale_trans)
    lo, hi = ax.get_ylim()
    gap = (hi - lo) * 0.045
    prev = None
    for x, y, label in sorted((e for e in ends if e), key=lambda e: e[1]):
        y = y if prev is None else max(y, prev + gap)
        prev = y
        ax.text(x, y, label, va="center", fontsize=8, color=INK_2, transform=ax.transData + nudge_right)


def plot_dir(results_dir: Path) -> list[Path]:
    runs = load_runs(results_dir)
    if not runs:
        raise SystemExit(f"no sweep JSON files in {results_dir}")
    experiments: dict[str, dict[str, dict]] = {}
    for lab, run in runs.items():
        experiments.setdefault(run["meta"].get("experiment", "main"), {})[lab] = run
    outputs = []
    for exp, group in experiments.items():
        outputs += _plot_experiment(results_dir, exp, group)
    return outputs


def _plot_experiment(results_dir: Path, exp: str, runs: dict[str, dict]) -> list[Path]:
    # Slot assignment is per chart but in a stable label order, so a config keeps its color.
    colors = {label: SERIES[i % len(SERIES)] for i, label in enumerate(runs)}
    outputs = []
    for mode, xlabel, log_x in MODES:
        series = {lab: [lv for lv in run["levels"] if lv["mode"] == mode] for lab, run in runs.items()}
        series = {lab: lvls for lab, lvls in series.items() if lvls}
        if not series:
            continue
        direct = len(series) <= 4
        first = next(iter(series.values()))[0]
        fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), facecolor=SURFACE)
        panels = [
            ("Output throughput", "tokens / s", "output_token_throughput"),
            ("Per-token latency (TPOT p50)", "ms", "p50_tpot_ms"),
            ("Time to first token (p99)", "ms", "p99_ttft_ms"),
        ]
        for ax, (title, ylabel, key) in zip(axes.flat, panels):
            _style(ax, title, ylabel, xlabel)
            ends = [_line(ax, [lv["level"] for lv in lvls], [lv[key] for lv in lvls], colors[lab], lab)
                    for lab, lvls in series.items()]
            if log_x:
                ax.set_xscale("log", base=2)
                ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
                ax.xaxis.set_minor_formatter(NullFormatter())
            if direct:
                _direct_labels(ax, ends)
        # Pareto view: what throughput does each config buy at a given per-token latency?
        ax = axes.flat[3]
        _style(ax, "Throughput vs latency trade-off", "output tokens / s", "TPOT p50 (ms)")
        ends = [_line(ax, [lv["p50_tpot_ms"] for lv in lvls], [lv["output_token_throughput"] for lv in lvls],
                      colors[lab], lab) for lab, lvls in series.items()]
        if direct:
            _direct_labels(ax, ends)

        handles, labels = axes.flat[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper right", ncol=len(labels), frameon=False, fontsize=9)
        fig.suptitle(
            f"{exp}  ·  {first['model']}  ·  {first['input_len']} in / {first['output_len']} out tokens"
            + (f"  ·  {first['shared_prefix_len']}-token shared prefix" if first.get("shared_prefix_len") else ""),
            x=0.01, ha="left", fontsize=12, color=INK,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        out = results_dir / f"{exp}_{mode}.png"
        fig.savefig(out, dpi=150, facecolor=SURFACE)
        plt.close(fig)
        outputs.append(out)
    return outputs
