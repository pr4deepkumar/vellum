"""Charts from sweep results: one small-multiple panel per metric, one line per config.

An experiment holding the from-scratch engine's rungs (scratch-*) gets two charts: the usual one
with the reference engines and the highest rung, and a ladder chart with the rungs on the x-axis.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter, NullFormatter  # noqa: E402
from matplotlib.transforms import ScaledTranslation  # noqa: E402

# Categorical slots in fixed order (validated palette, light surface).
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SCRATCH_RUNGS = ("naive", "lastlogit", "nosync", "gqa", "sdpa", "chunked", "graph", "awq", "triton")
SCRATCH_LADDER = [f"scratch-{r}" for r in SCRATCH_RUNGS]
PREFERRED_ORDER = ["hf-fp16", "fp16", "vllm-fp16", "vllm-awq", *SCRATCH_LADDER, "awq", "fp8",
                   "prefix-off", "prefix-on"]
# Color follows the config, not its rank on a chart: hf-fp16 keeps slot 1 whatever sits beside
# it. Every scratch-* config is "your engine" and shares slot 4 -- the engine chart shows one
# rung at a time, and the ladder chart draws all of them as one line.
ENGINE_SLOT = 3
FIXED_SLOT = {"hf-fp16": 0, "fp16": 0, "prefix-off": 0, "vllm-fp16": 1, "awq": 1, "prefix-on": 1,
              "vllm-awq": 2}
LEGEND_COLS = 4
MODES = (
    ("concurrency", "concurrent requests (log2)", True),
    ("batch_size", "batch size (log2)", True),
    ("request_rate", "offered load (req/s)", False),
)
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
# What the charts call each config and experiment; the result files keep their short labels.
NAMES = {"hf-fp16": "HF Transformers fp16", "vllm-fp16": "vLLM fp16", "vllm-awq": "vLLM AWQ 4-bit",
         "fp16": "fp16", "awq": "AWQ 4-bit", "prefix-off": "prefix caching off", "prefix-on": "prefix caching on"}
TITLES = {"engine": "Engines on fixed batches", "quant": "vLLM server, fp16 vs AWQ 4-bit",
          "prefix": "vLLM server, shared-prefix workload"}


def _name(label: str) -> str:
    if label.startswith("scratch-"):
        return f"VELLUM {label.removeprefix('scratch-')}"
    return NAMES.get(label, label)


def _subtitle(results_dir: Path, level: dict) -> str:
    model = level["model"].rstrip("/").split("/")[-1]
    return f"{model}  ·  {results_dir.name.replace('-', ' ')}  ·  {level['input_len']} in / {level['output_len']} out tokens"


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
        ladder = {lab: run for lab, run in group.items() if lab.startswith("scratch-")}
        if not ladder:
            outputs += _plot_experiment(results_dir, exp, group)
            continue
        others = {lab: run for lab, run in group.items() if lab not in ladder}
        top = max(ladder, key=_rung_index)
        outputs += _plot_experiment(results_dir, exp, {**others, top: ladder[top]})
        if len(ladder) > 1:
            outputs += _plot_ladder(results_dir, exp, ladder, others)
    return outputs


def _rung_index(label: str) -> int:
    rung = label.removeprefix("scratch-")
    return SCRATCH_RUNGS.index(rung) if rung in SCRATCH_RUNGS else -1


def _slot(label: str) -> int | None:
    return ENGINE_SLOT if label.startswith("scratch-") else FIXED_SLOT.get(label)


def _colors(labels) -> dict[str, str]:
    """Known configs take their fixed slot; anything else takes the next free one, in order."""
    slots: dict[str, int] = {}
    for lab in labels:
        s = _slot(lab)
        if s is not None and s not in slots.values():
            slots[lab] = s
    free = iter(s for s in range(len(SERIES)) if s not in slots.values())
    for i, lab in enumerate(labels):
        if lab not in slots:
            slots[lab] = next(free, i)
    return {lab: SERIES[slots[lab] % len(SERIES)] for lab in labels}


def _legend(fig, handles, labels) -> float:
    """Draw the figure legend; return the top of the area left for the panels."""
    if len(labels) <= 3:  # one short row fits right of the title
        fig.legend(handles, labels, loc="upper right", ncol=len(labels), frameon=False, fontsize=9)
        return 0.95
    # A long legend gets its own rows under the title. matplotlib fills legend columns
    # top-down, so interleave the entries to make the rows read left to right in order.
    rows = -(-len(labels) // LEGEND_COLS)
    order = [r * LEGEND_COLS + c for c in range(LEGEND_COLS) for r in range(rows)
             if r * LEGEND_COLS + c < len(labels)]
    fig.legend([handles[i] for i in order], [labels[i] for i in order], loc="upper left",
               bbox_to_anchor=(0.01, 0.955), ncol=LEGEND_COLS, frameon=False, fontsize=9)
    return 0.945 - 0.033 * rows


def _plot_experiment(results_dir: Path, exp: str, runs: dict[str, dict]) -> list[Path]:
    colors = _colors(list(runs))
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
            ends = [_line(ax, [lv["level"] for lv in lvls], [lv[key] for lv in lvls], colors[lab], _name(lab))
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
                      colors[lab], _name(lab)) for lab, lvls in series.items()]
        if direct:
            _direct_labels(ax, ends)

        top = _legend(fig, *axes.flat[0].get_legend_handles_labels())
        fig.suptitle(
            f"{TITLES.get(exp, exp)}  ·  {_subtitle(results_dir, first)}"
            + (f"  ·  {first['shared_prefix_len']}-token shared prefix" if first.get("shared_prefix_len") else ""),
            x=0.01, ha="left", fontsize=12, color=INK,
        )
        fig.tight_layout(rect=(0, 0, 1, top))
        out = results_dir / f"{exp}_{mode}.png"
        fig.savefig(out, dpi=150, facecolor=SURFACE)
        plt.close(fig)
        outputs.append(out)
    return outputs


def _value(v: float) -> str:
    return f"{v:.0f}" if abs(v) >= 100 else f"{v:.1f}"


def _plot_ladder(results_dir: Path, exp: str, ladder: dict[str, dict], refs: dict[str, dict]) -> list[Path]:
    """What each rung bought: the engine rung by rung at fixed batch sizes, with the reference
    engines as labelled horizontal lines. Batch sizes are the smallest and largest that every
    rung completed, so each point compares like with like."""
    def done(run):
        return {lv["level"]: lv for lv in run["levels"] if lv["mode"] == "batch_size" and lv["num_ok"]}

    levels = {lab: done(run) for lab, run in ladder.items()}
    common = set.intersection(*(set(d) for d in levels.values()))
    if not common:
        return []
    small, big = min(common), max(common)
    colors = _colors(list(refs))
    panels = (
        (f"Per-token latency at batch {small}", "ms", "p50_tpot_ms", small),
        (f"Throughput at batch {big}", "output tokens / s", "output_token_throughput", big),
        (f"Time to first token at batch {big}", "ms", "p50_ttft_ms", big),
        (f"Peak GPU memory at batch {big}", "GiB", "peak_mem_gib", big),
    )
    rungs = [lab.removeprefix("scratch-") for lab in ladder]
    xs = list(range(len(rungs)))
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), facecolor=SURFACE)
    for ax, (title, ylabel, key, bs) in zip(axes.flat, panels):
        _style(ax, title, ylabel, "rung (each keeps every rung before it)")
        ys = [levels[lab][bs].get(key) for lab in ladder]
        pts = [(x, y) for x, y in zip(xs, ys) if y is not None]
        if not pts:
            ax.text(0.5, 0.5, "not recorded", transform=ax.transAxes, ha="center", color=INK_2, fontsize=9)
        ends, line_end = [], xs[-1] + 0.35
        for lab, run in refs.items():
            v = done(run).get(bs, {}).get(key)
            if v is not None:  # a reference line stops past the last rung; its label sits beyond
                ax.hlines(v, -0.5, line_end, color=colors[lab], linewidth=1.5, zorder=1, label=_name(lab))
                ends.append((line_end, v, _name(lab)))
        _line(ax, xs, ys, SERIES[ENGINE_SLOT], "VELLUM")
        for x, y in {pts[0], pts[-1]} if pts else ():
            ax.annotate(_value(y), (x, y), textcoords="offset points", xytext=(0, 8), ha="center",
                        fontsize=8, color=INK)
        ax.set_xticks(xs, rungs, rotation=30 if len(rungs) > 8 else 0, ha="right" if len(rungs) > 8 else "center")
        ax.set_xlim(-0.5, line_end + 0.2 * len(xs) + 0.4)  # the right margin holds the reference labels
        ax.margins(y=0.18)
        _direct_labels(ax, ends)
    top = _legend(fig, *axes.flat[0].get_legend_handles_labels())
    first = next(iter(ladder.values()))["levels"][0]
    fig.suptitle(f"VELLUM optimization ladder  ·  {_subtitle(results_dir, first)}", x=0.01, ha="left",
                 fontsize=12, color=INK)
    fig.tight_layout(rect=(0, 0, 1, top))
    out = results_dir / f"{exp}_ladder.png"
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return [out]
