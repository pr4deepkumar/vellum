"""Markdown summary tables from sweep results (paste into README / write-up)."""

from __future__ import annotations

from pathlib import Path

from .plot import load_runs

_LOAD_FMT = {"concurrency": "c={}", "request_rate": "{} rps", "batch_size": "bs={}"}


def _n(v, fmt="{:.1f}") -> str:
    return "–" if v is None else fmt.format(v)


def cost_per_million(tok_per_s: float | None, gpu_cost_per_hour: float | None) -> float | None:
    """USD per 1M tokens if one GPU, billed by the hour, is kept busy at this rate."""
    if not tok_per_s or gpu_cost_per_hour is None:
        return None
    return gpu_cost_per_hour / (tok_per_s * 3600) * 1e6


def _level_table(run: dict, cost: float | None) -> list[str]:
    head = "| load | out tok/s | req/s | TTFT p50 | TTFT p99 | TPOT p50 | TPOT p99 | E2E p99 | peak KV | fails |"
    sep = "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
    if cost is not None:
        head += " $/1M out |"
        sep += "---:|"
    rows = [head, sep]
    for lv in run["levels"]:
        row = (
            f"| {_LOAD_FMT[lv['mode']].format(lv['level'])} | {_n(lv['output_token_throughput'], '{:.0f}')} "
            f"| {_n(lv['request_throughput'], '{:.2f}')} | {_n(lv['p50_ttft_ms'])} | {_n(lv['p99_ttft_ms'])} "
            f"| {_n(lv['p50_tpot_ms'])} | {_n(lv['p99_tpot_ms'])} | {_n(lv['p99_e2e_ms'], '{:.0f}')} "
            f"| {_n(lv.get('max_kv_cache_usage'), '{:.0%}')} | {lv['num_failed']} |"
        )
        if cost is not None:
            row += f" {_n(cost_per_million(lv['output_token_throughput'], cost), '${:.3f}')} |"
        rows.append(row)
    return rows


def _comparison(runs: dict[str, dict], mode: str, cost: float | None) -> list[str]:
    """Head-to-head at the highest load level every config in this mode reached."""
    in_mode = {lab: r for lab, r in runs.items() if any(lv["mode"] == mode for lv in r["levels"])}
    if len(in_mode) < 2:
        return []
    # Only levels every config completed (e.g. HF may OOM at large batch sizes).
    common = set.intersection(*[{lv["level"] for lv in r["levels"] if lv["mode"] == mode and lv["num_ok"]}
                                for r in in_mode.values()])
    if not common:
        return []
    level = max(common)
    pick = {lab: next(lv for lv in r["levels"] if lv["mode"] == mode and lv["level"] == level and lv["num_ok"])
            for lab, r in in_mode.items()}
    base_label = next(iter(pick))
    base = pick[base_label]["output_token_throughput"]
    out = [f"### Head-to-head at {_LOAD_FMT[mode].format(level)} (relative to {base_label})", ""]
    head, sep = "| config | out tok/s | speed-up | TTFT p50 | TPOT p50 |", "|---|---:|---:|---:|---:|"
    if cost is not None:
        head += " $/1M out |"
        sep += "---:|"
    out += [head, sep]
    for lab, lv in pick.items():
        row = (f"| {lab} | {lv['output_token_throughput']:.0f} | {lv['output_token_throughput'] / base:.2f}× "
               f"| {_n(lv['p50_ttft_ms'])} ms | {_n(lv['p50_tpot_ms'])} ms |")
        if cost is not None:
            row += f" {_n(cost_per_million(lv['output_token_throughput'], cost), '${:.3f}')} |"
        out.append(row)
    return out + [""]


def report_dir(results_dir: Path, gpu_cost_per_hour: float | None = None) -> str:
    runs = load_runs(results_dir)
    lines = []
    if gpu_cost_per_hour is not None:
        lines += [f"_Cost assumes one GPU at ${gpu_cost_per_hour:.2f}/hr kept fully busy at the measured rate._", ""]
    for label, run in runs.items():
        meta = run["meta"]
        lines.append(f"### {label}  (`{meta['model']}`)")
        if meta.get("server_args"):
            lines.append(f"\n`{meta['server_args']}`")
        lines.append("")
        lines += _level_table(run, gpu_cost_per_hour)
        lines.append("")
    experiments: dict[str, dict[str, dict]] = {}
    for label, run in runs.items():
        experiments.setdefault(run["meta"].get("experiment", "main"), {})[label] = run
    for exp, group in experiments.items():
        for mode in _LOAD_FMT:
            block = _comparison(group, mode, gpu_cost_per_hour)
            if block:
                block[0] = block[0].replace("### ", f"### [{exp}] ", 1)
            lines += block
    return "\n".join(lines)
