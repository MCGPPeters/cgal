"""Generate a baseline-vs-CGAL comparison report from a Monty matrix run.

Reads ``eval_stats.csv`` (Monty's BASIC logger) for each (experiment, arm)
pair under the logs root, plus optionally OTel JSONL counters from the
collector's file exporter, then writes a markdown report with one
section per experiment and a summary table.

Designed to degrade gracefully — missing arms or missing files just get
``n/a`` rows, the report is always emitted.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

try:
    import pandas as pd
except ImportError:  # pragma: no cover -- pandas is a Monty dep, but be defensive
    pd = None  # type: ignore[assignment]


ARMS = ("baseline", "cgal")


def find_eval_stats(arm_dir: Path) -> Path | None:
    """Monty's BASIC logger writes ``eval_stats.csv`` somewhere under the
    log dir; the exact subfolder depends on the experiment name.
    """
    if not arm_dir.exists():
        return None
    matches = sorted(arm_dir.rglob("eval_stats.csv"))
    return matches[-1] if matches else None


def summarise_eval_stats(csv_path: Path) -> dict[str, Any]:
    """Pull the metrics that matter for a CGAL comparison."""
    if pd is None or csv_path is None or not csv_path.exists():
        return {}
    df = pd.read_csv(csv_path)
    n = len(df)
    if n == 0:
        return {"episodes": 0}

    out: dict[str, Any] = {"episodes": n, "stats_path": str(csv_path)}

    # Accuracy: Monty marks correct via terminal_condition or primary_performance.
    if "primary_performance" in df.columns:
        correct = df["primary_performance"].astype(str).str.lower().isin(
            {"correct", "correct_mlh"}
        )
        out["accuracy"] = float(correct.mean())
        out["n_correct"] = int(correct.sum())
    if "terminal_condition" in df.columns:
        out["terminal_conditions"] = (
            df["terminal_condition"].value_counts().to_dict()
        )

    for col in ("num_steps", "monty_matching_steps", "time"):
        if col in df.columns:
            out[f"mean_{col}"] = float(df[col].mean())

    return out


def load_otel_counters(otel_dir: Path, experiment: str) -> dict[str, dict[str, float]]:
    """Aggregate OTel counter values per arm for one experiment, if available."""
    out: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    metrics_file = otel_dir / "metrics.jsonl"
    if not metrics_file.exists():
        return {}

    for line in metrics_file.read_text().splitlines():
        try:
            doc = json.loads(line)
        except json.JSONDecodeError:
            continue
        for rm in doc.get("resourceMetrics", []):
            attrs = {a["key"]: a.get("value", {}) for a in rm.get("resource", {}).get("attributes", [])}
            exp_attr = attrs.get("experiment", {}).get("stringValue")
            if exp_attr and exp_attr != experiment:
                continue
            for sm in rm.get("scopeMetrics", []):
                for m in sm.get("metrics", []):
                    sum_section = m.get("sum") or m.get("gauge") or {}
                    for dp in sum_section.get("dataPoints", []):
                        dp_attrs = {a["key"]: a.get("value", {}) for a in dp.get("attributes", [])}
                        cgal = dp_attrs.get("cgal_enabled", {}).get("stringValue", "false")
                        arm = "cgal" if cgal == "true" else "baseline"
                        val = dp.get("asInt") or dp.get("asDouble") or 0
                        out[arm][m["name"]] += float(val)
    return {k: dict(v) for k, v in out.items()}


def fmt(v: Any) -> str:
    if v is None:
        return "n/a"
    if isinstance(v, float):
        return f"{v:.3f}"
    if isinstance(v, dict):
        return ", ".join(f"{k}={v}" for k, v in v.items())
    return str(v)


def render(experiments: list[str], data: dict[str, dict[str, dict[str, Any]]],
           otel: dict[str, dict[str, dict[str, float]]]) -> str:
    lines: list[str] = []
    lines.append("# Monty baseline vs. CGAL comparison report")
    lines.append("")
    lines.append("Generated automatically by `generate_report.py`. Each experiment")
    lines.append("was run twice — once with the stock learning module and once with")
    lines.append("all four CGAL mixins enabled (`consensus_gating`, `novelty_detection`,")
    lines.append("`salience_replay`, `trust_weights`).")
    lines.append("")

    # Summary table.
    lines.append("## Summary")
    lines.append("")
    lines.append("| Experiment | Arm | Episodes | Accuracy | Mean steps | Mean time (s) |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: |")
    for exp in experiments:
        for arm in ARMS:
            s = data.get(exp, {}).get(arm, {})
            lines.append(
                f"| {exp} | {arm} | {fmt(s.get('episodes'))} | "
                f"{fmt(s.get('accuracy'))} | {fmt(s.get('mean_num_steps') or s.get('mean_monty_matching_steps'))} | "
                f"{fmt(s.get('mean_time'))} |"
            )
    lines.append("")

    # Delta table.
    lines.append("## Deltas (cgal − baseline)")
    lines.append("")
    lines.append("| Experiment | Δ accuracy | Δ mean steps | Δ mean time (s) |")
    lines.append("| --- | ---: | ---: | ---: |")
    for exp in experiments:
        b = data.get(exp, {}).get("baseline", {})
        c = data.get(exp, {}).get("cgal", {})
        def delta(key: str) -> str:
            if key not in b or key not in c:
                return "n/a"
            return f"{c[key] - b[key]:+.3f}"
        steps_key = "mean_num_steps" if "mean_num_steps" in b else "mean_monty_matching_steps"
        lines.append(f"| {exp} | {delta('accuracy')} | {delta(steps_key)} | {delta('mean_time')} |")
    lines.append("")

    # Per-experiment sections.
    for exp in experiments:
        lines.append(f"## {exp}")
        lines.append("")
        for arm in ARMS:
            s = data.get(exp, {}).get(arm, {})
            lines.append(f"### {arm}")
            lines.append("")
            if not s:
                lines.append("_No `eval_stats.csv` produced — see run.log._")
                lines.append("")
                continue
            lines.append("| Metric | Value |")
            lines.append("| --- | --- |")
            for k, v in s.items():
                lines.append(f"| {k} | {fmt(v)} |")
            lines.append("")
        otel_exp = otel.get(exp, {})
        if otel_exp:
            lines.append("**OTel counters**")
            lines.append("")
            lines.append("| Metric | baseline | cgal |")
            lines.append("| --- | ---: | ---: |")
            keys = sorted(set(otel_exp.get("baseline", {})) | set(otel_exp.get("cgal", {})))
            for k in keys:
                lines.append(
                    f"| {k} | {fmt(otel_exp.get('baseline', {}).get(k))} "
                    f"| {fmt(otel_exp.get('cgal', {}).get(k))} |"
                )
            lines.append("")

    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--logs-root", type=Path, default=Path("/logs"))
    p.add_argument("--otel-dir", type=Path, default=Path("/otel-out"))
    p.add_argument("--experiments", nargs="+", required=True)
    p.add_argument("--output", type=Path, default=Path("/logs/COMPARISON_REPORT.md"))
    args = p.parse_args(argv)

    data: dict[str, dict[str, dict[str, Any]]] = {}
    otel: dict[str, dict[str, dict[str, float]]] = {}
    for exp in args.experiments:
        data[exp] = {}
        for arm in ARMS:
            arm_dir = args.logs_root / exp / arm
            data[exp][arm] = summarise_eval_stats(find_eval_stats(arm_dir))
        otel[exp] = load_otel_counters(args.otel_dir, exp)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render(args.experiments, data, otel))
    print(f"[generate_report] wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
