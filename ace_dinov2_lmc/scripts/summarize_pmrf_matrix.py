#!/usr/bin/env python3
"""Summarize a PMRF Stage1 matrix with single/pmrf/adapter_ffn runs."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, List, Optional

METHODS = ("single", "pmrf", "adapter_ffn")
METRICS = (
    "50cm_5deg", "25cm_5deg", "10cm_5deg", "5cm_5deg",
    "2cm_2deg", "1cm_1deg", "median_deg", "median_cm",
)

RUNTIME_RE = re.compile(
    r"\[LMC-(?:Eval)?Runtime\]\[(?P<stage>[^\]]+)\].*?call=(?P<call>\d+)"
    r".*?entropy_mean=(?P<entropy_mean>[-+0-9.eE]+)"
    r".*?effective_tokens=(?P<effective_tokens>[-+0-9.eE]+)"
    r".*?avg_max=(?P<avg_max>[-+0-9.eE]+)"
    r".*?raw_norm=(?P<raw_norm>[-+0-9.eE]+)"
    r".*?attn_out_norm=(?P<attn_out_norm>[-+0-9.eE]+)"
    r".*?fused_norm=(?P<fused_norm>[-+0-9.eE]+)"
)
EXTRA_RE = re.compile(r"extra=\{(?P<extra>[^}]*)\}")


def to_float(value: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def read_stage1_summary(summary_path: Path) -> Dict[str, object]:
    with summary_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            if row.get("method") == "ace_fcn_stage1" and row.get("status") == "ok":
                out: Dict[str, object] = {"status": "ok", "frames": int(row.get("frames") or 0)}
                for metric in METRICS:
                    out[metric] = to_float(row.get(metric, "nan"))
                out["output"] = row.get("output", "")
                return out
    return {"status": "missing"}


def parse_extra(extra: str) -> Dict[str, object]:
    parsed: Dict[str, object] = {}
    for token in extra.split():
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        if value in ("True", "False"):
            parsed[key] = value == "True"
            continue
        try:
            parsed[key] = float(value)
        except ValueError:
            parsed[key] = value
    return parsed


def read_runtime(log_path: Path, max_rows: int = 50000) -> Dict[str, object]:
    rows: List[Dict[str, object]] = []
    if not log_path.exists():
        return {"rows": 0}
    with log_path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if "[LMC-Runtime]" not in line and "[LMC-EvalRuntime]" not in line:
                continue
            match = RUNTIME_RE.search(line)
            if not match:
                continue
            row: Dict[str, object] = {
                "stage": match.group("stage"),
                "call": int(match.group("call")),
                "entropy_mean": to_float(match.group("entropy_mean")),
                "effective_tokens": to_float(match.group("effective_tokens")),
                "avg_max": to_float(match.group("avg_max")),
                "raw_norm": to_float(match.group("raw_norm")),
                "attn_out_norm": to_float(match.group("attn_out_norm")),
                "fused_norm": to_float(match.group("fused_norm")),
            }
            extra_match = EXTRA_RE.search(line)
            row.update(parse_extra(extra_match.group("extra") if extra_match else ""))
            rows.append(row)
            if len(rows) >= max_rows:
                break
    if not rows:
        return {"rows": 0}
    summary: Dict[str, object] = {"rows": len(rows)}
    for key in ("entropy_mean", "effective_tokens", "avg_max", "raw_norm", "attn_out_norm", "fused_norm"):
        vals = [float(r[key]) for r in rows if isinstance(r.get(key), (int, float))]
        if vals:
            summary[f"runtime_{key}_mean"] = mean(vals)
            summary[f"runtime_{key}_last"] = vals[-1]
    for key in (
        "reread_delta_norm", "anchor_reread_cosine",
        "adapter_delta_norm", "anchor_adapter_cosine",
        "a1_a2_js_mean", "a1_a2_js_p50", "a1_a2_js_p90",
        "a1_a2_top1_agreement", "a1_a2_top5_overlap",
    ):
        vals = [float(r[key]) for r in rows if isinstance(r.get(key), (int, float))]
        if vals:
            summary[f"runtime_{key}_mean"] = mean(vals)
            summary[f"runtime_{key}_last"] = vals[-1]
    return summary


def subtract(a: Dict[str, object], b: Dict[str, object], keys: Iterable[str]) -> Dict[str, float]:
    return {key: float(a.get(key, float("nan"))) - float(b.get(key, float("nan"))) for key in keys}


def format_float(value: object) -> str:
    if not isinstance(value, (int, float)):
        return ""
    return f"{float(value):.4f}"


def write_tsv(path: Path, rows: Dict[str, Dict[str, object]]) -> None:
    runtime_keys = sorted({key for row in rows.values() for key in row if key.startswith("runtime_")})
    fieldnames = ["method", "status", "frames", *METRICS, *runtime_keys, "output"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for method, row in rows.items():
            writer.writerow({key: row.get(key, "") for key in fieldnames} | {"method": method})


def write_md(path: Path, rows: Dict[str, Dict[str, object]]) -> None:
    lines = []
    lines.append("# PMRF Matrix Summary")
    lines.append("")
    lines.append("| method | 50cm/5deg | 25cm/5deg | 10cm/5deg | 5cm/5deg | 2cm/2deg | 1cm/1deg | med rot | med trans |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for method in METHODS:
        row = rows.get(method, {})
        lines.append(
            f"| {method} | {format_float(row.get('50cm_5deg'))} | {format_float(row.get('25cm_5deg'))} | "
            f"{format_float(row.get('10cm_5deg'))} | {format_float(row.get('5cm_5deg'))} | "
            f"{format_float(row.get('2cm_2deg'))} | {format_float(row.get('1cm_1deg'))} | "
            f"{format_float(row.get('median_deg'))} | {format_float(row.get('median_cm'))} |"
        )
    lines.append("")
    if "single" in rows:
        lines.append("## Delta vs single")
        lines.append("")
        lines.append("| method | 10cm/5deg | 5cm/5deg | 2cm/2deg | med trans |")
        lines.append("| --- | ---: | ---: | ---: | ---: |")
        base = rows["single"]
        for method in ("pmrf", "adapter_ffn"):
            if method in rows:
                d = subtract(rows[method], base, ("10cm_5deg", "5cm_5deg", "2cm_2deg", "median_cm"))
                lines.append(
                    f"| {method} | {d['10cm_5deg']:.4f} | {d['5cm_5deg']:.4f} | "
                    f"{d['2cm_2deg']:.4f} | {d['median_cm']:.4f} |"
                )
    if "pmrf" in rows and "adapter_ffn" in rows:
        lines.append("")
        lines.append("## Delta PMRF vs adapter_ffn")
        d = subtract(rows["pmrf"], rows["adapter_ffn"], ("10cm_5deg", "5cm_5deg", "2cm_2deg", "median_cm"))
        lines.append("")
        lines.append(f"- 10cm/5deg: {d['10cm_5deg']:.4f}")
        lines.append(f"- 5cm/5deg: {d['5cm_5deg']:.4f}")
        lines.append(f"- 2cm/2deg: {d['2cm_2deg']:.4f}")
        lines.append(f"- median translation cm: {d['median_cm']:.4f}")
    runtime_display = (
        "runtime_reread_delta_norm_mean", "runtime_anchor_reread_cosine_mean",
        "runtime_adapter_delta_norm_mean", "runtime_anchor_adapter_cosine_mean",
        "runtime_a1_a2_js_mean_mean", "runtime_a1_a2_top1_agreement_mean",
        "runtime_a1_a2_top5_overlap_mean",
    )
    if any(any(key in row for key in runtime_display) for row in rows.values()):
        lines.append("")
        lines.append("## Runtime Refinement Diagnostics")
        lines.append("")
        lines.append("| method | reread_delta | adapter_delta | reread_cos | adapter_cos | JS(A1,A2) | top1 agree | top5 overlap |")
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
        for method in METHODS:
            row = rows.get(method, {})
            lines.append(
                f"| {method} | {format_float(row.get('runtime_reread_delta_norm_mean'))} | "
                f"{format_float(row.get('runtime_adapter_delta_norm_mean'))} | "
                f"{format_float(row.get('runtime_anchor_reread_cosine_mean'))} | "
                f"{format_float(row.get('runtime_anchor_adapter_cosine_mean'))} | "
                f"{format_float(row.get('runtime_a1_a2_js_mean_mean'))} | "
                f"{format_float(row.get('runtime_a1_a2_top1_agreement_mean'))} | "
                f"{format_float(row.get('runtime_a1_a2_top5_overlap_mean'))} |"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("matrix_root", type=Path)
    parser.add_argument("--output-prefix", type=Path, default=None)
    args = parser.parse_args()

    root = args.matrix_root
    rows: Dict[str, Dict[str, object]] = {}
    for method in METHODS:
        method_root = root / method
        row = read_stage1_summary(method_root / "summary.tsv")
        log_path = method_root / "stage1_local_ace_memory_it12" / "train_wayspots_bears.log"
        row.update(read_runtime(log_path))
        rows[method] = row

    output_prefix = args.output_prefix or (root / "pmrf_matrix_summary")
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    write_tsv(output_prefix.with_suffix(".tsv"), rows)
    write_md(output_prefix.with_suffix(".md"), rows)
    output_prefix.with_suffix(".json").write_text(json.dumps(rows, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote {output_prefix.with_suffix('.tsv')}")
    print(f"Wrote {output_prefix.with_suffix('.md')}")
    print(f"Wrote {output_prefix.with_suffix('.json')}")


if __name__ == "__main__":
    main()
