#!/usr/bin/env python3
"""Aggregate ACE-DINOv2-LMC summary.tsv files by metric-wise best values."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Dict, Iterable, Tuple


HIGHER_IS_BETTER = [
    "50cm_5deg",
    "25cm_5deg",
    "10cm_5deg",
    "5cm_5deg",
    "2cm_2deg",
    "1cm_1deg",
]
LOWER_IS_BETTER = ["median_deg", "median_cm"]
METRICS = HIGHER_IS_BETTER + LOWER_IS_BETTER


def parse_float(value: str) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(parsed) or math.isinf(parsed):
        return None
    return parsed


def is_better(metric: str, candidate: float, current: float | None) -> bool:
    if current is None:
        return True
    if metric in LOWER_IS_BETTER:
        return candidate < current
    return candidate > current


def iter_summary_rows(run_root: Path, summary_name: str) -> Iterable[Tuple[Path, Dict[str, str]]]:
    for summary_path in sorted(run_root.rglob(summary_name)):
        with summary_path.open("r", newline="") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                row["_summary_path"] = str(summary_path)
                yield summary_path, row


def variant_from_path(run_root: Path, summary_path: Path) -> str:
    parent = summary_path.parent
    try:
        rel = parent.relative_to(run_root)
    except ValueError:
        return parent.name or "."
    if str(rel) == ".":
        return "."
    return rel.parts[0]


def metric_source(row: Dict[str, str], metric: str) -> str:
    return row.get(f"source_{metric}") or row.get("output") or row.get("_summary_path", "")


def aggregate(run_root: Path, summary_name: str) -> Dict[Tuple[str, str, str], Dict[str, object]]:
    groups: Dict[Tuple[str, str, str], Dict[str, object]] = {}

    for summary_path, row in iter_summary_rows(run_root, summary_name):
        if row.get("status") and row["status"] != "ok":
            continue
        scene = row.get("scene", "")
        method = row.get("method", "")
        if not scene or not method:
            continue

        variant = variant_from_path(run_root, summary_path)
        key = (variant, scene, method)
        group = groups.setdefault(
            key,
            {
                "variant": variant,
                "scene": scene,
                "method": method,
                "best": {},
                "sources": {},
            },
        )

        for metric in METRICS:
            value = parse_float(row.get(metric, ""))
            if value is None:
                continue
            best = group["best"].get(metric)
            if is_better(metric, value, best):
                group["best"][metric] = value
                group["sources"][metric] = metric_source(row, metric)

    return groups


def format_value(value: float | None) -> str:
    if value is None:
        return "nan"
    return f"{value:.3f}"


def print_markdown(groups: Dict[Tuple[str, str, str], Dict[str, object]], include_sources: bool) -> None:
    header = ["variant", "scene", "method", *METRICS]
    print("| " + " | ".join(header) + " |")
    print("| " + " | ".join(["---"] * len(header)) + " |")
    for key in sorted(groups):
        group = groups[key]
        best = group["best"]
        row = [
            str(group["variant"]),
            str(group["scene"]),
            str(group["method"]),
            *[format_value(best.get(metric)) for metric in METRICS],
        ]
        print("| " + " | ".join(row) + " |")

    if include_sources:
        print()
        print("Metric sources:")
        for key in sorted(groups):
            group = groups[key]
            label = f"{group['variant']} / {group['scene']} / {group['method']}"
            print(f"\n{label}")
            for metric in METRICS:
                source = group["sources"].get(metric)
                if source:
                    print(f"- {metric}: {source}")


def write_tsv(groups: Dict[Tuple[str, str, str], Dict[str, object]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as f:
        fieldnames = ["variant", "scene", "method", *METRICS, *[f"source_{m}" for m in METRICS]]
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for key in sorted(groups):
            group = groups[key]
            best = group["best"]
            sources = group["sources"]
            row = {
                "variant": group["variant"],
                "scene": group["scene"],
                "method": group["method"],
            }
            for metric in METRICS:
                row[metric] = format_value(best.get(metric))
                row[f"source_{metric}"] = sources.get(metric, "")
            writer.writerow(row)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root", type=Path, help="Run root to scan recursively for summary.tsv files.")
    parser.add_argument("--summary-name", default="summary.tsv", help="Summary filename to scan for.")
    parser.add_argument("--sources", action="store_true", help="Print per-metric source paths after the table.")
    parser.add_argument("--tsv", type=Path, help="Optional path to write a machine-readable TSV.")
    args = parser.parse_args()

    run_root = args.run_root.expanduser().resolve()
    groups = aggregate(run_root, args.summary_name)
    if not groups:
        raise SystemExit(f"No ok rows found under {run_root} with name {args.summary_name}")

    print_markdown(groups, include_sources=args.sources)
    if args.tsv:
        write_tsv(groups, args.tsv.expanduser().resolve())
        print(f"\nWrote TSV: {args.tsv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
