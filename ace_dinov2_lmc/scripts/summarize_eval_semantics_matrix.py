#!/usr/bin/env python3
"""Summarize the common-scale x effective-ratio-cap evaluation matrix."""

from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path


METRICS = {
    "25cm_5deg": ("accuracy_25cm5deg_pct", max),
    "10cm_5deg": ("accuracy_10cm5deg_pct", max),
    "5cm_5deg": ("accuracy_5cm5deg_pct", max),
    "2cm_2deg": ("accuracy_2cm2deg_pct", max),
    "1cm_1deg": ("accuracy_1cm1deg_pct", max),
    "median_deg": ("median_rotation_deg", min),
    "median_cm": ("median_translation_cm", min),
}
CONFIG_RE = re.compile(r"^c(?P<common>[01])_cap(?P<cap>0|006|008)$")


def read_kv(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#") or "\t" not in line:
            continue
        key, value = line.split("\t", 1)
        values[key] = value
    return values


def expected_semantics(config: str) -> tuple[float, float]:
    match = CONFIG_RE.fullmatch(config)
    if match is None:
        raise ValueError(f"Unrecognized config directory: {config}")
    common = float(match.group("common"))
    cap = {"0": 0.0, "006": 0.006, "008": 0.008}[match.group("cap")]
    return common, cap


def close(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=0.0, abs_tol=1e-9)


def summarize(root: Path) -> list[dict[str, str]]:
    grouped: dict[tuple[str, str], list[tuple[Path, dict[str, str]]]] = {}
    for path in sorted(root.glob("*/wayspots_*/eval_summary_*.txt")):
        config = path.parents[1].name
        scene = path.parent.name
        grouped.setdefault((config, scene), []).append((path, read_kv(path)))

    if not grouped:
        raise RuntimeError(f"No eval_summary files found under {root}")

    rows: list[dict[str, str]] = []
    for (config, scene), candidates in sorted(grouped.items()):
        common, cap = expected_semantics(config)
        if len(candidates) != 3:
            raise RuntimeError(f"Expected 3 seeds for {config}/{scene}, found {len(candidates)}")
        for path, values in candidates:
            effective_common = float(values["eval_lmc_fusion_reread_common_scale_effective"])
            effective_cap = float(values["eval_lmc_fusion_reread_effective_ratio_cap_effective"])
            if not close(effective_common, common) or not close(effective_cap, cap):
                raise RuntimeError(
                    f"Semantic mismatch in {path}: effective=({effective_common}, {effective_cap}), "
                    f"expected=({common}, {cap})"
                )

        row = {
            "scene": scene,
            "config": config,
            "common_scale": f"{common:g}",
            "effective_ratio_cap": f"{cap:g}",
            "num_seeds": str(len(candidates)),
            "semantic_status": "ok",
        }
        for metric, (input_key, selector) in METRICS.items():
            values = [(float(candidate[input_key]), path) for path, candidate in candidates]
            best_value, best_path = selector(values, key=lambda item: item[0])
            row[metric] = f"{best_value:.4f}"
            row[f"source_{metric}"] = str(best_path)
        rows.append(row)
    return rows


def write_outputs(root: Path, rows: list[dict[str, str]]) -> None:
    metric_names = list(METRICS)
    fields = [
        "scene", "config", "common_scale", "effective_ratio_cap", "num_seeds",
        "semantic_status", *metric_names, *(f"source_{name}" for name in metric_names),
    ]
    with (root / "summary.tsv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# PMRF Evaluation Semantics Matrix",
        "",
        "Each metric is independently selected across all three deterministic evaluation seeds.",
        "",
        "| scene | config | Acc25 | Acc10 | Acc5 | Acc2 | Acc1 | MedR | MedT cm |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['scene']} | {row['config']} | {row['25cm_5deg']} | "
            f"{row['10cm_5deg']} | {row['5cm_5deg']} | {row['2cm_2deg']} | "
            f"{row['1cm_1deg']} | {row['median_deg']} | {row['median_cm']} |"
        )
    (root / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("matrix_root", type=Path)
    args = parser.parse_args()
    root = args.matrix_root.resolve()
    rows = summarize(root)
    write_outputs(root, rows)
    print(root / "summary.tsv")
    print(root / "summary.md")


if __name__ == "__main__":
    main()
