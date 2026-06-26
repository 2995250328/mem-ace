#!/usr/bin/env python3
"""Inventory legacy flat 04_evaluation directories and suggest canonical parents."""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path


DEFAULT_BASE = Path("/data/xwh/ace_dinov2_lmc/04_evaluation")


def classify_dataset(name: str) -> tuple[str, str]:
    n = name.lower()
    if "cambridge" in n or any(s in n for s in ["greatcourt", "kings", "hospital", "shop", "stmary"]):
        return "cambridge", "cambridge keyword"
    if "glace_official" in n:
        return "cambridge", "glace official keyword"
    if "indoor6" in n or re.search(r"\bscene[0-9]", n):
        return "indoor6", "indoor6/scene keyword"
    if "wayspots" in n or any(s in n for s in ["bears", "squarebench", "cubes", "tendrils"]):
        return "wayspots", "wayspots scene keyword"
    if any(s in n for s in ["pmrf", "qknorm", "reread", "centered", "quick_v3", "highhypo_pmrf"]):
        return "wayspots", "fusion keyword defaults to wayspots"
    if any(s in n for s in ["dsac", "gpu", "train_logs", "train_compare", "concurrency"]):
        return "shared", "infrastructure keyword"
    return "shared", "unknown dataset"


def classify_track(name: str) -> tuple[str, str]:
    n = name.lower()
    if any(s in n for s in ["dsac", "concurrency", "probe", "train_logs", "train_compare"]):
        return "diagnostics", "diagnostics keyword"
    if any(s in n for s in ["repro", "restore", "legacy", "old", "paper_results"]):
        return "reproduction", "reproduction keyword"
    if any(s in n for s in ["sparse_depth", "extracted", "memory"]):
        return "memory", "memory keyword"
    if any(s in n for s in ["baseline", "glace_official", "ace_fcn"]):
        return "baseline", "baseline keyword"
    if "stage2" in n:
        return "stage2", "stage2 keyword"
    if any(s in n for s in ["pmrf", "fusion", "qknorm", "centered", "reread", "adapter", "v3"]):
        return "fusion", "fusion keyword"
    if "stage1" in n or "single" in n or "local_stage1" in n or "global_long" in n:
        return "stage1", "stage1/single keyword"
    if "efficiency" in n:
        return "training_efficiency", "efficiency keyword"
    return "scratch", "unknown track"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--output", type=Path, default=None, help="Write TSV to this path; default stdout")
    args = parser.parse_args()

    rows = []
    for path in sorted(args.base.iterdir()):
        if not path.is_dir():
            continue
        name = path.name
        if name.startswith("_") or name in {"wayspots", "cambridge", "indoor6", "shared"}:
            continue
        dataset, dataset_reason = classify_dataset(name)
        track, track_reason = classify_track(name)
        suggested_parent = args.base / dataset / track / "legacy_flat"
        rows.append(
            {
                "name": name,
                "dataset": dataset,
                "track": track,
                "dataset_reason": dataset_reason,
                "track_reason": track_reason,
                "suggested_parent": str(suggested_parent),
                "suggested_path": str(suggested_parent / name),
            }
        )

    out = args.output.open("w", newline="") if args.output else sys.stdout
    try:
        writer = csv.DictWriter(
            out,
            fieldnames=[
                "name",
                "dataset",
                "track",
                "dataset_reason",
                "track_reason",
                "suggested_parent",
                "suggested_path",
            ],
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerows(rows)
    finally:
        if args.output:
            out.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
