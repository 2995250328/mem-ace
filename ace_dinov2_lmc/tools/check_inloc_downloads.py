#!/usr/bin/env python3
"""Audit staged InLoc archives and flag HTML/corrupt downloads early."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


EXPECTED_FILES = [
    "cutouts.tar.gz",
    "iphone7_query.tar.gz",
    "DUC1_cutouts.tar.bz2",
    "DUC2_cutouts_1.tar.gz",
    "DUC2_cutouts_2.tar.gz",
    "CSE3_cutouts.tar.bz2",
    "CSE4_cutouts_1.tar.gz",
    "CSE4_cutouts_2.tar.gz",
    "CSE5_cutouts_1.tar.gz",
    "CSE5_cutouts_2.tar.gz",
    "DUC1_scans.tar.bz2",
    "DUC2_scans.tar.bz2",
    "CSE3_scans.tar.bz2",
    "CSE4_scans.tar.bz2",
    "CSE5_scans.tar.bz2",
    "DUC1_align.zip",
    "DUC2_align.zip",
    "CSE3_align.zip",
    "CSE4_align.zip",
    "CSE5_align.zip",
]


def _classify_file(path: Path) -> tuple[str, str]:
    if not path.exists():
        return "missing", "missing"
    size = path.stat().st_size
    with path.open("rb") as handle:
        head = handle.read(512)
    lowered = head.lower()
    if lowered.startswith(b"<!doctype html") or lowered.startswith(b"<html") or b"<html" in lowered[:256]:
        return "html", "html_or_redirect"
    if head.startswith(b"\x1f\x8b"):
        return "ok", "gzip"
    if head.startswith(b"BZh"):
        return "ok", "bzip2"
    if head.startswith(b"PK\x03\x04"):
        return "ok", "zip"
    if size < 1024 * 1024:
        return "suspect", "too_small"
    return "suspect", "unknown_magic"


def audit(raw_root: Path) -> dict[str, object]:
    files = []
    for name in EXPECTED_FILES:
        path = raw_root / name
        status, kind = _classify_file(path)
        files.append(
            {
                "name": name,
                "exists": path.exists(),
                "size_bytes": path.stat().st_size if path.exists() else 0,
                "status": status,
                "kind": kind,
                "path": str(path),
            }
        )
    return {
        "raw_root": str(raw_root),
        "expected_count": len(EXPECTED_FILES),
        "ok_count": sum(item["status"] == "ok" for item in files),
        "missing_count": sum(item["status"] == "missing" for item in files),
        "html_count": sum(item["status"] == "html" for item in files),
        "suspect_count": sum(item["status"] == "suspect" for item in files),
        "files": files,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit staged InLoc raw archives.")
    parser.add_argument("--raw-root", type=Path, default=Path("/data/xwh/dataset_staging/inloc/raw"))
    parser.add_argument("--write-json", type=Path, default=None, help="Optional output JSON path.")
    args = parser.parse_args()

    report = audit(args.raw_root)
    if args.write_json is not None:
        args.write_json.parent.mkdir(parents=True, exist_ok=True)
        args.write_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
