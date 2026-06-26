#!/usr/bin/env python3
"""Create a canonical ACE-DINOv2-LMC evaluation run root and manifest."""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import re
from pathlib import Path


DEFAULT_BASE = Path("/data/xwh/ace_dinov2_lmc/04_evaluation")
PROJECT_ROOT = Path("/home/xwh/project/ace_depth")

DATASETS = {"wayspots", "cambridge", "indoor6", "shared"}
TRACKS = {
    "baseline",
    "stage1",
    "stage2",
    "fusion",
    "training_efficiency",
    "reproduction",
    "diagnostics",
    "paper_results",
    "memory",
    "scratch",
}


def slug(value: str, *, allow_empty: bool = False) -> str:
    value = value.strip().lower()
    value = value.replace("+", "plus")
    value = re.sub(r"[\s/\\:]+", "_", value)
    value = re.sub(r"[^a-z0-9._-]+", "", value)
    value = re.sub(r"_+", "_", value).strip("._-")
    if not value and not allow_empty:
        raise ValueError("empty slug after normalization")
    if ".." in value:
        raise ValueError(f"unsafe slug: {value!r}")
    return value


def normalize_gpu_tag(value: str) -> str:
    value = value.strip().lower()
    if value in {"", "none", "no", "nogpu"}:
        return "nogpu"
    if value in {"cpu"}:
        return "cpu"
    if value.startswith("gpu"):
        value = value[3:]
    digits = re.sub(r"[^0-9]", "", value)
    if not digits:
        raise ValueError(f"invalid GPU tag: {value!r}")
    return f"gpu{digits}"


def parse_kv(items: list[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"--extra-manifest entries must be key=value, got {item!r}")
        key, value = item.split("=", 1)
        result[slug(key)] = value
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--dataset", required=True, choices=sorted(DATASETS))
    parser.add_argument("--track", required=True, choices=sorted(TRACKS))
    parser.add_argument("--method", required=True, help="Stable method slug, e.g. concat_glace")
    parser.add_argument("--scope", required=True, help="Scene-set slug, e.g. court_stmary or all")
    parser.add_argument("--protocol", required=True, help="Main schedule/eval contract")
    parser.add_argument("--gpus", required=True, help="0, 01, gpu23, cpu, or nogpu")
    parser.add_argument("--date", default=_dt.date.today().strftime("%Y%m%d"))
    parser.add_argument("--scenes", nargs="*", default=[])
    parser.add_argument("--launch-script", default="")
    parser.add_argument("--status", default="planned", choices=["planned", "running", "complete", "failed", "archived"])
    parser.add_argument("--metric-policy", default="")
    parser.add_argument("--extra-manifest", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--create", action="store_true")
    parser.add_argument("--allow-existing", action="store_true")
    parser.add_argument("--print-export", action="store_true")
    args = parser.parse_args()

    if not re.fullmatch(r"\d{8}", args.date):
        raise SystemExit(f"--date must be YYYYMMDD, got {args.date!r}")

    method = slug(args.method)
    scope = slug(args.scope)
    protocol = slug(args.protocol)
    gpu_tag = normalize_gpu_tag(args.gpus)
    leaf = f"{args.date}_{scope}_{protocol}_{gpu_tag}"
    run_root = args.base / args.dataset / args.track / method / leaf

    manifest = {
        "created_at": _dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "project_root": str(PROJECT_ROOT),
        "run_root": str(run_root),
        "dataset": args.dataset,
        "track": args.track,
        "method": method,
        "scope": scope,
        "protocol": protocol,
        "gpu_tag": gpu_tag,
        "scenes": args.scenes,
        "launch_script": args.launch_script,
        "status": args.status,
        "metric_policy": args.metric_policy,
    }
    manifest.update(parse_kv(args.extra_manifest))

    if args.create:
        if run_root.exists() and not args.allow_existing:
            raise SystemExit(f"run root already exists: {run_root}\nUse --allow-existing if this is intentional.")
        run_root.mkdir(parents=True, exist_ok=True)
        for child in ["logs", "summaries", "artifacts", "scripts"]:
            (run_root / child).mkdir(exist_ok=True)
        manifest_path = run_root / "manifest.json"
        if manifest_path.exists() and not args.allow_existing:
            raise SystemExit(f"manifest already exists: {manifest_path}\nUse --allow-existing if this is intentional.")
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    if args.print_export:
        print(f"export RUN_ROOT={run_root}")
    else:
        print(f"RUN_ROOT={run_root}")
    print(f"MANIFEST={run_root / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
