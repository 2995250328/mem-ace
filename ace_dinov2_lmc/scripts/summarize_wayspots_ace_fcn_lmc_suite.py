#!/usr/bin/env python3
"""Summarize Wayspots ACE/GLACE baselines plus ACE-FCN two-stage LMC runs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path


THRESHOLDS = (
    ("50cm_5deg", "50cm/5deg"),
    ("25cm_5deg", "25cm/5deg"),
    ("10cm_5deg", "10cm/5deg"),
    ("5cm_5deg", "5cm/5deg"),
    ("2cm_2deg", "2cm/2deg"),
    ("1cm_1deg", "1cm/1deg"),
)


def _fmt(value: float | int | str | None) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, int):
        return str(value)
    if math.isnan(value):
        return "nan"
    return f"{value:.3f}"


def _read_tsv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", errors="replace") as f:
        return list(csv.DictReader(f, delimiter="	"))


def _read_devices(status_path: Path) -> dict[tuple[str, str], tuple[str, str]]:
    rows = _read_tsv_rows(status_path)
    devices: dict[tuple[str, str], tuple[str, str]] = {}
    for row in rows:
        scene = row.get("scene", "")
        method = row.get("method", "")
        if not scene or not method or method == "all":
            continue
        train_device = row.get("train_device") or row.get("device") or ""
        eval_device = row.get("eval_device") or row.get("device") or ""
        devices[(scene, method)] = (train_device, eval_device)
    return devices


def _find_latest_post_train_eval(root: Path) -> Path | None:
    matches = sorted(root.glob("**/post_train_eval.txt"))
    return matches[-1] if matches else None


def _find_latest_post_train_eval_any(roots: list[Path]) -> Path | None:
    matches: list[Path] = []
    for root in roots:
        if root.exists():
            matches.extend(root.glob("**/post_train_eval.txt"))
    if not matches:
        return None
    return max(matches, key=lambda p: (p.stat().st_mtime, str(p)))


def _stage_roots(suite_root: Path, stage_dirname: str, scene: str) -> list[Path]:
    roots = [suite_root / stage_dirname / "extracted" / scene / "dino_ace_lmc_ace_g"]
    for stage_dir in sorted(suite_root.glob(f"{stage_dirname}*")):
        if stage_dir.name == stage_dirname:
            continue
        candidate = stage_dir / "extracted" / scene / "dino_ace_lmc_ace_g"
        if candidate.exists():
            roots.append(candidate)
    return roots


def _parse_post_train_eval(path: Path) -> dict[str, float | int | str]:
    text = path.read_text(encoding="utf-8", errors="replace")
    kv: dict[str, str] = {}
    for line in text.splitlines():
        if "\t" not in line:
            continue
        key, value = line.split("\t", 1)
        kv[key.strip()] = value.strip()
    if "median_rotation_deg" in kv:
        return {
            "status": "ok",
            "frames": int(kv.get("total_frames", 0) or 0),
            "median_deg": float(kv["median_rotation_deg"]),
            "median_cm": float(kv["median_translation_cm"]),
            "50cm_5deg": float("nan"),
            "25cm_5deg": float(kv["accuracy_25cm5deg_pct"]),
            "10cm_5deg": float(kv["accuracy_10cm5deg_pct"]),
            "5cm_5deg": float(kv["accuracy_5cm5deg_pct"]),
            "2cm_2deg": float(kv["accuracy_2cm2deg_pct"]),
            "1cm_1deg": float(kv["accuracy_1cm1deg_pct"]),
        }

    metrics_match = re.search(
        r"Median:\s*([0-9.]+)\s*deg,\s*([0-9.]+)\s*cm\s*\|\s*"
        r"25cm/5deg:\s*([0-9.]+)%\s*\|\s*"
        r"10cm/5deg:\s*([0-9.]+)%\s*\|\s*"
        r"5cm/5deg:\s*([0-9.]+)%\s*\|\s*"
        r"2cm/2deg:\s*([0-9.]+)%\s*\|\s*"
        r"1cm/1deg:\s*([0-9.]+)%",
        text,
    )
    frames_match = re.search(r"Frames:\s*(\d+)", text)
    if not metrics_match:
        raise ValueError(f"Could not parse metrics from {path}")
    return {
        "status": "ok",
        "frames": int(frames_match.group(1)) if frames_match else 0,
        "median_deg": float(metrics_match.group(1)),
        "median_cm": float(metrics_match.group(2)),
        "50cm_5deg": float("nan"),  # not in post_train_eval format, will be nan
        "25cm_5deg": float(metrics_match.group(3)),
        "10cm_5deg": float(metrics_match.group(4)),
        "5cm_5deg": float(metrics_match.group(5)),
        "2cm_2deg": float(metrics_match.group(6)),
        "1cm_1deg": float(metrics_match.group(7)),
    }


def _missing_metrics(status: str) -> dict[str, float | int | str]:
    out: dict[str, float | int | str] = {
        "status": status,
        "frames": 0,
        "median_deg": float("nan"),
        "median_cm": float("nan"),
    }
    for key, _ in THRESHOLDS:
        out[key] = float("nan")
    return out


def _summarize_pose_error_log(path: Path) -> dict[str, float | int | str]:
    rot_deg: list[float] = []
    trans_cm: list[float] = []
    if not path.exists():
        return _missing_metrics("missing")

    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 11:
                continue
            try:
                rot_deg.append(float(parts[-3]))
                trans_cm.append(float(parts[-2]) * 100.0)
            except ValueError:
                continue

    if not rot_deg:
        return _missing_metrics("missing")

    out: dict[str, float | int | str] = {
        "status": "ok",
        "frames": int(len(rot_deg)),
        "median_deg": float(sorted(rot_deg)[len(rot_deg) // 2]) if len(rot_deg) % 2 == 1 else float(__import__('numpy').median(rot_deg)),
        "median_cm": float(sorted(trans_cm)[len(trans_cm) // 2]) if len(trans_cm) % 2 == 1 else float(__import__('numpy').median(trans_cm)),
    }
    import numpy as np
    r = np.asarray(rot_deg, dtype=np.float64)
    t = np.asarray(trans_cm, dtype=np.float64)
    out["median_deg"] = float(np.median(r))
    out["median_cm"] = float(np.median(t))
    for key, label in THRESHOLDS:
        max_cm, max_deg = {
            "50cm_5deg": (50.0, 5.0),
            "25cm_5deg": (25.0, 5.0),
            "10cm_5deg": (10.0, 5.0),
            "5cm_5deg": (5.0, 5.0),
            "2cm_2deg": (2.0, 2.0),
            "1cm_1deg": (1.0, 1.0),
        }[key]
        out[key] = float(((r < max_deg) & (t < max_cm)).mean() * 100.0)
    return out


def _read_memory_sanity(path: Path) -> dict[str, str]:
    if not path.exists():
        return {
            "memory_hard_pass": "missing",
            "memory_pooled_points": "",
            "memory_cosine_margin": "",
            "memory_nn3d_ratio": "",
        }
    payload = json.loads(path.read_text(encoding="utf-8"))
    report = payload.get("sanity_report", payload)
    stats = payload.get("extraction_stats", payload)
    return {
        "memory_hard_pass": str(report.get("hard_pass", "")),
        "memory_pooled_points": _fmt(stats.get("pooled_points")),
        "memory_cosine_margin": _fmt(report.get("nn_cosine_margin_median")),
        "memory_nn3d_ratio": _fmt(report.get("nn_3d_error_ratio_median")),
    }


def _row(
    *,
    scene: str,
    method: str,
    source_path: Path | None,
    metrics: dict[str, float | int | str],
    train_device: str = "",
    eval_device: str = "",
    memory_feature_space: str = "",
    global_head: str = "",
    memory_hard_pass: str = "",
    memory_pooled_points: str = "",
    memory_cosine_margin: str = "",
    memory_nn3d_ratio: str = "",
) -> dict[str, str]:
    return {
        "scene": scene,
        "method": method,
        "status": _fmt(metrics["status"]),
        "frames": _fmt(metrics["frames"]),
        "50cm_5deg": _fmt(metrics["50cm_5deg"]),
        "25cm_5deg": _fmt(metrics["25cm_5deg"]),
        "10cm_5deg": _fmt(metrics["10cm_5deg"]),
        "5cm_5deg": _fmt(metrics["5cm_5deg"]),
        "2cm_2deg": _fmt(metrics["2cm_2deg"]),
        "1cm_1deg": _fmt(metrics["1cm_1deg"]),
        "median_deg": _fmt(metrics["median_deg"]),
        "median_cm": _fmt(metrics["median_cm"]),
        "train_device": train_device,
        "eval_device": eval_device,
        "memory_feature_space": memory_feature_space,
        "global_head": global_head,
        "memory_hard_pass": memory_hard_pass,
        "memory_pooled_points": memory_pooled_points,
        "memory_cosine_margin": memory_cosine_margin,
        "memory_nn3d_ratio": memory_nn3d_ratio,
        "output": str(source_path) if source_path is not None else "",
    }


def summarize(
    *,
    suite_root: Path,
    baseline_root: Path,
    scenes: list[str],
    memory_dirname: str,
    stage1_dirname: str,
    stage2_dirname: str,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    baseline_devices = _read_devices(baseline_root / "status.tsv")
    for scene in scenes:
        ace_pose = baseline_root / scene / "ace" / f"poses_{scene}_post_train.txt"
        ace_train_device, ace_eval_device = baseline_devices.get((scene, "ace"), ("", ""))
        rows.append(
            _row(
                scene=scene,
                method="ace",
                source_path=ace_pose if ace_pose.exists() else None,
                metrics=_summarize_pose_error_log(ace_pose),
                train_device=ace_train_device,
                eval_device=ace_eval_device,
            )
        )

        glace_pose = baseline_root / scene / "glace" / f"poses_{scene}_post_train.txt"
        glace_train_device, glace_eval_device = baseline_devices.get((scene, "glace"), ("", ""))
        rows.append(
            _row(
                scene=scene,
                method="glace",
                source_path=glace_pose if glace_pose.exists() else None,
                metrics=_summarize_pose_error_log(glace_pose),
                train_device=glace_train_device,
                eval_device=glace_eval_device,
            )
        )

        memory_path = suite_root / memory_dirname / scene / "memory_ace_fcn_sparse_sp_r4.pt"
        sanity = _read_memory_sanity(memory_path.with_suffix(memory_path.suffix + ".sanity.json"))

        stage1_eval = _find_latest_post_train_eval_any(_stage_roots(suite_root, stage1_dirname, scene))
        rows.append(
            _row(
                scene=scene,
                method="ace_fcn_stage1",
                source_path=stage1_eval,
                metrics=_parse_post_train_eval(stage1_eval) if stage1_eval else _missing_metrics("missing"),
                memory_feature_space="ace_fcn",
                global_head="none",
                **sanity,
            )
        )

        stage2_eval = _find_latest_post_train_eval_any(_stage_roots(suite_root, stage2_dirname, scene))
        rows.append(
            _row(
                scene=scene,
                method="ace_fcn_stage2_glace_concat",
                source_path=stage2_eval,
                metrics=_parse_post_train_eval(stage2_eval) if stage2_eval else _missing_metrics("missing"),
                memory_feature_space="ace_fcn",
                global_head="glace_concat",
                **sanity,
            )
        )
    return rows


def write_outputs(rows: list[dict[str, str]], suite_root: Path) -> None:
    suite_root.mkdir(parents=True, exist_ok=True)
    fields = [
        "scene",
        "method",
        "status",
        "frames",
        "50cm_5deg",
        "25cm_5deg",
        "10cm_5deg",
        "5cm_5deg",
        "2cm_2deg",
        "1cm_1deg",
        "median_deg",
        "median_cm",
        "train_device",
        "eval_device",
        "memory_feature_space",
        "global_head",
        "memory_hard_pass",
        "memory_pooled_points",
        "memory_cosine_margin",
        "memory_nn3d_ratio",
        "output",
    ]
    with (suite_root / "summary.tsv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter="	")
        writer.writeheader()
        writer.writerows(rows)

    with (suite_root / "summary.md").open("w", encoding="utf-8") as f:
        f.write(
            "| scene | method | status | frames | 50cm/5deg | 25cm/5deg | 10cm/5deg | 5cm/5deg | 2cm/2deg | 1cm/1deg | "
            "median deg | median cm | memory | global head | sanity pass | pooled pts | cosine margin | nn3d ratio |\n"
        )
        f.write(
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---:|---:|---:|\n"
        )
        for row in rows:
            f.write(
                f"| {row['scene']} | {row['method']} | {row['status']} | {row['frames']} | "
                f"{row['50cm_5deg']} | {row['25cm_5deg']} | {row['10cm_5deg']} | {row['5cm_5deg']} | {row['2cm_2deg']} | {row['1cm_1deg']} | "
                f"{row['median_deg']} | {row['median_cm']} | {row['memory_feature_space']} | {row['global_head']} | "
                f"{row['memory_hard_pass']} | {row['memory_pooled_points']} | {row['memory_cosine_margin']} | {row['memory_nn3d_ratio']} |\n"
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite-root", type=Path, required=True)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--scenes", nargs="+", required=True)
    parser.add_argument("--memory-dirname", type=str, default="memory")
    parser.add_argument("--stage1-dirname", type=str, default="stage1_local_ace_memory_it12")
    parser.add_argument("--stage2-dirname", type=str, default="stage2_glace_concat_it12")
    args = parser.parse_args()

    rows = summarize(
        suite_root=args.suite_root.resolve(),
        baseline_root=args.baseline_root.resolve(),
        scenes=args.scenes,
        memory_dirname=args.memory_dirname,
        stage1_dirname=args.stage1_dirname,
        stage2_dirname=args.stage2_dirname,
    )
    write_outputs(rows, args.suite_root.resolve())
    print(args.suite_root.resolve() / "summary.tsv")
    print(args.suite_root.resolve() / "summary.md")


if __name__ == "__main__":
    main()
