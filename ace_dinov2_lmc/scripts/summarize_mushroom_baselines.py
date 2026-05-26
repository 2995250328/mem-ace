#!/usr/bin/env python3
"""Summarize MuSHRoom baseline outputs with a common metric set."""

from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path

import numpy as np


THRESHOLDS = (
    ("25cm_5deg", 25.0, 5.0),
    ("10cm_5deg", 10.0, 5.0),
    ("5cm_5deg", 5.0, 5.0),
    ("2cm_2deg", 2.0, 2.0),
    ("1cm_1deg", 1.0, 1.0),
)


def _read_matrix(path: Path) -> np.ndarray:
    return np.loadtxt(path, dtype=np.float64).reshape(4, 4)


def _frame_key(path_or_name: str | Path) -> str:
    stem = Path(path_or_name).stem
    if stem.endswith(".pose"):
        stem = stem[: -len(".pose")]
    return stem


def _canonical_frame_key(path_or_name: str | Path) -> str:
    key = _frame_key(path_or_name)
    match = re.search(r"(\d+)$", key)
    return match.group(1) if match else key


def _quat_wxyz_to_rot(q: list[float]) -> np.ndarray:
    w, x, y, z = q
    n = math.sqrt(w * w + x * x + y * y + z * z)
    if n == 0:
        return np.eye(3, dtype=np.float64)
    w, x, y, z = (w / n, x / n, y / n, z / n)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _rotation_error_deg(est_c2w: np.ndarray, gt_c2w: np.ndarray) -> float:
    rel = est_c2w[:3, :3] @ gt_c2w[:3, :3].T
    cos_angle = (float(np.trace(rel)) - 1.0) * 0.5
    cos_angle = max(-1.0, min(1.0, cos_angle))
    return math.degrees(math.acos(cos_angle))


def _metrics(rot_deg: list[float], trans_cm: list[float]) -> dict[str, float | int | str]:
    if not rot_deg:
        out: dict[str, float | int | str] = {
            "status": "missing",
            "frames": 0,
            "median_deg": float("nan"),
            "median_cm": float("nan"),
        }
        for key, _, _ in THRESHOLDS:
            out[key] = float("nan")
        return out

    r = np.asarray(rot_deg, dtype=np.float64)
    t = np.asarray(trans_cm, dtype=np.float64)
    out = {
        "status": "ok",
        "frames": int(len(r)),
        "median_deg": float(np.median(r)),
        "median_cm": float(np.median(t)),
    }
    for key, max_cm, max_deg in THRESHOLDS:
        out[key] = float(np.mean((r < max_deg) & (t < max_cm)) * 100.0)
    return out


def _summarize_pose_error_log(path: Path) -> dict[str, float | int | str]:
    rot_deg: list[float] = []
    trans_cm: list[float] = []
    if not path.exists():
        return _metrics(rot_deg, trans_cm)

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
    return _metrics(rot_deg, trans_cm)


def _read_ace_pose_estimates(path: Path) -> dict[str, np.ndarray]:
    estimates: dict[str, np.ndarray] = {}
    if not path.exists():
        return estimates

    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 10:
                continue
            try:
                quat = [float(x) for x in parts[1:5]]
                trans = [float(x) for x in parts[5:8]]
            except ValueError:
                continue
            w2c = np.eye(4, dtype=np.float64)
            w2c[:3, :3] = _quat_wxyz_to_rot(quat)
            w2c[:3, 3] = np.asarray(trans, dtype=np.float64)
            c2w = np.linalg.inv(w2c)
            estimates[_canonical_frame_key(parts[0])] = c2w
    return estimates


def _summarize_aceg_registered(scene_path: Path, registered_pose_file: Path) -> dict[str, float | int | str]:
    estimates = _read_ace_pose_estimates(registered_pose_file)
    rot_deg: list[float] = []
    trans_cm: list[float] = []
    for gt_path in sorted((scene_path / "test" / "poses").glob("*.txt")):
        est = estimates.get(_canonical_frame_key(gt_path))
        if est is None:
            continue
        gt = _read_matrix(gt_path)
        rot_deg.append(_rotation_error_deg(est, gt))
        trans_cm.append(float(np.linalg.norm(est[:3, 3] - gt[:3, 3]) * 100.0))
    return _metrics(rot_deg, trans_cm)



def _read_devices(run_root: Path) -> dict[tuple[str, str], tuple[str, str]]:
    status_path = run_root / "status.tsv"
    devices: dict[tuple[str, str], tuple[str, str]] = {}
    if not status_path.exists():
        return devices
    with status_path.open("r", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            scene = row.get("scene", "")
            method = row.get("method", "")
            if not scene or not method or method == "all":
                continue
            train_device = row.get("train_device", "") or "unknown"
            eval_device = row.get("eval_device", "") or "unknown"
            devices[(scene, method)] = (train_device, eval_device)
    return devices

def _row(scene: str, method: str, output: Path, metrics: dict[str, float | int | str], train_device: str = "unknown", eval_device: str = "unknown") -> dict[str, str]:
    def fmt(value: float | int | str) -> str:
        if isinstance(value, str):
            return value
        if isinstance(value, int):
            return str(value)
        if math.isnan(value):
            return "nan"
        return f"{value:.3f}"

    row = {
        "scene": scene,
        "method": method,
        "status": str(metrics["status"]),
        "frames": fmt(metrics["frames"]),
        "median_deg": fmt(metrics["median_deg"]),
        "median_cm": fmt(metrics["median_cm"]),
        "train_device": train_device,
        "eval_device": eval_device,
        "output": str(output),
    }
    for key, _, _ in THRESHOLDS:
        row[key] = fmt(metrics[key])
    return row


def summarize(run_root: Path, dataset_root: Path, scenes: list[str], methods: list[str]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    devices = _read_devices(run_root)
    for scene in scenes:
        scene_path = dataset_root / scene
        for method in methods:
            method_dir = run_root / scene / method
            if method == "aceg25":
                pose_file = method_dir / f"{scene}_aceg25_registered_poses.txt"
                metrics = _summarize_aceg_registered(scene_path, pose_file)
                train_device, eval_device = devices.get((scene, method), ("unknown", "unknown"))
                rows.append(_row(scene, method, pose_file, metrics, train_device, eval_device))
            else:
                pose_file = method_dir / f"poses_{scene}_post_train.txt"
                metrics = _summarize_pose_error_log(pose_file)
                train_device, eval_device = devices.get((scene, method), ("unknown", "unknown"))
                rows.append(_row(scene, method, pose_file, metrics, train_device, eval_device))
    return rows


def write_outputs(rows: list[dict[str, str]], run_root: Path) -> None:
    run_root.mkdir(parents=True, exist_ok=True)
    tsv_path = run_root / "summary.tsv"
    fields = [
        "scene",
        "method",
        "status",
        "frames",
        "25cm_5deg",
        "10cm_5deg",
        "5cm_5deg",
        "2cm_2deg",
        "1cm_1deg",
        "median_deg",
        "median_cm",
        "train_device",
        "eval_device",
        "output",
    ]
    with tsv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)

    md_path = run_root / "summary.md"
    with md_path.open("w", encoding="utf-8") as f:
        f.write("| scene | method | status | frames | train device | eval device | 25cm/5deg | 10cm/5deg | 5cm/5deg | 2cm/2deg | 1cm/1deg | median deg | median cm |\n")
        f.write("|---|---|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|\n")
        for row in rows:
            f.write(
                f"| {row['scene']} | {row['method']} | {row['status']} | {row['frames']} | "
                f"{row['train_device']} | {row['eval_device']} | "
                f"{row['25cm_5deg']} | {row['10cm_5deg']} | {row['5cm_5deg']} | "
                f"{row['2cm_2deg']} | {row['1cm_1deg']} | {row['median_deg']} | {row['median_cm']} |\n"
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--scenes", nargs="+", required=True)
    parser.add_argument("--methods", nargs="+", required=True)
    args = parser.parse_args()

    rows = summarize(args.run_root.resolve(), args.dataset_root.resolve(), args.scenes, args.methods)
    write_outputs(rows, args.run_root.resolve())
    print(args.run_root.resolve() / "summary.tsv")
    print(args.run_root.resolve() / "summary.md")


if __name__ == "__main__":
    main()
