#!/usr/bin/env python3
"""Check whether WAI depth files can support ACE aux coordinate supervision."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _ids_from_dir(path: Path, prefix: str = "", suffixes: tuple[str, ...] | None = None) -> dict[str, Path]:
    if not path.exists():
        raise FileNotFoundError(path)
    out: dict[str, Path] = {}
    for file in path.iterdir():
        if not file.is_file():
            continue
        if suffixes and file.suffix.lower() not in suffixes:
            continue
        stem = file.stem
        if prefix and stem.startswith(prefix):
            stem = stem[len(prefix):]
        out[stem] = file
    return out


def _depth_stats(paths: list[Path], max_samples: int) -> list[str]:
    lines = []
    if not paths:
        return lines
    sample_paths = paths[:max_samples]
    for path in sample_paths:
        arr = np.load(path)
        finite = np.isfinite(arr)
        valid = finite & (arr > 0)
        values = arr[valid]
        if values.size == 0:
            lines.append(f"{path.name}: shape={arr.shape} dtype={arr.dtype} valid_ratio=0.000000")
            continue
        lines.append(
            f"{path.name}: shape={arr.shape} dtype={arr.dtype} "
            f"valid_ratio={float(valid.mean()):.6f} "
            f"min={float(values.min()):.4f} median={float(np.median(values)):.4f} max={float(values.max()):.4f}"
        )
    return lines


def _frame_intrinsics(frame: dict) -> np.ndarray:
    return np.asarray(
        [
            [float(frame["fl_x"]), 0.0, float(frame["cx"])],
            [0.0, float(frame["fl_y"]), float(frame["cy"])],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def _pose_key(pose: np.ndarray, decimals: int = 5) -> tuple:
    return tuple(np.round(pose.reshape(-1), decimals=decimals).tolist())


def _scene_meta_alignment(ace_train_root: Path, wai_scene_root: Path, depth_kind: str) -> tuple[list[Path], dict]:
    meta_path = wai_scene_root / "scene_meta.json"
    if not meta_path.exists():
        return [], {"status": "missing_scene_meta", "meta": str(meta_path)}

    frames = json.load(meta_path.open("r", encoding="utf-8")).get("frames", [])
    pose_to_frames: dict[tuple, list[dict]] = {}
    meta_poses = []
    for frame in frames:
        if "transform_matrix" not in frame or depth_kind not in frame:
            continue
        pose = np.asarray(frame["transform_matrix"], dtype=np.float64)
        if pose.shape != (4, 4):
            continue
        pose_to_frames.setdefault(_pose_key(pose), []).append(frame)
        meta_poses.append((pose.reshape(-1), frame))
    if not meta_poses:
        return [], {"status": "no_valid_frames", "meta": str(meta_path)}

    rgb_files = sorted((ace_train_root / "rgb").iterdir())
    pose_files = sorted((ace_train_root / "poses").iterdir())
    calibration_files = sorted((ace_train_root / "calibration").iterdir())
    meta_pose_mat = np.stack([p for p, _ in meta_poses], axis=0)

    matched_paths = []
    matched = 0
    fallback_pose = 0
    bad_intrinsics = 0
    missing_depth = 0
    unmatched = 0
    examples = []
    for idx, pose_file in enumerate(pose_files):
        ace_pose = np.loadtxt(pose_file).astype(np.float64)
        ace_k = np.loadtxt(calibration_files[idx]).astype(np.float64)
        if ace_pose.shape != (4, 4) or ace_k.shape != (3, 3):
            unmatched += 1
            continue

        candidates = pose_to_frames.get(_pose_key(ace_pose), [])
        chosen = None
        if candidates:
            chosen = min(candidates, key=lambda fr: float(np.max(np.abs(_frame_intrinsics(fr) - ace_k))))
        else:
            pose_diffs = np.max(np.abs(meta_pose_mat - ace_pose.reshape(1, -1)), axis=1)
            best_idx = int(np.argmin(pose_diffs))
            if float(pose_diffs[best_idx]) <= 1e-4:
                chosen = meta_poses[best_idx][1]
                fallback_pose += 1

        if chosen is None:
            unmatched += 1
            continue
        k_diff = float(np.max(np.abs(_frame_intrinsics(chosen) - ace_k)))
        if k_diff > 1e-2:
            bad_intrinsics += 1
            continue
        depth_path = wai_scene_root / chosen[depth_kind]
        if not depth_path.exists():
            missing_depth += 1
            continue
        matched += 1
        matched_paths.append(depth_path)
        if len(examples) < 8:
            examples.append(f"{rgb_files[idx].name}->{chosen.get('frame_name')}->{Path(chosen[depth_kind]).name}")

    return matched_paths, {
        "status": "complete" if matched == len(rgb_files) else "partial",
        "meta": str(meta_path),
        "ace_rgb": len(rgb_files),
        "matched": matched,
        "unmatched": unmatched,
        "fallback_pose": fallback_pose,
        "bad_intrinsics": bad_intrinsics,
        "missing_depth": missing_depth,
        "examples": examples,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ace-train-root", required=True, type=Path, help="ACE scene train dir, e.g. indoor6_ace/scene2a/train")
    parser.add_argument("--wai-scene-root", required=True, type=Path, help="WAI scene train dir, e.g. wai_data/indoor6/scene2a_train")
    parser.add_argument("--depth-kind", default="gt_depth", choices=["gt_depth", "colmap_depth"])
    parser.add_argument("--max-stats", type=int, default=8)
    args = parser.parse_args()

    ace_rgb = _ids_from_dir(args.ace_train_root / "rgb", suffixes=(".jpg", ".jpeg", ".png"))
    wai_depth = _ids_from_dir(args.wai_scene_root / args.depth_kind, prefix="image-", suffixes=(".npy",))

    ace_ids = set(ace_rgb)
    depth_ids = set(wai_depth)
    matched = sorted(ace_ids & depth_ids)
    missing = sorted(ace_ids - depth_ids)
    extra = sorted(depth_ids - ace_ids)

    print(f"ace_rgb={len(ace_rgb)} depth={len(wai_depth)} matched={len(matched)} missing={len(missing)} extra={len(extra)}")
    if ace_ids:
        print(f"ace_range={min(ace_ids)}..{max(ace_ids)}")
    if depth_ids:
        print(f"depth_range={min(depth_ids)}..{max(depth_ids)}")
    print(f"missing_first20={missing[:20]}")
    print(f"extra_first20={extra[:20]}")

    meta_paths, meta_report = _scene_meta_alignment(args.ace_train_root, args.wai_scene_root, args.depth_kind)
    print("scene_meta_alignment:")
    for key, value in meta_report.items():
        print(f"  {key}={value}")

    matched_paths = meta_paths if meta_paths else [wai_depth[idx] for idx in matched]
    print("depth_stats:")
    for line in _depth_stats(matched_paths, args.max_stats):
        print(f"  {line}")

    if meta_report.get("status") == "complete":
        print("status=complete")
        print("note=Use scene_meta pose/calibration alignment; filename-only alignment is not authoritative.")
    elif missing:
        print("status=partial")
        print("note=Filename-only alignment is partial. Prefer scene_meta pose/calibration alignment or handle missing frames.")
    else:
        print("status=complete")


if __name__ == "__main__":
    main()
