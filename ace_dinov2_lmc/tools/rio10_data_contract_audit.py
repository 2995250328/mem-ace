#!/usr/bin/env python3
"""Audit the ACE-format RIO10 scene01 data contract against WAI metadata."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
from PIL import Image


ACE_SCENE = Path("/data/xwh/RIO10_ace/scene01_seq01_01")
WAI_ROOT = Path("/data/xwh/RIO10_wai/mapanything_wai")
IMAGE_HEIGHT = 518
PATCH_SIZE = 14


def round_to_patch(size: float) -> int:
    return int(round(float(size) / PATCH_SIZE) * PATCH_SIZE)


def list_split(split_root: Path) -> tuple[list[Path], list[Path], list[Path]]:
    return (
        sorted((split_root / "rgb").iterdir()),
        sorted((split_root / "poses").iterdir()),
        sorted((split_root / "calibration").iterdir()),
    )


def stem_rgb(path: Path) -> str:
    return path.stem


def stem_pose(path: Path) -> str:
    name = path.name
    if name.endswith(".pose.txt"):
        return name[: -len(".pose.txt")]
    return path.stem


def stem_calib(path: Path) -> str:
    name = path.name
    if name.endswith(".calibration"):
        return name[: -len(".calibration")]
    return path.stem


def load_wai_meta(scene_name: str) -> dict[str, dict]:
    meta_path = WAI_ROOT / scene_name / "scene_meta.json"
    with meta_path.open("r", encoding="utf-8") as f:
        meta = json.load(f)
    return {frame["frame_name"]: frame for frame in meta["frames"]}


def ace_frame_to_wai_frame(split: str, ace_stem: str) -> tuple[str, str]:
    if split == "train":
        scene = "scene01_seq01_01_train"
        prefix = "seq01_01"
    elif split == "test":
        scene = "scene01_seq01_02_test"
        prefix = "seq01_02"
    else:
        raise ValueError(split)
    return scene, f"{prefix}_{ace_stem}"


def compare_samples(split: str, stems: list[str]) -> list[str]:
    scene_name, _ = ace_frame_to_wai_frame(split, "frame-000000")
    meta_by_name = load_wai_meta(scene_name)
    lines = []
    for stem in stems:
        _, wai_frame_name = ace_frame_to_wai_frame(split, stem)
        frame = meta_by_name[wai_frame_name]
        pose_ace = np.loadtxt(ACE_SCENE / split / "poses" / f"{stem}.pose.txt")
        calib_ace = np.loadtxt(ACE_SCENE / split / "calibration" / f"{stem}.calibration")
        pose_wai = np.asarray(frame["transform_matrix"], dtype=np.float64).reshape(4, 4)
        calib_wai = np.asarray(
            [
                [float(frame["fl_x"]), 0.0, float(frame["cx"])],
                [0.0, float(frame["fl_y"]), float(frame["cy"])],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        pose_max_abs = float(np.max(np.abs(pose_ace - pose_wai)))
        calib_max_abs = float(np.max(np.abs(calib_ace - calib_wai)))
        rgb_path = ACE_SCENE / split / "rgb" / f"{stem}.jpg"
        target = os.path.realpath(rgb_path)
        image_rel = str(frame.get("image") or frame.get("file_path"))
        expected_tail = image_rel.replace("/", os.sep)
        target_ok = target.endswith(expected_tail)
        lines.append(
            f"{split} {stem}: pose_max_abs={pose_max_abs:.3e} "
            f"calib_max_abs={calib_max_abs:.3e} symlink_ok={target_ok} target={target}"
        )
    return lines


def camera_stats(split: str, pose_files: list[Path]) -> str:
    centers = []
    for path in pose_files:
        pose = np.loadtxt(path)
        centers.append(pose[:3, 3])
    arr = np.asarray(centers, dtype=np.float64)
    mins = arr.min(axis=0)
    maxs = arr.max(axis=0)
    mean = arr.mean(axis=0)
    radius = np.linalg.norm(arr - mean[None], axis=1)
    return (
        f"{split} centers mean={mean.round(4).tolist()} "
        f"min={mins.round(4).tolist()} max={maxs.round(4).tolist()} "
        f"radius_m p50/p95/max={np.percentile(radius, [50, 95, 100]).round(4).tolist()}"
    )


def resized_intrinsics_check(split: str, stem: str) -> str:
    rgb_path = ACE_SCENE / split / "rgb" / f"{stem}.jpg"
    calib = np.loadtxt(ACE_SCENE / split / "calibration" / f"{stem}.calibration")
    with Image.open(rgb_path) as img:
        orig_w, orig_h = img.size
    target_h = round_to_patch(IMAGE_HEIGHT)
    target_w = round_to_patch(orig_w * target_h / max(1, orig_h))
    h_scale = target_h / orig_h
    current_w = target_w
    w_scale = target_w / current_w
    resized = calib.copy()
    resized[0, 0] *= h_scale * w_scale
    resized[1, 1] *= h_scale
    resized[0, 2] *= h_scale * w_scale
    resized[1, 2] *= h_scale
    return (
        f"{split} {stem}: orig_wh=({orig_w},{orig_h}) resized_wh=({target_w},{target_h}) "
        f"fx/fy/cx/cy={resized[0,0]:.4f}/{resized[1,1]:.4f}/{resized[0,2]:.4f}/{resized[1,2]:.4f}"
    )


def main() -> None:
    print(f"ACE scene: {ACE_SCENE}")
    all_stems = {}
    pose_files_by_split = {}
    for split in ("train", "test"):
        split_root = ACE_SCENE / split
        rgb_files, pose_files, calib_files = list_split(split_root)
        pose_files_by_split[split] = pose_files
        rgb_stems = [stem_rgb(p) for p in rgb_files]
        pose_stems = [stem_pose(p) for p in pose_files]
        calib_stems = [stem_calib(p) for p in calib_files]
        all_stems[split] = set(rgb_stems)
        stem_ok = rgb_stems == pose_stems == calib_stems
        print(
            f"{split}: rgb={len(rgb_files)} poses={len(pose_files)} calibration={len(calib_files)} "
            f"stem_alignment={stem_ok}"
        )
        if not stem_ok:
            mismatch = sorted((set(rgb_stems) ^ set(pose_stems)) | (set(rgb_stems) ^ set(calib_stems)))[:10]
            print(f"{split}: stem mismatches sample={mismatch}")
        sample_idx = [0, len(rgb_stems) // 2, len(rgb_stems) - 1]
        sample_stems = [rgb_stems[i] for i in sample_idx]
        for line in compare_samples(split, sample_stems):
            print(line)
        print(camera_stats(split, pose_files))
        print(resized_intrinsics_check(split, sample_stems[0]))

    overlap = all_stems["train"] & all_stems["test"]
    print(f"train/test filename stem overlap={len(overlap)} (expected nonzero because sequences reuse frame numbers)")
    train_target = os.path.realpath(ACE_SCENE / "train" / "rgb" / "frame-000000.jpg")
    test_target = os.path.realpath(ACE_SCENE / "test" / "rgb" / "frame-000000.jpg")
    print(f"train frame-000000 source={train_target}")
    print(f"test  frame-000000 source={test_target}")
    print("expected split mapping: train -> seq01_01, test/public-val -> seq01_02, hidden test would be seq01_03+")


if __name__ == "__main__":
    main()
