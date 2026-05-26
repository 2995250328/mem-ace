#!/usr/bin/env python3
"""Convert MuSHRoom iPhone short_capture into ACE test-only scenes.

Input layout:
    <iphone-root>/room_datasets/<room>/iphone/short_capture/
        images/frame_00001.jpg
        depth/frame_00001.png
        transformations_colmap.json

Output layout:
    <output-root>/<room>/
        test/{rgb,poses,calibration,gt_depth}

The converter uses transformations_colmap.json by default because it stores the
metric COLMAP-aligned camera track for iPhone captures. Images and depth maps are
symlinked by default; poses and calibration files are written in ACE format.
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path

import numpy as np


def _link_or_copy(src: Path, dst: Path, *, copy_files: bool) -> None:
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if copy_files:
        shutil.copy2(src, dst)
    else:
        dst.symlink_to(src)


def _camera_matrix(meta: dict, frame: dict) -> np.ndarray:
    def get_float(key: str) -> float:
        if key in frame:
            return float(frame[key])
        return float(meta[key])

    return np.asarray(
        [
            [get_float("fl_x"), 0.0, get_float("cx")],
            [0.0, get_float("fl_y"), get_float("cy")],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def _frame_stem(file_path: str) -> str:
    return Path(file_path).stem


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _convert_split(
    split_root: Path,
    out_split: Path,
    *,
    transform_file: str,
    copy_files: bool,
    require_depth: bool,
) -> dict[str, int]:
    meta_path = split_root / transform_file
    if not meta_path.is_file():
        raise FileNotFoundError(meta_path)
    meta = _load_json(meta_path)

    for subdir in ("rgb", "poses", "calibration", "gt_depth"):
        (out_split / subdir).mkdir(parents=True, exist_ok=True)

    written = 0
    missing_images = 0
    missing_depth = 0
    bad_pose = 0

    for frame in meta.get("frames", []):
        image_rel = frame.get("file_path")
        if not image_rel:
            missing_images += 1
            continue
        image_src = (split_root / str(image_rel).lstrip("./")).resolve()
        if not image_src.exists():
            missing_images += 1
            continue

        depth_rel = frame.get("depth_file_path")
        depth_src = None
        if depth_rel:
            candidate = (split_root / str(depth_rel).lstrip("./")).resolve()
            if candidate.exists():
                depth_src = candidate
        if depth_src is None:
            fallback = split_root / "depth" / f"{_frame_stem(image_rel)}.png"
            if fallback.exists():
                depth_src = fallback.resolve()
        if depth_src is None:
            missing_depth += 1
            if require_depth:
                continue

        pose = np.asarray(frame.get("transform_matrix"), dtype=np.float64)
        if pose.shape != (4, 4):
            bad_pose += 1
            continue

        stem = _frame_stem(image_rel)
        image_dst = out_split / "rgb" / f"{stem}{image_src.suffix.lower()}"
        pose_dst = out_split / "poses" / f"{stem}.pose.txt"
        calib_dst = out_split / "calibration" / f"{stem}.calibration"

        _link_or_copy(image_src, image_dst, copy_files=copy_files)
        if depth_src is not None:
            depth_dst = out_split / "gt_depth" / f"{stem}{depth_src.suffix.lower()}"
            _link_or_copy(depth_src, depth_dst, copy_files=copy_files)

        np.savetxt(pose_dst, pose, fmt="%.10f")
        np.savetxt(calib_dst, _camera_matrix(meta, frame), fmt="%.10f")
        written += 1

    return {
        "written": written,
        "frames": len(meta.get("frames", [])),
        "missing_images": missing_images,
        "missing_depth": missing_depth,
        "bad_pose": bad_pose,
    }


def convert_room(
    iphone_root: Path,
    output_root: Path,
    room: str,
    *,
    split: str,
    transform_file: str,
    copy_files: bool,
    require_depth: bool,
    skip_incomplete: bool,
) -> bool:
    split_root = iphone_root / "room_datasets" / room / "iphone" / split
    if not split_root.exists():
        if skip_incomplete:
            print(f"[skip] {room}: missing {split_root}")
            return False
        raise FileNotFoundError(split_root)

    out_scene = output_root / room
    output_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{room}.iphone.tmp.", dir=output_root) as tmp_dir:
        tmp_scene = Path(tmp_dir)
        stats = _convert_split(
            split_root,
            tmp_scene / "test",
            transform_file=transform_file,
            copy_files=copy_files,
            require_depth=require_depth,
        )
        if stats["written"] == 0:
            if skip_incomplete:
                print(f"[skip] {room}: no frames converted from {split_root}")
                return False
            raise RuntimeError(f"No frames converted from {split_root}")
        if out_scene.exists():
            shutil.rmtree(out_scene)
        tmp_scene.rename(out_scene)

    print(
        f"ACE iPhone scene: {out_scene} test written={stats['written']} "
        f"(json_frames={stats['frames']} missing_images={stats['missing_images']} "
        f"missing_depth={stats['missing_depth']} bad_pose={stats['bad_pose']})"
    )
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iphone-root", type=Path, default=Path("/data/xwh/MuSHRoom/iphone"))
    parser.add_argument("--output-root", type=Path, default=Path("/data/xwh/MuSHRoom_ace_iphone"))
    parser.add_argument("--rooms", nargs="+", required=True, help="Room names, e.g. coffee_room classroom vr_room")
    parser.add_argument("--split", type=str, default="short_capture", choices=["short_capture", "long_capture"])
    parser.add_argument("--transform-file", type=str, default="transformations_colmap.json")
    parser.add_argument("--copy-files", action="store_true", help="Copy RGB/depth instead of symlinking.")
    parser.add_argument("--require-depth", action="store_true", help="Skip frames without depth files.")
    parser.add_argument("--skip-incomplete", action="store_true", help="Skip rooms that cannot be converted.")
    args = parser.parse_args()

    converted = 0
    skipped = 0
    for room in args.rooms:
        ok = convert_room(
            args.iphone_root,
            args.output_root,
            room,
            split=args.split,
            transform_file=args.transform_file,
            copy_files=args.copy_files,
            require_depth=args.require_depth,
            skip_incomplete=args.skip_incomplete,
        )
        converted += int(ok)
        skipped += int(not ok)
    print(f"summary: converted={converted} skipped={skipped}")


if __name__ == "__main__":
    main()
