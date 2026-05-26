#!/usr/bin/env python3
"""Convert one RIO10 WAI train/test pair into ACE directory layout.

The converter writes the same structure used by indoor6_ace:

    <output_root>/<scene_id>/train/{rgb,poses,calibration}
    <output_root>/<scene_id>/test/{rgb,poses,calibration}

Images are symlinked by default. Poses are WAI c2w transform_matrix files, and
calibration files are 3x3 pinhole intrinsics matrices.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np


def _frame_short_name(frame_name: str) -> str:
    if "frame-" in frame_name:
        return frame_name[frame_name.index("frame-") :]
    return Path(frame_name).stem


def _write_split(scene_root: Path, out_split: Path, *, copy_images: bool) -> tuple[int, int]:
    meta_path = scene_root / "scene_meta.json"
    with meta_path.open("r", encoding="utf-8") as f:
        meta = json.load(f)

    for subdir in ("rgb", "poses", "calibration"):
        (out_split / subdir).mkdir(parents=True, exist_ok=True)

    written = 0
    missing_images = 0
    for frame in meta.get("frames", []):
        frame_name = str(frame["frame_name"])
        stem = _frame_short_name(frame_name)
        image_rel = frame.get("image") or frame.get("file_path")
        if image_rel is None:
            missing_images += 1
            continue
        image_src = scene_root / str(image_rel)
        if not image_src.exists():
            missing_images += 1
            continue

        image_dst = out_split / "rgb" / f"{stem}{image_src.suffix.lower()}"
        if image_dst.exists() or image_dst.is_symlink():
            image_dst.unlink()
        if copy_images:
            shutil.copy2(image_src, image_dst)
        else:
            image_dst.symlink_to(image_src)

        pose = np.asarray(frame["transform_matrix"], dtype=np.float64).reshape(4, 4)
        intrinsics = np.asarray(
            [
                [float(frame["fl_x"]), 0.0, float(frame["cx"])],
                [0.0, float(frame["fl_y"]), float(frame["cy"])],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        np.savetxt(out_split / "poses" / f"{stem}.pose.txt", pose, fmt="%.10f")
        np.savetxt(out_split / "calibration" / f"{stem}.calibration", intrinsics, fmt="%.10f")
        written += 1

    return written, missing_images


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wai-root", type=Path, default=Path("/data/xwh/RIO10_wai/mapanything_wai"))
    parser.add_argument("--output-root", type=Path, default=Path("/data/xwh/RIO10_ace"))
    parser.add_argument("--train-scene", type=str, required=True)
    parser.add_argument("--test-scene", type=str, required=True)
    parser.add_argument("--scene-id", type=str, required=True)
    parser.add_argument("--copy-images", action="store_true", help="Copy images instead of symlinking.")
    args = parser.parse_args()

    out_scene = args.output_root / args.scene_id
    train_count, train_missing = _write_split(
        args.wai_root / args.train_scene,
        out_scene / "train",
        copy_images=args.copy_images,
    )
    test_count, test_missing = _write_split(
        args.wai_root / args.test_scene,
        out_scene / "test",
        copy_images=args.copy_images,
    )
    print(f"ACE scene: {out_scene}")
    print(f"train frames: {train_count} missing_images={train_missing}")
    print(f"test frames: {test_count} missing_images={test_missing}")


if __name__ == "__main__":
    main()
