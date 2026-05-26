#!/usr/bin/env python3
"""Convert MuSHRoom Kinect captures into Indoor6-compatible WAI scenes.

Input layout:
    <mushroom_root>/room_datasets/<room>/kinect/
        long_capture/{images,depth,pose,intrinsic}/
        short_capture/{images,depth,pose,intrinsic}/

Output layout:
    <output_root>/<room>_train/
        scene_meta.json
        images/*.png
        gt_depth/*.npz
    <output_root>/<room>_test/
        scene_meta.json
        images/*.png
        gt_depth/*.npz

The converter uses pose/*.txt as OpenCV c2w 4x4 matrices and stores depth in
meters as compressed float32 NumPy arrays.
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path

import imageio.v2 as imageio
import numpy as np


def _numeric_stem(path: Path) -> int:
    return int(path.stem)


def _read_intrinsic_color(path: Path) -> np.ndarray:
    mat = np.loadtxt(path).astype(np.float64)
    if mat.shape == (4, 4):
        return mat[:3, :3]
    if mat.shape == (3, 3):
        return mat
    raise ValueError(f"Unexpected intrinsic shape {mat.shape} at {path}")


def _link_or_copy(src: Path, dst: Path, *, copy_files: bool) -> None:
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if copy_files:
        shutil.copy2(src, dst)
    else:
        dst.symlink_to(src)


def _collect_frames(split_root: Path) -> list[int]:
    image_ids = {_numeric_stem(p) for p in (split_root / "images").glob("*.png")}
    depth_ids = {_numeric_stem(p) for p in (split_root / "depth").glob("*.png")}
    pose_ids = {_numeric_stem(p) for p in (split_root / "pose").glob("*.txt")}
    return sorted(image_ids & depth_ids & pose_ids)


def _split_ready(split_root: Path) -> tuple[bool, str]:
    missing = [
        name for name in ("images", "depth", "pose")
        if not (split_root / name).is_dir()
    ]
    if missing:
        return False, f"missing dirs: {','.join(missing)}"
    intrinsic_path = split_root / "intrinsic" / "intrinsic_color.txt"
    if not intrinsic_path.is_file():
        return False, f"missing RGB intrinsics: {intrinsic_path}"
    frame_ids = _collect_frames(split_root)
    if not frame_ids:
        return False, "no common image/depth/pose frame ids"
    return True, f"frames={len(frame_ids)}"


def _depth_png_to_meters(depth_png: Path) -> np.ndarray:
    raw = imageio.imread(depth_png)
    if raw.ndim != 2:
        raise ValueError(f"Expected 2D depth PNG at {depth_png}, got shape={raw.shape}")
    if raw.dtype != np.uint16:
        raw = raw.astype(np.uint16, copy=False)
    depth_m = raw.astype(np.float32) / 1000.0
    return depth_m


def _frame_record(
    frame_id: int,
    pose: np.ndarray,
    intrinsics: np.ndarray,
    image_rel: str,
    depth_rel: str,
    height: int,
    width: int,
) -> dict:
    stem = f"{frame_id:06d}"
    return {
        "frame_name": stem,
        "image": image_rel,
        "file_path": image_rel,
        "transform_matrix": pose.tolist(),
        "h": int(height),
        "w": int(width),
        "fl_x": float(intrinsics[0, 0]),
        "fl_y": float(intrinsics[1, 1]),
        "cx": float(intrinsics[0, 2]),
        "cy": float(intrinsics[1, 2]),
        "gt_depth": depth_rel,
    }


def _write_scene(split_root: Path, out_scene: Path, scene_name: str, *, copy_files: bool) -> dict[str, int]:
    (out_scene / "images").mkdir(parents=True, exist_ok=True)
    (out_scene / "gt_depth").mkdir(parents=True, exist_ok=True)

    intrinsics = _read_intrinsic_color(split_root / "intrinsic" / "intrinsic_color.txt")
    frames = []
    frame_names = {}
    written = 0

    for idx, frame_id in enumerate(_collect_frames(split_root)):
        stem = f"{frame_id:06d}"
        image_src = (split_root / "images" / f"{frame_id}.png").resolve()
        depth_src = split_root / "depth" / f"{frame_id}.png"
        pose_src = split_root / "pose" / f"{frame_id}.txt"

        pose = np.loadtxt(pose_src).astype(np.float64)
        if pose.shape != (4, 4):
            raise ValueError(f"Unexpected pose shape {pose.shape} at {pose_src}")

        image_dst = out_scene / "images" / f"{stem}.png"
        depth_dst = out_scene / "gt_depth" / f"{stem}.npz"
        _link_or_copy(image_src, image_dst, copy_files=copy_files)

        depth_m = _depth_png_to_meters(depth_src)
        np.savez_compressed(depth_dst, arr_0=depth_m)

        frames.append(
            _frame_record(
                frame_id,
                pose,
                intrinsics,
                image_rel=f"images/{stem}.png",
                depth_rel=f"gt_depth/{stem}.npz",
                height=depth_m.shape[0],
                width=depth_m.shape[1],
            )
        )
        frame_names[stem] = idx
        written += 1

    scene_meta = {
        "scene_name": scene_name,
        "dataset_name": "mushroom",
        "version": "0.1",
        "shared_intrinsics": True,
        "camera_model": "PINHOLE",
        "camera_convention": "opencv",
        "scale_type": "metric",
        "frame_modalities": {
            "image": {"frame_key": "image", "format": "image"},
            "gt_depth": {"frame_key": "gt_depth", "format": "numpy"},
        },
        "frames": frames,
        "frame_names": frame_names,
    }
    (out_scene / "scene_meta.json").write_text(
        json.dumps(scene_meta, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    image_count = len(list((split_root / "images").glob("*.png")))
    depth_count = len(list((split_root / "depth").glob("*.png")))
    pose_count = len(list((split_root / "pose").glob("*.txt")))
    return {
        "written": written,
        "image_count": image_count,
        "depth_count": depth_count,
        "pose_count": pose_count,
        "dropped": min(image_count, depth_count, pose_count) - written,
    }


def convert_room(
    mushroom_root: Path,
    output_root: Path,
    room: str,
    *,
    copy_files: bool,
    skip_incomplete: bool,
) -> bool:
    room_root = mushroom_root / "room_datasets" / room / "kinect"
    if not room_root.exists():
        if skip_incomplete:
            print(f"[skip] {room}: missing {room_root}")
            return False
        raise FileNotFoundError(room_root)

    split_map = {
        f"{room}_train": room_root / "long_capture",
        f"{room}_test": room_root / "short_capture",
    }
    for scene_name, split_root in split_map.items():
        ready, reason = _split_ready(split_root)
        if not ready:
            if skip_incomplete:
                print(f"[skip] {scene_name}: {reason}")
                return False
            raise FileNotFoundError(f"{scene_name}: {reason}")

    output_root.mkdir(parents=True, exist_ok=True)
    stats_by_scene = {}
    with tempfile.TemporaryDirectory(prefix=f".{room}.tmp.", dir=output_root) as tmp_dir:
        tmp_root = Path(tmp_dir)
        for scene_name, split_root in split_map.items():
            stats_by_scene[scene_name] = _write_scene(
                split_root,
                tmp_root / scene_name,
                scene_name,
                copy_files=copy_files,
            )
        for scene_name in split_map:
            final_scene = output_root / scene_name
            if final_scene.exists():
                shutil.rmtree(final_scene)
            (tmp_root / scene_name).rename(final_scene)

    for scene_name, stats in stats_by_scene.items():
        print(
            f"WAI scene: {output_root / scene_name} "
            f"written={stats['written']} "
            f"(images={stats['image_count']} depth={stats['depth_count']} poses={stats['pose_count']})"
        )
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mushroom-root", type=Path, default=Path("/data/xwh/MuSHRoom/kinect"))
    parser.add_argument("--output-root", type=Path, default=Path("/data/xwh/MuSHRoom_wai/kinect"))
    parser.add_argument("--rooms", nargs="+", required=True, help="Room names, e.g. coffee_room classroom vr_room")
    parser.add_argument("--copy-files", action="store_true", help="Copy RGB images instead of symlinking.")
    parser.add_argument("--skip-incomplete", action="store_true", help="Skip rooms with incomplete Kinect files.")
    args = parser.parse_args()

    converted = 0
    skipped = 0
    for room in args.rooms:
        ok = convert_room(
            args.mushroom_root,
            args.output_root,
            room,
            copy_files=args.copy_files,
            skip_incomplete=args.skip_incomplete,
        )
        converted += int(ok)
        skipped += int(not ok)
    print(f"summary: converted={converted} skipped={skipped}")


if __name__ == "__main__":
    main()
