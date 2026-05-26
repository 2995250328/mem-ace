#!/usr/bin/env python3
"""Convert MuSHRoom Kinect rooms into ACE directory layout.

Input layout:
    <mushroom_root>/room_datasets/<room>/kinect/
        long_capture/{images,depth,pose,intrinsic}/
        short_capture/{images,depth,pose,intrinsic}/

Output layout:
    <output_root>/<scene_id>/
        train/{rgb,poses,calibration,depth,gt_depth}
        test/{rgb,poses,calibration,depth,gt_depth}

The converter uses:
  - transformations_colmap.json as shared-world OpenGL c2w, converted to OpenCV c2w by default
  - pose/*.txt as OpenCV c2w 4x4 only when --pose-source raw_pose is set
  - intrinsic/intrinsic_color.txt as the RGB intrinsics
  - depth/*.png as uint16 millimeter depth

Files are symlinked by default.
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path

import numpy as np


OPENGL_TO_OPENCV_C2W = np.diag([1.0, -1.0, -1.0, 1.0])


def _read_intrinsic_color(path: Path) -> np.ndarray:
    mat = np.loadtxt(path).astype(np.float64)
    if mat.shape == (4, 4):
        return mat[:3, :3]
    if mat.shape == (3, 3):
        return mat
    raise ValueError(f"Unexpected intrinsic shape {mat.shape} at {path}")


def _numeric_stem(path: Path) -> int:
    return int(path.stem)


def _link_or_copy(src: Path, dst: Path, *, copy_files: bool) -> None:
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if copy_files:
        shutil.copy2(src, dst)
    else:
        dst.symlink_to(src)


def _collect_frames(split_root: Path) -> list[int]:
    image_dir = split_root / "images"
    depth_dir = split_root / "depth"
    pose_dir = split_root / "pose"
    image_ids = {_numeric_stem(p) for p in image_dir.glob("*.png")}
    depth_ids = {_numeric_stem(p) for p in depth_dir.glob("*.png")}
    pose_ids = {_numeric_stem(p) for p in pose_dir.glob("*.txt")}
    return sorted(image_ids & depth_ids & pose_ids)


def _split_ready(split_root: Path, pose_source: str) -> tuple[bool, str]:
    required_dirs = ["images", "depth", "pose"]
    missing = [name for name in required_dirs if not (split_root / name).is_dir()]
    if missing:
        return False, f"missing dirs: {','.join(missing)}"
    intrinsic_path = split_root / "intrinsic" / "intrinsic_color.txt"
    if not intrinsic_path.is_file():
        return False, f"missing RGB intrinsics: {intrinsic_path}"
    if pose_source != "raw_pose":
        json_name = "transformations_colmap.json" if pose_source == "transformations_colmap" else "transformations.json"
        if not (split_root / json_name).is_file():
            return False, f"missing {json_name}"
    frame_ids = _collect_frames(split_root)
    if not frame_ids:
        return False, "no common image/depth/pose frame ids"
    return True, f"frames={len(frame_ids)}"


def _load_transform_json_poses(split_root: Path, json_name: str) -> dict[int, np.ndarray]:
    path = split_root / json_name
    with path.open("r") as f:
        data = json.load(f)

    poses = {}
    for frame in data.get("frames", []):
        frame_id = _numeric_stem(Path(frame["file_path"]))
        pose_gl = np.asarray(frame["transform_matrix"], dtype=np.float64)
        if pose_gl.shape != (4, 4):
            raise ValueError(f"Unexpected transform shape {pose_gl.shape} in {path} for frame {frame_id}")
        poses[frame_id] = pose_gl @ OPENGL_TO_OPENCV_C2W
    return poses


def _load_pose_map(split_root: Path, pose_source: str) -> dict[int, np.ndarray] | None:
    if pose_source == "raw_pose":
        return None
    json_name = "transformations_colmap.json" if pose_source == "transformations_colmap" else "transformations.json"
    return _load_transform_json_poses(split_root, json_name)


def _write_split(split_root: Path, out_split: Path, *, copy_files: bool, pose_source: str) -> dict[str, int | str]:
    for subdir in ("rgb", "poses", "calibration", "depth", "gt_depth"):
        (out_split / subdir).mkdir(parents=True, exist_ok=True)

    intrinsic_path = split_root / "intrinsic" / "intrinsic_color.txt"
    intrinsics = _read_intrinsic_color(intrinsic_path)
    frame_ids = _collect_frames(split_root)
    pose_map = _load_pose_map(split_root, pose_source)
    if pose_map is not None:
        frame_ids = sorted(set(frame_ids) & set(pose_map))

    written = 0
    for frame_id in frame_ids:
        stem = f"{frame_id:06d}"
        rgb_src = split_root / "images" / f"{frame_id}.png"
        depth_src = split_root / "depth" / f"{frame_id}.png"
        pose_src = split_root / "pose" / f"{frame_id}.txt"

        rgb_dst = out_split / "rgb" / f"{stem}.png"
        depth_dst = out_split / "gt_depth" / f"{stem}.png"
        depth_compat_dst = out_split / "depth" / f"{stem}.png"
        pose_dst = out_split / "poses" / f"{stem}.pose.txt"
        calib_dst = out_split / "calibration" / f"{stem}.calibration"

        _link_or_copy(rgb_src.resolve(), rgb_dst, copy_files=copy_files)
        _link_or_copy(depth_src.resolve(), depth_dst, copy_files=copy_files)
        _link_or_copy(depth_src.resolve(), depth_compat_dst, copy_files=copy_files)
        pose = pose_map[frame_id] if pose_map is not None else np.loadtxt(pose_src).astype(np.float64)
        if pose.shape != (4, 4):
            raise ValueError(f"Unexpected pose shape {pose.shape} at {pose_src}")
        np.savetxt(pose_dst, pose, fmt="%.10f")
        np.savetxt(calib_dst, intrinsics, fmt="%.10f")
        written += 1

    image_count = len(list((split_root / "images").glob("*.png")))
    depth_count = len(list((split_root / "depth").glob("*.png")))
    pose_count = len(list((split_root / "pose").glob("*.txt")))
    return {
        "written": written,
        "image_count": image_count,
        "depth_count": depth_count,
        "pose_count": pose_count,
        "dropped": min(image_count, depth_count, pose_count) - written,
        "pose_source": pose_source,
    }


def convert_room(
    mushroom_root: Path,
    output_root: Path,
    room: str,
    *,
    copy_files: bool,
    scene_id: str | None,
    skip_incomplete: bool,
    pose_source: str,
) -> bool:
    room_root = mushroom_root / "room_datasets" / room / "kinect"
    if not room_root.exists():
        if skip_incomplete:
            print(f"[skip] {room}: missing {room_root}")
            return False
        raise FileNotFoundError(room_root)

    for split_name, split_dir in (("train", room_root / "long_capture"), ("test", room_root / "short_capture")):
        ready, reason = _split_ready(split_dir, pose_source)
        if not ready:
            if skip_incomplete:
                print(f"[skip] {room}: {split_name} {reason}")
                return False
            raise FileNotFoundError(f"{room} {split_name}: {reason}")

    out_scene = output_root / (scene_id or room)
    output_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{out_scene.name}.tmp.", dir=output_root) as tmp_dir:
        tmp_scene = Path(tmp_dir)
        train_stats = _write_split(
            room_root / "long_capture",
            tmp_scene / "train",
            copy_files=copy_files,
            pose_source=pose_source,
        )
        test_stats = _write_split(
            room_root / "short_capture",
            tmp_scene / "test",
            copy_files=copy_files,
            pose_source=pose_source,
        )
        if out_scene.exists():
            shutil.rmtree(out_scene)
        tmp_scene.rename(out_scene)

    print(f"ACE scene: {out_scene}")
    print(
        f"train written={train_stats['written']} "
        f"(images={train_stats['image_count']} depth={train_stats['depth_count']} poses={train_stats['pose_count']} "
        f"pose_source={train_stats['pose_source']})"
    )
    print(
        f"test written={test_stats['written']} "
        f"(images={test_stats['image_count']} depth={test_stats['depth_count']} poses={test_stats['pose_count']} "
        f"pose_source={test_stats['pose_source']})"
    )
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mushroom-root", type=Path, default=Path("/data/xwh/MuSHRoom/kinect"))
    parser.add_argument("--output-root", type=Path, default=Path("/data/xwh/MuSHRoom_ace"))
    parser.add_argument("--rooms", nargs="+", required=True, help="Room names, e.g. coffee_room classroom vr_room")
    parser.add_argument("--copy-files", action="store_true", help="Copy images/depth instead of symlinking.")
    parser.add_argument("--skip-incomplete", action="store_true", help="Skip rooms with incomplete Kinect files.")
    parser.add_argument(
        "--pose-source",
        choices=("transformations_colmap", "transformations", "raw_pose"),
        default="transformations_colmap",
        help=(
            "Pose source. transformations_colmap keeps long/short captures in the shared COLMAP world and converts "
            "OpenGL c2w to OpenCV c2w; raw_pose preserves per-capture local pose/*.txt."
        ),
    )
    parser.add_argument(
        "--scene-id-prefix",
        type=str,
        default="",
        help="Optional prefix for output ACE scene ids, e.g. mushroom_kinect_",
    )
    args = parser.parse_args()

    args.output_root.mkdir(parents=True, exist_ok=True)
    converted = 0
    skipped = 0
    for room in args.rooms:
        scene_id = f"{args.scene_id_prefix}{room}" if args.scene_id_prefix else room
        ok = convert_room(
            args.mushroom_root,
            args.output_root,
            room,
            copy_files=args.copy_files,
            scene_id=scene_id,
            skip_incomplete=args.skip_incomplete,
            pose_source=args.pose_source,
        )
        converted += int(ok)
        skipped += int(not ok)
    print(f"summary: converted={converted} skipped={skipped}")


if __name__ == "__main__":
    main()
