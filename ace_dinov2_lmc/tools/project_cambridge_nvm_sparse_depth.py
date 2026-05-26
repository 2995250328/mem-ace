#!/usr/bin/env python3
"""Project Cambridge VisualSFM NVM points into sparse depth maps.

The ACE Cambridge setup keeps each scene's ``reconstruction.nvm`` next to the
converted ``train/test/{rgb,poses,calibration}`` folders. This script projects
the NVM sparse 3D points into the converted ACE images and writes metric sparse
depth maps as ``<split>/sparse_depth/<rgb_stem>.npz``.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import imageio.v2 as imageio
import numpy as np


def _converted_name(nvm_image_name: str) -> str:
    path = Path(nvm_image_name)
    png_name = str(path.with_suffix(".png"))
    return png_name.replace("/", "_")


def _read_calibration(path: Path, width: int, height: int) -> np.ndarray:
    calib = np.loadtxt(path).astype(np.float64)
    if np.size(calib) == 1:
        focal = float(np.asarray(calib).reshape(-1)[0])
        return np.array(
            [[focal, 0.0, width / 2.0], [0.0, focal, height / 2.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
    if calib.shape == (3, 3):
        return calib
    raise ValueError(f"Unexpected calibration shape {calib.shape} at {path}")


def _load_nvm_observations(nvm_path: Path) -> tuple[dict[str, int], dict[int, list[np.ndarray]]]:
    lines = nvm_path.read_text(encoding="utf-8", errors="replace").splitlines()
    non_empty = [line.strip() for line in lines if line.strip()]
    if not non_empty or not non_empty[0].startswith("NVM"):
        raise ValueError(f"Not an NVM file: {nvm_path}")

    num_cams = int(non_empty[1])
    cam_lines = non_empty[2:2 + num_cams]
    points_count_idx = 2 + num_cams
    num_points = int(non_empty[points_count_idx])
    point_lines = non_empty[points_count_idx + 1:points_count_idx + 1 + num_points]

    image_to_cam_idx = {}
    for cam_idx, line in enumerate(cam_lines):
        image_name = line.split()[0]
        image_to_cam_idx[_converted_name(image_name)] = cam_idx

    points_by_cam: dict[int, list[np.ndarray]] = defaultdict(list)
    for line in point_lines:
        parts = line.split()
        if len(parts) < 7:
            continue
        xyz = np.array([float(parts[0]), float(parts[1]), float(parts[2])], dtype=np.float64)
        n_obs = int(parts[6])
        for obs_idx in range(n_obs):
            base = 7 + obs_idx * 4
            if base + 3 >= len(parts):
                break
            cam_idx = int(parts[base])
            points_by_cam[cam_idx].append(xyz)

    return image_to_cam_idx, points_by_cam


def _project_depth(
    points_world: np.ndarray,
    c2w: np.ndarray,
    K: np.ndarray,
    width: int,
    height: int,
    max_depth_m: float,
) -> tuple[np.ndarray, int]:
    depth = np.zeros((height, width), dtype=np.float32)
    if points_world.size == 0:
        return depth, 0

    w2c = np.linalg.inv(c2w)
    pts_h = np.concatenate([points_world, np.ones((points_world.shape[0], 1), dtype=np.float64)], axis=1)
    pts_cam = (w2c @ pts_h.T).T[:, :3]
    z = pts_cam[:, 2]
    valid = np.isfinite(z) & (z > 0.0) & (z < float(max_depth_m))
    if not np.any(valid):
        return depth, 0

    pts_cam = pts_cam[valid]
    z = z[valid]
    u = K[0, 0] * (pts_cam[:, 0] / z) + K[0, 2]
    v = K[1, 1] * (pts_cam[:, 1] / z) + K[1, 2]
    px = np.rint(u).astype(np.int64)
    py = np.rint(v).astype(np.int64)
    inside = (px >= 0) & (px < width) & (py >= 0) & (py < height)
    if not np.any(inside):
        return depth, 0

    px = px[inside]
    py = py[inside]
    z = z[inside].astype(np.float32)
    order = np.argsort(z)
    written = 0
    for x, y, d in zip(px[order], py[order], z[order]):
        if depth[y, x] == 0.0:
            depth[y, x] = d
            written += 1
    return depth, written


def _convert_split(scene_root: Path, split: str, image_to_cam_idx, points_by_cam, max_depth_m: float) -> tuple[int, int]:
    split_root = scene_root / split
    rgb_dir = split_root / "rgb"
    pose_dir = split_root / "poses"
    calib_dir = split_root / "calibration"
    out_dir = split_root / "sparse_depth"
    out_dir.mkdir(parents=True, exist_ok=True)

    rgb_files = sorted(rgb_dir.iterdir())
    pose_files = sorted(pose_dir.iterdir())
    calib_files = sorted(calib_dir.iterdir())
    if not (len(rgb_files) == len(pose_files) == len(calib_files)):
        raise ValueError(
            f"Count mismatch at {split_root}: "
            f"rgb={len(rgb_files)} poses={len(pose_files)} calibration={len(calib_files)}"
        )

    frames = 0
    total_points = 0
    for rgb_path, pose_path, calib_path in zip(rgb_files, pose_files, calib_files):
        cam_idx = image_to_cam_idx.get(rgb_path.name)
        if cam_idx is None:
            continue

        image = imageio.imread(rgb_path)
        height, width = int(image.shape[0]), int(image.shape[1])
        c2w = np.loadtxt(pose_path).astype(np.float64)
        K = _read_calibration(calib_path, width, height)
        pts_list = points_by_cam.get(cam_idx, [])
        points_world = np.stack(pts_list, axis=0) if pts_list else np.zeros((0, 3), dtype=np.float64)
        depth, written = _project_depth(points_world, c2w, K, width, height, max_depth_m)
        np.savez_compressed(out_dir / f"{rgb_path.stem}.npz", arr_0=depth)
        frames += 1
        total_points += written
    return frames, total_points


def convert_scene(scene_root: Path, max_depth_m: float) -> None:
    nvm_path = scene_root / "reconstruction.nvm"
    if not nvm_path.is_file():
        raise FileNotFoundError(nvm_path)
    image_to_cam_idx, points_by_cam = _load_nvm_observations(nvm_path)
    train_frames, train_points = _convert_split(scene_root, "train", image_to_cam_idx, points_by_cam, max_depth_m)
    test_frames, test_points = _convert_split(scene_root, "test", image_to_cam_idx, points_by_cam, max_depth_m)
    print(
        f"{scene_root.name}: train={train_frames} points={train_points} "
        f"test={test_frames} points={test_points}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Project Cambridge reconstruction.nvm into sparse depth maps.")
    parser.add_argument("cambridge_root", type=Path, help="Root containing Cambridge_* ACE scenes, or one scene root.")
    parser.add_argument("--scenes", nargs="+", default=None, help="Scene names under cambridge_root.")
    parser.add_argument("--max-depth-m", type=float, default=1000.0)
    args = parser.parse_args()

    root = args.cambridge_root.resolve()
    if args.scenes:
        scenes = [root / scene for scene in args.scenes]
    elif (root / "reconstruction.nvm").is_file():
        scenes = [root]
    else:
        scenes = sorted(p for p in root.iterdir() if p.is_dir() and (p / "reconstruction.nvm").is_file())
    if not scenes:
        raise SystemExit(f"No Cambridge scenes with reconstruction.nvm under {root}")
    for scene in scenes:
        convert_scene(scene, args.max_depth_m)


if __name__ == "__main__":
    main()
