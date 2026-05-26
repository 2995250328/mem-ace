#!/usr/bin/env python3
"""Convert ACE-format scenes into lightweight WAI scene_meta scenes.

Input scene layout:
  <scene>/
    train/{rgb,poses,calibration}
    test/{rgb,poses,calibration}

Output layout:
  <output_root>/<scene>_train/
    scene_meta.json
    images/*
  <output_root>/<scene>_test/
    scene_meta.json
    images/*

The converter preserves ACE's sorted-file pairing contract. This is suitable
for datasets such as Cambridge and Wayspots where RGB, pose, and calibration
files already have one-to-one sorted ordering, but no depth modality exists.
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path

import imageio.v2 as imageio
import numpy as np


def _iter_scene_roots(root: Path, scenes: list[str] | None) -> list[Path]:
    if scenes:
        return [root / scene for scene in scenes]
    if (root / "train" / "rgb").is_dir() and (root / "test" / "rgb").is_dir():
        return [root]
    return sorted(
        p for p in root.iterdir()
        if p.is_dir() and (p / "train" / "rgb").is_dir() and (p / "test" / "rgb").is_dir()
    )


def _link_or_copy(src: Path, dst: Path, *, copy_files: bool) -> None:
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if copy_files:
        shutil.copy2(src, dst)
    else:
        dst.symlink_to(src.resolve())


def _read_pose(path: Path) -> np.ndarray:
    pose = np.loadtxt(path).astype(np.float64)
    if pose.shape == (4, 4):
        return pose
    if pose.shape == (3, 4):
        out = np.eye(4, dtype=np.float64)
        out[:3, :] = pose
        return out
    raise ValueError(f"Unexpected pose shape {pose.shape} at {path}")


def _read_calibration(path: Path, width: int, height: int) -> np.ndarray:
    calib = np.loadtxt(path).astype(np.float64)
    if np.size(calib) == 1:
        focal = float(np.asarray(calib).reshape(-1)[0])
        return np.array(
            [[focal, 0.0, width / 2.0], [0.0, focal, height / 2.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
    calib = np.asarray(calib, dtype=np.float64)
    if calib.shape == (3, 3):
        return calib
    raise ValueError(f"Unexpected calibration shape {calib.shape} at {path}")


def _image_size(path: Path) -> tuple[int, int]:
    image = imageio.imread(path)
    if image.ndim < 2:
        raise ValueError(f"Unexpected image shape {image.shape} at {path}")
    return int(image.shape[1]), int(image.shape[0])


def _depth_source_dirs(split_root: Path) -> list[str]:
    return [
        name
        for name in (
            "sparse_depth",
            "sparse_depth_mapanything",
            "sparse_depth_sampling_sp",
            "depth",
            "gt_depth",
            "colmap_depth",
        )
        if (split_root / name).is_dir()
    ]


def _find_depth_file(depth_dir: Path, stem: str) -> Path | None:
    for suffix in (".npz", ".npy", ".png", ".tif", ".tiff", ".exr", ".stable.depth.png", ".depth.png"):
        path = depth_dir / f"{stem}{suffix}"
        if path.is_file():
            return path
    return None


def _depth_format(path: Path) -> str:
    if path.suffix.lower() in {".npz", ".npy"}:
        return "numpy"
    if path.suffix.lower() in {".png", ".tif", ".tiff", ".exr"}:
        return "depth"
    return "numpy"


def _collect_split(split_root: Path) -> tuple[list[Path], list[Path], list[Path]]:
    rgb_dir = split_root / "rgb"
    pose_dir = split_root / "poses"
    calib_dir = split_root / "calibration"
    missing = [str(p) for p in (rgb_dir, pose_dir, calib_dir) if not p.is_dir()]
    if missing:
        raise FileNotFoundError(f"Missing ACE split dirs: {', '.join(missing)}")
    rgb_files = sorted(p for p in rgb_dir.iterdir() if p.is_file())
    pose_files = sorted(p for p in pose_dir.iterdir() if p.is_file())
    calib_files = sorted(p for p in calib_dir.iterdir() if p.is_file())
    if not (len(rgb_files) == len(pose_files) == len(calib_files)):
        raise ValueError(
            f"Count mismatch at {split_root}: "
            f"rgb={len(rgb_files)} poses={len(pose_files)} calibration={len(calib_files)}"
        )
    if not rgb_files:
        raise ValueError(f"No RGB files in {rgb_dir}")
    return rgb_files, pose_files, calib_files


def _write_wai_split(
    scene_root: Path,
    split: str,
    out_scene: Path,
    *,
    dataset_name: str,
    copy_files: bool,
) -> int:
    split_root = scene_root / split
    rgb_files, pose_files, calib_files = _collect_split(split_root)
    (out_scene / "images").mkdir(parents=True, exist_ok=True)
    depth_modalities = _depth_source_dirs(split_root)
    for modality in depth_modalities:
        (out_scene / modality).mkdir(parents=True, exist_ok=True)

    frame_modalities = {"image": {"frame_key": "image", "format": "image"}}
    for modality in depth_modalities:
        sample_depth = None
        for rgb_path in rgb_files:
            sample_depth = _find_depth_file(split_root / modality, rgb_path.stem)
            if sample_depth is not None:
                break
        frame_modalities[modality] = {"frame_key": modality, "format": _depth_format(sample_depth) if sample_depth else "numpy"}

    frames = []
    frame_names: dict[str, int] = {}
    for idx, (rgb_path, pose_path, calib_path) in enumerate(zip(rgb_files, pose_files, calib_files)):
        frame_name = rgb_path.stem
        # Keep frame names unique even when extensions/directories collide.
        if frame_name in frame_names:
            frame_name = f"{idx:06d}_{frame_name}"

        width, height = _image_size(rgb_path)
        pose = _read_pose(pose_path)
        K = _read_calibration(calib_path, width, height)

        image_rel = f"images/{frame_name}{rgb_path.suffix.lower()}"
        _link_or_copy(rgb_path, out_scene / image_rel, copy_files=copy_files)

        frame = {
            "frame_name": frame_name,
            "image": image_rel,
            "file_path": image_rel,
            "transform_matrix": pose.tolist(),
            "h": height,
            "w": width,
            "fl_x": float(K[0, 0]),
            "fl_y": float(K[1, 1]),
            "cx": float(K[0, 2]),
            "cy": float(K[1, 2]),
        }
        for modality in depth_modalities:
            depth_src = _find_depth_file(split_root / modality, rgb_path.stem)
            if depth_src is None:
                continue
            depth_rel = f"{modality}/{frame_name}{depth_src.suffix.lower()}"
            _link_or_copy(depth_src, out_scene / depth_rel, copy_files=copy_files)
            frame[modality] = depth_rel
        frames.append(frame)
        frame_names[frame_name] = idx

    scene_meta = {
        "scene_name": out_scene.name,
        "dataset_name": dataset_name,
        "version": "0.1",
        "shared_intrinsics": False,
        "camera_model": "PINHOLE",
        "camera_convention": "opencv",
        "scale_type": "metric",
        "frame_modalities": frame_modalities,
        "frames": frames,
        "frame_names": frame_names,
    }
    (out_scene / "scene_meta.json").write_text(
        json.dumps(scene_meta, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return len(frames)


def convert_scene(scene_root: Path, output_root: Path, *, dataset_name: str, copy_files: bool) -> None:
    scene_root = scene_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    stats: dict[str, int] = {}
    with tempfile.TemporaryDirectory(prefix=f".{scene_root.name}.tmp.", dir=output_root) as tmp_dir:
        tmp_root = Path(tmp_dir)
        for split in ("train", "test"):
            out_scene = tmp_root / f"{scene_root.name}_{split}"
            stats[split] = _write_wai_split(
                scene_root,
                split,
                out_scene,
                dataset_name=dataset_name,
                copy_files=copy_files,
            )
        for split in ("train", "test"):
            final_scene = output_root / f"{scene_root.name}_{split}"
            if final_scene.exists():
                shutil.rmtree(final_scene)
            (tmp_root / f"{scene_root.name}_{split}").rename(final_scene)

    print(
        f"WAI scene: {scene_root.name} -> {output_root} "
        f"train={stats['train']} test={stats['test']}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert ACE-format scenes to WAI scene_meta scenes.")
    parser.add_argument("ace_root", type=Path, help="ACE scene root or root containing multiple ACE scenes.")
    parser.add_argument("output_root", type=Path, help="Output root for WAI scenes.")
    parser.add_argument("--scenes", nargs="+", default=None, help="Scene names under ace_root to convert.")
    parser.add_argument("--dataset-name", default="ace_converted", help="dataset_name stored in scene_meta.json.")
    parser.add_argument("--copy-files", action="store_true", help="Copy RGB images instead of symlinking.")
    args = parser.parse_args()

    scenes = _iter_scene_roots(args.ace_root.resolve(), args.scenes)
    if not scenes:
        raise SystemExit(f"No ACE-format scenes found under {args.ace_root}")
    for scene in scenes:
        if not scene.is_dir():
            raise FileNotFoundError(scene)
        convert_scene(scene, args.output_root.resolve(), dataset_name=args.dataset_name, copy_files=args.copy_files)


if __name__ == "__main__":
    main()
