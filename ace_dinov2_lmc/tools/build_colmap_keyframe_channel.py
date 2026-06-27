#!/usr/bin/env python3
"""Build a patch-aligned COLMAP keyframe channel for STGS inter-frame ACE loss."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from tqdm import tqdm


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


@dataclass(frozen=True)
class Frame:
    image_idx: int
    name: str
    stem: str
    rgb_path: Path
    orig_width: int
    orig_height: int
    model_width: int
    model_height: int
    scale_x: float
    scale_y: float
    c2w: np.ndarray


def _frame_id(path: Path) -> str:
    stem = path.stem
    return stem.rsplit("_", 1)[-1] if "_" in stem else stem


def _round_to_multiple(value: float, multiple: int) -> int:
    if multiple <= 1:
        return max(1, int(round(value)))
    return max(multiple, int(round(float(value) / float(multiple)) * multiple))


def _target_size(orig_h: int, orig_w: int, image_height: int, image_width: int | None, round_multiple: int) -> tuple[int, int]:
    target_h = _round_to_multiple(float(image_height), round_multiple)
    if image_width is None:
        target_w = _round_to_multiple(float(orig_w) * float(target_h) / float(max(1, orig_h)), round_multiple)
    else:
        target_w = _round_to_multiple(float(image_width), round_multiple)
    return target_h, target_w


def _load_frames(split_root: Path, image_height: int, image_width: int | None, round_multiple: int) -> list[Frame]:
    rgb_dir = split_root / "rgb"
    pose_dir = split_root / "poses"
    poses = {_frame_id(p): p for p in pose_dir.iterdir() if p.is_file()}
    frames: list[Frame] = []
    for image_idx, rgb_path in enumerate(sorted(p for p in rgb_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)):
        fid = _frame_id(rgb_path)
        if fid not in poses:
            raise FileNotFoundError(f"Missing pose for {rgb_path.name}: expected id={fid}")
        image = imageio.imread(rgb_path)
        orig_h, orig_w = int(image.shape[0]), int(image.shape[1])
        model_h, model_w = _target_size(orig_h, orig_w, image_height, image_width, round_multiple)
        frames.append(
            Frame(
                image_idx=int(image_idx),
                name=rgb_path.name,
                stem=rgb_path.stem,
                rgb_path=rgb_path,
                orig_width=orig_w,
                orig_height=orig_h,
                model_width=model_w,
                model_height=model_h,
                scale_x=float(model_w) / float(max(1, orig_w)),
                scale_y=float(model_h) / float(max(1, orig_h)),
                c2w=np.loadtxt(poses[fid]).astype(np.float64),
            )
        )
    if not frames:
        raise ValueError(f"No RGB frames found under {rgb_dir}")
    return frames


def _track_length(point) -> int:
    track = getattr(point, "track", None)
    elems = getattr(track, "elements", None) if track is not None else None
    if elems is not None:
        return len(elems)
    try:
        return len(track)
    except Exception:
        return 0


def _point_depth(frame: Frame, xyz: np.ndarray) -> float:
    w2c = np.linalg.inv(frame.c2w)
    xyz_h = np.array([float(xyz[0]), float(xyz[1]), float(xyz[2]), 1.0], dtype=np.float64)
    return float((w2c @ xyz_h)[2])


def _camera_center(frame: Frame) -> np.ndarray:
    return frame.c2w[:3, 3].astype(np.float64)


def _parallax_deg(frame_a: Frame, frame_b: Frame, xyz: np.ndarray) -> float:
    va = xyz.astype(np.float64) - _camera_center(frame_a)
    vb = xyz.astype(np.float64) - _camera_center(frame_b)
    na = float(np.linalg.norm(va))
    nb = float(np.linalg.norm(vb))
    if na <= 1e-9 or nb <= 1e-9:
        return 0.0
    cosang = float(np.dot(va, vb) / (na * nb))
    cosang = min(1.0, max(-1.0, cosang))
    return float(math.degrees(math.acos(cosang)))


def _xy_model(frame: Frame, xy_original: np.ndarray) -> np.ndarray:
    return np.array([xy_original[0] * frame.scale_x, xy_original[1] * frame.scale_y], dtype=np.float32)


def _feature_yx_and_center(xy_model: np.ndarray, frame: Frame, output_subsample: int) -> tuple[np.ndarray, np.ndarray, float, bool]:
    grid_w = int(math.ceil(float(frame.model_width) / float(output_subsample)))
    grid_h = int(math.ceil(float(frame.model_height) / float(output_subsample)))
    x_cell = int(math.floor(float(xy_model[0]) / float(output_subsample)))
    y_cell = int(math.floor(float(xy_model[1]) / float(output_subsample)))
    inside = 0 <= x_cell < grid_w and 0 <= y_cell < grid_h
    center = np.array(
        [output_subsample * (x_cell + 0.5), output_subsample * (y_cell + 0.5)],
        dtype=np.float32,
    )
    alignment_error = float(np.linalg.norm(xy_model.astype(np.float32) - center))
    return np.array([y_cell, x_cell], dtype=np.int32), center, alignment_error, inside


def _load_sparse_depth(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npz":
        data = np.load(path, allow_pickle=False)
        key = "arr_0" if "arr_0" in data else data.files[0]
        return np.asarray(data[key])
    if path.suffix.lower() == ".npy":
        return np.load(path, allow_pickle=False)
    return np.asarray(imageio.imread(path))


def _sparse_depth_lookup(sparse_dir: Path | None, frame: Frame, xy_original: np.ndarray, radius_px: int) -> bool:
    if sparse_dir is None:
        return True
    candidates = [
        sparse_dir / f"{frame.stem}.npz",
        sparse_dir / f"{frame.stem}.npy",
        sparse_dir / f"{frame.stem}.png",
    ]
    depth_path = next((p for p in candidates if p.exists()), None)
    if depth_path is None:
        return False
    depth = _load_sparse_depth(depth_path)
    if depth.ndim == 3:
        depth = np.squeeze(depth)
    x = int(round(float(xy_original[0])))
    y = int(round(float(xy_original[1])))
    h, w = depth.shape[:2]
    if x < 0 or x >= w or y < 0 or y >= h:
        return False
    r = max(0, int(radius_px))
    patch = depth[max(0, y - r): min(h, y + r + 1), max(0, x - r): min(w, x + r + 1)]
    return bool(np.any(np.asarray(patch, dtype=np.float64) > 0.0))


def _image_name_index(reconstruction, frames: list[Frame]) -> dict[int, Frame]:
    by_name: dict[str, Frame] = {}
    for frame in frames:
        by_name[frame.name] = frame
        by_name[frame.stem] = frame
        by_name[frame.rgb_path.as_posix()] = frame
    out: dict[int, Frame] = {}
    for image in reconstruction.images.values():
        candidates = [str(image.name), Path(str(image.name)).name, Path(str(image.name)).stem]
        frame = next((by_name[c] for c in candidates if c in by_name), None)
        if frame is not None:
            out[int(image.image_id)] = frame
    return out


def _collect_observations(reconstruction, point, image_id_to_frame: dict[int, Frame]):
    observations = []
    for element in point.track.elements:
        image_id = int(element.image_id)
        frame = image_id_to_frame.get(image_id)
        if frame is None:
            continue
        try:
            image = reconstruction.images[image_id]
            point2d = image.points2D[element.point2D_idx]
        except Exception:
            continue
        if not point2d.has_point3D():
            continue
        xy = np.asarray(point2d.xy, dtype=np.float64)
        if not np.all(np.isfinite(xy)):
            continue
        observations.append((frame, xy))
    return observations


def _row_score(row: dict) -> tuple[float, float, int, float]:
    return (
        float(row["anchor_alignment_error_px"]),
        -float(row["parallax_deg"]),
        -int(row["track_length"]),
        float(row["colmap_reproj_error"]),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scene_root", type=Path, help="ACE-format scene root containing train/rgb, train/poses, train/calibration.")
    parser.add_argument("--split", default="train")
    parser.add_argument("--model-dir", type=Path, required=True, help="COLMAP sparse/triangulated model directory.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Default: <scene>/<split>/colmap_keyframe_channel_v1")
    parser.add_argument("--image-resolution", type=int, default=512)
    parser.add_argument("--image-width", type=int, default=None)
    parser.add_argument("--output-subsample", type=int, default=8, help="ACE-FCN=8, DINOv2=14.")
    parser.add_argument("--round-image-multiple", type=int, default=1, help="ACE-FCN=1, DINOv2=14.")
    parser.add_argument("--min-track-length", type=int, default=4)
    parser.add_argument("--max-reproj-error", type=float, default=2.0)
    parser.add_argument("--min-depth-m", type=float, default=0.1)
    parser.add_argument("--max-depth-m", type=float, default=1000.0)
    parser.add_argument("--min-parallax-deg", type=float, default=2.0)
    parser.add_argument("--max-parallax-deg", type=float, default=60.0)
    parser.add_argument("--max-anchor-alignment-px", type=float, default=2.0)
    parser.add_argument("--alignment-sigma-px", type=float, default=None)
    parser.add_argument("--sparse-depth-dir", type=Path, default=None, help="Optional sparse-depth dir; if set, anchors must hit a positive sparse-depth seed.")
    parser.add_argument("--sparse-depth-radius-px", type=int, default=0)
    parser.add_argument("--skip-depth-filter", action="store_true", help="Use only COLMAP observations and 2D filters; disables ACE-pose positive-depth checks.")
    args = parser.parse_args()

    import pycolmap

    split_root = args.scene_root / args.split
    frames = _load_frames(split_root, args.image_resolution, args.image_width, args.round_image_multiple)
    reconstruction = pycolmap.Reconstruction(str(args.model_dir))
    image_id_to_frame = _image_name_index(reconstruction, frames)
    if not image_id_to_frame:
        raise RuntimeError("No COLMAP images matched ACE rgb filenames.")

    output_dir = args.output_dir or (split_root / "colmap_keyframe_channel_v1")
    output_dir.mkdir(parents=True, exist_ok=True)
    sigma = float(args.alignment_sigma_px or max(args.max_anchor_alignment_px, 1e-6))
    best_by_cell: dict[tuple[int, int, int], dict] = {}
    points_seen = 0
    points_kept = 0
    candidate_rows = 0

    for point3d_id, point in tqdm(reconstruction.points3D.items(), desc="Build keyframe channel"):
        points_seen += 1
        track_len = _track_length(point)
        if track_len < int(args.min_track_length):
            continue
        reproj_error = float(getattr(point, "error", 0.0))
        if not np.isfinite(reproj_error) or reproj_error > float(args.max_reproj_error):
            continue
        xyz = np.asarray(point.xyz, dtype=np.float64)
        if not np.all(np.isfinite(xyz)):
            continue
        observations = _collect_observations(reconstruction, point, image_id_to_frame)
        if len(observations) < 2:
            continue
        points_kept += 1

        valid_obs = []
        for frame, xy in observations:
            xy_m = _xy_model(frame, xy)
            inside_xy = 0.0 <= xy_m[0] <= frame.model_width and 0.0 <= xy_m[1] <= frame.model_height
            if not inside_xy:
                continue
            if not args.skip_depth_filter:
                depth = _point_depth(frame, xyz)
                if not np.isfinite(depth) or depth <= args.min_depth_m or depth >= args.max_depth_m:
                    continue
            valid_obs.append((frame, xy, xy_m))
        if len(valid_obs) < 2:
            continue

        for anchor_frame, anchor_xy, anchor_xy_m in valid_obs:
            feature_yx, patch_center, align_err, inside_feature = _feature_yx_and_center(
                anchor_xy_m,
                anchor_frame,
                int(args.output_subsample),
            )
            if not inside_feature or align_err > float(args.max_anchor_alignment_px):
                continue
            if not _sparse_depth_lookup(args.sparse_depth_dir, anchor_frame, anchor_xy, int(args.sparse_depth_radius_px)):
                continue

            best_target = None
            best_parallax = -1.0
            for target_frame, target_xy, target_xy_m in valid_obs:
                if target_frame.image_idx == anchor_frame.image_idx:
                    continue
                parallax = _parallax_deg(anchor_frame, target_frame, xyz)
                if parallax < float(args.min_parallax_deg) or parallax > float(args.max_parallax_deg):
                    continue
                if parallax > best_parallax:
                    best_parallax = parallax
                    best_target = (target_frame, target_xy, target_xy_m, parallax)
            if best_target is None:
                continue

            target_frame, target_xy, target_xy_m, parallax = best_target
            alignment_weight = float(math.exp(-0.5 * (align_err / max(sigma, 1e-6)) ** 2))
            row = {
                "anchor_image_idx": int(anchor_frame.image_idx),
                "anchor_xy_original": anchor_xy.astype(np.float32),
                "anchor_xy_model": anchor_xy_m.astype(np.float32),
                "anchor_feature_yx": feature_yx.astype(np.int32),
                "anchor_patch_center_xy": patch_center.astype(np.float32),
                "anchor_alignment_error_px": float(align_err),
                "anchor_is_sparse_seed": True,
                "alignment_weight": alignment_weight,
                "point3D_id": int(point3d_id),
                "target_image_idx": int(target_frame.image_idx),
                "target_xy_original": target_xy.astype(np.float32),
                "target_xy_model": target_xy_m.astype(np.float32),
                "target_track_flag": True,
                "track_flag": True,
                "parallax_deg": float(parallax),
                "colmap_reproj_error": float(reproj_error),
                "track_length": int(track_len),
                "track_xyz_world": xyz.astype(np.float32),
            }
            key = (int(anchor_frame.image_idx), int(feature_yx[0]), int(feature_yx[1]))
            prev = best_by_cell.get(key)
            if prev is None or _row_score(row) < _row_score(prev):
                best_by_cell[key] = row
            candidate_rows += 1

    rows = list(best_by_cell.values())
    rows.sort(key=lambda r: (r["anchor_image_idx"], int(r["anchor_feature_yx"][0]), int(r["anchor_feature_yx"][1])))

    def stack(name, dtype=None):
        if not rows:
            return np.zeros((0,), dtype=dtype or np.float32)
        arr = np.stack([np.asarray(row[name]) for row in rows], axis=0)
        return arr.astype(dtype) if dtype is not None else arr

    def array(name, dtype):
        return np.asarray([row[name] for row in rows], dtype=dtype)

    out_npz = output_dir / "keyframe_channel.npz"
    np.savez_compressed(
        out_npz,
        schema_version=np.asarray("colmap_keyframe_channel_v1"),
        anchor_image_idx=array("anchor_image_idx", np.int64),
        anchor_xy_original=stack("anchor_xy_original", np.float32).reshape(-1, 2),
        anchor_xy_model=stack("anchor_xy_model", np.float32).reshape(-1, 2),
        anchor_feature_yx=stack("anchor_feature_yx", np.int32).reshape(-1, 2),
        anchor_patch_center_xy=stack("anchor_patch_center_xy", np.float32).reshape(-1, 2),
        anchor_alignment_error_px=array("anchor_alignment_error_px", np.float32),
        anchor_is_sparse_seed=array("anchor_is_sparse_seed", bool),
        alignment_weight=array("alignment_weight", np.float32),
        point3D_id=array("point3D_id", np.int64),
        target_image_idx=array("target_image_idx", np.int64),
        target_xy_original=stack("target_xy_original", np.float32).reshape(-1, 2),
        target_xy_model=stack("target_xy_model", np.float32).reshape(-1, 2),
        target_track_flag=array("target_track_flag", bool),
        track_flag=array("track_flag", bool),
        parallax_deg=array("parallax_deg", np.float32),
        colmap_reproj_error=array("colmap_reproj_error", np.float32),
        track_length=array("track_length", np.int32),
        track_xyz_world=stack("track_xyz_world", np.float32).reshape(-1, 3),
    )

    summary = {
        "schema_version": "colmap_keyframe_channel_v1",
        "scene_root": str(args.scene_root),
        "split": str(args.split),
        "model_dir": str(args.model_dir),
        "num_train_frames": len(frames),
        "num_colmap_images_matched": len(image_id_to_frame),
        "points_seen": points_seen,
        "points_kept_after_point_filters": points_kept,
        "candidate_rows_before_cell_dedup": candidate_rows,
        "rows": len(rows),
        "output_subsample": int(args.output_subsample),
        "round_image_multiple": int(args.round_image_multiple),
        "image_resolution": int(args.image_resolution),
        "max_anchor_alignment_px": float(args.max_anchor_alignment_px),
        "min_parallax_deg": float(args.min_parallax_deg),
        "max_parallax_deg": float(args.max_parallax_deg),
        "min_track_length": int(args.min_track_length),
        "max_reproj_error": float(args.max_reproj_error),
        "sparse_depth_dir": str(args.sparse_depth_dir) if args.sparse_depth_dir is not None else None,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote {out_npz}")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
