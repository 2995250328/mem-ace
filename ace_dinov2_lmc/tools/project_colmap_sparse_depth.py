#!/usr/bin/env python3
"""Project a COLMAP point cloud into ACE-format sparse depth using GT poses."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from tqdm import tqdm


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


@dataclass(frozen=True)
class Frame:
    name: str
    rgb_path: Path
    width: int
    height: int
    focal: float
    c2w: np.ndarray


def _frame_id(path: Path) -> str:
    stem = path.stem
    return stem.rsplit("_", 1)[-1] if "_" in stem else stem


def _read_focal(path: Path) -> float:
    calib = np.loadtxt(path).astype(np.float64)
    flat = np.asarray(calib).reshape(-1)
    if flat.size == 1:
        return float(flat[0])
    if calib.shape == (3, 3):
        return float(calib[0, 0])
    raise ValueError(f"Unsupported calibration shape {calib.shape} at {path}")


def _load_frames(split_root: Path) -> list[Frame]:
    rgb_dir = split_root / "rgb"
    pose_dir = split_root / "poses"
    calib_dir = split_root / "calibration"
    poses = {_frame_id(p): p for p in pose_dir.iterdir() if p.is_file()}
    calibs = {_frame_id(p): p for p in calib_dir.iterdir() if p.is_file()}
    frames: list[Frame] = []
    for rgb_path in sorted(p for p in rgb_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS):
        fid = _frame_id(rgb_path)
        image = imageio.imread(rgb_path)
        frames.append(
            Frame(
                name=rgb_path.name,
                rgb_path=rgb_path,
                width=int(image.shape[1]),
                height=int(image.shape[0]),
                focal=_read_focal(calibs[fid]),
                c2w=np.loadtxt(poses[fid]).astype(np.float64),
            )
        )
    if not frames:
        raise ValueError(f"No frames found under {rgb_dir}")
    return frames


def _track_length(point) -> int:
    track = getattr(point, "track", None)
    if track is None:
        return 0
    elems = getattr(track, "elements", None)
    if elems is not None:
        return len(elems)
    try:
        return len(track)
    except TypeError:
        return 0


def _load_filtered_points(model_dir: Path, min_track_length: int, max_reproj_error: float):
    import pycolmap

    reconstruction = pycolmap.Reconstruction(str(model_dir))
    points = []
    colors = []
    observed_by_image: dict[int, list[tuple[np.ndarray, np.ndarray]]] = {}
    for point in reconstruction.points3D.values():
        if _track_length(point) < min_track_length:
            continue
        error = float(getattr(point, "error", 0.0))
        if not np.isfinite(error) or error > max_reproj_error:
            continue
        xyz = np.asarray(point.xyz, dtype=np.float64)
        if not np.all(np.isfinite(xyz)):
            continue
        color = (
            np.asarray(point.color, dtype=np.uint8)
            if hasattr(point, "color")
            else np.array([255, 255, 255], dtype=np.uint8)
        )
        points.append(xyz)
        colors.append(color)
        for element in point.track.elements:
            try:
                image = reconstruction.images[element.image_id]
            except Exception:
                continue
            point2d = image.points2D[element.point2D_idx]
            if not point2d.has_point3D():
                continue
            xy = np.asarray(point2d.xy, dtype=np.float64)
            if np.all(np.isfinite(xy)):
                observed_by_image.setdefault(int(element.image_id), []).append((xyz, xy))
    if not points:
        return (
            reconstruction,
            np.zeros((0, 3), dtype=np.float64),
            np.zeros((0, 3), dtype=np.uint8),
            observed_by_image,
        )
    return reconstruction, np.stack(points, axis=0), np.stack(colors, axis=0), observed_by_image


def _nms_order(depth: np.ndarray, ys: np.ndarray, xs: np.ndarray, mode: str) -> np.ndarray:
    vals = depth[ys, xs]
    if mode == "near":
        return np.argsort(vals)
    if mode == "far":
        return np.argsort(-vals)
    if mode == "uniform":
        keys = ((xs.astype(np.uint64) * 73856093) ^ (ys.astype(np.uint64) * 19349663)) & np.uint64(0xFFFFFFFF)
        return np.argsort(keys, kind="stable")
    raise ValueError(mode)


def _depth_nms(depth: np.ndarray, radius: int, max_points: int, rank: str) -> tuple[np.ndarray, int]:
    if radius <= 0 and max_points <= 0:
        return depth, int(np.count_nonzero(depth > 0))
    ys, xs = np.where(depth > 0)
    if ys.size == 0:
        return depth, 0
    order = _nms_order(depth, ys, xs, rank)
    ys = ys[order]
    xs = xs[order]
    if radius <= 0:
        keep = np.arange(ys.size, dtype=np.int64)
        if max_points > 0:
            keep = keep[:max_points]
    else:
        r2 = float(radius * radius)
        cell = max(1, radius)
        occupied: dict[tuple[int, int], list[tuple[int, int]]] = {}
        keep_list: list[int] = []
        for i, (y, x) in enumerate(zip(ys, xs)):
            gx = int(x) // cell
            gy = int(y) // cell
            blocked = False
            for ny in (gy - 1, gy, gy + 1):
                for nx in (gx - 1, gx, gx + 1):
                    for ky, kx in occupied.get((ny, nx), []):
                        dy = float(y - ky)
                        dx = float(x - kx)
                        if dx * dx + dy * dy <= r2:
                            blocked = True
                            break
                    if blocked:
                        break
                if blocked:
                    break
            if blocked:
                continue
            keep_list.append(i)
            occupied.setdefault((gy, gx), []).append((int(y), int(x)))
            if max_points > 0 and len(keep_list) >= max_points:
                break
        keep = np.asarray(keep_list, dtype=np.int64)
    out = np.zeros_like(depth)
    out[ys[keep], xs[keep]] = depth[ys[keep], xs[keep]]
    return out, int(keep.size)


def _project(points: np.ndarray, frame: Frame, min_depth: float, max_depth: float) -> tuple[np.ndarray, int]:
    depth = np.zeros((frame.height, frame.width), dtype=np.float32)
    if points.size == 0:
        return depth, 0
    w2c = np.linalg.inv(frame.c2w)
    pts_h = np.concatenate([points, np.ones((points.shape[0], 1), dtype=np.float64)], axis=1)
    pts_cam = (w2c @ pts_h.T).T[:, :3]
    z = pts_cam[:, 2]
    valid = np.isfinite(z) & (z > min_depth) & (z < max_depth)
    if not np.any(valid):
        return depth, 0
    pts_cam = pts_cam[valid]
    z = z[valid]
    u = frame.focal * (pts_cam[:, 0] / z) + frame.width / 2.0
    v = frame.focal * (pts_cam[:, 1] / z) + frame.height / 2.0
    px = np.rint(u).astype(np.int64)
    py = np.rint(v).astype(np.int64)
    inside = (px >= 0) & (px < frame.width) & (py >= 0) & (py < frame.height)
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


def _project_observed(
    observations: list[tuple[np.ndarray, np.ndarray]],
    frame: Frame,
    min_depth: float,
    max_depth: float,
) -> tuple[np.ndarray, int]:
    depth = np.zeros((frame.height, frame.width), dtype=np.float32)
    if not observations:
        return depth, 0
    points = np.stack([p for p, _ in observations], axis=0)
    xys = np.stack([xy for _, xy in observations], axis=0)
    w2c = np.linalg.inv(frame.c2w)
    pts_h = np.concatenate([points, np.ones((points.shape[0], 1), dtype=np.float64)], axis=1)
    pts_cam = (w2c @ pts_h.T).T[:, :3]
    z = pts_cam[:, 2]
    px = np.rint(xys[:, 0]).astype(np.int64)
    py = np.rint(xys[:, 1]).astype(np.int64)
    valid = (
        np.isfinite(z)
        & (z > min_depth)
        & (z < max_depth)
        & (px >= 0)
        & (px < frame.width)
        & (py >= 0)
        & (py < frame.height)
    )
    if not np.any(valid):
        return depth, 0
    px = px[valid]
    py = py[valid]
    z = z[valid].astype(np.float32)
    order = np.argsort(z)
    written = 0
    for x, y, d in zip(px[order], py[order], z[order]):
        if depth[y, x] == 0.0:
            depth[y, x] = d
            written += 1
    return depth, written


def _write_ply(path: Path, points: np.ndarray, colors: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for (x, y, z), (r, g, b) in zip(points.astype(np.float32), colors.astype(np.uint8)):
            f.write(f"{x} {y} {z} {int(r)} {int(g)} {int(b)}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scene_root", type=Path)
    parser.add_argument("--split", default="train", choices=["train", "test"])
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-subdir", required=True)
    parser.add_argument("--min-track-length", type=int, default=4)
    parser.add_argument("--max-reproj-error", type=float, default=2.0)
    parser.add_argument("--min-depth-m", type=float, default=0.2)
    parser.add_argument("--max-depth-m", type=float, default=100.0)
    parser.add_argument("--depth-nms-radius", type=int, default=4)
    parser.add_argument("--depth-nms-max-points", type=int, default=0)
    parser.add_argument("--depth-nms-rank", default="uniform", choices=["uniform", "near", "far"])
    parser.add_argument(
        "--visibility-mode",
        default="observed",
        choices=["observed", "all"],
        help="observed uses only COLMAP track observations for each image; all projects every point into every image.",
    )
    parser.add_argument("--export-filtered-ply", type=Path, default=None)
    args = parser.parse_args()

    split_root = args.scene_root / args.split
    frames = _load_frames(split_root)
    reconstruction, points, colors, observed_by_image = _load_filtered_points(
        args.model_dir,
        args.min_track_length,
        args.max_reproj_error,
    )
    image_id_by_name = {image.name: int(image.image_id) for image in reconstruction.images.values()}
    print(
        f"Filtered points: {len(points)} "
        f"(min_track_length={args.min_track_length}, max_reproj_error={args.max_reproj_error})"
    )
    if args.export_filtered_ply is not None:
        _write_ply(args.export_filtered_ply, points, colors)
        print(f"Filtered PLY written: {args.export_filtered_ply}")

    out_dir = split_root / args.output_subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    total_written = 0
    nonempty = 0
    for frame in tqdm(frames, desc="Project filtered sparse depth"):
        if args.visibility_mode == "observed":
            image_id = image_id_by_name.get(frame.name)
            observations = observed_by_image.get(image_id, []) if image_id is not None else []
            depth, _ = _project_observed(observations, frame, args.min_depth_m, args.max_depth_m)
        else:
            depth, _ = _project(points, frame, args.min_depth_m, args.max_depth_m)
        depth, written = _depth_nms(depth, args.depth_nms_radius, args.depth_nms_max_points, args.depth_nms_rank)
        np.savez_compressed(out_dir / f"{frame.rgb_path.stem}.npz", arr_0=depth)
        total_written += written
        nonempty += int(written > 0)
    print(f"Sparse depth written: {out_dir}")
    print(f"Non-empty frames: {nonempty}/{len(frames)}, projected pixels: {total_written}")
    print(f"Average points/frame: {total_written / max(1, len(frames)):.1f}")


if __name__ == "__main__":
    main()
