#!/usr/bin/env python3
"""Extract RIO10-style stable sparse depth from MuSHRoom Kinect captures."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np


@dataclass(frozen=True)
class FrameRecord:
    frame_id: int
    image_path: Path
    depth_path: Path
    pose_path: Path


def _read_intrinsics(path: Path) -> np.ndarray:
    k = np.loadtxt(path).astype(np.float64)
    if k.shape == (4, 4):
        k = k[:3, :3]
    if k.shape != (3, 3):
        raise ValueError(f"Unexpected intrinsics shape {k.shape} at {path}")
    return k


def _discover_frames(split_root: Path) -> list[FrameRecord]:
    image_ids = {int(p.stem): p for p in (split_root / "images").glob("*.png")}
    depth_ids = {int(p.stem): p for p in (split_root / "depth").glob("*.png")}
    pose_ids = {int(p.stem): p for p in (split_root / "pose").glob("*.txt")}
    frame_ids = sorted(set(image_ids) & set(depth_ids) & set(pose_ids))
    return [
        FrameRecord(fid, image_ids[fid], depth_ids[fid], pose_ids[fid])
        for fid in frame_ids
    ]


def _read_rgb(path: Path) -> np.ndarray:
    rgb = imageio.imread(path)
    if rgb.ndim != 3 or rgb.shape[2] < 3:
        raise ValueError(f"Unexpected RGB shape {rgb.shape} at {path}")
    return np.asarray(rgb[..., :3], dtype=np.uint8)


def _read_depth_m(path: Path) -> np.ndarray:
    raw = imageio.imread(path)
    if raw.ndim != 2:
        raise ValueError(f"Unexpected depth shape {raw.shape} at {path}")
    return raw.astype(np.float32) / 1000.0


def _read_pose(path: Path) -> np.ndarray:
    pose = np.loadtxt(path).astype(np.float64)
    if pose.shape != (4, 4):
        raise ValueError(f"Unexpected pose shape {pose.shape} at {path}")
    return pose


def _gradient_candidates(rgb: np.ndarray, depth_m: np.ndarray, pixel_step: int, texture_percentile: float) -> tuple[np.ndarray, np.ndarray]:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.sqrt(gx * gx + gy * gy)

    valid = np.isfinite(depth_m) & (depth_m > 0.0)
    grid = np.zeros_like(valid, dtype=bool)
    grid[::pixel_step, ::pixel_step] = True
    mask = valid & grid
    if not np.any(mask):
        return np.empty((0,), dtype=np.int32), np.empty((0,), dtype=np.int32)

    values = grad[mask]
    threshold = float(np.quantile(values, np.clip(texture_percentile, 0.0, 1.0)))
    keep = mask & (grad >= threshold)
    vs, us = np.where(keep)
    return us.astype(np.int32), vs.astype(np.int32)


def _backproject(us: np.ndarray, vs: np.ndarray, z_m: np.ndarray, k: np.ndarray) -> np.ndarray:
    fx = float(k[0, 0])
    fy = float(k[1, 1])
    cx = float(k[0, 2])
    cy = float(k[1, 2])
    x = (us.astype(np.float64) - cx) / fx * z_m
    y = (vs.astype(np.float64) - cy) / fy * z_m
    return np.stack([x, y, z_m.astype(np.float64)], axis=1)


def _transform_points(c2w: np.ndarray, pts_cam: np.ndarray) -> np.ndarray:
    homo = np.concatenate([pts_cam, np.ones((pts_cam.shape[0], 1), dtype=np.float64)], axis=1)
    world = (c2w @ homo.T).T
    return world[:, :3]


def _project_to_image(w2c: np.ndarray, pts_world: np.ndarray, k: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    homo = np.concatenate([pts_world, np.ones((pts_world.shape[0], 1), dtype=np.float64)], axis=1)
    cam = (w2c @ homo.T).T[:, :3]
    z = cam[:, 2]
    fx = float(k[0, 0])
    fy = float(k[1, 1])
    cx = float(k[0, 2])
    cy = float(k[1, 2])
    u = fx * (cam[:, 0] / np.maximum(z, 1e-8)) + cx
    v = fy * (cam[:, 1] / np.maximum(z, 1e-8)) + cy
    return u, v, z


def _validate_support(
    pts_world: np.ndarray,
    neighbor_frames: list[FrameRecord],
    k: np.ndarray,
    min_depth: float,
    max_depth: float,
    depth_abs_tol: float,
    depth_rel_tol: float,
) -> np.ndarray:
    support = np.zeros((pts_world.shape[0],), dtype=np.int16)
    for frame in neighbor_frames:
        depth = _read_depth_m(frame.depth_path)
        h, w = depth.shape
        w2c = np.linalg.inv(_read_pose(frame.pose_path))
        u, v, z = _project_to_image(w2c, pts_world, k)
        finite_proj = np.isfinite(u) & np.isfinite(v) & np.isfinite(z)
        ui = np.zeros_like(z, dtype=np.int32)
        vi = np.zeros_like(z, dtype=np.int32)
        ui[finite_proj] = np.rint(u[finite_proj]).astype(np.int32)
        vi[finite_proj] = np.rint(v[finite_proj]).astype(np.int32)
        inside = (
            finite_proj
            & (z > min_depth)
            & (z < max_depth)
            & (ui >= 0)
            & (vi >= 0)
            & (ui < w)
            & (vi < h)
        )
        if not np.any(inside):
            continue
        target = np.zeros_like(z, dtype=np.float64)
        target[inside] = depth[vi[inside], ui[inside]]
        finite = inside & np.isfinite(target) & (target >= min_depth) & (target <= max_depth)
        diff = np.abs(target - z)
        consistent = finite & ((diff <= depth_abs_tol) | (diff <= depth_rel_tol * np.maximum(target, 1e-6)))
        support += consistent.astype(np.int16)
    return support


def _write_sparse_depth(path: Path, shape: tuple[int, int], us: np.ndarray, vs: np.ndarray, z_m: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sparse = np.zeros(shape, dtype=np.uint16)
    sparse[vs.astype(np.int32), us.astype(np.int32)] = np.clip(np.rint(z_m * 1000.0), 0, np.iinfo(np.uint16).max).astype(np.uint16)
    imageio.imwrite(path, sparse)


def _write_overlay(path: Path, rgb: np.ndarray, us: np.ndarray, vs: np.ndarray, z_m: np.ndarray, radius: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    overlay = rgb.copy()
    if z_m.size:
        z_min = float(np.percentile(z_m, 2))
        z_max = float(np.percentile(z_m, 98))
        denom = max(z_max - z_min, 1e-6)
        norm = np.clip((z_m - z_min) / denom, 0.0, 1.0)
        colors_bgr = cv2.applyColorMap((norm * 255.0).astype(np.uint8), cv2.COLORMAP_TURBO)
        colors_rgb = colors_bgr[:, 0, ::-1]
        for u, v, color in zip(us, vs, colors_rgb):
            cv2.circle(overlay, (int(u), int(v)), radius, color.tolist(), thickness=-1)
    imageio.imwrite(path, overlay)


def _process_split(
    split_root: Path,
    out_dir: Path,
    *,
    frame_stride: int,
    neighbor_radius: int,
    pixel_step: int,
    texture_percentile: float,
    min_views: int,
    min_depth: float,
    max_depth: float,
    depth_abs_tol: float,
    depth_rel_tol: float,
    write_overlays: bool,
    overlay_radius: int,
) -> dict[str, int]:
    k = _read_intrinsics(split_root / "intrinsic" / "intrinsic_color.txt")
    frames = _discover_frames(split_root)
    if not frames:
        raise ValueError(f"No complete frames under {split_root}")

    frame_by_id = {frame.frame_id: frame for frame in frames}
    ordered_ids = [frame.frame_id for frame in frames]
    ref_ids = ordered_ids[::max(int(frame_stride), 1)]
    kept_frames = 0
    kept_points = 0

    for ref_id in ref_ids:
        frame = frame_by_id[ref_id]
        rgb = _read_rgb(frame.image_path)
        depth = _read_depth_m(frame.depth_path)
        us, vs = _gradient_candidates(rgb, depth, pixel_step, texture_percentile)
        if us.size == 0:
            continue

        z_ref = depth[vs, us].astype(np.float64)
        pts_world = _transform_points(_read_pose(frame.pose_path), _backproject(us, vs, z_ref, k))

        neighbor_frames = []
        for offset in range(-neighbor_radius, neighbor_radius + 1):
            idx = ref_id + offset
            if idx in frame_by_id:
                neighbor_frames.append(frame_by_id[idx])

        support = _validate_support(
            pts_world,
            neighbor_frames,
            k,
            min_depth,
            max_depth,
            depth_abs_tol,
            depth_rel_tol,
        )
        keep = support >= int(min_views)
        if not np.any(keep):
            continue

        stem = f"{ref_id:06d}"
        _write_sparse_depth(out_dir / "sparse_depth" / f"{stem}.stable.depth.png", depth.shape, us[keep], vs[keep], z_ref[keep])
        if write_overlays:
            _write_overlay(out_dir / "overlays" / f"{stem}.stable.overlay.png", rgb, us[keep], vs[keep], z_ref[keep], overlay_radius)
        kept_frames += 1
        kept_points += int(np.sum(keep))

    return {
        "frames_total": len(frames),
        "frames_written": kept_frames,
        "points_written": kept_points,
    }


def _link_sparse_depth_to_wai(ace_sparse_dir: Path, wai_scene: Path, *, write_overlays: bool) -> None:
    wai_sparse_dir = wai_scene / "sparse_depth"
    wai_sparse_dir.mkdir(parents=True, exist_ok=True)
    for stale in wai_sparse_dir.glob("*.png"):
        if stale.is_symlink():
            stale.unlink()
    for src in ace_sparse_dir.glob("*.png"):
        dst = wai_sparse_dir / src.name
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        dst.symlink_to(src.resolve())

    if write_overlays:
        wai_overlay_dir = wai_scene / "overlays"
        wai_overlay_dir.mkdir(parents=True, exist_ok=True)
        for stale in wai_overlay_dir.glob("*.png"):
            if stale.is_symlink():
                stale.unlink()
        for src in (ace_sparse_dir.parent / "overlays").glob("*.png"):
            dst = wai_overlay_dir / src.name
            if dst.exists() or dst.is_symlink():
                dst.unlink()
            dst.symlink_to(src.resolve())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mushroom-root", type=Path, default=Path("/data/xwh/MuSHRoom/kinect"))
    parser.add_argument("--ace-root", type=Path, default=Path("/data/xwh/MuSHRoom_ace"))
    parser.add_argument("--wai-root", type=Path, default=Path("/data/xwh/MuSHRoom_wai/kinect"))
    parser.add_argument("--rooms", nargs="+", required=True)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--neighbor-radius", type=int, default=3)
    parser.add_argument("--pixel-step", type=int, default=16)
    parser.add_argument("--texture-percentile", type=float, default=0.70)
    parser.add_argument("--min-views", type=int, default=3)
    parser.add_argument("--min-depth", type=float, default=0.1)
    parser.add_argument("--max-depth", type=float, default=20.0)
    parser.add_argument("--depth-abs-tol", type=float, default=0.03)
    parser.add_argument("--depth-rel-tol", type=float, default=0.02)
    parser.add_argument("--write-overlays", action="store_true")
    parser.add_argument("--overlay-radius", type=int, default=3)
    parser.add_argument("--skip-wai", action="store_true", help="Only write ACE sparse_depth outputs.")
    parser.add_argument("--link-existing-only", action="store_true", help="Only link existing ACE sparse_depth into WAI scenes.")
    args = parser.parse_args()

    for room in args.rooms:
        room_root = args.mushroom_root / "room_datasets" / room / "kinect"
        split_specs = [
            ("train", room_root / "long_capture", args.ace_root / room / "train"),
            ("test", room_root / "short_capture", args.ace_root / room / "test"),
        ]
        for split_name, split_root, ace_out in split_specs:
            if args.link_existing_only:
                ace_sparse_dir = ace_out / "sparse_depth"
                if not ace_sparse_dir.exists():
                    raise FileNotFoundError(f"Missing ACE sparse_depth dir: {ace_sparse_dir}")
                print(f"ACE sparse depth: {ace_sparse_dir} existing_files={len(list(ace_sparse_dir.glob('*.png')))}")
            else:
                stats = _process_split(
                    split_root,
                    ace_out,
                    frame_stride=args.frame_stride,
                    neighbor_radius=args.neighbor_radius,
                    pixel_step=args.pixel_step,
                    texture_percentile=args.texture_percentile,
                    min_views=args.min_views,
                    min_depth=args.min_depth,
                    max_depth=args.max_depth,
                    depth_abs_tol=args.depth_abs_tol,
                    depth_rel_tol=args.depth_rel_tol,
                    write_overlays=args.write_overlays,
                    overlay_radius=args.overlay_radius,
                )
                print(f"ACE sparse depth: {ace_out / 'sparse_depth'} {stats}")

            if not args.skip_wai:
                wai_scene = args.wai_root / f"{room}_{split_name}"
                _link_sparse_depth_to_wai(ace_out / "sparse_depth", wai_scene, write_overlays=args.write_overlays)
                print(f"WAI sparse depth: {wai_scene / 'sparse_depth'} linked_from={ace_out / 'sparse_depth'}")


if __name__ == "__main__":
    main()
