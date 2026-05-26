#!/usr/bin/env python3
"""Keep sparse-depth pixels that agree with SuperPoint keypoint locations."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from tqdm import tqdm


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def _load_gray(path: Path) -> np.ndarray:
    image = imageio.imread(path)
    if image.ndim == 2:
        gray = image.astype(np.float32)
    elif image.ndim == 3:
        rgb = image[..., :3].astype(np.float32)
        gray = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    else:
        raise ValueError(f"Unsupported image shape {image.shape} at {path}")
    if gray.max() > 1.0:
        gray /= 255.0
    return gray.astype(np.float32, copy=False)


def _resize_long_edge(image: np.ndarray, max_long_edge: int) -> tuple[np.ndarray, float]:
    if max_long_edge <= 0:
        return image, 1.0
    height, width = image.shape[:2]
    long_edge = max(height, width)
    if long_edge <= max_long_edge:
        return image, 1.0
    import cv2

    scale = float(max_long_edge) / float(long_edge)
    resized = cv2.resize(
        image,
        (max(8, int(round(width * scale))), max(8, int(round(height * scale)))),
        interpolation=cv2.INTER_AREA,
    )
    return resized.astype(np.float32, copy=False), scale


def _load_depth(path: Path) -> np.ndarray:
    z = np.load(path)
    if "arr_0" in z.files:
        return z["arr_0"]
    for key in ("depth", "sparse_depth", "depth_map", "z"):
        if key in z.files:
            return z[key]
    raise KeyError(f"No depth array in {path}, keys={z.files}")


def _extract_superpoint_xy(
    image_path: Path,
    frontend,
    max_keypoints: int,
    resize_max_long_edge: int,
) -> np.ndarray:
    image = _load_gray(image_path)
    image_infer, scale = _resize_long_edge(image, resize_max_long_edge)
    corners, _, _ = frontend.run(image_infer)
    if corners is None or corners.shape[1] == 0:
        return np.zeros((0, 2), dtype=np.float32)
    corners = corners.astype(np.float32, copy=False)
    if max_keypoints > 0 and corners.shape[1] > max_keypoints:
        corners = corners[:, :max_keypoints]
    xy = corners[:2, :].T
    if scale != 1.0:
        xy /= scale
    return xy.astype(np.float32, copy=False)


def _intersect_depth_with_keypoints(depth: np.ndarray, keypoints_xy: np.ndarray, radius: int) -> tuple[np.ndarray, int]:
    out = np.zeros_like(depth)
    ys, xs = np.where(depth > 0)
    if ys.size == 0 or keypoints_xy.size == 0:
        return out, 0

    radius = max(0, int(radius))
    if radius == 0:
        px = np.rint(keypoints_xy[:, 0]).astype(np.int64)
        py = np.rint(keypoints_xy[:, 1]).astype(np.int64)
        valid = (px >= 0) & (px < depth.shape[1]) & (py >= 0) & (py < depth.shape[0])
        px = px[valid]
        py = py[valid]
        keep = depth[py, px] > 0
        out[py[keep], px[keep]] = depth[py[keep], px[keep]]
        return out, int(np.count_nonzero(out > 0))

    cell = radius
    grid: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for y, x in zip(ys, xs):
        grid.setdefault((int(y) // cell, int(x) // cell), []).append((int(y), int(x)))

    r2 = float(radius * radius)
    kept: dict[tuple[int, int], float] = {}
    for x_f, y_f in keypoints_xy:
        gx = int(round(float(x_f))) // cell
        gy = int(round(float(y_f))) // cell
        best = None
        best_d2 = None
        for ny in (gy - 1, gy, gy + 1):
            for nx in (gx - 1, gx, gx + 1):
                for y, x in grid.get((ny, nx), []):
                    dx = float(x) - float(x_f)
                    dy = float(y) - float(y_f)
                    d2 = dx * dx + dy * dy
                    if d2 <= r2 and (best_d2 is None or d2 < best_d2):
                        best = (y, x)
                        best_d2 = d2
        if best is not None:
            y, x = best
            kept[(y, x)] = float(depth[y, x])

    for (y, x), value in kept.items():
        out[y, x] = value
    return out, len(kept)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scene_root", type=Path)
    parser.add_argument("--split", default="train", choices=["train", "test"])
    parser.add_argument("--input-subdir", required=True)
    parser.add_argument("--output-subdir", required=True)
    parser.add_argument("--superpoint-repo", type=Path, default=Path("/home/xwh/project/ace_depth"))
    parser.add_argument("--superpoint-weights", type=Path, default=Path("superpoint_v1.pth"))
    parser.add_argument("--superpoint-conf-thresh", type=float, default=0.03)
    parser.add_argument("--superpoint-max-keypoints", type=int, default=2048)
    parser.add_argument("--superpoint-nms-dist", type=int, default=6)
    parser.add_argument("--superpoint-nn-thresh", type=float, default=0.7)
    parser.add_argument("--superpoint-resize-max-long-edge", type=int, default=1600)
    parser.add_argument("--match-radius", type=int, default=4)
    parser.add_argument("--use-gpu", action="store_true")
    args = parser.parse_args()

    candidate_paths = [
        args.superpoint_repo,
        Path("/home/xwh/project/ace_depth"),
        Path("/data/xwh/SuperPointPretrainedNetwork"),
    ]
    for candidate in candidate_paths:
        if candidate.is_dir() and str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))
    import torch
    from superpoint import SuperPointFrontend, SuperPointNet

    frontend = SuperPointFrontend(
        superpoint_net=SuperPointNet(),
        weights_path=str(args.superpoint_weights),
        nms_dist=int(args.superpoint_nms_dist),
        conf_thresh=float(args.superpoint_conf_thresh),
        nn_thresh=float(args.superpoint_nn_thresh),
        cuda=bool(args.use_gpu and torch.cuda.is_available()),
    )

    split_root = args.scene_root / args.split
    rgb_dir = split_root / "rgb"
    depth_dir = split_root / args.input_subdir
    out_dir = split_root / args.output_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    before_total = 0
    after_total = 0
    nonempty = 0
    images = sorted(p for p in rgb_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    for image_path in tqdm(images, desc="Filter depth by SuperPoint"):
        depth_path = depth_dir / f"{image_path.stem}.npz"
        if not depth_path.is_file():
            raise FileNotFoundError(depth_path)
        depth = _load_depth(depth_path)
        keypoints_xy = _extract_superpoint_xy(
            image_path,
            frontend,
            max_keypoints=args.superpoint_max_keypoints,
            resize_max_long_edge=args.superpoint_resize_max_long_edge,
        )
        filtered, kept = _intersect_depth_with_keypoints(depth, keypoints_xy, args.match_radius)
        np.savez_compressed(out_dir / depth_path.name, arr_0=filtered)
        before_total += int(np.count_nonzero(depth > 0))
        after_total += kept
        nonempty += int(kept > 0)

    count = max(1, len(images))
    print(f"Input: {depth_dir}")
    print(f"Output: {out_dir}")
    print(f"Frames: {len(images)}, non-empty: {nonempty}/{len(images)}")
    print(f"Avg points/frame: {before_total / count:.1f} -> {after_total / count:.1f}")


if __name__ == "__main__":
    main()
