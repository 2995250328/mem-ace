#!/usr/bin/env python3
"""Extract an ACE-FCN feature-space memory bank.

This is intentionally separate from the MapAnything/DINO/GLACE BSE extractor:
the contract here is simple and explicit:

  query feature  = frozen ACE FCN encoder feature, stride 8, dim 512
  memory feature = frozen ACE FCN encoder feature, stride 8, dim 512
  memory point   = patch-level world scene coordinate from GT depth

The first version requires ACE-format ``train/depth``. If a scene only has
RGB+pose+intrinsics, this script fails instead of silently producing pseudo
coordinates.
"""

import argparse
import json
import logging
import math
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from skimage import io as skio
from skimage.transform import resize
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from ace_network_ace import RegressorACE
from dataset_origin import CamLocDataset


_logger = logging.getLogger("extract_memory_ace_fcn")


def _strtobool(x):
    if isinstance(x, bool):
        return x
    v = str(x).strip().lower()
    if v in ("1", "true", "yes", "on"):
        return True
    if v in ("0", "false", "no", "off"):
        return False
    raise argparse.ArgumentTypeError(f"invalid bool: {x!r}")


def _resolve_train_root(scene_path: Path) -> Path:
    scene_path = scene_path.resolve()
    if scene_path.name == "train":
        return scene_path
    return scene_path / "train"


def _scene_name_from_path(scene_path: Path) -> str:
    scene_path = scene_path.resolve()
    if scene_path.name in ("train", "test", "val"):
        return scene_path.parent.name
    return scene_path.name


def _valid_scene_coords(coords_b3hw: torch.Tensor) -> torch.Tensor:
    return torch.isfinite(coords_b3hw).all(dim=1, keepdim=True) & (coords_b3hw.abs().sum(dim=1, keepdim=True) > 0)


def _depth_key(path: Path) -> str:
    stem = path.stem
    for suffix in (".stable.depth", ".depth"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
    return stem


def _depth_keys(path: Path) -> list[str]:
    key = _depth_key(path)
    keys = [key]
    # Indoor6 ACE RGB files are named like "003537.jpg", while the
    # COLMAP-derived depth files in WAI-style folders use "image-003537.npz".
    if key.startswith("image-") and key[len("image-"):].isdigit():
        keys.append(key[len("image-"):])
    return keys


def _build_depth_path_map(depth_dir: Path):
    depth_dir = Path(depth_dir)
    if not depth_dir.exists():
        raise FileNotFoundError(f"depth_dir does not exist: {depth_dir}")
    paths = [p for p in depth_dir.iterdir() if p.is_file()]
    out = {}
    for path in paths:
        for key in _depth_keys(path):
            out[key] = path
    if not out:
        raise FileNotFoundError(f"depth_dir contains no files: {depth_dir}")
    return out


def _load_depth(path: Path) -> np.ndarray:
    path = Path(path)
    if path.suffix.lower() == ".npy":
        depth = np.load(path, allow_pickle=False).astype(np.float64, copy=False)
    elif path.suffix.lower() == ".npz":
        depth = np.load(path, allow_pickle=False)["arr_0"].astype(np.float64, copy=False)
    else:
        raw = skio.imread(path)
        depth = np.asarray(raw).astype(np.float64, copy=False)
        if path.suffix.lower() in {".png", ".tif", ".tiff"} and np.issubdtype(np.asarray(raw).dtype, np.integer):
            depth = depth / 1000.0
    if depth.ndim == 3:
        depth = np.squeeze(depth)
    if depth.ndim != 2:
        raise ValueError(f"Unsupported depth shape {depth.shape} in {path}")
    return np.where(np.isfinite(depth), depth, 0.0)


def _sample_patch_depth_nearest_valid(
    depth: np.ndarray,
    stride: int,
    coords_h: int,
    coords_w: int,
    *,
    depth_min: float,
    depth_max: float,
):
    patch_depth = np.zeros((coords_h, coords_w), dtype=np.float64)
    patch_px = np.zeros((coords_h, coords_w), dtype=np.float64)
    patch_py = np.zeros((coords_h, coords_w), dtype=np.float64)
    patch_valid = np.zeros((coords_h, coords_w), dtype=bool)
    image_h, image_w = depth.shape
    half = int(stride) // 2
    for gy in range(coords_h):
        cy = min(gy * stride + half, image_h - 1)
        y0 = gy * stride
        y1 = min((gy + 1) * stride, image_h)
        for gx in range(coords_w):
            cx = min(gx * stride + half, image_w - 1)
            x0 = gx * stride
            x1 = min((gx + 1) * stride, image_w)
            center_depth = depth[cy, cx]
            if np.isfinite(center_depth) and depth_min <= center_depth <= depth_max:
                patch_depth[gy, gx] = center_depth
                patch_px[gy, gx] = cx
                patch_py[gy, gx] = cy
                patch_valid[gy, gx] = True
                continue
            window = depth[y0:y1, x0:x1]
            valid = np.isfinite(window) & (window >= depth_min) & (window <= depth_max)
            if not np.any(valid):
                continue
            yy, xx = np.where(valid)
            abs_y = yy + y0
            abs_x = xx + x0
            best = int(np.argmin((abs_y - cy) ** 2 + (abs_x - cx) ** 2))
            py = int(abs_y[best])
            px = int(abs_x[best])
            patch_depth[gy, gx] = depth[py, px]
            patch_px[gy, gx] = px
            patch_py[gy, gx] = py
            patch_valid[gy, gx] = True
    return patch_depth, patch_px, patch_py, patch_valid


def _depth_to_patch_scene_coords(
    depth: np.ndarray,
    pose_c2w: torch.Tensor,
    intrinsics: torch.Tensor,
    image_hw,
    stride: int,
    *,
    depth_min: float,
    depth_max: float,
):
    image_h, image_w = image_hw
    if tuple(depth.shape) != (image_h, image_w):
        depth = resize(depth, (image_h, image_w), order=0, preserve_range=True, anti_aliasing=False)
    coords_h = math.ceil(image_h / stride)
    coords_w = math.ceil(image_w / stride)
    depth_patch, px, py, valid = _sample_patch_depth_nearest_valid(
        depth,
        stride,
        coords_h,
        coords_w,
        depth_min=depth_min,
        depth_max=depth_max,
    )
    coords = torch.zeros((3, coords_h, coords_w), dtype=torch.float32)
    if not np.any(valid):
        return coords
    K = intrinsics.detach().float().cpu().numpy()
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    eye = np.ones((4, coords_h, coords_w), dtype=np.float64)
    eye[0] = ((px - cx) / fx) * depth_patch
    eye[1] = ((py - cy) / fy) * depth_patch
    eye[2] = depth_patch
    eye[:, ~valid] = 0.0
    pose = pose_c2w.detach().float().cpu().numpy()
    scene = np.matmul(pose, eye.reshape(4, -1)).reshape(4, coords_h, coords_w)
    coords[:, :coords_h, :coords_w] = torch.from_numpy(scene[:3]).float()
    return coords


def _flatten_feature_grid(features_bchw, coords_b3hw, image_index: int, stride: int, samples_per_image: int, generator):
    if coords_b3hw.shape[-2:] != features_bchw.shape[-2:]:
        coords_b3hw = F.interpolate(coords_b3hw, size=features_bchw.shape[-2:], mode="nearest")
    valid = _valid_scene_coords(coords_b3hw)[0, 0]
    ys, xs = torch.where(valid)
    if ys.numel() == 0:
        return None
    if samples_per_image > 0 and ys.numel() > samples_per_image:
        perm = torch.randperm(ys.numel(), generator=generator)[:samples_per_image]
        ys = ys[perm]
        xs = xs[perm]
    feats = features_bchw[0, :, ys, xs].permute(1, 0).contiguous().float().cpu()
    pts = coords_b3hw[0, :, ys, xs].permute(1, 0).contiguous().float().cpu()
    img_idx = torch.full((pts.shape[0],), int(image_index), dtype=torch.long)
    pixel_xy = torch.stack(
        (
            xs.cpu().float() * float(stride) + float(stride) * 0.5,
            ys.cpu().float() * float(stride) + float(stride) * 0.5,
        ),
        dim=1,
    )
    return pts, feats, img_idx, pixel_xy


def _voxel_pool(points, features, image_indices, pixel_xy, voxel_size: float, max_points: int, generator):
    if voxel_size <= 0:
        pooled_points = points
        pooled_features = features
        pooled_image_indices = image_indices
        pooled_pixel_xy = pixel_xy
        cluster_sizes = torch.ones(points.shape[0], dtype=torch.long)
    else:
        voxel_ids = torch.floor(points / float(voxel_size)).to(torch.int64)
        unique_voxels, inverse = torch.unique(voxel_ids, dim=0, return_inverse=True)
        n_vox = unique_voxels.shape[0]
        pooled_points = torch.zeros((n_vox, 3), dtype=torch.float32)
        pooled_features = torch.zeros((n_vox, features.shape[1]), dtype=torch.float32)
        pooled_pixel_xy = torch.zeros((n_vox, 2), dtype=torch.float32)
        cluster_sizes = torch.bincount(inverse, minlength=n_vox).to(torch.long)
        pooled_points.index_add_(0, inverse, points.float())
        pooled_features.index_add_(0, inverse, features.float())
        pooled_pixel_xy.index_add_(0, inverse, pixel_xy.float())
        denom = cluster_sizes.clamp_min(1).float().unsqueeze(1)
        pooled_points = pooled_points / denom
        pooled_features = pooled_features / denom
        pooled_pixel_xy = pooled_pixel_xy / denom

        # Keep one trace image index per voxel. This is for debugging only, so
        # the first contributor is sufficient.
        order = torch.argsort(inverse)
        sorted_inverse = inverse[order]
        first = torch.ones_like(sorted_inverse, dtype=torch.bool)
        first[1:] = sorted_inverse[1:] != sorted_inverse[:-1]
        pooled_image_indices = image_indices[order[first]]

    if max_points > 0 and pooled_points.shape[0] > max_points:
        keep = torch.randperm(pooled_points.shape[0], generator=generator)[:max_points]
        pooled_points = pooled_points[keep]
        pooled_features = pooled_features[keep]
        pooled_image_indices = pooled_image_indices[keep]
        pooled_pixel_xy = pooled_pixel_xy[keep]
        cluster_sizes = cluster_sizes[keep]

    return pooled_points, pooled_features, pooled_image_indices, pooled_pixel_xy, cluster_sizes


def _summary(values):
    values = values.detach().float().cpu()
    if values.numel() == 0:
        return {"count": 0}
    return {
        "count": int(values.numel()),
        "mean": float(values.mean().item()),
        "std": float(values.std(unbiased=False).item()) if values.numel() > 1 else 0.0,
        "median": float(values.median().item()),
        "p90": float(torch.quantile(values, 0.90).item()),
    }


def _nearest_sanity_report(query_points, query_features, memory_points, memory_features, *, max_queries, chunk_size, seed):
    gen = torch.Generator().manual_seed(int(seed) + 177)
    n_query = query_points.shape[0]
    if n_query > max_queries:
        q_idx = torch.randperm(n_query, generator=gen)[:max_queries]
        query_points = query_points[q_idx]
        query_features = query_features[q_idx]

    qf = F.normalize(query_features.float(), dim=1, eps=1e-6)
    mf = F.normalize(memory_features.float(), dim=1, eps=1e-6)
    best_sim = []
    best_idx = []
    for start in range(0, qf.shape[0], chunk_size):
        q = qf[start:start + chunk_size]
        sim = q @ mf.t()
        val, idx = sim.max(dim=1)
        best_sim.append(val.cpu())
        best_idx.append(idx.cpu())
    best_sim = torch.cat(best_sim, dim=0)
    best_idx = torch.cat(best_idx, dim=0)

    rand_idx = torch.randint(0, memory_features.shape[0], (query_features.shape[0],), generator=gen)
    random_sim = (qf.cpu() * mf[rand_idx].cpu()).sum(dim=1)
    nn_3d = torch.linalg.norm(query_points.float() - memory_points[best_idx].float(), dim=1)
    random_3d = torch.linalg.norm(query_points.float() - memory_points[rand_idx].float(), dim=1)

    return {
        "num_queries": int(query_features.shape[0]),
        "nn_cosine": _summary(best_sim),
        "random_cosine": _summary(random_sim),
        "nn_3d_error_m": _summary(nn_3d),
        "random_3d_error_m": _summary(random_3d),
        "nn_cosine_margin_median": float(best_sim.median().item() - random_sim.median().item()),
        "nn_3d_error_ratio_median": float(nn_3d.median().item() / max(float(random_3d.median().item()), 1e-6)),
    }


def main():
    parser = argparse.ArgumentParser(description="Extract ACE-FCN feature-space memory.")
    parser.add_argument("scene", type=Path, help="ACE scene root or ACE train split.")
    parser.add_argument("output", type=Path, help="Output memory_ace_fcn.pt path.")
    parser.add_argument("--ace_encoder_path", type=Path, default=Path("/home/xwh/project/ace_depth/ace_encoder_pretrained.pt"))
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--image_resolution", type=int, default=512)
    parser.add_argument("--samples_per_image", type=int, default=1024, help="Valid stride-8 patches sampled per image; <=0 keeps all.")
    parser.add_argument("--coord_source", type=str, default="depth_gt", choices=["depth_gt", "sparse_depth", "colmap_depth"], help="Coordinate source recorded in metadata.")
    parser.add_argument("--depth_dir", type=Path, default=None, help="Depth directory. Defaults to <scene>/train/depth for depth_gt.")
    parser.add_argument("--depth_min", type=float, default=1e-6, help="Minimum valid metric depth for memory extraction.")
    parser.add_argument("--depth_max", type=float, default=1000.0, help="Maximum valid metric depth for memory extraction.")
    parser.add_argument("--voxel_size", type=float, default=0.05)
    parser.add_argument("--max_points", type=int, default=300000)
    parser.add_argument("--num_head_blocks", type=int, default=4)
    parser.add_argument("--use_homogeneous", type=_strtobool, default=True)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--max_images", type=int, default=0, help="Debug only: stop after this many images; <=0 uses all images.")
    parser.add_argument("--seed", type=int, default=1305)
    parser.add_argument("--sanity_max_queries", type=int, default=4096)
    parser.add_argument("--sanity_chunk_size", type=int, default=256)
    parser.add_argument("--sanity_hard_fail", type=_strtobool, default=True)
    parser.add_argument("--sanity_cosine_margin", type=float, default=0.02)
    parser.add_argument("--sanity_3d_ratio_max", type=float, default=0.8)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    t0 = time.time()
    torch.manual_seed(int(args.seed))
    gen = torch.Generator().manual_seed(int(args.seed))

    train_root = _resolve_train_root(args.scene)
    depth_dir = Path(args.depth_dir) if args.depth_dir is not None else train_root / "depth"
    depth_paths = _build_depth_path_map(depth_dir)
    if not args.ace_encoder_path.exists():
        raise FileNotFoundError(f"ACE encoder checkpoint not found: {args.ace_encoder_path}")

    device = torch.device(args.device)
    dataset = CamLocDataset(
        train_root,
        mode=0,
        sparse=False,
        augment=False,
        image_height=int(args.image_resolution),
        use_half=False,
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=int(args.num_workers))
    regressor = RegressorACE.create_from_encoder(
        encoder_path=args.ace_encoder_path,
        mean=torch.as_tensor(dataset.mean_cam_center).float(),
        num_head_blocks=int(args.num_head_blocks),
        use_homogeneous=bool(args.use_homogeneous),
        num_encoder_features=512,
        freeze_backbone=True,
    ).to(device).eval()

    points_chunks = []
    feature_chunks = []
    image_idx_chunks = []
    pixel_xy_chunks = []
    all_poses = []
    all_intrinsics = []
    total_valid = 0
    total_kept = 0
    skipped = 0
    stride = int(regressor.OUTPUT_SUBSAMPLE)

    with torch.no_grad():
        for image_idx, batch in enumerate(loader):
            if int(args.max_images) > 0 and image_idx >= int(args.max_images):
                break
            image, _, pose, _, intrinsics, _, _, filenames = batch
            image = image.to(device, non_blocking=True).float()
            filename = filenames[0] if isinstance(filenames, (list, tuple)) else filenames
            rgb_stem = Path(str(filename)).stem
            all_poses.append(pose[0].float().cpu())
            all_intrinsics.append(intrinsics[0].float().cpu())
            depth_path = depth_paths.get(rgb_stem)
            if depth_path is None:
                skipped += 1
                continue
            depth = _load_depth(depth_path)
            coords = _depth_to_patch_scene_coords(
                depth,
                pose[0],
                intrinsics[0],
                image_hw=(int(image.shape[-2]), int(image.shape[-1])),
                stride=int(regressor.OUTPUT_SUBSAMPLE),
                depth_min=float(args.depth_min),
                depth_max=float(args.depth_max),
            ).unsqueeze(0)
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                features = regressor.get_features(image).float().cpu()
            coords_for_count = coords
            if coords_for_count.shape[-2:] != features.shape[-2:]:
                coords_for_count = F.interpolate(coords_for_count, size=features.shape[-2:], mode="nearest")
            valid_count = int(_valid_scene_coords(coords_for_count).sum().item())
            total_valid += valid_count
            flat = _flatten_feature_grid(
                features,
                coords,
                image_idx,
                stride,
                int(args.samples_per_image),
                gen,
            )
            if flat is None:
                skipped += 1
                continue
            pts, feats, img_idx, pixel_xy = flat
            total_kept += int(pts.shape[0])
            points_chunks.append(pts)
            feature_chunks.append(feats)
            image_idx_chunks.append(img_idx)
            pixel_xy_chunks.append(pixel_xy)
            if (image_idx + 1) % 50 == 0:
                _logger.info(
                    "Processed %d/%d images, raw_valid=%d, sampled=%d",
                    image_idx + 1,
                    len(dataset),
                    total_valid,
                    total_kept,
                )

    if not points_chunks:
        raise RuntimeError("No valid ACE-FCN memory samples were extracted.")

    raw_points = torch.cat(points_chunks, dim=0)
    raw_features = torch.cat(feature_chunks, dim=0)
    raw_image_indices = torch.cat(image_idx_chunks, dim=0)
    raw_pixel_xy = torch.cat(pixel_xy_chunks, dim=0)

    pooled_points, pooled_features, pooled_image_indices, pooled_pixel_xy, cluster_sizes = _voxel_pool(
        raw_points,
        raw_features,
        raw_image_indices,
        raw_pixel_xy,
        float(args.voxel_size),
        int(args.max_points),
        gen,
    )

    report = _nearest_sanity_report(
        raw_points,
        raw_features,
        pooled_points,
        pooled_features,
        max_queries=int(args.sanity_max_queries),
        chunk_size=int(args.sanity_chunk_size),
        seed=int(args.seed),
    )
    hard_fail_reasons = []
    if report["nn_cosine_margin_median"] <= float(args.sanity_cosine_margin):
        hard_fail_reasons.append(
            f"nn_cosine_margin_median={report['nn_cosine_margin_median']:.4f} <= {args.sanity_cosine_margin:.4f}"
        )
    if report["nn_3d_error_ratio_median"] >= float(args.sanity_3d_ratio_max):
        hard_fail_reasons.append(
            f"nn_3d_error_ratio_median={report['nn_3d_error_ratio_median']:.4f} >= {args.sanity_3d_ratio_max:.4f}"
        )
    report["hard_fail_reasons"] = hard_fail_reasons
    report["hard_pass"] = len(hard_fail_reasons) == 0

    scene_center = torch.as_tensor(dataset.mean_cam_center).float()
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "pooled_points": pooled_points.contiguous(),
        "pooled_features": pooled_features.contiguous(),
        "scene_center": scene_center,
        "points_world": pooled_points.contiguous(),
        "image_indices": pooled_image_indices.contiguous(),
        "pixel_xy": pooled_pixel_xy.contiguous(),
        "cluster_sizes": cluster_sizes.contiguous(),
        "all_poses": torch.stack(all_poses, dim=0) if all_poses else None,
        "all_intrinsics": torch.stack(all_intrinsics, dim=0) if all_intrinsics else None,
        "patch_stride": float(stride),
        "output_subsample": stride,
        "voxel_size": float(args.voxel_size),
        "original_views": int(len(dataset)),
        "scene": _scene_name_from_path(args.scene),
        "feature_dim": 512,
        "feature_source": "ace_fcn",
        "encoder_path": str(args.ace_encoder_path),
        "model_backend": "ace_fcn_lmc",
        "data_backend": "ace",
        "coord_source": str(args.coord_source),
        "depth_dir": str(depth_dir),
        "coord_frame": "world",
        "normalization": "none",
        "depth_min": float(args.depth_min),
        "depth_max": float(args.depth_max),
        "pooling_mode": "voxel_mean" if float(args.voxel_size) > 0 else "none",
        "sampling_mode": "per_image_random_valid_patch",
        "samples_per_image": int(args.samples_per_image),
        "layers_idx": ["ace_fcn"],
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "sanity_report": report,
        "extraction_stats": {
            "num_images": int(len(dataset)),
            "skipped_images": int(skipped),
            "raw_valid_patches": int(total_valid),
            "raw_sampled_patches": int(raw_points.shape[0]),
            "pooled_points": int(pooled_points.shape[0]),
            "feature_norm": _summary(pooled_features.norm(dim=1)),
            "elapsed_seconds": float(time.time() - t0),
        },
    }
    torch.save(payload, output)
    report_path = output.with_suffix(output.suffix + ".sanity.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump({"sanity_report": report, "extraction_stats": payload["extraction_stats"]}, f, indent=2)

    _logger.info("Saved ACE-FCN memory: %s", output)
    _logger.info("Saved sanity report: %s", report_path)
    _logger.info(
        "Sanity: hard_pass=%s cosine_margin=%.4f nn3d/random3d=%.4f pooled=%d raw=%d",
        report["hard_pass"],
        report["nn_cosine_margin_median"],
        report["nn_3d_error_ratio_median"],
        int(pooled_points.shape[0]),
        int(raw_points.shape[0]),
    )
    if hard_fail_reasons and bool(args.sanity_hard_fail):
        raise SystemExit("ACE-FCN memory sanity failed: " + "; ".join(hard_fail_reasons))


if __name__ == "__main__":
    main()
