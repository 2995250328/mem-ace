#!/usr/bin/env python3
"""Visualize ACE RGB against WAI aux depth using pose/calibration alignment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def _load_numpy_array(path: Path) -> np.ndarray:
    path = Path(path)
    if path.suffix.lower() == ".npz":
        return np.load(path, allow_pickle=False)["arr_0"]
    return np.load(path, allow_pickle=False)


def _pose_key(pose: np.ndarray, decimals: int = 5) -> tuple:
    return tuple(np.round(pose.reshape(-1), decimals=decimals).tolist())


def _frame_intrinsics(frame: dict) -> np.ndarray:
    return np.asarray(
        [
            [float(frame["fl_x"]), 0.0, float(frame["cx"])],
            [0.0, float(frame["fl_y"]), float(frame["cy"])],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def _sorted_files(path: Path, suffixes: tuple[str, ...]) -> list[Path]:
    files = [p for p in path.iterdir() if p.is_file() and p.suffix.lower() in suffixes]
    return sorted(files, key=lambda p: p.stem)


def _load_scene_meta_frames(wai_scene_root: Path, depth_kind: str) -> tuple[list[tuple[np.ndarray, dict]], dict[tuple, list[dict]]]:
    meta_path = wai_scene_root / "scene_meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing scene_meta.json: {meta_path}")
    frames = json.load(meta_path.open("r", encoding="utf-8")).get("frames", [])
    meta_poses: list[tuple[np.ndarray, dict]] = []
    pose_to_frames: dict[tuple, list[dict]] = {}
    for frame in frames:
        if "transform_matrix" not in frame or depth_kind not in frame:
            continue
        pose = np.asarray(frame["transform_matrix"], dtype=np.float64)
        if pose.shape != (4, 4):
            continue
        meta_poses.append((pose.reshape(-1), frame))
        pose_to_frames.setdefault(_pose_key(pose), []).append(frame)
    if not meta_poses:
        raise ValueError(f"No usable frames with {depth_kind} in {meta_path}")
    return meta_poses, pose_to_frames


def _match_depth_frame(
    ace_pose: np.ndarray,
    ace_k: np.ndarray,
    meta_poses: list[tuple[np.ndarray, dict]],
    pose_to_frames: dict[tuple, list[dict]],
) -> tuple[dict, bool, float, float]:
    candidates = pose_to_frames.get(_pose_key(ace_pose), [])
    fallback_pose = False
    if candidates:
        chosen = min(candidates, key=lambda fr: float(np.max(np.abs(_frame_intrinsics(fr) - ace_k))))
        pose_diff = 0.0
    else:
        meta_pose_mat = np.stack([p for p, _ in meta_poses], axis=0)
        pose_diffs = np.max(np.abs(meta_pose_mat - ace_pose.reshape(1, -1)), axis=1)
        best_idx = int(np.argmin(pose_diffs))
        pose_diff = float(pose_diffs[best_idx])
        if pose_diff > 1e-4:
            raise ValueError(f"No pose match in scene_meta; best max abs pose diff={pose_diff:.6g}")
        chosen = meta_poses[best_idx][1]
        fallback_pose = True
    k_diff = float(np.max(np.abs(_frame_intrinsics(chosen) - ace_k)))
    if k_diff > 1e-2:
        raise ValueError(f"Matched pose but intrinsics differ too much: max abs K diff={k_diff:.6g}")
    return chosen, fallback_pose, pose_diff, k_diff


def _resize_depth_nearest(depth: np.ndarray, size_wh: tuple[int, int]) -> np.ndarray:
    image = Image.fromarray(depth.astype(np.float32), mode="F")
    image = image.resize(size_wh, Image.Resampling.NEAREST)
    return np.asarray(image, dtype=np.float32)


def _load_resized_valid_depth(depth_path: Path, size_wh: tuple[int, int], depth_min: float, depth_max: float) -> tuple[np.ndarray, np.ndarray]:
    depth = _load_numpy_array(depth_path).astype(np.float32)
    if depth.ndim == 3:
        depth = np.squeeze(depth)
    depth = np.where(np.isfinite(depth), depth, 0.0).astype(np.float32)
    depth_rs = _resize_depth_nearest(depth, size_wh)
    valid = np.isfinite(depth_rs) & (depth_rs > depth_min) & (depth_rs < depth_max)
    return depth, valid


def _sample_patch_depth_nearest_valid(
    depth: np.ndarray,
    stride: int,
    depth_min: float,
    depth_max: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return one representative valid depth pixel per DINO patch."""
    image_h, image_w = depth.shape
    coords_h = int(np.ceil(image_h / stride))
    coords_w = int(np.ceil(image_w / stride))
    patch_depth = np.zeros((coords_h, coords_w), dtype=np.float32)
    patch_px = np.zeros((coords_h, coords_w), dtype=np.float32)
    patch_py = np.zeros((coords_h, coords_w), dtype=np.float32)
    patch_valid = np.zeros((coords_h, coords_w), dtype=bool)

    half = stride // 2
    for gy in range(coords_h):
        cy = min(gy * stride + half, image_h - 1)
        y0 = gy * stride
        y1 = min((gy + 1) * stride, image_h)
        for gx in range(coords_w):
            cx = min(gx * stride + half, image_w - 1)
            x0 = gx * stride
            x1 = min((gx + 1) * stride, image_w)

            center_depth = depth[cy, cx]
            if np.isfinite(center_depth) and depth_min < center_depth < depth_max:
                patch_depth[gy, gx] = center_depth
                patch_px[gy, gx] = cx
                patch_py[gy, gx] = cy
                patch_valid[gy, gx] = True
                continue

            window = depth[y0:y1, x0:x1]
            valid = np.isfinite(window) & (window > depth_min) & (window < depth_max)
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


def _patch_valid_count(depth: np.ndarray, stride: int, depth_min: float, depth_max: float) -> int:
    _, _, _, patch_valid = _sample_patch_depth_nearest_valid(depth, stride, depth_min, depth_max)
    return int(patch_valid.sum())


def _choose_sample_indices(valid: np.ndarray, depth: np.ndarray, max_points: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    ys, xs = np.where(valid)
    if ys.size <= max_points:
        return ys, xs

    rng = np.random.default_rng(seed)
    # Stratify by depth so the overlay exposes near/far alignment instead of only dense regions.
    vals = depth[ys, xs]
    bins = np.quantile(vals, np.linspace(0.0, 1.0, 9))
    chosen = []
    per_bin = max(1, max_points // 8)
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (vals >= lo) & (vals <= hi)
        idx = np.where(mask)[0]
        if idx.size == 0:
            continue
        take = min(per_bin, idx.size)
        chosen.append(rng.choice(idx, size=take, replace=False))
    if chosen:
        chosen_idx = np.unique(np.concatenate(chosen))
    else:
        chosen_idx = np.empty((0,), dtype=np.int64)
    if chosen_idx.size < max_points:
        remaining = np.setdiff1d(np.arange(ys.size), chosen_idx, assume_unique=False)
        take = min(max_points - chosen_idx.size, remaining.size)
        if take > 0:
            chosen_idx = np.concatenate([chosen_idx, rng.choice(remaining, size=take, replace=False)])
    elif chosen_idx.size > max_points:
        chosen_idx = rng.choice(chosen_idx, size=max_points, replace=False)
    return ys[chosen_idx], xs[chosen_idx]


def _choose_patch_points(
    patch_depth: np.ndarray,
    patch_px: np.ndarray,
    patch_py: np.ndarray,
    patch_valid: np.ndarray,
    max_points: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    gy, gx = np.where(patch_valid)
    if gy.size == 0:
        return (
            np.empty((0,), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
        )
    values = patch_depth[gy, gx]
    if gy.size <= max_points:
        return patch_py[gy, gx], patch_px[gy, gx], values

    rng = np.random.default_rng(seed)
    bins = np.quantile(values, np.linspace(0.0, 1.0, 9))
    chosen = []
    per_bin = max(1, max_points // 8)
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (values >= lo) & (values <= hi)
        idx = np.where(mask)[0]
        if idx.size == 0:
            continue
        take = min(per_bin, idx.size)
        chosen.append(rng.choice(idx, size=take, replace=False))
    chosen_idx = np.unique(np.concatenate(chosen)) if chosen else np.empty((0,), dtype=np.int64)
    if chosen_idx.size < max_points:
        remaining = np.setdiff1d(np.arange(gy.size), chosen_idx, assume_unique=False)
        take = min(max_points - chosen_idx.size, remaining.size)
        if take > 0:
            chosen_idx = np.concatenate([chosen_idx, rng.choice(remaining, size=take, replace=False)])
    elif chosen_idx.size > max_points:
        chosen_idx = rng.choice(chosen_idx, size=max_points, replace=False)
    return patch_py[gy[chosen_idx], gx[chosen_idx]], patch_px[gy[chosen_idx], gx[chosen_idx]], values[chosen_idx]


def make_visualization(args: argparse.Namespace) -> Path:
    ace_train_root = args.ace_train_root
    wai_scene_root = args.wai_scene_root
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    rgb_files = _sorted_files(ace_train_root / "rgb", (".jpg", ".jpeg", ".png"))
    pose_files = _sorted_files(ace_train_root / "poses", (".txt", ".pose"))
    calibration_files = _sorted_files(ace_train_root / "calibration", (".txt", ".calibration"))
    if not rgb_files:
        raise FileNotFoundError(f"No ACE RGB files under {ace_train_root / 'rgb'}")
    if not (len(rgb_files) == len(pose_files) == len(calibration_files)):
        raise ValueError(
            f"ACE file count mismatch: rgb={len(rgb_files)} poses={len(pose_files)} calib={len(calibration_files)}"
        )

    meta_poses, pose_to_frames = _load_scene_meta_frames(wai_scene_root, args.depth_kind)

    if args.ace_stem:
        stem_to_idx = {p.stem: i for i, p in enumerate(rgb_files)}
        if args.ace_stem not in stem_to_idx:
            raise ValueError(f"ACE stem {args.ace_stem!r} not found.")
        idx = stem_to_idx[args.ace_stem]
    else:
        idx = int(args.ace_index)
    if idx < 0:
        best = None
        scan_count = min(len(rgb_files), int(args.auto_scan_count))
        for cand_idx in range(scan_count):
            try:
                cand_pose = np.loadtxt(pose_files[cand_idx]).astype(np.float64)
                cand_k = np.loadtxt(calibration_files[cand_idx]).astype(np.float64)
                cand_frame, _, _, _ = _match_depth_frame(cand_pose, cand_k, meta_poses, pose_to_frames)
                cand_depth_path = wai_scene_root / cand_frame[args.depth_kind]
                cand_rgb = Image.open(rgb_files[cand_idx])
                cand_depth, _ = _load_resized_valid_depth(
                    cand_depth_path,
                    cand_rgb.size,
                    args.depth_min,
                    args.depth_max,
                )
                cand_depth_rs = _resize_depth_nearest(cand_depth, cand_rgb.size)
                count = _patch_valid_count(cand_depth_rs, args.patch_stride, args.depth_min, args.depth_max)
            except Exception:
                continue
            if best is None or count > best[0]:
                best = (count, cand_idx)
        if best is None:
            raise ValueError(f"Auto frame selection failed over first {scan_count} frames.")
        idx = int(best[1])
        print(f"[auto] selected ace_index={idx} stem={rgb_files[idx].stem} valid_patch_count={best[0]}")
    if idx < 0 or idx >= len(rgb_files):
        raise IndexError(f"ACE index out of range: {idx}, count={len(rgb_files)}")

    ace_pose = np.loadtxt(pose_files[idx]).astype(np.float64)
    ace_k = np.loadtxt(calibration_files[idx]).astype(np.float64)
    if ace_pose.shape != (4, 4) or ace_k.shape != (3, 3):
        raise ValueError(f"Unexpected ACE pose/K shape: pose={ace_pose.shape}, K={ace_k.shape}")

    frame, fallback_pose, pose_diff, k_diff = _match_depth_frame(ace_pose, ace_k, meta_poses, pose_to_frames)
    depth_path = wai_scene_root / frame[args.depth_kind]
    if not depth_path.exists():
        raise FileNotFoundError(depth_path)

    rgb = Image.open(rgb_files[idx]).convert("RGB")
    rgb_np = np.asarray(rgb)
    depth, valid = _load_resized_valid_depth(depth_path, rgb.size, args.depth_min, args.depth_max)
    depth_rs = _resize_depth_nearest(depth, rgb.size)

    # Match training-buffer sampling: one representative valid depth pixel per DINO patch.
    stride = int(args.patch_stride)
    patch_depth, patch_px, patch_py, patch_valid = _sample_patch_depth_nearest_valid(
        depth_rs,
        stride,
        args.depth_min,
        args.depth_max,
    )
    ys, xs, values = _choose_patch_points(
        patch_depth,
        patch_px,
        patch_py,
        patch_valid,
        args.max_points,
        args.seed,
    )

    if values.size == 0:
        raise ValueError("No valid sparse depth points selected for visualization.")

    stem = f"{ace_train_root.parent.name}_{rgb_files[idx].stem}_to_{Path(frame[args.depth_kind]).stem}"
    rgb_out = out_dir / f"{stem}_rgb.png"
    overlay_out = out_dir / f"{stem}_sparse_depth_overlay.png"
    triptych_out = out_dir / f"{stem}_triptych.png"
    meta_out = out_dir / f"{stem}_meta.json"

    rgb.save(rgb_out)

    vmin = float(np.percentile(values, 2))
    vmax = float(np.percentile(values, 98))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        vmin, vmax = float(values.min()), float(values.max())

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(rgb_np)
    sc = ax.scatter(xs, ys, c=values, s=args.point_size, cmap="turbo", vmin=vmin, vmax=vmax, edgecolors="white", linewidths=0.25)
    ax.set_title(f"{rgb_files[idx].name} -> {Path(frame[args.depth_kind]).name}")
    ax.axis("off")
    cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Depth (m)")
    fig.tight_layout()
    fig.savefig(overlay_out, dpi=args.dpi)
    plt.close(fig)

    depth_vis = np.ma.masked_where(~valid, depth_rs)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    axes[0].imshow(rgb_np)
    axes[0].set_title("ACE RGB")
    axes[0].axis("off")
    axes[1].imshow(rgb_np)
    sc = axes[1].scatter(xs, ys, c=values, s=args.point_size, cmap="turbo", vmin=vmin, vmax=vmax, edgecolors="white", linewidths=0.25)
    axes[1].set_title("Sparse depth on RGB")
    axes[1].axis("off")
    im = axes[2].imshow(depth_vis, cmap="turbo", vmin=vmin, vmax=vmax)
    axes[2].set_title("Resized WAI gt_depth")
    axes[2].axis("off")
    fig.colorbar(sc, ax=axes[1], fraction=0.046, pad=0.04, label="Depth (m)")
    fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04, label="Depth (m)")
    fig.suptitle(
        f"pose_fallback={fallback_pose}, pose_diff={pose_diff:.2e}, K_diff={k_diff:.2e}, "
        f"valid_sparse={values.size}",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(triptych_out, dpi=args.dpi)
    plt.close(fig)

    meta = {
        "ace_rgb": str(rgb_files[idx]),
        "ace_pose": str(pose_files[idx]),
        "ace_calibration": str(calibration_files[idx]),
        "wai_depth": str(depth_path),
        "wai_frame_name": frame.get("frame_name"),
        "depth_kind": args.depth_kind,
        "fallback_pose_match": fallback_pose,
        "pose_diff_max_abs": pose_diff,
        "intrinsics_diff_max_abs": k_diff,
        "rgb_size_wh": list(rgb.size),
        "depth_shape_raw": list(depth.shape),
        "sparse_points": int(values.size),
        "valid_patch_count": int(patch_valid.sum()),
        "depth_min_selected": float(values.min()),
        "depth_median_selected": float(np.median(values)),
        "depth_max_selected": float(values.max()),
        "rgb_output": str(rgb_out),
        "overlay_output": str(overlay_out),
        "triptych_output": str(triptych_out),
    }
    meta_out.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))
    return triptych_out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ace-train-root", type=Path, default=Path("/home/xwh/data/indoor6_ace/scene2a/train"))
    parser.add_argument("--wai-scene-root", type=Path, default=Path("/home/xwh/data/mapanything-dataset/wai_data/indoor6/scene2a_train"))
    parser.add_argument("--depth-kind", default="gt_depth", choices=["gt_depth", "colmap_depth"])
    parser.add_argument("--ace-index", type=int, default=0)
    parser.add_argument("--ace-stem", default="", help="Optional ACE RGB stem such as 003676; overrides --ace-index.")
    parser.add_argument("--auto-scan-count", type=int, default=512, help="When --ace-index is negative, scan this many frames and choose the densest valid-depth patch frame.")
    parser.add_argument("--output-dir", type=Path, default=Path("memory_extraction/04_evaluation/aux_depth_alignment_vis"))
    parser.add_argument("--patch-stride", type=int, default=14)
    parser.add_argument("--max-points", type=int, default=180)
    parser.add_argument("--point-size", type=float, default=34.0)
    parser.add_argument("--depth-min", type=float, default=0.1)
    parser.add_argument("--depth-max", type=float, default=100.0)
    parser.add_argument("--seed", type=int, default=1305)
    parser.add_argument("--dpi", type=int, default=140)
    args = parser.parse_args()
    make_visualization(args)


if __name__ == "__main__":
    main()
