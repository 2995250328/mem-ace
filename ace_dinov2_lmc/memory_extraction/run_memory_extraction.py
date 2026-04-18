"""
BSE-enhanced memory extraction for Map-Anything pipeline.
Complete implementation with HIGH priority fixes:
- Try/finally cleanup for temp files
- Adapter interface for model/dataset decoupling
- GPU memory cleanup between views
- Schema versioning
- Input validation
"""

import argparse
import os
import sys
import json
import time
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Optional, Tuple, Protocol, Any
from dataclasses import dataclass, field

# Add map-anything to path (optional - can use adapter instead)
_WORKSPACE_ROOT = Path(__file__).resolve().parent.parent.parent.parent
MAP_ANYTHING_PATH = _WORKSPACE_ROOT / "map-anything"
if MAP_ANYTHING_PATH.exists():
    sys.path.insert(0, str(MAP_ANYTHING_PATH))
# 与 ~/project/uniception 软链配合：优先用可编辑副本，避免只改 conda 里一份
if (_WORKSPACE_ROOT / "uniception" / "__init__.py").is_file():
    sys.path.insert(0, str(_WORKSPACE_ROOT))

# Local modules
from .bse_pooling import BSEPooler
from .welford_meter import WelfordNormalizer


# =============================================================================
# Visualization — prefer mapanything.tasks.run_memory_extraction (same as fps_memory.sh)
# =============================================================================
# Map-anything uses matplotlib scatter only over *valid* depth pixels; Indoor6 is sparse → fast.
# Dense valid depth (e.g. after interpolation) would make scatter O(N) artists and stall; then we
# fall back to a vectorized colormap (same turbo palette).

_SCATTER_SAFE_MAX_VALID = 80_000
_MA_DEPTH_SCATTER = None
_VIS_USE_MA = False
try:
    import mapanything.tasks.run_memory_extraction as _ma_rme  # type: ignore[import-not-found]

    save_pcd_with_open3d = _ma_rme.save_pcd_with_open3d
    _save_raw_rgb = _ma_rme._save_raw_rgb
    _raw_depth_to_point_cloud_count = _ma_rme._raw_depth_to_point_cloud_count
    _save_model_input_vis = _ma_rme._save_model_input_vis
    _MA_DEPTH_SCATTER = _ma_rme._save_depth_on_rgb_vis
    _VIS_USE_MA = True
except Exception:
    _MA_DEPTH_SCATTER = None
    _VIS_USE_MA = False


def _colors_to_uint8_n3(colors: Any, n: int) -> Optional[np.ndarray]:
    """Return [N,3] uint8 RGB or None if invalid."""
    if colors is None:
        return None
    if isinstance(colors, torch.Tensor):
        c = np.asarray(colors.detach().cpu().numpy())
    else:
        c = np.asarray(colors)
    if c.size == 0 or c.shape[0] != n:
        return None
    if c.ndim == 1:
        return None
    if c.shape[-1] < 3:
        return None
    c = c.reshape(n, -1)[:, :3].astype(np.float64)
    mx = float(np.nanmax(c)) if c.size else 0.0
    if mx <= 1.01:
        c = np.clip(c, 0.0, 1.0) * 255.0
    else:
        c = np.clip(c, 0.0, 255.0)
    return c.astype(np.uint8)


def _write_ply_ascii_xyz_rgb(points_np: np.ndarray, filename: str, colors_u8: Optional[np.ndarray]) -> None:
    """Write ASCII PLY with x,y,z and optional uchar red,green,blue (no Open3D)."""
    pts = np.asarray(points_np, dtype=np.float64).reshape(-1, 3)
    n = pts.shape[0]
    if n == 0:
        return
    cu8 = colors_u8
    if cu8 is not None and (cu8.shape[0] != n or cu8.shape[-1] != 3):
        cu8 = None
    with open(filename, "w", encoding="ascii") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {n}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        if cu8 is not None:
            f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for i in range(n):
            x, y, z = pts[i]
            if cu8 is not None:
                r, g, b = int(cu8[i, 0]), int(cu8[i, 1]), int(cu8[i, 2])
                f.write(f"{x:.6f} {y:.6f} {z:.6f} {r} {g} {b}\n")
            else:
                f.write(f"{x:.6f} {y:.6f} {z:.6f}\n")


def _fallback_save_pcd_with_open3d(points, filename, colors=None):
    if isinstance(points, torch.Tensor):
        points_np = np.asarray(points.detach().cpu().numpy())
    else:
        points_np = np.asarray(points)

    if points_np.size == 0:
        print(f"[Vis] Skipping PLY save (zero points): {filename}", flush=True)
        return

    points_np = points_np.reshape(-1, 3)
    n = points_np.shape[0]
    colors_u8 = _colors_to_uint8_n3(colors, n)

    o3d_ok = False
    try:
        import open3d as o3d  # type: ignore[import-not-found]

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points_np.astype(np.float64))
        if colors_u8 is not None:
            c = (colors_u8.astype(np.float64) / 255.0)
            pcd.colors = o3d.utility.Vector3dVector(np.clip(c, 0.0, 1.0))
        success = o3d.io.write_point_cloud(filename, pcd, write_ascii=False, print_progress=False)
        o3d_ok = bool(success)
        if o3d_ok:
            print(f"[Vis] Point cloud saved (Open3D): {filename}", flush=True)
    except Exception as e:
        print(f"[Vis] Open3D PLY failed ({type(e).__name__}: {e}); trying ASCII PLY...", flush=True)

    if not o3d_ok:
        try:
            _write_ply_ascii_xyz_rgb(points_np, filename, colors_u8)
            if os.path.isfile(filename) and os.path.getsize(filename) > 0:
                suf = " with RGB" if colors_u8 is not None else ""
                print(f"[Vis] Point cloud saved (ASCII PLY{suf}): {filename}", flush=True)
            else:
                print(f"[Vis] Failed to write PLY: {filename}", flush=True)
        except Exception as e:
            print(f"[Vis] ASCII PLY write failed ({type(e).__name__}: {e}): {filename}", flush=True)


def _fallback_save_model_input_vis(img_tensor, save_path, percentile_low=2, percentile_high=98):
    """Save normalized image tensor sent to the model as a viewable PNG."""
    try:
        from PIL import Image as PILImage
        t = img_tensor.detach().cpu().float().numpy()
        if t.ndim == 4:
            t = t[0]
        if t.ndim == 3 and t.shape[0] == 3:
            t = np.transpose(t, (1, 2, 0))
        if t.ndim != 3:
            return
        lo = np.percentile(t, percentile_low)
        hi = np.percentile(t, percentile_high)
        if hi <= lo:
            hi = lo + 1.0
        t = np.clip((t - lo) / (hi - lo), 0, 1) * 255
        PILImage.fromarray(t.astype(np.uint8)).save(save_path)
    except Exception as e:
        print(f"[Vis] Failed to save model input to {save_path}: {e}", flush=True)


def _fallback_save_raw_rgb(rgb_data, save_path):
    """Save dataset 'img' / img_uint8 to PNG (PIL, numpy HWC or CHW, or tensor)."""
    try:
        from PIL import Image as PILImage
        if hasattr(rgb_data, "save"):
            rgb_data.save(save_path)
            return
        t = rgb_data
        if isinstance(t, torch.Tensor):
            t = t.detach().cpu().numpy()
        t = np.asarray(t)
        if t.ndim == 3 and t.shape[0] == 3:
            t = np.transpose(t, (1, 2, 0))
        if t.ndim != 3 or t.shape[2] not in (3, 4):
            return
        if t.dtype != np.uint8:
            # If the tensor looks like ImageNet-normalized (has negatives / mean around 0),
            # unnormalize to [0,1] before saving; otherwise treat as [0,1] or [0,255].
            t_f = t.astype(np.float32)
            if np.isfinite(t_f).any() and (float(np.nanmin(t_f)) < -0.05 or float(np.nanmax(t_f)) > 1.5):
                mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
                std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
                t_rgb = t_f[..., :3] * std[None, None, :] + mean[None, None, :]
                t_rgb = np.clip(t_rgb, 0.0, 1.0)
                if t_f.shape[2] == 4:
                    a = np.clip(t_f[..., 3:4], 0.0, 1.0)
                    t_f = np.concatenate([t_rgb, a], axis=2)
                else:
                    t_f = t_rgb
                t = (t_f * 255.0).astype(np.uint8)
            else:
                if t_f.max() <= 1.01:
                    t = (np.clip(t_f, 0, 1) * 255).astype(np.uint8)
                else:
                    t = np.clip(t_f, 0, 255).astype(np.uint8)
        PILImage.fromarray(t).save(save_path)
    except Exception as e:
        print(f"[Vis] Failed to save raw RGB to {save_path}: {e}", flush=True)


def _fallback_raw_depth_to_point_cloud_count(depth_np, intrinsics, pose_c2w, depth_min=0.1, depth_max=6.0):
    """Back-project valid depth pixels to world; return count and optional points."""
    if isinstance(intrinsics, torch.Tensor):
        intrinsics = intrinsics.cpu().numpy()
    if isinstance(pose_c2w, torch.Tensor):
        pose_c2w = pose_c2w.cpu().numpy()
    intrinsics = np.asarray(intrinsics).reshape(3, 3)
    pose_c2w = np.asarray(pose_c2w).reshape(4, 4)
    H, W = depth_np.shape
    depth_flat = depth_np.ravel()
    valid = (depth_flat > depth_min) & (depth_flat < depth_max) & np.isfinite(depth_flat)
    n_valid = int(np.sum(valid))
    if n_valid == 0:
        return 0, None
    u = np.arange(W, dtype=np.float64)
    v = np.arange(H, dtype=np.float64)
    u, v = np.meshgrid(u, v, indexing="xy")
    u = u.ravel()[valid]
    v = v.ravel()[valid]
    z = depth_flat[valid].astype(np.float64)
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    x_cam = (u - cx) * z / fx
    y_cam = (v - cy) * z / fy
    pts_cam = np.stack([x_cam, y_cam, z], axis=1)
    R = pose_c2w[:3, :3]
    t = pose_c2w[:3, 3]
    pts_world = (pts_cam @ R.T) + t
    return n_valid, pts_world


if not _VIS_USE_MA:
    save_pcd_with_open3d = _fallback_save_pcd_with_open3d
    _save_model_input_vis = _fallback_save_model_input_vis
    _save_raw_rgb = _fallback_save_raw_rgb
    _raw_depth_to_point_cloud_count = _fallback_raw_depth_to_point_cloud_count


def _tensor_or_array_to_hw3_uint8(t: Any, H: int, W: int) -> Optional[np.ndarray]:
    """Convert img tensor/array to uint8 (H,W,3); resize if needed."""
    try:
        from PIL import Image as PILImage
        if isinstance(t, torch.Tensor):
            t = t.detach().cpu().numpy()
        t = np.asarray(t)
        if t.ndim == 4:
            t = t[0]
        if t.ndim == 3 and t.shape[0] == 3:
            t = np.transpose(t, (1, 2, 0))
        if t.ndim != 3 or t.shape[2] != 3:
            return None
        t = t.astype(np.float32)
        if t.size and t.max() > 1.01:
            t = np.clip(t, 0, 255)
        else:
            t = np.clip(t, 0, 1) * 255
        t = t.astype(np.uint8)
        if t.shape[:2] != (H, W):
            t = np.array(PILImage.fromarray(t).resize((W, H), resample=PILImage.BILINEAR))
        return t
    except Exception:
        return None


def _save_turbo_depth_colorbar_png(
    d_min: float, d_max: float, out_path: str, label: str = "depth (m)"
) -> None:
    """Standalone vertical colorbar: turbo colormap ↔ depth value (meters)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib as mpl
    except ImportError:
        return
    fig, ax = plt.subplots(figsize=(1.35, 4.8))
    cmap = plt.get_cmap("turbo")
    norm = mpl.colors.Normalize(vmin=float(d_min), vmax=float(d_max))
    mpl.colorbar.ColorbarBase(ax, cmap=cmap, norm=norm, orientation="vertical")
    ax.set_ylabel(label, fontsize=9)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _save_depth_overlay_with_colorbar_png(
    overlay_rgb_u8: np.ndarray,
    d_min: float,
    d_max: float,
    out_path: str,
    title: str = "depth on RGB (turbo)",
) -> None:
    """Side-by-side: RGB overlay image + turbo colorbar (depth in meters)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib as mpl
    except ImportError:
        return
    if overlay_rgb_u8.ndim != 3 or overlay_rgb_u8.shape[2] != 3:
        return
    H, W = overlay_rgb_u8.shape[:2]
    fig_w = max(9.0, min(16.0, W / 95.0 + 1.4))
    fig_h = max(5.0, min(14.0, H / 95.0))
    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 0.07], wspace=0.08)
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.imshow(overlay_rgb_u8)
    ax0.axis("off")
    ax0.set_title(title, fontsize=10)
    ax1 = fig.add_subplot(gs[0, 1])
    cmap = plt.get_cmap("turbo")
    norm = mpl.colors.Normalize(vmin=float(d_min), vmax=float(d_max))
    cb = mpl.colorbar.ColorbarBase(ax1, cmap=cmap, norm=norm, orientation="vertical")
    cb.set_label("depth (m)", fontsize=9)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def _compute_dynamic_depth_range(
    depth_np: np.ndarray,
    valid: np.ndarray,
    q_low: float = 2.0,
    q_high: float = 98.0,
) -> Tuple[float, float]:
    """Compute per-view depth range using percentiles over valid pixels.

    This makes small depth ranges visually discriminative (avoids a nearly-flat colormap).
    """
    if depth_np.size == 0 or not np.any(valid):
        return 0.0, 1.0
    d = depth_np[valid].astype(np.float64)
    d = d[np.isfinite(d)]
    if d.size == 0:
        return 0.0, 1.0
    lo = float(np.percentile(d, q_low))
    hi = float(np.percentile(d, q_high))
    if not np.isfinite(lo):
        lo = float(np.nanmin(d))
    if not np.isfinite(hi):
        hi = float(np.nanmax(d))
    if hi <= lo:
        hi = lo + 1e-3
    return lo, hi


def _save_depth_on_rgb_vis_dense(
    depth_np,
    save_dir,
    prefix,
    rgb_tensor=None,
    rgb_path=None,
    point_radius=6,
    d_min: Optional[float] = None,
    d_max: Optional[float] = None,
):
    """
    Dense-depth fast path: turbo colormap on full H×W + PIL (same palette as map-anything).
    """
    del point_radius
    try:
        from PIL import Image as PILImage
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[Vis] PIL/matplotlib not installed; skip depth-on-RGB figures", flush=True)
        return

    H, W = depth_np.shape
    valid = (depth_np > 0) & np.isfinite(depth_np)
    gray_bg = np.full((H, W, 3), 240, dtype=np.uint8)

    if not np.any(valid):
        PILImage.fromarray(gray_bg).save(os.path.join(save_dir, f"{prefix}_depth_vis.png"))
        PILImage.fromarray(gray_bg).save(os.path.join(save_dir, f"{prefix}_depth_on_rgb.png"))
        return

    if d_min is None or d_max is None:
        d_min, d_max = _compute_dynamic_depth_range(depth_np, valid, q_low=2.0, q_high=98.0)

    norm = np.zeros((H, W), dtype=np.float32)
    norm[valid] = (depth_np[valid].astype(np.float32) - d_min) / (d_max - d_min)
    norm = np.clip(norm, 0.0, 1.0)

    cmap = plt.get_cmap("turbo")
    rgba = cmap(norm)
    depth_rgb = (np.clip(rgba[:, :, :3], 0.0, 1.0) * 255.0).astype(np.uint8)
    depth_vis = np.where(valid[:, :, None], depth_rgb, gray_bg)
    PILImage.fromarray(depth_vis).save(os.path.join(save_dir, f"{prefix}_depth_vis.png"))

    rgb_bg = gray_bg.copy()
    if rgb_path and os.path.exists(rgb_path):
        try:
            rgb_bg = np.array(PILImage.open(rgb_path).convert("RGB"))
            if rgb_bg.shape[:2] != (H, W):
                rgb_bg = np.array(PILImage.fromarray(rgb_bg).resize((W, H), resample=PILImage.BILINEAR))
        except Exception:
            pass
    elif rgb_tensor is not None:
        arr = _tensor_or_array_to_hw3_uint8(rgb_tensor, H, W)
        if arr is not None:
            rgb_bg = arr

    alpha = 0.45
    comp = rgb_bg.astype(np.float32)
    ov = depth_rgb.astype(np.float32)
    blended = np.where(valid[:, :, None], comp * (1.0 - alpha) + ov * alpha, comp)
    on_rgb = np.clip(blended, 0.0, 255.0).astype(np.uint8)
    out_combined = os.path.join(save_dir, f"{prefix}_depth_on_rgb.png")
    # Combined: overlay + colorbar in one image (no separate bar file).
    _save_depth_overlay_with_colorbar_png(
        on_rgb, d_min, d_max, out_combined,
        title=f"{prefix} depth on RGB (dense turbo)",
    )


def _save_depth_on_rgb_vis_scatter_local(
    depth_np,
    save_dir,
    prefix,
    rgb_tensor=None,
    rgb_path=None,
    point_radius=6,
    d_min: Optional[float] = None,
    d_max: Optional[float] = None,
):
    """
    Local scatter-style visualization for sparse depth (Indoor6).
    Matches map-anything behaviour: draw colored depth points over RGB with a radius,
    so sparse depth does not look like "no overlay".
    """
    try:
        from PIL import Image as PILImage
        from PIL import ImageDraw
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[Vis] PIL/matplotlib not installed; skip depth-on-RGB figures", flush=True)
        return

    H, W = depth_np.shape
    valid = (depth_np > 0) & np.isfinite(depth_np)
    gray_bg = np.full((H, W, 3), 240, dtype=np.uint8)

    # Resolve RGB background
    rgb_bg = gray_bg.copy()
    if rgb_path and os.path.exists(rgb_path):
        try:
            rgb_bg = np.array(PILImage.open(rgb_path).convert("RGB"))
            if rgb_bg.shape[:2] != (H, W):
                rgb_bg = np.array(PILImage.fromarray(rgb_bg).resize((W, H), resample=PILImage.BILINEAR))
        except Exception:
            rgb_bg = gray_bg.copy()
    elif rgb_tensor is not None:
        arr = _tensor_or_array_to_hw3_uint8(rgb_tensor, H, W)
        if arr is not None:
            rgb_bg = arr

    if not np.any(valid):
        PILImage.fromarray(gray_bg).save(os.path.join(save_dir, f"{prefix}_depth_vis.png"))
        PILImage.fromarray(rgb_bg).save(os.path.join(save_dir, f"{prefix}_depth_on_rgb.png"))
        return

    ys, xs = np.nonzero(valid)
    d = depth_np[ys, xs].astype(np.float32)
    if d_min is None or d_max is None:
        d_min, d_max = _compute_dynamic_depth_range(depth_np, valid, q_low=2.0, q_high=98.0)
    t = np.clip((d - d_min) / (d_max - d_min), 0.0, 1.0)

    cmap = plt.get_cmap("turbo")
    colors = (np.clip(cmap(t)[:, :3], 0.0, 1.0) * 255.0).astype(np.uint8)

    # Depth-only visualization: draw points on gray background
    depth_vis_img = PILImage.fromarray(gray_bg)
    draw_depth = ImageDraw.Draw(depth_vis_img)

    # Depth-on-RGB visualization: draw points on RGB
    on_rgb_img = PILImage.fromarray(rgb_bg)
    draw_on = ImageDraw.Draw(on_rgb_img)

    # Make sparse points visually salient:
    # - draw a colored ring (not a 1px dot)
    # - add a high-contrast outline so points remain visible on dark/bright areas
    n_valid = int(len(xs))
    # adaptive radius for sparse depth: keep it visible even when n_valid is tiny
    r = int(max(point_radius, 4 if n_valid < 2_000 else 3))
    r_out = int(max(r + 2, r))
    outline_dark = (0, 0, 0)
    outline_light = (255, 255, 255)

    for (y, x), c in zip(zip(ys.tolist(), xs.tolist()), colors.tolist()):
        col = tuple(int(v) for v in c)
        # Solid dot with high-contrast outline:
        # 1) outer black filled disk (halo)
        x0, y0 = x - r_out, y - r_out
        x1, y1 = x + r_out, y + r_out
        draw_depth.ellipse([x0, y0, x1, y1], outline=None, fill=outline_dark)
        draw_on.ellipse([x0, y0, x1, y1], outline=None, fill=outline_dark)
        # 2) inner white filled disk (thin rim for contrast)
        x0, y0 = x - (r_out - 1), y - (r_out - 1)
        x1, y1 = x + (r_out - 1), y + (r_out - 1)
        draw_depth.ellipse([x0, y0, x1, y1], outline=None, fill=outline_light)
        draw_on.ellipse([x0, y0, x1, y1], outline=None, fill=outline_light)
        # 3) main colored filled disk
        x0, y0 = x - r, y - r
        x1, y1 = x + r, y + r
        draw_depth.ellipse([x0, y0, x1, y1], outline=None, fill=col)
        draw_on.ellipse([x0, y0, x1, y1], outline=None, fill=col)

    out_vis = os.path.join(save_dir, f"{prefix}_depth_vis.png")
    out_combined = os.path.join(save_dir, f"{prefix}_depth_on_rgb.png")
    depth_vis_img.save(out_vis)

    # Combined: overlay + colorbar in one image (no separate bar file).
    _save_depth_overlay_with_colorbar_png(
        np.asarray(on_rgb_img),
        d_min,
        d_max,
        out_combined,
        title=f"{prefix} depth on RGB (sparse scatter)",
    )


def _save_depth_on_rgb_vis(depth_np, save_dir, prefix, rgb_tensor=None, rgb_path=None, point_radius=6):
    """
    Match map-anything/fps_memory: scatter over valid depth pixels when count is modest
    (sparse Indoor6); otherwise dense colormap so the run does not stall.
    """
    valid = (depth_np > 0) & np.isfinite(depth_np)
    n_valid = int(np.sum(valid))
    # Use per-view dynamic depth range for the colorbar to make small depth variations visible.
    d_min, d_max = _compute_dynamic_depth_range(depth_np, valid, q_low=2.0, q_high=98.0)

    # Prefer scatter for sparse depth; dense colormap for very dense valid masks.
    if n_valid <= _SCATTER_SAFE_MAX_VALID:
        _save_depth_on_rgb_vis_scatter_local(
            depth_np, save_dir, prefix,
            rgb_tensor=rgb_tensor, rgb_path=rgb_path, point_radius=point_radius,
            d_min=d_min, d_max=d_max,
        )
        return

    _save_depth_on_rgb_vis_dense(
        depth_np, save_dir, prefix,
        rgb_tensor=rgb_tensor, rgb_path=rgb_path, point_radius=point_radius,
        d_min=d_min, d_max=d_max,
    )


def export_memory_views_visualization(
    raw_batches: List[Dict[str, Any]],
    memory_views: List[Dict[str, torch.Tensor]],
    output_run_dir: str,
    depth_min: float,
    depth_max: float,
    view_records: Optional[List[Dict[str, Any]]] = None,
) -> None:
    """
    After memory views are chosen and loaded: save RGB, depth colormap, RGB+depth overlay,
    raw-depth point counts, optional PLY for view 0, and normalized model-input previews.
    """
    depth_vis_dir = os.path.join(output_run_dir, "depth_validation")
    model_input_dir = os.path.join(output_run_dir, "model_input_vis")
    os.makedirs(depth_vis_dir, exist_ok=True)
    os.makedirs(model_input_dir, exist_ok=True)

    raw_depth_point_counts: List[Tuple[int, int]] = []
    for i, raw_data in enumerate(raw_batches):
        rgb_raw = raw_data.get("img_uint8", raw_data.get("img", raw_data.get("image")))
        if isinstance(rgb_raw, (list, tuple)):
            rgb_raw = rgb_raw[0]
        if rgb_raw is not None:
            _save_raw_rgb(rgb_raw, os.path.join(depth_vis_dir, f"view_{i:02d}_rgb_raw.png"))

        rgb_tensor = raw_data.get("img_uint8")
        if rgb_tensor is None:
            rgb_tensor = raw_data.get("img", raw_data.get("image"))
        if isinstance(rgb_tensor, (list, tuple)):
            rgb_tensor = rgb_tensor[0] if rgb_tensor else None
        if hasattr(rgb_tensor, "convert"):
            rgb_tensor = np.array(rgb_tensor.convert("RGB"))
        rgb_path = raw_data.get("image_path", raw_data.get("rgb_path", None))
        if isinstance(rgb_path, (list, tuple)):
            rgb_path = rgb_path[0] if rgb_path else None

        depth = raw_data.get("depthmap", raw_data.get("depth", raw_data.get("gt_depth", raw_data.get("depth_z"))))
        if depth is None:
            print(f"[Vis] view_{i:02d}: no depth in raw_batch; skip depth figures", flush=True)
            continue
        if isinstance(depth, (list, tuple)):
            depth = depth[0]
        if isinstance(depth, torch.Tensor):
            depth_np = depth.detach().float().cpu()
            depth_np = normalize_depth_to_hw(depth_np).numpy()
        else:
            depth_np = np.asarray(depth, dtype=np.float32).squeeze()
        if depth_np.ndim != 2:
            print(f"[Vis] view_{i:02d}: depth squeeze to 2D failed, shape={depth_np.shape}; skip", flush=True)
            continue

        # Clamp depth used for visualization / raw PLY to the configured valid range.
        depth_vis_np = depth_np.copy()
        invalid_mask = (~np.isfinite(depth_vis_np)) | (depth_vis_np <= depth_min) | (depth_vis_np >= depth_max)
        depth_vis_np[invalid_mask] = 0.0

        intr = raw_data.get("camera_intrinsics", raw_data.get("intrinsics"))
        pose = raw_data.get("camera_pose", raw_data.get("pose"))
        if isinstance(intr, (list, tuple)):
            intr = intr[0]
        if isinstance(pose, (list, tuple)):
            pose = pose[0]
        if intr is not None and pose is not None:
            n_pts, pts_world = _raw_depth_to_point_cloud_count(
                depth_vis_np, intr, pose, depth_min=depth_min, depth_max=depth_max
            )
            raw_depth_point_counts.append((i, n_pts))
            if i == 0 and pts_world is not None and len(pts_world) > 0:
                raw_ply_path = os.path.join(depth_vis_dir, "view_00_raw_depth_pointcloud.ply")
                colors_for_ply = None
                try:
                    # Try to align colors with the same valid-depth mask used in
                    # _raw_depth_to_point_cloud_count (ravel() row-major order).
                    H, W = depth_vis_np.shape
                    rgb_img = None
                    if rgb_tensor is not None:
                        rgb_img = _tensor_or_array_to_hw3_uint8(rgb_tensor, H, W)
                    if rgb_img is None and rgb_raw is not None:
                        # rgb_raw may be PIL/numpy/tensor; best-effort conversion
                        if hasattr(rgb_raw, "convert"):
                            from PIL import Image as PILImage  # local import for safety
                            rgb_img = np.asarray(rgb_raw.convert("RGB"))
                        else:
                            rgb_img = np.asarray(rgb_raw)
                    if rgb_img is not None and rgb_img.ndim == 3 and rgb_img.shape[2] >= 3:
                        rgb_img = rgb_img[:, :, :3]
                        valid_depth = (depth_vis_np > depth_min) & (depth_vis_np < depth_max) & np.isfinite(depth_vis_np)
                        valid_flat = valid_depth.reshape(-1)
                        rgb_flat = rgb_img.reshape(-1, 3)
                        if valid_flat.shape[0] == rgb_flat.shape[0] and valid_flat.sum() > 0:
                            colors_for_ply = rgb_flat[valid_flat]
                except Exception:
                    # 颜色兜底：保持 None，不影响深度/RGB 的导出
                    colors_for_ply = None
                try:
                    save_pcd_with_open3d(
                        torch.from_numpy(pts_world),
                        raw_ply_path,
                        colors=(torch.from_numpy(colors_for_ply) if colors_for_ply is not None else None),
                    )
                    if os.path.isfile(raw_ply_path) and os.path.getsize(raw_ply_path) > 0:
                        print(
                            f"[Vis] View 0: raw depth -> {n_pts} points (range [{depth_min}, {depth_max}] m), saved to {raw_ply_path}",
                            flush=True,
                        )
                    else:
                        print(f"[Vis] View 0: PLY not written (empty/missing): {raw_ply_path}", flush=True)
                except Exception as e:
                    # 点云可视化失败不应影响后续 depth/RGB 图片导出
                    print(f"[Vis] View 0 PLY export failed ({type(e).__name__}: {e}); continue", flush=True)

        _save_depth_on_rgb_vis(depth_vis_np, depth_vis_dir, f"view_{i:02d}", rgb_tensor=rgb_tensor, rgb_path=rgb_path)

    if raw_depth_point_counts:
        raw_counts_only = [c for _, c in raw_depth_point_counts]
        print(f"[Vis] Raw depth point counts per view: {raw_depth_point_counts}", flush=True)
        print(
            "[Vis] Raw depth summary: total=%d, mean=%.1f, median=%.1f, min=%d, max=%d"
            % (
                int(np.sum(raw_counts_only)),
                float(np.mean(raw_counts_only)),
                float(np.median(raw_counts_only)),
                int(np.min(raw_counts_only)),
                int(np.max(raw_counts_only)),
            ),
            flush=True,
        )
    print(f"[Vis] Depth validation: {depth_vis_dir} (rgb_raw, _depth_vis, _depth_on_rgb per view)", flush=True)

    def _annotate_png(path: str, lines: List[str]) -> None:
        try:
            from PIL import Image as PILImage, ImageDraw

            img = PILImage.open(path).convert("RGB")
            draw = ImageDraw.Draw(img)
            text = "\n".join([line for line in lines if line])
            if not text:
                img.save(path)
                return
            bbox = draw.multiline_textbbox((0, 0), text, spacing=2)
            pad = 4
            x0, y0 = 4, 4
            x1 = min(img.width, x0 + (bbox[2] - bbox[0]) + pad * 2)
            y1 = min(img.height, y0 + (bbox[3] - bbox[1]) + pad * 2)
            draw.rectangle([x0, y0, x1, y1], fill=(0, 0, 0))
            draw.multiline_text((x0 + pad, y0 + pad), text, fill=(255, 255, 255), spacing=2)
            img.save(path)
        except Exception as e:
            print(f"[Vis] Failed to annotate model input {path}: {e}", flush=True)

    for i, view in enumerate(memory_views):
        if "img" in view:
            out_path = os.path.join(model_input_dir, f"view_{i:02d}_model_input.png")
            _save_model_input_vis(
                view["img"],
                out_path,
            )
            rec = view_records[i] if view_records is not None and i < len(view_records) else {}
            source = str(rec.get("source", "unknown"))
            actual_idx = rec.get("actual_flat_idx", None)
            fps_idx = rec.get("fps_idx_at_view_position", None)
            _annotate_png(
                out_path,
                [
                    f"view={i:02d}",
                    f"src={os.path.basename(source)}",
                    f"actual_idx={actual_idx} fps_idx={fps_idx}",
                    f"outer={rec.get('outer_index')} inner={rec.get('inner_index')}",
                ],
            )
    print(f"[Vis] Model input images (normalized) saved to: {model_input_dir}", flush=True)


# =============================================================================
# Schema Versioning
# =============================================================================

MEMORY_SCHEMA_VERSION = "1.4"  # Adds pooled_GT legacy aliases plus selection metadata.
CHECKPOINT_FILE = "extraction_checkpoint.json"

# ace_depth/checkpoints/dinov2_vitl14_pretrain.pth → 常指向共享存储的软链接
_ACE_DEPTH_ROOT = Path(__file__).resolve().parents[2]


def resolve_dinov2_checkpoint_path(explicit: Optional[str] = None) -> str:
    """解析 DINOv2 权重路径：环境变量 > explicit > 仓库内软链接 > 共享存储 > 旧路径。"""
    for key in ("DINOV2_CHECKPOINT", "DINOV2_PRETRAINED_PATH"):
        p = os.environ.get(key, "").strip()
        if p and os.path.isfile(p):
            return p
    if explicit:
        p = explicit.strip()
        if p and os.path.isfile(p):
            return p
    repo_link = _ACE_DEPTH_ROOT / "checkpoints" / "dinov2_vitl14_pretrain.pth"
    if repo_link.is_file():
        return str(repo_link)
    candidates = (
        "/mnt/storage/xwh/checkpoints/dinov2_vitl14_pretrain.pth",
        "/Data/xwh/checkpoints/dinov2_vitl14_pretrain.pth",
    )
    for p in candidates:
        if os.path.isfile(p):
            return p
    return str(candidates[0])


_DEFAULT_DINO_CHECKPOINT = resolve_dinov2_checkpoint_path(None)


WAI_VIEW_MODE_ALIASES = {
    "fps_flat": "fps_flat",
    "fps_strict": "fps_flat",
    "strict_fps": "fps_flat",
    "original_multiview": "original_multiview",
    "first_fps_covis": "original_multiview",
    "first_fps_covis40": "original_multiview",
    "mapanything_original": "original_multiview",
    "anchor_support": "anchor_support",
    "asb": "anchor_support",
}


def canonicalize_wai_view_mode(mode: str) -> str:
    """Normalize user-facing WAI selection aliases to the internal protocol name."""
    key = str(mode or "fps_flat").strip().lower()
    try:
        return WAI_VIEW_MODE_ALIASES[key]
    except KeyError as exc:
        valid = ", ".join(sorted(WAI_VIEW_MODE_ALIASES))
        raise ValueError(f"Unsupported wai_view_mode={mode!r}. Valid values/aliases: {valid}") from exc


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class ExtractionConfig:
    """Memory 提取流水线配置（与 parse_args / extract_memory.sh 一一对应）。

    数据流概览：load_dataset → select_memory_views → prepare_batch_input →
    特征提取器(MapAnything/DINOv2) → 反投影+BSE 体素池化 → save_memory。
    """

    # 数据集根目录（WAI 时为含各 scene 子目录的 ROOT；ace loader 时为 ACE 数据根）
    dataset_path: str
    # 输出的 memory 文件路径（通常为 .pt，含特征与元数据）
    output_path: str
    # 参与提取的 memory 视角数量（均匀从训练集中抽样）
    n_memory: int = 100
    # PyTorch 设备，如 cuda:0（配合 shell 中 CUDA_VISIBLE_DEVICES 时，0 表示可见的第一块卡）
    device: str = "cuda:0"
    # BSE 体素边长（米）；越小点越密、显存/时间越大
    voxel_size: float = 0.05
    # 是否使用 BSE（边界感知）池化；当前关闭会 NotImplementedError
    use_bse: bool = True
    # Pooling 模式：bse=voxel hash + Otsu split；simple=简单 voxel mean
    pool_mode: str = "bse"
    # 预池化模式：per_view=每个 view 先池化；global_only=直接缓存 raw points，Pass 2 再全局池化
    prepool_mode: str = "per_view"
    # BSE 内是否用 Otsu 自动估计深度分箱阈值（与稀疏/噪声深度相关）
    use_otsu: bool = True
    # BSE 内 feature similarity 分布近似单峰时跳过 split 的 std 阈值
    unimodal_threshold: float = 0.02
    # Hybrid mode: per-voxel cosine-sim std threshold for selective split
    hybrid_split_min_std: float = 0.05
    # Hybrid mode: minimum points in a voxel to consider splitting
    hybrid_min_cluster_size: int = 4
    # Hybrid mode: minimum points in each Otsu branch to accept split
    hybrid_min_split_points: int = 2
    # Pass 2 完成逐 view 拼接后，是否做跨 view 的全局体素合并
    global_merge: bool = True
    # Pass 2 全局合并体素边长；None 时沿用 voxel_size
    global_merge_voxel_size: Optional[float] = None
    # 反投影与可视化时保留的深度范围 (min_m, max_m)，单位米
    depth_valid_range: Tuple[float, float] = (0.1, 6.0)
    # 网格中心深度采样：与 map-anything generate_patch_point_cloud / fps_memory 语义一致
    # nearest_valid 适合 Indoor6 稀疏深度；nearest 适合稠密 GT；median 为 3×3 邻域
    patch_depth_sampling: str = "nearest_valid"
    # 两遍处理时 chunk/checkpoint 临时目录（建议 /dev/shm 加速 IO）
    temp_dir: str = "/dev/shm"
    # MapAnything Hydra 模型名（如 mapanything_store_intermediates_ace）
    model_str: Optional[str] = None
    # 自定义 YAML 配置路径；None 时用内置 default
    model_config: Optional[str] = None
    # MapAnything 权重 .pth；None 时 main 里可能有默认路径
    model_checkpoint: Optional[str] = None
    # DINOv2 预训练权重路径（dinov2 模式；mapanything 的 encoder 也复用同一解析结果）
    dinov2_checkpoint: str = field(default_factory=lambda: resolve_dinov2_checkpoint_path(None))
    # 是否走 patch 网格提取路径（与双线性采样路径二选一语义）
    use_patch_based: bool = False
    # 是否对反投影点云做统计离群点剔除（SOR）
    enable_sor: bool = False
    # SOR：每个点的 k 近邻数量
    sor_k: int = 20
    # SOR：距离超过 mean + ratio*std 的点剔除
    sor_std_ratio: float = 2.0
    # 数据集类型：auto 则 detect_dataset_type；或 7scenes/indoor6/custom
    dataset_type: str = "auto"
    # WAI 场景名（如 chess_train）；None 时用 dataset_path  basename
    scene_name: Optional[str] = None
    # 加载器：wai=MapAnything WAI 格式；ace=CamLocDatasetDINOv2（ACE 管线）
    dataset_loader: str = "wai"
    # WAI view loading protocol:
    # fps_flat: dataset[idx] returns one view so actual model inputs match the
    #           original FPS-style memory list.
    # original_multiview: dataset[idx] returns n_memory covisibility views for
    #                     the first FPS anchor, matching map-anything fps_memory.sh.
    # anchor_support: reference-aware Anchor+Support selection in a single
    #                 MapAnything forward.
    wai_view_mode: str = "fps_flat"
    # anchor_support scoring knobs.
    anchor_support_alpha: float = 1.0
    anchor_support_eps: float = 1e-6
    anchor_support_tau: float = -1.0
    # anchor_support soft penalty exp(-lambda * dist_to_ref / ref_scale).
    covis_ref_lambda: float = 0.0
    # 0 auto-selects a conservative anchor count.
    covis_anchor_count: int = 0
    covis_support_per_anchor: int = 3
    covis_support_tau: float = 1e-4
    covis_support_min_neighbors: int = 3
    covis_safe_dist_to_ref: float = 3.0
    covis_far_view_budget: int = 8
    covis_far_anchor_budget: int = 2
    covis_coverage_beta: float = 1.0
    covis_candidate_pool_ratio: float = 1.0
    covis_ma_safe_asb: bool = False
    covis_ma_safe_pool_ratio: float = 3.0
    covis_native_group_asb: bool = False
    covis_native_anchor_candidates: int = 64
    covis_adaptive_asb: bool = False
    covis_adaptive_scene_stats: bool = True
    covis_adaptive_view_count: bool = False
    covis_adaptive_min_views: int = 28
    covis_ref_dist_max_limit: float = 4.2
    covis_ref_dist_mean_limit: float = 2.7
    covis_min_coverage_gain: float = 0.01
    covis_gain_patience: int = 2
    covis_target_coverage_mean: float = 0.55
    covis_target_coverage_max: float = 2.0
    # 体素内多条视线方向池化：mean/dominant/first/all
    ray_pool_strategy: str = "mean"
    # 是否保存多种 ray 表示供对比（BSEPooler）
    save_all_ray_strategies: bool = True
    # 特征骨干：mapanything（与 fps_memory 对齐）或 dinov2（DPT 式多尺度）
    use_model: str = "mapanything"
    # DINOv2 取的 block 索引列表；None 则默认 8 层 [2,5,8,11,14,17,20,23]
    dinov2_intermediate_layers: Optional[List[int]] = None
    # MapAnything infer 后：预测位姿相对 GT 的平移阈值（米），用于 OK/HIGH 分级
    pose_eval_translation_ok_m: float = 0.1
    # 为 True 时：任一非参考视角超阈值则进程以退出码 1 结束
    pose_eval_strict: bool = False
    # PoseEval 自检剪枝：第一次 MapAnything infer 后删除最差 view 并重新 infer。
    pose_prune_after_infer: bool = False
    pose_prune_min_views: int = 20
    pose_prune_min_keep_ratio: float = 0.70
    pose_prune_mad_k: float = 2.5
    # Debug：将关键中间数据/统计 dump 到输出目录（用于对齐 ace vs wai）
    debug_dump: bool = False
    # Debug：最多 dump 前 N 个视角（避免目录过大）
    debug_dump_max_views: int = 3
    # Debug：是否额外保存少量数值快照（npz），体积更大
    debug_dump_save_npz: bool = False
    # Diagnostics：保存 BSE 生成阶段的 feature 相似度统计，默认关闭
    feature_diagnostics: bool = False
    # Diagnostics：每个空间 voxel 最多取多少个 raw points 做 pairwise 统计
    feature_diag_max_points_per_voxel: int = 32
    # Diagnostics：最多保存多少条 same-voxel cross-view pair 样本
    feature_diag_max_pair_samples: int = 200000
    # Diagnostics：直方图 bin 数
    feature_diag_bins: int = 100


# =============================================================================
# Adapter Interfaces (HIGH Priority - Decoupling)
# =============================================================================

class FeatureExtractor(Protocol):
    """Protocol for feature extraction models."""

    def extract(
        self,
        images: torch.Tensor,
        depths: torch.Tensor,
        poses: torch.Tensor,
        intrinsics: torch.Tensor
    ) -> Dict[int, List[torch.Tensor]]:
        """
        Extract multi-scale features from input views.

        Args:
            images: [B, 3, H, W] RGB images
            depths: [B, 1, H, W] depth maps
            poses: [B, 4, 4] camera poses
            intrinsics: [B, 3, 3] camera intrinsics

        Returns:
            Dict mapping view_idx to list of multi-scale features
        """
        ...


class MapAnythingExtractor:
    """
    Adapter for Map-Anything model with multi-scale feature extraction.

    Loads the model using Hydra YAML config + OmegaConf, aligned with
    fps_memory.sh's `model=mapanything_store_intermediates_ace` config.
    Falls back to direct checkpoint loading if Hydra config unavailable.
    """

    # Target intermediate layers for multi-scale feature extraction
    # Must match original MapAnything: [0, 6, 12, 18] + final = 5 layers × 768 = 3840D
    TARGET_INTERM_LAYERS = [0, 6, 12, 18]

    def __init__(
        self,
        model_str: str = "mapanything",
        model_config: str = "default",
        checkpoint: Optional[str] = None,
        device: str = "cuda:0",
        dinov2_checkpoint: Optional[str] = None,
    ):
        """
        Initialize Map-Anything model adapter.

        Builds config manually from resolved YAML values, bypassing Hydra runtime.
        Falls back to YAML loading + partial resolution if manual config fails.

        Args:
            model_str: 仅用于兼容；实际注册名必须是 mapanything（见 init_model）。
            model_config: "default" to use built-in config, or path to custom YAML
            checkpoint: Path to pretrained MapAnything checkpoint
            device: Device to run on
            dinov2_checkpoint: DINOv2 预训练权重（encoder）；省略则与 run_memory_extraction 同一解析逻辑
        """
        self.device = device
        try:
            from mapanything.models import init_model  # pyright: ignore[reportMissingImports]

            dino_path = resolve_dinov2_checkpoint_path(dinov2_checkpoint)
            # Build model config manually (bypasses Hydra interpolation issues)
            model_config_dict = self._build_manual_config(dino_path)

            # Try init_model first (wraps model_factory → MapAnything)
            try:
                from omegaconf import OmegaConf  # pyright: ignore[reportMissingImports]
                cfg = OmegaConf.create(model_config_dict)
                # MODEL_CONFIGS 仅包含 "mapanything"，Hydra 名如 mapanything_store_intermediates_ace 会失败
                self.model = init_model("mapanything", cfg)
            except Exception:
                # Fallback: direct MapAnything construction
                from mapanything.models.mapanything import MapAnything  # pyright: ignore[reportMissingImports]
                self.model = MapAnything(**model_config_dict)

            self.model = self.model.to(device).eval()

            if checkpoint:
                state = torch.load(checkpoint, map_location='cpu', weights_only=False)
                self.model.load_state_dict(
                    state.get("model", state), strict=False
                )

            # Force store intermediate features
            # Key: Do NOT set info_sharing_storage_path - let save_filename work directly
            if hasattr(self.model, "model_config"):
                self.model.model_config.store_info_sharing_intermediate_features = True
            if hasattr(self.model, "store_info_sharing_intermediate_features"):
                self.model.store_info_sharing_intermediate_features = True

        except ImportError as e:
            raise ImportError(
                f"Cannot import map-anything. Ensure it's installed or use DINOv2Extractor. Error: {e}"
            )

    @staticmethod
    def _build_manual_config(dinov2_checkpoint_path: str) -> dict:
        """Build complete MapAnything config dict with all Hydra interpolations resolved."""
        return {
            "name": "mapanything",
            "encoder_config": {
                "encoder_str": "dinov2",
                "name": "dinov2_large",
                "data_norm_type": "dinov2",
                "size": "large",
                "with_registers": False,
                "uses_torch_hub": False,
                "pretrained_checkpoint_path": dinov2_checkpoint_path,
                "gradient_checkpointing": False,
            },
            "info_sharing_config": {
                "model_type": "alternating_attention",
                "model_return_type": "intermediate_features",
                "custom_positional_encoding": None,
                "dpt_indices": [11, 17],
                "module_args": {
                    "name": "aat_24_layers_ifr",
                    "indices": list(range(24)),
                    "norm_intermediate": False,
                    "size": "24_layers",
                    "depth": 24,
                    "distinguish_ref_and_non_ref_views": True,
                    "gradient_checkpointing": False,
                },
            },
            "pred_head_config": {
                "type": "dpt+pose",
                "feature_head": {
                    "feature_dim": 256,
                    "hooks": [0, 1, 2, 3],
                    "checkpoint_gradient": False,
                },
                "regressor_head": {
                    "output_dim": 6,
                    "checkpoint_gradient": False,
                },
                "pose_head": {
                    "num_resconv_block": 2,
                    "rot_representation_dim": 4,
                },
                "scale_head": {
                    "output_dim": 1,
                },
                "adaptor_type": "raydirs+depth+pose+confidence+mask",
                "dpt_adaptor": {
                    "name": "raydirs+depth+pose+confidence+mask+scale",
                    "ray_directions_mode": "linear",
                    "ray_directions_normalize_to_unit_sphere": True,
                    "ray_directions_normalize_to_unit_image_plane": False,
                    "ray_directions_vmin": -np.inf,
                    "ray_directions_vmax": np.inf,
                    "ray_directions_clamp_min_of_z_dir": False,
                    "ray_directions_z_dir_min": -np.inf,
                    "depth_mode": "exp",
                    "depth_vmin": 0,
                    "depth_vmax": np.inf,
                    "confidence_type": "exp",
                    "confidence_vmin": 1,
                    "confidence_vmax": np.inf,
                },
                "pose_adaptor": {
                    "name": "raydirs+depth+pose+confidence+mask+scale",
                    "cam_trans_mode": "linear",
                    "cam_trans_vmin": -np.inf,
                    "cam_trans_vmax": np.inf,
                    "quaternions_mode": "linear",
                    "quaternions_normalize": True,
                    "quaternions_vmin": -np.inf,
                    "quaternions_vmax": np.inf,
                },
                "scale_adaptor": {
                    "name": "raydirs+depth+pose+confidence+mask+scale",
                    "mode": "exp",
                    "vmin": 1e-8,
                    "vmax": np.inf,
                },
                "gradient_checkpointing": False,
            },
            "geometric_input_config": {
                "overall_prob": 1,
                "dropout_prob": 0,
                "ray_dirs_prob": 1,
                "depth_prob": 1,
                "cam_prob": 1,
                "sparse_depth_prob": 0,
                "sparsification_removal_percent": 0,
                "depth_scale_norm_all_prob": 0,
                "pose_scale_norm_all_prob": 0,
                "ray_dirs_encoder_config": {
                    "name": "ray_dirs_encoder",
                    "in_chans": 3,
                    "encoder_str": "dense_rep_encoder",
                    "apply_pe": False,
                },
                "depth_encoder_config": {
                    "name": "depth_encoder",
                    "in_chans": 1,
                    "encoder_str": "dense_rep_encoder",
                    "apply_pe": False,
                },
                "cam_rot_encoder_config": {
                    "name": "cam_rot_quats_encoder",
                    "in_chans": 4,
                    "encoder_str": "global_rep_encoder",
                },
                "cam_trans_encoder_config": {
                    "name": "cam_trans_encoder",
                    "in_chans": 3,
                    "encoder_str": "global_rep_encoder",
                },
                "scale_encoder_config": {
                    "name": "scale_encoder",
                    "in_chans": 1,
                    "encoder_str": "global_rep_encoder",
                },
            },
            "pretrained_checkpoint_path": None,
            "load_specific_pretrained_submodules": False,
            "specific_pretrained_submodules": [],
            "torch_hub_force_reload": False,
            "store_info_sharing_intermediate_features": True,
            "info_sharing_storage_device": None,
            "info_sharing_storage_path": None,
            "use_register_tokens_from_encoder": False,
            "info_sharing_mlp_layer_str": "mlp",
        }

    def extract(
        self,
        images: torch.Tensor,
        depths: torch.Tensor,
        poses: torch.Tensor,
        intrinsics: torch.Tensor
    ) -> Dict[int, Dict[str, Any]]:
        """
        Extract features using Map-Anything model.
        Note: This method signature is kept for compatibility but internally
        we need to pass the views directly to model.infer().
        """
        raise NotImplementedError(
            "MapAnythingExtractor should not use this extract() method. "
            "Use extract_from_views() instead with prepared view dicts."
        )

    def _process_saved_features(self, saved_data: Dict, n_views: int) -> Dict[int, Dict[str, Any]]:
        """Process stored features from model memory"""
        result = {}
        raw_interm = saved_data.get("intermediate", [])
        raw_final = saved_data.get("final")

        for i in range(n_views):
            feat_list = []

            # Extract from intermediate layers
            if raw_interm:
                for layer_idx in self.TARGET_INTERM_LAYERS:
                    if layer_idx < len(raw_interm):
                        interm_output = raw_interm[layer_idx]
                        # Handle MultiViewTransformerOutput object
                        if hasattr(interm_output, 'features'):
                            feat = interm_output.features[i]
                        elif isinstance(interm_output, dict) and "features" in interm_output:
                            feat = interm_output["features"][i]
                        else:
                            continue
                        if isinstance(feat, torch.Tensor):
                            feat_list.append(feat.detach().cpu())

            # Extract from final layer
            if raw_final:
                if hasattr(raw_final, 'features'):
                    feat = raw_final.features[i]
                elif isinstance(raw_final, dict) and "features" in raw_final:
                    feat = raw_final["features"][i]
                else:
                    feat = None
                if feat is not None and isinstance(feat, torch.Tensor):
                    feat_list.append(feat.detach().cpu())

            grid_H, grid_W = None, None
            if len(feat_list) > 0:
                first_feat = feat_list[0]
                if first_feat.ndim == 3:
                    B, L, C = first_feat.shape
                    S = int(L ** 0.5)
                    if S * S == L:
                        grid_H, grid_W = S, S
                elif first_feat.ndim == 4:
                    B, C, H, W = first_feat.shape
                    grid_H, grid_W = H, W

            result[i] = {
                'features': feat_list,
                'grid_H': grid_H,
                'grid_W': grid_W,
                'cls_token': None,  # Populated below
            }

            # CLS / scale / register tokens from raw_final (align with map-anything ace/utils.py):
            # - additional_token_features: fused scale token, often batch dim 1 for the whole MV batch
            # - additional_token_features_per_view: list of per-view tensors (preferred when present)
            if raw_final:
                tpv = None
                if isinstance(raw_final, dict):
                    tpv = raw_final.get("additional_token_features_per_view")
                if tpv is None and hasattr(raw_final, "additional_token_features_per_view"):
                    tpv = raw_final.additional_token_features_per_view
                if tpv is not None and isinstance(tpv, (list, tuple)) and i < len(tpv):
                    tv = tpv[i]
                    if isinstance(tv, torch.Tensor):
                        result[i]["cls_token"] = tv.detach().cpu().view(-1)
                else:
                    token_feat = None
                    if isinstance(raw_final, dict) and "additional_token_features" in raw_final:
                        token_feat = raw_final["additional_token_features"]
                    elif hasattr(raw_final, "additional_token_features"):
                        token_feat = raw_final.additional_token_features
                    if token_feat is not None and isinstance(token_feat, torch.Tensor):
                        tb = token_feat.shape[0]
                        if tb >= n_views:
                            result[i]["cls_token"] = token_feat[i].detach().cpu().view(-1)
                        elif tb == 1:
                            # Fused scale token: same for all views
                            result[i]["cls_token"] = token_feat[0].detach().cpu().view(-1)
                        else:
                            idx = min(i, tb - 1)
                            result[i]["cls_token"] = token_feat[idx].detach().cpu().view(-1)

        return result


class DINOv2Extractor:
    """
    DINOv2-based feature extractor with DPT-style multi-scale intermediate features.

    Extracts features from intermediate transformer blocks, mimicking DPT's approach:
    - Early blocks: low-level texture/edge features
    - Mid blocks: mid-level semantic features
    - Late blocks: high-level semantic features

    This enables degradation experiments where specific layers can be selected or
    dropped during fusion, exactly like MapAnything's intermediate feature selection.
    """

    # Default DPT-style layer selection for ViT-L/14 (24 blocks)
    # Mirrors the DPT head's hook pattern: evenly spaced across depth
    DEFAULT_INTERMEDIATE_LAYERS = [2, 5, 8, 11, 14, 17, 20, 23]

    def __init__(
        self,
        checkpoint_path: str,
        device: str = "cuda:0",
        intermediate_layers: Optional[List[int]] = None,
    ):
        """
        Initialize DINOv2 extractor.

        Args:
            checkpoint_path: Path to DINOv2 weights
            device: Device to run on
            intermediate_layers: Block indices to extract features from.
                                 None = use DEFAULT_INTERMEDIATE_LAYERS (8 layers, DPT-style).
                                 Use [23] for single-scale (final block only).
        """
        self.device = device
        self.intermediate_layers = (
            intermediate_layers if intermediate_layers is not None
            else self.DEFAULT_INTERMEDIATE_LAYERS
        )
        self.model = self._load_dinov2(checkpoint_path)

    def _load_dinov2(self, checkpoint_path: str) -> torch.nn.Module:
        """Load DINOv2 model, preferring local checkpoint to avoid hub compatibility issues."""
        if os.path.exists(checkpoint_path):
            print(f"[DINOv2] Loading from local checkpoint: {checkpoint_path}")
            # Build model from cached hub code but skip downloading weights
            try:
                model = torch.hub.load(
                    'facebookresearch/dinov2', 'dinov2_vitl14',
                    pretrained=False, verbose=False
                )
            except (TypeError, SyntaxError, Exception) as e:
                print(f"[DINOv2] Hub load failed ({e}), building DINOv2 from timm")
                import timm
                model = timm.create_model('vit_large_patch14_dinov2', pretrained=False, dynamic_img_size=True)
            state = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            # Handle different checkpoint formats
            if isinstance(state, dict) and 'state_dict' in state:
                state = state['state_dict']
            # Convert any half-precision weights to float32 for consistency
            state = {k: v.float() if v.is_floating_point() else v for k, v in state.items()}
            model.load_state_dict(state, strict=False)
        else:
            try:
                model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')
            except (TypeError, SyntaxError, Exception) as e:
                raise RuntimeError(
                    f"DINOv2 hub load failed ({e}) and no local checkpoint at {checkpoint_path}. "
                    "Please provide a valid --dinov2_checkpoint path."
                )

        model = model.to(self.device)
        model.eval()
        return model

    def _extract_multiscale(
        self, images: torch.Tensor
    ) -> Dict[int, Dict[str, Any]]:
        """
        Extract multi-scale features by hooking into intermediate transformer blocks.

        Returns features from each specified block, similar to DPT's intermediate
        feature extraction. Each layer's output is separately spatialized.

        Args:
            images: [B, 3, H, W] input images

        Returns:
            Dict mapping view_idx to {
                'features': list of [C_i, H_i, W_i] tensors (one per layer),
                'grid_H': int, 'grid_W': int,
                'layer_indices': list[int] — which blocks were extracted
            }
        """
        B = len(images)
        n_blocks = len(self.model.blocks)
        layer_outputs = {idx: [] for idx in self.intermediate_layers}

        hooks = []
        captured = {}

        def make_hook(block_idx):
            def hook_fn(module, input, output):
                captured[block_idx] = output.detach()
            return hook_fn

        # Register forward hooks on target blocks
        for idx in self.intermediate_layers:
            if idx < n_blocks:
                hooks.append(
                    self.model.blocks[idx].register_forward_hook(make_hook(idx))
                )

        try:
            with torch.no_grad():
                # Run full forward — hooks capture intermediate outputs
                out = self.model.forward_features(images)

                # Handle both dict (facebookresearch/dinov2 hub) and tensor (timm) returns
                n_registers = getattr(self.model, 'num_register_tokens', 0)
                n_prefix = 1 + n_registers  # CLS + registers

                if isinstance(out, dict):
                    # facebookresearch/dinov2 hub format
                    x_norm_patchtokens = out["x_norm_patchtokens"]  # [B, N, C]
                    x_norm_clstoken = out.get("x_norm_clstoken")    # [B, C] or None
                else:
                    # timm format: [B, N_total, C] where N_total = 1(CLS) + N_patches
                    x_norm_clstoken = out[:, 0]                     # [B, C]
                    x_norm_patchtokens = out[:, 1:]                  # [B, N, C]

                # Collect hooked intermediate features
                # Strip CLS token (and register tokens) from intermediate outputs
                n_registers = getattr(self.model, 'num_register_tokens', 0)
                n_prefix = 1 + n_registers  # CLS + registers
                for idx in self.intermediate_layers:
                    if idx in captured:
                        raw = captured[idx]
                        # raw shape: [B, N_total, C] where N_total = 1(CLS) + N_registers + N_patches
                        if raw.ndim == 3 and raw.shape[1] > n_prefix:
                            layer_outputs[idx] = raw[:, n_prefix:]  # [B, N_patches, C]
                        else:
                            layer_outputs[idx] = raw

                # Always add final layer (x_norm_patchtokens) as a separate feature
                # This matches MapAnything's behavior: intermediate layers + final
                layer_outputs['final'] = x_norm_patchtokens

            # Build result per view
            features_dict = {}
            for i in range(B):
                feat_list = []
                for idx in self.intermediate_layers:
                    if idx in layer_outputs and len(layer_outputs[idx]) > 0:
                        feat = layer_outputs[idx][i]  # [N, C] or [C, H, W]
                        if isinstance(feat, torch.Tensor):
                            feat_list.append(feat.cpu())

                # Infer grid dimensions from first feature
                grid_H, grid_W = None, None
                if feat_list:
                    first = feat_list[0]
                    if first.ndim == 2:  # [N, C] → infer spatial dims
                        N, C = first.shape
                        S = int(N ** 0.5)
                        if S * S == N:
                            grid_H, grid_W = S, S
                    elif first.ndim == 3:  # [C, H, W]
                        grid_H, grid_W = first.shape[1], first.shape[2]

                features_dict[i] = {
                    'features': feat_list,
                    'grid_H': grid_H,
                    'grid_W': grid_W,
                    'layer_indices': list(self.intermediate_layers),
                    'cls_token': x_norm_clstoken[i].cpu() if x_norm_clstoken is not None else None,
                }

            return features_dict

        finally:
            # Always remove hooks
            for h in hooks:
                h.remove()

    def extract(
        self,
        images: torch.Tensor,
        depths: torch.Tensor,
        poses: torch.Tensor,
        intrinsics: torch.Tensor
    ) -> Dict[int, Dict[str, Any]]:
        """
        Extract multi-scale DPT-style features using DINOv2.

        Returns:
            Dict mapping view_idx to {
                'features': list of [C, H, W] tensors from intermediate blocks,
                'grid_H': int, 'grid_W': int,
                'layer_indices': list[int]
            }
        """
        return self._extract_multiscale(images)


# =============================================================================
# Input Validation (MEDIUM Priority)
# =============================================================================

def validate_inputs(
    points: torch.Tensor,
    features: torch.Tensor,
    colors: torch.Tensor,
    ray_dirs: torch.Tensor
) -> None:
    """
    Validate input tensors for BSE pooling.

    Raises:
        ValueError: If inputs are invalid
    """
    if points.dim() != 2 or points.shape[1] != 3:
        raise ValueError(f"points must be [N, 3], got {points.shape}")

    if features.dim() != 2:
        raise ValueError(f"features must be [N, C], got {features.shape}")

    if colors.dim() != 2 or colors.shape[1] != 3:
        raise ValueError(f"colors must be [N, 3], got {colors.shape}")

    if ray_dirs.dim() != 2 or ray_dirs.shape[1] != 3:
        raise ValueError(f"ray_dirs must be [N, 3], got {ray_dirs.shape}")

    N = points.shape[0]
    if not (len(features) == len(colors) == len(ray_dirs) == N):
        raise ValueError(
            f"Length mismatch: points={N}, features={len(features)}, "
            f"colors={len(colors)}, ray_dirs={len(ray_dirs)}"
        )

    # Check for NaN/Inf
    for name, tensor in [("points", points), ("features", features),
                          ("colors", colors), ("ray_dirs", ray_dirs)]:
        if torch.isnan(tensor).any():
            raise ValueError(f"{name} contains NaN values")
        if torch.isinf(tensor).any():
            raise ValueError(f"{name} contains Inf values")


# =============================================================================
# Core Helper Functions
# =============================================================================

def prepare_batch_input(
    raw_data: Dict[str, torch.Tensor],
    device: torch.device,
    mode: str = "memory"
) -> Dict[str, torch.Tensor]:
    """
    Convert dataset raw data to model input format (MapAnything compatible).
    Based on mapanything/tasks/run_memory_extraction.py
    """
    view_input = {}

    # Image
    img_raw = raw_data.get("img", raw_data.get("image"))
    if isinstance(img_raw, (list, tuple)):
        img_raw = img_raw[0]

    if isinstance(img_raw, np.ndarray):
        img = torch.from_numpy(img_raw)
    else:
        img = img_raw

    # Force float32 BEFORE device transfer (ACE path may have float16)
    if isinstance(img, torch.Tensor) and img.dtype != torch.float32:
        img = img.float()

    img = img.to(device)

    if img.ndim == 3:
        img = img.unsqueeze(0)
    B = img.shape[0]

    view_input["img"] = img
    view_input["data_norm_type"] = ["dinov2"] * B

    # Intrinsics
    intr = raw_data.get("camera_intrinsics", raw_data.get("intrinsics"))
    if isinstance(intr, (list, tuple)):
        intr = intr[0]
    if intr is not None:
        if isinstance(intr, np.ndarray):
            intr = torch.from_numpy(intr)
        intr = intr.to(device)
        if intr.ndim == 2:
            intr = intr.unsqueeze(0)
        view_input["intrinsics"] = intr

    if mode == "memory":
        # Pose
        pose = raw_data.get("camera_pose", raw_data.get("pose"))
        if isinstance(pose, (list, tuple)):
            pose = pose[0]
        if pose is not None:
            if isinstance(pose, np.ndarray):
                pose = torch.from_numpy(pose)
            pose = pose.to(device)
            if pose.ndim == 2:
                pose = pose.unsqueeze(0)
            view_input["camera_poses"] = pose

        # Depth
        depth = raw_data.get("depthmap", raw_data.get("depth", raw_data.get("depth_z")))
        if isinstance(depth, (list, tuple)):
            depth = depth[0]

        if depth is not None:
            if isinstance(depth, np.ndarray):
                depth = torch.from_numpy(depth)
            depth = depth.to(device).float()
            # Ensure [B, H, W, 1] format
            if depth.ndim == 2:
                depth = depth.unsqueeze(0).unsqueeze(-1)  # [H,W] -> [1,H,W,1]
            elif depth.ndim == 3:
                # [H,W,1] is channel-last; [1,H,W] or [B,H,W] is batch-first — do not confuse them.
                if depth.shape[-1] == 1:
                    depth = depth.squeeze(-1).unsqueeze(0).unsqueeze(-1)  # [H,W,1] -> [1,H,W,1]
                else:
                    depth = depth.unsqueeze(-1)  # [B,H,W] -> [B,H,W,1]
            elif depth.ndim == 4:
                # Already 4D, ensure last dim is 1
                if depth.shape[-1] != 1:
                    # [B,H,W,C] -> squeeze C, then unsqueeze to get [B,H,W,1]
                    depth = depth.squeeze(-1).unsqueeze(-1)
            view_input["depth_z"] = depth

        view_input["is_metric_scale"] = torch.ones((B,), dtype=torch.bool, device=device)

    return view_input


def print_pose_prediction_vs_gt(
    memory_gt_poses: List[torch.Tensor],
    predictions: Any,
    translation_ok_m: float = 0.1,
    return_errors: bool = False,
) -> Any:
    """
    After MapAnything infer(), compare predicted camera_poses to dataset GT.
    Same SE(3) alignment as mapanything/tasks/run_memory_extraction.py (Prediction Evaluation):
    align predicted reference to GT reference, then measure per-view translation / rotation error.

    Returns True if every non-reference view has translation error below translation_ok_m.
    """
    has_preds = (
        isinstance(predictions, list)
        and len(predictions) > 0
        and isinstance(predictions[0], dict)
        and "camera_poses" in predictions[0]
    )
    if not has_preds:
        print("[PoseEval] Skipped: infer() output has no per-view 'camera_poses'.")
        return (True, [], []) if return_errors else True
    if len(memory_gt_poses) == 0:
        print("[PoseEval] Skipped: no GT poses in prepared views (camera_poses missing in batch).")
        return (True, [], []) if return_errors else True

    n = min(len(memory_gt_poses), len(predictions))
    if n < len(memory_gt_poses) or n < len(predictions):
        print(
            f"[PoseEval] Warning: truncating to n={n} "
            f"(GT poses={len(memory_gt_poses)}, predictions={len(predictions)})."
        )

    print("\n" + "=" * 60)
    print(f"{'PoseEval':^60}")
    print(f"{'View':<5} | {'Trans Err (m)':<15} | {'Rot Err (deg)':<15} | Status")
    print("-" * 60)

    t_errs: List[float] = []
    r_errs: List[float] = []

    T_gt_ref = memory_gt_poses[0].double().cpu()
    if T_gt_ref.ndim == 3:
        T_gt_ref = T_gt_ref.squeeze(0)

    T_pred_ref = predictions[0]["camera_poses"].double().cpu()
    if T_pred_ref.ndim == 3:
        T_pred_ref = T_pred_ref.squeeze(0)
    T_pred_ref_inv = torch.linalg.inv(T_pred_ref)

    all_trans_ok = True
    for i in range(n):
        pred_pose = predictions[i]["camera_poses"].double().cpu()
        if pred_pose.ndim == 3:
            pred_pose = pred_pose.squeeze(0)

        T_rel_pred = torch.matmul(T_pred_ref_inv, pred_pose)
        T_pred_world = torch.matmul(T_gt_ref, T_rel_pred)

        T_gt_i = memory_gt_poses[i].double().cpu()
        if T_gt_i.ndim == 3:
            T_gt_i = T_gt_i.squeeze(0)

        t_diff = T_pred_world[:3, 3] - T_gt_i[:3, 3]
        t_err = float(torch.norm(t_diff).item())
        t_errs.append(t_err)

        R_pred = T_pred_world[:3, :3]
        R_gt = T_gt_i[:3, :3]
        matmul_res = torch.matmul(R_pred, R_gt.transpose(0, 1))
        trace = matmul_res.trace()
        cos_theta = torch.clamp((trace - 1) / 2, -1.0, 1.0)
        angle_rad = torch.acos(cos_theta)
        r_err = float(torch.rad2deg(angle_rad).item())
        r_errs.append(r_err)

        if i == 0:
            status = "REF"
        else:
            status = "OK" if t_err < translation_ok_m else "HIGH"
            if t_err >= translation_ok_m:
                all_trans_ok = False
        print(f"{i:<5} | {t_err:.4f}          | {r_err:.4f}          | {status}")

    print("-" * 60)
    print(f"Mean  | {float(np.mean(t_errs)):.4f} m        | {float(np.mean(r_errs)):.4f} deg")

    # Scale-aligned diagnostics (same as map-anything)
    t_gt_norms: List[float] = []
    t_pred_norms: List[float] = []
    for i in range(n):
        T_gt_i = memory_gt_poses[i].double().cpu()
        if T_gt_i.ndim == 3:
            T_gt_i = T_gt_i.squeeze(0)
        pred_pose = predictions[i]["camera_poses"].double().cpu()
        if pred_pose.ndim == 3:
            pred_pose = pred_pose.squeeze(0)
        T_rel_pred = torch.matmul(T_pred_ref_inv, pred_pose)
        T_pred_world = torch.matmul(T_gt_ref, T_rel_pred)
        t_gt_norms.append(float(torch.norm(T_gt_i[:3, 3]).item()))
        t_pred_norms.append(float(torch.norm(T_pred_world[:3, 3]).item()))
    t_gt_arr = np.array(t_gt_norms, dtype=np.float64)
    t_pred_arr = np.array(t_pred_norms, dtype=np.float64)
    valid = t_pred_arr > 1e-6
    if np.sum(valid) > 0:
        scale = float(np.median(t_gt_arr[valid] / t_pred_arr[valid]))
        t_errs_scaled: List[float] = []
        r_errs_scaled: List[float] = []
        for i in range(n):
            pred_pose = predictions[i]["camera_poses"].double().cpu()
            if pred_pose.ndim == 3:
                pred_pose = pred_pose.squeeze(0)
            T_rel_pred = torch.matmul(T_pred_ref_inv, pred_pose)
            T_pred_world = torch.matmul(T_gt_ref, T_rel_pred).clone()
            T_pred_world[:3, 3] *= scale
            T_gt_i = memory_gt_poses[i].double().cpu()
            if T_gt_i.ndim == 3:
                T_gt_i = T_gt_i.squeeze(0)
            t_errs_scaled.append(float(torch.norm(T_pred_world[:3, 3] - T_gt_i[:3, 3]).item()))
            R_pred = T_pred_world[:3, :3]
            R_gt = T_gt_i[:3, :3]
            matmul_res = torch.matmul(R_pred, R_gt.transpose(0, 1))
            cos_theta = torch.clamp((matmul_res.trace() - 1) / 2, -1.0, 1.0)
            r_errs_scaled.append(float(torch.rad2deg(torch.acos(cos_theta)).item()))
        print(
            f"[PoseEval][Scale-aligned] scale={scale:.4f} -> "
            f"Mean Trans Err={float(np.mean(t_errs_scaled)):.4f} m, "
            f"Rot Err={float(np.mean(r_errs_scaled)):.4f} deg"
        )

    if all_trans_ok:
        print("[PoseEval] All non-reference views: translation error < "
              f"{translation_ok_m} m (rotation still reported above).")
    else:
        print(
            "[PoseEval] Some views exceed translation threshold "
            f"({translation_ok_m} m); check calibration / depth / model weights."
        )
    print("=" * 60 + "\n")
    if return_errors:
        return all_trans_ok, t_errs, r_errs
    return all_trans_ok


def _pose_prune_keep_indices(
    trans_errors: List[float],
    translation_ok_m: float,
    min_views: int,
    min_keep_ratio: float,
    mad_k: float,
    max_views: Optional[int] = None,
) -> Tuple[List[int], Dict[str, Any]]:
    """Keep reference plus the lower-error non-reference views."""
    n = len(trans_errors)
    if n <= 1:
        return list(range(n)), {"reason": "not_enough_views"}

    errs = np.asarray(trans_errors, dtype=np.float64)
    nonref = np.arange(1, n, dtype=np.int64)
    nonref_errs = errs[nonref]
    finite = np.isfinite(nonref_errs)
    if not finite.any():
        return list(range(n)), {"reason": "no_finite_errors"}

    valid_errs = nonref_errs[finite]
    median = float(np.nanmedian(valid_errs))
    mad = float(np.nanmedian(np.abs(valid_errs - median)))
    robust_sigma = 1.4826 * mad
    robust_threshold = median + max(float(mad_k), 0.0) * robust_sigma
    if not np.isfinite(robust_threshold) or robust_threshold <= 0:
        robust_threshold = float(translation_ok_m)

    min_keep = max(1, int(np.ceil(float(n) * max(float(min_keep_ratio), 0.0))), int(min_views))
    min_keep = min(min_keep, n)
    target_keep = min_keep
    if max_views is not None and int(max_views) > 0:
        target_keep = max(min_keep, min(int(max_views), n))
    drop_budget = max(0, n - target_keep)

    drop_candidates = [int(i) for i in nonref if np.isfinite(errs[int(i)])]
    drop_candidates.sort(key=lambda i: float(errs[i]), reverse=True)
    drop = set(drop_candidates[:drop_budget])
    keep = [i for i in range(n) if i not in drop]
    stats = {
        "n": int(n),
        "min_keep": int(min_keep),
        "target_keep": int(target_keep),
        "max_views": int(max_views) if max_views is not None and int(max_views) > 0 else None,
        "drop_budget": int(drop_budget),
        "median": median,
        "mad": mad,
        "robust_threshold": float(robust_threshold),
        "translation_ok_m": float(translation_ok_m),
        "dropped": sorted(drop),
        "dropped_errors": {int(i): float(errs[int(i)]) for i in sorted(drop)},
    }
    return keep, stats


def process_multiscale_features_to_grid(
    feat_list: List[torch.Tensor],
    apply_l2_norm: bool = False,
    patch_size: int = 14,
    img_H: Optional[int] = None,
    img_W: Optional[int] = None,
) -> Tuple[torch.Tensor, int, int]:
    """
    Process multi-scale features (DPT-style): align to max grid resolution and concatenate.

    This is similar to mapanything's process_multiscale_features_to_grid.

    Args:
        feat_list: List of feature tensors from different layers
                  Each can be (B, L, C) or (B, C, H, W)
        apply_l2_norm: If True, apply L2 normalization before concatenation

    Returns:
        aligned_features: (N, C_total) where N = grid_H * grid_W
        grid_H, grid_W: Grid dimensions
    """
    import math

    aligned_list = []
    max_H, max_W = 0, 0
    reshaped_feats = []

    # Step 1: Reshape all features to spatial format and find max resolution
    for feat in feat_list:
        if feat.ndim == 2:
            # (N, C) -> (1, C, H, W) — single-view features from DINOv2 hooks
            N, C = feat.shape
            # Strip CLS token if present (try N-1 as perfect square/rect)
            _strip_cls_single = False
            S = int(math.sqrt(N))
            if S * S != N:
                # Try stripping 1 CLS token
                N_minus = N - 1
                S_m = int(math.sqrt(N_minus))
                if S_m * S_m == N_minus:
                    N = N_minus
                    _strip_cls_single = True
                    S = S_m
            if S * S == N:
                H_p, W_p = S, S
            else:
                # Try aspect ratio (non-square grid, e.g. 37×49)
                W_p = int(math.sqrt(N))
                H_p = N // W_p
                if H_p * W_p != N:
                    # Try stripping CLS for rect
                    N_minus = N - 1 if not _strip_cls_single else N
                    W_p = int(math.sqrt(N_minus))
                    H_p = N_minus // W_p
                    if H_p * W_p == N_minus:
                        N = N_minus
                        _strip_cls_single = True
                    else:
                        H_p, W_p = S, S  # fallback
            if _strip_cls_single:
                feat = feat[1:]  # strip CLS
            feat = feat.reshape(1, C, H_p, W_p)

        elif feat.ndim == 3:
            # (B, L, C) -> (B, C, H, W)
            B, L, C = feat.shape
            # Strip CLS token if present
            _strip_cls_batch = False
            S = int(math.sqrt(L))
            if S * S != L:
                L_minus = L - 1
                S_m = int(math.sqrt(L_minus))
                if S_m * S_m == L_minus:
                    L = L_minus
                    _strip_cls_batch = True
                    S = S_m
            if S * S == L:
                H_p, W_p = S, S
            else:
                # Try aspect ratio (non-square grid)
                W_p = int(math.sqrt(L))
                H_p = L // W_p
                if H_p * W_p != L:
                    L_minus = L - 1 if not _strip_cls_batch else L
                    W_p = int(math.sqrt(L_minus))
                    H_p = L_minus // W_p
                    if H_p * W_p == L_minus:
                        L = L_minus
                        _strip_cls_batch = True
                    else:
                        H_p, W_p = S, S  # fallback
            feat = feat.transpose(1, 2).reshape(B, C, H_p, W_p)

        if feat.ndim == 4:
            B, C, H_p, W_p = feat.shape
            reshaped_feats.append((feat, H_p, W_p))
            max_H = max(max_H, H_p)
            max_W = max(max_W, W_p)

    # Step 2: Align all features to max resolution using bilinear interpolation
    for feat, H_p, W_p in reshaped_feats:
        if (H_p, W_p) != (max_H, max_W):
            feat = F.interpolate(feat, size=(max_H, max_W), mode='bilinear', align_corners=False)

        # Apply L2 normalization if requested
        if apply_l2_norm:
            feat = F.normalize(feat, p=2, dim=1)

        # Flatten: (B, C, H, W) -> (B*H*W, C)
        feat_flat = feat.permute(0, 2, 3, 1).reshape(-1, feat.shape[1])
        aligned_list.append(feat_flat)

    # Step 3: Concatenate along channel dimension
    aligned_features = torch.cat(aligned_list, dim=1)

    return aligned_features, max_H, max_W


def _infer_feature_hw(num_tokens: int, target_H: int, target_W: int) -> Tuple[int, int, bool]:
    """Infer feature map size, optionally stripping one CLS token."""
    import math

    if num_tokens <= 0:
        raise ValueError(f"Cannot infer feature grid from num_tokens={num_tokens}")

    ratio = float(target_H) / max(float(target_W), 1.0)

    def _try(n: int) -> Optional[Tuple[int, int]]:
        w = max(1, int(math.sqrt(n / ratio)))
        h = n // w
        if h * w == n:
            return h, w
        s = int(math.sqrt(n))
        if s * s == n:
            return s, s
        return None

    direct = _try(num_tokens)
    if direct is not None:
        return direct[0], direct[1], False

    stripped = _try(num_tokens - 1)
    if stripped is not None:
        return stripped[0], stripped[1], True

    s = int(math.sqrt(num_tokens))
    return s, max(1, num_tokens // max(s, 1)), False


def process_multiscale_features_to_image(
    feat_list: List[torch.Tensor],
    target_H: int,
    target_W: int,
) -> Tuple[torch.Tensor, int, int]:
    """
    Original pooled baseline: align every feature level to image resolution.

    This mirrors mapanything/tasks/run_memory_extraction.py
    ``process_multiscale_features`` used by fps_memory.sh when
    USE_PATCH_BASED_EXTRACTION=False.
    """
    aligned_list = []
    for feat in feat_list:
        if feat.ndim == 2:
            n_tokens, channels = feat.shape
            H_p, W_p, strip_cls = _infer_feature_hw(n_tokens, target_H, target_W)
            if strip_cls:
                feat = feat[1:]
            feat = feat.reshape(1, H_p, W_p, channels).permute(0, 3, 1, 2)

        elif feat.ndim == 3:
            B, n_tokens, channels = feat.shape
            H_p, W_p, strip_cls = _infer_feature_hw(n_tokens, target_H, target_W)
            if strip_cls:
                feat = feat[:, 1:]
            feat = feat.transpose(1, 2).reshape(B, channels, H_p, W_p)

        if feat.ndim != 4:
            raise ValueError(f"Unsupported feature tensor shape for image alignment: {tuple(feat.shape)}")

        if feat.shape[-2:] != (target_H, target_W):
            feat = F.interpolate(feat, size=(target_H, target_W), mode='bilinear', align_corners=False)

        feat_flat = feat.permute(0, 2, 3, 1).reshape(-1, feat.shape[1])
        aligned_list.append(feat_flat)

    return torch.cat(aligned_list, dim=1), target_H, target_W


def process_multiscale_features(
    feat_list: List[torch.Tensor],
    target_size: Optional[Tuple[int, int]] = None
) -> torch.Tensor:
    """
    Legacy: Align multi-scale features to unified grid and concatenate.

    (Kept for backward compatibility, use process_multiscale_features_to_grid instead)

    Args:
        feat_list: List of tensors [B, C_i, H_i, W_i] from different layers
        target_size: Target (H, W) size, uses largest if None

    Returns:
        Concatenated tensor [B, C_total, H_max, W_max]
    """
    aligned_features, grid_H, grid_W = process_multiscale_features_to_grid(feat_list, apply_l2_norm=False)
    B = feat_list[0].shape[0] if feat_list[0].ndim == 4 else 1
    C_total = aligned_features.shape[1]
    return aligned_features.reshape(B, C_total, grid_H, grid_W)


def normalize_depth_to_hw(depth: torch.Tensor) -> torch.Tensor:
    """
    Collapse common depth layouts to 2D [H, W] for pixel sampling.

    Indoor6 / WAI often store depth as [H, W, 1] (channel-last). A naive
    ``depth[0]`` (treating dim0 as batch) wrongly yields [W, 1] or [518, 1],
    which makes all grid samples hit an invalid strip and Welford sees no points.
    Mirrors the 3D handling in mapanything/tasks/run_memory_extraction.py.
    """
    d = depth
    if d.ndim == 2:
        return d
    if d.ndim == 4:
        if d.shape[1] == 1:
            return d[0, 0]
        if d.shape[-1] == 1:
            return d[0, :, :, 0]
        d = d[0]
        if d.ndim == 3 and d.shape[-1] == 1:
            return d[:, :, 0]
        raise ValueError(f"Unsupported 4D depth after batch strip: {tuple(depth.shape)}")
    if d.ndim == 3:
        if d.shape[-1] == 1:
            return d.squeeze(-1)
        if d.shape[0] == 1:
            return d[0]
        return d[0]
    raise ValueError(f"Unsupported depth tensor ndim={d.ndim}, shape={tuple(depth.shape)}")


def mapanything_pooled_color_image(
    view_data: Dict[str, Any],
    image_H: int,
    image_W: int,
    device: torch.device,
) -> Optional[torch.Tensor]:
    """
    Return the color tensor used by the original fps_memory.py pooled path.

    The original pooled script did not use raw RGB for `pooled_colors`; it took
    the exact model input image, resized if needed, then min-max normalized it
    per view before flattening/sampling. Keep that behavior for `pool_mode=pooled`
    so the payload is comparable to `*_pooled_GT.pt`.
    """
    img = view_data.get("img") if isinstance(view_data, dict) else None
    if not isinstance(img, torch.Tensor):
        return None

    img = img.detach().to(device).float()
    if img.ndim == 3:
        img = img.unsqueeze(0)
    if img.ndim != 4 or img.shape[1] != 3:
        return None

    if img.shape[-2:] != (image_H, image_W):
        img = F.interpolate(img, size=(image_H, image_W), mode="bilinear", align_corners=False)

    img_min = img.min()
    img_max = img.max()
    if img_max > img_min:
        img = (img - img_min) / (img_max - img_min)
    else:
        img = torch.zeros_like(img)
    return img.clamp(0.0, 1.0)


def unproject_with_pixel_features(
    view_data: Dict[str, torch.Tensor],
    features_flat: torch.Tensor,
    image_H: int,
    image_W: int,
    depth_valid_range: Tuple[float, float] = (0.1, 6.0),
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Original pooled baseline: use all valid depth pixels and image-resolution features.

    Point generation mirrors mapanything/tasks/run_memory_extraction.py
    ``generate_gt_point_cloud`` for USE_PATCH_BASED_EXTRACTION=False.
    """
    depth_min, depth_max = float(depth_valid_range[0]), float(depth_valid_range[1])

    if "scene_coords" in view_data:
        coords = view_data["scene_coords"].float()
        if coords.ndim == 3:
            coords = coords.unsqueeze(0)
        if coords.shape[-2:] != (image_H, image_W):
            coords = F.interpolate(coords, size=(image_H, image_W), mode='nearest')
        points_world = coords[0].permute(1, 2, 0).reshape(-1, 3)
        valid_mask = torch.ones(points_world.shape[0], dtype=torch.bool, device=points_world.device)
    else:
        depth = view_data.get('depth', view_data.get('depthmap', view_data.get('depth_z', view_data.get('gt_depth'))))
        if depth is None:
            raise KeyError("No depth found in view_data (tried 'depth', 'depthmap', 'depth_z', 'gt_depth')")
        depth = normalize_depth_to_hw(depth).float()
        if depth.shape != (image_H, image_W):
            depth = F.interpolate(
                depth.view(1, 1, *depth.shape),
                size=(image_H, image_W),
                mode='nearest',
            )[0, 0]

        pose = view_data.get('camera_pose', view_data.get('pose'))
        if pose is None:
            raise KeyError("No pose found in view_data (tried 'camera_pose', 'pose')")
        if pose.ndim == 3:
            pose = pose[0]

        K = view_data.get('camera_intrinsics', view_data.get('intrinsics'))
        if K is None:
            raise KeyError("No intrinsics found in view_data (tried 'camera_intrinsics' and 'intrinsics')")
        if K.ndim == 3:
            K = K[0]

        v, u = torch.meshgrid(
            torch.arange(image_H, device=depth.device),
            torch.arange(image_W, device=depth.device),
            indexing='ij',
        )
        u = u.flatten().float()
        v = v.flatten().float()
        z = depth.flatten()
        valid_mask = (z > depth_min) & (z < depth_max)

        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        x_cam = (u - cx) * z / fx
        y_cam = (v - cy) * z / fy
        pts_cam = torch.stack([x_cam, y_cam, z], dim=-1)

        R = pose[:3, :3]
        t = pose[:3, 3]
        points_world = pts_cam @ R.T + t.unsqueeze(0)

    valid_mask_feat = valid_mask.to(features_flat.device)
    points_world = points_world[valid_mask]
    features_filtered = features_flat[valid_mask_feat].to(points_world.device)

    pose = view_data.get('camera_pose', view_data.get('pose'))
    if pose is not None and pose.ndim == 3:
        pose = pose[0]
    if pose is not None:
        camera_center = pose[:3, 3].to(points_world.device)
    else:
        camera_center = points_world.mean(dim=0) if len(points_world) else torch.zeros(3, device=points_world.device)

    if len(points_world) == 0:
        return (
            torch.zeros(0, 3, device=features_filtered.device),
            torch.zeros(0, 3, device=features_filtered.device),
            torch.zeros(0, 3, device=features_filtered.device),
            features_filtered,
            torch.zeros(0, 3, device=features_filtered.device),
        )

    ray_dirs = F.normalize(points_world - camera_center.unsqueeze(0), dim=-1, eps=1e-6)
    camera_centers = camera_center.unsqueeze(0).expand(len(points_world), -1)

    if 'images' in view_data or 'img' in view_data or 'image' in view_data:
        rgb = view_data.get('images', view_data.get('img', view_data.get('image')))
        if isinstance(rgb, torch.Tensor):
            if rgb.ndim == 4:
                rgb = rgb[0]
            if rgb.ndim == 3 and rgb.shape[-1] == 3:
                rgb = rgb.permute(2, 0, 1)
            if rgb.ndim == 3 and rgb.shape[0] == 3:
                rgb = rgb.to(points_world.device).float()
                if rgb.shape[-2:] != (image_H, image_W):
                    rgb = F.interpolate(rgb.unsqueeze(0), size=(image_H, image_W), mode='bilinear', align_corners=False)[0]
                colors_all = rgb.permute(1, 2, 0).reshape(-1, 3)
                colors = colors_all[valid_mask.to(colors_all.device)]
            else:
                colors = torch.ones(len(points_world), 3, device=points_world.device) * 0.5
        else:
            colors = torch.ones(len(points_world), 3, device=points_world.device) * 0.5
    else:
        colors = torch.ones(len(points_world), 3, device=points_world.device) * 0.5

    return points_world, ray_dirs, colors, features_filtered, camera_centers


def unproject_with_grid_features(
    view_data: Dict[str, torch.Tensor],
    features_flat: torch.Tensor,
    grid_H: int,
    grid_W: int,
    depth_valid_range: Tuple[float, float] = (0.1, 6.0),
    sampling_method: str = 'nearest_valid'
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Unproject grid-resolution features to 3D using real depth.

    Grid layout and depth sampling match ``mapanything.tasks.run_memory_extraction.generate_patch_point_cloud``
    (fps_memory.sh ``patch_depth_sampling``): ``nearest`` | ``median`` | ``nearest_valid``.

    Feature flatten order from ``process_multiscale_features_to_grid`` is row-major over (grid_H, grid_W)
    with ``indexing='ij'`` on linspace v then u — same as map-anything patch path.
    """
    depth_min, depth_max = float(depth_valid_range[0]), float(depth_valid_range[1])

    # Get depth from batch_gpu (try multiple key names)
    depth = view_data.get('depth', view_data.get('depthmap', view_data.get('depth_z', view_data.get('gt_depth'))))
    if depth is None:
        raise KeyError("No depth found in view_data (tried 'depth', 'depthmap', 'depth_z', 'gt_depth')")

    depth = normalize_depth_to_hw(depth)

    pose = view_data.get('camera_pose', view_data.get('pose'))
    if pose is None:
        raise KeyError("No pose found in view_data (tried 'camera_pose', 'pose')")
    if pose.ndim == 3:
        pose = pose[0]  # [B, 4, 4] -> [4, 4]

    K = view_data.get('camera_intrinsics', view_data.get('intrinsics'))
    if K is None:
        raise KeyError("No intrinsics found in view_data (tried 'camera_intrinsics' and 'intrinsics')")
    if K.ndim == 3:
        K = K[0]  # [B, 3, 3] -> [3, 3]

    img_H, img_W = depth.shape
    depth_2d = depth

    # Extract intrinsics
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    # Extract rotation and translation. WAI/MapAnything poses are c2w, so
    # translation is already the camera center in world coordinates.
    R = pose[:3, :3]  # [3, 3]
    t = pose[:3, 3]   # [3]
    camera_center = t  # [3]

    # Grid centers — same as generate_patch_point_cloud (indexing='ij' → [grid_H, grid_W])
    grid_centers_v = torch.linspace(0, img_H - 1, grid_H, device=depth.device)
    grid_centers_u = torch.linspace(0, img_W - 1, grid_W, device=depth.device)
    grid_v, grid_u = torch.meshgrid(grid_centers_v, grid_centers_u, indexing='ij')
    grid_y_int = torch.clamp(grid_v.long(), 0, img_H - 1)
    grid_x_int = torch.clamp(grid_u.long(), 0, img_W - 1)

    if sampling_method == 'median':
        patch_size = 3
        half_patch = patch_size // 2
        depth_patches = []
        for dy in range(-half_patch, half_patch + 1):
            for dx in range(-half_patch, half_patch + 1):
                y_idx = torch.clamp(grid_y_int + dy, 0, img_H - 1)
                x_idx = torch.clamp(grid_x_int + dx, 0, img_W - 1)
                depth_patches.append(depth_2d[y_idx, x_idx])
        depth_patches_t = torch.stack(depth_patches, dim=0)
        z_sampled = torch.median(depth_patches_t, dim=0)[0]
    elif sampling_method == 'nearest_valid':
        radius = 10
        z_list = []
        for i in range(grid_H):
            for j in range(grid_W):
                gy, gx = int(grid_y_int[i, j].item()), int(grid_x_int[i, j].item())
                y0 = max(0, gy - radius)
                y1 = min(img_H, gy + radius + 1)
                x0 = max(0, gx - radius)
                x1 = min(img_W, gx + radius + 1)
                patch = depth_2d[y0:y1, x0:x1]
                vm = (patch > depth_min) & (patch < depth_max) & torch.isfinite(patch)
                if vm.any():
                    z_list.append(patch[vm].median().reshape(1))
                else:
                    z_list.append(torch.zeros(1, device=depth.device, dtype=depth.dtype))
        z_sampled = torch.cat(z_list, dim=0).view(grid_H, grid_W)
    else:
        z_sampled = depth_2d[grid_y_int, grid_x_int]

    depth_sampled = z_sampled.reshape(-1)
    grid_u = grid_u.reshape(-1)
    grid_v = grid_v.reshape(-1)

    valid_mask = (depth_sampled > depth_min) & (depth_sampled < depth_max)

    # features_flat may live on CPU (e.g. MapAnything stored intermediates) while depth/grid are on GPU
    valid_mask_feat = valid_mask.to(features_flat.device)

    # Only keep valid points
    u_valid = grid_u[valid_mask]
    v_valid = grid_v[valid_mask]
    z_valid = depth_sampled[valid_mask]

    if len(z_valid) == 0:
        # Return empty tensors with correct shapes
        return (
            torch.zeros(0, 3, device=depth.device),
            torch.zeros(0, 3, device=depth.device),
            torch.zeros(0, 3, device=depth.device),
            torch.zeros(0, features_flat.shape[1], device=depth.device),
            torch.zeros(0, 3, device=depth.device)
        )

    # Back-project to camera coordinates
    x_cam = (u_valid - cx) * z_valid / fx
    y_cam = (v_valid - cy) * z_valid / fy
    z_cam = z_valid

    pts_cam = torch.stack([x_cam, y_cam, z_cam], dim=-1)  # [N, 3]

    # Transform to world coordinates
    pts_world = pts_cam @ R.T + t.unsqueeze(0)  # [N, 3]

    # Compute ray directions
    ray_dirs = pts_world - camera_center.unsqueeze(0)
    ray_dirs = F.normalize(ray_dirs, dim=-1, eps=1e-6)

    # Extract colors at grid centers
    # Prefer `images` (same contract as map-anything). If absent, try common keys.
    if 'images' in view_data or 'img' in view_data or 'image' in view_data:
        rgb = None
        if 'images' in view_data:
            rgb = view_data['images'][0]  # [3, H, W]
        elif 'img' in view_data:
            rgb = view_data['img']
            if rgb.ndim == 4:
                rgb = rgb[0]
        elif 'image' in view_data:
            rgb = view_data['image']
            if rgb.ndim == 4:
                rgb = rgb[0]

        # Ensure [3,H,W] and match depth resolution
        if rgb is not None and isinstance(rgb, torch.Tensor):
            if rgb.ndim == 3 and rgb.shape[0] == 3:
                pass
            elif rgb.ndim == 3 and rgb.shape[-1] == 3:
                rgb = rgb.permute(2, 0, 1)
            else:
                rgb = None
        if rgb is not None and isinstance(rgb, torch.Tensor) and rgb.shape[-2:] != (img_H, img_W):
            rgb = F.interpolate(rgb.unsqueeze(0), size=(img_H, img_W), mode='bilinear', align_corners=False)[0]
        # Sample colors using bilinear interpolation
        # Normalize u, v to [-1, 1] for grid_sample
        u_norm = 2.0 * grid_u[valid_mask] / (img_W - 1) - 1.0
        v_norm = 2.0 * grid_v[valid_mask] / (img_H - 1) - 1.0
        grid_coords = torch.stack([u_norm, v_norm], dim=-1).unsqueeze(0).unsqueeze(0)  # [1, 1, N, 2]
        if rgb is None:
            colors = torch.ones(len(pts_world), 3, device=depth.device) * 0.5
        else:
            rgb_dev = rgb.unsqueeze(0).to(depth.device)
            grid_coords = grid_coords.to(depth.device)

            rgb_sampled = F.grid_sample(
                rgb_dev, grid_coords,
                mode='bilinear', align_corners=False, padding_mode='border'
            )  # [1, 3, 1, N]

            colors = rgb_sampled[0, :, 0, :].T  # [N, 3]
    else:
        colors = torch.ones(len(pts_world), 3, device=depth.device) * 0.5

    # Filter features to match valid points; align device with geometry for BSE pooler
    features_filtered = features_flat[valid_mask_feat].to(depth.device)

    # Camera centers for each point (all same for a single view)
    camera_centers = camera_center.unsqueeze(0).expand(len(pts_world), -1)  # [N, 3]

    return pts_world, ray_dirs, colors, features_filtered, camera_centers


def unproject_with_real_depth(
    view_data: Dict[str, torch.Tensor],
    features: torch.Tensor,
    depth_valid_range: Tuple[float, float] = (0.1, 6.0)
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Unproject features to 3D using real depth.

    Args:
        view_data: Dict with 'depths', 'camera_poses', 'camera_intrinsics', 'images'
        features: [1, C, H, W] feature tensor
        depth_valid_range: (min, max) valid depth in meters

    Returns:
        points: [N, 3] world coordinates
        ray_dirs: [N, 3] unit ray directions
        colors: [N, 3] RGB colors
        features_flat: [N, C] features
        camera_centers: [N, 3] camera center for each point (for Plücker encoding)
    """
    depth = view_data['depths'][0, 0]  # [H, W]
    pose = view_data['camera_poses'][0]  # [4, 4]
    K = view_data.get('camera_intrinsics', view_data.get('intrinsics'))[0]  # [3, 3]

    H, W = depth.shape

    # Resize features to match depth resolution
    if features.shape[-2:] != (H, W):
        features = F.interpolate(
            features, size=(H, W), mode='bilinear', align_corners=False
        )

    # Create pixel grid
    u, v = torch.meshgrid(
        torch.arange(W, device=depth.device),
        torch.arange(H, device=depth.device),
        indexing='xy'
    )
    u = u.float()
    v = v.float()

    # Filter valid depth
    valid_mask = (depth > depth_valid_range[0]) & (depth < depth_valid_range[1])

    # Optional: morphological dilation for depth boundaries
    # This helps with depth discontinuities at object edges
    # valid_mask = morphological_dilation(valid_mask.unsqueeze(0), kernel_size=3).squeeze(0)

    u_valid = u[valid_mask]
    v_valid = v[valid_mask]
    Z_valid = depth[valid_mask]

    if len(Z_valid) == 0:
        # No valid depth, return empty tensors
        device = depth.device
        C = features.shape[1]
        return (
            torch.zeros(0, 3, device=device),
            torch.zeros(0, 3, device=device),
            torch.zeros(0, 3, device=device),
            torch.zeros(0, C, device=device),
            torch.zeros(0, 3, device=device)
        )

    # Unproject to camera coordinates
    # P_cam = Z * K^-1 * [u, v, 1]^T
    K_inv = torch.inverse(K)
    uv1 = torch.stack([u_valid, v_valid, torch.ones_like(u_valid)], dim=0)  # [3, N]
    P_cam = K_inv @ uv1  # [3, N]
    P_cam = P_cam * Z_valid.unsqueeze(0)  # [3, N]

    # Transform to world coordinates
    # P_world = R * P_cam + t
    R = pose[:3, :3]
    t = pose[:3, 3]
    P_world = (R @ P_cam).T + t  # [N, 3]

    # Compute ray directions
    camera_center = t  # [3]
    ray_dirs = P_world - camera_center.unsqueeze(0)  # [N, 3]
    ray_dirs = F.normalize(ray_dirs, dim=1, eps=1e-6)

    # Camera centers for each point (all same for a single view)
    # This is needed for Plücker ray encoding
    camera_centers = camera_center.unsqueeze(0).expand(len(P_world), -1)  # [N, 3]

    # Extract features at valid pixels
    features_flat = features[0, :, v_valid.long(), u_valid.long()].T  # [N, C]

    # Extract colors (if available)
    if 'images' in view_data:
        rgb = view_data['images'][0]  # [3, H, W]
        colors = rgb[:, v_valid.long(), u_valid.long()].T  # [N, 3]
    else:
        colors = torch.ones(len(P_world), 3, device=P_world.device) * 0.5

    return P_world, ray_dirs, colors, features_flat, camera_centers


def sor_filter(
    points: torch.Tensor,
    k: int = 20,
    std_ratio: float = 2.0
) -> torch.Tensor:
    """
    Statistical Outlier Removal filter.

    Args:
        points: [N, 3] point cloud
        k: Number of neighbors
        std_ratio: Standard deviation multiplier

    Returns:
        Boolean mask of inliers
    """
    if len(points) < k + 1:
        return torch.ones(len(points), dtype=torch.bool, device=points.device)

    # Compute pairwise distances
    dists = torch.cdist(points, points)

    # Get k-nearest distances (excluding self)
    knn_dists = dists.topk(k + 1, largest=False).values[:, 1:]  # Exclude self
    mean_knn_dists = knn_dists.mean(dim=1)

    # Compute global statistics
    global_mean = mean_knn_dists.mean()
    global_std = mean_knn_dists.std()

    # Filter outliers
    threshold = global_mean + std_ratio * global_std
    return mean_knn_dists < threshold


def sor_filter_mapanything_compatible(
    points: torch.Tensor,
    nb_neighbors: int,
    std_ratio: float,
) -> torch.Tensor:
    """SOR mask matching map-anything's Open3D path, with torch fallback."""
    nb_neighbors = max(1, int(nb_neighbors))
    std_ratio = float(std_ratio)
    if len(points) < nb_neighbors + 1:
        return torch.ones(len(points), dtype=torch.bool, device=points.device)

    try:
        import open3d as o3d  # type: ignore[import-not-found]

        pts_cpu = points.detach().float().cpu()
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts_cpu.numpy())
        _, ind = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
        mask_np = np.zeros(len(pts_cpu), dtype=bool)
        mask_np[ind] = True
        return torch.from_numpy(mask_np).to(points.device)
    except Exception as e:
        print(
            f"[SOR] Open3D SOR unavailable ({type(e).__name__}: {e}).",
            flush=True,
        )

        # The torch fallback is O(N^2) in memory because it forms a full cdist
        # matrix. That is fine for small patch clouds, but pixel-level Indoor6
        # views can have 100k+ valid points and would request hundreds of GB.
        max_torch_points = 12000
        if int(points.shape[0]) > max_torch_points:
            print(
                f"[SOR] Skipping torch fallback for {int(points.shape[0])} points "
                f"(>{max_torch_points}) to avoid O(N^2) memory use.",
                flush=True,
            )
            return torch.ones(len(points), dtype=torch.bool, device=points.device)

        print("[SOR] Using torch fallback for small point cloud.", flush=True)
        return sor_filter(points, k=nb_neighbors, std_ratio=std_ratio)


def mapanything_sor_params(num_points: int) -> Tuple[int, float]:
    """Original fps_memory.sh SOR schedule used before voxel pooling."""
    if num_points < 5000:
        return min(20, max(5, num_points // 100)), 2.5
    return 50, 1.5


# =============================================================================
# Two-Pass Processing with HIGH Priority Fixes
# =============================================================================

def save_checkpoint(
    checkpoint_path: str,
    completed_views: List[int],
    chunk_paths: List[str]
) -> None:
    """Save extraction checkpoint for resume capability."""
    with open(checkpoint_path, 'w') as f:
        json.dump({
            'completed': completed_views,
            'paths': chunk_paths
        }, f)


def load_checkpoint(checkpoint_path: str) -> Tuple[List[int], List[str]]:
    """Load extraction checkpoint."""
    if not os.path.exists(checkpoint_path):
        return [], []
    with open(checkpoint_path, 'r') as f:
        data = json.load(f)
    return data.get('completed', []), data.get('paths', [])


def global_voxel_merge(
    *,
    points: torch.Tensor,
    features: torch.Tensor,
    colors: torch.Tensor,
    ray_dirs: torch.Tensor,
    ray_dirs_mean: torch.Tensor,
    cluster_sizes: torch.Tensor,
    voxel_size: float,
    ray_dirs_dominant: Optional[torch.Tensor] = None,
    ray_dirs_first: Optional[torch.Tensor] = None,
    plucker_rays: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    """Merge cross-view duplicates in world coordinates by voxel averaging."""
    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError(f"global_voxel_merge expects points shape (N, >=3), got {tuple(points.shape)}")
    if points.shape[0] == 0:
        raise ValueError("global_voxel_merge received empty points tensor.")
    if voxel_size <= 0:
        raise ValueError(f"global_voxel_merge expects voxel_size > 0, got {voxel_size}")

    quantized = torch.floor(points[:, :3] / float(voxel_size)).to(torch.int64)
    _, inverse = torch.unique(quantized, dim=0, return_inverse=True)
    n_voxels = int(inverse.max().item()) + 1

    def _scatter_mean(src: torch.Tensor) -> torch.Tensor:
        if src.ndim != 2:
            raise ValueError(f"scatter mean expects rank-2 tensor, got shape {tuple(src.shape)}")
        acc_dtype = torch.float32 if src.dtype in (torch.float16, torch.bfloat16) else src.dtype
        out = torch.zeros(n_voxels, src.shape[1], device=src.device, dtype=acc_dtype)
        out.index_add_(0, inverse, src.to(acc_dtype))
        cnt = torch.zeros(n_voxels, 1, device=src.device, dtype=acc_dtype)
        cnt.index_add_(0, inverse, torch.ones(src.shape[0], 1, device=src.device, dtype=acc_dtype))
        return out / cnt.clamp(min=1)

    def _scatter_sum(src: torch.Tensor) -> torch.Tensor:
        if src.ndim != 1:
            raise ValueError(f"scatter sum expects rank-1 tensor, got shape {tuple(src.shape)}")
        out = torch.zeros(n_voxels, device=src.device, dtype=src.dtype)
        out.index_add_(0, inverse, src)
        return out

    merged = {
        "points": _scatter_mean(points.float()),
        "features": _scatter_mean(features.float()),
        "colors": _scatter_mean(colors.float()),
        "ray_dirs": torch.nn.functional.normalize(_scatter_mean(ray_dirs.float()), dim=1, eps=1e-6),
        "ray_dirs_mean": torch.nn.functional.normalize(_scatter_mean(ray_dirs_mean.float()), dim=1, eps=1e-6),
        "cluster_sizes": _scatter_sum(cluster_sizes.to(torch.long)),
    }

    if ray_dirs_dominant is not None:
        merged["ray_dirs_dominant"] = torch.nn.functional.normalize(
            _scatter_mean(ray_dirs_dominant.float()), dim=1, eps=1e-6
        )
    if ray_dirs_first is not None:
        merged["ray_dirs_first"] = torch.nn.functional.normalize(
            _scatter_mean(ray_dirs_first.float()), dim=1, eps=1e-6
        )
    if plucker_rays is not None:
        pooled_plucker = _scatter_mean(plucker_rays.float())
        pooled_plucker[:, :3] = torch.nn.functional.normalize(pooled_plucker[:, :3], dim=1, eps=1e-6)
        merged["plucker_rays"] = pooled_plucker

    return merged


def two_pass_processing(
    memory_views: List[Dict[str, torch.Tensor]],
    raw_batches: List[Dict[str, Any]],  # Original data for depth/intrinsics
    features_dict: Dict[int, Dict[str, List[torch.Tensor]]],
    pooler: BSEPooler,
    welford: WelfordNormalizer,
    config: ExtractionConfig
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, float, Dict[str, torch.Tensor]]:
    """
    Two-pass extraction with configurable per-view pre-pooling.

    Pass 1:
      - prepool_mode=per_view: Extract -> Unproject -> Pool -> Save per-view pooled chunks
      - prepool_mode=global_only: Extract -> Unproject -> Save raw per-view chunks
    Pass 2:
      - Assemble chunks
      - If global_only: run one-shot global pooling
      - Optional global voxel merge
      - Recompute mu/sigma -> normalize

    Also collects view-level camera information (pose, intrinsics, Plücker rays).

    Args:
        memory_views: List of processed view dicts (model inputs)
        raw_batches: List of original view dicts (for depth/intrinsics access)
        features_dict: Dict mapping view_idx to multi-scale features
        pooler: BSEPooler instance
        welford: WelfordNormalizer instance
        config: Extraction configuration

    Returns:
        result: Dict containing pooled tensors (points, features, colors, ray_dirs, etc.)
        mu_scene: [3] scene mean
        sigma_scene: scene std
        view_info: Dict containing view-level camera information
            - camera_centers: [M, 3] camera centers
            - camera_rotations: [M, 3, 3] rotation matrices
            - camera_intrinsics: [M, 3, 3] intrinsic matrices
            - plucker_main_rays: [M, 6] Plücker coordinates for main rays
            where M = number of memory views
    """
    # Get device from memory_views
    device = memory_views[0]['img'].device if len(memory_views) > 0 else torch.device('cuda:0')

    os.makedirs(config.temp_dir, exist_ok=True)
    chunk_paths = []
    checkpoint_path = os.path.join(config.temp_dir, CHECKPOINT_FILE)

    # Check for resume
    completed_views, existing_chunks = load_checkpoint(checkpoint_path)
    if completed_views:
        # Validate checkpoint: /dev/shm is volatile and chunk files may be missing.
        existing_chunks = [p for p in existing_chunks if isinstance(p, str)]
        missing = [p for p in existing_chunks if not os.path.exists(p)]
        if missing or len(existing_chunks) != len(completed_views):
            print(
                f"[Resume] Checkpoint found but chunks are inconsistent "
                f"(completed={len(completed_views)}, paths={len(existing_chunks)}, missing={len(missing)}); "
                f"discarding resume state and restarting.",
                flush=True,
            )
            completed_views = []
            chunk_paths = []
            try:
                os.remove(checkpoint_path)
            except Exception:
                pass
        else:
            print(f"[Resume] Found {len(completed_views)} completed views", flush=True)
            chunk_paths = existing_chunks

    try:
        # Pass 1: extract -> optional per-view pool -> save chunks
        print(f"[Pass 1] Extracting with prepool_mode={config.prepool_mode}...")

        # Collect view-level camera information
        view_camera_centers = []
        view_camera_rotations = []
        view_camera_intrinsics = []
        view_plucker_main_rays = []
        all_cls_tokens = []
        saved_grid_H = None
        saved_grid_W = None
        saved_img_H = None
        saved_img_W = None

        for view_idx, (view_data, raw_batch) in enumerate(tqdm(zip(memory_views, raw_batches))):
            # Skip if already completed (resume capability)
            if view_idx in completed_views:
                continue

            # Get multi-scale features for this view
            view_info = features_dict[view_idx]
            feat_list = view_info['features']
            grid_H = view_info.get('grid_H', 37)  # Default for DINOv2
            grid_W = view_info.get('grid_W', 37)

            # Collect CLS token (scale token) for this view
            cls_token = view_info.get('cls_token')
            if cls_token is not None:
                all_cls_tokens.append(cls_token)

            # Prepare geometry data from raw_batch (original data)
            batch_gpu = {}
            for key in ["depth", "depthmap", "depth_z", "gt_depth", "intrinsics", "camera_intrinsics", "camera_pose", "pose", "scene_coords"]:
                if key in raw_batch:
                    val = raw_batch[key]
                    if isinstance(val, (list, tuple)): val = val[0]
                    if isinstance(val, np.ndarray): val = torch.from_numpy(val)
                    if isinstance(val, torch.Tensor): batch_gpu[key] = val.to(device)

            depth_for_hw = batch_gpu.get('depth', batch_gpu.get('depthmap', batch_gpu.get('depth_z', batch_gpu.get('gt_depth'))))
            if depth_for_hw is not None:
                image_H, image_W = normalize_depth_to_hw(depth_for_hw).shape
            else:
                image_H, image_W = view_data['img'].shape[-2:]

            # Process features: use_patch_based=False keeps the original
            # fps_memory pixel-based path; patch experiments use grid centers.
            if config.use_patch_based:
                features_flat, grid_H, grid_W = process_multiscale_features_to_grid(
                    feat_list, apply_l2_norm=False
                )
            else:
                features_flat, grid_H, grid_W = process_multiscale_features_to_image(
                    feat_list, int(image_H), int(image_W)
                )

            if saved_grid_H is None:
                saved_grid_H = int(grid_H)
                saved_grid_W = int(grid_W)
                saved_img_H = int(image_H)
                saved_img_W = int(image_W)

            if config.pool_mode == "pooled":
                compat_img = mapanything_pooled_color_image(
                    view_data,
                    int(image_H),
                    int(image_W),
                    device,
                )
                if compat_img is not None:
                    batch_gpu["images"] = compat_img

            # Add RGB for color sampling (required for colored PLY, esp. Indoor6).
            # Prefer raw uint8 if present; otherwise fall back to processed model input.
            if "images" not in batch_gpu:
                img_src = raw_batch.get("img_uint8", raw_batch.get("img", raw_batch.get("image")))
                if isinstance(img_src, (list, tuple)):
                    img_src = img_src[0] if img_src else None
                img_t = None
                if isinstance(img_src, torch.Tensor):
                    img_t = img_src.detach()
                elif isinstance(img_src, np.ndarray):
                    img_t = torch.from_numpy(img_src)
                if img_t is None and isinstance(view_data, dict) and "img" in view_data and isinstance(view_data["img"], torch.Tensor):
                    img_t = view_data["img"].detach()

                if img_t is not None:
                    # Normalize to float [0,1], shape [1,3,H,W]
                    if img_t.ndim == 4:
                        img_t = img_t[0]
                    if img_t.ndim == 3 and img_t.shape[0] == 3:
                        pass
                    elif img_t.ndim == 3 and img_t.shape[-1] == 3:
                        img_t = img_t.permute(2, 0, 1)
                    else:
                        img_t = None
                if img_t is not None:
                    img_t = img_t.to(device).float()
                    if img_t.max() > 1.01:
                        img_t = img_t / 255.0
                    img_t = torch.clamp(img_t, 0.0, 1.0)
                    batch_gpu["images"] = img_t.unsqueeze(0)

            # Provide RGB to unprojection for colored PLY (used by unproject_with_grid_features via 'images')
            # Keep this strictly local to visualization/color sampling; does not affect model inputs.
            if "images" not in batch_gpu:
                rgb_src = raw_batch.get("img_uint8", raw_batch.get("img", raw_batch.get("image")))
                if isinstance(rgb_src, (list, tuple)):
                    rgb_src = rgb_src[0] if rgb_src else None
                try:
                    if rgb_src is not None:
                        if hasattr(rgb_src, "convert"):
                            rgb_np = np.asarray(rgb_src.convert("RGB"))
                            rgb_t = torch.from_numpy(rgb_np)
                        elif isinstance(rgb_src, np.ndarray):
                            rgb_t = torch.from_numpy(rgb_src)
                        elif isinstance(rgb_src, torch.Tensor):
                            rgb_t = rgb_src
                        else:
                            rgb_t = torch.as_tensor(rgb_src)

                        # Normalize layout to [1,3,H,W]
                        if rgb_t.ndim == 4:
                            rgb_t = rgb_t[0]
                        if rgb_t.ndim == 3 and rgb_t.shape[0] == 3:
                            rgb_chw = rgb_t
                        elif rgb_t.ndim == 3 and rgb_t.shape[-1] in (3, 4):
                            rgb_chw = rgb_t[..., :3].permute(2, 0, 1)
                        else:
                            rgb_chw = None

                        if rgb_chw is not None:
                            rgb_chw = rgb_chw.contiguous()
                            if rgb_chw.dtype != torch.float32:
                                rgb_chw = rgb_chw.float()
                            if rgb_chw.max() > 1.1:
                                rgb_chw = rgb_chw / 255.0
                            rgb_chw = torch.clamp(rgb_chw, 0.0, 1.0)
                            batch_gpu["images"] = rgb_chw.unsqueeze(0).to(device)
                except Exception:
                    # If RGB extraction fails, fall back to gray colors in unproject_with_grid_features
                    pass

            # Debug: Check depth data
            depth_val = batch_gpu.get('depth', batch_gpu.get('depthmap', batch_gpu.get('depth_z')))
            if depth_val is not None:
                print(f"  [View {view_idx}] Depth: min={depth_val.min().item():.3f}, max={depth_val.max().item():.3f}, "
                      f"mean={depth_val.mean().item():.3f}", flush=True)
            else:
                print(f"  [View {view_idx}] WARNING: No depth data found in raw_batch!", flush=True)
                print(f"  Available keys: {list(raw_batch.keys())}", flush=True)

            # Geometry path: match map-anything pooled baseline when not patch-based.
            if config.use_patch_based:
                points, ray_dirs, colors, features_flat, camera_centers = unproject_with_grid_features(
                    batch_gpu,
                    features_flat,
                    grid_H,
                    grid_W,
                    config.depth_valid_range,
                    sampling_method=config.patch_depth_sampling,
                )
            else:
                points, ray_dirs, colors, features_flat, camera_centers = unproject_with_pixel_features(
                    batch_gpu,
                    features_flat,
                    grid_H,
                    grid_W,
                    config.depth_valid_range,
                )

            # Skip if no valid points
            if len(points) == 0:
                print(f"[Warning] No valid points for view {view_idx}")
                continue

            # Optional: SOR filtering (MEDIUM priority)
            if config.enable_sor and config.pool_mode != "pooled":
                print(
                    f"[SOR] Running per-view SOR "
                    f"(n={int(points.shape[0])}, nb_neighbors={int(config.sor_k)}, "
                    f"std_ratio={float(config.sor_std_ratio):.3f})...",
                    flush=True,
                )
                inlier_mask = sor_filter_mapanything_compatible(
                    points,
                    nb_neighbors=config.sor_k,
                    std_ratio=config.sor_std_ratio,
                )
                print(
                    f"[SOR] Per-view SOR removed {int((~inlier_mask).sum().item())} / "
                    f"{int(inlier_mask.numel())} points.",
                    flush=True,
                )
                points = points[inlier_mask]
                ray_dirs = ray_dirs[inlier_mask]
                colors = colors[inlier_mask]
                features_flat = features_flat[inlier_mask]
                camera_centers = camera_centers[inlier_mask]

            # Validate inputs
            validate_inputs(points, features_flat, colors, ray_dirs)

            if config.prepool_mode == "per_view":
                chunk_payload = pooler.pool(
                    points, features_flat, colors, ray_dirs,
                    camera_centers=camera_centers
                )
                # Log-only stats before any global merge.
                welford.update(chunk_payload['points'])
            elif config.prepool_mode == "global_only":
                chunk_payload = {
                    'points': points.detach().cpu(),
                    'features': features_flat.detach().cpu(),
                    'colors': colors.detach().cpu(),
                    'ray_dirs': ray_dirs.detach().cpu(),
                    'camera_centers': camera_centers.detach().cpu(),
                    'view_ids': torch.full((points.shape[0],), int(view_idx), dtype=torch.long).cpu(),
                }
                welford.update(points)
            else:
                raise ValueError(f"Unsupported prepool_mode={config.prepool_mode!r}")

            # Collect view-level camera information from raw_batch
            pose = raw_batch.get("camera_pose", raw_batch.get("pose"))
            if isinstance(pose, (list, tuple)):
                pose = pose[0]
            if isinstance(pose, np.ndarray):
                pose = torch.from_numpy(pose)
            pose = pose.to(device)
            if pose.ndim == 2:
                pose = pose.unsqueeze(0)
            pose = pose[0]  # [4, 4]

            intrinsics = raw_batch.get("camera_intrinsics", raw_batch.get("intrinsics"))
            if isinstance(intrinsics, (list, tuple)):
                intrinsics = intrinsics[0]
            if isinstance(intrinsics, np.ndarray):
                intrinsics = torch.from_numpy(intrinsics)
            intrinsics = intrinsics.to(device)
            if intrinsics.ndim == 2:
                intrinsics = intrinsics.unsqueeze(0)
            intrinsics = intrinsics[0]  # [3, 3]

            # Extract camera center and rotation from c2w pose.
            R = pose[:3, :3]  # [3, 3]
            t = pose[:3, 3]   # [3]
            camera_center = t  # [3]

            # Main ray direction (camera forward direction: look along -Z axis in OpenGL convention)
            # In our coordinate system, the camera looks along +Z axis (after R transformation)
            main_ray_dir = R[:, 2]  # [3] - third column of rotation matrix

            # Compute Plücker coordinates for main ray: (direction, moment)
            # moment = camera_center × direction
            moment = torch.cross(camera_center, main_ray_dir, dim=0)
            plucker_main_ray = torch.cat([main_ray_dir, moment])  # [6]

            view_camera_centers.append(camera_center.cpu())
            view_camera_rotations.append(R.cpu())
            view_camera_intrinsics.append(intrinsics.cpu())
            view_plucker_main_rays.append(plucker_main_ray.cpu())

            # Save temporary chunk
            chunk_path = os.path.join(config.temp_dir, f"chunk_{view_idx}.pt")
            torch.save(chunk_payload, chunk_path)
            chunk_paths.append(chunk_path)

            # Save checkpoint for resume
            completed_views.append(view_idx)
            save_checkpoint(checkpoint_path, completed_views, chunk_paths)

            # HIGH Priority: GPU memory cleanup between views
            torch.cuda.empty_cache()

        # Pass 2: assemble chunks -> optional global pool/merge -> normalize once
        print("[Pass 2] Assembling raw chunks...")
        if welford.count > 0:
            premerge_mu, premerge_sigma = welford.finalize()
            print(
                f"[Pass 2] Pre-merge stats (log only): mu={premerge_mu.tolist()}, sigma={premerge_sigma:.6f}",
                flush=True,
            )

        all_points = []
        all_ray_dirs = []
        all_features = []
        all_colors = []
        all_camera_centers = []
        all_view_ids = []
        all_ray_dirs_mean = []
        all_ray_dirs_dominant = []
        all_ray_dirs_first = []
        all_plucker_rays = []
        all_cluster_sizes = []

        for chunk_path in tqdm(chunk_paths):
            if not os.path.exists(chunk_path):
                raise FileNotFoundError(
                    f"Missing chunk file during Pass2: {chunk_path}. "
                    f"If you changed N_VIEWS or temp_dir was cleared, remove {checkpoint_path} and rerun."
                )
            chunk = torch.load(chunk_path)
            all_points.append(chunk['points'])
            all_features.append(chunk['features'])
            all_colors.append(chunk['colors'])

            if config.prepool_mode == "global_only":
                all_ray_dirs.append(chunk['ray_dirs'])
                all_camera_centers.append(chunk['camera_centers'])
                if 'view_ids' in chunk:
                    all_view_ids.append(chunk['view_ids'])
            else:
                all_ray_dirs.append(chunk['ray_dirs'])
                all_ray_dirs_mean.append(chunk['ray_dirs_mean'])
                all_cluster_sizes.append(chunk['cluster_sizes'])

                # Optional ray representations
                if 'ray_dirs_dominant' in chunk:
                    all_ray_dirs_dominant.append(chunk['ray_dirs_dominant'])
                if 'ray_dirs_first' in chunk:
                    all_ray_dirs_first.append(chunk['ray_dirs_first'])
                if 'plucker_rays' in chunk:
                    all_plucker_rays.append(chunk['plucker_rays'])

        if config.prepool_mode == "global_only":
            print(f"[Pass 2] Running one-shot global pooling with pool_mode={config.pool_mode}...", flush=True)
            raw_points_global = torch.cat(all_points, dim=0).float()
            raw_features_global = torch.cat(all_features, dim=0).float()
            raw_colors_global = torch.cat(all_colors, dim=0).float()
            raw_ray_dirs_global = torch.cat(all_ray_dirs, dim=0).float()
            raw_camera_centers_global = torch.cat(all_camera_centers, dim=0).float()
            raw_view_ids_global = (
                torch.cat(all_view_ids, dim=0).long()
                if all_view_ids
                else torch.zeros(raw_points_global.shape[0], dtype=torch.long)
            )
            if config.pool_mode == "pooled" and config.enable_sor:
                sor_k, sor_std = mapanything_sor_params(int(raw_points_global.shape[0]))
                print(
                    f"[Pool] Running map-anything-compatible global SOR before pooled voxel mean "
                    f"(n={raw_points_global.shape[0]}, nb_neighbors={sor_k}, std_ratio={sor_std})...",
                    flush=True,
                )
                inlier_mask = sor_filter_mapanything_compatible(
                    raw_points_global.to(device),
                    nb_neighbors=sor_k,
                    std_ratio=sor_std,
                ).cpu()
                print(
                    f"[Pool] Global SOR removed {int((~inlier_mask).sum().item())} / "
                    f"{int(inlier_mask.numel())} raw points.",
                    flush=True,
                )
                raw_points_global = raw_points_global[inlier_mask]
                raw_features_global = raw_features_global[inlier_mask]
                raw_colors_global = raw_colors_global[inlier_mask]
                raw_ray_dirs_global = raw_ray_dirs_global[inlier_mask]
                raw_camera_centers_global = raw_camera_centers_global[inlier_mask]
                raw_view_ids_global = raw_view_ids_global[inlier_mask]
            save_feature_similarity_diagnostics(
                os.path.dirname(os.path.abspath(config.output_path)),
                config,
                pooler,
                raw_points_global,
                raw_features_global,
                raw_view_ids_global,
            )
            pooled_global = pooler.pool(
                raw_points_global.to(device),
                raw_features_global.to(device),
                raw_colors_global.to(device),
                raw_ray_dirs_global.to(device),
                camera_centers=raw_camera_centers_global.to(device),
            )
            final_points_world = pooled_global['points'].float().cpu()
            final_ray_dirs = pooled_global['ray_dirs'].float().cpu()
            final_ray_dirs_mean = pooled_global['ray_dirs_mean'].float().cpu()
            final_features = pooled_global['features'].float().cpu()
            final_colors = pooled_global['colors'].float().cpu()
            final_cluster_sizes = pooled_global['cluster_sizes'].long().cpu()
            optional_world = {}
            if 'ray_dirs_dominant' in pooled_global:
                optional_world['ray_dirs_dominant'] = pooled_global['ray_dirs_dominant'].float().cpu()
            if 'ray_dirs_first' in pooled_global:
                optional_world['ray_dirs_first'] = pooled_global['ray_dirs_first'].float().cpu()
            if 'plucker_rays' in pooled_global:
                optional_world['plucker_rays'] = pooled_global['plucker_rays'].float().cpu()
        else:
            # Concatenate per-view pooled chunks in world coordinates
            final_points_world = torch.cat(all_points, dim=0).float()
            final_ray_dirs = torch.cat(all_ray_dirs, dim=0).float()
            final_ray_dirs_mean = torch.cat(all_ray_dirs_mean, dim=0).float()
            final_features = torch.cat(all_features, dim=0).float()
            final_colors = torch.cat(all_colors, dim=0).float()
            final_cluster_sizes = torch.cat(all_cluster_sizes, dim=0).long()
            optional_world = {}
            if all_ray_dirs_dominant:
                optional_world['ray_dirs_dominant'] = torch.cat(all_ray_dirs_dominant, dim=0).float()
            if all_ray_dirs_first:
                optional_world['ray_dirs_first'] = torch.cat(all_ray_dirs_first, dim=0).float()
            if all_plucker_rays:
                optional_world['plucker_rays'] = torch.cat(all_plucker_rays, dim=0).float()

        n_before_merge = int(final_points_world.shape[0])
        if config.global_merge:
            merge_voxel_size = (
                float(config.global_merge_voxel_size)
                if config.global_merge_voxel_size is not None
                else float(config.voxel_size)
            )
            merged = global_voxel_merge(
                points=final_points_world,
                features=final_features,
                colors=final_colors,
                ray_dirs=final_ray_dirs,
                ray_dirs_mean=final_ray_dirs_mean,
                cluster_sizes=final_cluster_sizes,
                voxel_size=merge_voxel_size,
                ray_dirs_dominant=optional_world.get('ray_dirs_dominant'),
                ray_dirs_first=optional_world.get('ray_dirs_first'),
                plucker_rays=optional_world.get('plucker_rays'),
            )
            final_points_world = merged['points']
            final_features = merged['features']
            final_colors = merged['colors']
            final_ray_dirs = merged['ray_dirs']
            final_ray_dirs_mean = merged['ray_dirs_mean']
            final_cluster_sizes = merged['cluster_sizes']
            if 'ray_dirs_dominant' in merged:
                optional_world['ray_dirs_dominant'] = merged['ray_dirs_dominant']
            if 'ray_dirs_first' in merged:
                optional_world['ray_dirs_first'] = merged['ray_dirs_first']
            if 'plucker_rays' in merged:
                optional_world['plucker_rays'] = merged['plucker_rays']
            print(
                f"[Global Merge] {n_before_merge} -> {final_points_world.shape[0]} points "
                f"(removed {n_before_merge - int(final_points_world.shape[0])} cross-view duplicates, "
                f"voxel_size={merge_voxel_size})",
                flush=True,
            )
        else:
            print(f"[Global Merge] disabled; keeping {n_before_merge} points", flush=True)

        # Recompute normalization from final merged point cloud.
        mu_scene = final_points_world.mean(dim=0)
        centered = final_points_world - mu_scene.unsqueeze(0)
        sigma_scene = torch.sqrt((centered.pow(2).mean(dim=0)).mean()).item()
        if not np.isfinite(sigma_scene) or sigma_scene <= 1e-12:
            raise ValueError(f"Invalid sigma after global merge: {sigma_scene}")
        final_points = centered / sigma_scene
        print(
            f"[Pass 2] Final stats after merge: mu={mu_scene.tolist()}, sigma={sigma_scene:.6f}",
            flush=True,
        )

        # Build result dict with all ray representations
        result = {
            'points': final_points,
            'ray_dirs': final_ray_dirs,
            'ray_dirs_mean': final_ray_dirs_mean,
            'features': final_features,
            'colors': final_colors,
            'cluster_sizes': final_cluster_sizes
        }

        # Add optional ray representations
        if 'ray_dirs_dominant' in optional_world:
            result['ray_dirs_dominant'] = optional_world['ray_dirs_dominant']
        if 'ray_dirs_first' in optional_world:
            result['ray_dirs_first'] = optional_world['ray_dirs_first']
        if 'plucker_rays' in optional_world:
            result['plucker_rays'] = optional_world['plucker_rays']
        # NOTE: We remove pooled camera_centers since we now have view-level camera info
        # if all_camera_centers:
        #     result['camera_centers'] = torch.cat(all_camera_centers, dim=0)

        # Assemble view-level camera information
        camera_centers_stack = torch.stack(view_camera_centers, dim=0)      # [M, 3]
        scene_center_cam = camera_centers_stack.mean(dim=0)                 # [3] — mean camera center
        view_info = {
            'camera_centers': camera_centers_stack,                          # [M, 3]
            'camera_rotations': torch.stack(view_camera_rotations, dim=0),  # [M, 3, 3]
            'camera_intrinsics': torch.stack(view_camera_intrinsics, dim=0), # [M, 3, 3]
            'plucker_main_rays': torch.stack(view_plucker_main_rays, dim=0), # [M, 6]
            'all_scale_tokens': torch.stack(all_cls_tokens, dim=0) if all_cls_tokens else None,  # [M, D]
            'scene_center_cam': scene_center_cam,                           # [3] — mean camera center (for head.mean)
            'grid_H': saved_grid_H,
            'grid_W': saved_grid_W,
            'img_H': saved_img_H,
            'img_W': saved_img_W,
        }

        return result, mu_scene, sigma_scene, view_info

    finally:
        # HIGH Priority: Clean up temp files in try/finally
        print("[Cleanup] Removing temporary files...")
        for chunk_path in chunk_paths:
            if os.path.exists(chunk_path):
                os.remove(chunk_path)
        if os.path.exists(checkpoint_path):
            os.remove(checkpoint_path)


def save_memory(
    output_path: str,
    result: Dict[str, torch.Tensor],
    mu: torch.Tensor,
    sigma: float,
    view_info: Dict[str, torch.Tensor] = None,
    layers_idx: Optional[List] = None,
    scene_center: Optional[torch.Tensor] = None,
    config: Optional[ExtractionConfig] = None,
    selection_metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Save memory to .pt file with extended schema and versioning.

    Output schema includes:
    - Point-level data:
        - points: [N, 3] normalized coordinates, (points_world - mu) / sigma
        - points_norm: [N, 3] explicit normalized-coordinate alias
        - points_world: [N, 3] world coordinates
        - pooled_points: [N, 3] legacy pooled-format alias, kept in world coordinates
        - features / pooled_features: pooled feature tensor and legacy alias
        - ray_dirs: Primary ray directions (based on ray_pool_strategy)
        - ray_dirs_mean: Mean + normalized (baseline)
        - ray_dirs_dominant: Dominant ray per cluster (optional)
        - ray_dirs_first: First ray per cluster (optional)
        - plucker_rays: 6D Plücker coordinates for pooled rays (optional)
        - cluster_sizes: Number of points per cluster
    - View-level camera information (if provided):
        - camera_centers: [M, 3] camera centers for each memory view
        - camera_rotations: [M, 3, 3] rotation matrices for each memory view
        - camera_intrinsics: [M, 3, 3] intrinsic matrices for each memory view
        - plucker_main_rays: [M, 6] Plücker coordinates for main rays

    Args:
        output_path: Output file path
        result: Dict containing all pooled tensors
        mu: [3] scene mean
        sigma: scene std
        view_info: Dict containing view-level camera information (optional)
    """
    points_norm = result['points'].cpu().float()
    points_world = points_norm * float(sigma) + mu.cpu().float().view(1, 3)
    preserve_pooled_dtype = config is not None and config.pool_mode == "pooled"
    features = result['features'].cpu().float() if preserve_pooled_dtype else result['features'].cpu().half()
    mu_cpu = mu.cpu().float()
    sigma_tensor = torch.as_tensor(float(sigma), dtype=torch.float32)

    memory_dict = {
        'schema_version': MEMORY_SCHEMA_VERSION,
        'points': points_norm,                              # [N, 3], normalized: (points_world - mu) / sigma
        'points_norm': points_norm,                         # [N, 3], explicit normalized-coordinate alias
        'points_world': points_world,                       # [N, 3], world coordinates, avoids runtime de-normalization
        'pooled_points': points_world,                      # [N, 3], legacy pooled-format alias in world coordinates
        'ray_dirs': result['ray_dirs'].cpu().float(),       # [N, 3]
        'ray_dirs_mean': result['ray_dirs_mean'].cpu().float(),  # [N, 3]
        'features': features,                               # [N, C] pooled fp32, BSE fp16
        'pooled_features': features,                        # [N, C] legacy pooled-format alias
        'colors': result['colors'].cpu().float(),           # [N, 3]
        'pooled_colors': result['colors'].cpu().float(),    # [N, 3] legacy pooled-format alias
        'cluster_sizes': result['cluster_sizes'].cpu().long(),  # [N]
        'mu': mu_cpu,                                       # [3]
        'sigma': float(sigma),                              # scalar
        'normalization_mu': mu_cpu,                         # [3], explicit name consumed by training
        'normalization_sigma': sigma_tensor,                # scalar tensor, explicit name consumed by training
        'coordinate_space': {
            'points': 'normalized',
            'points_norm': 'normalized',
            'points_world': 'world',
            'pooled_points': 'world',
            'formula': 'points_norm = (points_world - normalization_mu) / normalization_sigma',
        },
    }

    all_poses = None
    all_intrinsics = None
    if view_info is not None and view_info.get('camera_centers') is not None and view_info.get('camera_rotations') is not None:
        centers = view_info['camera_centers'].cpu().float()
        rotations = view_info['camera_rotations'].cpu().float()
        if centers.ndim == 2 and rotations.ndim == 3 and rotations.shape[0] == centers.shape[0]:
            all_poses = torch.eye(4, dtype=torch.float32).unsqueeze(0).repeat(centers.shape[0], 1, 1)
            all_poses[:, :3, :3] = rotations
            all_poses[:, :3, 3] = centers
    if view_info is not None and view_info.get('camera_intrinsics') is not None:
        all_intrinsics = view_info['camera_intrinsics'].cpu().float()

    if all_poses is not None:
        # Legacy pooled_GT fields, kept alongside the extended camera metadata.
        memory_dict['all_poses'] = all_poses
        memory_dict['ref_pose'] = all_poses[0].clone()
        memory_dict['original_views'] = int(all_poses.shape[0])
    if all_intrinsics is not None:
        memory_dict['all_intrinsics'] = all_intrinsics
    memory_dict['patch_stride'] = 14.0

    if config is not None:
        if config.pool_mode == "pooled":
            memory_dict['type'] = (
                "pooled_gt_geometry_patch_based"
                if config.use_patch_based
                else "pooled_gt_geometry_enhanced_feats"
            )
            memory_dict['extraction_method'] = "patch_based" if config.use_patch_based else "pixel_based"
            memory_dict['l2_normalization'] = False
            memory_dict['depth_sampling_method'] = (
                config.patch_depth_sampling if config.use_patch_based else "pixel"
            )
            memory_dict['depth_valid_min'] = float(config.depth_valid_range[0])
            memory_dict['depth_valid_max'] = float(config.depth_valid_range[1])
            memory_dict['depth_sparsify_prob'] = 0.0
            memory_dict['depth_sparsify_removal_percent'] = 0.0
            memory_dict['depth_sparsify_seed'] = 3407
            if view_info is not None:
                memory_dict['grid_H'] = view_info.get('grid_H') if config.use_patch_based else None
                memory_dict['grid_W'] = view_info.get('grid_W') if config.use_patch_based else None
                memory_dict['img_H'] = view_info.get('img_H') if config.use_patch_based else None
                memory_dict['img_W'] = view_info.get('img_W') if config.use_patch_based else None
        memory_dict['pool_mode'] = config.pool_mode
        memory_dict['prepool_mode'] = config.prepool_mode
        memory_dict['global_merge'] = bool(config.global_merge)
        memory_dict['global_merge_voxel_size'] = (
            float(config.global_merge_voxel_size)
            if config.global_merge_voxel_size is not None
            else None
        )
        memory_dict['voxel_size'] = float(config.voxel_size)
        if config.scene_name is not None:
            memory_dict['scene'] = config.scene_name

    if selection_metadata:
        memory_dict['selection_metadata'] = selection_metadata
        for key in (
            'selection_strategy',
            'wai_view_mode',
            'selected_memory_indices',
            'memory_index_groups',
            'dataset_num_views',
            'scene_center_source',
        ):
            if key in selection_metadata:
                memory_dict[key] = selection_metadata[key]

    if scene_center is not None:
        memory_dict['scene_center'] = scene_center.cpu().float()  # [3] pooled-compatible scene anchor

    # Add optional ray representations
    if 'ray_dirs_dominant' in result:
        memory_dict['ray_dirs_dominant'] = result['ray_dirs_dominant'].cpu().float()
    if 'ray_dirs_first' in result:
        memory_dict['ray_dirs_first'] = result['ray_dirs_first'].cpu().float()
    if 'plucker_rays' in result:
        memory_dict['plucker_rays'] = result['plucker_rays'].cpu().float()  # [N, 6]

    # Add view-level camera information (if provided)
    if view_info is not None:
        memory_dict['view_camera_centers'] = view_info['camera_centers']          # [M, 3]
        memory_dict['view_camera_rotations'] = view_info['camera_rotations']      # [M, 3, 3]
        memory_dict['view_camera_intrinsics'] = view_info['camera_intrinsics']    # [M, 3, 3]
        memory_dict['view_plucker_main_rays'] = view_info['plucker_main_rays']    # [M, 6]
        if view_info.get('all_scale_tokens') is not None:
            memory_dict['all_scale_tokens'] = view_info['all_scale_tokens'].cpu().float()  # [M, D]
        if view_info.get('scene_center_cam') is not None:
            memory_dict['scene_center_cam'] = view_info['scene_center_cam'].cpu().float()  # [3]

    # Add layers_idx (for compatibility with pooled memory format)
    if layers_idx is not None:
        memory_dict['layers_idx'] = layers_idx

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

    torch.save(memory_dict, output_path)
    print(f"[Save] Memory saved to {output_path}")
    print(f"[Save] Schema version: {MEMORY_SCHEMA_VERSION}")
    print(f"[Save] Total points: {len(result['points'])}")
    print(f"[Save] Feature dim: {result['features'].shape[1]}")
    print(f"[Save] Scene mean: {mu.tolist()}")
    print(f"[Save] Scene sigma: {sigma:.4f}")
    if scene_center is not None:
        print(f"[Save] Scene center: {scene_center.cpu().float().tolist()}")
    if config is not None:
        print(
            f"[Save] Pooling: pool_mode={config.pool_mode}, prepool_mode={config.prepool_mode}, "
            f"global_merge={bool(config.global_merge)}"
        )
    print(f"[Save] World points: {tuple(memory_dict['points_world'].shape)}")
    print(f"[Save] Normalized points: {tuple(memory_dict['points_norm'].shape)}")
    print("[Save] Legacy aliases: pooled_points=points_world, pooled_features=features")
    print(f"[Save] Ray representations: {[k for k in memory_dict.keys() if 'ray' in k or 'plucker' in k]}")
    if view_info is not None:
        print(f"[Save] View count: {len(view_info['camera_centers'])}")
        print(f"[Save] View info: camera_centers, camera_rotations, camera_intrinsics, plucker_main_rays")
        if 'all_scale_tokens' in memory_dict:
            print(f"[Save] Scale tokens: {memory_dict['all_scale_tokens'].shape}")


def save_ply(
    output_path: str,
    result: Dict[str, torch.Tensor],
    mu: torch.Tensor,
    sigma: float
) -> None:
    """
    Save point cloud as PLY file for visualization.

    Saves points with colors (unnormalized from scene coordinates).

    Args:
        output_path: Output file path (will replace .pt with .ply)
        result: Dict containing pooled tensors
        mu: [3] scene mean
        sigma: scene std
    """
    # Determine PLY path
    ply_path = output_path.replace('.pt', '.ply')
    if ply_path == output_path:
        ply_path = output_path + '.ply'

    # Points in `result` are normalized as (p - mu) / sigma in two_pass_processing; invert for PLY
    pts = result['points'].cpu().float() * float(sigma) + mu.cpu().float().view(1, 3)
    points = pts.numpy().astype(np.float64)
    colors = result['colors'].cpu().numpy()

    # Normalize colors to [0, 1] if needed
    if colors.max() > 1.1:
        colors = colors / 255.0
    colors = np.clip(colors, 0, 1).astype(np.float64)

    def _write_xyz_rgb_ply(path: str, pts_xyz: np.ndarray, cols01: np.ndarray) -> None:
        with open(path, 'w') as f:
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {len(pts_xyz)}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")
            f.write("end_header\n")
            for i in range(len(pts_xyz)):
                x, y, z = pts_xyz[i]
                r, g, b = (cols01[i] * 255).astype(np.uint8)
                f.write(f"{x:.6f} {y:.6f} {z:.6f} {r} {g} {b}\n")

    # Write base PLY
    _write_xyz_rgb_ply(ply_path, points, colors)

    # Optional: visualization-only "splat" PLY to make points look larger in viewers.
    # PLY itself has no point-size attribute; viewers usually control it. This file is for quick inspection.
    splat = float(os.environ.get("PLY_VIS_SPLAT_RADIUS", "0") or "0")
    splat_k = int(os.environ.get("PLY_VIS_SPLAT_K", "12") or "12")
    if splat > 0:
        rng = np.random.default_rng(0)
        k = max(1, splat_k)
        offs = rng.normal(size=(k, 3)).astype(np.float64)
        norms = np.linalg.norm(offs, axis=1, keepdims=True) + 1e-12
        offs = offs / norms
        radii = rng.random(size=(k, 1)).astype(np.float64) ** (1.0 / 3.0)
        offs = offs * radii * float(splat)
        offs = np.vstack([np.zeros((1, 3), dtype=np.float64), offs])

        pts_vis = (points[:, None, :] + offs[None, :, :]).reshape(-1, 3)
        cols_vis = np.repeat(colors, repeats=offs.shape[0], axis=0)
        ply_vis_path = ply_path.replace(".ply", "_vis_splat.ply")
        _write_xyz_rgb_ply(ply_vis_path, pts_vis, cols_vis)
        print(f"[PLY] Vis splat PLY saved to {ply_vis_path} (radius={splat}, k={k})")

    print(f"[PLY] Point cloud saved to {ply_path}")
    print(f"[PLY] Points: {len(points):,}")


# =============================================================================
# ACE Dataset Wrapper
# =============================================================================

class ACEDatasetWithDepth:
    """Wrapper for CamLocDatasetDINOv2 that adds depth loading from depth/ directory."""
    def __init__(self, base_dataset):
        self.base_dataset = base_dataset
        self.depth_dir = Path(base_dataset.rgb_files[0].parent.parent) / 'depth'

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        from skimage import io
        # Get base data (image, mask, pose, intrinsics, etc.)
        data = self.base_dataset[idx]

        # Load depth from depth/ directory
        rgb_file = self.base_dataset.rgb_files[self.base_dataset.valid_file_indices[idx]]
        depth_file = self.depth_dir / rgb_file.name.replace('.color.png', '.depth.png')

        if depth_file.exists():
            depth = io.imread(str(depth_file))
            depth = depth.astype(np.float32) / 1000.0  # mm to meters
        else:
            depth = np.zeros((self.base_dataset.image_height,
                            data[0].shape[2] if data[0].ndim == 3 else data[0].shape[1]),
                           dtype=np.float32)

        # Return as dict compatible with WAI format
        return {
            'img': data[0],
            'depthmap': depth,
            'camera_pose': data[2],
            'camera_intrinsics': data[4],
            'image': data[0],
            'pose': data[2],
            'intrinsics': data[4],
        }

# =============================================================================
# Dataset Detection
# =============================================================================

def detect_dataset_type(dataset_path: str) -> str:
    """
    Auto-detect dataset type from path.

    Returns:
        "7scenes" or "indoor6" or "custom"
    """
    path_lower = dataset_path.lower()
    if "7scenes" in path_lower or "7-scenes" in path_lower:
        return "7scenes"
    elif "indoor6" in path_lower or "indoor-6" in path_lower:
        return "indoor6"
    return "custom"


def load_dataset(
    dataset_path: str,
    dataset_type: str = "auto",
    scene_name: Optional[str] = None,
    n_views: int = 1,
    dataset_loader: str = "wai",
):
    """
    Load dataset based on type and loader.

    For mapanything WAI datasets, returns a sequential-view-mode dataset. Keep
    ``n_views=1`` when each FPS index should map to exactly one model input
    view; WAI returns an anchor-centered multi-view sample when ``n_views>1``.

    Args:
        dataset_path: ROOT directory for WAI datasets (e.g. /data/.../7scenes).
        dataset_type: "7scenes", "indoor6", "custom", or "auto"
        scene_name: Specific scene to load (e.g. "chess_train"). Required when
                    dataset_path points to a ROOT directory containing multiple scenes.
        n_views: Number of views per sample for WAI. Use 1 for true FPS-frame
                 inputs.

    Returns:
        Dataset object where __getitem__(idx) returns a view dict or a list of
        view dicts with keys such as img, depthmap, camera_pose, and intrinsics.
    """
    if dataset_type == "auto":
        dataset_type = detect_dataset_type(dataset_path)

    # === ACE dataset loader branch ===
    if dataset_loader == "ace":
        print(f"[Dataset] Using ACE loader (CamLocDatasetDINOv2)")
        # Add parent directory to path for imports
        parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        if parent_dir not in sys.path:
            sys.path.insert(0, parent_dir)
        from dataset_dinov2 import CamLocDatasetDINOv2
        base_ds = CamLocDatasetDINOv2(dataset_path, mode=2)
        return ACEDatasetWithDepth(base_ds)

    # === WAI / MapAnything dataset loader branch ===
    if dataset_loader != "wai":
        raise ValueError(f"Unsupported dataset_loader={dataset_loader!r}. Use 'wai' or 'ace'.")

    from mapanything.datasets.wai.seven_scenes import SevenScenesWAI  # pyright: ignore[reportMissingImports]
    from mapanything.datasets.wai.indoor6 import Indoor6WAI  # pyright: ignore[reportMissingImports]

    # Determine dataset_metadata_dir
    metadata_dir = os.environ.get(
        "MAPANYTHING_DATASET_METADATA_DIR",
        "/mnt/storage/xwh/map-anything/mapanything_dataset_metadata"
    )

    # Common kwargs for BaseDataset
    base_kwargs = dict(
        resolution=518,
        data_norm_type='dinov2',
        transform='imgnorm',
    )

    if dataset_type == "7scenes":
        return SevenScenesWAI(
            ROOT=dataset_path,
            dataset_metadata_dir=metadata_dir,
            split='train',
            sample_specific_scene=True,
            specific_scene_name=scene_name,
            sequential_view_mode=True,
            num_views=n_views,
            **base_kwargs,
        )
    if dataset_type == "indoor6":
        return Indoor6WAI(
            ROOT=dataset_path,
            dataset_metadata_dir=metadata_dir,
            split='train',
            sample_specific_scene=True,
            specific_scene_name=scene_name,
            sequential_view_mode=True,
            num_views=n_views,
            **base_kwargs,
        )
    raise ValueError(f"Unsupported dataset_type={dataset_type!r} for WAI loader.")


def _pose_to_center_np(pose: Any) -> Optional[np.ndarray]:
    """pose [4,4] -> camera center [3] in world coords, or None."""
    if pose is None:
        return None
    if isinstance(pose, torch.Tensor):
        p = pose.detach().cpu().float().numpy()
    else:
        p = np.asarray(pose)
    if p.size == 16:
        p = p.reshape(4, 4)
    if p.shape != (4, 4):
        return None
    t = p[:3, 3]
    # WAI/MapAnything poses are camera-to-world transforms; translation is the
    # camera center in world coordinates.
    c = t
    if not np.isfinite(c).all():
        return None
    return c.astype(np.float64)


def _to_numpy_safe(x: Any) -> Optional[np.ndarray]:
    """Best-effort tensor/array -> numpy (CPU), without changing values."""
    if x is None:
        return None
    try:
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        if isinstance(x, np.ndarray):
            return x
        return np.asarray(x)
    except Exception:
        return None


def _stat_np(arr: Optional[np.ndarray]) -> Dict[str, Any]:
    if arr is None:
        return {"present": False}
    a = np.asarray(arr)
    out: Dict[str, Any] = {"present": True, "shape": list(a.shape), "dtype": str(a.dtype)}
    try:
        if a.size > 0 and np.issubdtype(a.dtype, np.number):
            finite = np.isfinite(a)
            out["finite_ratio"] = float(finite.mean()) if finite.size else None
            if finite.any():
                af = a[finite]
                out.update(
                    min=float(np.min(af)),
                    max=float(np.max(af)),
                    mean=float(np.mean(af)),
                    std=float(np.std(af)),
                )
    except Exception:
        pass
    return out


def _pose_stats(pose_4x4: Optional[np.ndarray]) -> Dict[str, Any]:
    """Summarize pose; supports [4,4] or [1,4,4]."""
    p = pose_4x4
    if p is None:
        return {"present": False}
    p = np.asarray(p)
    if p.size == 16:
        p = p.reshape(4, 4)
    if p.ndim == 3 and p.shape[0] == 1 and p.shape[1:] == (4, 4):
        p = p[0]
    if p.shape != (4, 4):
        return {"present": True, "shape": list(p.shape), "error": "not_4x4"}
    R = p[:3, :3]
    t = p[:3, 3]
    c = t
    return {
        "present": True,
        "t": [float(x) for x in t.tolist()],
        "c": [float(x) for x in c.tolist()],
        "R_det": float(np.linalg.det(R)),
        "R_orth_err": float(np.linalg.norm(R.T @ R - np.eye(3))),
    }


def debug_dump_views(
    output_run_dir: str,
    config: "ExtractionConfig",
    raw_batches: List[Any],
    memory_views: List[Dict[str, Any]],
    depth_min: float,
    depth_max: float,
) -> Optional[str]:
    """Dump raw/processed per-view stats to <run_dir>/debug_dump for ace vs wai comparison."""
    if not getattr(config, "debug_dump", False):
        return None
    dump_dir = os.path.join(output_run_dir, "debug_dump")
    os.makedirs(dump_dir, exist_ok=True)

    n = min(len(raw_batches), len(memory_views), int(getattr(config, "debug_dump_max_views", 3)))
    meta: Dict[str, Any] = {
        "schema": 1,
        "dataset_loader": config.dataset_loader,
        "dataset_type": config.dataset_type,
        "scene_name": config.scene_name,
        "dataset_path": config.dataset_path,
        "n_dump": n,
        "depth_valid_range": [float(depth_min), float(depth_max)],
        "patch_depth_sampling": config.patch_depth_sampling,
        "use_patch_based": bool(config.use_patch_based),
        "voxel_size": float(config.voxel_size),
        "pool_mode": config.pool_mode,
        "prepool_mode": config.prepool_mode,
        "use_otsu": bool(config.use_otsu),
        "unimodal_threshold": float(config.unimodal_threshold),
        "hybrid_split_min_std": float(config.hybrid_split_min_std),
        "hybrid_min_cluster_size": int(config.hybrid_min_cluster_size),
        "hybrid_min_split_points": int(config.hybrid_min_split_points),
        "global_merge_voxel_size": (
            float(config.global_merge_voxel_size)
            if config.global_merge_voxel_size is not None
            else None
        ),
        "ray_pool_strategy": config.ray_pool_strategy,
        "use_model": config.use_model,
        "model_str": config.model_str,
        "feature_diagnostics": bool(config.feature_diagnostics),
        "feature_diag_max_points_per_voxel": int(config.feature_diag_max_points_per_voxel),
        "feature_diag_max_pair_samples": int(config.feature_diag_max_pair_samples),
    }

    views_out: List[Dict[str, Any]] = []
    for i in range(n):
        raw = raw_batches[i]
        proc = memory_views[i]

        raw_dict = raw if isinstance(raw, dict) else {}

        raw_img = raw_dict.get("img", raw_dict.get("image", raw_dict.get("img_uint8")))
        raw_depth = raw_dict.get("depthmap", raw_dict.get("depth", raw_dict.get("depth_z")))
        raw_intr = raw_dict.get("camera_intrinsics", raw_dict.get("intrinsics"))
        raw_pose = raw_dict.get("camera_pose", raw_dict.get("pose"))
        raw_fn = raw_dict.get("filename")

        proc_img = proc.get("img")
        proc_depth = proc.get("depth_z")
        proc_intr = proc.get("intrinsics")
        proc_pose = proc.get("camera_poses")

        raw_depth_np = _to_numpy_safe(raw_depth)
        if raw_depth_np is not None and raw_depth_np.ndim == 4:
            # tolerate [B,H,W,1] etc
            raw_depth_np = np.squeeze(raw_depth_np)

        depth_valid_ratio = None
        try:
            if raw_depth_np is not None and raw_depth_np.ndim >= 2:
                d = np.asarray(raw_depth_np, dtype=np.float64)
                d = d.reshape(d.shape[0], d.shape[1])
                valid = np.isfinite(d) & (d > depth_min) & (d < depth_max)
                depth_valid_ratio = float(valid.mean())
        except Exception:
            depth_valid_ratio = None

        one = {
            "i": i,
            "raw": {
                "filename": raw_fn,
                "img": _stat_np(_to_numpy_safe(raw_img)),
                "depth": _stat_np(raw_depth_np),
                "intrinsics": _stat_np(_to_numpy_safe(raw_intr)),
                "pose": _pose_stats(_to_numpy_safe(raw_pose)),
                "depth_valid_ratio": depth_valid_ratio,
            },
            "processed": {
                "img": _stat_np(_to_numpy_safe(proc_img)),
                "depth_z": _stat_np(_to_numpy_safe(proc_depth)),
                "intrinsics": _stat_np(_to_numpy_safe(proc_intr)),
                "camera_poses": _pose_stats(_to_numpy_safe(proc_pose)),
            },
        }
        views_out.append(one)

        if getattr(config, "debug_dump_save_npz", False):
            # 只保存少量关键张量，避免太大：img(第一张)、depth(2D)、K、pose
            try:
                npz_path = os.path.join(dump_dir, f"view_{i:02d}.npz")
                img_np = _to_numpy_safe(proc_img)
                if img_np is not None and img_np.ndim == 4:
                    img_np = img_np[0]
                dz_np = _to_numpy_safe(proc_depth)
                if dz_np is not None:
                    dz_np = np.squeeze(dz_np)
                np.savez_compressed(
                    npz_path,
                    img=img_np,
                    depth_z=dz_np,
                    intrinsics=_to_numpy_safe(proc_intr),
                    camera_poses=_to_numpy_safe(proc_pose),
                )
            except Exception:
                pass

    out_path = os.path.join(dump_dir, "debug_dump.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"meta": meta, "views": views_out}, f, ensure_ascii=False, indent=2)
    return dump_dir


def _write_hist_csv(path: str, hist: torch.Tensor, edges: torch.Tensor) -> None:
    """Write a histogram as CSV with [bin_left, bin_right, count]."""
    with open(path, "w", encoding="utf-8") as f:
        f.write("bin_left,bin_right,count\n")
        h = hist.detach().cpu().long().tolist()
        e = edges.detach().cpu().float().tolist()
        for i, count in enumerate(h):
            f.write(f"{e[i]:.8f},{e[i + 1]:.8f},{int(count)}\n")


def _write_optional_hist_png(path: str, values: torch.Tensor, title: str, xlabel: str) -> None:
    """Best-effort PNG histogram writer; silently skips if matplotlib is unavailable."""
    try:
        mpl_cache = "/tmp/ace_dinov2_lmc_matplotlib"
        os.makedirs(mpl_cache, exist_ok=True)
        os.environ.setdefault("MPLCONFIGDIR", mpl_cache)
        os.environ.setdefault("XDG_CACHE_HOME", mpl_cache)
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        vals = values.detach().cpu().float().numpy()
        if vals.size == 0:
            return
        plt.figure(figsize=(6, 4), dpi=160)
        plt.hist(vals, bins=80, range=(-1.0, 1.0), color="#3b6ea8", alpha=0.9)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel("count")
        plt.tight_layout()
        plt.savefig(path)
        plt.close()
    except Exception:
        return


def save_feature_similarity_diagnostics(
    output_run_dir: str,
    config: "ExtractionConfig",
    pooler: BSEPooler,
    points: torch.Tensor,
    features: torch.Tensor,
    view_ids: torch.Tensor,
) -> Optional[str]:
    """Save BSE-generation feature similarity diagnostics.

    This uses raw unprojected points/features before the final global pooling.
    Same-space pairs are approximated by membership in the same world-space
    voxel, and only cross-view pairs are counted for the main consistency test.
    """
    if not getattr(config, "feature_diagnostics", False):
        return None

    if points.numel() == 0 or features.numel() == 0 or view_ids.numel() == 0:
        return None

    diag_dir = os.path.join(output_run_dir, "feature_diagnostics")
    os.makedirs(diag_dir, exist_ok=True)

    with torch.no_grad():
        points = points.detach().float().cpu()
        features = features.detach().float().cpu()
        view_ids = view_ids.detach().long().cpu()

        cluster_ids = pooler._voxel_hash(points)
        voxel_means = pooler._scatter_mean(features, cluster_ids)
        sim_to_mean = pooler._cosine_similarity(features, voxel_means[cluster_ids]).clamp(-1.0, 1.0)

        bins = max(10, int(getattr(config, "feature_diag_bins", 100)))
        sim_edges = torch.linspace(-1.0, 1.0, bins + 1)
        sim_hist = torch.histc(sim_to_mean, bins=bins, min=-1.0, max=1.0)
        _write_hist_csv(os.path.join(diag_dir, "bse_feature_to_voxel_mean_similarity_hist.csv"), sim_hist, sim_edges)
        _write_optional_hist_png(
            os.path.join(diag_dir, "bse_feature_to_voxel_mean_similarity_hist.png"),
            sim_to_mean,
            "BSE feature-to-voxel-mean cosine similarity",
            "cosine similarity",
        )

        max_points_per_voxel = max(2, int(getattr(config, "feature_diag_max_points_per_voxel", 32)))
        max_pair_samples = max(1, int(getattr(config, "feature_diag_max_pair_samples", 200000)))
        feat_norm = F.normalize(features, dim=1, eps=1e-6)

        pair_sims: List[torch.Tensor] = []
        pair_dists: List[torch.Tensor] = []
        cross_view_voxels = 0
        total_voxels = int(cluster_ids.max().item()) + 1
        generator = torch.Generator(device="cpu")
        generator.manual_seed(0)

        for cid in range(total_voxels):
            idx = (cluster_ids == cid).nonzero(as_tuple=True)[0]
            n = int(idx.numel())
            if n < 2:
                continue
            if torch.unique(view_ids[idx]).numel() < 2:
                continue
            cross_view_voxels += 1
            if n > max_points_per_voxel:
                perm = torch.randperm(n, generator=generator)[:max_points_per_voxel]
                idx = idx[perm]
                n = int(idx.numel())

            row, col = torch.triu_indices(n, n, offset=1)
            keep = view_ids[idx[row]] != view_ids[idx[col]]
            if not bool(keep.any()):
                continue
            row = row[keep]
            col = col[keep]
            sims = (feat_norm[idx[row]] * feat_norm[idx[col]]).sum(dim=1).clamp(-1.0, 1.0)
            dists = torch.linalg.norm(points[idx[row]] - points[idx[col]], dim=1)
            pair_sims.append(sims.cpu())
            pair_dists.append(dists.cpu())

            if sum(int(x.numel()) for x in pair_sims) >= max_pair_samples:
                break

        if pair_sims:
            same_space_sims = torch.cat(pair_sims)[:max_pair_samples]
            same_space_dists = torch.cat(pair_dists)[:max_pair_samples]
        else:
            same_space_sims = torch.empty(0)
            same_space_dists = torch.empty(0)

        same_hist = torch.histc(same_space_sims, bins=bins, min=-1.0, max=1.0) if same_space_sims.numel() else torch.zeros(bins)
        _write_hist_csv(os.path.join(diag_dir, "same_voxel_cross_view_feature_similarity_hist.csv"), same_hist, sim_edges)
        _write_optional_hist_png(
            os.path.join(diag_dir, "same_voxel_cross_view_feature_similarity_hist.png"),
            same_space_sims,
            "Same-voxel cross-view feature cosine similarity",
            "cosine similarity",
        )

        dist_max = float(max(float(getattr(config, "voxel_size", 0.05)) * 2.0, same_space_dists.max().item() if same_space_dists.numel() else 0.1))
        dist_edges = torch.linspace(0.0, dist_max, bins + 1)
        dist_hist = torch.histc(same_space_dists, bins=bins, min=0.0, max=dist_max) if same_space_dists.numel() else torch.zeros(bins)
        _write_hist_csv(os.path.join(diag_dir, "same_voxel_cross_view_spatial_distance_hist.csv"), dist_hist, dist_edges)

        neg_count = int(min(max_pair_samples, max(1000, same_space_sims.numel())))
        if features.shape[0] >= 2 and neg_count > 0:
            a = torch.randint(0, features.shape[0], (neg_count,), generator=generator)
            b = torch.randint(0, features.shape[0], (neg_count,), generator=generator)
            neg_keep = cluster_ids[a] != cluster_ids[b]
            a = a[neg_keep]
            b = b[neg_keep]
            neg_sims = (feat_norm[a] * feat_norm[b]).sum(dim=1).clamp(-1.0, 1.0) if a.numel() else torch.empty(0)
        else:
            neg_sims = torch.empty(0)
        neg_hist = torch.histc(neg_sims, bins=bins, min=-1.0, max=1.0) if neg_sims.numel() else torch.zeros(bins)
        _write_hist_csv(os.path.join(diag_dir, "random_different_voxel_feature_similarity_hist.csv"), neg_hist, sim_edges)
        _write_optional_hist_png(
            os.path.join(diag_dir, "random_different_voxel_feature_similarity_hist.png"),
            neg_sims,
            "Random different-voxel feature cosine similarity",
            "cosine similarity",
        )

        def _summary(x: torch.Tensor) -> Dict[str, Any]:
            if x.numel() == 0:
                return {"count": 0}
            xf = x.float()
            qs = torch.quantile(xf, torch.tensor([0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]))
            return {
                "count": int(xf.numel()),
                "mean": float(xf.mean().item()),
                "std": float(xf.std(unbiased=False).item()),
                "min": float(xf.min().item()),
                "max": float(xf.max().item()),
                "q01": float(qs[0].item()),
                "q05": float(qs[1].item()),
                "q10": float(qs[2].item()),
                "q25": float(qs[3].item()),
                "q50": float(qs[4].item()),
                "q75": float(qs[5].item()),
                "q90": float(qs[6].item()),
                "q95": float(qs[7].item()),
                "q99": float(qs[8].item()),
            }

        summary = {
            "schema": 1,
            "definition": {
                "bse_feature_to_voxel_mean_similarity": "cos(feature_i, mean_feature_of_same_world_voxel)",
                "same_voxel_cross_view_feature_similarity": "cos(feature_i, feature_j) for raw points in the same world voxel but from different views",
                "random_different_voxel_feature_similarity": "random feature-pair baseline where points are in different voxels",
            },
            "config": {
                "dataset_loader": config.dataset_loader,
                "dataset_type": config.dataset_type,
                "scene_name": config.scene_name,
                "pool_mode": config.pool_mode,
                "prepool_mode": config.prepool_mode,
                "voxel_size": float(config.voxel_size),
                "use_model": config.use_model,
                "use_otsu": bool(config.use_otsu),
                "unimodal_threshold": float(config.unimodal_threshold),
                "hybrid_split_min_std": float(config.hybrid_split_min_std),
            },
            "raw_points": int(points.shape[0]),
            "feature_dim": int(features.shape[1]),
            "total_voxels": total_voxels,
            "cross_view_voxels": int(cross_view_voxels),
            "bse_otsu_tau_global": float(pooler._otsu_threshold(sim_to_mean, check_unimodal=True)),
            "bse_feature_to_voxel_mean_similarity": _summary(sim_to_mean),
            "same_voxel_cross_view_feature_similarity": _summary(same_space_sims),
            "same_voxel_cross_view_spatial_distance_m": _summary(same_space_dists),
            "random_different_voxel_feature_similarity": _summary(neg_sims),
        }

        with open(os.path.join(diag_dir, "feature_similarity_summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"[FeatureDiag] Saved feature similarity diagnostics to: {diag_dir}", flush=True)
    return diag_dir


def _extract_centers_from_pose_dir(dataset_path: str, scene_name: Optional[str] = None) -> Optional[np.ndarray]:
    """Fast path: read pose txt from disk into centers [N,3].

    Supports both layouts:
    - ACE: <dataset_path>/poses
    - WAI: <dataset_path>/<scene_name>/poses   (dataset_path is the ROOT)
    """
    from pathlib import Path

    root = Path(dataset_path)
    candidates: List[Path] = []
    # ACE style
    candidates.append(root / "poses")
    # WAI style (ROOT + scene subdir)
    if scene_name:
        candidates.append(root / scene_name / "poses")

    pose_dir = None
    for c in candidates:
        if c.exists():
            pose_dir = c
            break
    if pose_dir is None:
        return None
    pose_files = sorted([p for p in pose_dir.iterdir() if p.is_file()])
    if not pose_files:
        return None

    centers: List[np.ndarray] = []
    for pth in pose_files:
        try:
            pose = np.loadtxt(str(pth), dtype=np.float64)
        except Exception:
            continue
        c = _pose_to_center_np(pose)
        if c is not None:
            centers.append(c)
    if not centers:
        return None
    return np.stack(centers, axis=0)


def _extract_centers_from_wai_scene_meta(dataset_root: str, scene_name: str) -> Optional[np.ndarray]:
    """Fast path for WAI: read <dataset_root>/<scene_name>/scene_meta.json and extract camera centers.

    Indices must align with WAI sequential_view_mode ordering, which uses sorted(frame_names).
    We therefore sort by scene_meta['frame_names'] keys (fallback: frame['frame_name'] list).
    """
    from pathlib import Path

    meta_path = Path(dataset_root) / scene_name / "scene_meta.json"
    if not meta_path.exists():
        return None
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            scene_meta = json.load(f)
    except Exception:
        return None

    frames = scene_meta.get("frames", [])
    if not isinstance(frames, list) or not frames:
        return None

    # Map frame_name -> frame dict for stable ordering
    name_to_frame: Dict[str, Any] = {}
    for fr in frames:
        if isinstance(fr, dict) and "frame_name" in fr:
            name_to_frame[str(fr["frame_name"])] = fr

    frame_names = scene_meta.get("frame_names")
    if isinstance(frame_names, dict) and frame_names:
        ordered_names = sorted([str(k) for k in frame_names.keys()])
    else:
        ordered_names = sorted(list(name_to_frame.keys()))

    centers: List[np.ndarray] = []
    for nm in ordered_names:
        fr = name_to_frame.get(nm)
        if not isinstance(fr, dict):
            continue
        pose = fr.get("transform_matrix") or fr.get("extrinsics")
        if pose is None:
            continue
        c2w = np.asarray(pose, dtype=np.float64)
        if c2w.size == 16:
            c2w = c2w.reshape(4, 4)
        if c2w.shape != (4, 4):
            continue
        # WAI stores cam2world; camera center is translation component.
        c = c2w[:3, 3]
        if np.isfinite(c).all():
            centers.append(c.astype(np.float64))

    if not centers:
        return None
    return np.stack(centers, axis=0)


def _extract_centers_from_dataset_items(dataset) -> np.ndarray:
    """Slow fallback: iterate dataset and grab pose only (may still trigger IO in __getitem__)."""
    from tqdm import tqdm

    centers: List[np.ndarray] = []
    for idx in tqdm(range(len(dataset)), desc="Extracting camera centers (FPS)", leave=False):
        sample = dataset[idx]
        pose = None
        if isinstance(sample, dict):
            pose = sample.get("camera_pose", sample.get("pose"))
        elif isinstance(sample, tuple):
            # ACE CamLocDatasetDINOv2: (image, mask, pose, pose_inv, intrinsics, intrinsics_inv, coords, filename)
            if len(sample) > 2:
                pose = sample[2]
        elif isinstance(sample, list):
            # WAI: list[dict] views, take first
            if sample and isinstance(sample[0], dict):
                pose = sample[0].get("camera_pose", sample[0].get("pose"))
        c = _pose_to_center_np(pose)
        if c is not None:
            centers.append(c)

    if not centers:
        raise RuntimeError("FPS view selection failed: no valid camera poses found.")
    return np.stack(centers, axis=0)


def fps_select_views(centers: np.ndarray, n_select: int, seed: int = 42,
                     force_include: Optional[List[int]] = None) -> List[int]:
    """Furthest Point Sampling on camera centers for spatial coverage.

    Args:
        force_include: List of frame indices that MUST be in the selection.
            Typical use: ``force_include=[0]`` to guarantee the scene origin frame
            is selected, so that MapAnything's internal reference frame aligns
            with the world origin.
    """
    centers = np.asarray(centers, dtype=np.float64).reshape(-1, 3)
    n_total = int(centers.shape[0])
    if n_select >= n_total:
        return list(range(n_total))

    if force_include is None:
        force_include = []

    # Validate force_include
    for idx in force_include:
        if idx < 0 or idx >= n_total:
            raise ValueError(f"force_include index {idx} out of range [0, {n_total})")

    n_forced = min(len(force_include), n_select)
    selected: List[int] = list(force_include[:n_forced])

    # If all slots are forced, return early
    if len(selected) >= n_select:
        return sorted(selected[:n_select])

    # Initialize min_dist from all forced seeds
    min_dist = np.full(n_total, np.inf, dtype=np.float64)
    for s in selected:
        d = np.linalg.norm(centers - centers[s], axis=1)
        min_dist = np.minimum(min_dist, d)

    # Fill remaining slots with standard FPS
    while len(selected) < n_select:
        next_idx = int(np.argmax(min_dist))
        selected.append(next_idx)
        if len(selected) < n_select:
            d = np.linalg.norm(centers - centers[next_idx], axis=1)
            min_dist = np.minimum(min_dist, d)

    return selected


def _extract_centers_from_flat_wai_dataset(dataset) -> Optional[np.ndarray]:
    """Fast path aligned with WAI flat_view_list order, without image IO."""
    flat_view_list = getattr(dataset, "flat_view_list", None)
    scene_meta_cache = getattr(dataset, "scene_meta_cache", None)
    if not isinstance(flat_view_list, list) or scene_meta_cache is None:
        return None

    centers: List[np.ndarray] = []
    for item in flat_view_list:
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            return None
        scene_name, frame_name = str(item[0]), str(item[1])
        meta = scene_meta_cache.get(scene_name) if hasattr(scene_meta_cache, "get") else None
        if not isinstance(meta, dict):
            return None

        frame_data = None
        frame_names = meta.get("frame_names")
        if isinstance(frame_names, dict) and frame_name in frame_names:
            idx = int(frame_names[frame_name])
            frames = meta.get("frames", [])
            if 0 <= idx < len(frames):
                frame_data = frames[idx]
        if frame_data is None:
            for fr in meta.get("frames", []):
                if isinstance(fr, dict) and str(fr.get("frame_name")) == frame_name:
                    frame_data = fr
                    break
        if not isinstance(frame_data, dict):
            return None

        pose = frame_data.get("transform_matrix") or frame_data.get("extrinsics")
        c = _pose_to_center_np(pose)
        if c is None:
            return None
        centers.append(c)

    if not centers:
        return None
    return np.stack(centers, axis=0)


def _load_wai_pairwise_covisibility(
    dataset_path: Optional[str],
    scene_name: Optional[str],
    n_expected: Optional[int] = None,
) -> Optional[np.ndarray]:
    """Load WAI scene_root/covisibility/v0/*.npy as mmap when available."""
    if not dataset_path or not scene_name:
        return None

    covis_dir = os.path.join(str(dataset_path), str(scene_name), "covisibility", "v0")
    if not os.path.isdir(covis_dir):
        print(f"[CovisFPS] No covisibility directory found: {covis_dir}", flush=True)
        return None

    npy_names = sorted([n for n in os.listdir(covis_dir) if n.endswith(".npy")])
    if not npy_names:
        print(f"[CovisFPS] No .npy covisibility file found under: {covis_dir}", flush=True)
        return None

    covis_path = os.path.join(covis_dir, npy_names[0])
    try:
        covis = np.load(covis_path, mmap_mode="r")
    except Exception as e:
        print(f"[CovisFPS] Failed to load covisibility {covis_path} ({type(e).__name__}: {e})", flush=True)
        return None

    if covis.ndim != 2 or covis.shape[0] != covis.shape[1]:
        print(f"[CovisFPS] Invalid covisibility shape {getattr(covis, 'shape', None)} at {covis_path}", flush=True)
        return None
    if n_expected is not None and covis.shape[0] < int(n_expected):
        print(
            f"[CovisFPS] Covisibility shape {covis.shape} is smaller than expected n={n_expected}; skip.",
            flush=True,
        )
        return None
    if n_expected is not None and covis.shape[0] > int(n_expected):
        covis = covis[: int(n_expected), : int(n_expected)]

    print(f"[CovisFPS] Loaded covisibility: {covis_path}, shape={tuple(covis.shape)}", flush=True)
    return covis


def _print_selection_coverage_stats(centers: np.ndarray, selected: List[int], tag: str) -> None:
    try:
        centers_np = np.asarray(centers, dtype=np.float64).reshape(-1, 3)
        sel = centers_np[np.asarray(selected, dtype=np.int64)]
        dists = np.linalg.norm(centers_np[:, None, :] - sel[None, :, :], axis=2).min(axis=1)
        ref_dists = np.linalg.norm(sel - sel[0:1, :], axis=1)
        print(
            f"[{tag}] Coverage nearest-center distance: "
            f"mean={dists.mean():.4f}m, median={np.median(dists):.4f}m, "
            f"max={dists.max():.4f}m",
            flush=True,
        )
        print(
            f"[{tag}] Selected distance-to-reference: "
            f"mean={ref_dists.mean():.4f}m, median={np.median(ref_dists):.4f}m, "
            f"max={ref_dists.max():.4f}m",
            flush=True,
        )
    except Exception as e:
        print(f"[{tag}] Coverage stats skipped ({type(e).__name__}: {e})", flush=True)


def _selection_coverage_summary(
    centers: np.ndarray,
    selected: List[int],
    ref_idx: Optional[int] = None,
    covisibility: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    centers_np = np.asarray(centers, dtype=np.float64).reshape(-1, 3)
    n_total = int(centers_np.shape[0])
    valid_selected: List[int] = []
    seen = set()
    for idx in selected:
        idx = int(idx)
        if 0 <= idx < n_total and idx not in seen:
            valid_selected.append(idx)
            seen.add(idx)
    if not valid_selected:
        raise ValueError("selection coverage summary requires at least one valid selected index")

    selected_arr = np.asarray(valid_selected, dtype=np.int64)
    ref = int(ref_idx) if ref_idx is not None and 0 <= int(ref_idx) < n_total else int(selected_arr[0])
    dmat = np.linalg.norm(centers_np[:, None, :] - centers_np[selected_arr][None, :, :], axis=2)
    nearest = dmat.min(axis=1)
    nearest_pos = dmat.argmin(axis=1)
    nearest_selected = selected_arr[nearest_pos]
    ref_dists = np.linalg.norm(centers_np[selected_arr] - centers_np[ref:ref + 1], axis=1)

    def _q(values: np.ndarray, q: float) -> float:
        return _percentile_or_default(values, q, 0.0)

    worst_order = np.argsort(-nearest)[: min(20, n_total)]
    coverage = {
        "mean": float(np.nanmean(nearest)),
        "median": _q(nearest, 50),
        "p80": _q(nearest, 80),
        "p90": _q(nearest, 90),
        "p95": _q(nearest, 95),
        "p99": _q(nearest, 99),
        "max": float(np.nanmax(nearest)),
        "worst_uncovered": [
            {
                "frame_idx": int(i),
                "nearest_selected_idx": int(nearest_selected[int(i)]),
                "distance_m": float(nearest[int(i)]),
            }
            for i in worst_order
        ],
    }
    ref_summary = {
        "mean": float(np.nanmean(ref_dists)),
        "median": _q(ref_dists, 50),
        "p90": _q(ref_dists, 90),
        "max": float(np.nanmax(ref_dists)),
    }

    X = centers_np - centers_np.mean(axis=0, keepdims=True)
    try:
        _, _, vh = np.linalg.svd(X, full_matrices=False)
        uv = X @ vh[:2].T
    except Exception:
        uv = centers_np[:, :2]
    mins = uv.min(axis=0)
    maxs = uv.max(axis=0)
    span = np.maximum(maxs - mins, 1e-9)
    grid = np.floor((uv - mins) / span * 8.0).astype(np.int64)
    grid = np.clip(grid, 0, 7)
    all_cells = {tuple(x) for x in grid}
    selected_cells = {tuple(x) for x in grid[selected_arr]}
    grid_summary = {
        "grid_size": 8,
        "selected_cells": int(len(selected_cells)),
        "occupied_cells": int(len(all_cells)),
        "occupancy_ratio": float(len(selected_cells) / max(len(all_cells), 1)),
    }

    out: Dict[str, Any] = {
        "num_scene_views": n_total,
        "num_selected_views": int(len(selected_arr)),
        "selected_indices": [int(i) for i in selected_arr.tolist()],
        "reference_index": int(ref),
        "coverage": coverage,
        "ref_distance": ref_summary,
        "pca_grid_occupancy": grid_summary,
    }

    if covisibility is not None:
        covis = np.asarray(covisibility)
        if covis.ndim == 2 and covis.shape[0] >= n_total and covis.shape[1] >= n_total:
            cov = np.maximum(covis[:n_total, :n_total], covis[:n_total, :n_total].T).astype(np.float64, copy=True)
            np.fill_diagonal(cov, 0.0)
            positive = cov[np.isfinite(cov) & (cov > 0)]
            if positive.size:
                tau = _percentile_or_default(positive, 95, float(np.nanmin(positive)))
                adj = cov[np.ix_(selected_arr, selected_arr)] >= tau
                seen_nodes = set()
                components: List[int] = []
                for start in range(len(selected_arr)):
                    if start in seen_nodes:
                        continue
                    stack = [start]
                    seen_nodes.add(start)
                    comp_size = 0
                    while stack:
                        node = stack.pop()
                        comp_size += 1
                        for nxt in np.flatnonzero(adj[node]):
                            nxt = int(nxt)
                            if nxt not in seen_nodes:
                                seen_nodes.add(nxt)
                                stack.append(nxt)
                    components.append(comp_size)
                degrees = adj.sum(axis=1).astype(np.int64)
                out["strong_covis_graph"] = {
                    "tau_q95": float(tau),
                    "component_sizes": sorted([int(c) for c in components], reverse=True),
                    "num_components": int(len(components)),
                    "degree_min": int(degrees.min()) if degrees.size else 0,
                    "degree_median": float(np.median(degrees)) if degrees.size else 0.0,
                    "degree_max": int(degrees.max()) if degrees.size else 0,
                    "isolated_count": int((degrees == 0).sum()) if degrees.size else 0,
                }

    return out


def _export_selection_coverage_diagnostics(
    dataset: Any,
    dataset_path: Optional[str],
    scene_name: Optional[str],
    selected: List[int],
    output_run_dir: str,
    max_views: int,
    adaptive_min_views: int,
    tag: str = "SelectionCoverage",
) -> None:
    """Save geometry-only selection coverage diagnostics without changing selection."""
    try:
        centers = _extract_centers_from_flat_wai_dataset(dataset)
        if centers is None and dataset_path is not None and scene_name:
            centers = _extract_centers_from_wai_scene_meta(dataset_path, scene_name)
        if centers is None and dataset_path is not None:
            centers = _extract_centers_from_pose_dir(dataset_path, scene_name=scene_name)
        if centers is None:
            centers = _extract_centers_from_dataset_items(dataset)
        if centers is None or len(centers) == 0:
            print(f"[{tag}] Skipped: no camera centers available.", flush=True)
            return

        ref_idx = int(selected[0]) if selected else 0
        covis = _load_wai_pairwise_covisibility(dataset_path, scene_name, n_expected=len(centers))
        summary = _selection_coverage_summary(centers, selected, ref_idx=ref_idx, covisibility=covis)

        fps_same = fps_select_views(centers, len(selected), force_include=[ref_idx])
        fps_max = fps_select_views(centers, int(max_views), force_include=[ref_idx])
        summary["baselines"] = {
            "fps_same_count": _selection_coverage_summary(centers, fps_same, ref_idx=ref_idx),
            "fps_max_views": _selection_coverage_summary(centers, fps_max, ref_idx=ref_idx),
        }

        dist_to_ref = np.linalg.norm(np.asarray(centers, dtype=np.float64) - centers[ref_idx:ref_idx + 1], axis=1)
        positive_ref = dist_to_ref[np.isfinite(dist_to_ref) & (dist_to_ref > 0)]
        ref_q80 = _percentile_or_default(positive_ref, 80, _percentile_or_default(positive_ref, 50, 1.0))
        target_mean, target_max, _ = _reference_safe_fps_coverage(
            centers,
            ref_idx,
            adaptive_min_views,
            ref_q80,
        )
        targets = {
            "source": f"ref_safe_fps_at_{int(adaptive_min_views)}",
            "ref_dist_q80_m": float(ref_q80),
            "coverage_mean_limit_m": float(target_mean * 1.10),
            "coverage_max_limit_m": float(target_max * 1.25),
            "coverage_p95_advisory_limit_m": float(target_max * 1.25 * 0.75),
        }
        cov = summary["coverage"]
        checks = {
            "coverage_mean_ok": bool(cov["mean"] <= targets["coverage_mean_limit_m"]),
            "coverage_max_ok": bool(cov["max"] <= targets["coverage_max_limit_m"]),
            "coverage_p95_ok": bool(cov["p95"] <= targets["coverage_p95_advisory_limit_m"]),
        }
        checks["status"] = "OK" if all(checks.values()) else ("WARN" if checks["coverage_mean_ok"] and checks["coverage_max_ok"] else "FAIL")
        summary["adaptive_coverage_targets"] = targets
        summary["coverage_checks"] = checks

        os.makedirs(output_run_dir, exist_ok=True)
        report_path = os.path.join(output_run_dir, "selection_coverage_report.json")
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        _save_selection_coverage_topdown(centers, selected, summary, output_run_dir)

        fps_same_cov = summary["baselines"]["fps_same_count"]["coverage"]
        print(
            f"[{tag}] Report saved: {report_path}; status={checks['status']}, "
            f"selected={summary['num_selected_views']}/{int(max_views)}, "
            f"coverage mean/p95/max={cov['mean']:.4f}/{cov['p95']:.4f}/{cov['max']:.4f}m, "
            f"limits mean/p95/max={targets['coverage_mean_limit_m']:.4f}/"
            f"{targets['coverage_p95_advisory_limit_m']:.4f}/{targets['coverage_max_limit_m']:.4f}m, "
            f"fps_same mean/max={fps_same_cov['mean']:.4f}/{fps_same_cov['max']:.4f}m",
            flush=True,
        )
    except Exception as e:
        print(f"[{tag}] Diagnostics skipped ({type(e).__name__}: {e})", flush=True)


def _save_selection_coverage_topdown(
    centers: np.ndarray,
    selected: List[int],
    summary: Dict[str, Any],
    output_run_dir: str,
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[SelectionCoverage] Top-down plot skipped ({type(e).__name__}: {e})", flush=True)
        return

    centers_np = np.asarray(centers, dtype=np.float64).reshape(-1, 3)
    selected_arr = np.asarray(summary.get("selected_indices", selected), dtype=np.int64)
    X = centers_np - centers_np.mean(axis=0, keepdims=True)
    try:
        _, _, vh = np.linalg.svd(X, full_matrices=False)
        uv = X @ vh[:2].T
    except Exception:
        uv = centers_np[:, :2]

    sel = centers_np[selected_arr]
    nearest = np.linalg.norm(centers_np[:, None, :] - sel[None, :, :], axis=2).min(axis=1)
    worst = [int(x["frame_idx"]) for x in summary.get("coverage", {}).get("worst_uncovered", [])[:10]]

    fig, ax = plt.subplots(figsize=(8, 7), dpi=160)
    sc = ax.scatter(uv[:, 0], uv[:, 1], c=nearest, s=4, cmap="viridis", alpha=0.85, linewidths=0)
    ax.scatter(uv[selected_arr, 0], uv[selected_arr, 1], s=42, c="#d62728", marker="x", linewidths=1.3, label="selected")
    if worst:
        worst_arr = np.asarray(worst, dtype=np.int64)
        ax.scatter(uv[worst_arr, 0], uv[worst_arr, 1], s=36, facecolors="none", edgecolors="#1f77b4", linewidths=1.0, label="worst uncovered")
        for idx in worst[:5]:
            ax.text(uv[idx, 0], uv[idx, 1], str(idx), fontsize=6, color="#1f77b4")
    ax.set_title("Selection Coverage Top-Down PCA")
    ax.set_xlabel("PCA-1")
    ax.set_ylabel("PCA-2")
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="best", fontsize=7)
    cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("distance to nearest selected view (m)")
    fig.tight_layout()
    out_path = os.path.join(output_run_dir, "selection_coverage_topdown.png")
    fig.savefig(out_path)
    plt.close(fig)
    print(f"[SelectionCoverage] Top-down plot saved: {out_path}", flush=True)


def _coverage_gain_scores(
    centers: np.ndarray,
    current_nearest: np.ndarray,
    candidates: np.ndarray,
) -> np.ndarray:
    """Mean nearest-center coverage improvement if each candidate is added."""
    if len(candidates) > 128:
        gains = np.zeros(len(candidates), dtype=np.float64)
        chunk_size = 256
        for start in range(0, len(candidates), chunk_size):
            chunk = candidates[start:start + chunk_size].astype(np.int64)
            d = np.linalg.norm(centers[:, None, :] - centers[chunk][None, :, :], axis=2)
            improved = np.maximum(current_nearest[:, None] - np.minimum(current_nearest[:, None], d), 0.0)
            gains[start:start + len(chunk)] = improved.mean(axis=0)
        return gains

    gains = np.zeros(len(candidates), dtype=np.float64)
    for j, idx in enumerate(candidates):
        d = np.linalg.norm(centers - centers[int(idx)], axis=1)
        gains[j] = float(np.maximum(current_nearest - np.minimum(current_nearest, d), 0.0).mean())
    return gains


def _percentile_or_default(values: np.ndarray, q: float, default: float) -> float:
    vals = np.asarray(values, dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return float(default)
    out = float(np.nanpercentile(vals, float(q)))
    return out if np.isfinite(out) else float(default)


def _reference_safe_fps_coverage(
    centers: np.ndarray,
    root: int,
    min_views: int,
    ref_dist_limit: float,
) -> Tuple[float, float, float]:
    """Estimate a scene-local coverage target from pose statistics only."""
    n_total = int(centers.shape[0])
    min_views = max(1, min(int(min_views), n_total))
    dist_to_ref = np.linalg.norm(centers - centers[int(root)], axis=1).astype(np.float64)
    allowed = np.isfinite(dist_to_ref) & (dist_to_ref <= float(ref_dist_limit))
    allowed[int(root)] = False

    selected_mask = np.zeros(n_total, dtype=bool)
    selected_mask[int(root)] = True
    current_nearest = np.linalg.norm(centers - centers[int(root)], axis=1).astype(np.float64)
    gains: List[float] = []

    while int(selected_mask.sum()) < min_views:
        candidate_mask = allowed & (~selected_mask)
        if not candidate_mask.any():
            candidate_mask = ~selected_mask
        candidates = np.flatnonzero(candidate_mask)
        if candidates.size == 0:
            break
        next_idx = int(candidates[int(np.argmax(current_nearest[candidates]))])
        new_nearest = np.minimum(current_nearest, np.linalg.norm(centers - centers[next_idx], axis=1))
        gains.append(float(np.maximum(current_nearest - new_nearest, 0.0).mean()))
        selected_mask[next_idx] = True
        current_nearest = new_nearest

    mean_cov = float(np.nanmean(current_nearest)) if current_nearest.size else 0.0
    max_cov = float(np.nanmax(current_nearest)) if current_nearest.size else 0.0
    recent = np.asarray(gains[-3:], dtype=np.float64)
    recent = recent[np.isfinite(recent) & (recent > 0)]
    gain_floor = float(np.nanmedian(recent) * 0.5) if recent.size else mean_cov * 0.015
    if not np.isfinite(gain_floor) or gain_floor <= 0:
        gain_floor = max(mean_cov * 0.015, 1e-6)
    return mean_cov, max_cov, gain_floor


def _estimate_adaptive_asb_scene_params(
    centers: np.ndarray,
    covis_strength: np.ndarray,
    root: int,
    n_select: int,
    adaptive_min_views: int,
    support_min_neighbors: int,
    log_tag: str,
) -> Dict[str, float]:
    """
    Derive adaptive ASB thresholds from the current scene's pose and covisibility
    distributions instead of using scene-specific meter-scale constants.
    """
    dist_to_ref = np.linalg.norm(centers - centers[int(root)], axis=1).astype(np.float64)
    positive_ref = dist_to_ref[np.isfinite(dist_to_ref) & (dist_to_ref > 0)]
    if positive_ref.size == 0:
        return {}

    ref_q50 = _percentile_or_default(positive_ref, 50, 1.0)
    ref_q80 = _percentile_or_default(positive_ref, 80, ref_q50)
    ref_q90 = _percentile_or_default(positive_ref, 90, ref_q80)
    ref_mean_local = float(np.nanmean(positive_ref[positive_ref <= ref_q80]))
    if not np.isfinite(ref_mean_local) or ref_mean_local <= 0:
        ref_mean_local = float(np.nanmean(positive_ref))

    cov_mean_at_min, cov_max_at_min, gain_floor = _reference_safe_fps_coverage(
        centers,
        root,
        adaptive_min_views,
        ref_q80,
    )

    positive_covis = covis_strength[np.isfinite(covis_strength) & (covis_strength > 0)]
    covis_tau = -1.0
    support_tau = 0.0
    if positive_covis.size:
        covis_tau = _percentile_or_default(positive_covis, 5, float(np.nanmin(positive_covis)))
        min_supportable = max(int(n_select), int(adaptive_min_views) * 2)
        for q in (95, 90, 75, 50, 25, 5):
            cand_tau = _percentile_or_default(positive_covis, q, float(np.nanmin(positive_covis)))
            support_counts = (covis_strength >= cand_tau).sum(axis=1).astype(np.int64)
            if int((support_counts >= int(support_min_neighbors)).sum()) >= min_supportable:
                support_tau = float(cand_tau)
                break
        if support_tau <= 0:
            support_tau = _percentile_or_default(positive_covis, 5, float(np.nanmin(positive_covis)))

    params = {
        "safe_dist_to_ref": float(ref_q50),
        "ref_dist_max_limit": float(ref_q80),
        "ref_dist_mean_limit": float(ref_mean_local),
        "target_coverage_mean": float(cov_mean_at_min * 1.10),
        "target_coverage_max": float(cov_max_at_min * 1.25),
        "min_coverage_gain": float(max(gain_floor, cov_mean_at_min * 0.01, 1e-6)),
        "covis_tau": float(covis_tau),
        "support_tau": float(support_tau),
    }

    print(
        f"[{log_tag}:SceneStats] pose ref_dist: q50={ref_q50:.4f}m, "
        f"q80={ref_q80:.4f}m, q90={ref_q90:.4f}m, local_mean<=q80={ref_mean_local:.4f}m",
        flush=True,
    )
    print(
        f"[{log_tag}:SceneStats] coverage target from ref-safe FPS@{adaptive_min_views}: "
        f"mean={params['target_coverage_mean']:.4f}m, max={params['target_coverage_max']:.4f}m, "
        f"min_gain={params['min_coverage_gain']:.4f}m",
        flush=True,
    )
    if positive_covis.size:
        cov_q05 = _percentile_or_default(positive_covis, 5, params["covis_tau"])
        cov_q95 = _percentile_or_default(positive_covis, 95, params["support_tau"])
        print(
            f"[{log_tag}:SceneStats] covis positive: q05={cov_q05:.6g}, "
            f"q95={cov_q95:.6g}, auto_tau={params['covis_tau']:.6g}, "
            f"auto_support_tau={params['support_tau']:.6g}",
            flush=True,
        )
    else:
        print(f"[{log_tag}:SceneStats] covis positive values unavailable; keep covis thresholds disabled.", flush=True)

    return params


def _prune_candidate_pool_to_budget(
    centers: np.ndarray,
    covis_strength: np.ndarray,
    candidate_order: List[int],
    target_select: int,
    root: int,
    support_tau: float,
    alpha: float,
    coverage_beta: float,
    eps: float,
    log_tag: str,
    adaptive_count: bool = False,
    min_select: Optional[int] = None,
    coverage_stop_mean: Optional[float] = None,
    coverage_stop_max: Optional[float] = None,
    coverage_repair_order: Optional[List[int]] = None,
    force_target_count: bool = False,
    coverage_first: bool = False,
) -> List[int]:
    """Prune an oversampled ASB candidate pool to the final inference budget.

    The pruning is greedy over the candidate pool, not over the whole dataset:
    coverage gain keeps spatial support, covisibility-to-selected discourages
    isolated views, and a local repair pass swaps low-connectivity views with
    nearby alternatives from the oversampled pool when that preserves coverage.
    """
    if len(candidate_order) <= target_select:
        return candidate_order[:target_select]

    target_select = max(1, min(int(target_select), len(candidate_order)))
    adaptive_count = bool(adaptive_count)
    if min_select is None:
        min_select = target_select
    min_select = max(1, min(int(min_select), target_select))
    if coverage_stop_mean is not None:
        coverage_stop_mean = float(coverage_stop_mean)
        if not np.isfinite(coverage_stop_mean) or coverage_stop_mean <= 0:
            coverage_stop_mean = None
    if coverage_stop_max is not None:
        coverage_stop_max = float(coverage_stop_max)
        if not np.isfinite(coverage_stop_max) or coverage_stop_max <= 0:
            coverage_stop_max = None
    root = int(root)
    candidates = []
    seen = set()
    for idx in candidate_order:
        idx = int(idx)
        if idx not in seen:
            candidates.append(idx)
            seen.add(idx)
    if root not in seen:
        candidates.insert(0, root)
    repair_candidates = list(candidates)
    if coverage_repair_order is not None:
        repair_seen = set(repair_candidates)
        for idx in coverage_repair_order:
            idx = int(idx)
            if idx not in repair_seen:
                repair_candidates.append(idx)
                repair_seen.add(idx)

    selected: List[int] = [root]
    selected_set = {root}
    current_nearest = np.linalg.norm(centers - centers[root], axis=1).astype(np.float64)
    covis_scale = float(np.nanmax(covis_strength[np.ix_(candidates, candidates)])) if candidates else 0.0
    if not np.isfinite(covis_scale) or covis_scale <= 0:
        covis_scale = 1.0

    strong_threshold = float(support_tau) if np.isfinite(support_tau) and support_tau > 0 else 0.0

    def _recompute_nearest(sel: List[int]) -> np.ndarray:
        if not sel:
            return np.full(centers.shape[0], np.inf, dtype=np.float64)
        sel_arr = np.asarray(sel, dtype=np.int64)
        return np.linalg.norm(centers[:, None, :] - centers[sel_arr][None, :, :], axis=2).min(axis=1)

    def _disconnected_count(sel: List[int]) -> int:
        if len(sel) <= 1 or strong_threshold <= 0:
            return 0
        count = 0
        sel_arr = np.asarray(sel, dtype=np.int64)
        for idx in sel:
            if int(idx) == root:
                continue
            others = sel_arr[sel_arr != int(idx)]
            if others.size == 0:
                count += 1
                continue
            conn = float(np.nanmax(covis_strength[int(idx), others]))
            if not np.isfinite(conn) or conn < strong_threshold:
                count += 1
        return count

    while len(selected) < target_select:
        remaining = np.asarray([idx for idx in candidates if idx not in selected_set], dtype=np.int64)
        if remaining.size == 0:
            break
        gains = _coverage_gain_scores(centers, current_nearest, remaining)
        max_gain = float(np.nanmax(gains)) if gains.size else 0.0
        if not np.isfinite(max_gain) or max_gain <= 0:
            max_gain = 1.0
        gain_norm = gains / max_gain

        selected_arr = np.asarray(selected, dtype=np.int64)
        conn = np.nanmax(covis_strength[np.ix_(remaining, selected_arr)], axis=1).astype(np.float64)
        conn[~np.isfinite(conn)] = 0.0
        conn_norm = np.clip(conn / covis_scale, 0.0, None)

        if coverage_first:
            score = np.power(eps + gain_norm, max(float(coverage_beta), 0.0))
            if alpha > 0:
                score = score + 0.10 * min(float(alpha), 2.0) * conn_norm
        else:
            score = np.power(eps + gain_norm, max(float(coverage_beta), 0.0)) * np.power(
                eps + conn_norm,
                max(float(alpha), 0.0),
            )
        next_idx = int(remaining[int(np.argmax(score))])
        selected.append(next_idx)
        selected_set.add(next_idx)
        current_nearest = np.minimum(current_nearest, np.linalg.norm(centers - centers[next_idx], axis=1))
        if adaptive_count and len(selected) >= min_select:
            cov_mean = float(np.nanmean(current_nearest)) if current_nearest.size else 0.0
            cov_max = float(np.nanmax(current_nearest)) if current_nearest.size else 0.0
            disconnected = _disconnected_count(selected)
            disconnected_limit = max(1, int(np.ceil(0.15 * float(max(len(selected) - 1, 1)))))
            coverage_ok = (
                (coverage_stop_mean is None or cov_mean <= coverage_stop_mean)
                and (coverage_stop_max is None or cov_max <= coverage_stop_max)
            )
            if disconnected <= disconnected_limit and coverage_ok:
                print(
                    f"[{log_tag}] Adaptive count stop: views={len(selected)}, "
                    f"coverage_mean={cov_mean:.4f}m, coverage_max={cov_max:.4f}m, "
                    f"disconnected={disconnected}/{disconnected_limit}, "
                    f"coverage_stop_mean={coverage_stop_mean if coverage_stop_mean is not None else -1:.4f}m, "
                    f"coverage_stop_max={coverage_stop_max if coverage_stop_max is not None else -1:.4f}m",
                    flush=True,
                )
                break

    if len(selected) < target_select and not adaptive_count:
        for idx in candidates:
            if idx not in selected_set:
                selected.append(int(idx))
                selected_set.add(int(idx))
            if len(selected) >= target_select:
                break

    positive_covis = covis_strength[np.isfinite(covis_strength) & (covis_strength > 0)]
    conn_threshold = float(support_tau) if np.isfinite(support_tau) and support_tau > 0 else 0.0
    if conn_threshold <= 0 and positive_covis.size:
        conn_threshold = _percentile_or_default(positive_covis, 5, 0.0)

    candidate_arr = np.asarray(candidates, dtype=np.int64)
    pool_centers = centers[candidate_arr]
    if len(candidate_arr) > 1:
        pair_d = np.linalg.norm(pool_centers[:, None, :] - pool_centers[None, :, :], axis=2)
        pair_d[pair_d <= 0] = np.nan
        nn_d = np.nanmin(pair_d, axis=1)
        nn_d = nn_d[np.isfinite(nn_d) & (nn_d > 0)]
        local_radius = float(np.nanpercentile(nn_d, 75) * 3.0) if nn_d.size else 0.0
    else:
        local_radius = 0.0
    if not np.isfinite(local_radius) or local_radius <= 0:
        local_radius = 0.25

    current_nearest = _recompute_nearest(selected)
    old_cov_mean = float(np.nanmean(current_nearest)) if current_nearest.size else 0.0
    repairs = 0
    for pos, idx in list(enumerate(selected)):
        if idx == root or len(selected) <= 1:
            continue
        others = [j for j in selected if j != idx]
        if not others:
            continue
        curr_conn = float(np.nanmax(covis_strength[int(idx), np.asarray(others, dtype=np.int64)]))
        if np.isfinite(curr_conn) and curr_conn >= conn_threshold:
            continue

        unselected = [j for j in candidates if j not in selected_set]
        if not unselected:
            continue
        dist_local = np.linalg.norm(centers[np.asarray(unselected, dtype=np.int64)] - centers[int(idx)], axis=1)
        local = [j for j, d in zip(unselected, dist_local) if float(d) <= local_radius]
        if not local:
            continue

        others_arr = np.asarray(others, dtype=np.int64)
        best_j = None
        best_score = curr_conn if np.isfinite(curr_conn) else 0.0
        best_cov = old_cov_mean
        for cand in local:
            cand_conn = float(np.nanmax(covis_strength[int(cand), others_arr]))
            if not np.isfinite(cand_conn) or cand_conn <= best_score:
                continue
            trial = others + [int(cand)]
            trial_centers = centers[np.asarray(trial, dtype=np.int64)]
            trial_cov = np.linalg.norm(centers[:, None, :] - trial_centers[None, :, :], axis=2).min(axis=1)
            trial_cov_mean = float(np.nanmean(trial_cov))
            if trial_cov_mean <= old_cov_mean * 1.05:
                best_j = int(cand)
                best_score = cand_conn
                best_cov = trial_cov_mean
        if best_j is not None:
            selected_set.remove(int(idx))
            selected[pos] = int(best_j)
            selected_set.add(int(best_j))
            current_nearest = _recompute_nearest(selected)
            old_cov_mean = best_cov
            repairs += 1

    coverage_repairs = 0
    if adaptive_count and coverage_stop_max is not None:
        tail_stop = float(coverage_stop_max) * 0.65
        if coverage_stop_mean is not None:
            tail_stop = max(tail_stop, float(coverage_stop_mean) * 1.8)
        while len(selected) < target_select:
            current_nearest = _recompute_nearest(selected)
            cov_max = float(np.nanmax(current_nearest)) if current_nearest.size else 0.0
            cov_p95 = _percentile_or_default(current_nearest, 95, cov_max)
            cov_p99 = _percentile_or_default(current_nearest, 99, cov_max)
            if not np.isfinite(cov_max):
                break
            if cov_max <= float(coverage_stop_max) and cov_p99 <= tail_stop:
                break

            unselected = [j for j in repair_candidates if j not in selected_set]
            if not unselected:
                break
            remaining = np.asarray(unselected, dtype=np.int64)
            selected_arr = np.asarray(selected, dtype=np.int64)
            conn = np.nanmax(covis_strength[np.ix_(remaining, selected_arr)], axis=1).astype(np.float64)
            conn[~np.isfinite(conn)] = 0.0

            worst_count = min(32, max(1, int(np.ceil(0.01 * float(centers.shape[0])))))
            worst_idx = np.argsort(-current_nearest)[:worst_count].astype(np.int64)
            d_worst = np.linalg.norm(centers[remaining, None, :] - centers[worst_idx][None, :, :], axis=2)
            new_worst_nearest = np.minimum(current_nearest[worst_idx][None, :], d_worst)
            worst_gain = (current_nearest[worst_idx][None, :] - new_worst_nearest).mean(axis=1)
            all_gain = _coverage_gain_scores(centers, current_nearest, remaining)
            tail_threshold = min(cov_p95, tail_stop)
            tail_idx = np.flatnonzero(current_nearest >= tail_threshold).astype(np.int64)
            if tail_idx.size == 0:
                tail_idx = worst_idx

            worst_frame = int(np.nanargmax(current_nearest))
            cluster_radius = max(float(coverage_stop_mean or 0.0), float(coverage_stop_max) * 0.35)
            if not np.isfinite(cluster_radius) or cluster_radius <= 0:
                cluster_radius = max(float(cov_p95), 0.5)
            dist_to_worst = np.linalg.norm(centers - centers[worst_frame], axis=1)
            cluster_idx = np.flatnonzero((dist_to_worst <= cluster_radius) & (current_nearest >= tail_threshold)).astype(np.int64)
            if cluster_idx.size < 8:
                tail_order = np.argsort(dist_to_worst[tail_idx])[: min(64, len(tail_idx))]
                cluster_idx = tail_idx[tail_order].astype(np.int64)
            d_cluster = np.linalg.norm(centers[remaining, None, :] - centers[cluster_idx][None, :, :], axis=2)
            new_cluster_nearest = np.minimum(current_nearest[cluster_idx][None, :], d_cluster)
            cluster_current_max = float(np.nanmax(current_nearest[cluster_idx])) if cluster_idx.size else cov_max
            cluster_new_max = np.nanmax(new_cluster_nearest, axis=1)
            cluster_max_reduction = cluster_current_max - cluster_new_max
            cluster_gain = (current_nearest[cluster_idx][None, :] - new_cluster_nearest).mean(axis=1)

            new_max = np.empty(len(remaining), dtype=np.float64)
            chunk_size = 256
            for start in range(0, len(remaining), chunk_size):
                chunk = remaining[start:start + chunk_size]
                d_all = np.linalg.norm(centers[:, None, :] - centers[chunk][None, :, :], axis=2)
                new_nearest = np.minimum(current_nearest[:, None], d_all)
                new_max[start:start + len(chunk)] = np.nanmax(new_nearest, axis=0)
            max_reduction = cov_max - new_max
            d_tail = np.linalg.norm(centers[remaining, None, :] - centers[tail_idx][None, :, :], axis=2)
            new_tail_nearest = np.minimum(current_nearest[tail_idx][None, :], d_tail)
            tail_gain = (current_nearest[tail_idx][None, :] - new_tail_nearest).mean(axis=1)
            improves_tail = np.isfinite(tail_gain) & (tail_gain > 1e-6)
            improves_max = np.isfinite(max_reduction) & (max_reduction > 1e-6)
            improves_cluster = (
                (np.isfinite(cluster_max_reduction) & (cluster_max_reduction > 1e-6))
                | (np.isfinite(cluster_gain) & (cluster_gain > 1e-6))
            )
            improves_any = improves_max | improves_tail | improves_cluster
            if not improves_any.any():
                print(
                    f"[{log_tag}] Coverage-hole repair stopped: no ref-safe candidate reduces "
                    f"coverage tail (max={cov_max:.4f}m, p99={cov_p99:.4f}m, "
                    f"tail_stop={tail_stop:.4f}m).",
                    flush=True,
                )
                break

            wg_max = float(np.nanmax(worst_gain)) if worst_gain.size else 0.0
            ag_max = float(np.nanmax(all_gain)) if all_gain.size else 0.0
            mr_max = float(np.nanmax(max_reduction[improves_max])) if improves_max.any() else 0.0
            tg_max = float(np.nanmax(tail_gain[improves_tail])) if improves_tail.any() else 0.0
            cmr_max = float(np.nanmax(cluster_max_reduction[np.isfinite(cluster_max_reduction)])) if cluster_max_reduction.size else 0.0
            cg_max = float(np.nanmax(cluster_gain[np.isfinite(cluster_gain)])) if cluster_gain.size else 0.0
            if not np.isfinite(wg_max) or wg_max <= 0:
                wg_max = 1.0
            if not np.isfinite(ag_max) or ag_max <= 0:
                ag_max = 1.0
            if not np.isfinite(mr_max) or mr_max <= 0:
                mr_max = 1.0
            if not np.isfinite(tg_max) or tg_max <= 0:
                tg_max = 1.0
            if not np.isfinite(cmr_max) or cmr_max <= 0:
                cmr_max = 1.0
            if not np.isfinite(cg_max) or cg_max <= 0:
                cg_max = 1.0
            conn_norm = np.clip(conn / covis_scale, 0.0, None)
            conn_bonus = np.clip(conn_norm, 0.0, 2.0)
            score = (
                3.0 * np.maximum(cluster_max_reduction, 0.0) / cmr_max
                + 2.0 * np.maximum(cluster_gain, 0.0) / cg_max
                + 2.0 * np.maximum(max_reduction, 0.0) / mr_max
                + 1.5 * np.maximum(tail_gain, 0.0) / tg_max
                + worst_gain / wg_max
                + 0.25 * (all_gain / ag_max)
                + 0.10 * conn_bonus
            )
            score[~improves_any] = -np.inf
            connected_mask = conn >= conn_threshold if conn_threshold > 0 else np.ones_like(conn, dtype=bool)
            if connected_mask.any() and np.isfinite(score[connected_mask]).any():
                score[~connected_mask] *= 0.25
            if score.size == 0 or not np.isfinite(score).any() or float(np.nanmax(score)) <= 0:
                break

            next_idx = int(remaining[int(np.nanargmax(score))])
            before_max = cov_max
            selected.append(next_idx)
            selected_set.add(next_idx)
            current_nearest = _recompute_nearest(selected)
            after_max = float(np.nanmax(current_nearest)) if current_nearest.size else 0.0
            coverage_repairs += 1
            print(
                f"[{log_tag}] Coverage-hole repair add idx={next_idx}: "
                f"views={len(selected)}/{target_select}, coverage_max={before_max:.4f}->{after_max:.4f}m, "
                f"worst_frame={worst_frame}, cluster={int(cluster_idx.size)}, "
                f"cluster_max={cluster_current_max:.4f}m, p99={cov_p99:.4f}m, tail_stop={tail_stop:.4f}m, "
                f"limit={float(coverage_stop_max):.4f}m",
                flush=True,
            )
            after_p99 = _percentile_or_default(current_nearest, 99, after_max)
            if after_max >= before_max - 1e-6 and after_p99 >= cov_p99 - 1e-6:
                break

    if force_target_count and len(selected) < target_select:
        fill_pool = repair_candidates if repair_candidates else candidates
        while len(selected) < target_select:
            remaining = np.asarray([idx for idx in fill_pool if idx not in selected_set], dtype=np.int64)
            if remaining.size == 0:
                break

            current_nearest = _recompute_nearest(selected)
            gains = _coverage_gain_scores(centers, current_nearest, remaining)
            max_gain = float(np.nanmax(gains)) if gains.size else 0.0
            if not np.isfinite(max_gain) or max_gain <= 0:
                max_gain = 1.0
            gain_norm = gains / max_gain

            selected_arr = np.asarray(selected, dtype=np.int64)
            conn = np.nanmax(covis_strength[np.ix_(remaining, selected_arr)], axis=1).astype(np.float64)
            conn[~np.isfinite(conn)] = 0.0
            conn_norm = np.clip(conn / covis_scale, 0.0, None)

            if coverage_first:
                score = np.power(eps + gain_norm, max(float(coverage_beta), 0.0))
                if alpha > 0:
                    score = score + 0.10 * min(float(alpha), 2.0) * conn_norm
            else:
                score = np.power(eps + gain_norm, max(float(coverage_beta), 0.0)) * np.power(
                    eps + conn_norm,
                    max(float(alpha), 0.0),
                )
            if score.size == 0 or not np.isfinite(score).any():
                break
            next_idx = int(remaining[int(np.nanargmax(score))])
            selected.append(next_idx)
            selected_set.add(next_idx)
        if len(selected) < target_select:
            print(
                f"[{log_tag}] WARNING: forced target_count requested but only selected "
                f"{len(selected)}/{target_select} views.",
                flush=True,
            )

    selected = selected[:target_select]
    print(
        f"[{log_tag}] Candidate-pool prune: pool={len(candidate_order)} -> selected={len(selected)}, "
        f"connectivity_repairs={repairs}, coverage_hole_repairs={coverage_repairs}, "
        f"local_radius={local_radius:.4f}m, "
        f"adaptive_count={adaptive_count}, coverage_first={coverage_first}, max_views={target_select}",
        flush=True,
    )
    return selected


def _native_covis_group_for_anchor(
    covis_strength: np.ndarray,
    root: int,
    n_select: int,
) -> List[int]:
    """Approximate WAI native multi-view group as root + strongest root-covis views."""
    n_total = int(covis_strength.shape[0])
    root = int(root)
    target = max(1, min(int(n_select), n_total))
    if target >= n_total:
        return list(range(n_total))

    row = covis_strength[root].astype(np.float64, copy=True)
    row[root] = -np.inf
    finite = np.isfinite(row)
    positive = finite & (row > 0)
    order = np.argsort(-np.where(positive, row, -np.inf), kind="stable")

    selected: List[int] = [root]
    seen = {root}
    for idx in order:
        idx = int(idx)
        if idx in seen or not positive[idx]:
            continue
        selected.append(idx)
        seen.add(idx)
        if len(selected) >= target:
            return selected

    # Fallback for sparse/empty covis rows: fill with nearest camera centers is
    # done by caller through FPS candidates, but keep this function total.
    order = np.argsort(-np.where(finite, row, -np.inf), kind="stable")
    for idx in order:
        idx = int(idx)
        if idx in seen:
            continue
        selected.append(idx)
        seen.add(idx)
        if len(selected) >= target:
            break
    return selected


def select_native_covis_group_anchor(
    centers: np.ndarray,
    covisibility: np.ndarray,
    n_select: int,
    initial_index: int,
    candidate_anchor_count: int = 64,
    log_tag: str = "CovisFPS:NativeGroupASB",
) -> Tuple[int, List[int]]:
    """
    Pick one WAI-native covisibility anchor whose native high-covis group has
    the best scene coverage among FPS-spread anchor candidates.

    This keeps MapAnything's original high-covis single-batch assumption: the
    final loaded views come from dataset[anchor] with num_views=n_select.
    """
    centers = np.asarray(centers, dtype=np.float64).reshape(-1, 3)
    n_total = int(centers.shape[0])
    target = max(1, min(int(n_select), n_total))
    covis = np.asarray(covisibility)
    covis_strength = np.maximum(covis[:n_total, :n_total], covis[:n_total, :n_total].T).astype(np.float64, copy=True)
    np.fill_diagonal(covis_strength, 0.0)

    ref_idx = int(initial_index)
    n_anchor_candidates = max(1, min(int(candidate_anchor_count), n_total))
    anchor_candidates = fps_select_views(centers, n_anchor_candidates, force_include=[ref_idx])

    best_anchor = ref_idx
    best_group = _native_covis_group_for_anchor(covis_strength, ref_idx, target)
    best_score = float("inf")
    best_cov_mean = float("inf")
    best_cov_max = float("inf")
    best_root_conn_median = 0.0

    for anchor in anchor_candidates:
        anchor = int(anchor)
        group = _native_covis_group_for_anchor(covis_strength, anchor, target)
        if len(group) < target:
            continue
        sel = np.asarray(group, dtype=np.int64)
        nearest = np.linalg.norm(centers[:, None, :] - centers[sel][None, :, :], axis=2).min(axis=1)
        cov_mean = float(np.nanmean(nearest))
        cov_max = float(np.nanmax(nearest))
        root_conn = covis_strength[anchor, sel[sel != anchor]]
        root_conn_median = float(np.nanmedian(root_conn)) if root_conn.size else 0.0
        # Max tail matters for memory coverage, but keep mean in the objective
        # so we do not choose an anchor with one-off tail improvement only.
        score = cov_mean + 0.25 * cov_max
        if score < best_score:
            best_score = score
            best_anchor = anchor
            best_group = group
            best_cov_mean = cov_mean
            best_cov_max = cov_max
            best_root_conn_median = root_conn_median

    ref_group = _native_covis_group_for_anchor(covis_strength, ref_idx, target)
    ref_sel = np.asarray(ref_group, dtype=np.int64)
    ref_nearest = np.linalg.norm(centers[:, None, :] - centers[ref_sel][None, :, :], axis=2).min(axis=1)
    print(
        f"[{log_tag}] selected_anchor={best_anchor}, ref_anchor={ref_idx}, "
        f"candidate_anchors={len(anchor_candidates)}, native_group_views={len(best_group)}, "
        f"coverage_mean/max={best_cov_mean:.4f}/{best_cov_max:.4f}m, "
        f"ref_anchor_coverage_mean/max={float(np.nanmean(ref_nearest)):.4f}/"
        f"{float(np.nanmax(ref_nearest)):.4f}m, "
        f"root_covis_median={best_root_conn_median:.6g}",
        flush=True,
    )
    return best_anchor, best_group


def anchor_support_select_views(
    centers: np.ndarray,
    n_select: int,
    covisibility: np.ndarray,
    initial_index: Optional[int] = None,
    alpha: float = 2.0,
    eps: float = 1e-6,
    tau: float = -1.0,
    ref_lambda: float = 0.5,
    anchor_count: int = 0,
    support_per_anchor: int = 3,
    support_tau: float = 1e-4,
    support_min_neighbors: int = 3,
    safe_dist_to_ref: float = 3.0,
    far_view_budget: int = 8,
    far_anchor_budget: int = 2,
    coverage_beta: float = 1.0,
    adaptive: bool = False,
    adaptive_min_views: int = 28,
    ref_dist_max_limit: float = 4.2,
    ref_dist_mean_limit: float = 2.7,
    min_coverage_gain: float = 0.01,
    gain_patience: int = 2,
    target_coverage_mean: float = 0.55,
    target_coverage_max: float = 2.0,
    adaptive_scene_stats: bool = True,
    adaptive_allow_early_stop: bool = True,
    candidate_pool_ratio: float = 1.0,
    adaptive_view_count: bool = False,
    ma_safe: bool = False,
    ma_safe_pool_ratio: float = 3.0,
    log_tag: str = "CovisFPS:AnchorSupport",
) -> List[int]:
    """
    Reference-aware Anchor+Support selection for one MapAnything forward.

    Anchors improve scene coverage, supports provide local high-covisibility
    evidence around anchors, and the far-view budgets prevent the selected batch
    from drifting into the large-baseline regime that breaks a single reference.
    """
    centers = np.asarray(centers, dtype=np.float64).reshape(-1, 3)
    n_total = int(centers.shape[0])
    target_select = min(int(n_select), n_total)
    candidate_pool_ratio = float(candidate_pool_ratio)
    use_full_candidate_pool = candidate_pool_ratio <= 0
    effective_candidate_pool_ratio = 1.0 if use_full_candidate_pool else max(candidate_pool_ratio, 1.0)
    n_select = (
        n_total
        if use_full_candidate_pool
        else min(max(target_select, int(np.ceil(float(target_select) * effective_candidate_pool_ratio))), n_total)
    )
    if target_select >= n_total:
        return list(range(n_total))

    covis = np.asarray(covisibility)
    if covis.shape[0] < n_total or covis.shape[1] < n_total:
        raise ValueError(f"covisibility shape {covis.shape} is smaller than centers length {n_total}")
    covis = covis[:n_total, :n_total]
    covis_strength = np.maximum(covis, covis.T).astype(np.float64, copy=True)
    np.fill_diagonal(covis_strength, 0.0)

    if initial_index is not None and 0 <= int(initial_index) < n_total:
        root = int(initial_index)
    else:
        root = 0

    dist_to_ref = np.linalg.norm(centers - centers[root], axis=1).astype(np.float64)
    positive_ref_dists = dist_to_ref[np.isfinite(dist_to_ref) & (dist_to_ref > 0)]
    ref_scale = float(np.nanmedian(positive_ref_dists)) if positive_ref_dists.size else 1.0
    if not np.isfinite(ref_scale) or ref_scale <= 0:
        ref_scale = 1.0

    eps = max(float(eps), 1e-12)
    alpha = max(float(alpha), 0.0)
    tau = float(tau)
    ref_lambda = max(float(ref_lambda), 0.0)
    support_per_anchor = max(0, int(support_per_anchor))
    support_tau = max(float(support_tau), 0.0)
    support_min_neighbors = max(0, int(support_min_neighbors))
    far_view_budget = max(0, int(far_view_budget))
    far_anchor_budget = max(0, int(far_anchor_budget))
    coverage_beta = max(float(coverage_beta), 0.0)
    adaptive = bool(adaptive)
    adaptive_scene_stats = bool(adaptive_scene_stats)
    adaptive_allow_early_stop = bool(adaptive_allow_early_stop)
    adaptive_view_count = bool(adaptive_view_count)
    ma_safe = bool(ma_safe)
    ma_safe_pool_ratio = max(float(ma_safe_pool_ratio), 1.0)
    if n_select > target_select:
        adaptive_allow_early_stop = False
    adaptive_min_views = max(1, min(int(adaptive_min_views), target_select))
    min_coverage_gain = max(float(min_coverage_gain), 0.0)
    gain_patience = max(1, int(gain_patience))
    ref_dist_max_limit = float(ref_dist_max_limit)
    ref_dist_mean_limit = float(ref_dist_mean_limit)
    target_coverage_mean = float(target_coverage_mean)
    target_coverage_max = float(target_coverage_max)

    if adaptive and adaptive_scene_stats:
        auto_params = _estimate_adaptive_asb_scene_params(
            centers,
            covis_strength,
            root,
            target_select,
            adaptive_min_views,
            support_min_neighbors,
            log_tag,
        )
        safe_dist_to_ref = auto_params.get("safe_dist_to_ref", float(safe_dist_to_ref))
        ref_dist_max_limit = auto_params.get("ref_dist_max_limit", ref_dist_max_limit)
        ref_dist_mean_limit = auto_params.get("ref_dist_mean_limit", ref_dist_mean_limit)
        target_coverage_mean = auto_params.get("target_coverage_mean", target_coverage_mean)
        target_coverage_max = auto_params.get("target_coverage_max", target_coverage_max)
        min_coverage_gain = auto_params.get("min_coverage_gain", min_coverage_gain)
        if tau < 0 and auto_params.get("covis_tau", -1.0) > 0:
            tau = float(auto_params["covis_tau"])
        if auto_params.get("support_tau", 0.0) > 0:
            support_tau = float(auto_params["support_tau"])

    use_ref_dist_max_limit = adaptive and np.isfinite(ref_dist_max_limit) and ref_dist_max_limit > 0
    use_ref_dist_mean_limit = adaptive and np.isfinite(ref_dist_mean_limit) and ref_dist_mean_limit > 0
    use_target_coverage_mean = adaptive and np.isfinite(target_coverage_mean) and target_coverage_mean > 0
    use_target_coverage_max = adaptive and np.isfinite(target_coverage_max) and target_coverage_max > 0
    adaptive_count_enabled = adaptive and adaptive_view_count
    allow_short_selection = adaptive_count_enabled and adaptive_allow_early_stop
    adaptive_count_min_select = target_select
    adaptive_count_coverage_stop_mean = None
    adaptive_count_coverage_stop_max = None
    if adaptive_count_enabled:
        adaptive_count_min_select = max(adaptive_min_views, int(np.ceil(0.85 * float(target_select))))
        adaptive_count_min_select = max(1, min(adaptive_count_min_select, target_select))
        if use_target_coverage_mean:
            adaptive_count_coverage_stop_mean = max(target_coverage_mean, 1e-6)
        if use_target_coverage_max:
            adaptive_count_coverage_stop_max = max(target_coverage_max, 1e-6)

    safe_dist_to_ref = float(safe_dist_to_ref)
    use_safe_dist = np.isfinite(safe_dist_to_ref) and safe_dist_to_ref > 0
    far_mask = (dist_to_ref > safe_dist_to_ref) if use_safe_dist else np.zeros(n_total, dtype=bool)

    support_counts = (covis_strength >= support_tau).sum(axis=1).astype(np.int64)
    supportable = support_counts >= support_min_neighbors
    supportable[root] = False
    n_supportable = int(supportable.sum())

    max_anchor_by_budget = max(1, (n_select - 1) // max(1, support_per_anchor + 1))
    if int(anchor_count) <= 0:
        anchor_count = min(8, max(4, n_select // 5), max_anchor_by_budget)
    anchor_count = max(1, min(int(anchor_count), max_anchor_by_budget, n_select - 1))

    print(
        f"[{log_tag}] Selecting {n_select} views: root={root}, anchors={anchor_count}, "
        f"support_per_anchor={support_per_anchor}, support_tau={support_tau:.6g}, "
        f"support_min_neighbors={support_min_neighbors}, safe_dist_to_ref={safe_dist_to_ref:.4f}m, "
        f"far_view_budget={far_view_budget}, far_anchor_budget={far_anchor_budget}, "
        f"ref_lambda={ref_lambda:.4f}, coverage_beta={coverage_beta:.3f}, "
        f"supportable_candidates={n_supportable}/{n_total}",
        flush=True,
    )
    if n_select > target_select:
        print(
            f"[{log_tag}] Candidate pool enabled: target_views={target_select}, "
            f"candidate_pool_views={n_select}, ratio="
            f"{'all' if use_full_candidate_pool else f'{effective_candidate_pool_ratio:.3f}'}",
            flush=True,
        )
    if adaptive:
        print(
            f"[{log_tag}] Adaptive enabled: scene_stats={adaptive_scene_stats}, "
            f"allow_early_stop={adaptive_allow_early_stop}, min_views={adaptive_min_views}, "
            f"ref_dist_max_limit={ref_dist_max_limit:.4f}m, "
            f"ref_dist_mean_limit={ref_dist_mean_limit:.4f}m, "
            f"target_coverage_mean={target_coverage_mean:.4f}m, "
            f"target_coverage_max={target_coverage_max:.4f}m, "
            f"min_coverage_gain={min_coverage_gain:.4f}m, gain_patience={gain_patience}, "
            f"adaptive_view_count={adaptive_count_enabled}, "
            f"adaptive_count_min_views={adaptive_count_min_select if adaptive_count_enabled else target_select}",
            flush=True,
        )

    if ma_safe:
        root_conn = covis_strength[root].astype(np.float64, copy=True)
        root_conn[root] = 0.0
        finite_conn = np.isfinite(root_conn)
        positive = finite_conn & (root_conn > 0)
        pool_size = min(
            n_total - 1,
            max(target_select - 1, int(np.ceil(float(target_select) * ma_safe_pool_ratio))),
        )
        supportable_mask = support_counts >= support_min_neighbors
        primary = np.flatnonzero(positive & supportable_mask).astype(np.int64)
        if primary.size < max(target_select - 1, 1):
            primary = np.flatnonzero(positive).astype(np.int64)
        if primary.size:
            primary = primary[np.argsort(-root_conn[primary], kind="stable")]
        candidates = [int(i) for i in primary[:pool_size].tolist() if int(i) != root]

        if len(candidates) < max(target_select - 1, 1):
            order = np.argsort(-np.where(finite_conn, root_conn, -np.inf), kind="stable")
            seen = set(candidates)
            for idx in order:
                idx = int(idx)
                if idx == root or idx in seen:
                    continue
                candidates.append(idx)
                seen.add(idx)
                if len(candidates) >= pool_size:
                    break

        candidate_order = [root] + candidates
        ordered = _prune_candidate_pool_to_budget(
            centers,
            covis_strength,
            candidate_order,
            target_select,
            root,
            support_tau,
            alpha,
            coverage_beta,
            eps,
            log_tag,
            force_target_count=True,
            coverage_first=True,
        )
        final_cov = np.linalg.norm(
            centers[:, None, :] - centers[np.asarray(ordered, dtype=np.int64)][None, :, :],
            axis=2,
        ).min(axis=1)
        ordered_ref = dist_to_ref[np.asarray(ordered, dtype=np.int64)]
        ordered_root_conn = root_conn[np.asarray([i for i in ordered if int(i) != root], dtype=np.int64)]
        root_conn_min = float(np.nanmin(ordered_root_conn)) if ordered_root_conn.size else 0.0
        root_conn_median = float(np.nanmedian(ordered_root_conn)) if ordered_root_conn.size else 0.0
        root_conn_p90 = _percentile_or_default(ordered_root_conn, 90, 0.0) if ordered_root_conn.size else 0.0
        print(
            f"[{log_tag}] MA-safe final stats: coverage_mean={float(np.nanmean(final_cov)):.4f}m, "
            f"coverage_max={float(np.nanmax(final_cov)):.4f}m, "
            f"ref_dist_mean={float(np.nanmean(ordered_ref)):.4f}m, "
            f"ref_dist_max={float(np.nanmax(ordered_ref)):.4f}m, "
            f"root_covis_min/median/p90={root_conn_min:.6g}/{root_conn_median:.6g}/{root_conn_p90:.6g}, "
            f"candidate_pool={len(candidate_order)}, pool_ratio={ma_safe_pool_ratio:.3f}",
            flush=True,
        )
        return ordered

    if use_full_candidate_pool:
        candidate_mask = np.ones(n_total, dtype=bool)
        candidate_mask[root] = False
        if adaptive and use_ref_dist_max_limit:
            candidate_mask &= dist_to_ref <= ref_dist_max_limit
        coverage_repair_order = [root] + [int(i) for i in np.flatnonzero(candidate_mask).astype(np.int64)]
        supportable_mask = support_counts >= support_min_neighbors
        if int((candidate_mask & supportable_mask).sum()) >= max(target_select * 2, target_select - 1):
            candidate_mask &= supportable_mask
        candidates = np.flatnonzero(candidate_mask).astype(np.int64)
        if candidates.size < max(target_select - 1, 1):
            candidates = np.flatnonzero(np.arange(n_total) != root).astype(np.int64)
        candidate_order = [root] + [int(i) for i in candidates]
        print(
            f"[{log_tag}] Full candidate source with MA-safe filtering: "
            f"candidate_pool={len(candidate_order)}, adaptive_ref_limit={adaptive and use_ref_dist_max_limit}, "
            f"supportable_filter={len(candidate_order) < n_total}.",
            flush=True,
        )
        use_adaptive_prune = adaptive and use_target_coverage_max
        prune_min_select = adaptive_count_min_select
        prune_coverage_stop_mean = adaptive_count_coverage_stop_mean
        prune_coverage_stop_max = adaptive_count_coverage_stop_max
        if use_adaptive_prune and not adaptive_count_enabled:
            prune_min_select = max(adaptive_min_views, int(np.ceil(0.85 * float(target_select))))
            prune_min_select = max(1, min(prune_min_select, target_select))
            if use_target_coverage_mean:
                prune_coverage_stop_mean = max(target_coverage_mean, 1e-6)
            if use_target_coverage_max:
                prune_coverage_stop_max = max(target_coverage_max, 1e-6)

        ordered = _prune_candidate_pool_to_budget(
            centers,
            covis_strength,
            candidate_order,
            target_select,
            root,
            support_tau,
            alpha,
            coverage_beta,
            eps,
            log_tag,
            adaptive_count=use_adaptive_prune,
            min_select=prune_min_select,
            coverage_stop_mean=prune_coverage_stop_mean,
            coverage_stop_max=prune_coverage_stop_max,
            coverage_repair_order=coverage_repair_order,
            force_target_count=True,
        )
        final_cov = np.linalg.norm(
            centers[:, None, :] - centers[np.asarray(ordered, dtype=np.int64)][None, :, :],
            axis=2,
        ).min(axis=1)
        ordered_ref = dist_to_ref[np.asarray(ordered, dtype=np.int64)]
        print(
            f"[{log_tag}] Full candidate pool final stats: coverage_mean={float(np.nanmean(final_cov)):.4f}m, "
            f"coverage_max={float(np.nanmax(final_cov)):.4f}m, "
            f"ref_dist_mean={float(np.nanmean(ordered_ref)):.4f}m, "
            f"ref_dist_max={float(np.nanmax(ordered_ref)):.4f}m, candidates={len(candidate_order)}",
            flush=True,
        )
        return ordered

    selected_mask = np.zeros(n_total, dtype=bool)
    selected_mask[root] = True
    selected_all: List[int] = [root]
    anchors: List[int] = []
    supports_by_anchor: Dict[int, List[int]] = {}
    fillers: List[int] = []
    current_nearest = np.linalg.norm(centers - centers[root], axis=1)

    max_support_count = float(np.nanmax(support_counts)) if support_counts.size else 1.0
    if not np.isfinite(max_support_count) or max_support_count <= 0:
        max_support_count = 1.0
    support_gamma = 0.5 * min(alpha, 2.0)

    def _far_view_count() -> int:
        return int(far_mask[np.asarray(selected_all, dtype=np.int64)].sum()) if selected_all else 0

    def _coverage_mean() -> float:
        return float(np.nanmean(current_nearest)) if current_nearest.size else 0.0

    def _coverage_max() -> float:
        return float(np.nanmax(current_nearest)) if current_nearest.size else 0.0

    def _selected_disconnected_count() -> int:
        if len(selected_all) <= 1 or support_tau <= 0:
            return 0
        count = 0
        sel_arr = np.asarray(selected_all, dtype=np.int64)
        for idx in selected_all:
            idx = int(idx)
            if idx == root:
                continue
            others = sel_arr[sel_arr != idx]
            if others.size == 0:
                count += 1
                continue
            conn = float(np.nanmax(covis_strength[idx, others]))
            if not np.isfinite(conn) or conn < support_tau:
                count += 1
        return count

    def _coverage_target_met() -> bool:
        if not allow_short_selection:
            return False
        checks: List[bool] = []
        if use_target_coverage_mean:
            checks.append(_coverage_mean() <= target_coverage_mean)
        if use_target_coverage_max:
            checks.append(_coverage_max() <= target_coverage_max)
        return bool(checks) and all(checks)

    def _selected_ref_dist_mean_after(idx: int) -> float:
        selected_arr = np.asarray(selected_all, dtype=np.int64)
        current_sum = float(dist_to_ref[selected_arr].sum()) if selected_arr.size else 0.0
        return (current_sum + float(dist_to_ref[int(idx)])) / float(len(selected_all) + 1)

    def _candidate_exceeds_ref_limits(idx: int) -> bool:
        if not adaptive:
            return False
        idx = int(idx)
        if use_ref_dist_max_limit and float(dist_to_ref[idx]) > ref_dist_max_limit:
            return True
        if use_ref_dist_mean_limit and _selected_ref_dist_mean_after(idx) > ref_dist_mean_limit:
            return True
        return False

    def _adaptive_count_target_met() -> bool:
        if not adaptive_count_enabled or len(selected_all) < adaptive_count_min_select:
            return False
        coverage_ok = (
            (adaptive_count_coverage_stop_mean is None or _coverage_mean() <= adaptive_count_coverage_stop_mean)
            and (adaptive_count_coverage_stop_max is None or _coverage_max() <= adaptive_count_coverage_stop_max)
        )
        disconnected = _selected_disconnected_count()
        disconnected_limit = max(1, int(np.ceil(0.15 * float(max(len(selected_all) - 1, 1)))))
        return coverage_ok and disconnected <= disconnected_limit

    def _filter_ref_limit_candidates(candidates: np.ndarray) -> np.ndarray:
        if not adaptive or candidates.size == 0:
            return candidates
        keep = np.asarray([not _candidate_exceeds_ref_limits(int(idx)) for idx in candidates], dtype=bool)
        return candidates[keep]

    def _add_view(idx: int) -> None:
        nonlocal current_nearest
        idx = int(idx)
        if selected_mask[idx]:
            return
        selected_mask[idx] = True
        selected_all.append(idx)
        d = np.linalg.norm(centers - centers[idx], axis=1)
        current_nearest = np.minimum(current_nearest, d)

    # 1) Coverage anchors, but only among nodes that have enough local support.
    far_anchor_used = 0
    for _ in range(anchor_count):
        candidate_mask = supportable & (~selected_mask)
        if not candidate_mask.any():
            print(f"[{log_tag}] No supportable anchor candidates remain; using unconstrained candidates.", flush=True)
            candidate_mask = ~selected_mask

        if use_safe_dist and far_anchor_used >= far_anchor_budget:
            near_mask = candidate_mask & (~far_mask)
            if near_mask.any():
                candidate_mask = near_mask

        candidates = np.flatnonzero(candidate_mask)
        if adaptive:
            safe_candidates = _filter_ref_limit_candidates(candidates)
            if safe_candidates.size == 0:
                print(
                    f"[{log_tag}] Stop anchor selection: no candidates satisfy adaptive ref-distance limits "
                    f"(selected={len(selected_all)}, coverage_mean={_coverage_mean():.4f}m, "
                    f"coverage_max={_coverage_max():.4f}m).",
                    flush=True,
                )
                break
            candidates = safe_candidates
        if candidates.size == 0:
            break

        gains = _coverage_gain_scores(centers, current_nearest, candidates)
        max_gain = float(np.nanmax(gains)) if gains.size else 0.0
        if not np.isfinite(max_gain) or max_gain <= 0:
            max_gain = 1.0
        gain_norm = gains / max_gain
        support_norm = np.log1p(support_counts[candidates].astype(np.float64)) / np.log1p(max_support_count)
        support_norm = np.clip(support_norm, 0.0, None)
        ref_penalty = np.exp(-ref_lambda * (dist_to_ref[candidates] / ref_scale))
        ref_penalty[~np.isfinite(ref_penalty)] = 0.0
        score = np.power(eps + gain_norm, coverage_beta) * np.power(eps + support_norm, support_gamma) * ref_penalty
        next_anchor = int(candidates[int(np.argmax(score))])

        if use_safe_dist and far_mask[next_anchor]:
            far_anchor_used += 1
        anchors.append(next_anchor)
        supports_by_anchor[next_anchor] = []
        _add_view(next_anchor)

    print(
        f"[{log_tag}] Anchors selected ({len(anchors)}): {anchors}; "
        f"far_anchors={int(far_mask[np.asarray(anchors, dtype=np.int64)].sum()) if anchors else 0}",
        flush=True,
    )

    # 2) Local supports for each anchor. Use high covisibility first, with a
    # small diversity bonus so support views do not collapse to the same camera.
    for anchor in anchors:
        for _ in range(support_per_anchor):
            if len(selected_all) >= n_select:
                break
            candidate_mask = (~selected_mask) & (covis_strength[anchor] >= support_tau)
            if use_safe_dist and _far_view_count() >= far_view_budget:
                near_mask = candidate_mask & (~far_mask)
                if near_mask.any():
                    candidate_mask = near_mask
            candidates = np.flatnonzero(candidate_mask)
            if adaptive:
                safe_candidates = _filter_ref_limit_candidates(candidates)
                if safe_candidates.size == 0:
                    break
                candidates = safe_candidates
            if candidates.size == 0:
                break

            covis_vals = covis_strength[anchor, candidates].astype(np.float64)
            covis_max = float(np.nanmax(covis_vals)) if covis_vals.size else 0.0
            if not np.isfinite(covis_max) or covis_max <= 0:
                covis_max = 1.0
            covis_norm = np.clip(covis_vals / covis_max, 0.0, None)

            chosen_supports = supports_by_anchor.get(anchor, [])
            if chosen_supports:
                support_centers = centers[np.asarray(chosen_supports, dtype=np.int64)]
                div = np.linalg.norm(centers[candidates, None, :] - support_centers[None, :, :], axis=2).min(axis=1)
                div_max = float(np.nanmax(div)) if div.size else 0.0
                div_norm = div / div_max if np.isfinite(div_max) and div_max > 0 else np.zeros_like(div)
            else:
                div_norm = np.zeros(len(candidates), dtype=np.float64)

            score = covis_norm * (1.0 + 0.25 * div_norm)
            next_support = int(candidates[int(np.argmax(score))])
            supports_by_anchor[anchor].append(next_support)
            _add_view(next_support)

    support_sizes = [len(supports_by_anchor.get(a, [])) for a in anchors]
    print(
        f"[{log_tag}] Support sizes per anchor: {support_sizes}; "
        f"selected_after_support={len(selected_all)}, far_views={_far_view_count()}",
        flush=True,
    )

    # 3) Fill remaining slots with coverage gain, covisibility to the selected
    # set, and the same reference-distance penalty / far-view budget.
    covis_scale = float(np.nanmax(covis_strength)) if covis_strength.size else 0.0
    if not np.isfinite(covis_scale) or covis_scale <= 0:
        covis_scale = 1.0
    low_gain_steps = 0
    while len(selected_all) < n_select:
        if _adaptive_count_target_met():
            print(
                f"[{log_tag}] Adaptive count stop before filler: selected={len(selected_all)}, "
                f"coverage_mean={_coverage_mean():.4f}m, coverage_max={_coverage_max():.4f}m, "
                f"disconnected={_selected_disconnected_count()}, max_views={target_select}.",
                flush=True,
            )
            break
        selected_arr = np.asarray(selected_all, dtype=np.int64)
        max_covis = np.nanmax(covis_strength[:, selected_arr], axis=1).astype(np.float64)
        max_covis[~np.isfinite(max_covis)] = 0.0

        candidate_mask = ~selected_mask
        if use_safe_dist and _far_view_count() >= far_view_budget:
            near_mask = candidate_mask & (~far_mask)
            if near_mask.any():
                candidate_mask = near_mask
        if tau >= 0:
            valid = candidate_mask & (max_covis >= tau)
            if valid.any():
                candidate_mask = valid

        candidates = np.flatnonzero(candidate_mask)
        if candidates.size == 0:
            break

        gains = _coverage_gain_scores(centers, current_nearest, candidates)
        max_gain = float(np.nanmax(gains)) if gains.size else 0.0
        if not np.isfinite(max_gain) or max_gain <= 0:
            max_gain = 1.0
        gain_norm = gains / max_gain
        covis_norm = np.clip(max_covis[candidates] / covis_scale, 0.0, None)
        ref_penalty = np.exp(-ref_lambda * (dist_to_ref[candidates] / ref_scale))
        ref_penalty[~np.isfinite(ref_penalty)] = 0.0
        score = np.power(eps + gain_norm, coverage_beta) * np.power(eps + covis_norm, alpha) * ref_penalty
        if adaptive:
            raw_best_pos = int(np.argmax(score))
            raw_best_idx = int(candidates[raw_best_pos])
            raw_best_risky = _candidate_exceeds_ref_limits(raw_best_idx)
            if allow_short_selection and len(selected_all) >= adaptive_min_views and _coverage_target_met() and raw_best_risky:
                print(
                    f"[{log_tag}] Adaptive stop before risky filler idx={raw_best_idx}: "
                    f"selected={len(selected_all)}, coverage_mean={_coverage_mean():.4f}m, "
                    f"coverage_max={_coverage_max():.4f}m, ref_dist={dist_to_ref[raw_best_idx]:.4f}m.",
                    flush=True,
                )
                break

            keep = np.asarray([not _candidate_exceeds_ref_limits(int(idx)) for idx in candidates], dtype=bool)
            if not keep.any():
                if allow_short_selection:
                    print(
                        f"[{log_tag}] Adaptive stop: no filler candidates satisfy ref-distance limits "
                        f"(selected={len(selected_all)}, coverage_mean={_coverage_mean():.4f}m, "
                        f"coverage_max={_coverage_max():.4f}m).",
                        flush=True,
                    )
                    break
                print(
                    f"[{log_tag}] Adaptive ref-distance limits exhausted; continuing to fill "
                    f"{target_select} views because adaptive_view_count=False.",
                    flush=True,
                )
            elif not keep.all():
                candidates = candidates[keep]
                gains = gains[keep]
                score = score[keep]

        best_pos = int(np.argmax(score))
        next_idx = int(candidates[best_pos])
        selected_gain = float(gains[best_pos]) if gains.size else 0.0
        fillers.append(next_idx)
        _add_view(next_idx)
        if allow_short_selection and len(selected_all) >= adaptive_min_views:
            if selected_gain < min_coverage_gain:
                low_gain_steps += 1
            else:
                low_gain_steps = 0
            if low_gain_steps >= gain_patience and _coverage_target_met():
                print(
                    f"[{log_tag}] Adaptive stop after low-gain fillers: selected={len(selected_all)}, "
                    f"last_gain={selected_gain:.4f}m, low_gain_steps={low_gain_steps}, "
                    f"coverage_mean={_coverage_mean():.4f}m, coverage_max={_coverage_max():.4f}m.",
                    flush=True,
                )
                break

    # Stable MapAnything input order: root first, then local anchor/support
    # blocks, then any coverage fillers. Never sort by frame index.
    ordered: List[int] = [root]
    seen = {root}
    for anchor in sorted(anchors, key=lambda i: float(dist_to_ref[int(i)])):
        if anchor not in seen:
            ordered.append(anchor)
            seen.add(anchor)
        for support in sorted(supports_by_anchor.get(anchor, []), key=lambda i: -float(covis_strength[anchor, int(i)])):
            if support not in seen:
                ordered.append(int(support))
                seen.add(int(support))
    for filler in fillers:
        if filler not in seen:
            ordered.append(int(filler))
            seen.add(int(filler))
    for idx in selected_all:
        if idx not in seen:
            ordered.append(int(idx))
            seen.add(int(idx))

    if len(ordered) > target_select:
        ordered = _prune_candidate_pool_to_budget(
            centers,
            covis_strength,
            ordered,
            target_select,
            root,
            support_tau,
            alpha,
            coverage_beta,
            eps,
            log_tag,
            adaptive_count=adaptive_count_enabled,
            min_select=adaptive_count_min_select,
            coverage_stop_mean=adaptive_count_coverage_stop_mean,
            coverage_stop_max=adaptive_count_coverage_stop_max,
        )
    else:
        ordered = ordered[:target_select]
    print(
        f"[{log_tag}] Final ordered views={len(ordered)}, anchors={len(anchors)}, "
        f"supports={sum(support_sizes)}, fillers={len(fillers)}, "
        f"far_views={int(far_mask[np.asarray(ordered, dtype=np.int64)].sum()) if ordered else 0}",
        flush=True,
    )
    if adaptive:
        ordered_ref = dist_to_ref[np.asarray(ordered, dtype=np.int64)] if ordered else np.asarray([], dtype=np.float64)
        ref_mean = float(np.nanmean(ordered_ref)) if ordered_ref.size else 0.0
        ref_max = float(np.nanmax(ordered_ref)) if ordered_ref.size else 0.0
        if ordered:
            final_cov = np.linalg.norm(
                centers[:, None, :] - centers[np.asarray(ordered, dtype=np.int64)][None, :, :],
                axis=2,
            ).min(axis=1)
            final_cov_mean = float(np.nanmean(final_cov))
            final_cov_max = float(np.nanmax(final_cov))
        else:
            final_cov_mean = 0.0
            final_cov_max = 0.0
        print(
            f"[{log_tag}] Adaptive final stats: coverage_mean={final_cov_mean:.4f}m, "
            f"coverage_max={final_cov_max:.4f}m, ref_dist_mean={ref_mean:.4f}m, "
            f"ref_dist_max={ref_max:.4f}m, min_views={adaptive_min_views}, max_views={target_select}",
            flush=True,
        )
        if len(ordered) < adaptive_min_views:
            print(
                f"[{log_tag}] WARNING: adaptive ref-distance limits stopped selection below min_views "
                f"({len(ordered)} < {adaptive_min_views}).",
                flush=True,
            )
    return ordered


def select_memory_views(
    dataset,
    n_memory: int,
    dataset_path: Optional[str] = None,
    scene_name: Optional[str] = None,
    force_include_origin: bool = True,
    selection_mode: str = "fps_flat",
    covis_alpha: float = 1.0,
    covis_eps: float = 1e-6,
    covis_tau: float = -1.0,
    covis_ref_lambda: float = 0.0,
    covis_anchor_count: int = 0,
    covis_support_per_anchor: int = 3,
    covis_support_tau: float = 1e-4,
    covis_support_min_neighbors: int = 3,
    covis_safe_dist_to_ref: float = 3.0,
    covis_far_view_budget: int = 8,
    covis_far_anchor_budget: int = 2,
    covis_coverage_beta: float = 1.0,
    covis_candidate_pool_ratio: float = 1.0,
    covis_ma_safe_asb: bool = False,
    covis_ma_safe_pool_ratio: float = 3.0,
    covis_native_group_asb: bool = False,
    covis_native_anchor_candidates: int = 64,
    covis_adaptive_asb: bool = False,
    covis_adaptive_scene_stats: bool = True,
    covis_adaptive_view_count: bool = False,
    covis_adaptive_allow_early_stop: bool = True,
    covis_adaptive_min_views: int = 28,
    covis_ref_dist_max_limit: float = 4.2,
    covis_ref_dist_mean_limit: float = 2.7,
    covis_min_coverage_gain: float = 0.01,
    covis_gain_patience: int = 2,
    covis_target_coverage_mean: float = 0.55,
    covis_target_coverage_max: float = 2.0,
) -> Tuple[List[int], Optional[np.ndarray], List[List[int]]]:
    """
    选择 memory 视角（优先 FPS）。

    优先级：
    1) 若可用：map-anything 的 select_optimal_memory_indices（它本身就是最优选帧逻辑）
    2) 否则：读取 <dataset_path>/poses 做 FPS（最快，不加载图像/深度）
    3) 再否则：遍历 dataset item 抽 pose 做 FPS（慢）

    Args:
        force_include_origin: Legacy name retained for CLI compatibility. For
            FPS-style modes, the fallback path now matches map-anything's
            select_optimal_memory_indices: seed FPS with the frame closest to
            the global mean camera center, not necessarily frame 0.
    """
    selection_mode = canonicalize_wai_view_mode(selection_mode)
    if selection_mode != "anchor_support":
        try:
            from mapanything.tasks.ace.memory_selection import select_optimal_memory_indices  # pyright: ignore[reportMissingImports]
            memory_indices, scene_center = select_optimal_memory_indices(dataset, n_memory)
            return memory_indices, scene_center, [memory_indices]
        except ImportError:
            pass

    centers = _extract_centers_from_flat_wai_dataset(dataset)
    if dataset_path is not None:
        # WAI fastest path: scene_meta.json already contains per-frame c2w.
        if centers is None and scene_name:
            centers = _extract_centers_from_wai_scene_meta(dataset_path, scene_name)
        # ACE / generic path: poses/ directory
        if centers is None:
            centers = _extract_centers_from_pose_dir(dataset_path, scene_name=scene_name)
    if centers is None:
        centers = _extract_centers_from_dataset_items(dataset)

    scene_center = centers.mean(axis=0).astype(np.float32) if centers is not None and len(centers) > 0 else None

    ref_idx = int(np.argmin(np.linalg.norm(centers - scene_center.reshape(1, 3), axis=1)))

    if selection_mode == "anchor_support":
        covis = _load_wai_pairwise_covisibility(dataset_path, scene_name, n_expected=len(centers))
        if covis is not None:
            if covis_native_group_asb:
                native_anchor, native_group = select_native_covis_group_anchor(
                    centers,
                    covis,
                    n_memory,
                    initial_index=ref_idx,
                    candidate_anchor_count=covis_native_anchor_candidates,
                )
                _print_selection_coverage_stats(centers, native_group, "AnchorSupportNativeGroupEstimate")
                return [int(native_anchor)], scene_center, [native_group]

            print(
                f"[AnchorSupport] Selecting {n_memory} views with ref_idx={ref_idx}, "
                f"alpha={covis_alpha}, eps={covis_eps}, tau={covis_tau}, ref_lambda={covis_ref_lambda}",
                flush=True,
            )
            indices = anchor_support_select_views(
                centers,
                n_memory,
                covis,
                initial_index=ref_idx,
                alpha=covis_alpha,
                eps=covis_eps,
                tau=covis_tau,
                ref_lambda=covis_ref_lambda,
                anchor_count=covis_anchor_count,
                support_per_anchor=covis_support_per_anchor,
                support_tau=covis_support_tau,
                support_min_neighbors=covis_support_min_neighbors,
                safe_dist_to_ref=covis_safe_dist_to_ref,
                far_view_budget=covis_far_view_budget,
                far_anchor_budget=covis_far_anchor_budget,
                coverage_beta=covis_coverage_beta,
                candidate_pool_ratio=covis_candidate_pool_ratio,
                ma_safe=covis_ma_safe_asb,
                ma_safe_pool_ratio=covis_ma_safe_pool_ratio,
                adaptive=covis_adaptive_asb,
                adaptive_scene_stats=covis_adaptive_scene_stats,
                adaptive_view_count=covis_adaptive_view_count,
                adaptive_allow_early_stop=covis_adaptive_allow_early_stop,
                adaptive_min_views=covis_adaptive_min_views,
                ref_dist_max_limit=covis_ref_dist_max_limit,
                ref_dist_mean_limit=covis_ref_dist_mean_limit,
                min_coverage_gain=covis_min_coverage_gain,
                gain_patience=covis_gain_patience,
                target_coverage_mean=covis_target_coverage_mean,
                target_coverage_max=covis_target_coverage_max,
            )
            groups = [indices]
            _print_selection_coverage_stats(centers, indices, "AnchorSupport")
            return indices, scene_center, groups
        print("[AnchorSupport] Falling back to metadata FPS because covisibility is unavailable.", flush=True)
        indices = fps_select_views(centers, n_memory, force_include=[ref_idx])
        return indices, scene_center, [indices]

    force_include = [ref_idx] if force_include_origin else None
    indices = fps_select_views(centers, n_memory, force_include=force_include)
    return indices, scene_center, [indices]


def convert_ace_tuple_to_dict(
    sample: tuple,
    dataset_path: str,
    image_height: int = 518,
) -> Dict[str, Any]:
    """
    Convert CamLocDatasetDINOv2 tuple output to dict format expected by
    prepare_batch_input and two_pass_processing.

    ACE dataset __getitem__ returns:
        (image, image_mask, pose, pose_inv, intrinsics, intrinsics_inv, coords, filename)

    Depth is NOT in the tuple — it must be loaded separately from the depth/ directory.

    Args:
        sample: Tuple from CamLocDatasetDINOv2.__getitem__
        dataset_path: Root path to the scene dataset (contains rgb/, depth/, poses/, etc.)
        image_height: Target image height (for depth scaling)

    Returns:
        Dict with keys: img, camera_pose, camera_intrinsics, depthmap, filename
    """
    if len(sample) != 8:
        raise ValueError(
            f"ACE 样本应为 8 元组，实际 len={len(sample)}。"
            "若数据为 WAI（list[dict]），请使用 --dataset_loader wai，勿走 convert_ace_tuple_to_dict。"
        )
    image, image_mask, pose, pose_inv, intrinsics, intrinsics_inv, coords, filename = sample

    # Derive depth file path from the rgb filename.
    # filename is like ".../rgb/seq-02-frame-000102.color.png"
    # depth is at   ".../depth/seq-02-frame-000102.depth.png"
    depth = None
    # 目标分辨率：与 CamLocDatasetDINOv2 输出的图像保持一致，确保 depth 与 intrinsics 对齐
    target_hw: Optional[Tuple[int, int]] = None
    try:
        if isinstance(image, torch.Tensor):
            target_hw = tuple(int(x) for x in image.shape[-2:])
        elif isinstance(image, np.ndarray):
            target_hw = tuple(int(x) for x in image.shape[-2:])
    except Exception:
        target_hw = None
    if filename and isinstance(filename, str):
        # Replace /rgb/ with /depth/ in the path
        depth_path = filename.replace("/rgb/", "/depth/")
        # Replace .color.png → .depth.png (or just strip extension and add .depth.png)
        base, ext = os.path.splitext(depth_path)
        if base.endswith(".color"):
            base = base[:-6]  # strip ".color"
        depth_path = base + ".depth" + ext

        if not os.path.exists(depth_path):
            # Try just replacing /rgb/ without renaming the suffix
            alt_path = filename.replace("/rgb/", "/depth/")
            if os.path.exists(alt_path):
                depth_path = alt_path

        if os.path.exists(depth_path):
            from skimage import io as skio

            depth_np = skio.imread(depth_path).astype(np.float64) / 1000.0  # mm -> meters

            # 若深度分辨率与图像不一致，则使用最近邻重采样到图像分辨率，使其与 intrinsics 一致
            if target_hw is not None and depth_np.shape[:2] != target_hw:
                try:
                    import torch.nn.functional as F

                    d = torch.from_numpy(depth_np).float().unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
                    d_resized = F.interpolate(d, size=target_hw, mode="nearest")
                    depth = d_resized.squeeze(0).squeeze(0)  # [H,W]
                except Exception as e:
                    print(
                        f"[Warning] Depth resize to {target_hw} failed ({type(e).__name__}: {e}); "
                        f"falling back to original depth resolution {depth_np.shape[:2]}",
                        flush=True,
                    )
                    depth = torch.from_numpy(depth_np).float()
            else:
                depth = torch.from_numpy(depth_np).float()
        else:
            print(f"[Warning] Depth file not found (tried {depth_path})")

    result = {
        "img": image,                        # [3, H, W] tensor (ImageNet normalized)
        "image_mask": image_mask,             # [1, H, W] bool
        "camera_pose": pose,                  # [4, 4] tensor
        "pose": pose,
        "pose_inv": pose_inv,
        "camera_intrinsics": intrinsics,      # [3, 3] tensor
        "intrinsics": intrinsics,
        "intrinsics_inv": intrinsics_inv,
        "coords": coords,
        "filename": filename,
    }
    if depth is not None:
        result["depthmap"] = depth            # [H, W] tensor in meters
        result["depth"] = depth

    return result


# =============================================================================
# Main Entry Point
# =============================================================================

def parse_args() -> ExtractionConfig:
    """解析命令行，构建 ExtractionConfig。"""
    parser = argparse.ArgumentParser(
        description='BSE memory extraction — 从多视角 RGB-D 提取 3D memory（体素特征 + 归一化统计）。'
    )

    # 位置参数
    parser.add_argument(
        'dataset_path',
        type=str,
        help='数据集路径：WAI 时为数据集 ROOT；ace loader 时为 ACE 数据根目录。',
    )
    parser.add_argument(
        'output_path',
        type=str,
        help='输出 memory 文件路径（.pt），含 pooled 特征与 view 元数据。',
    )

    parser.add_argument(
        '--n_memory',
        type=int,
        default=100,
        help='抽取多少个训练帧作为 memory 视角（均匀步长遍历数据集索引）。',
    )

    parser.add_argument(
        '--use_model',
        type=str,
        default='mapanything',
        choices=['mapanything', 'dinov2'],
        help='特征提取器：mapanything（默认，与 fps_memory 一致，中间层多尺度）；'
        'dinov2（DINOv2 块特征，DPT 式融合）。',
    )
    parser.add_argument(
        '--model_str',
        type=str,
        default=None,
        help='MapAnything 注册的模型名字符串（如 mapanything_store_intermediates_ace）。',
    )
    parser.add_argument(
        '--model_config',
        type=str,
        default=None,
        help='Hydra/OmegaConf 模型 YAML 路径；省略则用内置 default。',
    )
    parser.add_argument(
        '--model_checkpoint',
        type=str,
        default=None,
        help='MapAnything 权重文件；省略时 main() 可能使用固定默认路径。',
    )
    parser.add_argument(
        '--dinov2_checkpoint',
        type=str,
        default=_DEFAULT_DINO_CHECKPOINT,
        help='DINOv2 ViT 预训练权重（--use_model dinov2；mapanything 的 encoder 也使用同一解析路径）。',
    )
    parser.add_argument(
        '--dinov2_intermediate_layers',
        type=int,
        nargs='*',
        default=None,
        help='DINOv2 参与融合的 block 索引；默认 8 层 [2,5,8,11,14,17,20,23]；'
        '仅传 23 表示只用最后一层（单尺度）。',
    )

    parser.add_argument(
        '--use_bse',
        action='store_true',
        default=True,
        help='启用 BSE（边界感知）体素池化；关闭时当前实现会报错未实现。',
    )
    parser.add_argument(
        '--voxel_size',
        type=float,
        default=0.05,
        help='体素网格边长（米）；越小越细、内存与时间开销越大。',
    )
    parser.add_argument(
        '--pool_mode',
        type=str,
        default='bse',
        choices=['bse', 'pooled', 'simple', 'hybrid'],
        help=(
            'pooling 模式：'
            'bse=voxel hash + Otsu split；'
            'pooled=map-anything 原版 pooled_GT baseline，强制 raw 点全局 SOR 后一次 voxel mean，忽略 BSE/Otsu/prepool/global_merge 开关；'
            'simple=底层简单 voxel mean；hybrid=全局 simple mean + 选择性 split。'
        ),
    )
    parser.add_argument(
        '--prepool_mode',
        type=str,
        default='per_view',
        choices=['per_view', 'global_only'],
        help='预池化模式：per_view=每个 view 先池化；global_only=直接缓存 raw points，Pass 2 再全局池化。',
    )
    parser.add_argument(
        '--use_otsu',
        type=lambda x: str(x).strip().lower() in ('1', 'true', 'yes', 'on'),
        nargs='?',
        const='true',
        default=True,
        help='BSE 内对深度分箱使用 Otsu 自动阈值（适应稀疏/不均匀深度）。',
    )
    parser.add_argument(
        '--unimodal_threshold',
        type=float,
        default=0.02,
        help='BSE 内 feature similarity std 低于该值时认为近似单峰并跳过 split。',
    )
    parser.add_argument(
        '--hybrid_split_min_std',
        type=float,
        default=0.05,
        help='Hybrid: per-voxel cosine-sim std 阈值，低于此值不 split（默认 0.05）。',
    )
    parser.add_argument(
        '--hybrid_min_cluster_size',
        type=int,
        default=4,
        help='Hybrid: voxel 内点数低于此值不 split（默认 4）。',
    )
    parser.add_argument(
        '--hybrid_min_split_points',
        type=int,
        default=2,
        help='Hybrid: Otsu 后任一分支点数低于此值则回退 simple mean（默认 2）。',
    )
    parser.add_argument(
        '--global_merge',
        type=lambda x: str(x).strip().lower() in ('1', 'true', 'yes', 'on'),
        default=True,
        help='Pass 2 对所有 view 的 pooled 点做一次全局体素合并（默认 True）。',
    )
    parser.add_argument(
        '--global_merge_voxel_size',
        type=float,
        default=None,
        help='Pass 2 全局体素合并的 voxel size；默认 None 时沿用 --voxel_size。',
    )

    parser.add_argument(
        '--device',
        type=str,
        default='cuda:0',
        help='计算设备，如 cuda:0；若 shell 设置 CUDA_VISIBLE_DEVICES，则 0 表示可见首卡。',
    )
    parser.add_argument(
        '--use_patch_based',
        action='store_true',
        default=False,
        help='使用基于 patch 的提取路径（与双线性网格路径相对，见 two_pass_processing）。',
    )
    parser.add_argument(
        '--depth_valid_range',
        type=float,
        nargs=2,
        default=(0.1, 6.0),
        help='有效深度区间 (最小米, 最大米)；反投影与可视化时过滤。',
    )
    parser.add_argument(
        '--patch_depth_sampling',
        type=str,
        default='nearest_valid',
        choices=['nearest', 'median', 'nearest_valid'],
        help='每个特征网格中心的深度：nearest 最近像素；median 3×3 中位数；'
        'nearest_valid 在邻域找最近有效深度（稀疏 Indoor6 常用，对齐 fps_memory）。',
    )

    parser.add_argument(
        '--enable_sor',
        action='store_true',
        default=False,
        help='对反投影得到的点云做统计离群点剔除（Statistical Outlier Removal）。',
    )
    parser.add_argument(
        '--sor_k',
        type=int,
        default=20,
        help='SOR：每个点考虑的近邻个数 k。',
    )
    parser.add_argument(
        '--sor_std_ratio',
        type=float,
        default=2.0,
        help='SOR：到 k 邻域均值的距离超过 std_ratio 倍标准差则剔除。',
    )

    parser.add_argument(
        '--temp_dir',
        type=str,
        default='/dev/shm',
        help='两遍处理时各 view 的 chunk 与断点文件目录（建议内存盘减少 IO）。',
    )
    parser.add_argument(
        '--dataset_type',
        type=str,
        default='auto',
        choices=['auto', '7scenes', 'indoor6', 'custom'],
        help='数据集类型；auto 时根据路径启发式检测。',
    )
    parser.add_argument(
        '--scene_name',
        type=str,
        default=None,
        help='WAI 场景名（如 chess_train）；省略则用 dataset_path 的最后一段目录名。',
    )
    parser.add_argument(
        '--dataset_loader',
        type=str,
        default='wai',
        choices=['wai', 'ace'],
        help='数据加载：wai=MapAnything WAI；ace=ACE CamLocDatasetDINOv2。',
    )
    parser.add_argument(
        '--wai_view_mode',
        type=str,
        default='fps_flat',
        choices=sorted(WAI_VIEW_MODE_ALIASES),
        help=(
            'WAI 视图加载协议：fps_flat/fps_strict=每个 FPS index 只加载 1 张图，实际推理输入严格等于 FPS list；'
            'original_multiview/first_fps_covis=复现 map-anything fps_memory.sh，首个 FPS anchor 一次返回 n_memory 个 covisibility views；'
            'anchor_support/asb=当前 Reference-aware Anchor+Support 选帧。'
        ),
    )
    parser.add_argument(
        '--anchor_support_alpha',
        type=float,
        default=1.0,
        help='anchor_support 中 covisibility 项的指数权重；越大越偏向共视性。',
    )
    parser.add_argument(
        '--anchor_support_eps',
        type=float,
        default=1e-6,
        help='anchor_support score epsilon，避免零值导致分数为 0。',
    )
    parser.add_argument(
        '--anchor_support_tau',
        type=float,
        default=-1.0,
        help='anchor_support fill 阶段硬共视阈值；<0 表示关闭硬阈值，仅使用 soft score。',
    )
    parser.add_argument(
        '--covis_ref_lambda',
        type=float,
        default=0.0,
        help='anchor_support 下的 reference 距离软惩罚强度；0 表示不惩罚远离 reference 的候选。',
    )
    parser.add_argument(
        '--covis_anchor_count',
        type=int,
        default=0,
        help='anchor_support 模式下的 anchor 数；<=0 时按 n_memory 自动取保守值。',
    )
    parser.add_argument(
        '--covis_support_per_anchor',
        type=int,
        default=3,
        help='anchor_support 模式下每个 anchor 选择的高共视 support 数。',
    )
    parser.add_argument(
        '--covis_support_tau',
        type=float,
        default=1e-4,
        help='anchor_support 模式下 strong support neighbor 的共视阈值。',
    )
    parser.add_argument(
        '--covis_support_min_neighbors',
        type=int,
        default=3,
        help='anchor_support 模式下 anchor 至少需要多少个 strong support neighbors。',
    )
    parser.add_argument(
        '--covis_safe_dist_to_ref',
        type=float,
        default=3.0,
        help='anchor_support 模式下视为 reference-safe 的距离阈值；远视角由 budget 控制。',
    )
    parser.add_argument(
        '--covis_far_view_budget',
        type=int,
        default=8,
        help='anchor_support 模式下允许 dist_to_ref 超过 safe_dist 的总视角数。',
    )
    parser.add_argument(
        '--covis_far_anchor_budget',
        type=int,
        default=2,
        help='anchor_support 模式下允许 dist_to_ref 超过 safe_dist 的 anchor 数。',
    )
    parser.add_argument(
        '--covis_coverage_beta',
        type=float,
        default=1.0,
        help='anchor_support/fill 阶段 coverage_gain 项指数；越大越偏向覆盖收益。',
    )
    parser.add_argument(
        '--covis_candidate_pool_ratio',
        type=float,
        default=1.0,
        help='anchor_support 内部候选池倍率；>1 时先多选候选再裁回 n_memory；<=0 表示使用全场景候选池（仅建议诊断覆盖，不建议 MapAnything 单 batch 训练用）。',
    )
    parser.add_argument(
        '--covis_ma_safe_asb',
        action='store_true',
        help='启用 MapAnything-safe ASB：先取 reference 高共视候选池，再在池内 coverage-greedy，避免全局低共视 batch。',
    )
    parser.add_argument(
        '--covis_ma_safe_pool_ratio',
        type=float,
        default=3.0,
        help='MapAnything-safe ASB 的 root-covis 候选池倍率；候选数约为 n_memory * ratio。',
    )
    parser.add_argument(
        '--covis_native_group_asb',
        action='store_true',
        help='启用 native-group ASB：搜索覆盖较好的 WAI native covis anchor，然后由 dataset[anchor] 返回高共视 n_memory views。',
    )
    parser.add_argument(
        '--covis_native_anchor_candidates',
        type=int,
        default=64,
        help='native-group ASB 搜索的 FPS anchor 候选数量。',
    )
    parser.add_argument(
        '--covis_adaptive_asb',
        action='store_true',
        help='启用 adaptive ASB：按 scene meta 统计自适应调整 coverage/ref-distance 约束。',
    )
    parser.add_argument(
        '--covis_adaptive_view_count',
        action='store_true',
        help='anchor_support adaptive 模式下把 n_memory 仅作为 max views；连通性和覆盖足够时允许少选几张。',
    )
    parser.add_argument(
        '--covis_disable_adaptive_scene_stats',
        action='store_true',
        help='关闭 adaptive ASB 的场景统计自适应；关闭后使用手动传入的米制阈值。',
    )
    parser.add_argument(
        '--covis_adaptive_min_views',
        type=int,
        default=28,
        help='adaptive ASB 下至少保留的视角数；超过后才允许 early stop。',
    )
    parser.add_argument(
        '--covis_ref_dist_max_limit',
        type=float,
        default=4.2,
        help='adaptive ASB 候选硬约束：加入后 selected ref_dist_max 不超过该值；<=0 关闭。',
    )
    parser.add_argument(
        '--covis_ref_dist_mean_limit',
        type=float,
        default=2.7,
        help='adaptive ASB 候选硬约束：加入后 selected ref_dist_mean 不超过该值；<=0 关闭。',
    )
    parser.add_argument(
        '--covis_min_coverage_gain',
        type=float,
        default=0.01,
        help='adaptive ASB 低收益判定阈值：coverage_mean 改善小于该值计入 patience。',
    )
    parser.add_argument(
        '--covis_gain_patience',
        type=int,
        default=2,
        help='adaptive ASB 低 coverage gain 连续次数阈值；达到且 coverage target 满足后停止。',
    )
    parser.add_argument(
        '--covis_target_coverage_mean',
        type=float,
        default=0.55,
        help='adaptive ASB coverage target：nearest-center mean 小于等于该值才允许风险/低收益停止；<=0 关闭。',
    )
    parser.add_argument(
        '--covis_target_coverage_max',
        type=float,
        default=2.0,
        help='adaptive ASB coverage target：nearest-center max 小于等于该值才允许风险/低收益停止；<=0 关闭。',
    )
    parser.add_argument(
        '--ray_pool_strategy',
        type=str,
        default='mean',
        choices=['mean', 'dominant', 'first', 'all'],
        help='同一体素内多条视线方向的聚合方式（mean 平均方向等）。',
    )
    parser.add_argument(
        '--save_all_ray_strategies',
        action='store_true',
        default=True,
        help='为 true 时保存多种 ray 池化结果便于对比实验。',
    )

    parser.add_argument(
        '--pose_eval_translation_ok_m',
        type=float,
        default=0.1,
        help='MapAnything infer 后：预测位姿相对 GT 的平移误差（米）低于此视为 OK，否则 HIGH。',
    )
    parser.add_argument(
        '--pose_eval_strict',
        action='store_true',
        default=False,
        help='为 true 时：任一非参考视角平移误差超阈值则进程以退出码 1 结束。',
    )
    parser.add_argument(
        '--pose_prune_after_infer',
        action='store_true',
        default=False,
        help='第一次 MapAnything infer 后按 PoseEval 删除最差非 reference views，并用保留 views 重新 infer。',
    )
    parser.add_argument(
        '--pose_prune_min_views',
        type=int,
        default=20,
        help='PoseEval 剪枝后至少保留多少个 views（包含 reference）。',
    )
    parser.add_argument(
        '--pose_prune_min_keep_ratio',
        type=float,
        default=0.70,
        help='PoseEval 剪枝后至少保留的比例；与 pose_prune_min_views 取更严格者。',
    )
    parser.add_argument(
        '--pose_prune_mad_k',
        type=float,
        default=2.5,
        help='PoseEval 剪枝日志中的 robust threshold 系数；实际 drop 优先裁掉 translation_ok_m 以上的最高误差。',
    )

    parser.add_argument(
        '--debug_dump',
        action='store_true',
        default=False,
        help='在输出目录下导出 debug_dump（raw/processed 的关键统计与可选数值快照），用于对齐不同 loader 的数据差异。',
    )
    parser.add_argument(
        '--debug_dump_max_views',
        type=int,
        default=3,
        help='debug_dump 最多导出多少个视角（默认 3，建议 1~5）。',
    )
    parser.add_argument(
        '--debug_dump_save_npz',
        action='store_true',
        default=False,
        help='debug_dump 额外保存少量数值快照（.npz），方便离线精确对比；会显著增大输出体积。',
    )
    parser.add_argument(
        '--feature_diagnostics',
        action='store_true',
        default=False,
        help='保存 BSE 生成阶段的 feature 相似度诊断：voxel-mean 相似度、同 voxel 跨视图相似度和随机负样本直方图。'
        '当前只对 prepool_mode=global_only 的 raw points/features 有完整意义。',
    )
    parser.add_argument(
        '--feature_diag_max_points_per_voxel',
        type=int,
        default=32,
        help='feature_diagnostics：每个空间 voxel 最多采样多少个 raw points 做 pairwise 统计。',
    )
    parser.add_argument(
        '--feature_diag_max_pair_samples',
        type=int,
        default=200000,
        help='feature_diagnostics：最多累计多少条 same-voxel cross-view pair 样本。',
    )
    parser.add_argument(
        '--feature_diag_bins',
        type=int,
        default=100,
        help='feature_diagnostics：相似度/距离直方图的 bin 数。',
    )

    args = parser.parse_args()

    config = ExtractionConfig(
        dataset_path=args.dataset_path,
        output_path=args.output_path,
        n_memory=args.n_memory,
        device=args.device,
        voxel_size=args.voxel_size,
        use_bse=args.use_bse,
        pool_mode=args.pool_mode,
        prepool_mode=args.prepool_mode,
        use_otsu=args.use_otsu,
        unimodal_threshold=args.unimodal_threshold,
        hybrid_split_min_std=args.hybrid_split_min_std,
        hybrid_min_cluster_size=args.hybrid_min_cluster_size,
        hybrid_min_split_points=args.hybrid_min_split_points,
        global_merge=bool(args.global_merge),
        global_merge_voxel_size=args.global_merge_voxel_size,
        depth_valid_range=tuple(args.depth_valid_range),
        patch_depth_sampling=args.patch_depth_sampling,
        temp_dir=args.temp_dir,
        model_str=args.model_str,
        model_config=args.model_config,
        model_checkpoint=args.model_checkpoint,
        dinov2_checkpoint=resolve_dinov2_checkpoint_path(args.dinov2_checkpoint),
        use_patch_based=args.use_patch_based,
        enable_sor=args.enable_sor,
        sor_k=args.sor_k,
        sor_std_ratio=args.sor_std_ratio,
        dataset_type=args.dataset_type,
        scene_name=args.scene_name,
        dataset_loader=args.dataset_loader,
        wai_view_mode=canonicalize_wai_view_mode(args.wai_view_mode),
        anchor_support_alpha=float(args.anchor_support_alpha),
        anchor_support_eps=float(args.anchor_support_eps),
        anchor_support_tau=float(args.anchor_support_tau),
        covis_ref_lambda=float(args.covis_ref_lambda),
        covis_anchor_count=int(args.covis_anchor_count),
        covis_support_per_anchor=int(args.covis_support_per_anchor),
        covis_support_tau=float(args.covis_support_tau),
        covis_support_min_neighbors=int(args.covis_support_min_neighbors),
        covis_safe_dist_to_ref=float(args.covis_safe_dist_to_ref),
        covis_far_view_budget=int(args.covis_far_view_budget),
        covis_far_anchor_budget=int(args.covis_far_anchor_budget),
        covis_coverage_beta=float(args.covis_coverage_beta),
        covis_candidate_pool_ratio=float(args.covis_candidate_pool_ratio),
        covis_ma_safe_asb=bool(args.covis_ma_safe_asb),
        covis_ma_safe_pool_ratio=float(args.covis_ma_safe_pool_ratio),
        covis_native_group_asb=bool(args.covis_native_group_asb),
        covis_native_anchor_candidates=int(args.covis_native_anchor_candidates),
        covis_adaptive_asb=bool(args.covis_adaptive_asb),
        covis_adaptive_scene_stats=not bool(args.covis_disable_adaptive_scene_stats),
        covis_adaptive_view_count=bool(args.covis_adaptive_view_count),
        covis_adaptive_min_views=int(args.covis_adaptive_min_views),
        covis_ref_dist_max_limit=float(args.covis_ref_dist_max_limit),
        covis_ref_dist_mean_limit=float(args.covis_ref_dist_mean_limit),
        covis_min_coverage_gain=float(args.covis_min_coverage_gain),
        covis_gain_patience=int(args.covis_gain_patience),
        covis_target_coverage_mean=float(args.covis_target_coverage_mean),
        covis_target_coverage_max=float(args.covis_target_coverage_max),
        ray_pool_strategy=args.ray_pool_strategy,
        save_all_ray_strategies=args.save_all_ray_strategies,
        use_model=args.use_model,
        dinov2_intermediate_layers=args.dinov2_intermediate_layers,
        pose_eval_translation_ok_m=args.pose_eval_translation_ok_m,
        pose_eval_strict=args.pose_eval_strict,
        pose_prune_after_infer=args.pose_prune_after_infer,
        pose_prune_min_views=int(args.pose_prune_min_views),
        pose_prune_min_keep_ratio=float(args.pose_prune_min_keep_ratio),
        pose_prune_mad_k=float(args.pose_prune_mad_k),
        debug_dump=args.debug_dump,
        debug_dump_max_views=int(args.debug_dump_max_views),
        debug_dump_save_npz=args.debug_dump_save_npz,
        feature_diagnostics=args.feature_diagnostics,
        feature_diag_max_points_per_voxel=int(args.feature_diag_max_points_per_voxel),
        feature_diag_max_pair_samples=int(args.feature_diag_max_pair_samples),
        feature_diag_bins=int(args.feature_diag_bins),
    )

    if config.pool_mode == "pooled":
        # Pooled is an isolated baseline mode: concatenate raw points from all
        # views, then run exactly one global voxel mean. BSE/Otsu/per-view
        # prepooling/global post-merge switches must not alter this branch.
        config.prepool_mode = "global_only"
        config.use_otsu = False
        config.global_merge = False
        config.global_merge_voxel_size = None
        if not config.enable_sor:
            print("[Config] pool_mode=pooled: enabling original map-anything global SOR.", flush=True)
        config.enable_sor = True
        if os.path.basename(config.output_path) == "memory_bse.pt":
            config.output_path = os.path.join(os.path.dirname(config.output_path), "memory_pooled.pt")
            print(
                f"[Config] pool_mode=pooled: output basename adjusted to {config.output_path}",
                flush=True,
            )

    return config


def main():
    """Main entry point."""
    config = parse_args()
    device = torch.device(config.device)

    # Auto-derive scene_name from dataset_path if not provided
    scene_name = config.scene_name
    if scene_name is None:
        scene_name = os.path.basename(config.dataset_path.rstrip('/'))
        # ACE loader: dataset_path is like .../pgt_7scenes_heads/train → use parent dir name "pgt_7scenes_heads"
        # Strip common suffixes like _train, _test, _val, or plain train/test/val
        if config.dataset_loader == "ace" and scene_name in ("train", "test", "val"):
            parent = os.path.basename(os.path.dirname(config.dataset_path.rstrip('/')))
            if parent:
                scene_name = parent
    dataset_type = config.dataset_type
    if dataset_type == "auto":
        dataset_type = detect_dataset_type(config.dataset_path)

    print(f"[BSE Memory] Dataset: {config.dataset_path}")
    print(f"[BSE Memory] Scene: {scene_name} (type={dataset_type})")
    print(f"[BSE Memory] Output: {config.output_path}")
    print(f"[BSE Memory] N_MEMORY: {config.n_memory}, BSE: {config.use_bse}")
    print(f"[BSE Memory] Pool mode: {config.pool_mode}")
    print(f"[BSE Memory] Prepool mode: {config.prepool_mode}")
    print(f"[BSE Memory] Voxel size: {config.voxel_size}, Otsu: {config.use_otsu}")
    print(f"[BSE Memory] Unimodal threshold: {config.unimodal_threshold}")
    print(f"[BSE Memory] Global merge: {config.global_merge}")
    print(
        f"[BSE Memory] Global merge voxel size: "
        f"{config.global_merge_voxel_size if config.global_merge_voxel_size is not None else config.voxel_size}"
    )
    print(
        f"[Config] patch_depth_sampling: {config.patch_depth_sampling} "
        f"(align map-anything fps_memory: nearest_valid for sparse Indoor6 depth)",
        flush=True,
    )
    print(
        f"[Config] Depth valid range: [{config.depth_valid_range[0]:.3f}, {config.depth_valid_range[1]:.3f}] m",
        flush=True,
    )
    if config.feature_diagnostics:
        print(
            f"[FeatureDiag] enabled: max_points_per_voxel={config.feature_diag_max_points_per_voxel}, "
            f"max_pair_samples={config.feature_diag_max_pair_samples}, bins={config.feature_diag_bins}",
            flush=True,
        )
        if config.prepool_mode != "global_only":
            print(
                "[FeatureDiag] WARNING: full cross-view raw-feature diagnostics require prepool_mode=global_only; "
                "current run may not contain raw cross-view chunks.",
                flush=True,
            )

    if config.pool_mode == "hybrid":
        if config.prepool_mode != "global_only":
            raise ValueError(
                f"pool_mode=hybrid requires prepool_mode=global_only, "
                f"got prepool_mode={config.prepool_mode}"
            )
        if config.global_merge:
            raise ValueError(
                "pool_mode=hybrid requires global_merge=False. "
                "Global merge would undo selective splits."
            )
        print(
            f"[BSE Memory] Hybrid config: "
            f"split_min_std={config.hybrid_split_min_std}, "
            f"min_cluster_size={config.hybrid_min_cluster_size}, "
            f"min_split_points={config.hybrid_min_split_points}",
            flush=True,
        )

    # Initialize modules
    welford = WelfordNormalizer()
    if config.use_bse:
        pooler = BSEPooler(
            voxel_size=config.voxel_size,
            pool_mode=config.pool_mode,
            use_otsu=config.use_otsu,
            unimodal_threshold=config.unimodal_threshold,
            ray_pool_strategy=config.ray_pool_strategy,
            save_all_ray_strategies=config.save_all_ray_strategies,
            hybrid_split_min_std=config.hybrid_split_min_std,
            hybrid_min_cluster_size=config.hybrid_min_cluster_size,
            hybrid_min_split_points=config.hybrid_min_split_points,
        )
    else:
        raise NotImplementedError("Vanilla voxel pooling not implemented")

    # Load dataset
    print("[BSE Memory] Loading dataset...", flush=True)
    if config.dataset_loader == "wai" and (
        config.wai_view_mode == "original_multiview"
        or (config.wai_view_mode == "anchor_support" and config.covis_native_group_asb)
    ):
        # Reproduce map-anything fps_memory.sh: with sequential WAI datasets,
        # dataset[idx] returns an anchor-centered covisibility sample of
        # num_views views. The first FPS anchor can therefore fill all memory
        # slots by itself.
        dataset_num_views = config.n_memory
    else:
        # Strict FPS protocol: one dataset view per FPS index, so the actual
        # model inputs match select_optimal_memory_indices() exactly.
        dataset_num_views = 1
    print(
        f"[Data] WAI view mode: {config.wai_view_mode} (dataset_num_views={dataset_num_views})",
        flush=True,
    )
    train_dataset = load_dataset(
        config.dataset_path,
        dataset_type=dataset_type,
        scene_name=scene_name,
        n_views=dataset_num_views,
        dataset_loader=config.dataset_loader,
    )
    print(f"[BSE Memory] Train dataset: {len(train_dataset)} frames")

    requested_memory_views = int(config.n_memory)
    probe_memory_views = requested_memory_views
    if config.pose_prune_after_infer:
        keep_ratio = max(float(config.pose_prune_min_keep_ratio), 1e-3)
        probe_memory_views = int(np.ceil(float(requested_memory_views) / keep_ratio))
        probe_memory_views = max(requested_memory_views, probe_memory_views)
        probe_memory_views = min(probe_memory_views, len(train_dataset))
        print(
            f"[PosePrune] Probe selection enabled: target_views={requested_memory_views}, "
            f"probe_views={probe_memory_views}, keep_ratio={config.pose_prune_min_keep_ratio:.3f}",
            flush=True,
        )

    # Select memory views
    print("[BSE Memory] Selecting memory views...", flush=True)
    memory_indices, scene_center_from_cameras, memory_index_groups = select_memory_views(
        train_dataset,
        probe_memory_views,
        dataset_path=config.dataset_path,
        scene_name=scene_name,
        selection_mode=config.wai_view_mode,
        covis_alpha=config.anchor_support_alpha,
        covis_eps=config.anchor_support_eps,
        covis_tau=config.anchor_support_tau,
        covis_ref_lambda=config.covis_ref_lambda,
        covis_anchor_count=config.covis_anchor_count,
        covis_support_per_anchor=config.covis_support_per_anchor,
        covis_support_tau=config.covis_support_tau,
        covis_support_min_neighbors=config.covis_support_min_neighbors,
        covis_safe_dist_to_ref=config.covis_safe_dist_to_ref,
        covis_far_view_budget=config.covis_far_view_budget,
        covis_far_anchor_budget=config.covis_far_anchor_budget,
        covis_coverage_beta=config.covis_coverage_beta,
        covis_candidate_pool_ratio=config.covis_candidate_pool_ratio,
        covis_ma_safe_asb=config.covis_ma_safe_asb,
        covis_ma_safe_pool_ratio=config.covis_ma_safe_pool_ratio,
        covis_native_group_asb=config.covis_native_group_asb,
        covis_native_anchor_candidates=config.covis_native_anchor_candidates,
        covis_adaptive_asb=config.covis_adaptive_asb,
        covis_adaptive_scene_stats=config.covis_adaptive_scene_stats,
        covis_adaptive_view_count=config.covis_adaptive_view_count,
        covis_adaptive_allow_early_stop=not config.pose_prune_after_infer,
        covis_adaptive_min_views=config.covis_adaptive_min_views,
        covis_ref_dist_max_limit=config.covis_ref_dist_max_limit,
        covis_ref_dist_mean_limit=config.covis_ref_dist_mean_limit,
        covis_min_coverage_gain=config.covis_min_coverage_gain,
        covis_gain_patience=config.covis_gain_patience,
        covis_target_coverage_mean=config.covis_target_coverage_mean,
        covis_target_coverage_max=config.covis_target_coverage_max,
    )
    print(f"[BSE Memory] Selected {len(memory_indices)} views", flush=True)
    print(f"[Data] Selected memory indices ({len(memory_indices)}): {memory_indices}", flush=True)
    output_run_dir = os.path.dirname(os.path.abspath(config.output_path))
    _export_selection_coverage_diagnostics(
        train_dataset,
        config.dataset_path,
        scene_name,
        memory_indices,
        output_run_dir,
        max_views=requested_memory_views,
        adaptive_min_views=config.covis_adaptive_min_views,
    )

    # Prepare input views (fast path: load directly from disk, skip WAI heavy processing)
    print("[BSE Memory] Preparing input views...", flush=True)
    raw_batches = []  # Store original data for depth/intrinsics access
    memory_views = []  # Store processed data for model input
    memory_gt_poses: List[torch.Tensor] = []  # For post-infer pose vs GT check (MapAnything)
    loaded_view_records: List[Dict[str, Any]] = []
    did_print_compare = False

    def _stat_any(x: Any) -> str:
        try:
            if isinstance(x, torch.Tensor):
                t = x.detach()
                mn = float(t.min()) if t.numel() else float("nan")
                mx = float(t.max()) if t.numel() else float("nan")
                mean = float(t.float().mean()) if t.numel() else float("nan")
                return f"tensor dtype={t.dtype} shape={tuple(t.shape)} min={mn:.4g} max={mx:.4g} mean={mean:.4g}"
            a = np.asarray(x)
            if a.size == 0:
                return f"array dtype={a.dtype} shape={a.shape} (empty)"
            return (
                f"array dtype={a.dtype} shape={a.shape} "
                f"min={float(np.nanmin(a)):.4g} max={float(np.nanmax(a)):.4g} mean={float(np.nanmean(a)):.4g}"
            )
        except Exception as e:
            return f"<stat failed: {type(e).__name__}: {e}>"

    def _pick_first(raw: Any) -> Any:
        return raw[0] if isinstance(raw, (list, tuple)) and raw else raw

    def _print_loader_compare(raw_view: Dict[str, Any], processed: Dict[str, Any]) -> None:
        nonlocal did_print_compare
        if did_print_compare:
            return
        did_print_compare = True

        def _keys(d: Dict[str, Any]) -> List[str]:
            return sorted([str(k) for k in d.keys()])

        print("\n===================== LoaderCompare (view_00) =====================", flush=True)
        print(f"[Compare] dataset_loader={config.dataset_loader}, dataset_type={dataset_type}, scene_name={scene_name}", flush=True)
        print(f"[Compare] raw keys: {_keys(raw_view)}", flush=True)
        print(f"[Compare] processed keys: {_keys(processed)}", flush=True)

        raw_img = _pick_first(raw_view.get("img", raw_view.get("image", raw_view.get("img_uint8"))))
        print(f"[Compare] raw img: {_stat_any(raw_img)}", flush=True)
        if "img" in processed:
            print(f"[Compare] processed img: {_stat_any(processed['img'])}", flush=True)

        raw_depth = _pick_first(
            raw_view.get("depthmap", raw_view.get("depth", raw_view.get("gt_depth", raw_view.get("depth_z"))))
        )
        print(f"[Compare] raw depth: {_stat_any(raw_depth) if raw_depth is not None else '<None>'}", flush=True)

        raw_intr = _pick_first(raw_view.get("camera_intrinsics", raw_view.get("intrinsics")))
        raw_pose = _pick_first(raw_view.get("camera_pose", raw_view.get("pose")))
        if raw_intr is not None:
            print(f"[Compare] raw intrinsics: {_stat_any(raw_intr)}", flush=True)
        if raw_pose is not None:
            print(f"[Compare] raw pose: {_stat_any(raw_pose)}", flush=True)

        if "depth_z" in processed:
            print(f"[Compare] processed depth_z: {_stat_any(processed['depth_z'])}", flush=True)
        if "intrinsics" in processed:
            print(f"[Compare] processed intrinsics: {_stat_any(processed['intrinsics'])}", flush=True)
        if "camera_poses" in processed:
            print(f"[Compare] processed camera_poses: {_stat_any(processed['camera_poses'])}", flush=True)
        print("==================================================================\n", flush=True)

    flat_view_lookup: Dict[Tuple[str, str], int] = {}
    flat_view_list = getattr(train_dataset, "flat_view_list", None)
    if isinstance(flat_view_list, list):
        for flat_i, item in enumerate(flat_view_list):
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                flat_view_lookup[(str(item[0]), str(item[1]))] = flat_i
    memory_index_to_group: Dict[int, int] = {}
    for group_i, group in enumerate(memory_index_groups):
        for idx in group:
            memory_index_to_group[int(idx)] = int(group_i)

    def _first_scalar_str(x: Any) -> Optional[str]:
        if isinstance(x, (list, tuple)):
            x = x[0] if x else None
        if x is None:
            return None
        return str(x)

    def _view_source_name(view: Any) -> str:
        if not isinstance(view, dict):
            return "<non-dict-view>"
        for key in ("instance", "image_path", "rgb_path", "filename", "path"):
            val = _first_scalar_str(view.get(key))
            if val:
                return val
        return "<unknown>"

    def _lookup_actual_flat_idx(view: Any) -> Optional[int]:
        if not isinstance(view, dict) or not flat_view_lookup:
            return None
        label = _first_scalar_str(view.get("label", view.get("scene_name", scene_name))) or str(scene_name)
        source = _view_source_name(view)
        frame = os.path.basename(source)
        candidates = [frame]
        stem, ext = os.path.splitext(frame)
        if ext:
            candidates.append(stem)
        for cand in candidates:
            idx = flat_view_lookup.get((label, cand))
            if idx is not None:
                return idx
        return None

    def _write_loaded_view_records(records: List[Dict[str, Any]]) -> None:
        print("[Data] Actual model input views:", flush=True)
        for rec in records:
            print(
                "  [view_%02d] outer=%s inner=%s fps_idx_at_pos=%s actual_flat_idx=%s match_fps=%s source=%s"
                % (
                    int(rec["view_idx"]),
                    rec.get("outer_index"),
                    rec.get("inner_index"),
                    rec.get("fps_idx_at_view_position"),
                    rec.get("actual_flat_idx"),
                    rec.get("matches_fps_at_view_position"),
                    rec.get("source"),
                ),
                flush=True,
            )
        try:
            out_path = os.path.join(output_run_dir, "loaded_model_input_views.json")
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "selected_memory_indices": memory_indices,
                        "wai_view_mode": config.wai_view_mode,
                        "dataset_num_views": dataset_num_views,
                        "anchor_support": {
                            "alpha": float(config.anchor_support_alpha),
                            "eps": float(config.anchor_support_eps),
                            "tau": float(config.anchor_support_tau),
                            "ref_lambda": float(config.covis_ref_lambda),
                            "anchor_count": int(config.covis_anchor_count),
                            "support_per_anchor": int(config.covis_support_per_anchor),
                            "support_tau": float(config.covis_support_tau),
                            "support_min_neighbors": int(config.covis_support_min_neighbors),
                            "safe_dist_to_ref": float(config.covis_safe_dist_to_ref),
                            "far_view_budget": int(config.covis_far_view_budget),
                            "far_anchor_budget": int(config.covis_far_anchor_budget),
                            "coverage_beta": float(config.covis_coverage_beta),
                            "candidate_pool_ratio": float(config.covis_candidate_pool_ratio),
                            "ma_safe_asb": bool(config.covis_ma_safe_asb),
                            "ma_safe_pool_ratio": float(config.covis_ma_safe_pool_ratio),
                            "native_group_asb": bool(config.covis_native_group_asb),
                            "native_anchor_candidates": int(config.covis_native_anchor_candidates),
                            "adaptive_asb": bool(config.covis_adaptive_asb),
                            "adaptive_scene_stats": bool(config.covis_adaptive_scene_stats),
                            "adaptive_view_count": bool(config.covis_adaptive_view_count),
                            "adaptive_allow_early_stop": not bool(config.pose_prune_after_infer),
                            "adaptive_min_views": int(config.covis_adaptive_min_views),
                            "ref_dist_max_limit": float(config.covis_ref_dist_max_limit),
                            "ref_dist_mean_limit": float(config.covis_ref_dist_mean_limit),
                            "min_coverage_gain": float(config.covis_min_coverage_gain),
                            "gain_patience": int(config.covis_gain_patience),
                            "target_coverage_mean": float(config.covis_target_coverage_mean),
                            "target_coverage_max": float(config.covis_target_coverage_max),
                            "pose_prune_after_infer": bool(config.pose_prune_after_infer),
                            "pose_prune_min_views": int(config.pose_prune_min_views),
                            "pose_prune_min_keep_ratio": float(config.pose_prune_min_keep_ratio),
                            "pose_prune_mad_k": float(config.pose_prune_mad_k),
                        },
                        "memory_index_groups": memory_index_groups,
                        "records": records,
                    },
                    f,
                    ensure_ascii=False,
                    indent=2,
                )
            print(f"[Data] Loaded model input view records saved to: {out_path}", flush=True)
        except Exception as e:
            print(f"[Data] Failed to save loaded model input view records ({type(e).__name__}: {e})", flush=True)

    t0 = time.time()
    for i, idx in enumerate(memory_indices):
        if len(raw_batches) >= probe_memory_views:
            break
        raw_data = train_dataset[idx]

        # ACE loader: CamLocDatasetDINOv2 returns an 8-tuple per frame.
        # WAI loader: often returns list[dict] (one dict per view) — must NOT pass that to convert_ace_tuple_to_dict.
        if config.dataset_loader == "ace" and isinstance(raw_data, tuple):
            view = convert_ace_tuple_to_dict(raw_data, config.dataset_path)
            views = [view]
        elif isinstance(raw_data, list):
            views = raw_data
        else:
            views = [raw_data]

        for inner_j, view in enumerate(views):
            if len(raw_batches) >= probe_memory_views:
                break
            view_idx = len(raw_batches)
            actual_flat_idx = _lookup_actual_flat_idx(view)
            fps_idx_at_view_position = memory_indices[view_idx] if view_idx < len(memory_indices) else None
            loaded_view_records.append(
                {
                    "view_idx": view_idx,
                    "outer_index": i,
                    "inner_index": inner_j,
                    "selected_outer_fps_idx": int(idx) if isinstance(idx, (int, np.integer)) else idx,
                    "group_id": memory_index_to_group.get(int(idx)) if isinstance(idx, (int, np.integer)) else None,
                    "fps_idx_at_view_position": (
                        int(fps_idx_at_view_position)
                        if isinstance(fps_idx_at_view_position, (int, np.integer))
                        else fps_idx_at_view_position
                    ),
                    "actual_flat_idx": (
                        int(actual_flat_idx)
                        if isinstance(actual_flat_idx, (int, np.integer))
                        else actual_flat_idx
                    ),
                    "matches_fps_at_view_position": (
                        actual_flat_idx == fps_idx_at_view_position
                        if actual_flat_idx is not None and fps_idx_at_view_position is not None
                        else None
                    ),
                    "source": _view_source_name(view),
                    "label": _first_scalar_str(view.get("label")) if isinstance(view, dict) else None,
                }
            )
            raw_batches.append(view)  # Save original data
            processed = prepare_batch_input(view, device)
            memory_views.append(processed)
            if i == 0 and isinstance(view, dict) and isinstance(processed, dict):
                _print_loader_compare(view, processed)
            if "camera_poses" in processed:
                memory_gt_poses.append(processed["camera_poses"])
        if i == 0 or (i + 1) % 10 == 0:
            elapsed = time.time() - t0
            print(f"  [{i+1}/{len(memory_indices)} views loaded] {elapsed:.1f}s elapsed", flush=True)
    # Cap to the probe budget. Without PosePrune this equals n_memory; with
    # PosePrune the final target is restored after the probe infer.
    raw_batches = raw_batches[:probe_memory_views]
    memory_views = memory_views[:probe_memory_views]
    memory_gt_poses = memory_gt_poses[:probe_memory_views]
    loaded_view_records = loaded_view_records[:probe_memory_views]
    _write_loaded_view_records(loaded_view_records)
    actual_loaded_indices = [
        int(rec["actual_flat_idx"])
        for rec in loaded_view_records
        if isinstance(rec.get("actual_flat_idx"), (int, np.integer))
    ]
    if len(actual_loaded_indices) >= 2 and actual_loaded_indices != [int(i) for i in memory_indices[:len(actual_loaded_indices)]]:
        print(
            f"[SelectionCoverage] Recomputing coverage for actual loaded model-input views "
            f"({len(actual_loaded_indices)} views).",
            flush=True,
        )
        _export_selection_coverage_diagnostics(
            train_dataset,
            config.dataset_path,
            scene_name,
            actual_loaded_indices,
            output_run_dir,
            max_views=requested_memory_views,
            adaptive_min_views=config.covis_adaptive_min_views,
            tag="SelectionCoverage:LoadedViews",
        )
    print(f"  [All {len(memory_views)} views loaded in {time.time()-t0:.1f}s]", flush=True)

    try:
        print(f"[Vis] Exporting depth/RGB validation under run dir: {output_run_dir}", flush=True)
        export_memory_views_visualization(
            raw_batches,
            memory_views,
            output_run_dir,
            depth_min=config.depth_valid_range[0],
            depth_max=config.depth_valid_range[1],
            view_records=loaded_view_records,
        )
    except Exception as e:
        print(f"[Vis] Visualization skipped ({type(e).__name__}: {e})", flush=True)

    # Optional debug dump for ace vs wai alignment
    try:
        dump_dir = debug_dump_views(
            output_run_dir,
            config,
            raw_batches,
            memory_views,
            depth_min=float(config.depth_valid_range[0]),
            depth_max=float(config.depth_valid_range[1]),
        )
        if dump_dir:
            print(f"[DebugDump] Saved debug dump to: {dump_dir}", flush=True)
    except Exception as e:
        print(f"[DebugDump] Skipped ({type(e).__name__}: {e})", flush=True)

    # Initialize feature extractor
    print("[BSE Memory] Initializing feature extractor...", flush=True)
    use_mapanything = config.use_model == "mapanything"

    if use_mapanything:
        try:
            extractor = MapAnythingExtractor(
                "mapanything",
                config.model_config or "default",
                config.model_checkpoint or "/mnt/storage/xwh/checkpoints/facebook_map-anything.pth",
                device=str(device),
                dinov2_checkpoint=config.dinov2_checkpoint,
            )
            print("[BSE Memory] Using MapAnything extractor (default)")
        except (ImportError, Exception) as e:
            import traceback
            print(f"[BSE Memory] MapAnything unavailable: {e}")
            print(f"[BSE Memory] Full traceback:\n{traceback.format_exc()}")
            use_mapanything = False

    if not use_mapanything:
        layers = config.dinov2_intermediate_layers
        if layers is None:
            # Default DPT-style: 8 evenly-spaced layers across 24 blocks
            layers = [2, 5, 8, 11, 14, 17, 20, 23]
        print(f"[BSE Memory] Using DINOv2 extractor with DPT-style multi-scale (layers={layers})")
        extractor = DINOv2Extractor(
            config.dinov2_checkpoint,
            device=str(device),
            intermediate_layers=layers,
        )
        # Save layers_idx for memory file (add 'final' to match pooled format)
        layers_idx = layers + ['final']
    else:
        # MapAnything uses [0, 6, 12, 18, 'final'] by default
        layers_idx = [0, 6, 12, 18, 'final']

    # Before feature extraction, ensure depth entering the model respects the configured valid range.
    if config.depth_valid_range:
        d_lo, d_hi = float(config.depth_valid_range[0]), float(config.depth_valid_range[1])
        for v in memory_views:
            dz = v.get("depth_z")
            if isinstance(dz, torch.Tensor):
                # depth_z is [B,H,W,1]; zero-out values outside [d_lo, d_hi] or non-finite.
                mask_invalid = (~torch.isfinite(dz)) | (dz <= d_lo) | (dz >= d_hi)
                if mask_invalid.any():
                    dz = dz.clone()
                    dz[mask_invalid] = 0.0
                    v["depth_z"] = dz

    # Extract features
    print("[BSE Memory] Extracting features...", flush=True)

    if use_mapanything:
        # MapAnything must run as one batch so all features share the same
        # internal reference frame.
        with torch.no_grad():
            predictions = extractor.model.infer(
                memory_views,
                memory_efficient_inference=True,
                use_amp=False,
                ignore_depth_inputs=False,
                ignore_pose_inputs=False,
                ignore_calibration_inputs=False
            )

        pose_ok, trans_errors, _rot_errors = print_pose_prediction_vs_gt(
            memory_gt_poses,
            predictions,
            translation_ok_m=config.pose_eval_translation_ok_m,
            return_errors=True,
        )
        if config.pose_prune_after_infer and trans_errors:
            keep_indices, prune_stats = _pose_prune_keep_indices(
                trans_errors,
                translation_ok_m=config.pose_eval_translation_ok_m,
                min_views=max(config.pose_prune_min_views, requested_memory_views),
                min_keep_ratio=0.0,
                mad_k=config.pose_prune_mad_k,
                max_views=requested_memory_views,
            )
            if len(keep_indices) < len(memory_views):
                print(
                    f"[PosePrune] Dropping {len(memory_views) - len(keep_indices)} / {len(memory_views)} views "
                    f"after probe infer; target_keep={prune_stats.get('target_keep')}, "
                    f"median={prune_stats.get('median', 0.0):.4f}m, "
                    f"MAD={prune_stats.get('mad', 0.0):.4f}m, "
                    f"robust_threshold={prune_stats.get('robust_threshold', 0.0):.4f}m.",
                    flush=True,
                )
                print(f"[PosePrune] Dropped view errors: {prune_stats.get('dropped_errors', {})}", flush=True)

                raw_batches = [raw_batches[i] for i in keep_indices]
                memory_views = [memory_views[i] for i in keep_indices]
                memory_gt_poses = [memory_gt_poses[i] for i in keep_indices]
                loaded_view_records = [
                    {**loaded_view_records[i], "pruned_from_view_idx": int(loaded_view_records[i].get("view_idx", i)), "view_idx": new_i}
                    for new_i, i in enumerate(keep_indices)
                ]
                memory_indices = [
                    int(rec.get("actual_flat_idx", rec.get("selected_outer_fps_idx", idx)))
                    for rec, idx in zip(loaded_view_records, keep_indices)
                ]
                try:
                    out_path = os.path.join(output_run_dir, "pose_pruned_model_input_views.json")
                    with open(out_path, "w", encoding="utf-8") as f:
                        json.dump(
                            {
                                "pose_prune": prune_stats,
                                "selected_memory_indices_after_prune": memory_indices,
                                "records": loaded_view_records,
                            },
                            f,
                            ensure_ascii=False,
                            indent=2,
                        )
                    print(f"[PosePrune] Pruned input view records saved to: {out_path}", flush=True)
                except Exception as e:
                    print(f"[PosePrune] Failed to save prune records ({type(e).__name__}: {e})", flush=True)

                if hasattr(extractor.model, "get_info_sharing_intermediate_features"):
                    extractor.model.get_info_sharing_intermediate_features(clear=True)
                torch.cuda.empty_cache()
                print(f"[PosePrune] Re-running MapAnything infer with {len(memory_views)} kept views...", flush=True)
                with torch.no_grad():
                    predictions = extractor.model.infer(
                        memory_views,
                        memory_efficient_inference=True,
                        use_amp=False,
                        ignore_depth_inputs=False,
                        ignore_pose_inputs=False,
                        ignore_calibration_inputs=False,
                    )
                pose_ok, trans_errors, _rot_errors = print_pose_prediction_vs_gt(
                    memory_gt_poses,
                    predictions,
                    translation_ok_m=config.pose_eval_translation_ok_m,
                    return_errors=True,
                )
            else:
                print("[PosePrune] No views dropped by PoseEval gate.", flush=True)
        if config.pose_eval_strict and not pose_ok:
            print(
                "[PoseEval] Strict mode: exiting with code 1 (see table above).",
                flush=True,
            )
            sys.exit(1)

        # Add 'depths' key only after MapAnything infer(), because infer()
        # validates input keys strictly and rejects this compatibility alias.
        for view in memory_views:
            if 'depth_z' in view and 'depths' not in view:
                depth_z = view['depth_z']  # [B,H,W,1]
                view['depths'] = depth_z.permute(0, 3, 1, 2)  # [B,1,H,W]

        stored_features = extractor.model.get_info_sharing_intermediate_features()
        features_dict = extractor._process_saved_features(stored_features, len(memory_views))
    else:
        # DINOv2: batch tensors
        all_images = torch.cat([v['img'] for v in memory_views], dim=0).float()  # Ensure float32 for DINOv2
        all_depths = torch.cat([v['depth_z'] for v in memory_views], dim=0) if 'depth_z' in memory_views[0] else None
        all_poses = torch.cat([v['camera_poses'] for v in memory_views], dim=0)
        all_intrinsics = torch.cat([v['intrinsics'] for v in memory_views], dim=0)
        features_dict = extractor.extract(all_images, all_depths, all_poses, all_intrinsics)

    # Two-pass processing
    print("[BSE Memory] Two-pass processing...", flush=True)
    result, mu, sigma, view_info = two_pass_processing(
        memory_views, raw_batches, features_dict, pooler, welford, config
    )

    # Save memory
    pooled_scene_center = None
    if scene_center_from_cameras is not None:
        pooled_scene_center = torch.as_tensor(scene_center_from_cameras, dtype=torch.float32)

    selection_metadata = {
        "selection_strategy": canonicalize_wai_view_mode(config.wai_view_mode),
        "wai_view_mode": canonicalize_wai_view_mode(config.wai_view_mode),
        "selected_memory_indices": [int(x) for x in memory_indices],
        "memory_index_groups": [[int(x) for x in group] for group in memory_index_groups],
        "dataset_num_views": int(dataset_num_views),
        "scene_center_source": (
            "mapanything.select_optimal_memory_indices.mean_all_camera_centers"
            if canonicalize_wai_view_mode(config.wai_view_mode) != "anchor_support"
            else "metadata.mean_all_camera_centers"
        ),
        "loaded_view_records": loaded_view_records,
    }

    save_memory(
        config.output_path,
        result,
        mu,
        sigma,
        view_info,
        layers_idx,
        scene_center=pooled_scene_center,
        config=config,
        selection_metadata=selection_metadata,
    )

    # Save PLY for visualization
    save_ply(config.output_path, result, mu, sigma)

    print("[BSE Memory] Done!")


if __name__ == '__main__':
    main()
