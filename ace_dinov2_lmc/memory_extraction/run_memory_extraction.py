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

    for i, view in enumerate(memory_views):
        if "img" in view:
            _save_model_input_vis(
                view["img"],
                os.path.join(model_input_dir, f"view_{i:02d}_model_input.png"),
            )
    print(f"[Vis] Model input images (normalized) saved to: {model_input_dir}", flush=True)


# =============================================================================
# Schema Versioning
# =============================================================================

MEMORY_SCHEMA_VERSION = "1.3"  # Added world-coordinate points alongside normalized coordinates
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
    # Pass 2 完成逐 view 拼接后，是否做跨 view 的全局体素合并
    global_merge: bool = True
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
    # Debug：将关键中间数据/统计 dump 到输出目录（用于对齐 ace vs wai）
    debug_dump: bool = False
    # Debug：最多 dump 前 N 个视角（避免目录过大）
    debug_dump_max_views: int = 3
    # Debug：是否额外保存少量数值快照（npz），体积更大
    debug_dump_save_npz: bool = False


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
) -> bool:
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
        return True
    if len(memory_gt_poses) == 0:
        print("[PoseEval] Skipped: no GT poses in prepared views (camera_poses missing in batch).")
        return True

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
    return all_trans_ok


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

    # Extract rotation and translation
    R = pose[:3, :3]  # [3, 3]
    t = pose[:3, 3]   # [3]
    camera_center = -R.T @ t  # [3]

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
        if missing:
            print(
                f"[Resume] Checkpoint found but {len(missing)}/{len(existing_chunks)} chunk files are missing; "
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

            # Process features: align to grid and concatenate
            features_flat, _, _ = process_multiscale_features_to_grid(
                feat_list, apply_l2_norm=False
            )

            # Prepare geometry data from raw_batch (original data)
            batch_gpu = {}
            for key in ["depth", "depthmap", "depth_z", "gt_depth", "intrinsics", "camera_intrinsics", "camera_pose", "pose", "scene_coords"]:
                if key in raw_batch:
                    val = raw_batch[key]
                    if isinstance(val, (list, tuple)): val = val[0]
                    if isinstance(val, np.ndarray): val = torch.from_numpy(val)
                    if isinstance(val, torch.Tensor): batch_gpu[key] = val.to(device)

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

            # Unproject with grid features (samples depth at grid centers)
            points, ray_dirs, colors, features_flat, camera_centers = unproject_with_grid_features(
                batch_gpu,
                features_flat,
                grid_H,
                grid_W,
                config.depth_valid_range,
                sampling_method=config.patch_depth_sampling,
            )

            # Skip if no valid points
            if len(points) == 0:
                print(f"[Warning] No valid points for view {view_idx}")
                continue

            # Optional: SOR filtering (MEDIUM priority)
            if config.enable_sor:
                inlier_mask = sor_filter(
                    points, k=config.sor_k, std_ratio=config.sor_std_ratio
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

            # Extract camera center and rotation from pose
            # pose = [R | t; 0 0 0 1], center = -R^T @ t
            R = pose[:3, :3]  # [3, 3]
            t = pose[:3, 3]   # [3]
            camera_center = -R.T @ t  # [3]

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
            pooled_global = pooler.pool(
                torch.cat(all_points, dim=0).float().to(device),
                torch.cat(all_features, dim=0).float().to(device),
                torch.cat(all_colors, dim=0).float().to(device),
                torch.cat(all_ray_dirs, dim=0).float().to(device),
                camera_centers=torch.cat(all_camera_centers, dim=0).float().to(device),
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
            merged = global_voxel_merge(
                points=final_points_world,
                features=final_features,
                colors=final_colors,
                ray_dirs=final_ray_dirs,
                ray_dirs_mean=final_ray_dirs_mean,
                cluster_sizes=final_cluster_sizes,
                voxel_size=config.voxel_size,
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
                f"(removed {n_before_merge - int(final_points_world.shape[0])} cross-view duplicates)",
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
) -> None:
    """
    Save memory to .pt file with extended schema and versioning.

    Output schema includes:
    - Point-level data:
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
    points_world = result['points'].cpu().float() * float(sigma) + mu.cpu().float().view(1, 3)

    memory_dict = {
        'schema_version': MEMORY_SCHEMA_VERSION,
        'points': result['points'].cpu().float(),           # [N, 3]
        'points_world': points_world,                       # [N, 3] world coordinates, avoids runtime de-normalization
        'ray_dirs': result['ray_dirs'].cpu().float(),       # [N, 3]
        'ray_dirs_mean': result['ray_dirs_mean'].cpu().float(),  # [N, 3]
        'features': result['features'].cpu().half(),        # [N, C] fp16
        'colors': result['colors'].cpu().float(),           # [N, 3]
        'cluster_sizes': result['cluster_sizes'].cpu().long(),  # [N]
        'mu': mu.cpu().float(),                             # [3]
        'sigma': sigma                                      # scalar
    }

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
    print(f"[Save] World points: {tuple(memory_dict['points_world'].shape)}")
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

    For mapanything WAI datasets, returns a sequential-view-mode dataset where
    each index corresponds to one frame (for FPS memory selection).

    Args:
        dataset_path: ROOT directory for WAI datasets (e.g. /data/.../7scenes).
        dataset_type: "7scenes", "indoor6", "custom", or "auto"
        scene_name: Specific scene to load (e.g. "chess_train"). Required when
                    dataset_path points to a ROOT directory containing multiple scenes.
        n_views: Number of views per sample (1 for sequential frame-level access)

    Returns:
        Dataset object where __getitem__(idx) returns a single view dict with keys:
            img, depthmap, camera_pose, camera_intrinsics, dataset, label, instance
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
    R = p[:3, :3]
    t = p[:3, 3]
    c = -R.T @ t
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
    c = -R.T @ t
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
        "ray_pool_strategy": config.ray_pool_strategy,
        "use_model": config.use_model,
        "model_str": config.model_str,
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


def select_memory_views(
    dataset,
    n_memory: int,
    dataset_path: Optional[str] = None,
    scene_name: Optional[str] = None,
    force_include_origin: bool = True,
) -> Tuple[List[int], Optional[np.ndarray]]:
    """
    选择 memory 视角（优先 FPS）。

    优先级：
    1) 若可用：map-anything 的 select_optimal_memory_indices（它本身就是最优选帧逻辑）
    2) 否则：读取 <dataset_path>/poses 做 FPS（最快，不加载图像/深度）
    3) 再否则：遍历 dataset item 抽 pose 做 FPS（慢）

    Args:
        force_include_origin: When True (default), frame index 0 (scene origin) is
            always included in the selection. This ensures MapAnything's internal
            reference frame aligns with the scene world origin, eliminating
            implicit coordinate offsets between features and point cloud.
    """
    try:
        from mapanything.tasks.ace.memory_selection import select_optimal_memory_indices  # pyright: ignore[reportMissingImports]
        memory_indices, scene_center = select_optimal_memory_indices(dataset, n_memory)
        return memory_indices, scene_center
    except ImportError:
        pass

    centers = None
    if dataset_path is not None:
        # WAI fastest path: scene_meta.json already contains per-frame c2w.
        if scene_name:
            centers = _extract_centers_from_wai_scene_meta(dataset_path, scene_name)
        # ACE / generic path: poses/ directory
        if centers is None:
            centers = _extract_centers_from_pose_dir(dataset_path, scene_name=scene_name)
    if centers is None:
        centers = _extract_centers_from_dataset_items(dataset)

    force_include = [0] if force_include_origin else None
    indices = fps_select_views(centers, n_memory, force_include=force_include)
    indices.sort()  # 顺序化索引，利于后续按序 IO
    scene_center = centers.mean(axis=0).astype(np.float32) if centers is not None and len(centers) > 0 else None
    return indices, scene_center


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
        choices=['bse', 'simple'],
        help='pooling 模式：bse=voxel hash + Otsu split；simple=简单 voxel mean。',
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
        action='store_true',
        default=True,
        help='BSE 内对深度分箱使用 Otsu 自动阈值（适应稀疏/不均匀深度）。',
    )
    parser.add_argument(
        '--global_merge',
        type=lambda x: str(x).strip().lower() in ('1', 'true', 'yes', 'on'),
        default=True,
        help='Pass 2 对所有 view 的 pooled 点做一次全局体素合并（默认 True）。',
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

    args = parser.parse_args()

    return ExtractionConfig(
        dataset_path=args.dataset_path,
        output_path=args.output_path,
        n_memory=args.n_memory,
        device=args.device,
        voxel_size=args.voxel_size,
        use_bse=args.use_bse,
        pool_mode=args.pool_mode,
        prepool_mode=args.prepool_mode,
        use_otsu=args.use_otsu,
        global_merge=bool(args.global_merge),
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
        ray_pool_strategy=args.ray_pool_strategy,
        save_all_ray_strategies=args.save_all_ray_strategies,
        use_model=args.use_model,
        dinov2_intermediate_layers=args.dinov2_intermediate_layers,
        pose_eval_translation_ok_m=args.pose_eval_translation_ok_m,
        pose_eval_strict=args.pose_eval_strict,
        debug_dump=args.debug_dump,
        debug_dump_max_views=int(args.debug_dump_max_views),
        debug_dump_save_npz=args.debug_dump_save_npz,
    )


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
    print(f"[BSE Memory] Global merge: {config.global_merge}")
    print(
        f"[Config] patch_depth_sampling: {config.patch_depth_sampling} "
        f"(align map-anything fps_memory: nearest_valid for sparse Indoor6 depth)",
        flush=True,
    )
    print(
        f"[Config] Depth valid range: [{config.depth_valid_range[0]:.3f}, {config.depth_valid_range[1]:.3f}] m",
        flush=True,
    )

    # Initialize modules
    welford = WelfordNormalizer()
    if config.use_bse:
        pooler = BSEPooler(
            voxel_size=config.voxel_size,
            pool_mode=config.pool_mode,
            use_otsu=config.use_otsu,
            ray_pool_strategy=config.ray_pool_strategy,
            save_all_ray_strategies=config.save_all_ray_strategies
        )
    else:
        raise NotImplementedError("Vanilla voxel pooling not implemented")

    # Load dataset
    print("[BSE Memory] Loading dataset...", flush=True)
    train_dataset = load_dataset(
        config.dataset_path,
        dataset_type=dataset_type,
        scene_name=scene_name,
        n_views=1,
        dataset_loader=config.dataset_loader,
    )
    print(f"[BSE Memory] Train dataset: {len(train_dataset)} frames")

    # Select memory views
    print("[BSE Memory] Selecting memory views...", flush=True)
    memory_indices, scene_center_from_cameras = select_memory_views(
        train_dataset,
        config.n_memory,
        dataset_path=config.dataset_path,
        scene_name=scene_name,
    )
    print(f"[BSE Memory] Selected {len(memory_indices)} views", flush=True)
    print(f"[Data] Selected memory indices ({len(memory_indices)}): {memory_indices}", flush=True)

    # Prepare input views (fast path: load directly from disk, skip WAI heavy processing)
    print("[BSE Memory] Preparing input views...", flush=True)
    raw_batches = []  # Store original data for depth/intrinsics access
    memory_views = []  # Store processed data for model input
    memory_gt_poses: List[torch.Tensor] = []  # For post-infer pose vs GT check (MapAnything)
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
    t0 = time.time()
    for i, idx in enumerate(memory_indices):
        if len(raw_batches) >= config.n_memory:
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

        for view in views:
            if len(raw_batches) >= config.n_memory:
                break
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
    # Cap to exactly n_memory
    raw_batches = raw_batches[:config.n_memory]
    memory_views = memory_views[:config.n_memory]
    memory_gt_poses = memory_gt_poses[:config.n_memory]
    print(f"  [All {len(memory_views)} views loaded in {time.time()-t0:.1f}s]", flush=True)

    output_run_dir = os.path.dirname(os.path.abspath(config.output_path))
    try:
        print(f"[Vis] Exporting depth/RGB validation under run dir: {output_run_dir}", flush=True)
        export_memory_views_visualization(
            raw_batches,
            memory_views,
            output_run_dir,
            depth_min=config.depth_valid_range[0],
            depth_max=config.depth_valid_range[1],
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
        # MapAnything: call infer and get features from memory
        with torch.no_grad():
            predictions = extractor.model.infer(
                memory_views,
                memory_efficient_inference=True,
                use_amp=False,
                ignore_depth_inputs=False,
                ignore_pose_inputs=False,
                ignore_calibration_inputs=False
            )

        pose_ok = print_pose_prediction_vs_gt(
            memory_gt_poses,
            predictions,
            translation_ok_m=config.pose_eval_translation_ok_m,
        )
        if config.pose_eval_strict and not pose_ok:
            print(
                "[PoseEval] Strict mode: exiting with code 1 (see table above).",
                flush=True,
            )
            sys.exit(1)

        # Add 'depths' key for compatibility with two_pass_processing
        for view in memory_views:
            if 'depth_z' in view and 'depths' not in view:
                depth_z = view['depth_z']  # [B,H,W,1]
                view['depths'] = depth_z.permute(0, 3, 1, 2)  # [B,1,H,W]

        # Get features from model's internal storage
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

    save_memory(
        config.output_path,
        result,
        mu,
        sigma,
        view_info,
        layers_idx,
        scene_center=pooled_scene_center,
    )

    # Save PLY for visualization
    save_ply(config.output_path, result, mu, sigma)

    print("[BSE Memory] Done!")


if __name__ == '__main__':
    main()
