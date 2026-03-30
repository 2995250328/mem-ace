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
from dataclasses import dataclass

# Add map-anything to path (optional - can use adapter instead)
MAP_ANYTHING_PATH = Path(__file__).parent.parent.parent.parent / "map-anything"
if MAP_ANYTHING_PATH.exists():
    sys.path.insert(0, str(MAP_ANYTHING_PATH))

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


def _fallback_save_pcd_with_open3d(points, filename, colors=None):
    try:
        import open3d as o3d  # type: ignore[import-not-found]
    except ImportError:
        print(f"[Vis] open3d not installed; skip PLY: {filename}", flush=True)
        return
    if isinstance(points, torch.Tensor):
        points_np = np.asarray(points.detach().cpu().numpy())
    else:
        points_np = np.asarray(points)

    if points_np.size == 0:
        print(f"[Vis] Skipping PLY save (zero points): {filename}", flush=True)
        return

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_np.astype(np.float64))

    if colors is not None:
        if isinstance(colors, torch.Tensor):
            colors_np = np.asarray(colors.detach().cpu().numpy())
        else:
            colors_np = np.asarray(colors)
        if colors_np.size > 0 and colors_np.shape[0] == points_np.shape[0]:
            if np.nanmax(colors_np) > 1.1:
                colors_np = colors_np / 255.0
            colors_np = np.clip(colors_np, 0.0, 1.0)
            pcd.colors = o3d.utility.Vector3dVector(colors_np.astype(np.float64))

    success = o3d.io.write_point_cloud(filename, pcd, write_ascii=False, print_progress=False)
    if success:
        print(f"[Vis] Point cloud saved to: {filename}", flush=True)


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
            if t.max() <= 1.01:
                t = (np.clip(t, 0, 1) * 255).astype(np.uint8)
            else:
                t = np.clip(t, 0, 255).astype(np.uint8)
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


def _save_depth_on_rgb_vis_dense(depth_np, save_dir, prefix, rgb_tensor=None, rgb_path=None, point_radius=6):
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

    d_min = float(depth_np[valid].min())
    d_max = float(depth_np[valid].max())
    if d_max <= d_min:
        d_max = d_min + 1.0

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
    PILImage.fromarray(on_rgb).save(os.path.join(save_dir, f"{prefix}_depth_on_rgb.png"))


def _save_depth_on_rgb_vis(depth_np, save_dir, prefix, rgb_tensor=None, rgb_path=None, point_radius=6):
    """
    Match map-anything/fps_memory: scatter over valid depth pixels when count is modest
    (sparse Indoor6); otherwise dense colormap so the run does not stall.
    """
    valid = (depth_np > 0) & np.isfinite(depth_np)
    n_valid = int(np.sum(valid))
    if _MA_DEPTH_SCATTER is not None and n_valid <= _SCATTER_SAFE_MAX_VALID:
        _MA_DEPTH_SCATTER(
            depth_np, save_dir, prefix,
            rgb_tensor=rgb_tensor, rgb_path=rgb_path, point_radius=point_radius,
        )
        return
    if _MA_DEPTH_SCATTER is not None and n_valid > _SCATTER_SAFE_MAX_VALID:
        print(
            f"[Vis] {prefix}: {n_valid} valid depth pixels (>{_SCATTER_SAFE_MAX_VALID}) — "
            f"using dense turbo colormap (map-anything scatter would be too slow here)",
            flush=True,
        )
    _save_depth_on_rgb_vis_dense(
        depth_np, save_dir, prefix,
        rgb_tensor=rgb_tensor, rgb_path=rgb_path, point_radius=point_radius,
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
            depth = depth.detach().float().cpu()
            depth = normalize_depth_to_hw(depth).numpy()
        else:
            depth = np.asarray(depth, dtype=np.float32).squeeze()
        if depth.ndim != 2:
            print(f"[Vis] view_{i:02d}: depth squeeze to 2D failed, shape={depth.shape}; skip", flush=True)
            continue

        intr = raw_data.get("camera_intrinsics", raw_data.get("intrinsics"))
        pose = raw_data.get("camera_pose", raw_data.get("pose"))
        if isinstance(intr, (list, tuple)):
            intr = intr[0]
        if isinstance(pose, (list, tuple)):
            pose = pose[0]
        if intr is not None and pose is not None:
            n_pts, pts_world = _raw_depth_to_point_cloud_count(
                depth, intr, pose, depth_min=depth_min, depth_max=depth_max
            )
            raw_depth_point_counts.append((i, n_pts))
            if i == 0 and pts_world is not None and len(pts_world) > 0:
                raw_ply_path = os.path.join(depth_vis_dir, "view_00_raw_depth_pointcloud.ply")
                save_pcd_with_open3d(torch.from_numpy(pts_world), raw_ply_path, colors=None)
                print(f"[Vis] View 0: raw depth -> {n_pts} points (range [{depth_min}, {depth_max}] m), saved to {raw_ply_path}", flush=True)

        _save_depth_on_rgb_vis(depth, depth_vis_dir, f"view_{i:02d}", rgb_tensor=rgb_tensor, rgb_path=rgb_path)

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

MEMORY_SCHEMA_VERSION = "1.2"  # Added view-level camera information (pose, intrinsics, Plücker main rays)
CHECKPOINT_FILE = "extraction_checkpoint.json"


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class ExtractionConfig:
    """Configuration for memory extraction."""
    dataset_path: str
    output_path: str
    n_memory: int = 100
    device: str = "cuda:0"
    voxel_size: float = 0.05
    use_bse: bool = True
    use_otsu: bool = True
    depth_valid_range: Tuple[float, float] = (0.1, 6.0)
    # Grid depth sampling (same semantics as map-anything generate_patch_point_cloud / fps_memory.sh)
    patch_depth_sampling: str = "nearest_valid"
    temp_dir: str = "/dev/shm"
    model_str: Optional[str] = None
    model_config: Optional[str] = None
    model_checkpoint: Optional[str] = None
    dinov2_checkpoint: str = "/data/xwh/checkpoints/dinov2_vitl14_pretrain.pth"
    use_patch_based: bool = False
    enable_sor: bool = False
    sor_k: int = 20
    sor_std_ratio: float = 2.0
    dataset_type: str = "auto"
    scene_name: Optional[str] = None  # WAI scene name (e.g. chess_train, scene2a_train)
    dataset_loader: str = "wai"  # 'wai' (MapAnything WAI) or 'ace' (CamLocDatasetDINOv2)
    ray_pool_strategy: str = "mean"  # 'mean', 'dominant', 'first', 'all'
    save_all_ray_strategies: bool = True  # Save all ray representations for comparison
    use_model: str = "mapanything"  # 'mapanything', 'dinov2' — feature extractor to use
    dinov2_intermediate_layers: Optional[List[int]] = None  # DINOv2 block indices for DPT-style multi-scale; None=auto [2,5,8,11,14,17,20,23]
    # Post-infer pose check (same logic as mapanything/tasks/run_memory_extraction.py Prediction Evaluation)
    pose_eval_translation_ok_m: float = 0.1
    pose_eval_strict: bool = False  # If True, exit non-zero when any non-ref view exceeds translation threshold


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
    TARGET_INTERM_LAYERS = [2, 5, 8, 11, 14, 17, 20, 23]

    def __init__(self, model_str: str = "mapanything", model_config: str = "default",
                 checkpoint: Optional[str] = None, device: str = "cuda:0"):
        """
        Initialize Map-Anything model adapter.

        Builds config manually from resolved YAML values, bypassing Hydra runtime.
        Falls back to YAML loading + partial resolution if manual config fails.

        Args:
            model_str: Model architecture string (e.g. "mapanything")
            model_config: "default" to use built-in config, or path to custom YAML
            checkpoint: Path to pretrained checkpoint
            device: Device to run on
        """
        self.device = device
        try:
            from mapanything.models import init_model  # pyright: ignore[reportMissingImports]

            # Build model config manually (bypasses Hydra interpolation issues)
            model_config_dict = self._build_manual_config()

            # Try init_model first (wraps model_factory → MapAnything)
            try:
                from omegaconf import OmegaConf  # pyright: ignore[reportMissingImports]
                cfg = OmegaConf.create(model_config_dict)
                self.model = init_model(model_str, cfg)
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
    def _build_manual_config() -> dict:
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
                "pretrained_checkpoint_path": "/data/xwh/checkpoints/dinov2_vitl14_pretrain.pth",
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

            # Extract CLS / scale token from raw_final (same as map-anything's additional_token_features)
            if raw_final:
                token_feat = None
                if isinstance(raw_final, dict) and "additional_token_features" in raw_final:
                    token_feat = raw_final["additional_token_features"]
                elif hasattr(raw_final, 'additional_token_features'):
                    token_feat = raw_final.additional_token_features
                if token_feat is not None and isinstance(token_feat, torch.Tensor):
                    result[i]['cls_token'] = token_feat[i].detach().cpu().view(-1)

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
        """Load DINOv2 model."""
        try:
            # Try torch.hub first
            model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')
        except Exception:
            # Fallback to local checkpoint
            model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14', pretrained=False)
            state = torch.load(checkpoint_path, map_location='cpu')
            model.load_state_dict(state, strict=False)

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

                # Also get the final layer's patch tokens (after norm)
                x_norm_patchtokens = out["x_norm_patchtokens"]  # [B, N, C]
                x_norm_clstoken = out.get("x_norm_clstoken")    # [B, C] or None

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

                # For the last block (if not already hooked), use the normed output
                last_idx = self.intermediate_layers[-1]
                if last_idx not in layer_outputs or len(layer_outputs[last_idx]) == 0:
                    layer_outputs[last_idx] = x_norm_patchtokens

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
        img = torch.from_numpy(img_raw).to(device)
    else:
        img = img_raw.to(device)

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
    apply_l2_norm: bool = False
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
            S = int(math.sqrt(N))
            if S * S == N:
                H_p, W_p = S, S
            else:
                # Try aspect ratio
                W_p = int(math.sqrt(N))
                H_p = N // W_p
                if H_p * W_p != N:
                    H_p, W_p = S, S
            feat = feat.reshape(1, C, H_p, W_p)

        elif feat.ndim == 3:
            # (B, L, C) -> (B, C, H, W)
            B, L, C = feat.shape
            # Try to infer spatial dimensions
            S = int(math.sqrt(L))
            if S * S == L:
                H_p, W_p = S, S
            else:
                # Try aspect ratio
                ratio = 1.0
                W_p = int(math.sqrt(L / ratio))
                H_p = L // W_p
                if H_p * W_p != L:
                    H_p, W_p = S, S
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
    if 'images' in view_data:
        rgb = view_data['images'][0]  # [3, H, W]
        # Sample colors using bilinear interpolation
        # Normalize u, v to [-1, 1] for grid_sample
        u_norm = 2.0 * grid_u[valid_mask] / (img_W - 1) - 1.0
        v_norm = 2.0 * grid_v[valid_mask] / (img_H - 1) - 1.0
        grid_coords = torch.stack([u_norm, v_norm], dim=-1).unsqueeze(0).unsqueeze(0)  # [1, 1, N, 2]
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


def two_pass_processing(
    memory_views: List[Dict[str, torch.Tensor]],
    raw_batches: List[Dict[str, Any]],  # Original data for depth/intrinsics
    features_dict: Dict[int, Dict[str, List[torch.Tensor]]],
    pooler: BSEPooler,
    welford: WelfordNormalizer,
    config: ExtractionConfig
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, float, Dict[str, torch.Tensor]]:
    """
    Two-pass BSE + Welford normalization with HIGH priority fixes.

    Pass 1: Extract -> Unproject -> Pool -> Accumulate statistics -> Save chunks
    Pass 2: Finalize statistics -> Load chunks -> Normalize -> Concatenate

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
        print(f"[Resume] Found {len(completed_views)} completed views")
        chunk_paths = existing_chunks

    try:
        # Pass 1: Extract + pool + accumulate
        print("[Pass 1] Extracting and pooling...")

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

            # BSE pooling (with camera_centers for Plücker encoding)
            pooled = pooler.pool(
                points, features_flat, colors, ray_dirs,
                camera_centers=camera_centers
            )

            # Accumulate statistics
            welford.update(pooled['points'])

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
            torch.save(pooled, chunk_path)
            chunk_paths.append(chunk_path)

            # Save checkpoint for resume
            completed_views.append(view_idx)
            save_checkpoint(checkpoint_path, completed_views, chunk_paths)

            # HIGH Priority: GPU memory cleanup between views
            torch.cuda.empty_cache()

        # Pass 2: Normalize + assemble
        print("[Pass 2] Normalizing and assembling...")
        mu_scene, sigma_scene = welford.finalize()

        all_points = []
        all_ray_dirs = []
        all_ray_dirs_mean = []
        all_ray_dirs_dominant = []
        all_ray_dirs_first = []
        all_features = []
        all_colors = []
        all_plucker_rays = []
        all_camera_centers = []
        all_cluster_sizes = []

        for chunk_path in tqdm(chunk_paths):
            pooled = torch.load(chunk_path)
            # Normalize points
            pooled['points'] = (pooled['points'] - mu_scene) / sigma_scene
            all_points.append(pooled['points'])
            all_ray_dirs.append(pooled['ray_dirs'])
            all_ray_dirs_mean.append(pooled['ray_dirs_mean'])
            all_features.append(pooled['features'])
            all_colors.append(pooled['colors'])
            all_cluster_sizes.append(pooled['cluster_sizes'])

            # Optional ray representations
            if 'ray_dirs_dominant' in pooled:
                all_ray_dirs_dominant.append(pooled['ray_dirs_dominant'])
            if 'ray_dirs_first' in pooled:
                all_ray_dirs_first.append(pooled['ray_dirs_first'])
            if 'plucker_rays' in pooled:
                all_plucker_rays.append(pooled['plucker_rays'])
            if 'camera_centers' in pooled:
                all_camera_centers.append(pooled['camera_centers'])

        # Concatenate all chunks
        final_points = torch.cat(all_points, dim=0)
        final_ray_dirs = torch.cat(all_ray_dirs, dim=0)
        final_ray_dirs_mean = torch.cat(all_ray_dirs_mean, dim=0)
        final_features = torch.cat(all_features, dim=0)
        final_colors = torch.cat(all_colors, dim=0)
        final_cluster_sizes = torch.cat(all_cluster_sizes, dim=0)

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
        if all_ray_dirs_dominant:
            result['ray_dirs_dominant'] = torch.cat(all_ray_dirs_dominant, dim=0)
        if all_ray_dirs_first:
            result['ray_dirs_first'] = torch.cat(all_ray_dirs_first, dim=0)
        if all_plucker_rays:
            result['plucker_rays'] = torch.cat(all_plucker_rays, dim=0)
        # NOTE: We remove pooled camera_centers since we now have view-level camera info
        # if all_camera_centers:
        #     result['camera_centers'] = torch.cat(all_camera_centers, dim=0)

        # Assemble view-level camera information
        view_info = {
            'camera_centers': torch.stack(view_camera_centers, dim=0),      # [M, 3]
            'camera_rotations': torch.stack(view_camera_rotations, dim=0),  # [M, 3, 3]
            'camera_intrinsics': torch.stack(view_camera_intrinsics, dim=0), # [M, 3, 3]
            'plucker_main_rays': torch.stack(view_plucker_main_rays, dim=0), # [M, 6]
            'all_scale_tokens': torch.stack(all_cls_tokens, dim=0) if all_cls_tokens else None,  # [M, D]
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
    view_info: Dict[str, torch.Tensor] = None
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
    memory_dict = {
        'schema_version': MEMORY_SCHEMA_VERSION,
        'points': result['points'].cpu().float(),           # [N, 3]
        'ray_dirs': result['ray_dirs'].cpu().float(),       # [N, 3]
        'ray_dirs_mean': result['ray_dirs_mean'].cpu().float(),  # [N, 3]
        'features': result['features'].cpu().half(),        # [N, C] fp16
        'colors': result['colors'].cpu().float(),           # [N, 3]
        'cluster_sizes': result['cluster_sizes'].cpu().long(),  # [N]
        'mu': mu.cpu().float(),                             # [3]
        'sigma': sigma                                      # scalar
    }

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

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

    torch.save(memory_dict, output_path)
    print(f"[Save] Memory saved to {output_path}")
    print(f"[Save] Schema version: {MEMORY_SCHEMA_VERSION}")
    print(f"[Save] Total points: {len(result['points'])}")
    print(f"[Save] Feature dim: {result['features'].shape[1]}")
    print(f"[Save] Scene mean: {mu.tolist()}")
    print(f"[Save] Scene sigma: {sigma:.4f}")
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

    # Write PLY
    with open(ply_path, 'w') as f:
        # Header
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")

        # Vertices
        for i in range(len(points)):
            x, y, z = points[i]
            r, g, b = (colors[i] * 255).astype(np.uint8)
            f.write(f"{x:.6f} {y:.6f} {z:.6f} {r} {g} {b}\n")

    print(f"[PLY] Point cloud saved to {ply_path}")
    print(f"[PLY] Points: {len(points):,}")


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

    # === ACE dataset loader branch (NEW) ===
    if dataset_loader == "ace":
        print(f"[Dataset] Using ACE loader (CamLocDatasetDINOv2)")
        # Add parent directory to path for imports
        parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        if parent_dir not in sys.path:
            sys.path.insert(0, parent_dir)
        from dataset_dinov2 import CamLocDatasetDINOv2
        return CamLocDatasetDINOv2(dataset_path, mode=2)

    # === WAI dataset loader branch (EXISTING) ===
    try:
        from mapanything.datasets.wai.seven_scenes import SevenScenesWAI  # pyright: ignore[reportMissingImports]
        from mapanything.datasets.wai.indoor6 import Indoor6WAI  # pyright: ignore[reportMissingImports]

        # Determine dataset_metadata_dir
        metadata_dir = os.environ.get(
            "MAPANYTHING_DATASET_METADATA_DIR",
            "/data/xwh/map-anything/mapanything_dataset_metadata"
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
        elif dataset_type == "indoor6":
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
        else:
            raise ImportError("custom dataset type, falling back")
    except (ImportError, Exception) as e:
        print(f"[Warning] WAI dataset loading failed ({e}), using ACE dataset loader")
        from dataset_dinov2 import CamLocDatasetDINOv2
        return CamLocDatasetDINOv2(dataset_path, mode=2)


def select_memory_views(dataset, n_memory: int) -> List[int]:
    """
    Select optimal memory views from dataset.

    Args:
        dataset: Dataset object
        n_memory: Number of views to select

    Returns:
        List of selected indices
    """
    try:
        from mapanything.tasks.ace.memory_selection import select_optimal_memory_indices  # pyright: ignore[reportMissingImports]
        memory_indices, _ = select_optimal_memory_indices(dataset, n_memory)
        return memory_indices
    except ImportError:
        # Fallback: uniform sampling
        total = len(dataset)
        step = max(1, total // n_memory)
        return list(range(0, total, step))[:n_memory]


# =============================================================================
# Main Entry Point
# =============================================================================

def parse_args() -> ExtractionConfig:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='BSE-enhanced memory extraction')

    # Required arguments
    parser.add_argument('dataset_path', type=str, help='Dataset path')
    parser.add_argument('output_path', type=str, help='Output .pt file')

    # Memory selection
    parser.add_argument('--n_memory', type=int, default=100, help='Number of memory views')

    # Model configuration
    parser.add_argument('--use_model', type=str, default='mapanything',
                        choices=['mapanything', 'dinov2'],
                        help='Feature extractor: mapanything (default, align with fps_memory) or dinov2')
    parser.add_argument('--model_str', type=str, default=None, help='Model architecture string')
    parser.add_argument('--model_config', type=str, default=None, help='Model config path')
    parser.add_argument('--model_checkpoint', type=str, default=None, help='Model checkpoint path')
    parser.add_argument('--dinov2_checkpoint', type=str,
                        default='/data/xwh/checkpoints/dinov2_vitl14_pretrain.pth',
                        help='DINOv2 checkpoint path (for dinov2 mode)')
    parser.add_argument('--dinov2_intermediate_layers', type=int, nargs='*', default=None,
                        help='DINOv2 block indices for DPT-style multi-scale. '
                             'Default: [2,5,8,11,14,17,20,23] (8 evenly-spaced layers). '
                             'Use [23] for single-scale (last block only).')

    # BSE configuration
    parser.add_argument('--use_bse', action='store_true', default=True, help='Use BSE pooling')
    parser.add_argument('--voxel_size', type=float, default=0.05, help='BSE voxel size')
    parser.add_argument('--use_otsu', action='store_true', default=True, help='Use Otsu threshold')

    # Processing configuration
    parser.add_argument('--device', type=str, default='cuda:0', help='Device')
    parser.add_argument('--use_patch_based', action='store_true', default=False,
                        help='Use patch-based extraction')
    parser.add_argument('--depth_valid_range', type=float, nargs=2,
                        default=(0.1, 6.0), help='Valid depth range (min, max)')
    parser.add_argument(
        '--patch_depth_sampling',
        type=str,
        default='nearest_valid',
        choices=['nearest', 'median', 'nearest_valid'],
        help='Grid-center depth: nearest (dense GT), median (3×3), nearest_valid (Indoor6 sparse; '
             'same as fps_memory PATCH_DEPTH_SAMPLING)',
    )

    # SOR filtering
    parser.add_argument('--enable_sor', action='store_true', default=False,
                        help='Enable Statistical Outlier Removal')
    parser.add_argument('--sor_k', type=int, default=20, help='SOR k neighbors')
    parser.add_argument('--sor_std_ratio', type=float, default=2.0, help='SOR std ratio')

    # Advanced
    parser.add_argument('--temp_dir', type=str, default='/dev/shm',
                        help='Temporary directory for chunks')
    parser.add_argument('--dataset_type', type=str, default='auto',
                        choices=['auto', '7scenes', 'indoor6', 'custom'],
                        help='Dataset type (auto-detect by default)')
    parser.add_argument('--scene_name', type=str, default=None,
                        help='Scene name for WAI datasets (e.g. chess_train, scene2a_train). '
                             'If not provided, auto-derived from dataset_path.')
    parser.add_argument('--dataset_loader', type=str, default='wai',
                        choices=['wai', 'ace'],
                        help='Dataset loader: "wai" (MapAnything WAI format, default) or "ace" (ACE CamLocDatasetDINOv2)')

    # Ray pooling configuration
    parser.add_argument('--ray_pool_strategy', type=str, default='mean',
                        choices=['mean', 'dominant', 'first', 'all'],
                        help='Strategy for pooling ray directions')
    parser.add_argument('--save_all_ray_strategies', action='store_true', default=True,
                        help='Save all ray representations for comparison')

    # MapAnything infer() 后：预测位姿 vs 数据集 GT（与 fps_memory / map-anything run_memory_extraction 一致）
    parser.add_argument(
        '--pose_eval_translation_ok_m',
        type=float,
        default=0.1,
        help='Per-view translation error threshold (m) for OK vs HIGH; ref view is always REF',
    )
    parser.add_argument(
        '--pose_eval_strict',
        action='store_true',
        default=False,
        help='If set, exit with code 1 when any non-reference view exceeds --pose_eval_translation_ok_m',
    )

    args = parser.parse_args()

    return ExtractionConfig(
        dataset_path=args.dataset_path,
        output_path=args.output_path,
        n_memory=args.n_memory,
        device=args.device,
        voxel_size=args.voxel_size,
        use_bse=args.use_bse,
        use_otsu=args.use_otsu,
        depth_valid_range=tuple(args.depth_valid_range),
        patch_depth_sampling=args.patch_depth_sampling,
        temp_dir=args.temp_dir,
        model_str=args.model_str,
        model_config=args.model_config,
        model_checkpoint=args.model_checkpoint,
        dinov2_checkpoint=args.dinov2_checkpoint,
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
    )


def main():
    """Main entry point."""
    config = parse_args()
    device = torch.device(config.device)

    # Auto-derive scene_name from dataset_path if not provided
    scene_name = config.scene_name
    if scene_name is None:
        scene_name = os.path.basename(config.dataset_path.rstrip('/'))
    dataset_type = config.dataset_type
    if dataset_type == "auto":
        dataset_type = detect_dataset_type(config.dataset_path)

    print(f"[BSE Memory] Dataset: {config.dataset_path}")
    print(f"[BSE Memory] Scene: {scene_name} (type={dataset_type})")
    print(f"[BSE Memory] Output: {config.output_path}")
    print(f"[BSE Memory] N_MEMORY: {config.n_memory}, BSE: {config.use_bse}")
    print(f"[BSE Memory] Voxel size: {config.voxel_size}, Otsu: {config.use_otsu}")
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
    memory_indices = select_memory_views(train_dataset, config.n_memory)
    print(f"[BSE Memory] Selected {len(memory_indices)} views", flush=True)
    print(f"[Data] Selected memory indices ({len(memory_indices)}): {memory_indices}", flush=True)

    # Prepare input views (fast path: load directly from disk, skip WAI heavy processing)
    print("[BSE Memory] Preparing input views...", flush=True)
    raw_batches = []  # Store original data for depth/intrinsics access
    memory_views = []  # Store processed data for model input
    memory_gt_poses: List[torch.Tensor] = []  # For post-infer pose vs GT check (MapAnything)
    t0 = time.time()
    for i, idx in enumerate(memory_indices):
        if len(raw_batches) >= config.n_memory:
            break
        raw_data = train_dataset[idx]
        # WAI datasets return a list of view dicts; flatten to single views
        views = raw_data if isinstance(raw_data, list) else [raw_data]
        for view in views:
            if len(raw_batches) >= config.n_memory:
                break
            raw_batches.append(view)  # Save original data
            processed = prepare_batch_input(view, device)
            memory_views.append(processed)
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
    print(f"[Vis] Exporting depth/RGB validation under run dir: {output_run_dir}", flush=True)
    export_memory_views_visualization(
        raw_batches,
        memory_views,
        output_run_dir,
        depth_min=config.depth_valid_range[0],
        depth_max=config.depth_valid_range[1],
    )

    # Initialize feature extractor
    print("[BSE Memory] Initializing feature extractor...", flush=True)
    use_mapanything = config.use_model == "mapanything"

    if use_mapanything:
        try:
            extractor = MapAnythingExtractor(
                config.model_str or "mapanything_store_intermediates_ace",
                config.model_config or "default",
                config.model_checkpoint or "/data/xwh/checkpoints/facebook_map-anything.pth",
                device=str(device),
            )
            print("[BSE Memory] Using MapAnything extractor (default)")
        except (ImportError, Exception) as e:
            print(f"[BSE Memory] MapAnything unavailable ({e}), falling back to DINOv2 multi-scale")
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
        all_images = torch.cat([v['img'] for v in memory_views], dim=0)
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
    save_memory(config.output_path, result, mu, sigma, view_info)

    # Save PLY for visualization
    save_ply(config.output_path, result, mu, sigma)

    print("[BSE Memory] Done!")


if __name__ == '__main__':
    main()
