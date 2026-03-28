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
    ray_pool_strategy: str = "mean"  # 'mean', 'dominant', 'first', 'all'
    save_all_ray_strategies: bool = True  # Save all ray representations for comparison


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
    """Adapter for Map-Anything model with multi-scale feature extraction."""

    # Target intermediate layers (DPT-style: early, mid, late stages)
    TARGET_INTERM_LAYERS = [0, 6, 12, 18]

    def __init__(self, model_str: str, model_config: str, checkpoint: Optional[str] = None, device: str = "cuda:0"):
        """
        Initialize Map-Anything model adapter.

        Args:
            model_str: Model architecture string
            model_config: Path to model config
            checkpoint: Optional checkpoint path
            device: Device to run on
        """
        self.device = device
        try:
            from mapanything.models import init_model
            self.model = init_model(model_str, model_config)
            self.model = self.model.to(device).eval()
            if checkpoint:
                state = torch.load(checkpoint, map_location='cpu')
                self.model.load_state_dict(state.get("model", state), strict=False)

            # Force store intermediate features
            if hasattr(self.model, "model_config"):
                self.model.model_config.store_info_sharing_intermediate_features = True
        except ImportError as e:
            raise ImportError(
                f"Cannot import map-anything. Ensure it's installed or use DINOv2Extractor. Error: {e}"
            )

    def extract(
        self,
        images: torch.Tensor,
        depths: torch.Tensor,
        poses: torch.Tensor,
        intrinsics: torch.Tensor
    ) -> Dict[int, Dict[str, Any]]:
        """
        Extract features using Map-Anything model.

        Returns:
            Dict mapping view_idx to {'features': feat_list, 'grid_H': H, 'grid_W': W}
            where feat_list is a list of intermediate layer features.
        """
        # Prepare input views (mapanything format)
        input_views = []
        n_views = len(images)

        for i in range(n_views):
            view_dict = {
                'img': images[i:i+1],
                'depth': depths[i:i+1] if depths is not None else None,
                'camera_pose': poses[i:i+1],
                'camera_intrinsics': intrinsics[i:i+1]
            }
            input_views.append(view_dict)

        # Temporary file for saved features
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as tmp:
            save_path = tmp.name

        try:
            # Run inference with feature saving
            with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
                self.model.infer(
                    input_views,
                    memory_efficient_inference=True,
                    use_amp=True,
                    amp_dtype="bf16",
                    save_filename=save_path,
                    ignore_depth_inputs=False,
                    ignore_pose_inputs=False,
                    ignore_calibration_inputs=False
                )

            # Load saved features
            saved_data = torch.load(save_path, map_location='cpu')

            # Process features for each view
            result = {}
            raw_interm = saved_data.get("intermediate", [])
            raw_final = saved_data.get("final", {})

            for i in range(n_views):
                # Collect features from target intermediate layers + final layer
                feat_list = []

                for layer_idx in self.TARGET_INTERM_LAYERS:
                    if layer_idx < len(raw_interm):
                        feat = raw_interm[layer_idx]["features"][i]
                        if isinstance(feat, torch.Tensor):
                            feat_list.append(feat.detach().cpu())

                # Add final layer features
                if "features" in raw_final and i < len(raw_final["features"]):
                    feat = raw_final["features"][i]
                    if isinstance(feat, torch.Tensor):
                        feat_list.append(feat.detach().cpu())

                # Get grid dimensions from first feature
                grid_H, grid_W = None, None
                if len(feat_list) > 0:
                    first_feat = feat_list[0]
                    if first_feat.ndim == 3:  # (B, L, C) -> infer H, W
                        B, L, C = first_feat.shape
                        S = int(L ** 0.5)
                        if S * S == L:
                            grid_H, grid_W = S, S
                    elif first_feat.ndim == 4:  # (B, C, H, W)
                        B, C, H, W = first_feat.shape
                        grid_H, grid_W = H, W

                result[i] = {
                    'features': feat_list,
                    'grid_H': grid_H,
                    'grid_W': grid_W
                }

            return result

        finally:
            # Clean up temp file
            if os.path.exists(save_path):
                os.remove(save_path)


class DINOv2Extractor:
    """DINOv2-based feature extractor (standalone, no map-anything dependency)."""

    def __init__(self, checkpoint_path: str, device: str = "cuda:0"):
        """
        Initialize DINOv2 extractor.

        Args:
            checkpoint_path: Path to DINOv2 weights
            device: Device to run on
        """
        self.device = device
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

    def extract(
        self,
        images: torch.Tensor,
        depths: torch.Tensor,
        poses: torch.Tensor,
        intrinsics: torch.Tensor
    ) -> Dict[int, List[torch.Tensor]]:
        """Extract features using DINOv2."""
        features_dict = {}

        with torch.no_grad():
            for i in range(len(images)):
                img = images[i]  # [3, H, W]

                # DINOv2 expects ImageNet normalization (already done by dataset)
                # Extract patch features
                features = self.model.forward_features(img.unsqueeze(0))

                # Reshape to spatial grid
                # DINOv2 ViT-L/14: [1, N_patches, 1024]
                B, N, C = features.shape
                H_feat = W_feat = int(N ** 0.5)
                features = features.reshape(B, H_feat, W_feat, C).permute(0, 3, 1, 2)

                # Single-scale for DINOv2 (can be extended with intermediate layers)
                features_dict[i] = {'features': [features.squeeze(0)]}

        return features_dict


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
    Convert dataset raw data to model input format.

    Args:
        raw_data: Dict from dataset __getitem__
        device: Target device
        mode: "memory" or "query"

    Returns:
        Processed dict ready for model.infer()
    """
    # 1. Extract image (already normalized by dataset)
    image = raw_data['image'].to(device).unsqueeze(0)  # [1, 3, H, W]

    # 2. Extract depth (if available)
    depth = raw_data.get('depth', None)
    if depth is not None:
        depth = depth.to(device).unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]

    # 3. Extract pose [4x4 matrix]
    pose = raw_data['pose'].to(device).unsqueeze(0)  # [1, 4, 4]

    # 4. Extract intrinsics [3x3 matrix]
    intrinsics = raw_data['intrinsics'].to(device).unsqueeze(0)  # [1, 3, 3]

    return {
        'images': image,
        'depths': depth,
        'camera_poses': pose,
        'camera_intrinsics': intrinsics
    }


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
        if feat.ndim == 3:
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


def unproject_with_grid_features(
    view_data: Dict[str, torch.Tensor],
    features_flat: torch.Tensor,
    grid_H: int,
    grid_W: int,
    depth_valid_range: Tuple[float, float] = (0.1, 6.0),
    sampling_method: str = 'nearest'
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Unproject grid-resolution features to 3D using real depth.

    This samples depth at grid centers (like mapanything's generate_patch_point_cloud).

    Args:
        view_data: Dict with 'depths', 'camera_poses', 'camera_intrinsics', 'images'
        features_flat: [grid_H * grid_W, C] feature tensor (flattened)
        grid_H, grid_W: Grid dimensions
        depth_valid_range: (min, max) valid depth in meters
        sampling_method: 'nearest' or 'median' for depth sampling at grid centers

    Returns:
        points: [N, 3] world coordinates
        ray_dirs: [N, 3] unit ray directions
        colors: [N, 3] RGB colors
        features_flat: [N, C] features (may be filtered)
        camera_centers: [N, 3] camera center for each point
    """
    depth = view_data['depths'][0, 0]  # [H, W]
    pose = view_data['camera_poses'][0]  # [4, 4]
    K = view_data['camera_intrinsics'][0]  # [3, 3]
    img_H, img_W = depth.shape

    # Extract intrinsics
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    # Extract rotation and translation
    R = pose[:3, :3]  # [3, 3]
    t = pose[:3, 3]   # [3]
    camera_center = -R.T @ t  # [3]

    # Create grid center coordinates in image space
    # Map grid (u_grid, v_grid) to image (u_img, v_img)
    u_grid = torch.linspace(0, img_W - 1, grid_W, device=depth.device)
    v_grid = torch.linspace(0, img_H - 1, grid_H, device=depth.device)
    grid_u, grid_v = torch.meshgrid(u_grid, v_grid, indexing='xy')  # [grid_W, grid_H]

    # Flatten: [grid_H * grid_W]
    grid_u = grid_u.T.flatten()  # Row-major: v first, then u
    grid_v = grid_v.T.flatten()

    # Round to nearest integer for depth sampling
    u_int = grid_u.long().clamp(0, img_W - 1)
    v_int = grid_v.long().clamp(0, img_H - 1)

    # Sample depth at grid centers
    depth_sampled = depth[v_int, u_int]  # [N]

    # Filter valid depth
    valid_mask = (depth_sampled > depth_valid_range[0]) & (depth_sampled < depth_valid_range[1])

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

        rgb_sampled = F.grid_sample(
            rgb.unsqueeze(0), grid_coords,
            mode='bilinear', align_corners=False, padding_mode='border'
        )  # [1, 3, 1, N]

        colors = rgb_sampled[0, :, 0, :].T  # [N, 3]
    else:
        colors = torch.ones(len(pts_world), 3, device=depth.device) * 0.5

    # Filter features to match valid points
    features_filtered = features_flat[valid_mask]  # [N, C]

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
    K = view_data['camera_intrinsics'][0]  # [3, 3]

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
        memory_views: List of processed view dicts
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

        for view_idx, view_data in enumerate(tqdm(memory_views)):
            # Skip if already completed (resume capability)
            if view_idx in completed_views:
                continue

            # Get multi-scale features for this view
            view_info = features_dict[view_idx]
            feat_list = view_info['features']
            grid_H = view_info.get('grid_H', 37)  # Default for DINOv2
            grid_W = view_info.get('grid_W', 37)

            # Process features: align to grid and concatenate
            features_flat, _, _ = process_multiscale_features_to_grid(
                feat_list, apply_l2_norm=False
            )

            # Unproject with grid features (samples depth at grid centers)
            points, ray_dirs, colors, features_flat, camera_centers = unproject_with_grid_features(
                view_data, features_flat, grid_H, grid_W, config.depth_valid_range
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

            # Collect view-level camera information
            pose = view_data['camera_poses'][0]  # [4, 4]
            intrinsics = view_data['camera_intrinsics'][0]  # [3, 3]

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
            'plucker_main_rays': torch.stack(view_plucker_main_rays, dim=0)  # [M, 6]
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


def load_dataset(dataset_path: str, dataset_type: str = "auto"):
    """
    Load dataset based on type.

    Args:
        dataset_path: Path to dataset
        dataset_type: "7scenes", "indoor6", "custom", or "auto"

    Returns:
        Dataset object
    """
    if dataset_type == "auto":
        dataset_type = detect_dataset_type(dataset_path)

    try:
        from mapanything.datasets import SevenScenesWAI, Indoor6WAI

        if dataset_type == "7scenes":
            return SevenScenesWAI(dataset_path, split='train')
        elif dataset_type == "indoor6":
            return Indoor6WAI(dataset_path, split='train')
        else:
            # Fallback to ACE dataset loader
            from dataset_dinov2 import SevenScenes
            return SevenScenes(dataset_path, split='train')
    except ImportError:
        # Fallback to ACE dataset loader
        from dataset_dinov2 import SevenScenes
        return SevenScenes(dataset_path, split='train')


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
        from mapanything.tasks.ace.memory_selection import select_optimal_memory_indices
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
    parser.add_argument('--model_str', type=str, default=None, help='Model architecture string')
    parser.add_argument('--model_config', type=str, default=None, help='Model config path')
    parser.add_argument('--model_checkpoint', type=str, default=None, help='Model checkpoint path')
    parser.add_argument('--dinov2_checkpoint', type=str,
                        default='/data/xwh/checkpoints/dinov2_vitl14_pretrain.pth',
                        help='DINOv2 checkpoint path (for standalone mode)')

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

    # Ray pooling configuration
    parser.add_argument('--ray_pool_strategy', type=str, default='mean',
                        choices=['mean', 'dominant', 'first', 'all'],
                        help='Strategy for pooling ray directions')
    parser.add_argument('--save_all_ray_strategies', action='store_true', default=True,
                        help='Save all ray representations for comparison')

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
        ray_pool_strategy=args.ray_pool_strategy,
        save_all_ray_strategies=args.save_all_ray_strategies
    )


def main():
    """Main entry point."""
    config = parse_args()
    device = torch.device(config.device)

    print(f"[BSE Memory] Dataset: {config.dataset_path}")
    print(f"[BSE Memory] Output: {config.output_path}")
    print(f"[BSE Memory] N_MEMORY: {config.n_memory}, BSE: {config.use_bse}")
    print(f"[BSE Memory] Voxel size: {config.voxel_size}, Otsu: {config.use_otsu}")

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
    print("[BSE Memory] Loading dataset...")
    train_dataset = load_dataset(config.dataset_path, config.dataset_type)
    print(f"[BSE Memory] Train dataset: {len(train_dataset)} samples")

    # Select memory views
    print("[BSE Memory] Selecting memory views...")
    memory_indices = select_memory_views(train_dataset, config.n_memory)
    print(f"[BSE Memory] Selected {len(memory_indices)} views")

    # Prepare input views
    print("[BSE Memory] Preparing input views...")
    memory_views = []
    for idx in tqdm(memory_indices):
        raw_data = train_dataset[idx]
        processed = prepare_batch_input(raw_data, device)
        memory_views.append(processed)

    # Initialize feature extractor
    print("[BSE Memory] Initializing feature extractor...")
    if config.model_str and config.model_config:
        extractor = MapAnythingExtractor(
            config.model_str, config.model_config, config.model_checkpoint
        )
    else:
        print("[BSE Memory] Using standalone DINOv2 extractor")
        extractor = DINOv2Extractor(
            config.dinov2_checkpoint, device=str(device)
        )

    # Extract features
    print("[BSE Memory] Extracting features...")
    # Batch all views for efficient extraction
    all_images = torch.cat([v['images'] for v in memory_views], dim=0)
    all_depths = torch.cat([v['depths'] for v in memory_views], dim=0)
    all_poses = torch.cat([v['camera_poses'] for v in memory_views], dim=0)
    all_intrinsics = torch.cat([v['camera_intrinsics'] for v in memory_views], dim=0)

    features_dict = extractor.extract(all_images, all_depths, all_poses, all_intrinsics)

    # Two-pass processing
    print("[BSE Memory] Two-pass processing...")
    result, mu, sigma, view_info = two_pass_processing(
        memory_views, features_dict, pooler, welford, config
    )

    # Save memory
    save_memory(config.output_path, result, mu, sigma, view_info)

    print("[BSE Memory] Done!")


if __name__ == '__main__':
    main()
