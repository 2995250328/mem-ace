"""Single-frame and COLMAP-track inter-frame reprojection losses."""

import math
from dataclasses import dataclass

import torch

from .contracts import TrackBatch


@dataclass(frozen=True)
class ReprojectionConfig:
    depth_min: float = 0.1
    depth_max: float = 1000.0
    hard_clamp_px: float = 1000.0
    soft_clamp_px: float = 50.0
    soft_clamp_min_px: float = 1.0
    depth_target: float = 10.0


def to_homogeneous(points: torch.Tensor) -> torch.Tensor:
    return torch.cat([points, torch.ones_like(points[..., :1])], dim=-1)


def project_world_points(
    world_points: torch.Tensor,
    world_to_camera: torch.Tensor,
    intrinsics: torch.Tensor,
    depth_min: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Project paired world points with paired cameras."""
    camera_points = torch.bmm(
        world_to_camera,
        to_homogeneous(world_points).unsqueeze(-1),
    ).squeeze(-1)
    homogeneous_pixels = torch.bmm(intrinsics, camera_points.unsqueeze(-1)).squeeze(-1)
    depth = homogeneous_pixels[:, 2]
    pixels = homogeneous_pixels[:, :2] / depth.clamp(min=depth_min)[:, None]
    return pixels, depth, camera_points


def dynamic_tanh_sum(
    errors: torch.Tensor,
    step: int,
    total_steps: int,
    soft_clamp: float,
    soft_clamp_min: float,
    weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """ACE dynamic tanh robust penalty."""
    progress = min(1.0, max(0.0, float(step) / float(max(1, total_steps))))
    schedule = 1.0 - math.sqrt(max(0.0, 1.0 - progress**2))
    scale = (1.0 - schedule) * soft_clamp + soft_clamp_min
    penalty = scale * torch.tanh(errors / scale)
    if weights is not None:
        penalty = penalty * weights
    return penalty.sum()


def single_frame_reprojection_loss(
    predicted_world_points: torch.Tensor,
    source_pixels: torch.Tensor,
    source_world_to_camera: torch.Tensor,
    source_intrinsics: torch.Tensor,
    source_inverse_intrinsics: torch.Tensor,
    config: ReprojectionConfig,
    *,
    step: int,
    total_steps: int,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """ACE reprojection loss with the invalid-point camera-space proxy."""
    predicted_pixels, depth, camera_points = project_world_points(
        predicted_world_points,
        source_world_to_camera,
        source_intrinsics,
        config.depth_min,
    )
    errors = torch.linalg.vector_norm(predicted_pixels - source_pixels, ord=1, dim=1)
    finite = (
        torch.isfinite(predicted_pixels).all(dim=1)
        & torch.isfinite(camera_points).all(dim=1)
        & torch.isfinite(errors)
    )
    valid = (
        finite
        & (depth > config.depth_min)
        & (depth < config.depth_max)
        & (errors <= config.hard_clamp_px)
    )
    invalid = ~valid

    valid_loss = dynamic_tanh_sum(
        errors[valid],
        step,
        total_steps,
        config.soft_clamp_px,
        config.soft_clamp_min_px,
    )
    target_rays = torch.bmm(
        source_inverse_intrinsics,
        to_homogeneous(source_pixels).unsqueeze(-1),
    ).squeeze(-1)
    target_camera_points = config.depth_target * target_rays
    invalid_delta = torch.nan_to_num(
        (target_camera_points - camera_points).abs(),
        nan=0.0,
        posinf=1e4,
        neginf=1e4,
    )
    invalid_loss = invalid_delta[invalid].sum()
    normalizer = max(1, predicted_world_points.shape[0])
    loss = (valid_loss + invalid_loss) / normalizer
    return loss, {
        "valid_mask": valid,
        "pixel_l1": errors,
        "predicted_pixels": predicted_pixels,
    }


def inter_frame_track_reprojection_loss(
    predicted_world_points: torch.Tensor,
    tracks: TrackBatch,
    config: ReprojectionConfig,
    *,
    step: int,
    total_steps: int,
    max_error_px: float = 100.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Project source-frame predictions into target frames from the same track."""
    count = predicted_world_points.shape[0]
    tracks.validate(count)
    predicted_pixels, depth, _ = project_world_points(
        predicted_world_points,
        tracks.target_world_to_camera,
        tracks.target_intrinsics,
        config.depth_min,
    )
    errors = torch.linalg.vector_norm(
        predicted_pixels - tracks.target_pixels,
        ord=1,
        dim=1,
    )
    weights = tracks.weights.reshape(-1).clamp_min(0.0)
    valid = (
        tracks.valid.reshape(-1).bool()
        & (weights > 0)
        & torch.isfinite(predicted_pixels).all(dim=1)
        & torch.isfinite(errors)
        & (depth > config.depth_min)
        & (depth < config.depth_max)
    )
    selected_errors = errors[valid]
    if max_error_px > 0:
        selected_errors = selected_errors.clamp(max=max_error_px)
    loss = dynamic_tanh_sum(
        selected_errors,
        step,
        total_steps,
        config.soft_clamp_px,
        config.soft_clamp_min_px,
        weights=weights[valid],
    ) / max(1, count)
    return loss, {
        "valid_mask": valid,
        "pixel_l1": errors,
        "predicted_pixels": predicted_pixels,
    }
