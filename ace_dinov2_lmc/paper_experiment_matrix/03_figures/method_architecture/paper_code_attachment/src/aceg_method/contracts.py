"""Typed data contracts for the paper-facing ACE-G implementation."""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class SceneMemory:
    """Pooled scene memory aligned point-by-point in world coordinates."""

    points: torch.Tensor
    features: torch.Tensor
    scene_center: torch.Tensor

    def validate(self) -> "SceneMemory":
        if self.points.ndim not in (2, 3) or self.points.shape[-1] != 3:
            raise ValueError("points must have shape [N, 3] or [B, N, 3]")
        if self.features.ndim != self.points.ndim:
            raise ValueError("features and points must have the same rank")
        if self.features.shape[:-1] != self.points.shape[:-1]:
            raise ValueError("features must be aligned one-to-one with points")
        expected_center_shape = (3,) if self.points.ndim == 2 else (self.points.shape[0], 3)
        if tuple(self.scene_center.shape) != expected_center_shape:
            raise ValueError(f"scene_center must have shape {expected_center_shape}")
        if self.points.shape[-2] == 0:
            raise ValueError("scene memory must contain at least one point")
        for name, tensor in (
            ("points", self.points),
            ("features", self.features),
            ("scene_center", self.scene_center),
        ):
            if not torch.isfinite(tensor).all():
                raise ValueError(f"{name} contains NaN or Inf")
        return self

    def batched(self) -> "SceneMemory":
        self.validate()
        if self.points.ndim == 3:
            return self
        return SceneMemory(
            points=self.points.unsqueeze(0),
            features=self.features.unsqueeze(0),
            scene_center=self.scene_center.unsqueeze(0),
        )

    def to(self, device: torch.device | str) -> "SceneMemory":
        return SceneMemory(
            points=self.points.to(device),
            features=self.features.to(device),
            scene_center=self.scene_center.to(device),
        )


@dataclass(frozen=True)
class AnchoredSceneTokens:
    """Compressed scene representation: token content plus 3D anchor."""

    features: torch.Tensor
    anchors: torch.Tensor
    scene_center: torch.Tensor

    def validate(self) -> "AnchoredSceneTokens":
        if self.features.ndim != 3:
            raise ValueError("token features must have shape [B, K, C]")
        if self.anchors.shape != self.features.shape[:2] + (3,):
            raise ValueError("anchors must have shape [B, K, 3]")
        if self.scene_center.shape != (self.features.shape[0], 3):
            raise ValueError("scene_center must have shape [B, 3]")
        return self

    def detach(self) -> "AnchoredSceneTokens":
        return AnchoredSceneTokens(
            features=self.features.detach(),
            anchors=self.anchors.detach(),
            scene_center=self.scene_center.detach(),
        )


@dataclass(frozen=True)
class TrackBatch:
    """Target-frame geometry for COLMAP track reprojection."""

    target_pixels: torch.Tensor
    target_world_to_camera: torch.Tensor
    target_intrinsics: torch.Tensor
    valid: torch.Tensor
    weights: torch.Tensor

    def validate(self, count: int) -> "TrackBatch":
        if self.target_pixels.shape != (count, 2):
            raise ValueError("target_pixels must have shape [N, 2]")
        if self.target_world_to_camera.shape != (count, 3, 4):
            raise ValueError("target_world_to_camera must have shape [N, 3, 4]")
        if self.target_intrinsics.shape != (count, 3, 3):
            raise ValueError("target_intrinsics must have shape [N, 3, 3]")
        if self.valid.reshape(-1).shape[0] != count:
            raise ValueError("valid must contain N entries")
        if self.weights.reshape(-1).shape[0] != count:
            raise ValueError("weights must contain N entries")
        return self
