"""Cross-modal adaptors between ACE image features and scene point features."""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn


class FourierPositionEncoding(nn.Module):
    """Small Fourier encoder for 3D/6D geometry inputs."""

    def __init__(self, input_dim: int, num_bands: int = 6, max_freq: float = 16.0):
        super().__init__()
        self.input_dim = input_dim
        self.num_bands = num_bands
        self.output_dim = input_dim * (2 * num_bands + 1)
        freq = torch.logspace(0.0, math.log2(max_freq), num_bands, base=2.0)
        self.register_buffer("freq", freq.float())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xb = x.unsqueeze(-1) * self.freq
        sincos = torch.cat([xb.sin(), xb.cos()], dim=-1).flatten(-2)
        return torch.cat([x, sincos], dim=-1)


class RayPointCrossAttentionAdaptor(nn.Module):
    """Fuse per-sample ACE FCN features with a global scene point feature bank.

    Query geometry uses camera center and ray direction, so the adaptor can be
    used before scene-coordinate predictions are available. The point bank is
    downsampled to ``max_context_points`` for tractable attention.
    """

    def __init__(
        self,
        image_feature_dim: int = 512,
        point_feature_dim: int = 512,
        hidden_dim: int = 512,
        num_heads: int = 8,
        dropout: float = 0.1,
        max_context_points: int = 4096,
    ):
        super().__init__()
        self.image_feature_dim = image_feature_dim
        self.point_feature_dim = point_feature_dim
        self.hidden_dim = hidden_dim
        self.max_context_points = max_context_points

        self.ray_pe = FourierPositionEncoding(input_dim=6, num_bands=5)
        self.point_pe = FourierPositionEncoding(input_dim=3, num_bands=6)

        self.image_proj = nn.Linear(image_feature_dim, hidden_dim)
        self.ray_proj = nn.Linear(self.ray_pe.output_dim, hidden_dim)
        self.point_proj = nn.Linear(point_feature_dim, hidden_dim)
        self.point_pos_proj = nn.Linear(self.point_pe.output_dim, hidden_dim)

        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        self.out_proj = nn.Linear(hidden_dim, image_feature_dim)

    def _select_context(self, points: torch.Tensor, features: torch.Tensor):
        if self.max_context_points <= 0 or points.shape[0] <= self.max_context_points:
            return points, features
        # Deterministic uniform subsampling keeps training reproducible and cheap.
        idx = torch.linspace(
            0,
            points.shape[0] - 1,
            steps=self.max_context_points,
            device=points.device,
        ).long()
        return points.index_select(0, idx), features.index_select(0, idx)

    def forward(
        self,
        image_features: torch.Tensor,
        camera_centers: torch.Tensor,
        ray_dirs: torch.Tensor,
        point_coords: torch.Tensor,
        point_features: torch.Tensor,
        scene_center: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if image_features.dim() != 2:
            raise ValueError(f"image_features must be [N,C], got {tuple(image_features.shape)}.")
        if point_coords.dim() != 2 or point_features.dim() != 2:
            raise ValueError("point_coords and point_features must be rank-2 tensors.")

        dtype = image_features.dtype
        point_coords = point_coords.to(device=image_features.device, dtype=torch.float32)
        point_features = point_features.to(device=image_features.device, dtype=dtype)
        camera_centers = camera_centers.to(device=image_features.device, dtype=torch.float32)
        ray_dirs = ray_dirs.to(device=image_features.device, dtype=torch.float32)
        if scene_center is None:
            scene_center = point_coords.mean(dim=0)
        scene_center = scene_center.to(device=image_features.device, dtype=torch.float32)

        point_coords, point_features = self._select_context(point_coords, point_features)
        ray_dirs = torch.nn.functional.normalize(ray_dirs, dim=-1, eps=1e-6)

        ray_geo = torch.cat([camera_centers - scene_center, ray_dirs], dim=-1)
        query = self.image_proj(image_features) + self.ray_proj(self.ray_pe(ray_geo).to(dtype))

        rel_points = point_coords - scene_center
        key_value = self.point_proj(point_features) + self.point_pos_proj(self.point_pe(rel_points).to(dtype))

        query_b = query.unsqueeze(0)
        key_value_b = key_value.unsqueeze(0)
        attn_out, _ = self.attn(query_b, key_value_b, key_value_b, need_weights=False)
        x = self.norm1(query_b + self.dropout(attn_out))
        x = self.norm2(x + self.dropout(self.ffn(x)))
        return self.out_proj(x.squeeze(0))
