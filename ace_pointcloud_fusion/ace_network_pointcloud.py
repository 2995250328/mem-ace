"""Image encoders with a point-cloud fusion adaptor."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ace_fcn_lmc"))

from ace_network_dinov2 import DINOv2Encoder, Head  # noqa: E402
from ace_fcn_lmc.ace_network_ace import ACEEncoder  # noqa: E402

from .adaptor import RayPointCrossAttentionAdaptor

_logger = logging.getLogger(__name__)


class RegressorACEPointCloudFusion(nn.Module):
    """ACE original FCN encoder + ray-aware point-cloud feature adaptor + ACE head."""

    OUTPUT_SUBSAMPLE = 8

    def __init__(
        self,
        mean,
        num_head_blocks: int,
        use_homogeneous: bool,
        encoder_path,
        point_feature_dim: int,
        *,
        image_feature_dim: int = 512,
        freeze_backbone: bool = True,
        adaptor_hidden_dim: int = 512,
        adaptor_heads: int = 8,
        adaptor_dropout: float = 0.1,
        max_context_points: int = 4096,
    ):
        super().__init__()
        self.feature_dim = int(image_feature_dim)
        self.point_feature_dim = int(point_feature_dim)

        self.encoder = ACEEncoder(
            pretrained_path=encoder_path,
            out_channels=self.feature_dim,
            freeze_backbone=freeze_backbone,
        )
        self.adaptor = RayPointCrossAttentionAdaptor(
            image_feature_dim=self.feature_dim,
            point_feature_dim=self.point_feature_dim,
            hidden_dim=adaptor_hidden_dim,
            num_heads=adaptor_heads,
            dropout=adaptor_dropout,
            max_context_points=max_context_points,
        )
        self.heads = Head(mean, num_head_blocks, use_homogeneous, in_channels=self.feature_dim)

    @classmethod
    def create_from_encoder(
        cls,
        encoder_path,
        mean,
        num_head_blocks,
        use_homogeneous,
        point_feature_dim,
        *,
        num_encoder_features: int = 512,
        freeze_backbone: bool = True,
        adaptor_hidden_dim: int = 512,
        adaptor_heads: int = 8,
        adaptor_dropout: float = 0.1,
        max_context_points: int = 4096,
    ):
        _logger.info(
            "Creating RegressorACEPointCloudFusion(image_dim=%d, point_dim=%d)",
            num_encoder_features,
            point_feature_dim,
        )
        return cls(
            mean=mean,
            num_head_blocks=num_head_blocks,
            use_homogeneous=use_homogeneous,
            encoder_path=encoder_path,
            point_feature_dim=point_feature_dim,
            image_feature_dim=num_encoder_features,
            freeze_backbone=freeze_backbone,
            adaptor_hidden_dim=adaptor_hidden_dim,
            adaptor_heads=adaptor_heads,
            adaptor_dropout=adaptor_dropout,
            max_context_points=max_context_points,
        )

    def get_features(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.encoder(inputs)

    def fuse_buffer_features(
        self,
        image_features_bC: torch.Tensor,
        camera_centers_b3: torch.Tensor,
        ray_dirs_b3: torch.Tensor,
        point_bank,
    ) -> torch.Tensor:
        return self.adaptor(
            image_features=image_features_bC,
            camera_centers=camera_centers_b3,
            ray_dirs=ray_dirs_b3,
            point_coords=point_bank["points"],
            point_features=point_bank["features"],
            scene_center=point_bank.get("scene_center"),
        )

    def get_scene_coordinates(self, features: torch.Tensor) -> torch.Tensor:
        return self.heads(features)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # Plain image-only forward is kept for smoke tests; training uses
        # fuse_buffer_features() before calling get_scene_coordinates().
        return self.get_scene_coordinates(self.get_features(inputs))


class RegressorDINOPointCloudFusion(RegressorACEPointCloudFusion):
    """DINOv2 ViT-L/14 encoder + ray-aware point-cloud feature adaptor + ACE head."""

    OUTPUT_SUBSAMPLE = 14

    def __init__(
        self,
        mean,
        num_head_blocks: int,
        use_homogeneous: bool,
        dinov2_path,
        point_feature_dim: int,
        *,
        image_feature_dim: int = 1024,
        freeze_backbone: bool = True,
        adaptor_hidden_dim: int = 512,
        adaptor_heads: int = 8,
        adaptor_dropout: float = 0.1,
        max_context_points: int = 4096,
    ):
        nn.Module.__init__(self)
        self.feature_dim = int(image_feature_dim)
        self.point_feature_dim = int(point_feature_dim)

        self.encoder = DINOv2Encoder(
            pretrained_path=dinov2_path,
            out_channels=self.feature_dim,
            freeze_backbone=freeze_backbone,
            use_local_dinov2=True,
        )
        self.adaptor = RayPointCrossAttentionAdaptor(
            image_feature_dim=self.feature_dim,
            point_feature_dim=self.point_feature_dim,
            hidden_dim=adaptor_hidden_dim,
            num_heads=adaptor_heads,
            dropout=adaptor_dropout,
            max_context_points=max_context_points,
        )
        self.heads = Head(mean, num_head_blocks, use_homogeneous, in_channels=self.feature_dim)

    @classmethod
    def create_from_encoder(
        cls,
        dinov2_path,
        mean,
        num_head_blocks,
        use_homogeneous,
        point_feature_dim,
        *,
        num_encoder_features: int = 1024,
        freeze_backbone: bool = True,
        adaptor_hidden_dim: int = 512,
        adaptor_heads: int = 8,
        adaptor_dropout: float = 0.1,
        max_context_points: int = 4096,
    ):
        _logger.info(
            "Creating RegressorDINOPointCloudFusion(image_dim=%d, point_dim=%d)",
            num_encoder_features,
            point_feature_dim,
        )
        return cls(
            mean=mean,
            num_head_blocks=num_head_blocks,
            use_homogeneous=use_homogeneous,
            dinov2_path=dinov2_path,
            point_feature_dim=point_feature_dim,
            image_feature_dim=num_encoder_features,
            freeze_backbone=freeze_backbone,
            adaptor_hidden_dim=adaptor_hidden_dim,
            adaptor_heads=adaptor_heads,
            adaptor_dropout=adaptor_dropout,
            max_context_points=max_context_points,
        )
