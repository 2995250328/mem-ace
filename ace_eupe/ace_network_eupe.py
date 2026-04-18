#!/usr/bin/env python3
"""ACE regressor with an EUPE backbone.

This module mirrors the ACE regressor surface so the trainer/evaluator can swap
the backbone with minimal changes.
The default target is EUPE ViT-B/16; EUPE ConvNeXt-B is also supported as a
separate replacement with its native final stride.
"""

from __future__ import annotations

import logging
import math
import re
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

_logger = logging.getLogger(__name__)


BACKBONE_SPECS = {
    "eupe_vitb16": {"feature_dim": 768, "output_subsample": 16, "checkpoint": "EUPE-ViT-B.pt"},
    "eupe_convnext_base": {"feature_dim": 1024, "output_subsample": 32, "checkpoint": "EUPE-ConvNeXt-B.pt"},
}


def backbone_spec(model_name: str) -> dict[str, int | str]:
    if model_name not in BACKBONE_SPECS:
        supported = ", ".join(sorted(BACKBONE_SPECS))
        raise ValueError(f"Unsupported EUPE model {model_name!r}. Supported: {supported}")
    return BACKBONE_SPECS[model_name]


class EUPEEncoder(nn.Module):
    """EUPE encoder returning dense features as ``B,C,H/stride,W/stride``."""

    SUPPORTED_BACKBONES = BACKBONE_SPECS

    def __init__(
        self,
        eupe_root: str | Path,
        checkpoint_path: str | Path,
        model_name: str = "eupe_vitb16",
        out_channels: int | None = None,
        freeze_backbone: bool = True,
    ):
        super().__init__()

        spec = backbone_spec(model_name)
        self.eupe_root = Path(eupe_root).expanduser().resolve()
        self.checkpoint_path = Path(checkpoint_path).expanduser().resolve()
        self.model_name = model_name
        self.output_subsample = int(spec["output_subsample"])
        self.freeze_backbone = freeze_backbone

        if not self.eupe_root.exists():
            raise FileNotFoundError(f"EUPE repository not found: {self.eupe_root}")
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"EUPE checkpoint not found: {self.checkpoint_path}")

        if str(self.eupe_root) not in sys.path:
            sys.path.insert(0, str(self.eupe_root))

        _logger.info(
            "Loading EUPE backbone %s from repo=%s weights=%s",
            self.model_name,
            self.eupe_root,
            self.checkpoint_path,
        )
        self.backbone = torch.hub.load(
            str(self.eupe_root),
            self.model_name,
            source="local",
            pretrained=True,
            weights=str(self.checkpoint_path),
        )

        self.feature_dim = int(getattr(self.backbone, "embed_dim", int(spec["feature_dim"])))
        out_channels = self.feature_dim if out_channels is None else int(out_channels)
        self.out_channels = out_channels

        if self.freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            _logger.info("EUPE backbone frozen")

        self.projection = nn.Identity() if out_channels == self.feature_dim else nn.Conv2d(self.feature_dim, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return patch features for ImageNet-normalized RGB images."""
        param = next(self.backbone.parameters())
        if x.dtype != param.dtype:
            x = x.to(param.dtype)

        batch_size, _, height, width = x.shape
        if height % self.output_subsample != 0 or width % self.output_subsample != 0:
            raise ValueError(
                f"EUPE input size ({height}, {width}) must be divisible by {self.output_subsample} "
                f"for {self.model_name}."
            )

        with torch.set_grad_enabled(not self.freeze_backbone):
            features_dict = self.backbone.forward_features(x)
            patch_tokens = features_dict["x_norm_patchtokens"]

        grid_h = height // self.output_subsample
        grid_w = width // self.output_subsample
        if patch_tokens.shape[1] != grid_h * grid_w:
            raise RuntimeError(
                f"EUPE {self.model_name} returned {patch_tokens.shape[1]} tokens, "
                f"but expected {grid_h * grid_w} for grid {grid_h}x{grid_w}."
            )
        features = patch_tokens.transpose(1, 2).reshape(batch_size, self.feature_dim, grid_h, grid_w)
        return self.projection(features)


class Head(nn.Module):
    """ACE 1x1-conv scene-coordinate head."""

    def __init__(
        self,
        mean,
        num_head_blocks,
        use_homogeneous,
        homogeneous_min_scale=0.01,
        homogeneous_max_scale=4.0,
        in_channels=768,
    ):
        super().__init__()

        self.use_homogeneous = use_homogeneous
        self.in_channels = in_channels
        self.head_channels = 512

        self.head_skip = (
            nn.Identity()
            if self.in_channels == self.head_channels
            else nn.Conv2d(self.in_channels, self.head_channels, 1, 1, 0)
        )

        self.res3_conv1 = nn.Conv2d(self.in_channels, self.head_channels, 1, 1, 0)
        self.res3_conv2 = nn.Conv2d(self.head_channels, self.head_channels, 1, 1, 0)
        self.res3_conv3 = nn.Conv2d(self.head_channels, self.head_channels, 1, 1, 0)

        self.res_blocks = []
        for block in range(num_head_blocks):
            self.res_blocks.append(
                (
                    nn.Conv2d(self.head_channels, self.head_channels, 1, 1, 0),
                    nn.Conv2d(self.head_channels, self.head_channels, 1, 1, 0),
                    nn.Conv2d(self.head_channels, self.head_channels, 1, 1, 0),
                )
            )
            self.add_module(str(block) + "c0", self.res_blocks[block][0])
            self.add_module(str(block) + "c1", self.res_blocks[block][1])
            self.add_module(str(block) + "c2", self.res_blocks[block][2])

        self.fc1 = nn.Conv2d(self.head_channels, self.head_channels, 1, 1, 0)
        self.fc2 = nn.Conv2d(self.head_channels, self.head_channels, 1, 1, 0)

        if self.use_homogeneous:
            self.fc3 = nn.Conv2d(self.head_channels, 4, 1, 1, 0)
            self.register_buffer("max_scale", torch.tensor([homogeneous_max_scale]))
            self.register_buffer("min_scale", torch.tensor([homogeneous_min_scale]))
            self.register_buffer("max_inv_scale", 1.0 / self.max_scale)
            self.register_buffer("h_beta", math.log(2) / (1.0 - self.max_inv_scale))
            self.register_buffer("min_inv_scale", 1.0 / self.min_scale)
        else:
            self.fc3 = nn.Conv2d(self.head_channels, 3, 1, 1, 0)

        self.register_buffer("mean", mean.clone().detach().view(1, 3, 1, 1))

    def forward(self, res):
        x = F.relu(self.res3_conv1(res))
        x = F.relu(self.res3_conv2(x))
        x = F.relu(self.res3_conv3(x))
        res = self.head_skip(res) + x

        for res_block in self.res_blocks:
            x = F.relu(res_block[0](res))
            x = F.relu(res_block[1](x))
            x = F.relu(res_block[2](x))
            res = res + x

        sc = F.relu(self.fc1(res))
        sc = F.relu(self.fc2(sc))
        sc = self.fc3(sc)

        if self.use_homogeneous:
            h_slice = F.softplus(sc[:, 3, :, :].unsqueeze(1), beta=self.h_beta.item()) + self.max_inv_scale
            h_slice.clamp_(max=self.min_inv_scale)
            sc = sc[:, :3] / h_slice

        sc += self.mean
        return sc


class Regressor(nn.Module):
    """EUPE-based ACE scene-coordinate regressor."""

    OUTPUT_SUBSAMPLE = 16

    def __init__(
        self,
        mean,
        num_head_blocks,
        use_homogeneous,
        eupe_root,
        eupe_checkpoint,
        eupe_model_name="eupe_vitb16",
        num_encoder_features=768,
        freeze_backbone=True,
    ):
        super().__init__()

        spec = backbone_spec(eupe_model_name)
        self.OUTPUT_SUBSAMPLE = int(spec["output_subsample"])
        self.feature_dim = int(num_encoder_features)
        self.encoder = EUPEEncoder(
            eupe_root=eupe_root,
            checkpoint_path=eupe_checkpoint,
            model_name=eupe_model_name,
            out_channels=self.feature_dim,
            freeze_backbone=freeze_backbone,
        )
        self.heads = Head(mean, num_head_blocks, use_homogeneous, in_channels=self.feature_dim)

    @classmethod
    def create_from_encoder(
        cls,
        eupe_root,
        eupe_checkpoint,
        mean,
        num_head_blocks,
        use_homogeneous,
        eupe_model_name="eupe_vitb16",
        num_encoder_features=768,
        freeze_backbone=True,
    ):
        _logger.info("Creating Regressor with EUPE encoder, feature size: %s", num_encoder_features)
        return cls(
            mean,
            num_head_blocks,
            use_homogeneous,
            eupe_root,
            eupe_checkpoint,
            eupe_model_name,
            num_encoder_features,
            freeze_backbone,
        )

    @classmethod
    def create_from_state_dict(cls, state_dict, eupe_root, eupe_checkpoint, eupe_model_name="eupe_vitb16"):
        mean = torch.zeros((3,))
        pattern = re.compile(r"^heads\.\d+c0\.weight$")
        num_head_blocks = sum(1 for key in state_dict.keys() if pattern.match(key))
        use_homogeneous = state_dict["heads.fc3.weight"].shape[0] == 4
        num_encoder_features = state_dict["heads.res3_conv1.weight"].shape[1]

        regressor = cls(
            mean,
            num_head_blocks,
            use_homogeneous,
            eupe_root,
            eupe_checkpoint,
            eupe_model_name,
            num_encoder_features,
        )
        regressor.heads.load_state_dict(
            {key.replace("heads.", ""): value for key, value in state_dict.items() if key.startswith("heads.")}
        )
        return regressor

    @classmethod
    def create_from_split_state_dict(
        cls,
        eupe_root,
        eupe_checkpoint,
        head_state_dict,
        eupe_model_name="eupe_vitb16",
    ):
        mean = torch.zeros((3,))
        pattern = re.compile(r"^\d+c0\.weight$")
        num_head_blocks = sum(1 for key in head_state_dict.keys() if pattern.match(key))
        use_homogeneous = head_state_dict["fc3.weight"].shape[0] == 4
        num_encoder_features = head_state_dict["res3_conv1.weight"].shape[1]

        regressor = cls(
            mean,
            num_head_blocks,
            use_homogeneous,
            eupe_root,
            eupe_checkpoint,
            eupe_model_name,
            num_encoder_features,
        )
        regressor.heads.load_state_dict(head_state_dict)
        return regressor

    def get_features(self, inputs):
        return self.encoder(inputs)

    def get_scene_coordinates(self, features):
        return self.heads(features)

    def forward(self, inputs):
        features = self.get_features(inputs)
        return self.get_scene_coordinates(features)
