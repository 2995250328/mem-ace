# ace_network_ace.py
# ACE FCN encoder adapter + RegressorACE for use with the LMC pipeline.

import logging
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn

from ace_network import Encoder
from ace_network_dinov2 import Head

_logger = logging.getLogger(__name__)


class ACEEncoder(nn.Module):
    """
    Wraps the original ACE FCN Encoder for use in the LMC pipeline.

    Input : RGB images [B, 3, H, W]
    Output: [B, 512, H//8, W//8]
    """

    def __init__(self, pretrained_path, out_channels=512, freeze_backbone=True):
        super(ACEEncoder, self).__init__()

        self.out_channels = out_channels
        self.patch_size = 8
        self.freeze_backbone = freeze_backbone
        self.feature_dim = out_channels

        self.encoder = Encoder(out_channels=out_channels)

        _logger.info("Loading ACE FCN encoder from %s", pretrained_path)
        try:
            state_dict = torch.load(pretrained_path, map_location='cpu', weights_only=True)
        except TypeError:
            state_dict = torch.load(pretrained_path, map_location='cpu')

        if 'model' in state_dict:
            state_dict = state_dict['model']
        elif 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']

        self.encoder.load_state_dict(state_dict)

        if self.freeze_backbone:
            for param in self.encoder.parameters():
                param.requires_grad = False
            _logger.info("ACE FCN encoder frozen")

        # Buffer for RGB -> grayscale conversion weights
        self.register_buffer(
            "rgb_weights",
            torch.tensor([0.299, 0.587, 0.114], dtype=torch.float32).view(1, 3, 1, 1),
        )

    def forward(self, x):
        """
        Args:
            x: RGB images [B, 3, H, W]
        Returns:
            features: [B, out_channels, H//8, W//8]
        """
        # Ensure dtype consistency
        if x.dtype != self.rgb_weights.dtype:
            x = x.to(self.rgb_weights.dtype)

        # RGB -> grayscale
        gray = (x * self.rgb_weights).sum(dim=1, keepdim=True)  # [B, 1, H, W]

        with torch.set_grad_enabled(not self.freeze_backbone):
            features = self.encoder(gray)

        return features


class RegressorACE(nn.Module):
    """
    ACE FCN-based architecture for scene coordinate regression, compatible with
    the LMC pipeline (mirrors Regressor in ace_network_dinov2.py).

    Output is subsampled by a factor of 8 compared to the input.
    """

    OUTPUT_SUBSAMPLE = 8

    def __init__(self, mean, num_head_blocks, use_homogeneous,
                 encoder_path, num_encoder_features=512, freeze_backbone=True):
        super(RegressorACE, self).__init__()

        self.feature_dim = num_encoder_features

        self.encoder = ACEEncoder(
            pretrained_path=encoder_path,
            out_channels=self.feature_dim,
            freeze_backbone=freeze_backbone,
        )
        self.heads = Head(mean, num_head_blocks, use_homogeneous, in_channels=self.feature_dim)

    @classmethod
    def create_from_encoder(cls, encoder_path, mean, num_head_blocks, use_homogeneous,
                            num_encoder_features=512, freeze_backbone=True):
        _logger.info("Creating RegressorACE with FCN encoder, feature_dim=%d", num_encoder_features)
        return cls(mean, num_head_blocks, use_homogeneous, encoder_path,
                   num_encoder_features, freeze_backbone)

    @classmethod
    def create_from_state_dict(cls, state_dict, encoder_path):
        mean = torch.zeros((3,))
        pattern = re.compile(r"^heads\.\d+c0\.weight$")
        num_head_blocks = sum(1 for k in state_dict.keys() if pattern.match(k))
        use_homogeneous = state_dict["heads.fc3.weight"].shape[0] == 4
        num_encoder_features = state_dict['heads.res3_conv1.weight'].shape[1]

        _logger.info(
            "Creating RegressorACE from state_dict: head_blocks=%d, homogeneous=%s, feature_dim=%d",
            num_head_blocks, use_homogeneous, num_encoder_features,
        )
        regressor = cls(mean, num_head_blocks, use_homogeneous, encoder_path, num_encoder_features)
        regressor.heads.load_state_dict(
            {k.replace('heads.', ''): v for k, v in state_dict.items() if k.startswith('heads.')}
        )
        return regressor

    @classmethod
    def create_from_split_state_dict(cls, scene_path, encoder_path,
                                     num_head_blocks=4, use_homogeneous=True,
                                     freeze_backbone=True):
        """
        Create a RegressorACE from a scene directory (computes mean_cam_center from poses).

        scene_path   : Path to scene directory (contains train/poses/).
        encoder_path : Path to ace_encoder_pretrained.pt.
        """
        import numpy as np
        from pathlib import Path as _Path

        scene_path = _Path(scene_path)
        poses_dir = scene_path / "train" / "poses"
        pose_files = sorted(poses_dir.glob("*.txt"))

        cam_centers = []
        for pf in pose_files:
            pose = np.loadtxt(str(pf))
            if pose.shape == (4, 4):
                # Camera center = -R^T * t
                R = pose[:3, :3]
                t = pose[:3, 3]
                cam_centers.append(-R.T @ t)
            elif pose.shape == (3, 4):
                R = pose[:3, :3]
                t = pose[:3, 3]
                cam_centers.append(-R.T @ t)

        if cam_centers:
            mean_cam_center = np.mean(cam_centers, axis=0)
        else:
            mean_cam_center = np.zeros(3)

        mean = torch.tensor(mean_cam_center, dtype=torch.float32)
        _logger.info("RegressorACE mean_cam_center: %s", mean_cam_center)

        regressor = cls(
            mean=mean,
            num_head_blocks=num_head_blocks,
            use_homogeneous=use_homogeneous,
            encoder_path=encoder_path,
            num_encoder_features=512,
            freeze_backbone=freeze_backbone,
        )
        return regressor

    def get_features(self, inputs):
        return self.encoder(inputs)

    def get_scene_coordinates(self, features):
        return self.heads(features)

    def forward(self, inputs):
        features = self.get_features(inputs)
        return self.get_scene_coordinates(features)
