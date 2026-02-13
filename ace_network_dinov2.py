# Copyright © Niantic, Inc. 2022.
# Modified to use DINOv2 as encoder

import logging
import math
import re

import torch
import torch.nn as nn
import torch.nn.functional as F

_logger = logging.getLogger(__name__)


def _build_dinov2_vitl14_local():
    """Build DINOv2 ViT-L/14 from local depth_anything_v2 (no network)."""
    from depth_anything_v2.dinov2 import DINOv2
    return DINOv2("vitl")


def _build_dinov2_vitl14_hub():
    """Build DINOv2 ViT-L/14 via torch.hub (requires network)."""
    return torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14', pretrained=False)


class DINOv2Encoder(nn.Module):
    """
    DINOv2 ViT-L/14 encoder for feature extraction.

    Input: RGB images (3 channels), size must be multiple of 14
    Output: 1024-channel feature maps
    """

    def __init__(self, pretrained_path, out_channels=1024, freeze_backbone=True, use_local_dinov2=True):
        super(DINOv2Encoder, self).__init__()

        self.out_channels = out_channels
        self.patch_size = 14
        self.freeze_backbone = freeze_backbone

        # Build DINOv2 ViT-L/14: prefer local (offline), fallback to torch.hub
        _logger.info(f"Loading DINOv2 from {pretrained_path}")
        if use_local_dinov2:
            try:
                self.dinov2 = _build_dinov2_vitl14_local()
                _logger.info("DINOv2 backbone built from local depth_anything_v2 (offline)")
            except Exception as e:
                _logger.warning(f"Local DINOv2 build failed: {e}, falling back to torch.hub")
                self.dinov2 = _build_dinov2_vitl14_hub()
        else:
            self.dinov2 = _build_dinov2_vitl14_hub()

        # Load weights from local checkpoint
        try:
            state_dict = torch.load(pretrained_path, map_location='cpu', weights_only=True)
        except TypeError:
            state_dict = torch.load(pretrained_path, map_location='cpu')
        # Handle different state dict formats
        if 'model' in state_dict:
            state_dict = state_dict['model']
        elif 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']

        self.dinov2.load_state_dict(state_dict, strict=False)

        # Freeze backbone if specified
        if self.freeze_backbone:
            for param in self.dinov2.parameters():
                param.requires_grad = False
            _logger.info("DINOv2 backbone frozen")

        # DINOv2 ViT-L outputs 1024-dim features
        self.feature_dim = 1024

        # Optional: Add a projection layer if out_channels != 1024
        if out_channels != self.feature_dim:
            self.projection = nn.Conv2d(self.feature_dim, out_channels, 1, 1, 0)
        else:
            self.projection = nn.Identity()

    def forward(self, x):
        """
        Args:
            x: Input RGB images [B, 3, H, W], H and W must be multiples of 14
        Returns:
            features: [B, out_channels, H//14, W//14]
        """
        # Avoid Half input + float weights (e.g. when no autocast in create_training_buffer)
        if x.dtype != next(self.dinov2.parameters()).dtype:
            x = x.to(next(self.dinov2.parameters()).dtype)
        B, C, H, W = x.shape

        assert H % self.patch_size == 0 and W % self.patch_size == 0, \
            f"Input size ({H}, {W}) must be multiple of patch_size ({self.patch_size})"

        # Get patch tokens from DINOv2 (without CLS token)
        # DINOv2 forward_features returns dict with 'x_norm_patchtokens'
        with torch.set_grad_enabled(not self.freeze_backbone):
            features_dict = self.dinov2.forward_features(x)
            patch_tokens = features_dict['x_norm_patchtokens']  # [B, N, 1024]

        # Reshape patch tokens to spatial feature map
        num_patches_h = H // self.patch_size
        num_patches_w = W // self.patch_size

        # [B, N, 1024] -> [B, 1024, H//14, W//14]
        features = patch_tokens.transpose(1, 2).reshape(B, self.feature_dim, num_patches_h, num_patches_w)

        # Apply projection if needed
        features = self.projection(features)

        return features


class Head(nn.Module):
    """
    MLP network predicting per-pixel scene coordinates given a feature vector. All layers are 1x1 convolutions.
    """

    def __init__(self,
                 mean,
                 num_head_blocks,
                 use_homogeneous,
                 homogeneous_min_scale=0.01,
                 homogeneous_max_scale=4.0,
                 in_channels=1024):  # Changed default from 512 to 1024 for DINOv2
        super(Head, self).__init__()

        self.use_homogeneous = use_homogeneous
        self.in_channels = in_channels  # Number of encoder features.
        self.head_channels = 512  # Hardcoded.

        # We may need a skip layer if the number of features output by the encoder is different.
        self.head_skip = nn.Identity() if self.in_channels == self.head_channels else nn.Conv2d(self.in_channels,
                                                                                                self.head_channels, 1,
                                                                                                1, 0)

        self.res3_conv1 = nn.Conv2d(self.in_channels, self.head_channels, 1, 1, 0)
        self.res3_conv2 = nn.Conv2d(self.head_channels, self.head_channels, 1, 1, 0)
        self.res3_conv3 = nn.Conv2d(self.head_channels, self.head_channels, 1, 1, 0)

        self.res_blocks = []

        for block in range(num_head_blocks):
            self.res_blocks.append((
                nn.Conv2d(self.head_channels, self.head_channels, 1, 1, 0),
                nn.Conv2d(self.head_channels, self.head_channels, 1, 1, 0),
                nn.Conv2d(self.head_channels, self.head_channels, 1, 1, 0),
            ))

            super(Head, self).add_module(str(block) + 'c0', self.res_blocks[block][0])
            super(Head, self).add_module(str(block) + 'c1', self.res_blocks[block][1])
            super(Head, self).add_module(str(block) + 'c2', self.res_blocks[block][2])

        self.fc1 = nn.Conv2d(self.head_channels, self.head_channels, 1, 1, 0)
        self.fc2 = nn.Conv2d(self.head_channels, self.head_channels, 1, 1, 0)

        if self.use_homogeneous:
            self.fc3 = nn.Conv2d(self.head_channels, 4, 1, 1, 0)

            # Use buffers because they need to be saved in the state dict.
            self.register_buffer("max_scale", torch.tensor([homogeneous_max_scale]))
            self.register_buffer("min_scale", torch.tensor([homogeneous_min_scale]))
            self.register_buffer("max_inv_scale", 1. / self.max_scale)
            self.register_buffer("h_beta", math.log(2) / (1. - self.max_inv_scale))
            self.register_buffer("min_inv_scale", 1. / self.min_scale)
        else:
            self.fc3 = nn.Conv2d(self.head_channels, 3, 1, 1, 0)

        # Learn scene coordinates relative to a mean coordinate (e.g. center of the scene).
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
            # Dehomogenize coords:
            # Softplus ensures we have a smooth homogeneous parameter with a minimum value = self.max_inv_scale.
            h_slice = F.softplus(sc[:, 3, :, :].unsqueeze(1), beta=self.h_beta.item()) + self.max_inv_scale
            h_slice.clamp_(max=self.min_inv_scale)
            sc = sc[:, :3] / h_slice

        # Add the mean to the predicted coordinates.
        sc += self.mean

        return sc


class Regressor(nn.Module):
    """
    DINOv2-based architecture for scene coordinate regression.

    The network predicts 3d scene coordinates, the output is subsampled by a factor of 14 compared to the input.
    """

    OUTPUT_SUBSAMPLE = 14  # Changed from 8 to 14 for DINOv2

    def __init__(self, mean, num_head_blocks, use_homogeneous,
                 dinov2_path, num_encoder_features=1024, freeze_backbone=True):
        """
        Constructor.

        mean: Learn scene coordinates relative to a mean coordinate (e.g. the center of the scene).
        num_head_blocks: How many extra residual blocks to use in the head (one is always used).
        use_homogeneous: Whether to learn homogeneous or 3D coordinates.
        dinov2_path: Path to DINOv2 pretrained weights.
        num_encoder_features: Number of channels output of the encoder network.
        freeze_backbone: Whether to freeze DINOv2 backbone.
        """
        super(Regressor, self).__init__()

        self.feature_dim = num_encoder_features

        self.encoder = DINOv2Encoder(
            pretrained_path=dinov2_path,
            out_channels=self.feature_dim,
            freeze_backbone=freeze_backbone
        )
        self.heads = Head(mean, num_head_blocks, use_homogeneous, in_channels=self.feature_dim)

    @classmethod
    def create_from_encoder(cls, dinov2_path, mean, num_head_blocks, use_homogeneous,
                          num_encoder_features=1024, freeze_backbone=True):
        """
        Create a regressor using DINOv2 encoder.

        dinov2_path: Path to DINOv2 pretrained weights.
        mean: Learn scene coordinates relative to a mean coordinate (e.g. the center of the scene).
        num_head_blocks: How many extra residual blocks to use in the head (one is always used).
        use_homogeneous: Whether to learn homogeneous or 3D coordinates.
        num_encoder_features: Number of output channels (default 1024 for ViT-L).
        freeze_backbone: Whether to freeze DINOv2 backbone.
        """
        _logger.info(f"Creating Regressor with DINOv2 encoder, feature size: {num_encoder_features}")
        regressor = cls(mean, num_head_blocks, use_homogeneous, dinov2_path,
                       num_encoder_features, freeze_backbone)
        return regressor

    @classmethod
    def create_from_state_dict(cls, state_dict, dinov2_path):
        """
        Instantiate a regressor from a pretrained state dictionary.

        state_dict: pretrained state dictionary (head only).
        dinov2_path: Path to DINOv2 pretrained weights.
        """
        # Mean is zero (will be loaded from the state dict).
        mean = torch.zeros((3,))

        # Count how many head blocks are in the dictionary.
        pattern = re.compile(r"^heads\.\d+c0\.weight$")
        num_head_blocks = sum(1 for k in state_dict.keys() if pattern.match(k))

        # Whether the network uses homogeneous coordinates.
        use_homogeneous = state_dict["heads.fc3.weight"].shape[0] == 4

        # Number of output channels of the encoder (from head input).
        num_encoder_features = state_dict['heads.res3_conv1.weight'].shape[1]

        # Create a regressor.
        _logger.info(f"Creating regressor from pretrained state_dict:"
                     f"\n\tNum head blocks: {num_head_blocks}"
                     f"\n\tHomogeneous coordinates: {use_homogeneous}"
                     f"\n\tEncoder feature size: {num_encoder_features}")
        regressor = cls(mean, num_head_blocks, use_homogeneous, dinov2_path, num_encoder_features)

        # Load head weights only (encoder is loaded separately)
        regressor.heads.load_state_dict({k.replace('heads.', ''): v for k, v in state_dict.items() if k.startswith('heads.')})

        # Done.
        return regressor

    @classmethod
    def create_from_split_state_dict(cls, dinov2_path, head_state_dict):
        """
        Instantiate a regressor from DINOv2 encoder and a scene-specific head.

        dinov2_path: Path to DINOv2 pretrained weights.
        head_state_dict: scene-specific head state dictionary
        """
        # Mean is zero (will be loaded from the state dict).
        mean = torch.zeros((3,))

        # Count how many head blocks are in the dictionary.
        pattern = re.compile(r"^\d+c0\.weight$")
        num_head_blocks = sum(1 for k in head_state_dict.keys() if pattern.match(k))

        # Whether the network uses homogeneous coordinates.
        use_homogeneous = head_state_dict["fc3.weight"].shape[0] == 4

        # Number of output channels of the encoder (from head input).
        num_encoder_features = head_state_dict['res3_conv1.weight'].shape[1]

        # Create a regressor.
        _logger.info(f"Creating regressor from split state_dict:"
                     f"\n\tNum head blocks: {num_head_blocks}"
                     f"\n\tHomogeneous coordinates: {use_homogeneous}"
                     f"\n\tEncoder feature size: {num_encoder_features}")
        regressor = cls(mean, num_head_blocks, use_homogeneous, dinov2_path, num_encoder_features)

        # Load head weights.
        regressor.heads.load_state_dict(head_state_dict)

        # Done.
        return regressor

    def get_features(self, inputs):
        return self.encoder(inputs)

    def get_scene_coordinates(self, features):
        return self.heads(features)

    def forward(self, inputs):
        """
        Forward pass.

        Args:
            inputs: RGB images [B, 3, H, W], H and W must be multiples of 14
        """
        features = self.get_features(inputs)
        return self.get_scene_coordinates(features)
