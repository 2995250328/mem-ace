# Copyright © Niantic, Inc. 2022.

import logging
import math
import re

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from superpoint import SuperPointNet
from depth_anything_v2.dpt import DepthAnythingV2

_logger = logging.getLogger(__name__)


class RelativeDepthLoss(nn.Module):
    def __init__(self, weight, max_samples=3000):
        """
        Args:
            weight: Weight balancing pair_loss vs ssi/grad loss.
            max_samples: Maximum number of pixels to sample for pairwise loss to prevent OOM.
        """
        super().__init__()
        self.weight = weight
        self.max_samples = max_samples  # Limit for sampling

        # Register buffers to avoid device mismatch issues
        self.register_buffer('alpha', None)
        self.register_buffer('beta', None)

    def compute_scale_shift(self, pred, target, mask):
        """
        Compute scale (alpha) and shift (beta) using Least Squares.
        Input shapes are flattened vectors of valid pixels.
        """
        # Ensure inputs are 1D vectors
        if pred.numel() == 0:
            return torch.tensor(1.0, device=pred.device), torch.tensor(0.0, device=pred.device)

        # A: [N, 2], b: [N, 1]
        A = torch.stack([pred, torch.ones_like(pred)], dim=1)
        b = target.unsqueeze(1)

        # Solve Ax = b
        try:
            x = torch.linalg.lstsq(A, b).solution
            alpha, beta = x[:2, 0]
        except Exception:
            # Fallback if Singular Matrix
            alpha = torch.tensor(1.0, device=pred.device)
            beta = torch.tensor(0.0, device=pred.device)

        return alpha, beta

    def apply_scale_shift(self, pred, alpha, beta):
        return alpha * pred + beta

    def compute_gradient(self, img):
        D_dy = torch.zeros_like(img)
        D_dx = torch.zeros_like(img)
        D_dy[:-1, :] = img[1:, :] - img[:-1, :]
        D_dx[:, :-1] = img[:, 1:] - img[:, :-1]
        return D_dx, D_dy

    def gradient_matching_loss(self, pred, target, mask):
        pred_dx, pred_dy = self.compute_gradient(pred)
        target_dx, target_dy = self.compute_gradient(target)

        # Apply mask
        loss_x = torch.abs(pred_dx - target_dx) * mask
        loss_y = torch.abs(pred_dy - target_dy) * mask

        return (loss_x.sum() + loss_y.sum()) / (mask.sum() + 1e-5) / 2.0

    def compute_pairwise_diff_sampled(self, pred_sample, gt_sample, coords_sample, alpha, beta):
        """
        Compute pairwise depth difference matrix on SAMPLED points.
        Args:
            pred_sample: [S] Sampled predicted depth
            gt_sample: [S] Sampled GT depth
            coords_sample: [S, 2] (y, x) coordinates of sampled points
            alpha: Scalar
            beta: Scalar
        """
        # 1. Align prediction
        pred_aligned = alpha * pred_sample + beta

        # 2. Compute depth differences
        # diff_pred[i, j] = pred[i] - pred[j]
        diff_pred = pred_aligned.unsqueeze(1) - pred_aligned.unsqueeze(0)  # [S, S]
        diff_gt = gt_sample.unsqueeze(1) - gt_sample.unsqueeze(0)  # [S, S]

        diff = diff_pred - diff_gt

        # 3. Compute spatial distances (Euclidean)
        # coords_sample is [S, 2]
        # dist[i, j] = || coord[i] - coord[j] ||
        spatial_dist = torch.norm(
            coords_sample.unsqueeze(1) - coords_sample.unsqueeze(0),
            dim=-1
        ) + 1.0  # Add 1.0 to avoid large division, or 1e-3 as before

        # 4. Normalize diff by spatial distance
        return diff / spatial_dist

    def forward(self, pred_depth, gt_depth, valid_mask, image=None):
        """
        Args:
            pred_depth: (H, W) or (B, H, W)
            gt_depth: (H, W) or (B, H, W)
            valid_mask: (H, W) or (B, H, W)
        """
        # Ensure inputs are 2D (H, W) for simplicity, or handle Batch dim
        # This implementation assumes single image input for simplicity as per original code context,
        # but robust to Batch dimension if flattened correctly.

        # 1. Masking and Flattening
        valid_idx = torch.nonzero(valid_mask)  # Indices of valid pixels [N_valid, 2 or 3]
        num_valid = valid_idx.shape[0]

        if num_valid == 0:
            return torch.tensor(0.0, device=pred_depth.device, requires_grad=True)

        # Extract values at valid indices
        if pred_depth.dim() == 3:  # (B, H, W)
            pred_vec = pred_depth[valid_idx[:, 0], valid_idx[:, 1], valid_idx[:, 2]]
            gt_vec = gt_depth[valid_idx[:, 0], valid_idx[:, 1], valid_idx[:, 2]]
        else:  # (H, W)
            pred_vec = pred_depth[valid_idx[:, 0], valid_idx[:, 1]]
            gt_vec = gt_depth[valid_idx[:, 0], valid_idx[:, 1]]

        # --- Part 1: SSI Loss & Gradient Loss (Calculated on ALL valid pixels) ---

        # 1.1 Compute Scale & Shift globally
        alpha, beta = self.compute_scale_shift(pred_vec, gt_vec, None)

        # 1.2 SSI Loss
        pred_aligned_vec = self.apply_scale_shift(pred_vec, alpha, beta)
        ssi_loss = torch.mean((pred_aligned_vec - gt_vec) ** 2)

        # 1.3 Gradient Loss (Needs 2D structure, so we compute on full image with mask)
        grad_loss = self.gradient_matching_loss(pred_depth, gt_depth, valid_mask)

        # --- Part 2: Pairwise Loss (Calculated on SAMPLED pixels to prevent OOM) ---
        if num_valid > self.max_samples:
            # Random sampling
            perm = torch.randperm(num_valid, device=pred_depth.device)[:self.max_samples]
            sample_idx = valid_idx[perm]

            # Extract sampled values
            if pred_depth.dim() == 3:
                # If (B, H, W), coords are (b, y, x). We need spatial coords (y, x) for distance
                coords_vec = sample_idx[:, 1:].float()
                pred_sample = pred_depth[sample_idx[:, 0], sample_idx[:, 1], sample_idx[:, 2]]
                gt_sample = gt_depth[sample_idx[:, 0], sample_idx[:, 1], sample_idx[:, 2]]
            else:
                coords_vec = sample_idx.float()  # (y, x)
                pred_sample = pred_depth[sample_idx[:, 0], sample_idx[:, 1]]
                gt_sample = gt_depth[sample_idx[:, 0], sample_idx[:, 1]]

        else:
            # Use all pixels if count is small
            if pred_depth.dim() == 3:
                coords_vec = valid_idx[:, 1:].float()
            else:
                coords_vec = valid_idx.float()
            pred_sample = pred_vec
            gt_sample = gt_vec

        # Compute Pairwise Diff Matrix (Size is max S*S, e.g. 3000*3000 approx 36MB)
        pair_diff_matrix = self.compute_pairwise_diff_sampled(
            pred_sample, gt_sample, coords_vec, alpha, beta
        )

        # Pair loss is the mean of absolute differences
        pair_loss = torch.mean(torch.abs(pair_diff_matrix))

        # --- Combine Losses ---
        total_loss = (1 - self.weight) * (ssi_loss + grad_loss) + self.weight * pair_loss

        return torch.log(total_loss + 1)

def find_neighbors_with_confidence(coords, H, W, patch_size, include_diagonal=8):
    """
    Find neighbors for a 3D coordinate tensor with confidence values, filter duplicates
    based on confidence, and return the results.

    Args:
        coords (np.ndarray): Shape (N, 3), where each row contains (x, y, confidence).
        H (int): Image height.
        W (int): Image width.
        patch_size (int): Size of the patch.
        include_diagonal (bool): Whether to include diagonal neighbors.

    Returns:
        np.ndarray: Filtered neighbors with shape (M, 4), where each row contains
                    (x, y, confidence, original_index).
    """
    # Define neighbor offsets
    offsets =[(0,0)]
    if include_diagonal==4:
        offsets += [
            (-1, 0), (1, 0), (0, -1), (0, 1)  # Up, Down, Left, Right
        ]
    elif include_diagonal==8:
        offsets += [
            (-1, -1), (-1, 1), (1, -1), (1, 1)  # Diagonal neighbors
        ]

    # Initialize results storage
    neighbors = []
    #print(f'{H}   {W}')

    for idx, (x, y, confidence) in enumerate(coords):
        for dx, dy in offsets:
            nx, ny = x + dx * patch_size, y + dy * patch_size
            #print(f'{nx},{ny}')

            # Check if the neighbor is within bounds
            if 0 <= nx < W and 0 <= ny < H:
                neighbors.append((nx, ny, confidence, idx))
    # Convert to numpy array
    neighbors = np.array(neighbors, dtype=np.float32)
    #print(neighbors.shape)

    # Sort neighbors by coordinates and confidence (descending)
    neighbors = neighbors[np.lexsort((-neighbors[:, 2], neighbors[:, 1], neighbors[:, 0]))]

    # Remove duplicates by keeping the highest confidence
    unique_neighbors = []
    seen_coords = set()

    for x, y, conf, orig_idx in neighbors:
        coord_key = (x, y)
        if coord_key not in seen_coords:
            unique_neighbors.append((x, y, conf, orig_idx))
            seen_coords.add(coord_key)

    return np.array(unique_neighbors, dtype=np.float32)

class Encoder(nn.Module):
    """
    FCN encoder, used to extract features from the input images.

    The number of output channels is configurable, the default used in the paper is 512.
    """

    def __init__(self, out_channels=512):
        super(Encoder, self).__init__()

        self.out_channels = out_channels

        self.conv1 = nn.Conv2d(1, 32, 3, 1, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 2, 1)
        self.conv3 = nn.Conv2d(64, 128, 3, 2, 1)
        self.conv4 = nn.Conv2d(128, 256, 3, 2, 1)

        self.res1_conv1 = nn.Conv2d(256, 256, 3, 1, 1)
        self.res1_conv2 = nn.Conv2d(256, 256, 1, 1, 0)
        self.res1_conv3 = nn.Conv2d(256, 256, 3, 1, 1)

        self.res2_conv1 = nn.Conv2d(256, 512, 3, 1, 1)
        self.res2_conv2 = nn.Conv2d(512, 512, 1, 1, 0)
        self.res2_conv3 = nn.Conv2d(512, self.out_channels, 3, 1, 1)

        self.res2_skip = nn.Conv2d(256, self.out_channels, 1, 1, 0)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        res = F.relu(self.conv4(x))

        x = F.relu(self.res1_conv1(res))
        x = F.relu(self.res1_conv2(x))
        x = F.relu(self.res1_conv3(x))

        res = res + x

        x = F.relu(self.res2_conv1(res))
        x = F.relu(self.res2_conv2(x))
        x = F.relu(self.res2_conv3(x))

        x = self.res2_skip(res) + x

        return x


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
                 in_channels=512):
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
    FCN architecture for scene coordinate regression.

    The network predicts a 3d scene coordinates, the output is subsampled by a factor of 8 compared to the input.
    """

    OUTPUT_SUBSAMPLE = 8

    def __init__(self, mean, num_head_blocks, use_homogeneous,
                 num_encoder_features=512):
        """
        Constructor.

        mean: Learn scene coordinates relative to a mean coordinate (e.g. the center of the scene).
        num_head_blocks: How many extra residual blocks to use in the head (one is always used).
        use_homogeneous: Whether to learn homogeneous or 3D coordinates.
        num_encoder_features: Number of channels output of the encoder network.
        """
        super(Regressor, self).__init__()

        self.feature_dim = num_encoder_features

        self.encoder = Encoder(out_channels=self.feature_dim)
        self.superpoint = SuperPointNet()
        self.heads = Head(mean, num_head_blocks, use_homogeneous, in_channels=self.feature_dim)

    @classmethod
    def create_from_encoder(cls, encoder_state_dict, mean, num_head_blocks, use_homogeneous):
        """
        Create a regressor using a pretrained encoder, loading encoder-specific parameters from the state dict.

        encoder_state_dict: pretrained encoder state dictionary.
        mean: Learn scene coordinates relative to a mean coordinate (e.g. the center of the scene).
        num_head_blocks: How many extra residual blocks to use in the head (one is always used).
        use_homogeneous: Whether to learn homogeneous or 3D coordinates.
        """

        # Number of output channels of the last encoder layer.
        num_encoder_features = encoder_state_dict['res2_conv3.weight'].shape[0]

        # Create a regressor.
        _logger.info(f"Creating Regressor using pretrained encoder with {num_encoder_features} feature size.")
        regressor = cls(mean, num_head_blocks, use_homogeneous,num_encoder_features)

        # Load encoder weights.
        regressor.encoder.load_state_dict(encoder_state_dict)

        # Done.
        return regressor

    @classmethod
    def create_from_state_dict(cls, state_dict):
        """
        Instantiate a regressor from a pretrained state dictionary.

        state_dict: pretrained state dictionary.
        """
        # Mean is zero (will be loaded from the state dict).
        mean = torch.zeros((3,))

        # Count how many head blocks are in the dictionary.
        pattern = re.compile(r"^heads\.\d+c0\.weight$")
        num_head_blocks = sum(1 for k in state_dict.keys() if pattern.match(k))

        # Whether the network uses homogeneous coordinates.
        use_homogeneous = state_dict["heads.fc3.weight"].shape[0] == 4

        # Number of output channels of the last encoder layer.
        num_encoder_features = state_dict['encoder.res2_conv3.weight'].shape[0]

        # Create a regressor.
        _logger.info(f"Creating regressor from pretrained state_dict:"
                     f"\n\tNum head blocks: {num_head_blocks}"
                     f"\n\tHomogeneous coordinates: {use_homogeneous}"
                     f"\n\tEncoder feature size: {num_encoder_features}")
        regressor = cls(mean, num_head_blocks, use_homogeneous,num_encoder_features)

        # Load all weights.
        regressor.load_state_dict(state_dict)

        # Done.
        return regressor

    @classmethod
    def create_from_split_state_dict(cls, encoder_state_dict, network_state_dict):
        """
        Instantiate a regressor from a pretrained encoder (scene-agnostic) and a scene-specific head.

        encoder_state_dict: encoder state dictionary
        head_state_dict: scene-specific head state dictionary
        """
        # We simply merge the dictionaries and call the other constructor.
        merged_state_dict = {}

        for k, v in encoder_state_dict.items():
            merged_state_dict[f"encoder.{k}"] = v

        first_value = next(iter(network_state_dict.values()))
        if isinstance(first_value, dict):
            # 说明是嵌套的，需要两层遍历
            for key in network_state_dict:
                for k, v in network_state_dict[key].items():
                    merged_state_dict[f"{key}.{k}"] = v
        else:
            # 说明是普通的 state_dict，直接一层遍历
            for k, v in network_state_dict.items():
                merged_state_dict[f"heads.{k}"] = v
        # for key in merged_state_dict.keys():
        #     print(key)

        return cls.create_from_state_dict(merged_state_dict)

    def load_encoder(self, encoder_dict_file):
        """
        Load weights into the encoder network.
        """
        self.encoder.load_state_dict(torch.load(encoder_dict_file))

    def get_features(self, inputs):
        return self.encoder(inputs)

    def get_scene_coordinates(self, features):
        return self.heads(features)
    
    def forward(self, inputs):
        """
        Forward pass.
        """
        # def normalize_shape(tensor_in):
        #     """Bring tensor from shape BxCxHxW to BxNxC"""
        #     return tensor_in.flatten(2).transpose(1, 2)
        # B,_,H,W = inputs.shape
        # features = self.get_features(inputs)
        # _,_,fH,fW = features.shape
        # #print(features.shape)
        # image_BHW = inputs.squeeze(1)
        # image_HW = image_BHW.squeeze(0)
        # image_HW_np = image_HW.cpu().numpy().astype(np.float32)
        # pts, desc, _ = superpoint.run(image_HW_np)
        # pts_tensor = torch.from_numpy(pts)
        # # desc_tensor = torch.from_numpy(desc)
        # pts_tensor = pts_tensor.permute(1,0).to(torch.device("cuda"),non_blocking=True,dtype=features.dtype)
        # pts_tensor[:,:2] = (pts_tensor[:,:2]//self.OUTPUT_SUBSAMPLE)*self.OUTPUT_SUBSAMPLE + \
        #                     torch.tensor(4,device=pts_tensor.device,dtype=pts_tensor.dtype)
        # pts_np = pts_tensor.cpu().numpy().astype(np.float32)
        # position_confidence_idx = find_neighbors_with_confidence(pts_np,H,W,8,4)
        # position_confidence_idx = torch.from_numpy(position_confidence_idx)
        # patch_idx = compute_patch_indices(position_confidence_idx[:,:2],H,W)
        # patch_idx = torch.tensor(patch_idx,device=inputs.device,dtype=torch.int)
        # scene_coordinates_BCHW = self.get_scene_coordinates(features)
        # scene_coordinates_BNC = normalize_shape(scene_coordinates_BCHW)
        # scene_coordinates_BNC = scene_coordinates_BNC[:,patch_idx,:]
        
        # return scene_coordinates_BCHW,scene_coordinates_BNC,patch_idx,fH,fW
        features = self.get_features(inputs)
        return self.get_scene_coordinates(features)
