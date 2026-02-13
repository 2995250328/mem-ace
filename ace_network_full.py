import logging
import math
import re
import torch
import torch.nn as nn
import torch.nn.functional as F

_logger = logging.getLogger(__name__)

# ==================================================================================
# Part 1: Standard ACE Encoder (无需修改)
# ==================================================================================

class Encoder(nn.Module):
    """FCN encoder."""

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


# ==================================================================================
# Part 2: Modified Head (支持不确定性)
# ==================================================================================

class Head(nn.Module):
    """MLP Head with optional Uncertainty Prediction."""

    def __init__(self, mean, num_head_blocks, use_homogeneous, use_uncertainty=False, in_channels=512):
        super(Head, self).__init__()
        self.use_homogeneous = use_homogeneous
        self.use_uncertainty = use_uncertainty # 新增开关
        self.in_channels = in_channels
        self.head_channels = 512

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

        # --- 核心修改：决定输出通道数 ---
        # 基础坐标通道: 齐次(4) 或 欧氏(3)
        base_channels = 4 if self.use_homogeneous else 3

        # 总通道数: 如果开启不确定性，额外 +1 (用于 log_variance)
        self.final_out_channels = base_channels + (1 if self.use_uncertainty else 0)

        self.fc3 = nn.Conv2d(self.head_channels, self.final_out_channels, 1, 1, 0)

        # 齐次坐标相关 buffer
        if self.use_homogeneous:
            self.register_buffer("max_scale", torch.tensor([4.0]))
            self.register_buffer("min_scale", torch.tensor([0.01]))
            self.register_buffer("max_inv_scale", 1. / self.max_scale)
            self.register_buffer("h_beta", math.log(2) / (1. - self.max_inv_scale))
            self.register_buffer("min_inv_scale", 1. / self.min_scale)

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
        sc = self.fc3(sc) # Output shape: [B, final_out_channels, H, W]

        # --- 处理不确定性通道 ---
        if self.use_uncertainty:
            # 最后一个通道是 uncertainty (log_var)
            uncertainty = sc[:, -1:, :, :] # [B, 1, H, W]
            coords_raw = sc[:, :-1, :, :]  # [B, 3 or 4, H, W]
        else:
            uncertainty = None
            coords_raw = sc

        # --- 处理坐标归一化 ---
        if self.use_homogeneous:
            # coords_raw 是 4 通道: X, Y, Z, H
            h_slice = F.softplus(coords_raw[:, 3, :, :].unsqueeze(1), beta=self.h_beta.item()) + self.max_inv_scale
            h_slice.clamp_(max=self.min_inv_scale)
            coords = coords_raw[:, :3] / h_slice
        else:
            coords = coords_raw

        # 加上均值
        coords += self.mean

        # --- 重新组合 ---
        if self.use_uncertainty:
            # 返回 [X, Y, Z, LogVar] -> 4 通道
            return torch.cat([coords, uncertainty], dim=1)
        else:
            # 返回 [X, Y, Z] -> 3 通道
            return coords


# ==================================================================================
# Part 3: Main Regressor Class (核心回归器 - 适配不确定性)
# ==================================================================================

class Regressor(nn.Module):
    OUTPUT_SUBSAMPLE = 8

    def __init__(self, mean, num_head_blocks, use_homogeneous, num_encoder_features=512, use_uncertainty=False):
        super(Regressor, self).__init__()
        self.feature_dim = num_encoder_features
        self.use_uncertainty = use_uncertainty
        self.encoder = Encoder(out_channels=self.feature_dim)

        # 传递 use_uncertainty 到底层 Head
        self.heads = Head(mean, num_head_blocks, use_homogeneous,
                          use_uncertainty=self.use_uncertainty,
                          in_channels=self.feature_dim)

    def get_features(self, inputs):
        """Extract features (No intrinsics needed)."""
        return self.encoder(inputs)

    def get_scene_coordinates(self, features):
        """Regress scene coordinates from features."""
        return self.heads(features)

    def forward(self, inputs):
        features = self.get_features(inputs)
        return self.get_scene_coordinates(features)

    @classmethod
    def create_from_encoder(cls, encoder_state_dict, mean, num_head_blocks, use_homogeneous, options=None):
        num_encoder_features = encoder_state_dict['res2_conv3.weight'].shape[0]

        # 从 options 获取 uncertainty 设置，默认为 False (回退兼容)
        use_uncertainty = getattr(options, 'use_uncertainty', False) if options else False

        _logger.info(f"Creating Regressor: feat_dim={num_encoder_features}, Uncertainty={use_uncertainty}")
        regressor = cls(mean, num_head_blocks, use_homogeneous, num_encoder_features, use_uncertainty=use_uncertainty)
        regressor.encoder.load_state_dict(encoder_state_dict)
        return regressor

    @classmethod
    def create_from_state_dict(cls, state_dict, options=None):
        # 1. 自动推断 Homogeneous
        if "heads.fc3.weight" in state_dict:
            fc3_out_channels = state_dict["heads.fc3.weight"].shape[0]
        else:
            # Fallback 假设
            fc3_out_channels = 4

            # 2. 自动推断 Uncertainty
        # 逻辑：
        # - 如果 fc3 输出 5 通道 -> Homogeneous(4) + Uncertainty(1)
        # - 如果 fc3 输出 4 通道 -> 可能是 Homogeneous(4) 无 Uncert，或者 Non-Homo(3) + Uncert(1)
        # - 如果 fc3 输出 3 通道 -> Non-Homo(3) 无 Uncert

        # 默认假设
        use_homogeneous = True
        use_uncertainty = False

        if fc3_out_channels == 5:
            use_homogeneous = True
            use_uncertainty = True
        elif fc3_out_channels == 4:
            # 这是一个歧义情况，但在 ACE 中绝大多数情况 fc3=4 意味着 use_homogeneous=True 且 use_uncertainty=False
            # 除非你显式训练了一个 Non-Homo + Uncertainty 的模型。
            # 为了安全，这里我们优先假设是标准的 Homogeneous 模式。
            use_homogeneous = True
            use_uncertainty = False
        elif fc3_out_channels == 3:
            use_homogeneous = False
            use_uncertainty = False

        # 如果 options 强制指定了，我们可以覆盖 (仅在初始化空模型时，加载权重可能会报错)
        # 但为了加载预训练模型，最好还是遵从 state_dict 的形状
        if options is not None:
            opt_uncertainty = getattr(options, 'use_uncertainty', False)
            if opt_uncertainty != use_uncertainty:
                _logger.warning(f"Warning: Options requesting use_uncertainty={opt_uncertainty}, "
                                f"but loaded weights suggest use_uncertainty={use_uncertainty}. "
                                f"Using weights configuration.")

        num_head_blocks = 0
        while f"heads.{num_head_blocks}c0.weight" in state_dict:
            num_head_blocks += 1

        if "encoder.res2_conv3.weight" in state_dict:
            num_encoder_features = state_dict["encoder.res2_conv3.weight"].shape[0]
        elif "heads.res3_conv1.weight" in state_dict:
            num_encoder_features = state_dict["heads.res3_conv1.weight"].shape[1]
        else:
            num_encoder_features = 512

        mean = torch.zeros(3)
        regressor = cls(mean, num_head_blocks, use_homogeneous, num_encoder_features, use_uncertainty=use_uncertainty)

        # 加载参数
        regressor.load_state_dict(state_dict, strict=False)
        return regressor

    @classmethod
    def create_from_split_state_dict(cls, encoder_state_dict, head_state_dict):
        merged_state_dict = {}
        for k, v in encoder_state_dict.items():
            if not k.startswith("encoder."):
                merged_state_dict[f"encoder.{k}"] = v
            else:
                merged_state_dict[k] = v

        # 处理 Heads
        if isinstance(head_state_dict, dict) and "heads" in head_state_dict:
            for k, v in head_state_dict["heads"].items(): merged_state_dict[f"heads.{k}"] = v
        else:
            for k, v in head_state_dict.items():
                if k.startswith("heads."):
                    merged_state_dict[k] = v
                elif not k.startswith("intrinsics_fusion.") and not k == "intrinsics_fusion_mode":
                    merged_state_dict[f"heads.{k}"] = v

        return cls.create_from_state_dict(merged_state_dict)


# ==================================================================================
# Part 4: Relative Depth Loss (无需修改，保留以维持文件完整性)
# ==================================================================================
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