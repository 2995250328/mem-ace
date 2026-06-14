import logging
import random
import time
import numpy as np
import math
import cv2
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, sampler
import os
from datetime import datetime
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
from tqdm import tqdm

# 引入你的项目模块
from ace_util import get_pixel_grid, to_homogeneous
from ace_loss import ReproLoss
from ace_network_full import Regressor, RelativeDepthLoss
from dataset import CamLocDataset
from superpoint import SuperPointFrontend, SuperPointNet
from depth_anything_v2.dpt import DepthAnythingV2
import ace_vis_util as vutil

_logger = logging.getLogger(__name__)

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

# ==========================================
# 1. SegFormer Sky Estimator
# ==========================================
class SegFormerSkyEstimator:
    def __init__(self, device='cuda', model_path='/data/xwh/checkpoints/segformer_b0'):
        self.device = device
        _logger.info(f"Initializing SegFormer from local path: {model_path} ...")
        try:
            self.model = SegformerForSemanticSegmentation.from_pretrained(model_path, local_files_only=True).to(self.device).eval()
            self.mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
            self.std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)
            self.SKY_CLASS_ID = 2
        except Exception as e:
            _logger.error(f"Failed to load SegFormer: {e}")
            self.model = None

    @torch.no_grad()
    def get_sky_mask(self, image_tensor):
        if self.model is None: return torch.zeros_like(image_tensor)
        if image_tensor.shape[1] == 1: image_rgb = image_tensor.repeat(1, 3, 1, 1)
        else: image_rgb = image_tensor

        image_rgb = image_rgb.float()

        inputs = F.interpolate(image_rgb, size=(512, 512), mode='bilinear', align_corners=False)
        inputs = (inputs - self.mean) / self.std

        outputs = self.model(inputs)
        logits = F.interpolate(outputs.logits, size=(image_tensor.shape[2], image_tensor.shape[3]), mode='bilinear', align_corners=False)
        pred = logits.argmax(dim=1).unsqueeze(1)

        return (pred == self.SKY_CLASS_ID).float().to(image_tensor.dtype)

# ==========================================
# 2. Main Trainer Class
# ==========================================
class TrainerACE:
    def __init__(self, options):
        self.options = options
        self.device = torch.device('cuda:0')
        self.base_seed = 2089
        set_seed(self.base_seed)

        # Generators
        self.batch_generator = torch.Generator().manual_seed(self.base_seed + 1023)
        self.loader_generator = torch.Generator().manual_seed(self.base_seed + 511)
        self.sampling_generator = torch.Generator(device=self.device).manual_seed(self.base_seed + 4095)
        self.training_generator = torch.Generator().manual_seed(self.base_seed + 8191)

        self.iteration = 0
        self.num_data_loader_workers = 12

        # Dataset
        self.dataset = CamLocDataset(
            root_dir=self.options.scene / "train",
            mode=0,
            use_half=self.options.use_half,
            image_height=self.options.image_resolution,
            augment=self.options.use_aug,
            aug_rotation=self.options.aug_rotation,
            aug_scale_max=self.options.aug_scale,
            aug_scale_min=1 / self.options.aug_scale,
        )
        self.depth_iterator = None

        # Regressor
        encoder_state_dict = torch.load(self.options.encoder_path, map_location="cpu")
        self.regressor = Regressor.create_from_encoder(
            encoder_state_dict,
            mean=self.dataset.mean_cam_center,
            num_head_blocks=self.options.num_head_blocks,
            use_homogeneous=self.options.use_homogeneous
        ).to(self.device)
        self.regressor.train()

        # SuperPoint
        self.sp_net = SuperPointNet()
        self.superpoint = SuperPointFrontend(
            self.sp_net,
            weights_path=self.options.weights_path,
            nms_dist=self.options.nms_dist,
            conf_thresh=self.options.conf_thresh,
            nn_thresh=self.options.nn_thresh,
            cuda=True
        )

        # SegFormer
        segformer_path = getattr(self.options, 'segformer_path', '/mnt/storage/xwh/checkpoints/segformer_b0')
        self.sky_estimator = SegFormerSkyEstimator(device=self.device, model_path=segformer_path)

        # Sobel Kernels
        self.sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=self.device).view(1, 1, 3, 3)
        self.sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32, device=self.device).view(1, 1, 3, 3)

        # Depth Anything V2
        model_configs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
            'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
        }
        self.depth_anything = DepthAnythingV2(**model_configs[self.options.depth_encoder])
        self.depth_anything.load_state_dict(
            torch.load(f'/data/xwh/checkpoints/depth_anything_v2_{self.options.depth_encoder}.pth', map_location='cpu'))
        self.depth_anything = self.depth_anything.to(self.device).eval()

        # Optimizer & Scheduler
        chunk_size = getattr(self.options, 'buffer_chunk_size')
        total_size = self.options.training_buffer_size

        # [修改] 这里我们需要估算总 Steps，包含标准 Buffer 和 额外的 Active Buffer
        # 假设 Active Buffer 默认大小为 1 个 Chunk (如果 options 里没定)
        ACTIVE_BUFFER_SIZE = getattr(self.options, 'active_buffer_size', chunk_size)
        total_size_all = total_size + ACTIVE_BUFFER_SIZE

        num_chunks_total = math.ceil(total_size_all / chunk_size)
        steps_per_chunk = chunk_size // self.options.batch_size
        total_steps = num_chunks_total * self.options.epochs * steps_per_chunk

        self.optimizer = optim.AdamW(self.regressor.parameters(), lr=self.options.learning_rate_min)
        self.scheduler = optim.lr_scheduler.OneCycleLR(self.optimizer,
                                                       max_lr=self.options.learning_rate_max,
                                                       total_steps=total_steps,
                                                       cycle_momentum=False)
        self.scaler = GradScaler(enabled=self.options.use_half)

        self.pixel_grid_2HW = get_pixel_grid(self.regressor.OUTPUT_SUBSAMPLE).to(self.device)
        self.iterations = total_steps
        self.iterations_output = 100

        self.repro_loss = ReproLoss(
            total_iterations=self.iterations,
            soft_clamp=self.options.repro_loss_soft_clamp,
            soft_clamp_min=self.options.repro_loss_soft_clamp_min,
            type=self.options.repro_loss_type,
            circle_schedule=(self.options.repro_loss_schedule == 'circle')
        )
        self.depth_loss_func = RelativeDepthLoss(self.options.RelativeDepthLoss_weight).to(self.device)
        self.training_buffer = None

    def expand_neighbors_gpu(self, coords, H, W, patch_size, include_diagonal=8):
        if coords.shape[0] == 0: return coords
        offsets = [[0, 0]]
        if include_diagonal >= 4: offsets.extend([[-1, 0], [1, 0], [0, -1], [0, 1]])
        if include_diagonal == 8: offsets.extend([[-1, -1], [-1, 1], [1, -1], [1, 1]])
        offsets = torch.tensor(offsets, device=coords.device, dtype=coords.dtype) * patch_size
        expanded = (coords.unsqueeze(1) + offsets).reshape(-1, 2)
        valid_mask = (expanded[:, 0] >= 0) & (expanded[:, 0] < W) & (expanded[:, 1] >= 0) & (expanded[:, 1] < H)
        return torch.unique(expanded[valid_mask], dim=0)

    # -------------------------------------------------------------------------
    # 计算重投影误差图
    # -------------------------------------------------------------------------
    @torch.no_grad()
    def compute_reprojection_error_map(self, image_B1HW, gt_pose_inv, intrinsics):
        B, _, H_in, W_in = image_B1HW.shape

        # 1. 快速推理 (Feature Only, 输出是下采样过的)
        with autocast(enabled=self.options.use_half):
            features = self.regressor.get_features(image_B1HW)
            scene_coords = self.regressor.get_scene_coordinates(features).float() # (B, 3, H_out, W_out)

        # 获取输出特征图的尺寸 (H_out, W_out)
        _, _, H_out, W_out = scene_coords.shape
        Num_Points = H_out * W_out

        # 2. 投影回相机坐标
        coords_flat = scene_coords.view(B, 3, -1) # (B, 3, N_out)

        # [修复] ones 的大小必须匹配网络输出点数，而不是原图点数
        ones = torch.ones((B, 1, Num_Points), device=self.device)
        coords_homo = torch.cat([coords_flat, ones], dim=1) # (B, 4, N_out)

        cam_coords = torch.bmm(gt_pose_inv, coords_homo) # (B, 3, N_out)

        # 3. 投影回像素坐标
        uv_proj_homo = torch.bmm(intrinsics, cam_coords) # (B, 3, N_out)
        z = uv_proj_homo[:, 2:3, :] + 1e-6
        uv_pred = uv_proj_homo[:, :2, :] / z # (B, 2, N_out)

        # 4. 获取 GT 像素坐标
        # 确保 pixel_grid_2HW 尺寸匹配当前的 H_out, W_out
        # self.pixel_grid_2HW 通常是预设的最大尺寸，需要切片以匹配当前 batch
        current_grid = self.pixel_grid_2HW[:, :H_out, :W_out]
        grid = current_grid.reshape(2, -1).unsqueeze(0).expand(B, 2, -1) # (B, 2, N_out)

        # 5. 计算 L2 误差 (在低分辨率下计算)
        diff = uv_pred - grid
        error_map_low_res = torch.norm(diff, dim=1).view(B, 1, H_out, W_out)

        # 6. 过滤无效深度
        valid_z = (cam_coords[:, 2, :].view(B, 1, H_out, W_out) > 0.1) & \
                  (cam_coords[:, 2, :].view(B, 1, H_out, W_out) < 1000.0)
        error_map_low_res = error_map_low_res * valid_z.float()

        # 7. [关键] 上采样回原图尺寸
        # 这样才能和 image_mask, sky_mask 等全分辨率掩码对齐
        error_map_full = F.interpolate(error_map_low_res, size=(H_in, W_in), mode='bilinear', align_corners=False)

        return error_map_full

    def get_strategy_mask(self, image_gray, valid_mask, mode, active_error_map=None):
        B, _, H, W = image_gray.shape
        strategy_mask = torch.ones((B, 1, H, W), device=self.device, dtype=image_gray.dtype)

        if mode == 'active':
            if active_error_map is not None:
                err = active_error_map.float()
                weights = torch.full_like(err, 0.1)

                # Hard Examples
                hard_mask = (err > 5.0) & (err < 500.0)
                weights[hard_mask] = 5.0

                # Easy Examples
                easy_mask = (err < 2.0)
                weights[easy_mask] = 1.0

                strategy_mask = weights.to(dtype=image_gray.dtype)
            else:
                strategy_mask.fill_(1.0)

        elif mode == 'edge':
            img_pad = F.pad(image_gray, (1, 1, 1, 1), mode='replicate')
            curr_sobel_x = self.sobel_x.to(dtype=image_gray.dtype)
            curr_sobel_y = self.sobel_y.to(dtype=image_gray.dtype)
            grad_x = F.conv2d(img_pad, curr_sobel_x)
            grad_y = F.conv2d(img_pad, curr_sobel_y)
            edge_map = torch.sqrt(grad_x**2 + grad_y**2)
            strategy_mask = edge_map

        elif mode == 'ground':
            y_grid = torch.linspace(0, 1, H, device=self.device).view(1, 1, H, 1).expand(B, 1, H, W)
            strategy_mask = (y_grid > 0.6).to(dtype=image_gray.dtype)

        elif mode == 'sky':
            strategy_mask = self.sky_estimator.get_sky_mask(image_gray).to(dtype=image_gray.dtype)

        else:
            strategy_mask.fill_(1.0)

        if valid_mask is not None:
            if valid_mask.ndim == 3: valid_mask = valid_mask.unsqueeze(1)
            strategy_mask = strategy_mask * valid_mask.to(dtype=image_gray.dtype)

        return strategy_mask

    # =========================================================================
    # [完整修复版] 创建训练 Buffer
    # =========================================================================
    def create_training_buffer(self, chunk_size, mode='hybrid'):
        """
        mode:
          - 'hybrid': SuperPoint + Options策略 (Random/Edge)
          - 'active': 纯 Active Learning (Error Map)
        """
        # 1. [OOM Fix] 强制释放旧 Buffer，腾出显存
        if self.training_buffer is not None:
            # _logger.info("Cleaning up previous buffer to free GPU memory...")
            del self.training_buffer
            self.training_buffer = None
            import gc
            gc.collect()
            torch.cuda.empty_cache()

        SAMPLES_PER_IMG = self.options.samples_per_image

        # 2. 确定当前阶段策略
        if mode == 'active':
            CURRENT_SP_RATIO = 0.0
            CURRENT_MODE = 'active'
        else:
            CURRENT_SP_RATIO = getattr(self.options, 'sp_ratio', 0.5)
            # 如果配置写了 active 但当前是 hybrid 阶段，降级为 edge
            opt_mode = getattr(self.options, 'sampling_mode', 'random')
            CURRENT_MODE = opt_mode if opt_mode != 'active' else 'edge'

        torch.backends.cudnn.benchmark = False
        self.regressor.eval()

        # 3. 申请新 Buffer (Wrap in try-except for safety)
        try:
            self.training_buffer = {
                'features': torch.empty((chunk_size, self.regressor.feature_dim),
                                        dtype=(torch.float32, torch.float16)[self.options.use_half],
                                        device=self.device),
                'target_px': torch.empty((chunk_size, 2), dtype=torch.float32, device=self.device),
                'gt_poses_inv': torch.empty((chunk_size, 3, 4), dtype=torch.float32, device=self.device),
                'intrinsics': torch.empty((chunk_size, 3, 3), dtype=torch.float32, device=self.device),
                'intrinsics_inv': torch.empty((chunk_size, 3, 3), dtype=torch.float32, device=self.device)
            }
        except torch.cuda.OutOfMemoryError:
            _logger.error(
                f"OOM triggered when allocating buffer of size {chunk_size}. Try reducing buffer_chunk_size.")
            torch.cuda.empty_cache()
            raise

        batch_sampler = sampler.BatchSampler(sampler.RandomSampler(self.dataset, generator=self.batch_generator),
                                             batch_size=1, drop_last=False)
        loader = DataLoader(self.dataset, sampler=batch_sampler, batch_size=None,
                            num_workers=self.num_data_loader_workers)

        buffer_idx = 0
        patch_size = self.regressor.OUTPUT_SUBSAMPLE

        desc_str = f"Fill ({mode.upper()}): UseSP={CURRENT_SP_RATIO > 0}, Strategy={CURRENT_MODE}"

        with tqdm(total=chunk_size, desc=desc_str, unit="px") as pbar:
            with torch.no_grad():
                while buffer_idx < chunk_size:
                    for _, image_B1HW, image_mask_B1HW, _, gt_pose_inv_B44, intrinsics_B33, intrinsics_inv_B33, _, _ in loader:

                        image_B1HW = image_B1HW.to(self.device, non_blocking=True)
                        image_mask_B1HW = image_mask_B1HW.to(self.device, non_blocking=True)
                        gt_pose_inv_B44 = gt_pose_inv_B44.to(self.device, non_blocking=True)
                        intrinsics_B33 = intrinsics_B33.to(self.device, non_blocking=True)

                        B, _, H, W = image_B1HW.shape

                        img_rate = min(H, W) / 480.0
                        total_target = int(img_rate * SAMPLES_PER_IMG)

                        target_sp_count = int(total_target * CURRENT_SP_RATIO)

                        # 4. 计算 Sky Mask (SP 阶段必用)
                        sky_mask_map = self.sky_estimator.get_sky_mask(image_B1HW)

                        # 5. Active 模式下计算误差图
                        error_map = None
                        if CURRENT_MODE == 'active':
                            error_map = self.compute_reprojection_error_map(image_B1HW, gt_pose_inv_B44[:, :3],
                                                                            intrinsics_B33)

                        # =======================================================
                        # Part A: SuperPoint 采样 (Stable)
                        # =======================================================
                        sp_coords_all = torch.zeros((0, 2), device=self.device)
                        if target_sp_count > 0:
                            image_HW_np = image_B1HW.squeeze().cpu().numpy().astype(np.float32)
                            pts, _, _ = self.superpoint.run(image_HW_np)

                            if pts is not None and pts.shape[1] > 0:
                                sp_coords = torch.from_numpy(pts[:2].T).to(self.device, dtype=torch.float32)
                                sp_coords = (sp_coords // patch_size) * patch_size + (patch_size // 2)
                                sp_coords_all = self.expand_neighbors_gpu(sp_coords, H, W, patch_size,
                                                                          include_diagonal=4)

                                if sp_coords_all.shape[0] > 0:
                                    sp_y = sp_coords_all[:, 1].long().clamp(0, H - 1)
                                    sp_x = sp_coords_all[:, 0].long().clamp(0, W - 1)

                                    # SP 始终过滤天空
                                    valid_sp = (sky_mask_map[0, 0, sp_y, sp_x] < 0.5)
                                    in_img = (image_mask_B1HW[0, 0, sp_y, sp_x] > 0)
                                    valid_sp = valid_sp & in_img

                                    sp_coords_all = sp_coords_all[valid_sp]

                                    if sp_coords_all.shape[0] > target_sp_count:
                                        perm = torch.randperm(sp_coords_all.shape[0], device=self.device)
                                        sp_coords_all = sp_coords_all[perm[:target_sp_count]]

                        # =======================================================
                        # Part B: 补充/全量采样 (Robust Fix)
                        # =======================================================
                        needed_supplement = total_target - sp_coords_all.shape[0]
                        supplement_coords = torch.zeros((0, 2), device=self.device)

                        if needed_supplement > 0:
                            # 1. 获取权重图
                            weights_map = self.get_strategy_mask(image_B1HW, image_mask_B1HW, mode=CURRENT_MODE,
                                                                 active_error_map=error_map)

                            # 2. 叠加 Sky 过滤 (Active 模式除外)
                            if CURRENT_MODE != 'active' and CURRENT_MODE != 'sky':
                                weights_map = weights_map * (1.0 - sky_mask_map)

                            weights_flat = weights_map.view(-1)

                            # ---------------------------------------------------
                            # [Robust Sampling Logic]
                            # ---------------------------------------------------

                            # 分支 1: 均匀采样 (Random / Sky) -> 避免 multinomial
                            if CURRENT_MODE == 'random' or CURRENT_MODE == 'sky':
                                # 只要 mask > 0.5 就是有效区域
                                valid_indices = torch.nonzero(weights_flat > 0.5).reshape(-1)

                                if valid_indices.numel() > 0:
                                    count = min(valid_indices.numel(), needed_supplement)
                                    # 直接随机取索引，快且稳
                                    rand_idx = torch.randint(0, valid_indices.numel(), (count,), device=self.device)
                                    global_idx = valid_indices[rand_idx]

                                    # 转坐标
                                    ys = torch.div(global_idx, W, rounding_mode='floor')
                                    xs = global_idx % W
                                    supplement_coords = torch.stack((xs, ys), dim=1).float()
                                    supplement_coords = (supplement_coords // patch_size) * patch_size + (
                                                patch_size // 2)

                            # 分支 2: 加权采样 (Active / Edge) -> 需小心 multinomial
                            else:
                                valid_indices = torch.nonzero(weights_flat > 1e-4).reshape(-1)

                                if valid_indices.numel() > 0:
                                    count = min(valid_indices.numel(), needed_supplement)

                                    # [Safety] 强转 Float32 计算概率
                                    probs = weights_flat[valid_indices].float()

                                    # [Safety] 清理 NaN/Inf
                                    if not torch.isfinite(probs).all():
                                        probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)

                                    probs_sum = probs.sum()

                                    # [Safety] 检查和是否有效
                                    if probs_sum > 1e-6:
                                        # 归一化
                                        probs = probs / probs_sum

                                        # 双重检查归一化结果
                                        if torch.abs(probs.sum() - 1.0) < 1e-3:
                                            try:
                                                local_idx = torch.multinomial(probs, count, replacement=True)
                                                global_idx = valid_indices[local_idx]
                                            except RuntimeError:
                                                # 极罕见情况，回退
                                                rand_idx = torch.randint(0, valid_indices.numel(), (count,),
                                                                         device=self.device)
                                                global_idx = valid_indices[rand_idx]
                                        else:
                                            # 归一化失败回退
                                            rand_idx = torch.randint(0, valid_indices.numel(), (count,),
                                                                     device=self.device)
                                            global_idx = valid_indices[rand_idx]
                                    else:
                                        # 概率和为0回退
                                        rand_idx = torch.randint(0, valid_indices.numel(), (count,),
                                                                 device=self.device)
                                        global_idx = valid_indices[rand_idx]

                                    # 转坐标
                                    ys = torch.div(global_idx, W, rounding_mode='floor')
                                    xs = global_idx % W
                                    supplement_coords = torch.stack((xs, ys), dim=1).float()
                                    supplement_coords = (supplement_coords // patch_size) * patch_size + (
                                                patch_size // 2)

                        # =======================================================
                        # Part C: Merge & Write
                        # =======================================================
                        final_coords = torch.cat((sp_coords_all, supplement_coords), dim=0)
                        final_coords = torch.unique(final_coords, dim=0)

                        if final_coords.shape[0] == 0: continue

                        norm_x = 2.0 * final_coords[:, 0] / (W - 1) - 1.0
                        norm_y = 2.0 * final_coords[:, 1] / (H - 1) - 1.0
                        grid = torch.stack((norm_x, norm_y), dim=1).unsqueeze(0).unsqueeze(1)

                        with autocast(enabled=self.options.use_half):
                            features_map = self.regressor.get_features(image_B1HW)
                        grid = grid.to(features_map.dtype)

                        sampled_features = F.grid_sample(features_map, grid, align_corners=True, mode='bilinear')
                        features_NC = sampled_features.squeeze(2).permute(0, 2, 1).reshape(-1,
                                                                                           self.regressor.feature_dim)

                        num_pts = features_NC.shape[0]
                        available = chunk_size - buffer_idx
                        to_add = min(num_pts, available)

                        if to_add > 0:
                            end = buffer_idx + to_add
                            self.training_buffer['features'][buffer_idx:end] = features_NC[:to_add]
                            self.training_buffer['target_px'][buffer_idx:end] = final_coords[:to_add]
                            self.training_buffer['gt_poses_inv'][buffer_idx:end] = gt_pose_inv_B44[:, :3].expand(
                                to_add, 3, 4)
                            self.training_buffer['intrinsics'][buffer_idx:end] = intrinsics_B33.expand(to_add, 3, 3)
                            self.training_buffer['intrinsics_inv'][buffer_idx:end] = intrinsics_inv_B33.expand(
                                to_add, 3, 3)
                            buffer_idx = end
                            pbar.update(to_add)

                        if buffer_idx >= chunk_size: break

        self.regressor.train()

    def train(self):
        self.training_start = time.time()

        # 基础 Chunk 大小
        chunk_size = getattr(self.options, 'buffer_chunk_size')

        # 1. 阶段一：基础 Buffer (Standard/Hybrid)
        std_total_size = self.options.training_buffer_size
        num_std_chunks = math.ceil(std_total_size / chunk_size)

        # 2. 阶段二：主动学习 Buffer (Active)
        # 默认 Active Buffer 大小等于 1 个 Chunk (8M)，或者可以自定义
        active_buffer_size = getattr(self.options, 'active_buffer_size', chunk_size)
        num_active_chunks = math.ceil(active_buffer_size / chunk_size)

        total_chunks_log = num_std_chunks + num_active_chunks

        _logger.info(f"Training Strategy:")
        _logger.info(f"  Phase 1 (Hybrid): {std_total_size} samples ({num_std_chunks} chunks)")
        _logger.info(f"  Phase 2 (Active): {active_buffer_size} samples ({num_active_chunks} chunks)")

        chunk_counter = 0

        # --- Phase 1 Loop ---
        for i in range(num_std_chunks):
            chunk_counter += 1
            _logger.info(f"=== Chunk {chunk_counter}/{total_chunks_log} [Hybrid] ===")
            self.create_training_buffer(chunk_size, mode='hybrid')
            for self.epoch in range(self.options.epochs):
                self.run_epoch(chunk_size)

        # --- Phase 2 Loop ---
        for i in range(num_active_chunks):
            chunk_counter += 1
            _logger.info(f"=== Chunk {chunk_counter}/{total_chunks_log} [Active] ===")
            self.create_training_buffer(chunk_size, mode='active')
            for self.epoch in range(self.options.epochs):
                self.run_epoch(chunk_size)

        self.save_model()

    def get_depth_batch(self):
        if self.depth_iterator is None:
            batch_sampler = sampler.BatchSampler(sampler.RandomSampler(self.dataset, generator=self.batch_generator), batch_size=self.options.depth_batch_size, drop_last=False)
            depth_loader = DataLoader(self.dataset, sampler=batch_sampler, batch_size=None, num_workers=self.num_data_loader_workers)
            self.depth_iterator = iter(depth_loader)
        try: return next(self.depth_iterator)
        except StopIteration:
            batch_sampler = sampler.BatchSampler(sampler.RandomSampler(self.dataset, generator=self.batch_generator), batch_size=self.options.depth_batch_size, drop_last=False)
            depth_loader = DataLoader(self.dataset, sampler=batch_sampler, batch_size=None, num_workers=self.num_data_loader_workers)
            self.depth_iterator = iter(depth_loader)
            return next(self.depth_iterator)

    def run_epoch(self, current_buffer_size):
        torch.backends.cudnn.benchmark = True
        random_indices = torch.randperm(current_buffer_size, generator=self.training_generator)
        for batch_start in range(0, current_buffer_size, self.options.batch_size):
            batch_end = batch_start + self.options.batch_size
            if batch_end > current_buffer_size: continue
            indices = random_indices[batch_start:batch_end]
            ace_batch = {
                'features': self.training_buffer['features'][indices],
                'target_px': self.training_buffer['target_px'][indices],
                'gt_poses_inv': self.training_buffer['gt_poses_inv'][indices],
                'intrinsics': self.training_buffer['intrinsics'][indices],
                'intrinsics_inv': self.training_buffer['intrinsics_inv'][indices]
            }
            depth_batch = None
            if self.iteration > 500: depth_batch = self.get_depth_batch()
            self.training_step(ace_batch, depth_batch)
            self.iteration += 1

    def training_step(self, ace_data, depth_data_batch):
        self.optimizer.zero_grad(set_to_none=True)
        features_bC = ace_data['features']
        target_px_b2 = ace_data['target_px']
        gt_inv_poses_b34 = ace_data['gt_poses_inv']
        Ks_b33 = ace_data['intrinsics']
        invKs_b33 = ace_data['intrinsics_inv']
        batch_size = features_bC.shape[0]
        channels = features_bC.shape[1]
        features_bCHW = features_bC[None, None, ...].view(-1, 16, 32, channels).permute(0, 3, 1, 2)
        with autocast(enabled=self.options.use_half):
            pred_scene_coords_b3HW = self.regressor.get_scene_coordinates(features_bCHW)
        pred_scene_coords_b31 = pred_scene_coords_b3HW.permute(0, 2, 3, 1).flatten(0, 2).unsqueeze(-1).float()
        pred_scene_coords_b41 = to_homogeneous(pred_scene_coords_b31)
        pred_cam_coords_b31 = torch.bmm(gt_inv_poses_b34, pred_scene_coords_b41)
        pred_px_b31 = torch.bmm(Ks_b33, pred_cam_coords_b31)
        pred_px_b31[:, 2].clamp_(min=self.options.depth_min)
        pred_px_b21 = pred_px_b31[:, :2] / pred_px_b31[:, 2, None]
        reprojection_error_b2 = pred_px_b21.squeeze() - target_px_b2
        reprojection_error_b1 = torch.norm(reprojection_error_b2, dim=1, keepdim=True, p=1)
        invalid_min_depth_b1 = pred_cam_coords_b31[:, 2] < self.options.depth_min
        invalid_repro_b1 = reprojection_error_b1 > self.options.repro_loss_hard_clamp
        invalid_max_depth_b1 = pred_cam_coords_b31[:, 2] > self.options.depth_max
        valid_mask_b1 = ~(invalid_min_depth_b1 | invalid_repro_b1 | invalid_max_depth_b1)
        invalid_mask_b1 = ~valid_mask_b1
        loss_ace = self.repro_loss.compute(reprojection_error_b1[valid_mask_b1], self.iteration)
        pixel_grid_crop_b31 = to_homogeneous(target_px_b2.unsqueeze(2))
        target_camera_coords_b31 = self.options.depth_target * torch.bmm(invKs_b33, pixel_grid_crop_b31)
        loss_invalid = torch.abs(target_camera_coords_b31 - pred_cam_coords_b31).masked_select(invalid_mask_b1.unsqueeze(2)).sum()
        total_loss = (loss_ace + loss_invalid) / features_bC.shape[0]
        loss_depth_val = 0.0
        if depth_data_batch is not None:
            loss_depth = self.compute_depth_loss(depth_data_batch)
            weight = (self.iteration / (self.iterations * (1 / self.options.DepthLoss_weight)))
            weight = min(weight, 1.0)
            total_loss += weight * loss_depth
            loss_depth_val = loss_depth.item()
        self.scaler.scale(total_loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.scheduler.step()
        if self.iteration % self.iterations_output == 0:
            fraction_valid = float(valid_mask_b1.sum() / batch_size) * 100
            current_lr = self.optimizer.param_groups[0]['lr']
            gpu_mem = torch.cuda.max_memory_allocated() / (1024 ** 3)
            ace_val = loss_ace.item() / batch_size
            inv_val = loss_invalid.item() / batch_size
            _logger.info(f"Iter: {self.iteration:6d} | Total: {total_loss.item():.4f} | ACE: {ace_val:.4f} | Inv: {inv_val:.4f} | Depth: {loss_depth_val:.4f} | Valid: {fraction_valid:.1f}% | LR: {current_lr:.1e} | Mem: {gpu_mem:.1f}G")

    def compute_depth_loss(self, batch_data):
        image_RGB, image_B1HW, image_mask, gt_pose, gt_pose_inv, intrinsics, _, _, _ = batch_data
        image_B1HW = image_B1HW.to(self.device, non_blocking=True)
        gt_pose_inv = gt_pose_inv.to(self.device, non_blocking=True)
        mask_gpu = image_mask.to(self.device, non_blocking=True)

        if image_RGB.max() <= 1.05:
            image_RGB_np = (image_RGB * 255).clamp(0, 255).permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
        else:
            image_RGB_np = image_RGB.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)

        Batch_Size = image_RGB_np.shape[0]
        depth_t_list = []
        with torch.no_grad():
            with torch.cuda.device(self.device):
                for i in range(Batch_Size):
                    img_single = image_RGB_np[i]
                    depth_i = self.depth_anything.infer_image(
                        cv2.cvtColor(img_single, cv2.COLOR_RGB2BGR),
                        self.options.input_size
                    )
                    if isinstance(depth_i, np.ndarray):
                        depth_i = torch.from_numpy(depth_i).to(self.device)
                    else:
                        depth_i = depth_i.to(self.device)
                    depth_t_list.append(depth_i)

            depth_t = torch.stack(depth_t_list, dim=0).unsqueeze(1)
            depth_t_median = depth_t.view(Batch_Size, -1).median(dim=1, keepdim=True).values.unsqueeze(2).unsqueeze(3)
            depth_t_abs = (depth_t - depth_t_median).abs()
            depth_t_mean = depth_t_abs.view(Batch_Size, -1).mean(dim=1, keepdim=True).unsqueeze(2).unsqueeze(3)
            depth_t = (depth_t - depth_t_median) / (depth_t_mean + 1e-5)

            with autocast(enabled=self.options.use_half):
                features = self.regressor.get_features(image_B1HW)
                scene_coords = self.regressor.get_scene_coordinates(features)

            scene_coords = scene_coords.float()
            if torch.isnan(scene_coords).any() or torch.isinf(scene_coords).any():
                return torch.tensor(0.0, device=self.device, requires_grad=True)

            B, C, H, W = scene_coords.shape
            scene_coords_flat = scene_coords.permute(0, 2, 3, 1).flatten(0, 2).unsqueeze(-1)
            scene_coords_homo = to_homogeneous(scene_coords_flat)
            gt_pose_inv_expanded = gt_pose_inv[:, :3].float().unsqueeze(1).expand(B, H * W, 3, 4).reshape(-1, 3, 4)
            cam_coords = torch.bmm(gt_pose_inv_expanded, scene_coords_homo)
            pred_depth = cam_coords[:, 2].view(B, H, W)

        pred_depth = torch.clamp(pred_depth, min=0.1, max=1000.0)
        s_median = pred_depth.view(B, -1).median(dim=1, keepdim=True).values.unsqueeze(2)
        s_abs = (pred_depth - s_median).abs()
        s_mean = s_abs.view(B, -1).mean(dim=1, keepdim=True).unsqueeze(2)

        if torch.any(s_mean < 1e-6):
            pred_depth_norm = torch.zeros_like(pred_depth)
        else:
            pred_depth_norm = 1 - (pred_depth - s_median) / (s_mean + 1e-5)

        depth_teacher_resized = F.interpolate(depth_t, size=(H, W), mode='bilinear', align_corners=True)
        mask_resized = F.interpolate(mask_gpu.float(), size=(H, W), mode='nearest')
        image_resized = F.interpolate(image_RGB.to(self.device).float(), size=(H, W), mode='bilinear', align_corners=True)

        depth_teacher_resized = depth_teacher_resized.squeeze(1)
        mask_resized = mask_resized.squeeze(1).bool()

        if torch.isnan(pred_depth_norm).any():
            return torch.tensor(0.0, device=self.device, requires_grad=True)

        return self.depth_loss_func(pred_depth_norm, depth_teacher_resized, mask_resized, image_resized)

    def save_model(self):
        model_state_dict = {}
        model_state_dict["heads"] = {k: v.half() for k, v in self.regressor.heads.state_dict().items()}
        base_dir = os.path.dirname(self.options.output_map_file)
        base_name = os.path.basename(self.options.output_map_file)
        name_root, ext = os.path.splitext(base_name)
        if not ext: ext = ".pt"
        scene_name = os.path.basename(os.path.normpath(str(self.options.scene)))
        timestamp = datetime.now().strftime("%m%d_%H%M")
        new_filename = f"{name_root}_{scene_name}_{self.options.depth_encoder}_ep{self.options.epochs}_{timestamp}{ext}"
        save_path = os.path.join(base_dir, new_filename)
        if base_dir and not os.path.exists(base_dir): os.makedirs(base_dir)
        torch.save(model_state_dict, save_path)
        _logger.info(f"Saved trained weights intelligently to: {save_path}")