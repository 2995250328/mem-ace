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

# 引入你的项目模块
from ace_util import get_pixel_grid, to_homogeneous
from ace_loss import ReproLoss
from ace_network_depth import Regressor, RelativeDepthLoss
from dataset import CamLocDataset
from superpoint import SuperPointFrontend
from depth_anything_v2.dpt import DepthAnythingV2
import ace_vis_util as vutil

_logger = logging.getLogger(__name__)

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


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
        self.superpoint = SuperPointFrontend(
            self.regressor.superpoint,
            weights_path=self.options.weights_path,
            nms_dist=self.options.nms_dist,
            conf_thresh=self.options.conf_thresh,
            nn_thresh=self.options.nn_thresh,
            cuda=True
        )

        # Depth Anything V2
        model_configs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
            'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
        }
        self.depth_anything = DepthAnythingV2(**model_configs[self.options.depth_encoder])
        self.depth_anything.load_state_dict(
            torch.load(f'/mnt/storage/xwh/checkpoints/depth_anything_v2_{self.options.depth_encoder}.pth', map_location='cpu'))
        self.depth_anything = self.depth_anything.to(self.device).eval()

        # Optimizer & Scheduler
        self.optimizer = optim.AdamW(self.regressor.parameters(), lr=self.options.learning_rate_min)
        steps_per_epoch = self.options.onebuffer // self.options.batch_size
        self.scheduler = optim.lr_scheduler.OneCycleLR(self.optimizer,
                                                       max_lr=self.options.learning_rate_max,
                                                       epochs=self.options.epochs,
                                                       steps_per_epoch=steps_per_epoch,
                                                       cycle_momentum=False)
        self.scaler = GradScaler(enabled=self.options.use_half)

        # Misc
        self.pixel_grid_2HW = get_pixel_grid(self.regressor.OUTPUT_SUBSAMPLE).to(self.device)
        self.iterations = self.options.epochs * self.options.training_buffer_size // self.options.batch_size
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
        self.ace_visualizer = None

    def expand_neighbors_gpu(self, coords, H, W, patch_size, include_diagonal=8):
        """
        在 GPU 上进行邻域扩展和去重。
        coords: (N, 2) tensor, [x, y]
        Returns: (M, 2) 扩展后的坐标
        """
        if coords.shape[0] == 0:
            return coords

        # 定义偏移量
        offsets = [[0, 0]]
        if include_diagonal >= 4:
            offsets.extend([[-1, 0], [1, 0], [0, -1], [0, 1]])  # 上下左右
        if include_diagonal == 8:
            offsets.extend([[-1, -1], [-1, 1], [1, -1], [1, 1]])  # 对角线

        offsets = torch.tensor(offsets, device=coords.device, dtype=coords.dtype) * patch_size

        # (N, 1, 2) + (K, 2) -> (N, K, 2) -> (N*K, 2)
        expanded = (coords.unsqueeze(1) + offsets).reshape(-1, 2)

        # 边界检查
        valid_mask = (expanded[:, 0] >= 0) & (expanded[:, 0] < W) & \
                     (expanded[:, 1] >= 0) & (expanded[:, 1] < H)
        valid_coords = expanded[valid_mask]

        # 去重 (如果不关心哪个 SuperPoint 产生的，直接 unique 坐标即可)
        unique_coords = torch.unique(valid_coords, dim=0)
        return unique_coords

    def create_training_buffer(self):
        """构建训练 Buffer，使用混合采样策略：70% SuperPoint + 30% Random"""
        _logger.info("Starting creation of the training buffer (Hybrid Sampling Mode).")
        torch.backends.cudnn.benchmark = False
        self.regressor.eval()

        # Initialize Buffer
        self.training_buffer = {
            'features': torch.empty((self.options.onebuffer, self.regressor.feature_dim),
                                    dtype=(torch.float32, torch.float16)[self.options.use_half], device=self.device),
            'target_px': torch.empty((self.options.onebuffer, 2), dtype=torch.float32, device=self.device),
            'gt_poses_inv': torch.empty((self.options.onebuffer, 3, 4), dtype=torch.float32, device=self.device),
            'intrinsics': torch.empty((self.options.onebuffer, 3, 3), dtype=torch.float32, device=self.device),
            'intrinsics_inv': torch.empty((self.options.onebuffer, 3, 3), dtype=torch.float32, device=self.device)
        }

        batch_sampler = sampler.BatchSampler(sampler.RandomSampler(self.dataset, generator=self.batch_generator),
                                             batch_size=1, drop_last=False)
        loader = DataLoader(self.dataset, sampler=batch_sampler, batch_size=None,
                            num_workers=self.num_data_loader_workers)

        buffer_idx = 0
        patch_size = self.regressor.OUTPUT_SUBSAMPLE  # 通常是 8

        # --- [配置] 混合采样比例 ---
        SP_RATIO = 1  # SuperPoint 占比
        # -------------------------

        with torch.no_grad():
            while buffer_idx < self.options.onebuffer:
                for _, image_B1HW, image_mask_B1HW, _, gt_pose_inv_B44, intrinsics_B33, intrinsics_inv_B33, _, _ in loader:

                    # 1. 准备数据
                    image_B1HW = image_B1HW.to(self.device, non_blocking=True)
                    image_mask_B1HW = image_mask_B1HW.to(self.device, non_blocking=True)
                    B, _, H, W = image_B1HW.shape

                    # 计算本张图像的目标采样总数
                    img_rate = min(H, W) / 480.0
                    total_target_samples = int(img_rate * self.options.samples_per_image)

                    # 计算 SuperPoint 的配额
                    sp_target_count = int(total_target_samples * SP_RATIO)

                    # 2. SuperPoint 提取
                    image_HW_np = image_B1HW.squeeze().cpu().numpy().astype(np.float32)
                    pts, _, _ = self.superpoint.run(image_HW_np)  # (3, N)

                    sp_coords_expanded = torch.zeros((0, 2), device=self.device)

                    if pts is not None and pts.shape[1] > 0:
                        # 转 Tensor 并对齐到 Patch
                        sp_coords = torch.from_numpy(pts[:2].T).to(self.device, dtype=torch.float32)
                        sp_coords = (sp_coords // patch_size) * patch_size + (patch_size // 2)

                        # 邻域扩展
                        sp_coords_expanded = self.expand_neighbors_gpu(sp_coords, H, W, patch_size, include_diagonal=4)

                        # --- [关键修改]：SuperPoint 截断逻辑 ---
                        # 如果 SP 点太多，强制随机截断，给 Random 采样留出空间
                        if sp_coords_expanded.shape[0] > sp_target_count:
                            perm = torch.randperm(sp_coords_expanded.shape[0], device=self.device)
                            sp_coords_expanded = sp_coords_expanded[perm[:sp_target_count]]

                    # 3. 随机采样补充 (Random Sampling)
                    # 我们需要填满 total_target_samples。
                    # 如果 SP 只有 10 个，我们就随机采 (Total - 10) 个。
                    # 如果 SP 占满了 70%，我们就随机采 30%。
                    current_sp_count = sp_coords_expanded.shape[0]
                    needed_random = total_target_samples - current_sp_count

                    final_coords = sp_coords_expanded

                    if needed_random > 0:
                        # Resize mask logic
                        mask_bool = image_mask_B1HW.squeeze() > 0
                        valid_indices = torch.nonzero(mask_bool)  # (M, 2) [y, x]

                        if valid_indices.shape[0] > 0:
                            # 随机选 needed 个点
                            # 注意：如果 valid 区域很小，可能不够采，允许重复采样(replacement=True)或者取min
                            sample_count = min(valid_indices.shape[0], needed_random)
                            # 这里建议 replacement=True 以保证数量，或者像下面这样简单处理
                            idx = torch.randint(0, valid_indices.shape[0], (needed_random,), device=self.device)

                            supplement_yx = valid_indices[idx]
                            supplement_xy = torch.stack((supplement_yx[:, 1], supplement_yx[:, 0]), dim=1).float()

                            # 对齐到 patch 中心
                            supplement_xy = (supplement_xy // patch_size) * patch_size + (patch_size // 2)

                            # 合并
                            final_coords = torch.cat((final_coords, supplement_xy), dim=0)

                    # 4. 提取特征 (Grid Sample)
                    if final_coords.shape[0] == 0:
                        continue

                    # 归一化坐标 [-1, 1]
                    norm_x = 2.0 * final_coords[:, 0] / (W - 1) - 1.0
                    norm_y = 2.0 * final_coords[:, 1] / (H - 1) - 1.0
                    grid = torch.stack((norm_x, norm_y), dim=1).unsqueeze(0).unsqueeze(1)  # (1, 1, N, 2)

                    with autocast(enabled=self.options.use_half):
                        features_map = self.regressor.get_features(image_B1HW)
                    grid = grid.to(features_map.dtype)

                    sampled_features = F.grid_sample(features_map, grid, align_corners=True, mode='bilinear')
                    features_NC = sampled_features.squeeze(2).permute(0, 2, 1).reshape(-1, self.regressor.feature_dim)

                    # 5. 填入 Buffer
                    num_pts = features_NC.shape[0]
                    available = self.options.onebuffer - buffer_idx
                    to_add = min(num_pts, available)

                    if to_add > 0:
                        end = buffer_idx + to_add
                        self.training_buffer['features'][buffer_idx:end] = features_NC[:to_add]
                        self.training_buffer['target_px'][buffer_idx:end] = final_coords[:to_add]
                        self.training_buffer['gt_poses_inv'][buffer_idx:end] = gt_pose_inv_B44[:, :3].expand(to_add, 3,
                                                                                                             4)
                        self.training_buffer['intrinsics'][buffer_idx:end] = intrinsics_B33.expand(to_add, 3, 3)
                        self.training_buffer['intrinsics_inv'][buffer_idx:end] = intrinsics_inv_B33.expand(to_add, 3, 3)

                        buffer_idx = end
                        print(f'\rBuffer Fill: {buffer_idx}/{self.options.onebuffer}', end="")

                    if buffer_idx >= self.options.onebuffer:
                        break
        print()
        self.regressor.train()

    def train(self):
        """主训练循环"""
        self.training_start = time.time()
        num_fills = self.options.training_buffer_size // self.options.onebuffer

        for i in range(num_fills):
            _logger.info(f"Refilling buffer pass {i + 1}/{num_fills}")
            self.create_training_buffer()

            for self.epoch in range(self.options.epochs):
                self.run_epoch()

        self.save_model()

    def get_depth_batch(self):
        # 如果迭代器尚未初始化
        if self.depth_iterator is None:
            batch_sampler = sampler.BatchSampler(
                sampler.RandomSampler(self.dataset, generator=self.batch_generator),
                batch_size=self.options.depth_batch_size,  # <--- 补全参数
                drop_last=False  # <--- 补全参数
            )
            depth_loader = DataLoader(self.dataset, sampler=batch_sampler, batch_size=None,
                                      num_workers=self.num_data_loader_workers)
            self.depth_iterator = iter(depth_loader)

        try:
            return next(self.depth_iterator)
        except StopIteration:
            # 迭代器耗尽，重新创建
            batch_sampler = sampler.BatchSampler(
                sampler.RandomSampler(self.dataset, generator=self.batch_generator),
                batch_size=self.options.depth_batch_size,  # <--- 补全参数
                drop_last=False  # <--- 补全参数
            )
            depth_loader = DataLoader(self.dataset, sampler=batch_sampler, batch_size=None,
                                      num_workers=self.num_data_loader_workers)
            self.depth_iterator = iter(depth_loader)
            return next(self.depth_iterator)

    def run_epoch(self):
        """执行一个 Epoch"""
        torch.backends.cudnn.benchmark = True

        # 1. Shuffle Buffer
        random_indices = torch.randperm(self.options.onebuffer, generator=self.training_generator)

        # 3. 遍历 Buffer 进行 ACE 训练
        for batch_start in range(0, self.options.onebuffer, self.options.batch_size):
            batch_end = batch_start + self.options.batch_size
            if batch_end > self.options.onebuffer: continue

            # 取 ACE 数据
            indices = random_indices[batch_start:batch_end]
            ace_batch = {
                'features': self.training_buffer['features'][indices],
                'target_px': self.training_buffer['target_px'][indices],
                'gt_poses_inv': self.training_buffer['gt_poses_inv'][indices],
                'intrinsics': self.training_buffer['intrinsics'][indices],
                'intrinsics_inv': self.training_buffer['intrinsics_inv'][indices]
            }

            # 取 Depth 数据 (Warmup 之后)
            depth_batch = None
            if self.iteration > (0.5 * self.iterations):
                depth_batch = self.get_depth_batch()  # 使用辅助函数

            self.training_step(ace_batch, depth_batch)
            self.iteration += 1

    def training_step(self, ace_data, depth_data_batch):
        """
        Combined training step: ACE Reprojection Loss + Relative Depth Loss
        """
        self.optimizer.zero_grad(set_to_none=True)

        # --- Part A: ACE Reprojection Loss (Standard) ---
        features_bC = ace_data['features']
        target_px_b2 = ace_data['target_px']
        gt_inv_poses_b34 = ace_data['gt_poses_inv']
        Ks_b33 = ace_data['intrinsics']
        invKs_b33 = ace_data['intrinsics_inv']

        batch_size = features_bC.shape[0]

        # Fake BCHW for Regressor Head
        channels = features_bC.shape[1]
        features_bCHW = features_bC[None, None, ...].view(-1, 16, 32, channels).permute(0, 3, 1, 2)

        with autocast(enabled=self.options.use_half):
            pred_scene_coords_b3HW = self.regressor.get_scene_coordinates(features_bCHW)

        pred_scene_coords_b31 = pred_scene_coords_b3HW.permute(0, 2, 3, 1).flatten(0, 2).unsqueeze(-1).float()
        pred_scene_coords_b41 = to_homogeneous(pred_scene_coords_b31)
        pred_cam_coords_b31 = torch.bmm(gt_inv_poses_b34, pred_scene_coords_b41)
        pred_px_b31 = torch.bmm(Ks_b33, pred_cam_coords_b31)
        pred_px_b31[:, 2].clamp_(min=self.options.depth_min)  # Avoid div zero
        pred_px_b21 = pred_px_b31[:, :2] / pred_px_b31[:, 2, None]

        # Reprojection Error
        reprojection_error_b2 = pred_px_b21.squeeze() - target_px_b2
        reprojection_error_b1 = torch.norm(reprojection_error_b2, dim=1, keepdim=True, p=1)

        # Masks
        invalid_min_depth_b1 = pred_cam_coords_b31[:, 2] < self.options.depth_min
        invalid_repro_b1 = reprojection_error_b1 > self.options.repro_loss_hard_clamp
        invalid_max_depth_b1 = pred_cam_coords_b31[:, 2] > self.options.depth_max
        valid_mask_b1 = ~(invalid_min_depth_b1 | invalid_repro_b1 | invalid_max_depth_b1)
        invalid_mask_b1 = ~valid_mask_b1

        loss_ace = self.repro_loss.compute(reprojection_error_b1[valid_mask_b1], self.iteration)

        # Invalid points proxy loss
        pixel_grid_crop_b31 = to_homogeneous(target_px_b2.unsqueeze(2))
        target_camera_coords_b31 = self.options.depth_target * torch.bmm(invKs_b33, pixel_grid_crop_b31)
        loss_invalid = torch.abs(target_camera_coords_b31 - pred_cam_coords_b31).masked_select(
            invalid_mask_b1.unsqueeze(2)).sum()

        total_loss = (loss_ace + loss_invalid) / features_bC.shape[0]

        # --- Part B: Depth Anything Loss (Optional) ---
        loss_depth_val = 0.0
        if depth_data_batch is not None:
            loss_depth = self.compute_depth_loss(depth_data_batch)
            # Weighted addition
            weight = (self.iteration / (self.iterations * (1 / self.options.DepthLoss_weight)))
            total_loss += weight * loss_depth
            loss_depth_val = loss_depth.item()

        # Backprop
        self.scaler.scale(total_loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.scheduler.step()

        # --- Enhanced Debug Logging ---
        if self.iteration % self.iterations_output == 0:
            # 计算有效点比例
            fraction_valid = float(valid_mask_b1.sum() / batch_size) * 100
            # 获取当前学习率
            current_lr = self.optimizer.param_groups[0]['lr']
            # 获取显存占用 (GB)
            gpu_mem = torch.cuda.max_memory_allocated() / (1024 ** 3)

            # 计算归一化后的各部分 loss 数值用于显示
            ace_val = loss_ace.item() / batch_size
            inv_val = loss_invalid.item() / batch_size

            _logger.info(
                f"Iter: {self.iteration:6d} | "
                f"Total: {total_loss.item():.4f} | "
                f"ACE: {ace_val:.4f} | "
                f"Inv: {inv_val:.4f} | "
                f"Depth: {loss_depth_val:.4f} | "
                f"Valid: {fraction_valid:.1f}% | "
                f"LR: {current_lr:.1e} | "
                f"Mem: {gpu_mem:.1f}G"
            )

    def compute_depth_loss(self, batch_data):
        # 解包数据
        image_RGB, image_B1HW, image_mask, gt_pose, gt_pose_inv, intrinsics, _, _, _ = batch_data

        image_B1HW = image_B1HW.to(self.device, non_blocking=True)
        gt_pose_inv = gt_pose_inv.to(self.device, non_blocking=True)
        mask_gpu = image_mask.to(self.device, non_blocking=True)
        # 1. Student 输入准备：
        # 如果 dataset 返回的是 [0,1] 的 float，这里就不需要再除以 255.0 了，否则会变成 [0, 0.0039]
        # 建议检查 dataset。通常 CamLocDataset 输出已经是 Normalized 的。
        # 假设 image_RGB 是 [0, 1]，则直接：
        image_rgb_gpu = image_RGB.to(self.device, non_blocking=True).float()
        # 如果 image_RGB 是 [0, 255]，则保留你原来的 / 255.0

        # 2. Teacher 输入准备 [修复部分]
        # 必须先反归一化到 [0, 255] 再转 uint8
        if image_RGB.max() <= 1.05: # 简单的自动判断
             image_RGB_np = (image_RGB * 255).clamp(0, 255).permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
        else:
             image_RGB_np = image_RGB.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)

        with torch.no_grad():
            # 强制将默认 cuda 设备设为 self.device (cuda:2)
            # 这样 infer_image 内部生成的 tensor 就会在 cuda:2 上
            with torch.cuda.device(self.device):
                depth_t = self.depth_anything.infer_image(
                    cv2.cvtColor(image_RGB_np[0], cv2.COLOR_RGB2BGR),
                    self.options.input_size
                )
            if isinstance(depth_t, np.ndarray):
                depth_t = torch.from_numpy(depth_t).to(self.device).unsqueeze(0).unsqueeze(0)
            else:
                # 如果修改生效导致直接返回了 Tensor
                depth_t = depth_t.to(self.device).unsqueeze(0).unsqueeze(0)
            depth_t = (depth_t - depth_t.median()) / (depth_t.abs().mean() + 1e-5)

        # 3. Student 推理 (ACE)
            # 即使开启 autocast，也要确保最后回归坐标时回到 float32
            with autocast(enabled=self.options.use_half):
                features = self.regressor.get_features(image_B1HW)
                # 这里的输出可能是 float16，如果数值很大就会溢出
                scene_coords = self.regressor.get_scene_coordinates(features)

            # [修复 1]：立即强制转为 float32，防止后续几何计算溢出
            scene_coords = scene_coords.float()
            # 4. 投影 ACE 坐标到相机深度
            B, C, H, W = scene_coords.shape
            scene_coords_flat = scene_coords.permute(0, 2, 3, 1).flatten(0, 2).unsqueeze(-1)
            scene_coords_homo = to_homogeneous(scene_coords_flat)

            # [修复 2]：确保 Pose 矩阵也是 float32
            gt_pose_inv_expanded = gt_pose_inv[:, :3].float().unsqueeze(1).expand(B, H * W, 3, 4).reshape(-1, 3, 4)
            # 矩阵乘法必须在 float32 下进行
            cam_coords = torch.bmm(gt_pose_inv_expanded, scene_coords_homo)
            pred_depth = cam_coords[:, 2].view(B, H, W)

        # [修复 3]：防止预测出负深度或无限远，导致数学计算错误
        pred_depth = torch.clamp(pred_depth, min=0.1, max=1000.0)
        # Normalize Student
        depth_median = torch.median(pred_depth)
        depth_diff_mean = torch.mean(torch.abs(pred_depth - depth_median))
        # 更加安全的归一化
        pred_depth_norm = 1 - (pred_depth - depth_median) / (depth_diff_mean + 1e-5)

        # 5. 对齐尺寸 (使用已经移到 GPU 的 mask_gpu 和 image_rgb_gpu)
        # image_rgb_gpu 现在是 Float 类型，interpolate 不会再报错
        depth_teacher_resized = F.interpolate(depth_t, size=(H, W), mode='bilinear', align_corners=True)
        mask_resized = F.interpolate(mask_gpu.float(), size=(H, W), mode='nearest').bool()
        image_resized = F.interpolate(image_rgb_gpu, size=(H, W), mode='bilinear', align_corners=True)

        # 6. 压缩维度
        pred_depth_norm = pred_depth_norm.squeeze()
        depth_teacher_resized = depth_teacher_resized.squeeze()
        mask_resized = mask_resized.squeeze()
        image_resized = image_resized.squeeze()

        # 计算 Loss

        loss = self.depth_loss_func(pred_depth_norm, depth_teacher_resized, mask_resized, image_resized)

        if torch.isnan(loss):
            print("Nan loss detected")
            return torch.tensor(0.0, device=self.device)

        return loss

    def save_model(self):
        # 准备模型状态字典 (fp16)
        model_state_dict = {}
        model_state_dict["superpoint"] = {k: v.half() for k, v in self.regressor.superpoint.state_dict().items()}
        model_state_dict["heads"] = {k: v.half() for k, v in self.regressor.heads.state_dict().items()}

        # --- 智能命名逻辑 ---
        # 1. 获取基础路径信息
        base_dir = os.path.dirname(self.options.output_map_depth)
        base_name = os.path.basename(self.options.output_map_depth)
        name_root, ext = os.path.splitext(base_name)
        if not ext: ext = ".pt"  # 默认后缀

        # 2. 提取场景名 (假设 input path 是 datasets/scene_name)
        # 使用 os.path.normpath 去除末尾可能存在的 /
        scene_name = os.path.basename(os.path.normpath(str(self.options.scene)))

        # 3. 生成时间戳
        timestamp = datetime.now().strftime("%m%d_%H%M")

        # 4. 组合新文件名: 原名_场景_Encoder_Epochs_时间.pt
        # 例如: test_depth_stairs_vitb_ep16_1209_2046.pt
        new_filename = (
            f"{name_root}_"
            f"{scene_name}_"
            f"{self.options.depth_encoder}_"
            f"ep{self.options.epochs}_"
            f"{timestamp}"
            f"{ext}"
        )

        save_path = os.path.join(base_dir, new_filename)

        # 确保目录存在
        if base_dir and not os.path.exists(base_dir):
            os.makedirs(base_dir)

        torch.save(model_state_dict, save_path)
        _logger.info(f"Saved trained weights intelligently to: {save_path}")

        # (可选) 同时保存一份原始指定路径的副本，方便脚本后续调用
        # torch.save(model_state_dict, self.options.output_map_depth)
        # _logger.info(f"Also saved to original path: {self.options.output_map_depth}")