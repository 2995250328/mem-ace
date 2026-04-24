"""ACE-style trainers that fuse image features with precomputed point features."""

from __future__ import annotations

import logging
import random
import sys
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm
import torch
from torch.amp import autocast
from torch.utils.data import DataLoader, sampler
import torchvision.transforms.functional as TF

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ace_fcn_lmc"))

from ace_util import to_homogeneous  # noqa: E402
from trainer_ace_fcn import TrainerACEFCN  # noqa: E402

from .ace_network_pointcloud import RegressorACEPointCloudFusion, RegressorDINOPointCloudFusion
from .utonia_encoder import load_feature_bank

_logger = logging.getLogger(__name__)


class TrainerACEPointCloudFusion(TrainerACEFCN):
    """ACE FCN trainer with raw-image-feature buffer and point-bank fusion."""

    def __init__(self, options):
        point_feature_path = getattr(options, "point_feature_path", None)
        if point_feature_path is None:
            raise ValueError("--point_feature_path is required for point-cloud fusion training.")
        self._point_bank_cpu = load_feature_bank(point_feature_path, device="cpu")
        self._point_feature_dim = int(self._point_bank_cpu["features"].shape[-1])
        super().__init__(options)
        self.point_bank = load_feature_bank(point_feature_path, device=self.device)
        _logger.info(
            "[PointFusion] Loaded point bank: points=%d feature_dim=%d source=%s",
            int(self.point_bank["points"].shape[0]),
            self._point_feature_dim,
            point_feature_path,
        )

    def _create_regressor(self):
        regressor = RegressorACEPointCloudFusion.create_from_encoder(
            encoder_path=self.options.encoder_path,
            mean=self.dataset.mean_cam_center,
            num_head_blocks=self.options.num_head_blocks,
            use_homogeneous=self.options.use_homogeneous,
            point_feature_dim=self._point_feature_dim,
            num_encoder_features=512,
            freeze_backbone=self.options.freeze_backbone,
            adaptor_hidden_dim=getattr(self.options, "adaptor_hidden_dim", 512),
            adaptor_heads=getattr(self.options, "adaptor_heads", 8),
            adaptor_dropout=getattr(self.options, "adaptor_dropout", 0.1),
            max_context_points=getattr(self.options, "max_context_points", 4096),
        )
        _logger.info("Loaded ACE FCN encoder from: %s", self.options.encoder_path)
        return regressor

    @staticmethod
    def _flatten_bchw(tensor_bchw: torch.Tensor) -> torch.Tensor:
        return tensor_bchw.permute(0, 2, 3, 1).reshape(-1, tensor_bchw.shape[1])

    @staticmethod
    def _make_world_rays(
        pixel_positions_b2hw: torch.Tensor,
        pose_b44: torch.Tensor,
        intrinsics_inv_b33: torch.Tensor,
    ):
        bsz, _, height, width = pixel_positions_b2hw.shape
        px_bhw2 = pixel_positions_b2hw.permute(0, 2, 3, 1).reshape(bsz, height * width, 2)
        ones = torch.ones((bsz, height * width, 1), device=px_bhw2.device, dtype=px_bhw2.dtype)
        px_h = torch.cat([px_bhw2, ones], dim=-1)

        ray_cam = torch.matmul(px_h, intrinsics_inv_b33.transpose(1, 2))
        rot_c2w = pose_b44[:, :3, :3]
        ray_world = torch.matmul(ray_cam, rot_c2w.transpose(1, 2))
        ray_world = torch.nn.functional.normalize(ray_world, dim=-1, eps=1e-6)

        centers = pose_b44[:, None, :3, 3].expand(bsz, height * width, 3)
        return centers.reshape(-1, 3), ray_world.reshape(-1, 3)

    def create_training_buffer(self, buffer_size=None):
        """Fill ACE-style buffer with image features plus camera ray metadata."""
        torch.backends.cudnn.benchmark = False
        effective_size = self.options.training_buffer_size if buffer_size is None else buffer_size
        self._current_buffer_size = effective_size

        buffer_batch_size = getattr(self.options, "buffer_batch_size", 10)
        if buffer_batch_size > 1:
            buffer_image_width = getattr(self.options, "buffer_image_width", None)
            if buffer_image_width is None:
                buffer_image_width = (self.options.image_resolution * 4 // 3 + 13) // 14 * 14
            buffer_dataset = self._build_train_dataset(
                image_width=buffer_image_width,
                augment=False,
                aug_rotation=0,
                aug_scale_max=1.0,
                aug_scale_min=1.0,
            )
        else:
            buffer_dataset = self.dataset

        batch_sampler = sampler.BatchSampler(
            sampler.RandomSampler(buffer_dataset, generator=self.batch_generator),
            batch_size=buffer_batch_size,
            drop_last=False,
        )

        def seed_worker(worker_id):
            worker_seed = torch.initial_seed() % 2**32
            np.random.seed(worker_seed)
            random.seed(worker_seed)

        training_dataloader = DataLoader(
            dataset=buffer_dataset,
            sampler=batch_sampler,
            batch_size=None,
            worker_init_fn=seed_worker,
            generator=self.loader_generator,
            pin_memory=True,
            num_workers=self.num_data_loader_workers,
            persistent_workers=self.num_data_loader_workers > 0,
            timeout=60 if self.num_data_loader_workers > 0 else 0,
        )

        buffer_on_cpu = getattr(self.options, "buffer_on_cpu", False)
        buffer_device = torch.device("cpu") if buffer_on_cpu else self.device
        dtype_feature = torch.float16 if self.options.use_half else torch.float32

        self.training_buffer = {
            "features": torch.empty((effective_size, self.regressor.feature_dim), dtype=dtype_feature, device=buffer_device),
            "camera_centers": torch.empty((effective_size, 3), dtype=torch.float32, device=buffer_device),
            "ray_dirs": torch.empty((effective_size, 3), dtype=torch.float32, device=buffer_device),
            "target_px": torch.empty((effective_size, 2), dtype=torch.float32, device=buffer_device),
            "gt_poses_inv": torch.empty((effective_size, 3, 4), dtype=torch.float32, device=buffer_device),
            "intrinsics": torch.empty((effective_size, 3, 3), dtype=torch.float32, device=buffer_device),
            "intrinsics_inv": torch.empty((effective_size, 3, 3), dtype=torch.float32, device=buffer_device),
        }

        _logger.info("[PointFusion] Starting training buffer creation.")
        self.regressor.eval()
        with torch.no_grad():
            buffer_idx = 0
            dataset_passes = 0
            sampled_total = 0
            sampled_duplicates = 0
            pbar = tqdm(total=effective_size, unit="samples", unit_scale=True, desc="PointFusionBuffer")

            while buffer_idx < effective_size:
                dataset_passes += 1
                for batch in training_dataloader:
                    image_bchw, image_mask_b1hw, pose_b44, pose_inv_b44, intrinsics_b33, intrinsics_inv_b33, _, _ = batch
                    image_bchw = image_bchw.to(self.device, non_blocking=True)
                    image_mask_b1hw = image_mask_b1hw.to(self.device, non_blocking=True)
                    pose_b44 = pose_b44.to(self.device, non_blocking=True)
                    pose_inv_b44 = pose_inv_b44.to(self.device, non_blocking=True)
                    intrinsics_b33 = intrinsics_b33.to(self.device, non_blocking=True)
                    intrinsics_inv_b33 = intrinsics_inv_b33.to(self.device, non_blocking=True)

                    pose_inv_b34 = pose_inv_b44[:, :3, :] if pose_inv_b44.shape[1] == 4 else pose_inv_b44
                    if image_bchw.dtype == torch.float16:
                        image_bchw = image_bchw.float()

                    with autocast("cuda", enabled=self.options.use_half):
                        features_bchw = self.regressor.get_features(image_bchw)

                    bsz, _, height, width = features_bchw.shape
                    image_mask_b1hw = TF.resize(
                        image_mask_b1hw,
                        [height, width],
                        interpolation=TF.InterpolationMode.NEAREST,
                    ).bool()
                    if image_mask_b1hw.sum() == 0:
                        continue

                    pixel_positions_b2hw = self.pixel_grid_2HW[:, :height, :width].clone()
                    pixel_positions_b2hw = pixel_positions_b2hw.unsqueeze(0).expand(bsz, 2, height, width)
                    camera_centers_n3, ray_dirs_n3 = self._make_world_rays(
                        pixel_positions_b2hw,
                        pose_b44,
                        intrinsics_inv_b33,
                    )

                    num_locations = height * width
                    batch_data = {
                        "features": self._flatten_bchw(features_bchw),
                        "camera_centers": camera_centers_n3,
                        "ray_dirs": ray_dirs_n3,
                        "target_px": pixel_positions_b2hw.permute(0, 2, 3, 1).reshape(-1, 2),
                        "gt_poses_inv": pose_inv_b34[:, None].expand(bsz, num_locations, 3, 4).reshape(-1, 3, 4),
                        "intrinsics": intrinsics_b33[:, None].expand(bsz, num_locations, 3, 3).reshape(-1, 3, 3),
                        "intrinsics_inv": intrinsics_inv_b33[:, None].expand(bsz, num_locations, 3, 3).reshape(-1, 3, 3),
                    }

                    mask_n1 = image_mask_b1hw.float().permute(0, 2, 3, 1).reshape(-1)
                    replacement_cfg = getattr(self.options, "buffer_sampling_replacement", None)
                    use_replacement = True if replacement_cfg is None else bool(replacement_cfg)
                    features_to_select = min(self.options.samples_per_image * bsz, effective_size - buffer_idx)
                    if not use_replacement:
                        features_to_select = min(features_to_select, int((mask_n1 > 0).sum().item()))
                    if features_to_select <= 0:
                        continue

                    sample_idxs = torch.multinomial(
                        mask_n1,
                        features_to_select,
                        replacement=use_replacement,
                        generator=self.sampling_generator,
                    )
                    sampled_total += int(sample_idxs.numel())
                    sampled_duplicates += int(sample_idxs.numel() - torch.unique(sample_idxs).numel())

                    buffer_offset = buffer_idx + features_to_select
                    for key, value in batch_data.items():
                        self.training_buffer[key][buffer_idx:buffer_offset] = value[sample_idxs].to(
                            buffer_device,
                            non_blocking=True,
                        )
                    buffer_idx = buffer_offset
                    pbar.update(features_to_select)
                    pbar.set_postfix(n_pass=dataset_passes)
                    if buffer_idx >= effective_size:
                        break
            pbar.close()

        buffer_memory = sum(v.element_size() * v.nelement() for v in self.training_buffer.values()) / (1024**3)
        dup_ratio = (sampled_duplicates / sampled_total) if sampled_total > 0 else 0.0
        _logger.info(
            "[PointFusion] Created buffer %.2fGB, passes=%d, duplicate_ratio=%.4f.",
            buffer_memory,
            dataset_passes,
            dup_ratio,
        )
        self.regressor.train()

    def run_epoch(self):
        torch.backends.cudnn.benchmark = True
        current_buffer_size = getattr(self, "_current_buffer_size", self.options.training_buffer_size)
        random_indices = torch.randperm(current_buffer_size, generator=self.training_generator)

        for batch_start in range(0, current_buffer_size, self.options.batch_size):
            batch_end = batch_start + self.options.batch_size
            if batch_end > current_buffer_size:
                continue
            idx = random_indices[batch_start:batch_end]
            buf_dev = next(iter(self.training_buffer.values())).device
            to_dev = self.device if buf_dev.type == "cpu" else buf_dev
            self.training_step(
                self.training_buffer["features"][idx].contiguous().to(to_dev, non_blocking=True),
                self.training_buffer["camera_centers"][idx].contiguous().to(to_dev, non_blocking=True),
                self.training_buffer["ray_dirs"][idx].contiguous().to(to_dev, non_blocking=True),
                self.training_buffer["target_px"][idx].contiguous().to(to_dev, non_blocking=True),
                self.training_buffer["gt_poses_inv"][idx].contiguous().to(to_dev, non_blocking=True),
                self.training_buffer["intrinsics"][idx].contiguous().to(to_dev, non_blocking=True),
                self.training_buffer["intrinsics_inv"][idx].contiguous().to(to_dev, non_blocking=True),
            )
            self.iteration += 1

    def training_step(self, features_bC, camera_centers_b3, ray_dirs_b3, target_px_b2, gt_inv_poses_b34, Ks_b33, invKs_b33):
        batch_size = features_bC.shape[0]
        h, w = 16, batch_size // 16
        if h * w != batch_size:
            batch_size = h * w
            features_bC = features_bC[:batch_size]
            camera_centers_b3 = camera_centers_b3[:batch_size]
            ray_dirs_b3 = ray_dirs_b3[:batch_size]
            target_px_b2 = target_px_b2[:batch_size]
            gt_inv_poses_b34 = gt_inv_poses_b34[:batch_size]
            Ks_b33 = Ks_b33[:batch_size]
            invKs_b33 = invKs_b33[:batch_size]

        with autocast("cuda", enabled=self.options.use_half):
            fused_bC = self.regressor.fuse_buffer_features(
                features_bC,
                camera_centers_b3,
                ray_dirs_b3,
                self.point_bank,
            )
            fused_bCHW = fused_bC.view(1, h, w, self.regressor.feature_dim).permute(0, 3, 1, 2)
            pred_scene_coords_b3HW = self.regressor.get_scene_coordinates(fused_bCHW)

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
        invalid_mask_b1 = invalid_min_depth_b1 | invalid_repro_b1 | invalid_max_depth_b1
        valid_mask_b1 = ~invalid_mask_b1

        valid_reprojection_error_b1 = reprojection_error_b1[valid_mask_b1]
        loss_valid = self.repro_loss.compute(valid_reprojection_error_b1, self.iteration)

        pixel_grid_crop_b31 = to_homogeneous(target_px_b2.unsqueeze(2))
        target_camera_coords_b31 = self.options.depth_target * torch.bmm(invKs_b33, pixel_grid_crop_b31)
        invalid_mask_b11 = invalid_mask_b1.unsqueeze(2)
        loss_invalid = torch.abs(target_camera_coords_b31 - pred_cam_coords_b31).masked_select(invalid_mask_b11).sum()

        loss = (loss_valid + loss_invalid) / batch_size
        self.optimizer.zero_grad(set_to_none=True)
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

        if self.iteration % self.iterations_output == 0:
            time_since_start = time.time() - self.training_start
            fraction_valid = float(valid_mask_b1.sum() / batch_size)
            _logger.info(
                "Iteration: %6d / Epoch %03d|%03d, Loss: %.1f, Valid: %.1f%%, Time: %.2fs",
                self.iteration,
                self.epoch,
                self.options.epochs,
                loss.item(),
                fraction_valid * 100,
                time_since_start,
            )

        self.scheduler.step()
        return loss

    def save_model(self, output_path):
        checkpoint = {
            "model_type": "ace_pointcloud_fusion",
            "image_backbone": "ace_fcn",
            "heads_state_dict": self.regressor.heads.state_dict(),
            "adaptor_state_dict": self.regressor.adaptor.state_dict(),
            "point_feature_path": str(self.options.point_feature_path),
            "point_feature_dim": self._point_feature_dim,
            "encoder_path": str(self.options.encoder_path),
            "feature_dim": self.regressor.feature_dim,
        }
        torch.save(checkpoint, output_path)
        _logger.info("[PointFusion] Saved checkpoint to: %s", output_path)


class TrainerDINOPointCloudFusion(TrainerACEPointCloudFusion):
    """DINOv2 image-feature variant of point-cloud fusion training."""

    def _create_regressor(self):
        regressor = RegressorDINOPointCloudFusion.create_from_encoder(
            dinov2_path=self.options.dinov2_path,
            mean=self.dataset.mean_cam_center,
            num_head_blocks=self.options.num_head_blocks,
            use_homogeneous=self.options.use_homogeneous,
            point_feature_dim=self._point_feature_dim,
            num_encoder_features=1024,
            freeze_backbone=self.options.freeze_backbone,
            adaptor_hidden_dim=getattr(self.options, "adaptor_hidden_dim", 512),
            adaptor_heads=getattr(self.options, "adaptor_heads", 8),
            adaptor_dropout=getattr(self.options, "adaptor_dropout", 0.1),
            max_context_points=getattr(self.options, "max_context_points", 4096),
        )
        _logger.info("Loaded DINOv2 encoder from: %s", self.options.dinov2_path)
        return regressor

    def save_model(self, output_path):
        checkpoint = {
            "model_type": "dinov2_pointcloud_fusion",
            "image_backbone": "dinov2",
            "heads_state_dict": self.regressor.heads.state_dict(),
            "adaptor_state_dict": self.regressor.adaptor.state_dict(),
            "point_feature_path": str(self.options.point_feature_path),
            "point_feature_dim": self._point_feature_dim,
            "dinov2_path": str(self.options.dinov2_path),
            "feature_dim": self.regressor.feature_dim,
        }
        torch.save(checkpoint, output_path)
        _logger.info("[PointFusion-DINO] Saved checkpoint to: %s", output_path)
