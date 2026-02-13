import logging
import random
import time
import numpy as np
import math
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.cuda.amp import GradScaler
from torch.amp import autocast
from torch.utils.data import DataLoader, sampler
import os
from datetime import datetime

from ace_util import get_pixel_grid, to_homogeneous
from ace_loss import ReproLoss
from ace_network_dinov2 import Regressor
from dataset_dinov2 import CamLocDatasetDINOv2

_logger = logging.getLogger(__name__)


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


class TrainerACEDINOv2:
    def __init__(self, options):
        self.options = options
        self.device = torch.device('cuda:0')
        self.base_seed = 2089
        set_seed(self.base_seed)

        # Generators (randperm on CPU requires CPU generator)
        self.batch_generator = torch.Generator().manual_seed(self.base_seed + 1023)
        self.loader_generator = torch.Generator().manual_seed(self.base_seed + 511)
        self.sampling_generator_cpu = torch.Generator().manual_seed(self.base_seed + 4095)
        self.training_generator = torch.Generator().manual_seed(self.base_seed + 8191)

        self.iteration = 0
        self.num_data_loader_workers = 12

        # Dataset with DINOv2 support
        self.dataset = CamLocDatasetDINOv2(
            root_dir=self.options.scene / "train",
            mode=0,
            use_half=self.options.use_half,
            image_height=self.options.image_resolution,
            augment=self.options.use_aug,
            aug_rotation=self.options.aug_rotation,
            aug_scale_max=self.options.aug_scale,
            aug_scale_min=1 / self.options.aug_scale,
        )

        # Regressor with DINOv2 encoder
        self.regressor = Regressor.create_from_encoder(
            dinov2_path=self.options.dinov2_path,
            mean=self.dataset.mean_cam_center,
            num_head_blocks=self.options.num_head_blocks,
            use_homogeneous=self.options.use_homogeneous,
            num_encoder_features=1024,  # ViT-L/14 output dimension
            freeze_backbone=self.options.freeze_backbone
        ).to(self.device)
        self.regressor.train()

        # Optimizer & Scheduler
        self.optimizer = optim.AdamW(self.regressor.parameters(), lr=self.options.learning_rate_min)
        steps_per_epoch = self.options.training_buffer_size // self.options.batch_size
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=self.options.learning_rate_max,
            epochs=self.options.epochs,
            steps_per_epoch=steps_per_epoch,
            cycle_momentum=False
        )
        try:
            self.scaler = torch.amp.GradScaler('cuda', enabled=self.options.use_half)
        except AttributeError:
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

        self.training_buffer = None
        self.ace_visualizer = None

    def create_training_buffer(self):
        """Create training buffer by sampling features from training images."""
        _logger.info("Creating training buffer...")

        buffer_size = self.options.training_buffer_size
        samples_per_image = self.options.samples_per_image

        # Create dataloader
        dataloader = DataLoader(
            self.dataset,
            batch_size=1,
            shuffle=True,
            num_workers=self.num_data_loader_workers,
            generator=self.loader_generator,
            pin_memory=True
        )

        # Initialize buffer
        buffer_features = torch.zeros(buffer_size, self.regressor.feature_dim)
        buffer_coords = torch.zeros(buffer_size, 3)
        buffer_count = 0

        self.regressor.eval()

        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                if buffer_count >= buffer_size:
                    break

                image = batch[0].to(self.device)
                pose_inv = batch[3].to(self.device)
                intrinsics_inv = batch[5].to(self.device)

                # Buffer creation: run encoder in float32 to avoid Half/float mismatch (no autocast here)
                if image.dtype == torch.float16:
                    image = image.float()
                # Extract features
                features = self.regressor.get_features(image)
                B, C, H, W = features.shape

                # Sample random pixels
                num_samples = min(samples_per_image, H * W)
                indices = torch.randperm(H * W, generator=self.sampling_generator_cpu)[:num_samples]

                # Convert to 2D coordinates
                y_coords = indices // W
                x_coords = indices % W

                # Get features at sampled locations
                sampled_features = features[0, :, y_coords, x_coords].T  # [num_samples, C]

                # Compute 3D scene coordinates (on device for matmul with intrinsics_inv)
                pixel_coords = torch.stack([
                    x_coords * self.regressor.OUTPUT_SUBSAMPLE,
                    y_coords * self.regressor.OUTPUT_SUBSAMPLE
                ], dim=1).float().to(self.device)

                # Backproject to 3D (homogeneous: [1, num_samples, 3] for matmul with intrinsics_inv [3,3])
                pixel_coords_homo = to_homogeneous(pixel_coords.unsqueeze(0), dim=2)
                cam_coords = torch.matmul(intrinsics_inv, pixel_coords_homo.transpose(1, 2))
                cam_coords = cam_coords.transpose(1, 2)

                # Transform to world coordinates (placeholder - actual coords from network)
                scene_coords_pred = self.regressor.get_scene_coordinates(features)
                sampled_coords = scene_coords_pred[0, :, y_coords, x_coords].T  # [num_samples, 3]

                # Add to buffer
                space_left = buffer_size - buffer_count
                num_to_add = min(num_samples, space_left)

                buffer_features[buffer_count:buffer_count + num_to_add] = sampled_features[:num_to_add].cpu()
                buffer_coords[buffer_count:buffer_count + num_to_add] = sampled_coords[:num_to_add].cpu()
                buffer_count += num_to_add

                # Log every 10 batches: 10 images × samples_per_image (e.g. 1024) = 10240 samples per log line
                if (batch_idx + 1) % 10 == 0:
                    _logger.info(f"Buffer progress: {buffer_count}/{buffer_size}")

        self.regressor.train()

        _logger.info(f"Training buffer created with {buffer_count} samples")
        self.training_buffer = (buffer_features[:buffer_count], buffer_coords[:buffer_count])

    def training_step(self, batch):
        """Single training step."""
        image = batch[0].to(self.device)
        # Dataset returns 4x4 pose_inv; use 3x4 [R|t] for projection (same as verify_buffer_dinov2)
        pose_inv = batch[3].to(self.device)
        if pose_inv.shape[1] == 4 and pose_inv.shape[2] == 4:
            pose_inv = pose_inv[:, :3, :]  # [B, 3, 4]
        intrinsics = batch[4].to(self.device)
        intrinsics_inv = batch[5].to(self.device)

        depth_min = getattr(self.options, 'depth_min', 0.01)
        depth_max = getattr(self.options, 'depth_max', 1e6)
        repro_hard_clamp = getattr(self.options, 'repro_loss_hard_clamp', 1000.0)
        depth_target = getattr(self.options, 'depth_target', 1.0)

        with autocast('cuda', enabled=self.options.use_half):
            # Forward pass
            scene_coords = self.regressor(image)

            B, _, H, W = scene_coords.shape
            N = H * W
            pixel_grid_crop = self.pixel_grid_2HW[:, :H, :W].to(scene_coords.device)  # [2, H, W]

            # Scene coords to camera coords and project
            pred_scene = scene_coords.view(B, 3, -1)
            pred_scene_homo = to_homogeneous(pred_scene, dim=1)  # [B, 4, N]
            pred_cam = torch.bmm(pose_inv, pred_scene_homo)  # [B, 3, N]
            pred_px = torch.bmm(intrinsics, pred_cam)  # [B, 3, N]
            pred_px[:, 2].clamp_(min=depth_min)
            pred_px_2 = pred_px[:, :2] / pred_px[:, 2:3]  # [B, 2, N]

            target_px = pixel_grid_crop.reshape(2, -1).unsqueeze(0).expand(B, 2, -1)  # [B, 2, N]
            repro_err_b1N = torch.norm(pred_px_2 - target_px, dim=1, p=1)  # [B, N]

            invalid_min_depth = pred_cam[:, 2] < depth_min
            invalid_max_depth = pred_cam[:, 2] > depth_max
            invalid_repro = repro_err_b1N > repro_hard_clamp
            valid_mask = ~(invalid_min_depth | invalid_max_depth | invalid_repro)

            valid_repro_err = repro_err_b1N[valid_mask]
            loss_valid = self.repro_loss.compute(valid_repro_err, self.iteration) if valid_repro_err.numel() > 0 else scene_coords.sum() * 0.0

            invalid_mask_expand = invalid_min_depth | invalid_max_depth | invalid_repro
            target_px_homo = torch.cat([
                pixel_grid_crop.reshape(2, -1),
                torch.ones(1, N, device=scene_coords.device, dtype=scene_coords.dtype)
            ], dim=0).unsqueeze(0).expand(B, 3, N)
            target_cam = depth_target * torch.bmm(intrinsics_inv, target_px_homo)
            loss_invalid = torch.abs(target_cam - pred_cam).masked_select(invalid_mask_expand.unsqueeze(1)).sum()

            loss = loss_valid + loss_invalid
            loss = loss / (B * N)

        # Backward pass
        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.scheduler.step()

        return loss.item()

    def train(self):
        """Main training loop."""
        _logger.info("Starting training...")

        # Create training buffer
        self.create_training_buffer()

        # Training dataloader
        dataloader = DataLoader(
            self.dataset,
            batch_size=self.options.batch_size,
            shuffle=True,
            num_workers=self.num_data_loader_workers,
            generator=self.training_generator,
            pin_memory=True
        )

        # Training loop
        epoch_losses = []
        for epoch in range(self.options.epochs):
            epoch_loss = 0
            num_batches = 0

            for batch in dataloader:
                loss = self.training_step(batch)
                epoch_loss += loss
                num_batches += 1
                self.iteration += 1

                if self.iteration % self.iterations_output == 0:
                    avg_loss = epoch_loss / num_batches
                    lr = self.scheduler.get_last_lr()[0]
                    _logger.info(f"Iter {self.iteration}/{self.iterations}, "
                               f"Epoch {epoch+1}/{self.options.epochs}, "
                               f"Loss: {avg_loss:.4f}, LR: {lr:.6f}")

            avg_epoch_loss = epoch_loss / num_batches
            epoch_losses.append(avg_epoch_loss)
            _logger.info(f"Epoch {epoch+1} completed, Avg Loss: {avg_epoch_loss:.4f}")

        _logger.info("Training completed!")
        return epoch_losses

    def save_model(self, output_path):
        """Save trained model."""
        _logger.info(f"Saving model to {output_path}")
        # Save only the head network (scene-specific)
        torch.save(self.regressor.heads.state_dict(), output_path)
        _logger.info("Model saved successfully")
