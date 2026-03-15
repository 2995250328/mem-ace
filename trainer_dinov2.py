# Copyright © Niantic, Inc. 2022.
# DINOv2 variant: same logic as ace_trainer.py, only backbone and dataset swapped.

import logging
import random
import time

import numpy as np
from tqdm import tqdm
import torch
import torch.optim as optim
import torchvision.transforms.functional as TF
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torch.utils.data import sampler

from ace_util import get_pixel_grid, to_homogeneous
from ace_loss import ReproLoss
from ace_network_dinov2 import Regressor
from dataset_dinov2 import CamLocDatasetDINOv2
from dataset_wai_dinov2 import CamLocDatasetWAIDINOv2

_logger = logging.getLogger(__name__)


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


class TrainerACEDINOv2:
    """ACE trainer with DINOv2 backbone. Same flow as TrainerACE: buffer of features then train head only."""

    def __init__(self, options):
        self.options = options
        self.device = torch.device('cuda')
        self.base_seed = 2089
        set_seed(self.base_seed)

        self.batch_generator = torch.Generator().manual_seed(self.base_seed + 1023)
        self.loader_generator = torch.Generator().manual_seed(self.base_seed + 511)
        self.sampling_generator = torch.Generator(device=self.device).manual_seed(self.base_seed + 4095)
        self.training_generator = torch.Generator().manual_seed(self.base_seed + 8191)

        self.iteration = 0
        self.epoch = 0
        self.training_start = None
        self.num_data_loader_workers = 12

        # Dataset (DINOv2: RGB, resolution multiple of 14)
        self.dataset = self._build_train_dataset(
            image_width=None,
            augment=self.options.use_aug,
            aug_rotation=self.options.aug_rotation,
            aug_scale_max=self.options.aug_scale,
            aug_scale_min=1 / self.options.aug_scale,
        )

        _logger.info("Loaded training scan from: {} [backend={}] -- {} images, mean: {:.2f} {:.2f} {:.2f}".format(
            self._get_train_root(),
            getattr(self.options, "data_backend", "ace"),
            len(self.dataset),
            self.dataset.mean_cam_center[0],
            self.dataset.mean_cam_center[1],
            self.dataset.mean_cam_center[2]))

        # Regressor with DINOv2 encoder (only difference from ACE: create_from_encoder args)
        self.regressor = Regressor.create_from_encoder(
            dinov2_path=self.options.dinov2_path,
            mean=self.dataset.mean_cam_center,
            num_head_blocks=self.options.num_head_blocks,
            use_homogeneous=self.options.use_homogeneous,
            num_encoder_features=1024,
            freeze_backbone=self.options.freeze_backbone,
        )
        _logger.info("Loaded DINOv2 encoder from: {}".format(self.options.dinov2_path))

        self.regressor = self.regressor.to(self.device)
        self.regressor.train()

        self._current_buffer_size = self.options.training_buffer_size
        self._init_optimizer_scheduler()
        self.scaler = GradScaler("cuda", enabled=self.options.use_half)

        self.pixel_grid_2HW = get_pixel_grid(self.regressor.OUTPUT_SUBSAMPLE).to(self.device)
        self.iterations = self.options.epochs * self.options.training_buffer_size // self.options.batch_size
        self.iterations_output = 100

        self.repro_loss = ReproLoss(
            total_iterations=self.iterations,
            soft_clamp=self.options.repro_loss_soft_clamp,
            soft_clamp_min=self.options.repro_loss_soft_clamp_min,
            type=self.options.repro_loss_type,
            circle_schedule=(self.options.repro_loss_schedule == 'circle'),
        )

        self.training_buffer = None

    def _get_train_root(self):
        backend = getattr(self.options, "data_backend", "ace")
        if backend == "wai":
            # WAI uses a scene folder with scene_meta.json (e.g. .../indoor6/scene1_train).
            return self.options.scene
        return self.options.scene / "train"

    def _build_train_dataset(self, image_width=None, augment=False, aug_rotation=0, aug_scale_max=1.0, aug_scale_min=1.0):
        backend = getattr(self.options, "data_backend", "ace")
        root_dir = self._get_train_root()
        common = dict(
            root_dir=root_dir,
            mode=0,
            use_half=self.options.use_half,
            image_height=self.options.image_resolution,
            image_width=image_width,
            augment=augment,
            aug_rotation=aug_rotation,
            aug_scale_max=aug_scale_max,
            aug_scale_min=aug_scale_min,
        )
        if backend == "wai":
            return CamLocDatasetWAIDINOv2(
                **common,
                wai_repo_root=getattr(self.options, "wai_repo_root", None),
                wai_image_modality=getattr(self.options, "wai_image_modality", "image"),
            )
        return CamLocDatasetDINOv2(**common)

    def _init_optimizer_scheduler(self, buffer_size=None):
        effective_size = self.options.training_buffer_size if buffer_size is None else buffer_size
        steps_per_epoch = effective_size // self.options.batch_size
        self.optimizer = optim.AdamW(self.regressor.parameters(), lr=self.options.learning_rate_min)
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=self.options.learning_rate_max,
            epochs=self.options.epochs,
            steps_per_epoch=steps_per_epoch,
            cycle_momentum=False,
        )

    def reset_optimizer_scheduler(self, keep_optimizer_state=False, buffer_size=None):
        effective_size = self.options.training_buffer_size if buffer_size is None else buffer_size
        steps_per_epoch = effective_size // self.options.batch_size
        if not keep_optimizer_state:
            self.optimizer = optim.AdamW(self.regressor.parameters(), lr=self.options.learning_rate_min)
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=self.options.learning_rate_max,
            epochs=self.options.epochs,
            steps_per_epoch=steps_per_epoch,
            cycle_momentum=False,
        )

    def create_training_buffer(self, buffer_size=None):
        """Same as ace_trainer: fill GPU buffer with (features, target_px, gt_poses_inv, intrinsics, intrinsics_inv).
        When buffer_batch_size > 1, use a dataset with fixed (H,W) so that encoder runs on batches of images.
        """
        torch.backends.cudnn.benchmark = False
        effective_size = self.options.training_buffer_size if buffer_size is None else buffer_size
        self._current_buffer_size = effective_size

        buffer_batch_size = getattr(self.options, 'buffer_batch_size', 10)
        if buffer_batch_size > 1:
            # Fixed size so we can batch multiple images in one forward
            buffer_image_width = getattr(self.options, 'buffer_image_width', None)
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

        _logger.info("Starting creation of the training buffer.")

        buffer_on_cpu = getattr(self.options, "buffer_on_cpu", False)
        buffer_device = torch.device("cpu") if buffer_on_cpu else self.device
        if buffer_on_cpu:
            _logger.info("Buffer will be allocated on CPU (buffer_on_cpu=True) to avoid GPU OOM.")

        self.training_buffer = {
            'features': torch.empty(
                (effective_size, self.regressor.feature_dim),
                dtype=(torch.float32, torch.float16)[self.options.use_half],
                device=buffer_device,
            ),
            'target_px': torch.empty((effective_size, 2), dtype=torch.float32, device=buffer_device),
            'gt_poses_inv': torch.empty((effective_size, 3, 4), dtype=torch.float32, device=buffer_device),
            'intrinsics': torch.empty((effective_size, 3, 3), dtype=torch.float32, device=buffer_device),
            'intrinsics_inv': torch.empty((effective_size, 3, 3), dtype=torch.float32, device=buffer_device),
        }

        self.regressor.eval()
        with torch.no_grad():
            buffer_idx = 0
            dataset_passes = 0
            sampled_total = 0
            sampled_duplicates = 0
            pbar = tqdm(
                total=effective_size,
                unit="samples",
                unit_scale=True,
                desc="Buffer",
                dynamic_ncols=True,
            )

            while buffer_idx < effective_size:
                dataset_passes += 1
                for batch in training_dataloader:
                    # Dataset returns: image, image_mask, pose, pose_inv, intrinsics, intrinsics_inv, coords
                    image_BCHW, image_mask_B1HW, _, gt_pose_inv_B44, intrinsics_B33, intrinsics_inv_B33, _, _ = batch

                    image_BCHW = image_BCHW.to(self.device, non_blocking=True)
                    image_mask_B1HW = image_mask_B1HW.to(self.device, non_blocking=True)
                    gt_pose_inv_B44 = gt_pose_inv_B44.to(self.device, non_blocking=True)
                    intrinsics_B33 = intrinsics_B33.to(self.device, non_blocking=True)
                    intrinsics_inv_B33 = intrinsics_inv_B33.to(self.device, non_blocking=True)

                    # 4x4 -> 3x4 (same as ACE)
                    if gt_pose_inv_B44.shape[1] == 4:
                        gt_pose_inv_B34 = gt_pose_inv_B44[:, :3, :]
                    else:
                        gt_pose_inv_B34 = gt_pose_inv_B44

                    if image_BCHW.dtype == torch.float16:
                        image_BCHW = image_BCHW.float()

                    with autocast("cuda", enabled=self.options.use_half):
                        features_BCHW = self.regressor.get_features(image_BCHW)

                    B, C, H, W = features_BCHW.shape
                    image_mask_B1HW = TF.resize(image_mask_B1HW, [H, W], interpolation=TF.InterpolationMode.NEAREST)
                    image_mask_B1HW = image_mask_B1HW.bool()

                    if image_mask_B1HW.sum() == 0:
                        continue

                    pixel_positions_B2HW = self.pixel_grid_2HW[:, :H, :W].clone().unsqueeze(0).expand(B, 2, H, W)
                    gt_pose_inv = gt_pose_inv_B34.unsqueeze(1).expand(B, H * W, 3, 4).reshape(-1, 3, 4)
                    intrinsics = intrinsics_B33.unsqueeze(1).expand(B, H * W, 3, 3).reshape(-1, 3, 3)
                    intrinsics_inv = intrinsics_inv_B33.unsqueeze(1).expand(B, H * W, 3, 3).reshape(-1, 3, 3)

                    def normalize_shape(tensor_in):
                        return tensor_in.transpose(0, 1).flatten(1).transpose(0, 1)

                    batch_data = {
                        'features': normalize_shape(features_BCHW),
                        'target_px': normalize_shape(pixel_positions_B2HW),
                        'gt_poses_inv': gt_pose_inv,
                        'intrinsics': intrinsics,
                        'intrinsics_inv': intrinsics_inv,
                    }

                    image_mask_B1HW = image_mask_B1HW.float()
                    image_mask_N1 = normalize_shape(image_mask_B1HW)
                    replacement_cfg = getattr(self.options, "buffer_sampling_replacement", None)
                    use_replacement = True if replacement_cfg is None else bool(replacement_cfg)
                    features_to_select = min(
                        self.options.samples_per_image * B,
                        effective_size - buffer_idx,
                    )
                    if not use_replacement:
                        valid_count = int((image_mask_N1.view(-1) > 0).sum().item())
                        features_to_select = min(features_to_select, valid_count)
                    if features_to_select <= 0:
                        continue
                    sample_idxs = torch.multinomial(
                        image_mask_N1.view(-1),
                        features_to_select,
                        replacement=use_replacement,
                        generator=self.sampling_generator,
                    )
                    sampled_total += int(sample_idxs.numel())
                    sampled_duplicates += int(sample_idxs.numel() - torch.unique(sample_idxs).numel())

                    for k in batch_data:
                        batch_data[k] = batch_data[k][sample_idxs].to(buffer_device, non_blocking=True)
                    buffer_offset = buffer_idx + features_to_select
                    for k in batch_data:
                        self.training_buffer[k][buffer_idx:buffer_offset] = batch_data[k]

                    buffer_idx = buffer_offset
                    pbar.update(features_to_select)
                    pbar.set_postfix(n_pass=dataset_passes)
                    if buffer_idx >= effective_size:
                        break

            pbar.close()

        buffer_memory = sum(v.element_size() * v.nelement() for v in self.training_buffer.values()) / (1024**3)
        dup_ratio = (sampled_duplicates / sampled_total) if sampled_total > 0 else 0.0
        replacement_cfg = getattr(self.options, "buffer_sampling_replacement", None)
        use_replacement = True if replacement_cfg is None else bool(replacement_cfg)
        _logger.info("Created buffer of {:.2f}GB with {} passes (buffer_batch_size={}).".format(
            buffer_memory, dataset_passes, buffer_batch_size))
        _logger.info(
            "Buffer sampling stats: replacement=%s, duplicate_ratio=%.4f (%d/%d).",
            use_replacement,
            dup_ratio,
            sampled_duplicates,
            sampled_total,
        )
        self.regressor.train()

    def run_epoch(self):
        """Same as ace_trainer: shuffle buffer, iterate batches, training_step from buffer."""
        torch.backends.cudnn.benchmark = True
        current_buffer_size = getattr(self, "_current_buffer_size", self.options.training_buffer_size)
        random_indices = torch.randperm(current_buffer_size, generator=self.training_generator)

        for batch_start in range(0, current_buffer_size, self.options.batch_size):
            batch_end = batch_start + self.options.batch_size
            if batch_end > current_buffer_size:
                continue

            random_batch_indices = random_indices[batch_start:batch_end]
            buf_dev = next(iter(self.training_buffer.values())).device
            to_dev = self.device if buf_dev.type == "cpu" else buf_dev
            self.training_step(
                self.training_buffer['features'][random_batch_indices].contiguous().to(to_dev, non_blocking=True),
                self.training_buffer['target_px'][random_batch_indices].contiguous().to(to_dev, non_blocking=True),
                self.training_buffer['gt_poses_inv'][random_batch_indices].contiguous().to(to_dev, non_blocking=True),
                self.training_buffer['intrinsics'][random_batch_indices].contiguous().to(to_dev, non_blocking=True),
                self.training_buffer['intrinsics_inv'][random_batch_indices].contiguous().to(to_dev, non_blocking=True),
            )
            self.iteration += 1

    def training_step(self, features_bC, target_px_b2, gt_inv_poses_b34, Ks_b33, invKs_b33):
        """Same as ace_trainer: head-only forward, reprojection loss, invalid proxy loss."""
        batch_size = features_bC.shape[0]
        channels = features_bC.shape[1]
        # Reshape to fake BCHW for head (same as ACE: 16x32 for 1x1 conv; use 16 x (batch_size//16))
        h, w = 16, batch_size // 16
        if h * w != batch_size:
            batch_size = h * w
            features_bC = features_bC[:batch_size]
            target_px_b2 = target_px_b2[:batch_size]
            gt_inv_poses_b34 = gt_inv_poses_b34[:batch_size]
            Ks_b33 = Ks_b33[:batch_size]
            invKs_b33 = invKs_b33[:batch_size]
        features_bCHW = features_bC.view(1, h, w, channels).permute(0, 3, 1, 2)

        with autocast("cuda", enabled=self.options.use_half):
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
        invalid_mask_b1 = invalid_min_depth_b1 | invalid_repro_b1 | invalid_max_depth_b1
        valid_mask_b1 = ~invalid_mask_b1

        valid_reprojection_error_b1 = reprojection_error_b1[valid_mask_b1]
        loss_valid = self.repro_loss.compute(valid_reprojection_error_b1, self.iteration)

        pixel_grid_crop_b31 = to_homogeneous(target_px_b2.unsqueeze(2))
        target_camera_coords_b31 = self.options.depth_target * torch.bmm(invKs_b33, pixel_grid_crop_b31)
        invalid_mask_b11 = invalid_mask_b1.unsqueeze(2)
        loss_invalid = torch.abs(target_camera_coords_b31 - pred_cam_coords_b31).masked_select(invalid_mask_b11).sum()

        loss = loss_valid + loss_invalid
        loss /= batch_size

        self.optimizer.zero_grad(set_to_none=True)
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

        if self.iteration % self.iterations_output == 0:
            time_since_start = time.time() - self.training_start
            fraction_valid = float(valid_mask_b1.sum() / batch_size)
            _logger.info(
                'Iteration: {:6d} / Epoch {:03d}|{:03d}, Loss: {:.1f}, Valid: {:.1f}%, Time: {:.2f}s'.format(
                    self.iteration,
                    self.epoch,
                    self.options.epochs,
                    loss.item(),
                    fraction_valid * 100,
                    time_since_start,
                )
            )

        if hasattr(self.scheduler, 'total_steps') and self.iteration < self.scheduler.total_steps:
            self.scheduler.step()
        else:
            self.scheduler.step()

        return loss

    def train(self):
        """Same as ace_trainer: create buffer, run_epoch loop, save_model."""
        self.training_start = time.time()

        buffer_start = time.time()
        self.create_training_buffer()
        _logger.info("Filled training buffer in {:.1f}s.".format(time.time() - buffer_start))

        for self.epoch in range(self.options.epochs):
            self.run_epoch()

        self.save_model(self.options.output_map)
        _logger.info("Done. Total time: {:.1f}s.".format(time.time() - self.training_start))

    def save_model(self, output_path):
        """Same as ace_trainer: save head state_dict as half."""
        head_state_dict = self.regressor.heads.state_dict()
        for k in list(head_state_dict.keys()):
            head_state_dict[k] = head_state_dict[k].half()
        torch.save(head_state_dict, output_path)
        _logger.info("Saved trained head weights to: {}".format(output_path))
