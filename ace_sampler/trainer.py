import logging
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from torch.utils.data import sampler as torch_sampler

from ace_network import Regressor
from dataset_origin import CamLocDataset
from ace_util import get_pixel_grid
from .model import SamplerNet

_logger = logging.getLogger(__name__)


class SamplerTrainer:
    def __init__(self, options):
        self.options = options
        self.device = torch.device(options.device)

        self.dataset = CamLocDataset(
            root_dir=options.scene / 'train',
            mode=0,
            use_half=options.use_half,
            image_height=options.image_resolution,
            augment=options.use_aug,
            aug_rotation=options.aug_rotation,
            aug_scale_max=options.aug_scale,
            aug_scale_min=1.0 / options.aug_scale,
        )

        # Load frozen ACE regressor
        enc_sd = torch.load(options.encoder_path, map_location='cpu')
        head_sd = torch.load(options.ace_head, map_location='cpu')
        head_sd = {k: v.float() for k, v in head_sd.items()}
        self.regressor = Regressor.create_from_split_state_dict(enc_sd, head_sd)
        self.regressor = self.regressor.to(self.device).eval()
        for p in self.regressor.parameters():
            p.requires_grad_(False)

        self.sampler_net = SamplerNet(in_channels=self.regressor.feature_dim).to(self.device)
        self.optimizer = optim.AdamW(self.sampler_net.parameters(), lr=options.sampler_lr)
        self.scaler = GradScaler(enabled=options.use_half)

        self.pixel_grid = get_pixel_grid(self.regressor.OUTPUT_SUBSAMPLE).to(self.device)

    @torch.no_grad()
    def _compute_error_map(self, image_B1HW, gt_pose_inv_B44, intrinsics_B33):
        """Returns (error_map (B,1,Hf,Wf), features (B,C,Hf,Wf)). Error in pixels."""
        with autocast(enabled=self.options.use_half):
            features = self.regressor.get_features(image_B1HW)
            scene_coords = self.regressor.get_scene_coordinates(features).float()

        B, _, Hf, Wf = scene_coords.shape
        N = Hf * Wf

        coords_flat = scene_coords.view(B, 3, N)
        ones = torch.ones(B, 1, N, device=self.device)
        coords_homo = torch.cat([coords_flat, ones], dim=1)           # (B,4,N)
        cam_coords = torch.bmm(gt_pose_inv_B44[:, :3], coords_homo)   # (B,3,N)

        uv_homo = torch.bmm(intrinsics_B33, cam_coords)               # (B,3,N)
        z = uv_homo[:, 2:3].clamp(min=0.1)
        uv_pred = uv_homo[:, :2] / z                                  # (B,2,N)

        grid = self.pixel_grid[:, :Hf, :Wf].reshape(2, -1)
        grid = grid.unsqueeze(0).expand(B, 2, N)                      # (B,2,N)

        error = torch.norm(uv_pred - grid, dim=1)                     # (B,N)
        valid = (cam_coords[:, 2] > 0.1) & (cam_coords[:, 2] < 1000.0)
        error = error * valid.float()

        return error.view(B, 1, Hf, Wf), features

    def train(self):
        batch_gen = torch.Generator()
        batch_gen.manual_seed(2089)
        batch_sampler = torch_sampler.BatchSampler(
            torch_sampler.RandomSampler(self.dataset, generator=batch_gen),
            batch_size=1, drop_last=False)
        loader = DataLoader(self.dataset, sampler=batch_sampler,
                            batch_size=None, num_workers=4, pin_memory=True)

        for epoch in range(self.options.sampler_epochs):
            total_loss, n_steps = 0.0, 0
            for batch in loader:
                image_B1HW, _, pose, gt_pose_inv_B44, intrinsics_B33, _, _, _ = batch
                image_B1HW = image_B1HW.to(self.device, non_blocking=True)
                gt_pose_inv_B44 = gt_pose_inv_B44.to(self.device, non_blocking=True)
                intrinsics_B33 = intrinsics_B33.to(self.device, non_blocking=True)

                error_map, features = self._compute_error_map(
                    image_B1HW, gt_pose_inv_B44, intrinsics_B33)

                target = torch.exp(-self.options.sampler_alpha * error_map.clamp(0, 500))

                self.optimizer.zero_grad(set_to_none=True)
                with autocast(enabled=self.options.use_half):
                    pred = self.sampler_net(features.detach().float())
                    loss = F.mse_loss(pred, target)

                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()

                total_loss += loss.item()
                n_steps += 1
                if n_steps % 100 == 0:
                    _logger.info(f'Epoch {epoch} step {n_steps} loss {total_loss/n_steps:.4f}')

            avg = total_loss / max(n_steps, 1)
            _logger.info(f'Epoch {epoch} done. avg_loss={avg:.4f}')
            self.sampler_net.save(
                str(self.options.sampler_output),
                meta={'epoch': epoch, 'alpha': self.options.sampler_alpha,
                      'ace_head': str(self.options.ace_head)})

        _logger.info(f'Sampler saved to {self.options.sampler_output}')
