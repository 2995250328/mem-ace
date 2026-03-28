"""Multi-scene SamplerNet trainer.

Trains a single universal SamplerNet across N scenes simultaneously.
Each scene has its own frozen ACE regressor; the trainer selects the correct
one per batch using the scene_idx returned by MultiSceneDataset.
"""
import logging
import torch
import torch.nn.functional as F
import torch.optim as optim
from pathlib import Path
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from torch.utils.data import sampler as torch_sampler
from tqdm import tqdm

from ace_network import Regressor
from ace_util import get_pixel_grid
from .model import SamplerNet
from .multi_dataset import MultiSceneDataset, SceneEntry

_logger = logging.getLogger(__name__)


class MultiSceneSamplerTrainer:
    """Train a universal SamplerNet on multiple scenes.

    Args:
        entries  : list[SceneEntry]  — scene/head pairs, optionally with cluster info
        options  : argparse namespace (see add_multi_sampler_train_args)
    """

    def __init__(self, entries: list, options):
        self.options = options
        self.device  = torch.device('cuda')

        # ── Shared encoder (loaded once) ──────────────────────────────────
        enc_sd = torch.load(options.encoder_path, map_location='cpu',
                            weights_only=False)

        # ── Per-entry frozen regressors ───────────────────────────────────
        _logger.info(f'Loading {len(entries)} scene/cluster regressors...')
        self.regressors = []
        for i, e in enumerate(entries):
            head_sd = torch.load(e.head_path, map_location='cpu', weights_only=False)
            head_sd = {k: v.float() for k, v in head_sd.items()}
            reg = Regressor.create_from_split_state_dict(enc_sd, head_sd)
            reg = reg.to(self.device).eval()
            for p in reg.parameters():
                p.requires_grad_(False)
            self.regressors.append(reg)
            cluster_info = (f' [cluster {e.cluster_idx}/{e.num_clusters}]'
                            if e.num_clusters else '')
            _logger.info(f'  [{i:2d}] {Path(e.scene_path).name}{cluster_info}')

        feature_dim = self.regressors[0].feature_dim

        # ── Universal SamplerNet (larger capacity) ────────────────────────
        self.sampler_net = SamplerNet(
            in_channels  = feature_dim,
            mid_channels = options.mid_channels,
            num_branches = options.num_branches,
        ).to(self.device)
        n_params = sum(p.numel() for p in self.sampler_net.parameters())
        _logger.info(f'SamplerNet: {n_params/1e3:.1f}K params '
                     f'(mid={options.mid_channels}, branches={options.num_branches})')

        self.optimizer = optim.AdamW(self.sampler_net.parameters(),
                                     lr=options.sampler_lr)
        self.scaler    = GradScaler('cuda', enabled=options.use_half)
        self.pixel_grid = get_pixel_grid(
            self.regressors[0].OUTPUT_SUBSAMPLE).to(self.device)

        # ── Multi-scene dataset ───────────────────────────────────────────
        self.dataset = MultiSceneDataset(entries, options)
        _logger.info(f'Total training samples: {len(self.dataset)} '
                     f'across {len(entries)} entries')

    @torch.no_grad()
    def _compute_error_map(self, image_B1HW, gt_pose_inv_B44,
                           intrinsics_B33, scene_idx: int):
        """Reprojection error map using the scene-specific regressor."""
        reg = self.regressors[scene_idx]
        with autocast('cuda', enabled=self.options.use_half):
            features    = reg.get_features(image_B1HW)
            scene_coords = reg.get_scene_coordinates(features).float()

        B, _, Hf, Wf = scene_coords.shape
        N = Hf * Wf

        coords_flat = scene_coords.view(B, 3, N)
        ones        = torch.ones(B, 1, N, device=self.device)
        cam_coords  = torch.bmm(gt_pose_inv_B44[:, :3],
                                torch.cat([coords_flat, ones], dim=1))

        uv_homo = torch.bmm(intrinsics_B33, cam_coords)
        z       = uv_homo[:, 2:3].clamp(min=0.1)
        uv_pred = uv_homo[:, :2] / z

        grid  = self.pixel_grid[:, :Hf, :Wf].reshape(2, -1).unsqueeze(0).expand(B, 2, N)
        error = torch.norm(uv_pred - grid, dim=1)
        valid = (cam_coords[:, 2] > 0.1) & (cam_coords[:, 2] < 1000.0)
        error_map = (error * valid.float()).view(B, 1, Hf, Wf)

        return error_map, features

    def train(self):
        batch_gen = torch.Generator()
        batch_gen.manual_seed(2089)
        sampler = torch_sampler.RandomSampler(self.dataset, generator=batch_gen)
        loader = DataLoader(self.dataset, sampler=sampler,
                            batch_size=1, num_workers=4, pin_memory=True)

        for epoch in range(self.options.sampler_epochs):
            total_loss, n_steps = 0.0, 0
            _logger.info(f'Epoch {epoch}/{self.options.sampler_epochs} '
                         f'— {len(self.dataset)} samples')
            pbar = tqdm(loader, desc=f'Epoch {epoch}', unit='img',
                        dynamic_ncols=True)

            for batch in pbar:
                # MultiSceneDataset returns 9-tuple: 8 from CamLocDataset + scene_idx
                (image_B1HW, _, _pose, gt_pose_inv_B44,
                 intrinsics_B33, _intrinsics_inv, _coords, _fname,
                 scene_idx) = batch

                image_B1HW       = image_B1HW.to(self.device, non_blocking=True)
                gt_pose_inv_B44  = gt_pose_inv_B44.to(self.device, non_blocking=True)
                intrinsics_B33   = intrinsics_B33.to(self.device, non_blocking=True)
                s_idx            = int(scene_idx.item())

                error_map, features = self._compute_error_map(
                    image_B1HW, gt_pose_inv_B44, intrinsics_B33, s_idx)

                target = torch.exp(
                    -self.options.sampler_alpha * error_map.clamp(0, 500))

                self.optimizer.zero_grad(set_to_none=True)
                with autocast('cuda', enabled=self.options.use_half):
                    pred = self.sampler_net(features.detach().float())
                    loss = F.mse_loss(pred, target)

                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()

                total_loss += loss.item()
                n_steps    += 1
                pbar.set_postfix(loss=f'{total_loss/n_steps:.4f}')

            avg = total_loss / max(n_steps, 1)
            _logger.info(f'Epoch {epoch} done. avg_loss={avg:.4f}')
            self.sampler_net.save(
                str(self.options.sampler_output),
                meta={'epoch': epoch, 'alpha': self.options.sampler_alpha,
                      'num_scenes': len(self.regressors)})

        _logger.info(f'Universal sampler saved to {self.options.sampler_output}')
