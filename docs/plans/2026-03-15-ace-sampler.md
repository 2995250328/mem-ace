# ACE Learnable Sampler Implementation Plan

**Goal:** Add a learnable sampling module (`ace_sampler/`) that trains a confidence predictor from reprojection error and uses it to replace random sampling during ACE buffer filling.

**Architecture:** Two-phase: Phase 1 trains a lightweight SamplerNet (depthwise-sep CNN on encoder features) end-to-end using a frozen ACE model's reprojection error as supervision signal. Phase 2 plugs the trained SamplerNet into `ace_trainer.py`'s buffer filling as a drop-in replacement for `torch.multinomial`, using Part A (top-k confidence) + Part B (random supplement).

**Tech Stack:** PyTorch 2.0, existing `ace_network.py` (Regressor), `dataset_origin.py` (CamLocDataset), `ace_util.py` (get_pixel_grid, to_homogeneous), `ace_trainer_full.py` (expand_neighbors_gpu reference).

---

## Context

Original ACE fills its training buffer by uniformly random-sampling pixels from each image (masked by `image_mask`). The hypothesis is that sampling points with low reprojection error (i.e., points the current ACE model already predicts well) produces a higher-quality buffer. A lightweight SamplerNet is trained to predict per-pixel confidence from encoder features, then used to bias buffer sampling toward stable, well-localized points.

**Constraint:** All new code lives in `ace_sampler/`. Root gets one thin entry script. `train_ace.py` and all existing flows are untouched by default.

---

## Critical Files

- **Read:** `ace_trainer.py` lines 60-83 (dataset creation), 210-340 (create_training_buffer)
- **Read:** `ace_network.py` — `Regressor.create_from_split_state_dict`, `get_features`, `get_scene_coordinates`, `OUTPUT_SUBSAMPLE=8`
- **Read:** `ace_trainer_full.py` lines 173-181 (`expand_neighbors_gpu`), 186-232 (`compute_reprojection_error_map`)
- **Read:** `dataset_origin.py` — `CamLocDataset.__init__` augmentation params, `__getitem__` return tuple
- **Read:** `ace_util.py` — `get_pixel_grid`, `to_homogeneous`
- **Modify:** `ace_trainer.py` — add optional `sampler_net` path in `__init__`, branch in `create_training_buffer`
- **Create:** `ace_sampler/__init__.py`, `model.py`, `trainer.py`, `buffer_sampler.py`, `options.py`, `README.md`
- **Create:** `train_ace_sampler.py` (root, thin wrapper)

---

## Reuse

- `ace_util.get_pixel_grid(subsample)` → pixel center coordinates grid
- `ace_util.to_homogeneous(x)` → homogeneous coords
- `Regressor.create_from_split_state_dict(enc_sd, head_sd)` → load full ACE model
- `CamLocDataset` from `dataset_origin.py` with `augment=True, aug_rotation=15, aug_scale_max=1.5, aug_scale_min=1/1.5`
- `expand_neighbors_gpu` — copy verbatim from `ace_trainer_full.py:173-181` into `buffer_sampler.py`

---

## Task 1: Create `ace_sampler/` skeleton

**Files:**
- Create: `ace_sampler/__init__.py`
- Create: `ace_sampler/model.py`
- Create: `ace_sampler/options.py`
- Create: `ace_sampler/trainer.py`
- Create: `ace_sampler/buffer_sampler.py`
- Create: `ace_sampler/README.md`

**Step 1: Create `ace_sampler/__init__.py`**
```python
from .model import SamplerNet
from .trainer import SamplerTrainer
from .buffer_sampler import fill_buffer_with_sampler
```

**Step 2: Create `ace_sampler/model.py`**
```python
import torch
import torch.nn as nn

class SamplerNet(nn.Module):
    """Lightweight confidence predictor on top of ACE encoder features.
    Input:  (B, 512, H, W) encoder feature map
    Output: (B, 1,   H, W) confidence map in [0, 1]
    """
    def __init__(self, in_channels: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            # Block 1: depthwise + pointwise
            nn.Conv2d(in_channels, in_channels, 3, 1, 1, groups=in_channels, bias=False),
            nn.Conv2d(in_channels, 128, 1, bias=False),
            nn.ReLU(inplace=True),
            # Block 2
            nn.Conv2d(128, 128, 3, 1, 1, groups=128, bias=False),
            nn.Conv2d(128, 64, 1, bias=False),
            nn.ReLU(inplace=True),
            # Head
            nn.Conv2d(64, 1, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(features))

    def save(self, path: str, meta: dict = None):
        payload = {'state_dict': self.state_dict(), 'in_channels': self.net[1].in_channels}
        if meta:
            payload.update(meta)
        torch.save(payload, path)

    @classmethod
    def load(cls, path: str, device) -> 'SamplerNet':
        ckpt = torch.load(path, map_location=device)
        net = cls(in_channels=ckpt.get('in_channels', 512))
        net.load_state_dict(ckpt['state_dict'])
        return net.to(device)
```

**Step 3: Verify model forward pass (user runs)**
```bash
cd /home/xwh/project/ace_depth
python -c "
import sys; sys.path.insert(0, '.')
import torch
from ace_sampler.model import SamplerNet
net = SamplerNet(512)
x = torch.randn(1, 512, 60, 80)
y = net(x)
assert y.shape == (1, 1, 60, 80), y.shape
assert y.min() >= 0 and y.max() <= 1
print('OK params:', sum(p.numel() for p in net.parameters()))
"
```
Expected: `OK params: 49345` (approx)

---

## Task 2: `ace_sampler/options.py`

**File:** Create `ace_sampler/options.py`

```python
import argparse
from pathlib import Path

def _strtobool(v):
    return str(v).lower() in ('true', '1', 'yes')

def add_sampler_train_args(parser):
    """Phase 1: train SamplerNet e2e."""
    g = parser.add_argument_group('sampler_train')
    g.add_argument('scene', type=Path, help='scene root (contains train/)')
    g.add_argument('ace_head', type=Path, help='trained ACE head .pt')
    g.add_argument('sampler_output', type=Path, help='output sampler .pt')
    g.add_argument('--encoder_path', type=Path, default=Path('ace_encoder_pretrained.pt'))
    g.add_argument('--num_head_blocks', type=int, default=1)
    g.add_argument('--use_homogeneous', type=_strtobool, default=False)
    g.add_argument('--image_resolution', type=int, default=480)
    g.add_argument('--use_aug', type=_strtobool, default=True)
    g.add_argument('--aug_rotation', type=int, default=15)
    g.add_argument('--aug_scale', type=float, default=1.5)
    g.add_argument('--sampler_epochs', type=int, default=5)
    g.add_argument('--sampler_lr', type=float, default=1e-3)
    g.add_argument('--sampler_alpha', type=float, default=0.1,
                   help='confidence target: exp(-alpha * repro_error_px)')
    g.add_argument('--use_half', type=_strtobool, default=True)
    g.add_argument('--device', type=str, default='cuda')

def add_sampler_buffer_args(parser):
    """Phase 2: buffer filling options."""
    g = parser.add_argument_group('sampler_buffer')
    g.add_argument('--sampler_path', type=Path, default=None,
                   help='trained SamplerNet .pt; None = original random sampling')
    g.add_argument('--sampler_ratio', type=float, default=0.7,
                   help='fraction of samples from sampler top-k (rest random)')
    g.add_argument('--use_neighbors', type=_strtobool, default=False,
                   help='expand sampler selections to 4-connected neighbors')
```

---

## Task 3: `ace_sampler/trainer.py` (Phase 1)

**File:** Create `ace_sampler/trainer.py`

Checkpoint: trainer loads frozen ACE, iterates dataset, computes error map at feature resolution, trains SamplerNet with MSE loss.

```python
import logging, random, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from torch.utils.data import sampler as torch_sampler

from ace_network import Regressor
from dataset_origin import CamLocDataset
from ace_util import get_pixel_grid, to_homogeneous
from .model import SamplerNet

_logger = logging.getLogger(__name__)

class SamplerTrainer:
    def __init__(self, options):
        self.options = options
        self.device = torch.device(options.device)

        # Dataset — same augmentation as ACE training
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
        # head saved as half — convert to float for inference
        head_sd = {k: v.float() for k, v in head_sd.items()}
        self.regressor = Regressor.create_from_split_state_dict(enc_sd, head_sd)
        self.regressor = self.regressor.to(self.device).eval()
        for p in self.regressor.parameters():
            p.requires_grad_(False)

        # SamplerNet
        self.sampler_net = SamplerNet(in_channels=self.regressor.feature_dim).to(self.device)
        self.optimizer = optim.AdamW(self.sampler_net.parameters(), lr=options.sampler_lr)
        self.scaler = GradScaler(enabled=options.use_half)

        # Pixel grid for reprojection (at feature resolution)
        self.pixel_grid = get_pixel_grid(self.regressor.OUTPUT_SUBSAMPLE).to(self.device)

    @torch.no_grad()
    def _compute_error_map(self, image_B1HW, gt_pose_inv_B44, intrinsics_B33):
        """Returns error map (B,1,Hf,Wf) at feature resolution. Units: pixels."""
        with autocast(enabled=self.options.use_half):
            features = self.regressor.get_features(image_B1HW)          # (B,512,Hf,Wf)
            scene_coords = self.regressor.get_scene_coordinates(features).float()  # (B,3,Hf,Wf)

        B, _, Hf, Wf = scene_coords.shape
        N = Hf * Wf

        # Scene coords → camera coords
        coords_flat = scene_coords.view(B, 3, N)
        ones = torch.ones(B, 1, N, device=self.device)
        coords_homo = torch.cat([coords_flat, ones], dim=1)          # (B,4,N)
        cam_coords = torch.bmm(gt_pose_inv_B44[:, :3], coords_homo)  # (B,3,N)

        # Project to pixel space
        uv_homo = torch.bmm(intrinsics_B33, cam_coords)              # (B,3,N)
        z = uv_homo[:, 2:3].clamp(min=0.1)
        uv_pred = uv_homo[:, :2] / z                                 # (B,2,N)

        # GT pixel grid at feature resolution
        grid = self.pixel_grid[:, :Hf, :Wf].reshape(2, -1)
        grid = grid.unsqueeze(0).expand(B, 2, N)                     # (B,2,N)

        # L2 reprojection error per pixel
        error = torch.norm(uv_pred - grid, dim=1)                    # (B,N)
        # Mask invalid depth
        valid = (cam_coords[:, 2] > 0.1) & (cam_coords[:, 2] < 1000.0)
        error = error * valid.float()

        return error.view(B, 1, Hf, Wf), features

    def train(self):
        batch_gen = torch.Generator(); batch_gen.manual_seed(2089)
        batch_sampler = torch_sampler.BatchSampler(
            torch_sampler.RandomSampler(self.dataset, generator=batch_gen),
            batch_size=1, drop_last=False)
        loader = DataLoader(self.dataset, sampler=batch_sampler,
                            batch_size=None, num_workers=4, pin_memory=True)

        for epoch in range(self.options.sampler_epochs):
            total_loss, n_steps = 0.0, 0
            for batch in loader:
                _, image_B1HW, image_mask, _, gt_pose_inv_B44, intrinsics_B33, _, _, _ = batch
                image_B1HW = image_B1HW.to(self.device, non_blocking=True)
                gt_pose_inv_B44 = gt_pose_inv_B44.to(self.device, non_blocking=True)
                intrinsics_B33 = intrinsics_B33.to(self.device, non_blocking=True)

                error_map, features = self._compute_error_map(
                    image_B1HW, gt_pose_inv_B44, intrinsics_B33)

                # Confidence target: exp(-alpha * error), clamped for stability
                target = torch.exp(-self.options.sampler_alpha * error_map.clamp(0, 500))

                # Train SamplerNet
                self.optimizer.zero_grad(set_to_none=True)
                with autocast(enabled=self.options.use_half):
                    pred = self.sampler_net(features.detach().float())
                    loss = F.mse_loss(pred, target)

                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()

                total_loss += loss.item(); n_steps += 1
                if n_steps % 100 == 0:
                    _logger.info(f'Epoch {epoch} step {n_steps} loss {total_loss/n_steps:.4f}')

            avg = total_loss / max(n_steps, 1)
            _logger.info(f'Epoch {epoch} done. avg_loss={avg:.4f}')
            self.sampler_net.save(
                str(self.options.sampler_output),
                meta={'epoch': epoch, 'alpha': self.options.sampler_alpha,
                      'ace_head': str(self.options.ace_head)})

        _logger.info(f'Sampler saved to {self.options.sampler_output}')
```

---

## Task 4: `ace_sampler/buffer_sampler.py` (Phase 2)

**File:** Create `ace_sampler/buffer_sampler.py`

Checkpoint: drop-in replacement for ace_trainer.py sampling; Part A topk + Part B random; optional neighbor expansion.

```python
import torch, logging
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import sampler as torch_sampler
from torch.cuda.amp import autocast

_logger = logging.getLogger(__name__)

def expand_neighbors_gpu(coords, H, W, patch_size, include_diagonal=4):
    """Verbatim from ace_trainer_full.py:173-181."""
    if coords.shape[0] == 0: return coords
    offsets = [[0, 0]]
    if include_diagonal >= 4: offsets.extend([[-1,0],[1,0],[0,-1],[0,1]])
    if include_diagonal == 8: offsets.extend([[-1,-1],[-1,1],[1,-1],[1,1]])
    offsets = torch.tensor(offsets, device=coords.device, dtype=coords.dtype) * patch_size
    expanded = (coords.unsqueeze(1) + offsets).reshape(-1, 2)
    valid = (expanded[:,0]>=0)&(expanded[:,0]<W)&(expanded[:,1]>=0)&(expanded[:,1]<H)
    return torch.unique(expanded[valid], dim=0)

def fill_buffer_with_sampler(regressor, sampler_net, dataset, options, device,
                              pixel_grid_2HW, sampling_generator,
                              batch_generator, loader_generator):
    """Phase 2 buffer filling. Same schema as ace_trainer.py training_buffer."""
    buf_size = options.training_buffer_size
    spi = options.samples_per_image
    ratio = getattr(options, 'sampler_ratio', 0.7)
    use_nbr = getattr(options, 'use_neighbors', False)
    patch = regressor.OUTPUT_SUBSAMPLE

    buf = {
        'features':       torch.empty((buf_size, regressor.feature_dim),
                                      dtype=torch.float16 if options.use_half else torch.float32,
                                      device=device),
        'target_px':      torch.empty((buf_size, 2), dtype=torch.float32, device=device),
        'gt_poses_inv':   torch.empty((buf_size, 3, 4), dtype=torch.float32, device=device),
        'intrinsics':     torch.empty((buf_size, 3, 3), dtype=torch.float32, device=device),
        'intrinsics_inv': torch.empty((buf_size, 3, 3), dtype=torch.float32, device=device),
    }

    bs = torch_sampler.BatchSampler(
        torch_sampler.RandomSampler(dataset, generator=batch_generator),
        batch_size=1, drop_last=False)
    loader = DataLoader(dataset, sampler=bs, batch_size=None,
                        generator=loader_generator, num_workers=12,
                        pin_memory=True, persistent_workers=True, timeout=60)

    regressor.eval(); sampler_net.eval()
    idx = 0

    with torch.no_grad():
        while idx < buf_size:
            for _, img, mask, _, pose_inv, K, Kinv, _, _ in loader:
                img = img.to(device, non_blocking=True)
                mask = mask.to(device, non_blocking=True)
                pose_inv = pose_inv.to(device, non_blocking=True)
                K = K.to(device, non_blocking=True)
                Kinv = Kinv.to(device, non_blocking=True)
                B, _, H, W = img.shape

                with autocast(enabled=options.use_half):
                    feats = regressor.get_features(img)          # (B,512,Hf,Wf)
                _, _, Hf, Wf = feats.shape

                mask_f = F.interpolate(mask.float(), [Hf, Wf], mode='nearest').bool()[0,0]
                conf = sampler_net(feats.float()).squeeze() * mask_f.float()  # (Hf,Wf)

                total = min(spi, buf_size - idx)
                na = int(total * ratio)
                nb = total - na

                # Part A: top-k confidence
                flat = conf.view(-1)
                ka = min(na, int(mask_f.sum().item()))
                if ka > 0:
                    ti = torch.topk(flat, ka).indices
                    ya = torch.div(ti, Wf, rounding_mode='floor')
                    xa = ti % Wf
                    ca = torch.stack([xa, ya], 1).float() * patch + patch // 2
                    if use_nbr:
                        ca = expand_neighbors_gpu(ca, H, W, patch, include_diagonal=4)
                else:
                    ca = torch.zeros((0,2), device=device)

                # Part B: random supplement
                nb = min(nb, int(mask_f.float().sum().item()))
                if nb > 0:
                    ri = torch.multinomial(mask_f.view(-1).float(), nb,
                                           replacement=True, generator=sampling_generator)
                    yb = torch.div(ri, Wf, rounding_mode='floor')
                    xb = ri % Wf
                    cb = torch.stack([xb, yb], 1).float() * patch + patch // 2
                else:
                    cb = torch.zeros((0,2), device=device)

                coords = torch.cat([ca, cb], 0)
                if coords.shape[0] == 0: continue

                nx = 2.0 * coords[:,0] / (W-1) - 1.0
                ny = 2.0 * coords[:,1] / (H-1) - 1.0
                grid = torch.stack([nx, ny], 1).unsqueeze(0).unsqueeze(1)
                with autocast(enabled=options.use_half):
                    sf = F.grid_sample(feats, grid.to(feats.dtype),
                                       align_corners=True, mode='bilinear')
                fNC = sf.squeeze(2).permute(0,2,1).reshape(-1, regressor.feature_dim)

                n = fNC.shape[0]; add = min(n, buf_size - idx); end = idx + add
                buf['features'][idx:end]       = fNC[:add]
                buf['target_px'][idx:end]      = coords[:add]
                buf['gt_poses_inv'][idx:end]   = pose_inv[:,:3].expand(add,3,4)
                buf['intrinsics'][idx:end]     = K.expand(add,3,3)
                buf['intrinsics_inv'][idx:end] = Kinv.expand(add,3,3)
                idx = end
                if idx >= buf_size: break

    _logger.info(f'Sampler buffer filled: {idx} samples.')
    return buf
```

---

## Task 5: `train_ace_sampler.py` (root entry, Phase 1)

**File:** Create `train_ace_sampler.py` in project root

```python
#!/usr/bin/env python3
"""Phase 1 entry: train SamplerNet e2e from reprojection error."""
import argparse, logging
from ace_sampler.options import add_sampler_train_args
from ace_sampler.trainer import SamplerTrainer

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(name)s - %(message)s')
    parser = argparse.ArgumentParser(description='Train ACE SamplerNet (Phase 1)')
    add_sampler_train_args(parser)
    options = parser.parse_args()
    SamplerTrainer(options).train()
```

---

## Task 6: Modify `ace_trainer.py` for Phase 2 (minimal patch)

**File:** Modify `ace_trainer.py`

Add `--sampler_path` / `--sampler_ratio` / `--use_neighbors` to its options parser (or pass via options namespace), then in `__init__` load SamplerNet if path given, and in `create_training_buffer` branch on it.

**Patch to `__init__` (after regressor is created, ~line 101):**
```python
# Load optional SamplerNet for guided buffer sampling
self.sampler_net = None
sampler_path = getattr(self.options, 'sampler_path', None)
if sampler_path is not None:
    from ace_sampler.model import SamplerNet
    self.sampler_net = SamplerNet.load(str(sampler_path), self.device)
    self.sampler_net.eval()
    _logger.info(f'Loaded SamplerNet from {sampler_path}')
```

**Patch to `create_training_buffer` (replace the sampling section, ~lines 315-335):**
```python
if self.sampler_net is not None:
    from ace_sampler.buffer_sampler import fill_buffer_with_sampler
    self.training_buffer = fill_buffer_with_sampler(
        self.regressor, self.sampler_net, self.dataset,
        self.options, self.device, self.pixel_grid_2HW,
        self.sampling_generator, self.batch_generator, self.loader_generator)
    self.regressor.train()
    return
# --- original random sampling continues below (unchanged) ---
```

---

## Task 7: Verification commands

**Regression (user runs — original ACE must be unaffected):**
```bash
cd /home/xwh/project/ace_depth
conda activate ace
# No --sampler_path → original random sampling
python train_ace.py datasets/7scenes_chess output/chess_regression.pt --device cuda:0
python test_ace.py  datasets/7scenes_chess output/chess_regression.pt --device cuda:0
# Compare median error with pre-change baseline
```
Expected: same median translation/rotation error as before (within numerical tolerance).

**Phase 1 — train sampler:**
```bash
python train_ace_sampler.py \
    datasets/7scenes_chess \
    output/ace_models/chess.pt \
    output/ace_models/chess_sampler.pt \
    --device cuda:0 --sampler_epochs 3
# Expected: logs "Epoch 0 step 100 loss X.XXXX", saves chess_sampler.pt
```

**Phase 2 — buffer filling with sampler:**
```bash
python train_ace.py datasets/7scenes_chess output/chess_sampler_run.pt \
    --device cuda:0 \
    --sampler_path output/ace_models/chess_sampler.pt \
    --sampler_ratio 0.7
# Expected: logs "Loaded SamplerNet", "Sampler buffer filled: N samples"
```

**Model forward check:**
```bash
python -c "
import sys; sys.path.insert(0,'.')
import torch
from ace_sampler.model import SamplerNet
net = SamplerNet(512)
y = net(torch.randn(1,512,60,80))
assert y.shape==(1,1,60,80) and 0<=y.min() and y.max()<=1
print('OK params:', sum(p.numel() for p in net.parameters()))
"
```
