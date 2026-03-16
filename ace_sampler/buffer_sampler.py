import logging
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import sampler as torch_sampler
from torch.cuda.amp import autocast

_logger = logging.getLogger(__name__)


def expand_neighbors_gpu(coords, H, W, patch_size, include_diagonal=4):
    """Verbatim from ace_trainer_full.py:173-181."""
    if coords.shape[0] == 0:
        return coords
    offsets = [[0, 0]]
    if include_diagonal >= 4:
        offsets.extend([[-1, 0], [1, 0], [0, -1], [0, 1]])
    if include_diagonal == 8:
        offsets.extend([[-1, -1], [-1, 1], [1, -1], [1, 1]])
    offsets = torch.tensor(offsets, device=coords.device, dtype=coords.dtype) * patch_size
    expanded = (coords.unsqueeze(1) + offsets).reshape(-1, 2)
    valid = (expanded[:, 0] >= 0) & (expanded[:, 0] < W) & \
            (expanded[:, 1] >= 0) & (expanded[:, 1] < H)
    return torch.unique(expanded[valid], dim=0)


def fill_buffer_with_sampler(regressor, sampler_net, dataset, options, device,
                              sampling_generator, batch_generator, loader_generator):
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

    regressor.eval()
    sampler_net.eval()
    idx = 0

    with torch.no_grad():
        while idx < buf_size:
            for img, mask, _, pose_inv, K, Kinv, _, _ in loader:
                img = img.to(device, non_blocking=True)
                mask = mask.to(device, non_blocking=True)
                pose_inv = pose_inv.to(device, non_blocking=True)
                K = K.to(device, non_blocking=True)
                Kinv = Kinv.to(device, non_blocking=True)
                B, _, H, W = img.shape

                with autocast(enabled=options.use_half):
                    feats = regressor.get_features(img)
                _, _, Hf, Wf = feats.shape

                mask_f = F.interpolate(mask.float(), [Hf, Wf], mode='nearest').bool()[0, 0]
                conf = sampler_net(feats.float()).squeeze() * mask_f.float()

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
                    ca = torch.zeros((0, 2), device=device)

                # Part B: random supplement
                nb = min(nb, int(mask_f.float().sum().item()))
                if nb > 0:
                    ri = torch.multinomial(mask_f.view(-1).float(), nb,
                                           replacement=True, generator=sampling_generator)
                    yb = torch.div(ri, Wf, rounding_mode='floor')
                    xb = ri % Wf
                    cb = torch.stack([xb, yb], 1).float() * patch + patch // 2
                else:
                    cb = torch.zeros((0, 2), device=device)

                coords = torch.cat([ca, cb], 0)
                if coords.shape[0] == 0:
                    continue

                nx = 2.0 * coords[:, 0] / (W - 1) - 1.0
                ny = 2.0 * coords[:, 1] / (H - 1) - 1.0
                grid = torch.stack([nx, ny], 1).unsqueeze(0).unsqueeze(1)
                with autocast(enabled=options.use_half):
                    sf = F.grid_sample(feats, grid.to(feats.dtype),
                                       align_corners=True, mode='bilinear')
                fNC = sf.squeeze(2).permute(0, 2, 1).reshape(-1, regressor.feature_dim)

                n = fNC.shape[0]
                add = min(n, buf_size - idx)
                end = idx + add
                buf['features'][idx:end]       = fNC[:add]
                buf['target_px'][idx:end]      = coords[:add]
                buf['gt_poses_inv'][idx:end]   = pose_inv[:, :3].expand(add, 3, 4)
                buf['intrinsics'][idx:end]     = K.expand(add, 3, 3)
                buf['intrinsics_inv'][idx:end] = Kinv.expand(add, 3, 3)
                idx = end
                if idx >= buf_size:
                    break

    _logger.info(f'Sampler buffer filled: {idx} samples.')
    return buf
