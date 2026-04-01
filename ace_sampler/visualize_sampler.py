"""Visualize SamplerNet confidence map and sampled points vs SuperPoint on test images.

Usage:
    python ace_sampler/visualize_sampler.py \
        /mnt/storage/xwh/7Scenes/pgt_7scenes_chess \
        output/ace_models/7Scenes_pgt/pgt_7scenes_chess.pt \
        ace_sampler/04_evaluation/universal/20260323_223249_universal_sampler_all.pt \
        --out_dir ace_sampler/04_evaluation/vis_chess \
        --n_images 20 \
        --device cuda:0

Output per image (saved to --out_dir):
    <stem>_side.png  — 5-panel: RGB | SamplerNet conf | SamplerNet pts | SP heatmap | SP pts
"""

import argparse
import logging
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')
_logger = logging.getLogger(__name__)


# ── loaders ───────────────────────────────────────────────────────────────────

def load_regressor(encoder_path, head_path, device):
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from ace_network import Regressor
    enc_sd  = torch.load(encoder_path, map_location='cpu', weights_only=False)
    head_sd = torch.load(head_path,    map_location='cpu', weights_only=False)
    head_sd = {k: v.float() for k, v in head_sd.items()}
    reg = Regressor.create_from_split_state_dict(enc_sd, head_sd)
    return reg.to(device).eval()


def load_sampler(sampler_path, device):
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from ace_sampler.model import SamplerNet
    return SamplerNet.load(sampler_path, device).eval()


def load_superpoint(weights_path, device):
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from superpoint import SuperPointNet, SuperPointFrontend
    net = SuperPointNet()
    cuda = device.type == 'cuda'
    sp = SuperPointFrontend(
        superpoint_net=net,
        weights_path=str(weights_path),
        nms_dist=4,
        conf_thresh=0.015,
        nn_thresh=0.7,
        cuda=cuda,
    )
    return sp


# ── image helpers ─────────────────────────────────────────────────────────────

def load_image(path, height=480):
    """Returns grayscale tensor (1,1,H,W), gray numpy (H,W) float32, RGB uint8 (H,W,3)."""
    bgr   = cv2.imread(str(path))
    h0, w0 = bgr.shape[:2]
    w_new = int(round(w0 * height / h0))
    bgr_r = cv2.resize(bgr, (w_new, height))
    gray  = cv2.cvtColor(bgr_r, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    tensor = torch.from_numpy(gray).unsqueeze(0).unsqueeze(0)   # (1,1,H,W)
    rgb   = cv2.cvtColor(bgr_r, cv2.COLOR_BGR2RGB)
    return tensor, gray, rgb


# ── rendering helpers ─────────────────────────────────────────────────────────

def heatmap_overlay(score_hw: np.ndarray, rgb: np.ndarray,
                    colormap=cv2.COLORMAP_JET) -> np.ndarray:
    """Overlay a [0,1] score map on RGB image."""
    H, W = rgb.shape[:2]
    u8   = (score_hw * 255).clip(0, 255).astype(np.uint8)
    heat = cv2.applyColorMap(u8, colormap)
    heat = cv2.resize(heat, (W, H))
    heat_rgb = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    return (rgb * 0.5 + heat_rgb * 0.5).clip(0, 255).astype(np.uint8)


def draw_two_sets(rgb: np.ndarray, pts_a: np.ndarray, pts_b: np.ndarray,
                  color_a=(50, 220, 50), color_b=(220, 50, 50),
                  radius: int = 3) -> np.ndarray:
    """Draw two point sets on RGB (pts_b first so pts_a is on top)."""
    out = rgb.copy()
    for x, y in pts_b:
        cv2.circle(out, (int(x), int(y)), radius, color_b, -1)
    for x, y in pts_a:
        cv2.circle(out, (int(x), int(y)), radius, color_a, -1)
    return out


def draw_one_set(rgb: np.ndarray, pts: np.ndarray,
                 color=(255, 200, 0), radius: int = 3) -> np.ndarray:
    out = rgb.copy()
    for x, y in pts:
        cv2.circle(out, (int(x), int(y)), radius, color, -1)
    return out


def labeled_panel(img: np.ndarray, label: str) -> np.ndarray:
    p = img.copy()
    cv2.putText(p, label, (8, 24), cv2.FONT_HERSHEY_SIMPLEX,
                0.65, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(p, label, (8, 24), cv2.FONT_HERSHEY_SIMPLEX,
                0.65, (0, 0, 0), 1, cv2.LINE_AA)
    return p


def make_grid(*panels) -> np.ndarray:
    return np.concatenate(panels, axis=1)


# ── sampling helpers ──────────────────────────────────────────────────────────

def _nms_topk(conf: torch.Tensor, k: int, nms_size: int = 5) -> torch.Tensor:
    """Grid-based NMS: divide feature map into nms_size×nms_size non-overlapping
    cells, pick the argmax pixel in each cell (conf>0 only), sort by confidence,
    return top-k. If fewer than k cells have conf>0, returns all valid cells
    (caller should add random points to fill the budget).
    """
    if nms_size <= 1:
        return torch.topk(conf.view(-1), min(k, conf.numel())).indices

    Hf, Wf = conf.shape
    noise   = torch.rand_like(conf) * 1e-6          # tie-break within each cell
    conf_tb = conf + noise
    ph = (nms_size - Hf % nms_size) % nms_size
    pw = (nms_size - Wf % nms_size) % nms_size
    conf_pad = F.pad(conf_tb, (0, pw, 0, ph), value=-1.0)
    nh = conf_pad.shape[0] // nms_size
    nw = conf_pad.shape[1] // nms_size

    cells   = conf_pad.reshape(nh, nms_size, nw, nms_size).permute(0, 2, 1, 3)
    cells   = cells.reshape(nh, nw, nms_size * nms_size)
    argmax  = cells.max(dim=-1).indices             # (nh, nw)

    local_r = argmax // nms_size
    local_c = argmax % nms_size
    global_r = torch.arange(nh, device=conf.device).unsqueeze(1) * nms_size + local_r
    global_c = torch.arange(nw, device=conf.device).unsqueeze(0) * nms_size + local_c

    valid = (global_r < Hf) & (global_c < Wf) & \
            (conf[global_r.clamp(0, Hf-1), global_c.clamp(0, Wf-1)] > 0)
    gr, gc = global_r[valid], global_c[valid]

    # sort by original confidence descending
    order  = torch.argsort(conf[gr, gc], descending=True)
    gr, gc = gr[order], gc[order]
    flat   = gr * Wf + gc

    return flat[:k]   # may be fewer than k — caller handles the shortfall


def sampler_points(conf: torch.Tensor, patch: int, na: int, nb: int, device,
                   nms_size: int = 5):
    """
    Returns pts_a (NMS top-k) and pts_b (random) in full-image pixel coords (x,y).
    If NMS yields fewer than na points, the shortfall is added to Part B.
    """
    Hf, Wf = conf.shape

    ti   = _nms_topk(conf, na, nms_size=nms_size)   # may be < na
    ya_f = torch.div(ti, Wf, rounding_mode='floor')
    xa_f = ti % Wf
    pts_a = (torch.stack([xa_f, ya_f], 1).float() * patch + patch // 2).cpu().numpy()

    nb_actual = nb + (na - ti.numel())               # absorb shortfall
    ri   = torch.randint(0, Hf * Wf, (nb_actual,), device=device)
    yb_f = torch.div(ri, Wf, rounding_mode='floor')
    xb_f = ri % Wf
    pts_b = (torch.stack([xb_f, yb_f], 1).float() * patch + patch // 2).cpu().numpy()

    return pts_a, pts_b


def sp_points(sp_frontend, gray_np: np.ndarray, n_points: int):
    """
    Run SuperPoint and return top-n keypoints as (N,2) array of (x,y),
    plus the full-res heatmap (H,W) float32.
    """
    corners, _, heatmap = sp_frontend.run(gray_np)
    if corners is None or corners.shape[1] == 0:
        return np.zeros((0, 2)), np.zeros(gray_np.shape, dtype=np.float32)
    # corners: (3, N) sorted by confidence descending → take top-n
    k    = min(n_points, corners.shape[1])
    pts  = corners[:2, :k].T   # (k, 2) as (x, y)
    heat = heatmap if heatmap is not None else np.zeros(gray_np.shape, dtype=np.float32)
    # normalise heatmap to [0,1]
    mx = heat.max()
    if mx > 0:
        heat = heat / mx
    return pts, heat


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('scene',          type=Path)
    parser.add_argument('ace_head',       type=Path)
    parser.add_argument('sampler_path',   type=Path)
    parser.add_argument('--encoder_path', type=Path, default=Path('ace_encoder_pretrained.pt'))
    parser.add_argument('--sp_weights',   type=Path, default=Path('superpoint_v1.pth'),
                        help='SuperPoint weights; set to empty string to skip SP panels')
    parser.add_argument('--out_dir',      type=Path, default=Path('ace_sampler/04_evaluation/vis'))
    parser.add_argument('--split',        type=str,  default='test', choices=['train', 'test'])
    parser.add_argument('--n_images',     type=int,  default=20)
    parser.add_argument('--samples_per_image', type=int,   default=1024)
    parser.add_argument('--sampler_ratio',     type=float, default=0.7)
    parser.add_argument('--sampler_nms_size',  type=int,   default=5,
                        help='NMS window for SamplerNet (1=disabled)')
    parser.add_argument('--image_resolution', type=int,   default=480)
    parser.add_argument('--device',       type=str,  default='cuda:0')
    opt = parser.parse_args()

    device = torch.device(opt.device)
    opt.out_dir.mkdir(parents=True, exist_ok=True)

    _logger.info('Loading ACE regressor...')
    reg     = load_regressor(opt.encoder_path, opt.ace_head, device)
    _logger.info('Loading SamplerNet...')
    sampler = load_sampler(str(opt.sampler_path), device)

    use_sp = opt.sp_weights and Path(opt.sp_weights).exists()
    sp = None
    if use_sp:
        _logger.info(f'Loading SuperPoint from {opt.sp_weights}...')
        sp = load_superpoint(opt.sp_weights, device)
    else:
        _logger.warning('SuperPoint weights not found — SP panels will be skipped.')

    patch = reg.OUTPUT_SUBSAMPLE
    spi   = opt.samples_per_image
    na    = int(spi * opt.sampler_ratio)
    nb    = spi - na

    rgb_dir = opt.scene / opt.split / 'rgb'
    images  = sorted(rgb_dir.glob('*.png'))[:opt.n_images]
    _logger.info(f'Visualizing {len(images)} images from {rgb_dir}')

    for img_path in images:
        tensor, gray_np, rgb = load_image(img_path, opt.image_resolution)
        tensor = tensor.to(device)

        # ── SamplerNet ────────────────────────────────────────────────────────
        with torch.no_grad():
            feats = reg.get_features(tensor)            # (1,C,Hf,Wf)
            conf  = sampler(feats.float()).squeeze()     # (Hf,Wf)
        conf = conf.view(*conf.shape[-2:])              # ensure 2D

        conf_np = conf.cpu().float().numpy()
        pts_a, pts_b = sampler_points(conf, patch, na, nb, device,
                                       nms_size=opt.sampler_nms_size)

        # ── SuperPoint ────────────────────────────────────────────────────────
        if use_sp:
            sp_pts, sp_heat = sp_points(sp, gray_np, spi)

        # ── render panels ─────────────────────────────────────────────────────
        p_rgb      = labeled_panel(rgb, 'RGB')
        p_conf     = labeled_panel(heatmap_overlay(conf_np, rgb), 'SamplerNet conf')
        p_samp     = labeled_panel(
            draw_two_sets(rgb, pts_a, pts_b,
                          color_a=(50, 220, 50), color_b=(220, 50, 50)),
            f'SamplerNet pts (G={na} R={nb})')

        if use_sp:
            p_sp_heat  = labeled_panel(heatmap_overlay(sp_heat, rgb, cv2.COLORMAP_HOT),
                                       'SuperPoint heatmap')
            p_sp_pts   = labeled_panel(
                draw_one_set(rgb, sp_pts, color=(255, 200, 0)),
                f'SuperPoint pts (top-{len(sp_pts)})')
            combined = make_grid(p_rgb, p_conf, p_samp, p_sp_heat, p_sp_pts)
        else:
            combined = make_grid(p_rgb, p_conf, p_samp)

        stem = img_path.stem
        out_path = opt.out_dir / f'{stem}_side.png'
        cv2.imwrite(str(out_path), cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))
        _logger.info(f'  {stem}_side.png')

    _logger.info(f'Done. Results in {opt.out_dir}')


if __name__ == '__main__':
    main()
