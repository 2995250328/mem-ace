#!/usr/bin/env python3
"""
联合对比 EUPE (ViT-B / ConvNeXt-B)、DINOv2 ViT-L/14（项目 checkpoint）、ACE 原版 FCN 编码器的 patch 级特征可视化。

支持的可视化（除 PCA→RGB 外）：
  - L2 范数：每个空间位置特征向量的模长（语义/能量直观）。
  - 余弦相似度：每个位置与「全图 patch 特征均值」的余弦相似度（越亮越接近全局平均纹理）。

默认将输入 resize 到 448×448，使 448 同时被 8、14、16 整除，便于三套骨干共用同一几何尺度。

用法示例::

    cd /home/xwh/project/ace_depth
    python compare_backbone_visualization.py \\
      --rgb-dir /mnt/storage/xwh/indoor6_ace/scene2a/test/rgb --num-samples 3

依赖：torch、torchvision、numpy、matplotlib；推荐 sklearn（与 EUPE 脚本一致）。
EUPE 仓库默认同级的 ../EUPE；权重默认可通过参数覆盖。
"""

from __future__ import annotations

import argparse
import os
import random
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms as T

# ace_depth 为当前仓库根
ACE_ROOT = Path(__file__).resolve().parent
if str(ACE_ROOT) not in sys.path:
    sys.path.insert(0, str(ACE_ROOT))

from ace_network import Encoder  # noqa: E402
from ace_network_dinov2 import DINOv2Encoder  # noqa: E402

EUPE_ROOT = Path(os.environ.get("EUPE_ROOT", ACE_ROOT.parent / "EUPE"))


# -----------------------------------------------------------------------------
# 共用：特征网格 -> 可视化
# -----------------------------------------------------------------------------


def infer_grid_hw(num_patches: int) -> tuple[int, int]:
    s = int(round(num_patches**0.5))
    if s * s == num_patches:
        return s, s
    for h in range(int(num_patches**0.5), 0, -1):
        if num_patches % h == 0:
            return num_patches // h, h
    return 1, num_patches


def patch_tokens_to_pca_rgb(feat: np.ndarray) -> np.ndarray:
    """(N, C) -> (N, 3) in [0, 1]，z-score -> PCA-3 -> 各通道 min-max。"""
    x = feat.astype(np.float64)
    mean = x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True) + 1e-8
    x = (x - mean) / std
    try:
        from sklearn.decomposition import PCA  # type: ignore[import-not-found]

        y = PCA(n_components=3, svd_solver="full").fit_transform(x)
    except ImportError:
        _, _, vh = np.linalg.svd(x, full_matrices=False)
        y = x @ vh[:3].T
    rgb = np.zeros_like(y)
    for k in range(3):
        ch = y[:, k]
        lo, hi = float(ch.min()), float(ch.max())
        rgb[:, k] = (ch - lo) / (hi - lo + 1e-8)
    return np.clip(rgb, 0.0, 1.0)


def _to_map_1ch(values: np.ndarray, h: int, w: int) -> np.ndarray:
    v = values.reshape(h, w)
    lo, hi = float(v.min()), float(v.max())
    if hi - lo < 1e-12:
        return np.zeros_like(v)
    return (v - lo) / (hi - lo)


def spatial_tokens_l2(feat: np.ndarray, h: int, w: int) -> np.ndarray:
    n = np.linalg.norm(feat, axis=1)
    return _to_map_1ch(n, h, w)


def spatial_tokens_cosine_to_global_mean(feat: np.ndarray, h: int, w: int) -> np.ndarray:
    t = torch.from_numpy(feat.astype(np.float32))
    g = t.mean(dim=0, keepdim=True)
    t_n = F.normalize(t, dim=1)
    g_n = F.normalize(g, dim=1)
    sim = (t_n * g_n).sum(dim=1).numpy()
    return _to_map_1ch(sim, h, w)


def upsample_map(
    grid_hw: np.ndarray,
    out_hw: tuple[int, int],
    *,
    rgb: bool,
    mode: str = "nearest",
) -> np.ndarray:
    """grid: (H,W) or (H,W,3)"""
    t = torch.from_numpy(grid_hw.astype(np.float32))
    if rgb:
        t = t.permute(2, 0, 1).unsqueeze(0)
    else:
        t = t.unsqueeze(0).unsqueeze(0)
    if mode == "nearest":
        t = F.interpolate(t, size=out_hw, mode="nearest")
    else:
        t = F.interpolate(t, size=out_hw, mode="bilinear", align_corners=False)
    t = t.squeeze(0).cpu().numpy()
    if rgb:
        return np.clip(np.transpose(t, (1, 2, 0)), 0.0, 1.0)
    return t[0]


def colorize_gray(g: np.ndarray, cmap_name: str = "magma") -> np.ndarray:
    return plt.get_cmap(cmap_name)(g)[..., :3]


# -----------------------------------------------------------------------------
# 模型前向：统一起见得到 (N, C) 与 (H, W)
# -----------------------------------------------------------------------------


@torch.inference_mode()
def eupe_patch_tokens(model: torch.nn.Module, x: torch.Tensor) -> np.ndarray:
    out = model.forward_features(x)
    return out["x_norm_patchtokens"][0].float().cpu().numpy()


@torch.inference_mode()
def dinov2_patch_tokens(enc: DINOv2Encoder, x: torch.Tensor) -> np.ndarray:
    fd = enc.dinov2.forward_features(x)
    return fd["x_norm_patchtokens"][0].float().cpu().numpy()


@torch.inference_mode()
def ace_fcn_tokens(enc: Encoder, x_gray: torch.Tensor) -> np.ndarray:
    """x_gray: [1,1,H,W]"""
    fm = enc(x_gray)  # 1, C, H', W'
    b, c, h, w = fm.shape
    return fm.view(c, h * w).transpose(0, 1).contiguous().float().cpu().numpy()


def make_imagenet_transform(res: int) -> T.Compose:
    return T.Compose(
        [
            T.Resize((res, res)),
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )


def rgb_to_gray_luminance(t_rgb: torch.Tensor) -> torch.Tensor:
    """[B,3,H,W] -> [B,1,H,W]（与常见 luminance 一致）"""
    r, g, b = t_rgb[:, 0:1], t_rgb[:, 1:2], t_rgb[:, 2:3]
    return 0.299 * r + 0.587 * g + 0.114 * b


# -----------------------------------------------------------------------------
# 主图：一行模型 PCA + 可选第二行 L2 / 第三行 cosine
# -----------------------------------------------------------------------------


def build_model_column_panels(
    display_rgb: np.ndarray,
    feats: list[tuple[str, np.ndarray]],
    viz_modes: list[str],
    pca_interp: str,
) -> plt.Figure:
    """
    feats: list of (name, (N, C))
    viz_modes: 子集 of 'pca', 'norm', 'cosine'
    """
    n_models = len(feats)
    n_modes = len(viz_modes)
    cols = 1 + n_models
    fig, axes = plt.subplots(n_modes, cols, figsize=(3.2 * cols, 3.0 * n_modes), constrained_layout=True)
    if n_modes == 1:
        axes = np.array([axes])

    img_h, img_w = display_rgb.shape[0], display_rgb.shape[1]
    out_hw = (img_h, img_w)

    for mi, mode in enumerate(viz_modes):
        axes[mi, 0].imshow(display_rgb)
        axes[mi, 0].set_title("Input")
        axes[mi, 0].axis("off")

        for j, (name, feat) in enumerate(feats):
            n, c = feat.shape
            hi, wi = infer_grid_hw(n)
            ax = axes[mi, j + 1]
            if mode == "pca":
                pca = patch_tokens_to_pca_rgb(feat).reshape(hi, wi, 3)
                up = upsample_map(pca, out_hw, rgb=True, mode=pca_interp)
                ax.imshow(up)
                ax.set_title(f"{name}\n(PCA→RGB)")
            elif mode == "norm":
                g = spatial_tokens_l2(feat, hi, wi)
                up = upsample_map(g, out_hw, rgb=False, mode="nearest")
                ax.imshow(colorize_gray(up))
                ax.set_title(f"{name}\n‖f‖₂")
            elif mode == "cosine":
                g = spatial_tokens_cosine_to_global_mean(feat, hi, wi)
                up = upsample_map(g, out_hw, rgb=False, mode="nearest")
                ax.imshow(colorize_gray(up, "plasma"))
                ax.set_title(f"{name}\ncos→μ_patch")
            ax.axis("off")

    fig.suptitle(
        "Backbone comparison — EUPE / DINOv2-L14 / ACE-FCN (same spatial scale when possible)",
        fontsize=11,
    )
    return fig


# -----------------------------------------------------------------------------
# 采样与入口
# -----------------------------------------------------------------------------

_IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".JPG", ".JPEG", ".PNG"}


def _is_rgb_frame_file(p: Path) -> bool:
    """Skip 7Scenes depth/label maps that share extensions like .png."""
    if p.suffix not in _IMAGE_EXT:
        return False
    n = p.name.lower()
    if ".depth." in n or ".label." in n:
        return False
    return True


def collect_images(root: Path, recursive: bool) -> list[Path]:
    it = root.rglob("*") if recursive else root.iterdir()
    out = [p for p in it if p.is_file() and _is_rgb_frame_file(p)]
    return sorted(out, key=lambda p: p.as_posix().lower())


def sample_paths(paths: list[Path], n: int, mode: str, seed: int | None) -> list[Path]:
    if n <= 0:
        return []
    if len(paths) <= n:
        return list(paths)
    if mode == "first":
        return paths[:n]
    rng = random.Random(seed)
    return rng.sample(paths, n)


def safe_stem(path: Path) -> str:
    s = re.sub(r"[^\w.\-]+", "_", path.stem, flags=re.UNICODE)
    return s[:100]


def load_eupe(eupe_repo: Path, name: str, weights: Path, device: torch.device):
    m = torch.hub.load(str(eupe_repo), name, source="local", pretrained=True, weights=str(weights))
    return m.to(device).eval()


def main() -> None:
    p = argparse.ArgumentParser(description="Compare EUPE / DINOv2 / ACE FCN backbone visualizations")
    p.add_argument("--image", type=str, default=None)
    p.add_argument("--rgb-dir", type=str, default=None)
    p.add_argument("--recursive", action="store_true")
    p.add_argument("--num-samples", type=int, default=3)
    p.add_argument("--sample-mode", choices=("first", "random"), default="first")
    p.add_argument("--sample-seed", type=int, default=0)
    p.add_argument(
        "--out-dir",
        type=str,
        default=str(ACE_ROOT / "outputs" / "backbone_compare"),
    )
    p.add_argument("--resize", type=int, default=448, help="448 可被 8、14、16 整除，便于对齐")
    p.add_argument("--pca-interp", choices=("nearest", "bilinear"), default="nearest")
    p.add_argument(
        "--viz",
        type=str,
        default="pca,norm,cosine",
        help="comma: pca, norm, cosine",
    )
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--eupe-root", type=str, default=str(EUPE_ROOT))
    p.add_argument("--eupe-vit-ckpt", type=str, default=str(EUPE_ROOT / "checkpoints" / "EUPE-ViT-B.pt"))
    p.add_argument("--eupe-cnx-ckpt", type=str, default=str(EUPE_ROOT / "checkpoints" / "EUPE-ConvNeXt-B.pt"))
    p.add_argument(
        "--dinov2-ckpt",
        type=str,
        default=str(ACE_ROOT / "checkpoints" / "dinov2_vitl14_pretrain.pth"),
    )
    p.add_argument(
        "--ace-encoder",
        type=str,
        default=str(ACE_ROOT / "ace_encoder_pretrained.pt"),
    )
    p.add_argument("--no-eupe-vit", action="store_true")
    p.add_argument("--no-eupe-cnx", action="store_true")
    p.add_argument("--no-dinov2", action="store_true")
    p.add_argument("--no-ace", action="store_true")
    args = p.parse_args()

    viz_modes = [x.strip() for x in args.viz.split(",") if x.strip()]
    for v in viz_modes:
        if v not in {"pca", "norm", "cosine"}:
            raise SystemExit(f"Unknown viz mode: {v}")

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    eupe_repo = Path(args.eupe_root)
    tfm = make_imagenet_transform(args.resize)

    # ---------- load encoders ----------
    vit = cnx = dino = ace_enc = None
    if not args.no_eupe_vit:
        wv = Path(args.eupe_vit_ckpt)
        if not wv.is_file():
            print(f"[skip] EUPE ViT weights missing: {wv}")
        else:
            vit = load_eupe(eupe_repo, "eupe_vitb16", wv, device)
    if not args.no_eupe_cnx:
        wc = Path(args.eupe_cnx_ckpt)
        if not wc.is_file():
            print(f"[skip] EUPE ConvNeXt weights missing: {wc}")
        else:
            cnx = load_eupe(eupe_repo, "eupe_convnext_base", wc, device)
    if not args.no_dinov2:
        wd = Path(args.dinov2_ckpt)
        if not wd.is_file():
            print(f"[skip] DINOv2 checkpoint missing: {wd}")
        else:
            dino = DINOv2Encoder(wd, out_channels=1024, freeze_backbone=True, use_local_dinov2=True)
            dino = dino.to(device).eval()
    if not args.no_ace:
        wa = Path(args.ace_encoder)
        if not wa.is_file():
            print(f"[skip] ACE encoder missing: {wa}")
        else:
            sd = torch.load(wa, map_location="cpu", weights_only=True)
            if isinstance(sd, dict) and "model" in sd:
                sd = sd["model"]
            nc = sd["res2_conv3.weight"].shape[0]
            ace_enc = Encoder(out_channels=nc)
            ace_enc.load_state_dict(sd)
            ace_enc = ace_enc.to(device).eval()

    if vit is None and cnx is None and dino is None and ace_enc is None:
        raise SystemExit("No backbone loaded; check checkpoints and --no-* flags.")

    def process_one(pil: Image.Image, tag: str) -> None:
        x = tfm(pil).unsqueeze(0).to(device)
        disp = np.array(pil.resize((args.resize, args.resize))).astype(np.float32) / 255.0

        feats: list[tuple[str, np.ndarray]] = []
        if vit is not None:
            feats.append(("EUPE ViT-B", eupe_patch_tokens(vit, x)))
        if cnx is not None:
            feats.append(("EUPE ConvNeXt-B", eupe_patch_tokens(cnx, x)))
        if dino is not None:
            feats.append(("DINOv2-L/14", dinov2_patch_tokens(dino, x)))
        if ace_enc is not None:
            xg = rgb_to_gray_luminance(x)
            feats.append(("ACE FCN", ace_fcn_tokens(ace_enc, xg)))

        fig = build_model_column_panels(disp, feats, viz_modes=viz_modes, pca_interp=args.pca_interp)
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{tag}_compare.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"Saved {out_path}")

    # ---------- run ----------
    if args.rgb_dir:
        root = Path(args.rgb_dir).expanduser().resolve()
        imgs = collect_images(root, args.recursive)
        picked = sample_paths(imgs, args.num_samples, args.sample_mode, args.sample_seed)
        if not picked:
            raise SystemExit("No images found.")
        print(f"[batch] {len(picked)} image(s) -> {args.out_dir}")
        for i, path in enumerate(picked):
            pil = Image.open(path).convert("RGB")
            process_one(pil, f"{i:02d}_{safe_stem(path)}")
        return

    if args.image:
        pil = Image.open(args.image).convert("RGB")
        process_one(pil, "single")
        return

    # demo
    pil = Image.effect_mandelbrot((args.resize, args.resize), (-3, -2.25, 0, 0.75), 30).convert("RGB")
    process_one(pil, "synthetic")


if __name__ == "__main__":
    main()
