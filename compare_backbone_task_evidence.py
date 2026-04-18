#!/usr/bin/env python3
"""
Task-oriented backbone evidence plots for scene coordinate regression.

Adds two more qualitative views on top of single-image PCA:
1) local distinctiveness maps (where local features differ from neighbors),
2) cross-view similarity heatmaps between nearby frames.

These are more relevant to localization than PCA alone because scene coordinate
regression needs both spatially distinctive regions and stable cross-view
matching cues.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

try:
    import cv2  # type: ignore
except ImportError:  # pragma: no cover
    cv2 = None

ACE_ROOT = Path(__file__).resolve().parent
if str(ACE_ROOT) not in sys.path:
    sys.path.insert(0, str(ACE_ROOT))

from ace_network import Encoder  # noqa: E402
from ace_network_dinov2 import DINOv2Encoder  # noqa: E402
from compare_backbone_visualization import (  # noqa: E402
    ACE_ROOT as _ACE_ROOT_FROM_HELPER,
    EUPE_ROOT,
    ace_fcn_tokens,
    collect_images,
    colorize_gray,
    dinov2_patch_tokens,
    eupe_patch_tokens,
    infer_grid_hw,
    make_imagenet_transform,
    rgb_to_gray_luminance,
    safe_stem,
    sample_paths,
    upsample_map,
)

assert ACE_ROOT == _ACE_ROOT_FROM_HELPER


def load_eupe(eupe_repo: Path, name: str, weights: Path, device: torch.device):
    model = torch.hub.load(str(eupe_repo), name, source="local", pretrained=True, weights=str(weights))
    return model.to(device).eval()


def normalized_tokens(feat: np.ndarray) -> np.ndarray:
    x = feat.astype(np.float32)
    denom = np.linalg.norm(x, axis=1, keepdims=True) + 1e-8
    return x / denom


def local_distinctiveness_map(feat: np.ndarray) -> np.ndarray:
    """
    Mean cosine dissimilarity to 4-neighbors on the native feature grid.
    High values indicate locally distinctive regions, often better anchors for localization.
    """
    n, c = feat.shape
    h, w = infer_grid_hw(n)
    x = normalized_tokens(feat).reshape(h, w, c)
    acc = np.zeros((h, w), dtype=np.float32)
    cnt = np.zeros((h, w), dtype=np.float32)

    def add_pair(a_y, a_x, b_y, b_x):
        cos = np.sum(x[a_y, a_x] * x[b_y, b_x], axis=-1)
        dis = 1.0 - cos
        acc[a_y, a_x] += dis
        acc[b_y, b_x] += dis
        cnt[a_y, a_x] += 1.0
        cnt[b_y, b_x] += 1.0

    add_pair(slice(None, -1), slice(None), slice(1, None), slice(None))
    add_pair(slice(None), slice(None, -1), slice(None), slice(1, None))

    out = acc / np.maximum(cnt, 1.0)
    lo, hi = float(out.min()), float(out.max())
    if hi - lo < 1e-12:
        return np.zeros_like(out)
    return (out - lo) / (hi - lo)


def select_anchor_points(
    display_rgb: np.ndarray,
    num_anchors: int,
    min_distance: int,
) -> list[tuple[int, int]]:
    """Model-independent anchors from image corners/features; fallback to regular grid."""
    h, w = display_rgb.shape[:2]
    gray = (display_rgb.mean(axis=2) * 255.0).astype(np.uint8)
    anchors: list[tuple[int, int]] = []

    if cv2 is not None:
        pts = cv2.goodFeaturesToTrack(
            gray,
            maxCorners=num_anchors,
            qualityLevel=0.01,
            minDistance=float(min_distance),
            blockSize=7,
            useHarrisDetector=False,
        )
        if pts is not None:
            for p in pts.reshape(-1, 2):
                x, y = int(round(float(p[0]))), int(round(float(p[1])))
                x = int(np.clip(x, 0, w - 1))
                y = int(np.clip(y, 0, h - 1))
                anchors.append((x, y))

    if len(anchors) >= num_anchors:
        return anchors[:num_anchors]

    # Fallback: 2x2-ish regular grid.
    xs = np.linspace(w * 0.25, w * 0.75, num=max(2, math.ceil(num_anchors / 2)))
    ys = np.linspace(h * 0.25, h * 0.75, num=2)
    for yy in ys:
        for xx in xs:
            p = (int(round(xx)), int(round(yy)))
            if p not in anchors:
                anchors.append(p)
            if len(anchors) >= num_anchors:
                return anchors[:num_anchors]
    return anchors[:num_anchors]


def pixel_to_patch_xy(px: int, py: int, img_w: int, img_h: int, grid_w: int, grid_h: int) -> tuple[int, int]:
    gx = int(np.clip(px / max(img_w, 1) * grid_w, 0, grid_w - 1))
    gy = int(np.clip(py / max(img_h, 1) * grid_h, 0, grid_h - 1))
    return gx, gy


def patch_to_pixel_xy(gx: int, gy: int, img_w: int, img_h: int, grid_w: int, grid_h: int) -> tuple[float, float]:
    x = (gx + 0.5) * img_w / grid_w
    y = (gy + 0.5) * img_h / grid_h
    return x, y


def anchor_similarity_maps(
    feat_a: np.ndarray,
    feat_b: np.ndarray,
    anchors_px: list[tuple[int, int]],
    img_hw: tuple[int, int],
) -> tuple[list[np.ndarray], list[tuple[float, float]], list[float], list[float]]:
    """
    For each anchor in frame A, compute cosine similarity over all patches in frame B.
    Returns upsample-ready heatmaps on B-native grid, best-match pixel centers, top1 scores, and top1-top2 margins.
    """
    img_h, img_w = img_hw
    n_a, c = feat_a.shape
    n_b, _ = feat_b.shape
    h_a, w_a = infer_grid_hw(n_a)
    h_b, w_b = infer_grid_hw(n_b)
    fa = normalized_tokens(feat_a).reshape(h_a, w_a, c)
    fb = normalized_tokens(feat_b).reshape(h_b * w_b, c)

    maps = []
    best_xy = []
    top1_scores = []
    margins = []

    for px, py in anchors_px:
        gx, gy = pixel_to_patch_xy(px, py, img_w, img_h, w_a, h_a)
        query = fa[gy, gx]
        sim = fb @ query
        sim_grid = sim.reshape(h_b, w_b)
        flat = sim.reshape(-1)
        top1 = float(flat.max())
        if flat.size >= 2:
            idx2 = np.argpartition(flat, -2)[-2:]
            vals = np.sort(flat[idx2])
            margin = float(vals[-1] - vals[-2])
        else:
            margin = 0.0
        best_idx = int(flat.argmax())
        by, bx = divmod(best_idx, w_b)
        best_xy.append(patch_to_pixel_xy(bx, by, img_w, img_h, w_b, h_b))
        maps.append(sim_grid)
        top1_scores.append(top1)
        margins.append(margin)
    return maps, best_xy, top1_scores, margins


def normalize_map(m: np.ndarray) -> np.ndarray:
    lo, hi = float(m.min()), float(m.max())
    if hi - lo < 1e-12:
        return np.zeros_like(m)
    return (m - lo) / (hi - lo)


def build_distinctiveness_figure(
    display_rgb: np.ndarray,
    model_feats: list[tuple[str, np.ndarray]],
    title_suffix: str,
) -> plt.Figure:
    cols = 1 + len(model_feats)
    fig, axes = plt.subplots(1, cols, figsize=(3.2 * cols, 3.4), constrained_layout=True)
    axes[0].imshow(display_rgb)
    axes[0].set_title("Input")
    axes[0].axis("off")

    out_hw = (display_rgb.shape[0], display_rgb.shape[1])
    for i, (name, feat) in enumerate(model_feats, start=1):
        d = local_distinctiveness_map(feat)
        up = upsample_map(d, out_hw, rgb=False, mode="nearest")
        axes[i].imshow(colorize_gray(up, "magma"))
        axes[i].set_title(f"{name}\nlocal distinctiveness")
        axes[i].axis("off")

    fig.suptitle("Localization evidence — local distinctiveness" + (f" | {title_suffix}" if title_suffix else ""))
    return fig


def build_match_figure(
    display_a: np.ndarray,
    display_b: np.ndarray,
    anchors_px: list[tuple[int, int]],
    model_feats: list[tuple[str, np.ndarray, np.ndarray]],
    title_suffix: str,
) -> tuple[plt.Figure, list[str]]:
    """
    Rows=model, cols=2+K:
      col0: frame A with anchors
      col1: frame B with best-match points
      col2..: similarity heatmaps on frame B for each anchor
    """
    k = len(anchors_px)
    rows = len(model_feats)
    cols = 2 + k
    fig, axes = plt.subplots(rows, cols, figsize=(3.0 * cols, 2.8 * rows), constrained_layout=True)
    if rows == 1:
        axes = np.array([axes])

    colors = ["tab:red", "tab:blue", "tab:green", "tab:orange", "tab:purple", "tab:brown"]
    summary_lines: list[str] = []
    out_hw = (display_b.shape[0], display_b.shape[1])

    for r, (name, feat_a, feat_b) in enumerate(model_feats):
        sim_maps, best_xy, top1_scores, margins = anchor_similarity_maps(
            feat_a, feat_b, anchors_px, img_hw=(display_a.shape[0], display_a.shape[1])
        )

        ax0 = axes[r, 0]
        ax0.imshow(display_a)
        for i, (x, y) in enumerate(anchors_px):
            c = colors[i % len(colors)]
            ax0.scatter([x], [y], s=30, c=c)
            ax0.text(x + 4, y + 4, str(i + 1), color=c, fontsize=8, weight="bold")
        ax0.set_title(f"{name}\nframe A anchors")
        ax0.axis("off")

        ax1 = axes[r, 1]
        ax1.imshow(display_b)
        for i, (x, y) in enumerate(best_xy):
            c = colors[i % len(colors)]
            ax1.scatter([x], [y], s=30, c=c)
            ax1.text(x + 4, y + 4, str(i + 1), color=c, fontsize=8, weight="bold")
        ax1.set_title(f"{name}\nframe B best matches")
        ax1.axis("off")

        for i, sim_grid in enumerate(sim_maps):
            ax = axes[r, 2 + i]
            norm = normalize_map(sim_grid)
            up = upsample_map(norm, out_hw, rgb=False, mode="bilinear")
            ax.imshow(colorize_gray(up, "plasma"))
            bx, by = best_xy[i]
            c = colors[i % len(colors)]
            ax.scatter([bx], [by], s=25, c=c)
            ax.set_title(f"anchor {i+1}\ns={top1_scores[i]:.3f} m={margins[i]:.3f}")
            ax.axis("off")

        summary_lines.append(
            f"{name}\tavg_top1={np.mean(top1_scores):.4f}\tavg_margin={np.mean(margins):.4f}"
        )

    fig.suptitle("Localization evidence — cross-view cosine matching" + (f" | {title_suffix}" if title_suffix else ""))
    return fig, summary_lines


def load_models(args, device: torch.device):
    eupe_repo = Path(args.eupe_root)
    models = []

    if not args.no_eupe_vit:
        w = Path(args.eupe_vit_ckpt)
        if w.is_file():
            models.append(("EUPE ViT-B", load_eupe(eupe_repo, "eupe_vitb16", w, device), "eupe_vit"))
    if not args.no_eupe_cnx:
        w = Path(args.eupe_cnx_ckpt)
        if w.is_file():
            models.append(("EUPE ConvNeXt-B", load_eupe(eupe_repo, "eupe_convnext_base", w, device), "eupe_cnx"))
    if not args.no_dinov2:
        w = Path(args.dinov2_ckpt)
        if w.is_file():
            model = DINOv2Encoder(w, out_channels=1024, freeze_backbone=True, use_local_dinov2=True).to(device).eval()
            models.append(("DINOv2-L/14", model, "dino"))
    if not args.no_ace:
        w = Path(args.ace_encoder)
        if w.is_file():
            state = torch.load(w, map_location="cpu", weights_only=True)
            if isinstance(state, dict) and "model" in state:
                state = state["model"]
            out_channels = state["res2_conv3.weight"].shape[0]
            model = Encoder(out_channels=out_channels)
            model.load_state_dict(state)
            model = model.to(device).eval()
            models.append(("ACE FCN", model, "ace"))

    return models


def get_features_for_model(kind: str, model, x_rgb: torch.Tensor, x_gray: torch.Tensor) -> np.ndarray:
    if kind == "eupe_vit" or kind == "eupe_cnx":
        return eupe_patch_tokens(model, x_rgb)
    if kind == "dino":
        return dinov2_patch_tokens(model, x_rgb)
    if kind == "ace":
        return ace_fcn_tokens(model, x_gray)
    raise ValueError(kind)


def save_text(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Task-oriented backbone evidence plots for localization")
    parser.add_argument("--rgb-dir", type=str, required=True)
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--num-pairs", type=int, default=2)
    parser.add_argument("--pair-gap", type=int, default=1, help="Pair frame i with i+gap in sorted order")
    parser.add_argument("--sample-mode", choices=("first", "random"), default="first")
    parser.add_argument("--sample-seed", type=int, default=0)
    parser.add_argument("--num-anchors", type=int, default=4)
    parser.add_argument("--resize", type=int, default=448)
    parser.add_argument("--pca-interp", choices=("nearest", "bilinear"), default="nearest")
    parser.add_argument("--out-dir", type=str, default=str(ACE_ROOT / "outputs" / "backbone_task_evidence"))
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--eupe-root", type=str, default=str(EUPE_ROOT))
    parser.add_argument("--eupe-vit-ckpt", type=str, default=str(EUPE_ROOT / "checkpoints" / "EUPE-ViT-B.pt"))
    parser.add_argument("--eupe-cnx-ckpt", type=str, default=str(EUPE_ROOT / "checkpoints" / "EUPE-ConvNeXt-B.pt"))
    parser.add_argument("--dinov2-ckpt", type=str, default=str(ACE_ROOT / "checkpoints" / "dinov2_vitl14_pretrain.pth"))
    parser.add_argument("--ace-encoder", type=str, default=str(ACE_ROOT / "ace_encoder_pretrained.pt"))
    parser.add_argument("--no-eupe-vit", action="store_true")
    parser.add_argument("--no-eupe-cnx", action="store_true")
    parser.add_argument("--no-dinov2", action="store_true")
    parser.add_argument("--no-ace", action="store_true")
    args = parser.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    root = Path(args.rgb_dir).expanduser().resolve()
    all_imgs = collect_images(root, args.recursive)
    if len(all_imgs) <= args.pair_gap:
        raise SystemExit(f"Not enough images under {root} for pair_gap={args.pair_gap}")

    candidate_starts = all_imgs[: -args.pair_gap]
    picked_starts = sample_paths(candidate_starts, args.num_pairs, args.sample_mode, args.sample_seed)
    pairs = [(p, all_imgs[all_imgs.index(p) + args.pair_gap]) for p in picked_starts]

    models = load_models(args, device)
    if not models:
        raise SystemExit("No models loaded; check checkpoints and --no-* flags.")

    tfm = make_imagenet_transform(args.resize)
    min_distance = max(24, args.resize // 7)

    for i, (path_a, path_b) in enumerate(pairs):
        pil_a = Image.open(path_a).convert("RGB")
        pil_b = Image.open(path_b).convert("RGB")

        x_a = tfm(pil_a).unsqueeze(0).to(device)
        x_b = tfm(pil_b).unsqueeze(0).to(device)
        xg_a = rgb_to_gray_luminance(x_a)
        xg_b = rgb_to_gray_luminance(x_b)

        display_a = np.array(pil_a.resize((args.resize, args.resize))).astype(np.float32) / 255.0
        display_b = np.array(pil_b.resize((args.resize, args.resize))).astype(np.float32) / 255.0
        anchors = select_anchor_points(display_a, args.num_anchors, min_distance=min_distance)

        per_model_single = []
        per_model_pair = []
        for name, model, kind in models:
            feat_a = get_features_for_model(kind, model, x_a, xg_a)
            feat_b = get_features_for_model(kind, model, x_b, xg_b)
            per_model_single.append((name, feat_a))
            per_model_pair.append((name, feat_a, feat_b))

        tag = f"{i:02d}_{safe_stem(path_a)}__to__{safe_stem(path_b)}"

        fig1 = build_distinctiveness_figure(display_a, per_model_single, title_suffix=tag)
        p1 = out_dir / f"{tag}_distinctiveness.png"
        fig1.savefig(p1, dpi=150)
        plt.close(fig1)

        fig2, lines = build_match_figure(display_a, display_b, anchors, per_model_pair, title_suffix=tag)
        p2 = out_dir / f"{tag}_cross_view_match.png"
        fig2.savefig(p2, dpi=150)
        plt.close(fig2)

        txt_path = out_dir / f"{tag}_summary.txt"
        save_text(
            txt_path,
            [
                f"frame_a\t{path_a}",
                f"frame_b\t{path_b}",
                "note\tQualitative evidence only; no GT correspondences used here.",
                *lines,
            ],
        )
        print(f"[{i+1}/{len(pairs)}] saved {p1.name}, {p2.name}, {txt_path.name}")


if __name__ == "__main__":
    main()
