#!/usr/bin/env python3
# Copyright © Niantic, Inc. 2022.

import argparse
import logging
import random
import time
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader

import dsacstar
from ace_network import Regressor
from dataset import CamLocDataset
from ace_util import get_pixel_grid, to_homogeneous

_logger = logging.getLogger(__name__)

# --- 硬编码的目标帧列表 (默认模式使用) ---
TARGET_FRAMES = [
    "seq-01-frame-000511", "seq-02-frame-000706", "seq-04-frame-000852", "seq-02-frame-000517",
    "seq-02-frame-000348", "seq-02-frame-000974", "seq-06-frame-000581", "seq-01-frame-000753",
    "seq-04-frame-000907", "seq-04-frame-000677", "seq-01-frame-000956", "seq-02-frame-000625",
    "seq-06-frame-000661", "seq-02-frame-000784", "seq-02-frame-000396", "seq-04-frame-000511",
    "seq-01-frame-000342", "seq-04-frame-000142", "seq-02-frame-000431", "seq-06-frame-000941"
]

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def load_and_resize_image_from_disk(path, target_height):
    """
    从磁盘直接读取图片，并调整尺寸以匹配网络输入比例。
    """
    img = cv2.imread(str(path))
    if img is None:
        raise ValueError(f"Could not read image from {path}")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    h, w, _ = img.shape
    scale = target_height / h
    new_w = int(w * scale)
    new_h = target_height

    img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    return img_resized

def is_target_frame(file_path_str, target_list):
    """
    检查文件路径是否匹配目标列表中的任意一项。
    """
    for target in target_list:
        # 简单匹配: 如果文件名包含 target 字符串
        if target in file_path_str:
            return target

        # 结构化匹配: 处理 seq-01/frame-000511 的情况
        parts = target.split('-frame-')
        if len(parts) == 2:
            seq_part = parts[0]
            frame_part = "frame-" + parts[1]
            if seq_part in file_path_str and frame_part in file_path_str:
                return target
    return None

def main():
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(
        description='Visualize ACE: Combined Random & Specific Modes',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('scene', type=Path, help='path to a scene in the dataset folder.')
    parser.add_argument('network', type=Path, help='path to a trained network head weights.')
    parser.add_argument('--encoder_path', type=Path, default=Path(__file__).parent / "ace_encoder_pretrained.pt",
                        help='file containing pre-trained encoder weights')
    parser.add_argument('--image_resolution', type=int, default=480, help='base image resolution')

    # RANSAC
    parser.add_argument('--hypotheses', '-hyps', type=int, default=64, help='number of hypotheses')
    parser.add_argument('--threshold', '-t', type=float, default=10, help='inlier threshold in pixels')
    parser.add_argument('--inlieralpha', '-ia', type=float, default=100, help='alpha parameter for soft inlier count')
    parser.add_argument('--maxpixelerror', '-maxerrr', type=float, default=100, help='maximum reprojection error')

    # Vis Params
    parser.add_argument('--output_dir', type=Path, default='visualization_results_final', help='Folder to save visualizations')
    parser.add_argument('--vis_err_threshold', type=float, default=1.0, help='Threshold (px) for Green Points')

    # Heatmap Params
    parser.add_argument('--heatmap_max_error', type=float, default=300.0, help='Max error (px) for Heatmap (Blue limit)')
    parser.add_argument('--heatmap_alpha', type=float, default=0.3, help='Transparency of heatmap overlay')
    parser.add_argument('--heatmap_gamma', type=float, default=0.5, help='Contrast stretching gamma')

    # --- 模式切换参数 ---
    parser.add_argument('--random', action='store_true', help='如果设置，进行随机采样；否则（默认）仅处理特定的目标帧。')
    parser.add_argument('--vis_samples', type=int, default=10, help='随机模式下要采集的样本数量。')

    opt = parser.parse_args()

    device = torch.device("cuda")
    setup_seed(int(time.time()))

    opt.output_dir.mkdir(parents=True, exist_ok=True)

    # Load Model
    encoder_state_dict = torch.load(opt.encoder_path, map_location="cpu")
    head_state_dict = torch.load(opt.network, map_location="cpu")
    network = Regressor.create_from_split_state_dict(encoder_state_dict, head_state_dict)
    network = network.to(device)
    network.eval()

    # Load Dataset
    # 随机模式下 shuffle=True，否则 False 以便按顺序遍历寻找目标
    do_shuffle = opt.random
    dataset = CamLocDataset(
        opt.scene / "train",
        mode=0,
        image_height=opt.image_resolution,
        )
    loader = DataLoader(dataset, shuffle=do_shuffle, num_workers=4, batch_size=1)

    pixel_grid_2HW = get_pixel_grid(8).to(device)

    processed_count = 0

    if opt.random:
        _logger.info(f"Mode: RANDOM Sampling. Aiming for {opt.vis_samples} samples.")
    else:
        _logger.info(f"Mode: SPECIFIC TARGETS. Looking for {len(TARGET_FRAMES)} frames.")

    with torch.no_grad():
        for batch_idx, data in enumerate(loader):
            # 停止条件检查
            if opt.random:
                if processed_count >= opt.vis_samples:
                    break
            else:
                # 目标模式下，如果所有目标都找到了（可选），或者遍历完了
                if processed_count >= len(TARGET_FRAMES):
                    # 注意：如果数据集中有多个文件匹配同一个 target string，这里可能会处理更多
                    # 这里为了简单，我们不强制 break，或者您可以根据需要 break
                    pass

            # Unpack
            image_B1HW = data[1].to(device, non_blocking=True) # Gray Tensor
            gt_pose_inv_B44 = data[4].to(device, non_blocking=True)
            intrinsics_B33 = data[5].to(device, non_blocking=True)
            file_path_batch = data[-1]
            file_path = file_path_batch[0]

            # --- 筛选逻辑 ---
            target_label = ""
            if opt.random:
                # 随机模式：所有帧都是候选
                target_label = "Random Sample"
            else:
                # 特定模式：检查是否匹配
                matched_name = is_target_frame(file_path, TARGET_FRAMES)
                if matched_name is None:
                    continue # 跳过非目标帧
                target_label = matched_name

            # 1. Load Original Image from Disk
            try:
                img_vis = load_and_resize_image_from_disk(file_path, opt.image_resolution)
            except Exception as e:
                _logger.warning(f"Failed load: {e}")
                continue

            # 2. Inference
            with autocast(enabled=True):
                scene_coordinates_B3HW = network(image_B1HW)

            # RANSAC
            focal_length = intrinsics_B33[0, 0, 0].item()
            ppX, ppY = intrinsics_B33[0, 0, 2].item(), intrinsics_B33[0, 1, 2].item()
            inlier_count = dsacstar.forward_rgb(
                scene_coordinates_B3HW.float().cpu(),
                torch.zeros((4, 4)),
                opt.hypotheses, opt.threshold, focal_length, ppX, ppY, opt.inlieralpha, opt.maxpixelerror, 8
            )

            # 在随机模式下，我们可能想跳过质量太差的帧
            if opt.random and inlier_count < 100:
                continue

            processed_count += 1
            _logger.info(f"Processing [{processed_count}] | Type: {target_label} | File: {Path(file_path).name}")

            # Calc Error
            B, C, H_out, W_out = scene_coordinates_B3HW.shape
            pred_scene = to_homogeneous(scene_coordinates_B3HW.permute(0,2,3,1).flatten(0,2).unsqueeze(-1).float())
            gt_pose_exp = gt_pose_inv_B44[:, :3].unsqueeze(1).expand(B, H_out*W_out, 3, 4).reshape(-1, 3, 4)
            intrinsics_exp = intrinsics_B33.unsqueeze(1).expand(B, H_out*W_out, 3, 3).reshape(-1, 3, 3)

            pred_px = torch.bmm(intrinsics_exp, torch.bmm(gt_pose_exp, pred_scene))
            pred_px[:, 2].clamp_(min=0.1)
            pred_uv = pred_px[:, :2] / pred_px[:, 2, None]

            target_uv = pixel_grid_2HW[:, :H_out, :W_out].clone().reshape(2, -1).t()
            reprojection_error = torch.norm(pred_uv.squeeze() - target_uv, dim=1)

            # --- Visualization Generation ---

            # Heatmap Processing
            err_map_raw = reprojection_error.view(H_out, W_out).cpu().numpy()
            err_map_resized = cv2.resize(err_map_raw, (img_vis.shape[1], img_vis.shape[0]), interpolation=cv2.INTER_CUBIC)
            err_map_norm = np.clip(err_map_resized / opt.heatmap_max_error, 0, 1.0)
            err_map_gamma = np.power(err_map_norm, opt.heatmap_gamma)

            # Points
            valid_pts = target_uv[reprojection_error < opt.vis_err_threshold].cpu().numpy()

            # Plotting
            fig = plt.figure(figsize=(20, 9))
            gs = gridspec.GridSpec(2, 3, height_ratios=[6, 1], figure=fig)

            ax0 = fig.add_subplot(gs[0, 0])
            ax1 = fig.add_subplot(gs[0, 1])
            ax2 = fig.add_subplot(gs[0, 2])
            ax_legend = fig.add_subplot(gs[1, :])

            # Plot 1: Original
            ax0.imshow(img_vis)
            ax0.set_title(f"Original: {Path(file_path).name}", fontsize=14, fontweight='bold', pad=10)
            ax0.axis('off')

            # Plot 2: Points
            ax1.imshow(img_vis)
            if len(valid_pts) > 0:
                ax1.scatter(valid_pts[:,0], valid_pts[:,1], s=4, c='#00FF00', alpha=0.8)
            ax1.set_title(f"Points (Err < {opt.vis_err_threshold}px)\nInliers: {inlier_count}", fontsize=14, fontweight='bold', pad=10)
            ax1.axis('off')

            # Plot 3: Heatmap
            ax2.imshow(img_vis)
            im = ax2.imshow(err_map_gamma, cmap='jet_r', alpha=opt.heatmap_alpha, vmin=0, vmax=1.0)
            ax2.set_title(f"Confidence Heatmap", fontsize=14, fontweight='bold', pad=10)
            ax2.axis('off')

            cbar = plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
            cbar.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
            cbar.set_ticklabels(["0px", f"{opt.heatmap_max_error*0.25:.0f}", f"{opt.heatmap_max_error*0.5:.0f}", f"{opt.heatmap_max_error*0.75:.0f}", f"{opt.heatmap_max_error:.0f}+"])
            cbar.set_label('Reprojection Error (px)', fontsize=12)

            # Legend
            ax_legend.axis('off')
            legend_text = (
                f"Mode: {'RANDOM' if opt.random else 'SPECIFIC'} | Frame: {Path(file_path).name}\n"
                f"Left: Original (Disk Load). Middle: High Confidence Points (Green). Right: Heatmap (Gamma={opt.heatmap_gamma}).\n"
                f"Red = Low Error (High Confidence), Blue = High Error."
            )
            ax_legend.text(0.5, 0.5, legend_text, ha='center', va='center', fontsize=14,
                           bbox=dict(boxstyle="round,pad=1", fc="#f0f0f0", ec="black", alpha=0.3))

            plt.tight_layout()

            # 文件名：包含模式信息，避免覆盖
            prefix = "rand" if opt.random else "target"
            # 提取一个简短的文件名标识
            safe_name = Path(file_path).stem
            out_filename = f"{prefix}_{processed_count}_{safe_name}.png"

            out_path = opt.output_dir / out_filename
            plt.savefig(out_path, dpi=120)
            plt.close(fig)

    _logger.info("Done.")

if __name__ == '__main__':
    main()