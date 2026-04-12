#!/usr/bin/env python3
# test_ace_full.py
# 全功能测试脚本：支持模型结构自适应加载(3/4通道) + 不确定性过滤 + 内参融合推理 + 详细误差统计

import argparse
import logging
import math
import time
import os
from pathlib import Path
from distutils.util import strtobool

import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader

# 引入项目模块
import dsacstar
from dataset import CamLocDataset
import ace_vis_util as vutil
from ace_util import get_pixel_grid, to_homogeneous
from ace_visualizer import ACEVisualizer

# 引入重构后的网络
from ace_network_full import Regressor

_logger = logging.getLogger(__name__)


def _strtobool(x):
    return bool(strtobool(x))


def compute_reprojection_error(pred_scene_coords, gt_pose_inv, intrinsic, device, valid_mask=None):
    """
    计算回归的重投影误差 (仅用于分析回归质量，不参与 RANSAC)
    """
    c, h, w = pred_scene_coords.shape
    # 构造像素网格
    y_grid, x_grid = torch.meshgrid(torch.arange(0, h, device=device),
                                    torch.arange(0, w, device=device), indexing='ij')
    pixel_grid = torch.stack((x_grid, y_grid), dim=0).float().view(2, -1)

    # 准备预测的 3D 点
    scene_coords_flat = pred_scene_coords.view(3, -1)
    ones = torch.ones((1, h * w), device=device)
    scene_coords_homo = torch.cat((scene_coords_flat, ones), dim=0)

    # World -> Camera
    cam_coords = torch.mm(gt_pose_inv.to(device), scene_coords_homo)

    # 剔除相机背后的点
    depth = cam_coords[2, :]

    # 基础深度过滤
    mask = depth > 0.1

    # 如果传入了额外的有效性掩膜（例如不确定性过滤后的掩膜），进行合并
    if valid_mask is not None:
        mask = mask & valid_mask.view(-1)

    if mask.sum() == 0:
        return 0.0

    # 投影 Camera -> Pixel
    proj_homo = torch.mm(intrinsic.to(device), cam_coords[:3, :])
    proj_uv = proj_homo[:2] / (proj_homo[2:3] + 1e-6)

    # 计算 L2 误差
    diff = proj_uv[:, mask] - pixel_grid[:, mask]
    error = torch.norm(diff, dim=0)

    return error.mean().item()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(
        description='Test the full ACE model with Intrinsic Fusion and Uncertainty.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('scene', type=Path, help='path to scene folder')
    parser.add_argument('network', type=Path, help='path to the trained model (.pt file)')

    # [新增] 显式指定 Encoder 路径
    parser.add_argument('--encoder_path', type=Path, default=Path("ace_encoder_pretrained.pt"),
                        help='path to the pretrained encoder weights')

    # 测试参数
    parser.add_argument('--session', '-sid', default='', help='custom session name')
    parser.add_argument('--image_resolution', type=int, default=480, help='base image resolution')

    # [新增] 不确定性过滤参数
    parser.add_argument('--keep_percentile', type=float, default=0.6,
                        help='Percentage of confident points to keep if uncertainty channel exists (0.0 - 1.0). Default 0.6 (Top 60%)')

    # RANSAC 参数
    parser.add_argument('--hypotheses', '-hyps', type=int, default=64, help='RANSAC iterations')
    parser.add_argument('--threshold', '-t', type=float, default=10, help='inlier threshold (px)')
    parser.add_argument('--inlieralpha', '-ia', type=float, default=100, help='soft inlier alpha')
    parser.add_argument('--maxpixelerror', '-maxerrr', type=float, default=100,
                        help='max reprojection error for pose check')

    # 可视化参数
    parser.add_argument('--render_visualization', type=_strtobool, default=False, help='create video')
    parser.add_argument('--render_target_path', type=Path, default='renderings', help='target folder')
    parser.add_argument('--render_flipped_portrait', type=_strtobool, default=False, help='flag for portrait dataset')
    parser.add_argument('--render_sparse_queries', type=_strtobool, default=False, help='sparse queries flag')
    parser.add_argument('--render_pose_error_threshold', type=int, default=20, help='pose error threshold cm/deg')
    parser.add_argument('--render_map_depth_filter', type=int, default=10, help='clean up map points far away')
    parser.add_argument('--render_camera_z_offset', type=int, default=4, help='zoom out meters')
    parser.add_argument('--render_frame_skip', type=int, default=1, help='skip frames')

    opt = parser.parse_args()

    # Device Setup
    device = torch.device("cuda:0") # 默认改为 cuda:0，请根据实际情况调整

    # Paths & Naming
    scene_path = Path(opt.scene)
    network_path = Path(opt.network)
    model_stem = network_path.stem
    output_dir = network_path.parent

    # Output Files
    test_log_file = output_dir / f'stats_{model_stem}.txt'
    pose_log_file = output_dir / f'poses_{model_stem}.txt'
    plot_file = output_dir / f'plot_analysis_{model_stem}.png'

    # Dataset
    testset = CamLocDataset(
        scene_path / "test",
        mode=0,
        image_height=opt.image_resolution,
        )
    _logger.info(f'Test images found: {len(testset)}')
    testset_loader = DataLoader(testset, shuffle=False, num_workers=6)

    # --- Robust Model Loading Logic ---
    _logger.info(f"Loading network from: {network_path}")

    # 1. Load Head/Fusion Weights
    head_state_dict = torch.load(network_path, map_location="cpu")

    # 2. Load Encoder Weights
    if not opt.encoder_path.exists():
        _logger.warning(f"Encoder file not found at {opt.encoder_path}. Trying default path or skipping if keys exist in head dict.")

    encoder_state_dict = torch.load(opt.encoder_path, map_location="cpu")

    # 3. Create Network
    # 使用 create_from_split_state_dict，内部会自动处理通道数推断
    network = Regressor.create_from_split_state_dict(encoder_state_dict, head_state_dict)

    network = network.to(device)
    network.eval()

    # 获取融合模式
    if hasattr(network, 'intrinsics_fusion') and hasattr(network.intrinsics_fusion, 'mode'):
        fusion_mode = network.intrinsics_fusion.mode
    else:
        fusion_mode = getattr(network, 'intrinsics_fusion_mode', 'unknown')
    _logger.info(f"Fusion Mode Detected: {fusion_mode}")

    # Setup Logging
    test_log = open(test_log_file, 'w', 1)
    pose_log = open(pose_log_file, 'w', 1)

    # Metrics
    rErrs, tErrs, repro_means = [], [], []
    pct10_5, pct5, pct2, pct1 = 0, 0, 0, 0

    avg_batch_time = 0
    num_batches = 0

    # Visualizer
    ace_visualizer = None
    if opt.render_visualization:
        target_path = vutil.get_rendering_target_path(opt.render_target_path, opt.network)
        ace_visualizer = ACEVisualizer(target_path, opt.render_flipped_portrait,
                                       opt.render_map_depth_filter,
                                       reloc_vis_error_threshold=opt.render_pose_error_threshold)
        trainset = CamLocDataset(scene_path / "train", mode=0, image_height=opt.image_resolution)
        trainset_loader = DataLoader(trainset, shuffle=False, num_workers=6)
        ace_visualizer.setup_reloc_visualisation(len(testset), trainset_loader, network,
                                                 opt.render_camera_z_offset, opt.render_frame_skip)

    # --- Testing Loop ---
    _logger.info("Starting inference...")

    with torch.no_grad():
        # 注意：dataset 返回值数量可能随版本变化，这里使用通配符确保兼容性
        for data_batch in testset_loader:
            # 兼容性解包：确保取到核心数据
            image_B1HW = data_batch[1] # Tensor Image
            gt_pose_B44 = data_batch[3] # GT Pose
            gt_pose_inv_B44 = data_batch[4] # GT Pose Inv
            intrinsics_B33 = data_batch[5] # Intrinsics
            filenames = data_batch[-1] # Filenames usually last

            batch_start = time.time()

            image_B1HW = image_B1HW.to(device, non_blocking=True)
            intrinsics_B33 = intrinsics_B33.to(device, non_blocking=True)

            with autocast(enabled=True):
                # Network Inference
                prediction = network(image_B1HW) # [B, C, H, W]

            # --- [核心修改] 适配不确定性通道 ---
            # 检查输出通道数：3 (RGB/Coords) vs 4 (Coords + Uncertainty)
            B, C, H, W = prediction.shape

            if C == 4:
                # 分离坐标和不确定性
                pred_coords = prediction[:, :3, :, :]
                pred_log_var = prediction[:, 3, :, :] # log(sigma^2), 越小越确信

                # --- [策略] 百分位过滤 ---
                # 我们希望保留 log_var 最小的那部分点（最确信的点）
                # 展平 log_var 计算分位数
                flat_log_var = pred_log_var.view(B, -1) # [B, N]
                k = int(flat_log_var.shape[1] * opt.keep_percentile)

                # 找到阈值 (smallest k values)
                # kthvalue 返回 (values, indices)，我们只需要 value 作为阈值
                threshold, _ = torch.kthvalue(flat_log_var, k, dim=1, keepdim=True)
                threshold = threshold.view(B, 1, 1, 1)

                # 生成掩膜: log_var <= threshold 的是“好点”
                valid_mask = pred_log_var <= threshold # [B, 1, H, W]

                # 应用掩膜：不要设为0！设为一个极大的值 (例如 10000 米外)
                # 这样 DSAC* 的 RANSAC 会因为重投影误差过大而直接忽略它们
                scene_coordinates_B3HW = pred_coords.clone()

                # --- [修改这里] 将 0 改为 1e5 (或者 float('inf')) ---
                scene_coordinates_B3HW[~valid_mask.expand_as(scene_coordinates_B3HW)] = 100000.0

            else:
                # 标准 3 通道模式
                scene_coordinates_B3HW = prediction
                valid_mask = None # 全图有效

            # Move to CPU for RANSAC
            scene_coordinates_B3HW_cpu = scene_coordinates_B3HW.float().cpu()
            if valid_mask is not None:
                valid_mask_cpu = valid_mask.cpu()

            for i in range(image_B1HW.shape[0]):
                scene_coords_3HW = scene_coordinates_B3HW[i]  # GPU tensor (filtered)
                scene_coords_3HW_cpu = scene_coordinates_B3HW_cpu[i]  # CPU tensor (filtered)

                # 获取当前帧的 mask (如果有)
                curr_valid_mask = valid_mask[i] if valid_mask is not None else None

                gt_pose_44 = gt_pose_B44[i]
                gt_pose_inv_44 = gt_pose_inv_B44[i]
                intrinsics_33 = intrinsics_B33[i]
                frame_name = Path(filenames[i]).name

                # 1. 计算回归重投影误差 (传入 valid_mask 以只计算高置信度区域的误差)
                repro_val = compute_reprojection_error(scene_coords_3HW, gt_pose_inv_44, intrinsics_33, device, curr_valid_mask)
                repro_means.append(repro_val)

                # 2. RANSAC Pose Estimation
                # dsacstar.forward_rgb 接收 3通道坐标图。
                # 由于我们在上面已经把不确定的点置为 0 了，RANSAC 会自动忽略它们（如果不幸选中 0 点，也会因为重投影误差极大被剔除）
                focal_length = intrinsics_33[0, 0].item()
                ppX = intrinsics_33[0, 2].item()
                ppY = intrinsics_33[1, 2].item()
                out_pose = torch.zeros((4, 4))

                inlier_count = dsacstar.forward_rgb(
                    scene_coords_3HW_cpu.unsqueeze(0), # [1, 3, H, W]
                    out_pose,
                    opt.hypotheses,
                    opt.threshold,
                    focal_length,
                    ppX, ppY,
                    opt.inlieralpha,
                    opt.maxpixelerror,
                    network.OUTPUT_SUBSAMPLE,
                )

                # 3. Pose Error
                t_err = float(torch.norm(gt_pose_44[0:3, 3] - out_pose[0:3, 3]))
                gt_R = gt_pose_44[0:3, 0:3].numpy()
                out_R = out_pose[0:3, 0:3].numpy()

                # 安全计算旋转误差
                try:
                    r_err = np.linalg.norm(cv2.Rodrigues(np.matmul(out_R, np.transpose(gt_R)))[0]) * 180 / math.pi
                except Exception:
                    r_err = 180.0 # 异常时给最大误差

                _logger.info(f"[{frame_name}] R: {r_err:.2f}, T: {t_err * 100:.1f}cm, Repro: {repro_val:.2f}px")

                # Visualization Update
                if ace_visualizer is not None:
                    ace_visualizer.render_reloc_frame(
                        query_pose=gt_pose_44.numpy(), query_file=filenames[i],
                        est_pose=out_pose.numpy(), est_error=max(r_err, t_err * 100),
                        sparse_query=opt.render_sparse_queries)

                # Stats Accumulation
                rErrs.append(r_err)
                tErrs.append(t_err * 100)

                if r_err < 5 and t_err < 0.1: pct10_5 += 1
                if r_err < 5 and t_err < 0.05: pct5 += 1
                if r_err < 2 and t_err < 0.02: pct2 += 1
                if r_err < 1 and t_err < 0.01: pct1 += 1

                # Log Pose
                out_pose_inv = out_pose.inverse()
                t = out_pose_inv[0:3, 3]
                rot, _ = cv2.Rodrigues(out_pose_inv[0:3, 0:3].numpy())
                angle = np.linalg.norm(rot)
                axis = rot / (angle + 1e-6)
                q_w = math.cos(angle * 0.5)
                q_xyz = math.sin(angle * 0.5) * axis
                pose_log.write(
                    f"{frame_name} {q_w} {q_xyz[0].item()} {q_xyz[1].item()} {q_xyz[2].item()} {t[0]} {t[1]} {t[2]} {r_err} {t_err} {inlier_count}\n")

            avg_batch_time += time.time() - batch_start
            num_batches += 1

    # --- Analysis & Plotting ---
    total_frames = max(1, len(rErrs))
    pct10_5 = pct10_5 / total_frames * 100
    pct5 = pct5 / total_frames * 100
    pct2 = pct2 / total_frames * 100
    pct1 = pct1 / total_frames * 100
    median_r = np.median(rErrs) if rErrs else 0
    median_t = np.median(tErrs) if tErrs else 0
    mean_repro = np.mean(repro_means) if repro_means else 0

    fig, axs = plt.subplots(1, 3, figsize=(18, 5))

    def plot_smart_hist(ax, data, title, color, unit):
        if len(data) == 0:
            return

        # 1. 计算 95% 分位数，作为绘图的上限
        # 这样可以切掉 5% 的极端离群值，让主要分布展开
        limit = np.percentile(data, 95)

        # 如果数据非常集中（例如limit为0），强制给一点范围
        if limit < 1e-3: limit = max(data)

        # 2. 筛选用于绘图的数据
        vis_data = [x for x in data if x <= limit]

        # 3. 绘制直方图
        ax.hist(vis_data, bins=50, color=color, edgecolor='black', alpha=0.7)
        ax.set_title(f'{title}\nMedian: {np.median(data):.2f} {unit} (Displaying top 95%)')
        ax.set_xlabel(f'Error ({unit})')
        ax.set_ylabel('Count')
        ax.grid(True, linestyle='--', alpha=0.5)

    # 绘制三张图
    plot_smart_hist(axs[0], tErrs, 'Translation Error', 'skyblue', 'cm')
    plot_smart_hist(axs[1], rErrs, 'Rotation Error', 'salmon', 'deg')
    plot_smart_hist(axs[2], repro_means, 'Regr. Reprojection Error', 'lightgreen', 'px')

    plt.tight_layout()
    plt.savefig(plot_file)
    plt.close()

    # Final Log
    _logger.info("=" * 50)
    _logger.info(f"Model: {model_stem}")
    _logger.info(f"Accuracy (10cm/5deg): {pct10_5:.1f}%")
    _logger.info(f"Accuracy (5cm/5deg):  {pct5:.1f}%")
    _logger.info(f"Median Error: {median_r:.2f}deg, {median_t:.2f}cm")
    _logger.info(f"Mean Reprojection: {mean_repro:.2f}px")

    test_log.write(f"Acc_10_5: {pct10_5}\nMedian_R: {median_r}\nMedian_T: {median_t}\nMean_Repro: {mean_repro}\n")
    test_log.close()
    pose_log.close()