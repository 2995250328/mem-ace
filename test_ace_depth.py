#!/usr/bin/env python3
# Copyright © Niantic, Inc. 2022.

import argparse
import logging
import math
import time
from distutils.util import strtobool
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt  # 引入绘图库

import dsacstar
from ace_network_depth import Regressor
from dataset import CamLocDataset

import ace_vis_util as vutil
from ace_util import get_pixel_grid, to_homogeneous
from ace_visualizer import ACEVisualizer
from superpoint import SuperPointFrontend

_logger = logging.getLogger(__name__)


def _strtobool(x):
    return bool(strtobool(x))


def compute_reprojection_error(pred_scene_coords, gt_pose_inv, intrinsic, device):
    """
    计算回归的重投影误差。
    利用 GT Pose 将预测的 3D 坐标投影回 2D，与像素网格对比。
    pred_scene_coords: [3, H, W]
    gt_pose_inv: [4, 4] (World -> Camera)
    intrinsic: [3, 3]
    """
    c, h, w = pred_scene_coords.shape

    # 1. 构造像素网格 (Ground Truth Pixels)
    y_grid, x_grid = torch.meshgrid(torch.arange(0, h, device=device),
                                    torch.arange(0, w, device=device), indexing='ij')
    pixel_grid = torch.stack((x_grid, y_grid), dim=0).float().view(2, -1)  # [2, H*W]

    # 2. 准备预测的 3D 点
    scene_coords_flat = pred_scene_coords.view(3, -1)  # [3, H*W]
    ones = torch.ones((1, h * w), device=device)
    scene_coords_homo = torch.cat((scene_coords_flat, ones), dim=0)  # [4, H*W]

    # 3. 变换到相机坐标系 (World -> Camera)
    cam_coords = torch.mm(gt_pose_inv.to(device), scene_coords_homo)  # [4, H*W]

    # 4. 剔除 Z <= 0 的点 (在相机背面)
    depth = cam_coords[2, :]
    valid_mask = depth > 0.1

    if valid_mask.sum() == 0:
        return torch.tensor(0.0)

    # 5. 投影到像素平面
    # K * [X, Y, Z]^T -> [u*z, v*z, z]^T
    proj_homo = torch.mm(intrinsic.to(device), cam_coords[:3, :])  # [3, H*W]
    proj_uv = proj_homo[:2] / (proj_homo[2:3] + 1e-6)  # [2, H*W]

    # 6. 计算 L2 误差
    diff = proj_uv[:, valid_mask] - pixel_grid[:, valid_mask]
    error = torch.norm(diff, dim=0)  # [N_valid]

    return error.mean().item()


if __name__ == '__main__':
    # Setup logging.
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(
        description='Test a trained network on a specific scene.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('scene', type=Path,
                        help='path to a scene in the dataset folder, e.g. "datasets/Cambridge_GreatCourt"')

    parser.add_argument('network', type=Path, help='path to a network trained for the scene (just the head weights)')

    parser.add_argument('--encoder_path', type=Path, default=Path(__file__).parent / "ace_encoder_pretrained.pt",
                        help='file containing pre-trained encoder weights')

    parser.add_argument('--session', '-sid', default='',
                        help='custom session name appended to output files, '
                             'useful to separate different runs of a script')

    parser.add_argument('--image_resolution', type=int, default=480, help='base image resolution')

    # DSACStar RANSAC parameters. ACE Keeps them at default.
    parser.add_argument('--hypotheses', '-hyps', type=int, default=64,
                        help='number of hypotheses, i.e. number of RANSAC iterations')

    parser.add_argument('--threshold', '-t', type=float, default=10,
                        help='inlier threshold in pixels (RGB) or centimeters (RGB-D)')

    parser.add_argument('--inlieralpha', '-ia', type=float, default=100,
                        help='alpha parameter of the soft inlier count; controls the softness of the '
                             'hypotheses score distribution; lower means softer')

    parser.add_argument('--maxpixelerror', '-maxerrr', type=float, default=100,
                        help='maximum reprojection (RGB, in px) or 3D distance (RGB-D, in cm) error when checking '
                             'pose consistency towards all measurements; error is clamped to this value for stability')

    # Visualization params
    parser.add_argument('--render_visualization', type=_strtobool, default=False,
                        help='create a video of the mapping process')
    parser.add_argument('--render_target_path', type=Path, default='renderings',
                        help='target folder for renderings')
    parser.add_argument('--render_flipped_portrait', type=_strtobool, default=False,
                        help='flag for wayspots dataset where images are sideways portrait')
    parser.add_argument('--render_sparse_queries', type=_strtobool, default=False,
                        help='set to true if your queries are not a smooth video')
    parser.add_argument('--render_pose_error_threshold', type=int, default=20,
                        help='pose error threshold for the visualisation in cm/deg')
    parser.add_argument('--render_map_depth_filter', type=int, default=10,
                        help='to clean up the ACE point cloud remove points too far away')
    parser.add_argument('--render_camera_z_offset', type=int, default=4,
                        help='zoom out of the scene by moving render camera backwards, in meters')
    parser.add_argument('--render_frame_skip', type=int, default=1,
                        help='skip every xth frame for long and dense query sequences')

    # SuperPoint params
    parser.add_argument('--weights_path', type=str, default='superpoint_v1.pth',
                        help='Path to pretrained weights file.')
    parser.add_argument('--nms_dist', type=int, default=1,
                        help='Non Maximum Suppression (NMS) distance.')
    parser.add_argument('--conf_thresh', type=float, default=0.05,
                        help='Detector confidence threshold.')
    parser.add_argument('--nn_thresh', type=float, default=0.7,
                        help='Descriptor matching threshold.')

    opt = parser.parse_args()

    device = torch.device("cuda:1")
    cuda = device == torch.device("cuda:1")
    num_workers = 6

    scene_path = Path(opt.scene)
    network_path = Path(opt.network)
    encoder_path = Path(opt.encoder_path)

    # --- 智能文件命名逻辑 ---
    # 获取模型的基础名称 (不带后缀)，例如 "pgt_7scenes_stairs_vitb_ep28"
    model_stem = network_path.stem
    output_dir = network_path.parent

    # 构建各类输出文件的路径
    test_log_file = output_dir / f'stats_{model_stem}.txt'
    pose_log_file = output_dir / f'poses_{model_stem}.txt'
    big_error_file = output_dir / f'list_big_error_{model_stem}.txt'
    little_error_file = output_dir / f'list_little_error_{model_stem}.txt'
    plot_file = output_dir / f'plot_analysis_{model_stem}.png'

    # Setup dataset.
    testset = CamLocDataset(
        scene_path / "test",
        mode=0,
        image_height=opt.image_resolution,
    )
    _logger.info(f'Test images found: {len(testset)}')

    # Setup dataloader.
    testset_loader = DataLoader(testset, shuffle=False, num_workers=num_workers)

    # Load network.
    encoder_state_dict = torch.load(encoder_path, map_location="cpu")
    network_state_dict = torch.load(network_path, map_location="cpu")
    network = Regressor.create_from_split_state_dict(encoder_state_dict, network_state_dict)

    # SuperPoint (Used inside Regressor or separately? Keeping initialization as per original code)
    superpoint = SuperPointFrontend(network.superpoint, opt.weights_path, opt.nms_dist, opt.conf_thresh, opt.nn_thresh,
                                    cuda)

    network = network.to(device)
    network.eval()

    _logger.info(f"Saving aggregate statistics to: {test_log_file}")
    _logger.info(f"Saving per-frame poses to: {pose_log_file}")

    test_log = open(test_log_file, 'w', 1)
    pose_log = open(pose_log_file, 'w', 1)

    # Metrics
    avg_batch_time = 0
    num_batches = 0

    rErrs = []  # 旋转误差列表
    tErrs = []  # 平移误差列表
    repro_means = []  # 平均重投影误差列表

    # Threshold counters
    pct10_5_up = 0
    pct10_5 = 0
    pct5 = 0
    pct2 = 0
    pct1 = 0

    big_error_list = []
    little_error_list = []

    # Visualizer Setup
    ace_visualizer = None
    if opt.render_visualization:
        target_path = vutil.get_rendering_target_path(opt.render_target_path, opt.network)
        ace_visualizer = ACEVisualizer(target_path,
                                       opt.render_flipped_portrait,
                                       opt.render_map_depth_filter,
                                       reloc_vis_error_threshold=opt.render_pose_error_threshold)
        trainset = CamLocDataset(scene_path / "train", mode=0, image_height=opt.image_resolution)
        trainset_loader = DataLoader(trainset, shuffle=False, num_workers=num_workers)
        ace_visualizer.setup_reloc_visualisation(
            frame_count=len(testset),
            data_loader=trainset_loader,
            network=network,
            camera_z_offset=opt.render_camera_z_offset,
            reloc_frame_skip=opt.render_frame_skip)

    # --- Testing Loop ---
    testing_start_time = time.time()

    with torch.no_grad():
        for image_RGB, image_B1HW, _, gt_pose_B44, gt_pose_inv_B44, intrinsics_B33, _, _, filenames in testset_loader:
            batch_start_time = time.time()

            image_B1HW = image_B1HW.to(device, non_blocking=True)

            # Predict scene coordinates
            # 注意：此处使用 autocast 可能会影响重投影精度，但速度快。若追求分析精度可暂时关掉。
            with autocast(enabled=True):
                scene_coordinates_B3HW = network(image_B1HW)

            # Move to CPU for RANSAC, keep GPU copy for Reprojection Error calculation
            scene_coordinates_B3HW_cpu = scene_coordinates_B3HW.float().cpu()

            for i in range(image_B1HW.shape[0]):
                scene_coords_3HW = scene_coordinates_B3HW_cpu[i]
                gt_pose_44 = gt_pose_B44[i]
                gt_pose_inv_44 = gt_pose_inv_B44[i]  # Dataloader 需要返回 inverse pose
                intrinsics_33 = intrinsics_B33[i]
                frame_path = filenames[i]
                frame_name = Path(frame_path).name

                # --- 1. 计算回归重投影误差 (新增功能) ---
                # 使用 GPU 上的原始预测值
                repro_error_val = compute_reprojection_error(
                    scene_coordinates_B3HW[i],
                    gt_pose_inv_44,
                    intrinsics_33,
                    device
                )
                repro_means.append(repro_error_val)

                # --- 2. RANSAC Pose Estimation ---
                focal_length = intrinsics_33[0, 0].item()
                ppX = intrinsics_33[0, 2].item()
                ppY = intrinsics_33[1, 2].item()

                out_pose = torch.zeros((4, 4))

                inlier_count = dsacstar.forward_rgb(
                    scene_coords_3HW.unsqueeze(0),
                    out_pose,
                    opt.hypotheses,
                    opt.threshold,
                    focal_length,
                    ppX,
                    ppY,
                    opt.inlieralpha,
                    opt.maxpixelerror,
                    network.OUTPUT_SUBSAMPLE,
                )

                # --- 3. Compute Pose Errors ---
                t_err = float(torch.norm(gt_pose_44[0:3, 3] - out_pose[0:3, 3]))

                gt_R = gt_pose_44[0:3, 0:3].numpy()
                out_R = out_pose[0:3, 0:3].numpy()
                r_err_mat = np.matmul(out_R, np.transpose(gt_R))
                r_err_vec = cv2.Rodrigues(r_err_mat)[0]
                r_err = np.linalg.norm(r_err_vec) * 180 / math.pi

                _logger.info(
                    f"[{frame_name}] R_Err: {r_err:.2f}deg, T_Err: {t_err * 100:.1f}cm, Repro_Err: {repro_error_val:.2f}px")

                if ace_visualizer is not None:
                    ace_visualizer.render_reloc_frame(
                        query_pose=gt_pose_44.numpy(),
                        query_file=frame_path,
                        est_pose=out_pose.numpy(),
                        est_error=max(r_err, t_err * 100),
                        sparse_query=opt.render_sparse_queries)

                rErrs.append(r_err)
                tErrs.append(t_err * 100)

                # Classification
                if r_err >= 5 or t_err >= 0.1:
                    pct10_5_up += 1
                    big_error_list.append(frame_name)
                elif r_err < 2 and t_err < 0.02:
                    little_error_list.append(frame_name)

                if r_err < 5 and t_err < 0.1: pct10_5 += 1
                if r_err < 5 and t_err < 0.05: pct5 += 1
                if r_err < 2 and t_err < 0.02: pct2 += 1
                if r_err < 1 and t_err < 0.01: pct1 += 1

                # Save Pose Log
                out_pose_inv = out_pose.inverse()
                t = out_pose_inv[0:3, 3]
                rot, _ = cv2.Rodrigues(out_pose_inv[0:3, 0:3].numpy())
                angle = np.linalg.norm(rot)
                axis = rot / (angle + 1e-6)
                q_w = math.cos(angle * 0.5)
                q_xyz = math.sin(angle * 0.5) * axis

                pose_log.write(f"{frame_name} "
                               f"{q_w} {q_xyz[0].item()} {q_xyz[1].item()} {q_xyz[2].item()} "
                               f"{t[0]} {t[1]} {t[2]} "
                               f"{r_err} {t_err} {inlier_count}\n")

            avg_batch_time += time.time() - batch_start_time
            num_batches += 1

    total_frames = len(rErrs)

    # --- 保存大/小误差列表 ---
    with open(big_error_file, "w") as f:
        f.write("\n".join(big_error_list))
    with open(little_error_file, "w") as f:
        f.write("\n".join(little_error_list))
    _logger.info(f"Saved error lists to {big_error_file} and {little_error_file}")

    # --- 统计与绘图 (Visual Analysis) ---
    tErrs = np.array(tErrs)
    rErrs = np.array(rErrs)
    repro_means = np.array(repro_means)

    median_rErr = np.median(rErrs)
    median_tErr = np.median(tErrs)
    mean_repro = np.mean(repro_means)

    # 计算百分比
    pct10_5 = pct10_5 / total_frames * 100
    pct5 = pct5 / total_frames * 100
    pct2 = pct2 / total_frames * 100
    pct1 = pct1 / total_frames * 100

    # 绘制统计图
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))

    # 1. Translation Error Histogram
    axs[0].hist(tErrs, bins=50, color='skyblue', edgecolor='black')
    axs[0].set_title(f'Translation Error Distribution\nMedian: {median_tErr:.2f} cm')
    axs[0].set_xlabel('Error (cm)')
    axs[0].set_ylabel('Count')

    # 2. Rotation Error Histogram
    axs[1].hist(rErrs, bins=50, color='salmon', edgecolor='black')
    axs[1].set_title(f'Rotation Error Distribution\nMedian: {median_rErr:.2f} deg')
    axs[1].set_xlabel('Error (deg)')

    # 3. Reprojection Error Histogram
    # 过滤掉极端的重投影误差以便更好显示分布
    vis_repro = repro_means[repro_means < np.percentile(repro_means, 95)]
    axs[2].hist(vis_repro, bins=50, color='lightgreen', edgecolor='black')
    axs[2].set_title(f'Regression Reprojection Error (Mean per Frame)\nGlobal Mean: {mean_repro:.2f} px')
    axs[2].set_xlabel('Error (px)')

    plt.tight_layout()
    plt.savefig(plot_file)
    plt.close()
    _logger.info(f"Saved analysis plot to {plot_file}")

    # --- Final Logging ---
    avg_time = avg_batch_time / num_batches

    _logger.info("===================================================")
    _logger.info(f"Model: {model_stem}")
    _logger.info(f'10cm/5deg: {pct10_5:.1f}%')
    _logger.info(f'5cm/5deg:  {pct5:.1f}%')
    _logger.info(f'2cm/2deg:  {pct2:.1f}%')
    _logger.info(f'1cm/1deg:  {pct1:.1f}%')
    _logger.info(f"Median Error: {median_rErr:.2f}deg, {median_tErr:.2f}cm")
    _logger.info(f"Mean Reprojection Error: {mean_repro:.2f}px")
    _logger.info(f"Avg processing time: {avg_time * 1000:.1f}ms")

    test_log.write(f"Model: {model_stem}\n")
    test_log.write(f"Acc_10_5: {pct10_5}\n")
    test_log.write(f"Acc_5_5: {pct5}\n")
    test_log.write(f"Acc_2_2: {pct2}\n")
    test_log.write(f"Acc_1_1: {pct1}\n")
    test_log.write(f"Median_R: {median_rErr}\n")
    test_log.write(f"Median_T: {median_tErr}\n")
    test_log.write(f"Mean_Repro: {mean_repro}\n")

    test_log.close()
    pose_log.close()