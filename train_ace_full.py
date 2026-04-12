#!/usr/bin/env python3
# train_ace_full.py
# 启动脚本：解析参数并启动全功能 ACE 训练 (ACE + Depth + SuperPoint + Intrinsic Fusion)

import argparse
import logging
import os
import sys
from pathlib import Path
from distutils.util import strtobool

# ==================================================================================
# [关键修改] 在导入 PyTorch 之前设置 CUDA_VISIBLE_DEVICES
# 这样可以欺骗 PyTorch，让它只看得到指定的显卡，从而避免占用 GPU 0 的显存。
# ==================================================================================
def setup_cuda_environment():
    # 创建一个临时的 parser 只为了读取 device 参数
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument('--device', type=str, default='cuda:3', help='Target device (e.g., cuda:2)')
    pre_args, _ = pre_parser.parse_known_args()

    device_str = pre_args.device
    if 'cuda' in device_str and ':' in device_str:
        # 提取物理显卡 ID (例如 "cuda:2" -> "2")
        gpu_id = device_str.split(':')[-1]
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
        print(f"Info: Set CUDA_VISIBLE_DEVICES = {gpu_id} (Physical GPU)")
        print(f"Info: Inside PyTorch, this will be mapped to 'cuda:0'")
    else:
        # 如果是 'cuda' (默认0) 或 'cpu'，不做特殊处理
        pass
setup_cuda_environment()

# ==================================================================================
# 现在可以安全导入 PyTorch 和项目模块了
# ==================================================================================
import torch
from ace_trainer_full import TrainerACE

# 配置日志
logging.basicConfig(level=logging.INFO)
_logger = logging.getLogger(__name__)


def _strtobool(x):
    return bool(strtobool(x))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train ACE with Depth Assistance, SuperPoint Sampling, and Intrinsic Fusion.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # --- 1. 基础路径参数 ---
    parser.add_argument('scene', type=Path, help='Path to the scene folder (e.g. datasets/Cambridge/GreatCourt)')
    parser.add_argument('--encoder_path', type=Path, default=Path(__file__).parent / "ace_encoder_pretrained.pt",
                        help='file containing pre-trained encoder weights')

    # --- 2. 训练输出与设备 ---
    parser.add_argument('output_map_file', type=Path,
                        help='target file for the trained network')
    parser.add_argument('--device', type=str, default='cuda:2',
                        help='Device to use. Note: If using a specific ID (e.g. cuda:2), logic handles CUDA_VISIBLE_DEVICES automatically.')

    # --- 3. 优化器与训练轮次 ---
    parser.add_argument('--learning_rate_min', type=float, default=0.0005, help='Minimum learning rate (OneCycleLR)')
    parser.add_argument('--learning_rate_max', type=float, default=0.005, help='Maximum learning rate (OneCycleLR)')
    parser.add_argument('--epochs', type=int, default=28, help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=5120, help='Batch size for coordinate regression')
    parser.add_argument('--training_buffer_size', type=int, default=24000000,
                        help='Total number of patches in the training buffer')
    parser.add_argument('--buffer_chunk_size', type=int, default=12000000,
                        help='patches per buffer refill; defaults to ~2M so fusion updates feed the next chunk')
    parser.add_argument('--sp_ratio', type=int, default=0.5,
                        help='使用superpoint采样点所占得比例')
    parser.add_argument('--sampling_mode', type=str, default='random',
                        help='所使用的补充采样模式')

    # --- 4. ACE 核心参数 ---
    parser.add_argument('--num_head_blocks', type=int, default=8,
                        help='Number of residual blocks in the Regressor head')
    parser.add_argument('--use_homogeneous', type=_strtobool, default=True,
                        help='Use homogeneous coordinates for regression')
    parser.add_argument('--use_half', type=_strtobool, default=True, help='Use FP16 (AMP) for training')
    parser.add_argument('--use_uncertainty', type=_strtobool, default=False,
                        help='Enable aleatoric uncertainty prediction (output 4 channels instead of 3).')

    # --- 5. 数据增强 ---
    parser.add_argument('--use_aug', type=_strtobool, default=True, help='Use data augmentation')
    parser.add_argument('--aug_rotation', type=int, default=15, help='Max rotation augmentation in degrees')
    parser.add_argument('--aug_scale', type=float, default=1.5, help='Max scale augmentation')
    parser.add_argument('--image_resolution', type=int, default=480, help='Height of input images')

    # --- 6. 采样策略 (SuperPoint) ---
    parser.add_argument('--samples_per_image', type=int, default=1024, help='Target number of samples per image')
    parser.add_argument('--weights_path', type=str, default='superpoint_v1.pth', help='Path to SuperPoint weights')
    parser.add_argument('--nms_dist', type=int, default=4, help='SuperPoint NMS distance')
    parser.add_argument('--conf_thresh', type=float, default=0.015, help='SuperPoint confidence threshold')
    parser.add_argument('--nn_thresh', type=float, default=0.7,
                        help='SuperPoint descriptor matching threshold (not used for sampling but required by init)')

    # --- 7. Depth Anything & Loss 参数 ---
    parser.add_argument('--depth_encoder', type=str, default='vits', choices=['vits', 'vitb', 'vitl', 'vitg'],
                        help='Depth Anything model version')
    parser.add_argument('--depth_batch_size', type=int, default=2,
                        help='Batch size for Depth Distillation branch (keep small to avoid OOM)')
    parser.add_argument('--depth_min', type=float, default=0.1, help='Minimum valid depth')
    parser.add_argument('--depth_max', type=float, default=1000.0, help='Maximum valid depth')
    parser.add_argument('--depth_target', type=float, default=10.0,
                        help='Default target depth for invalid points (proxy loss)')
    parser.add_argument('--input_size', type=int, default=518)

    # RelativeDepthLoss 内部权重 (平衡 ssi/grad 和 pairwise loss)
    # 注意：这不同于任务间的不确定性加权。这是 Depth Loss 内部不同项的平衡。
    parser.add_argument('--RelativeDepthLoss_weight', type=float, default=1.0,
                        help='Weight inside RelativeDepthLoss (balances Pairwise vs Grad/SSI)')
    parser.add_argument('--DepthLoss_weight', type=float, default=0.2,
                        help='回归时所用损失中深度损失的权重')

    # --- 8. ACE Reprojection Loss 参数 ---
    parser.add_argument('--repro_loss_type', type=str, default='dyntanh', choices=['l1', 'l2', 'dyntanh'],
                        help='Loss function type')
    parser.add_argument('--repro_loss_schedule', type=str, default='circle', choices=['circle', 'linear'],
                        help='Schedule for loss clamping')
    parser.add_argument('--repro_loss_soft_clamp', type=float, default=50.0, help='Soft clamping threshold')
    parser.add_argument('--repro_loss_soft_clamp_min', type=float, default=1.0, help='Minimum soft clamping threshold')
    parser.add_argument('--repro_loss_hard_clamp', type=float, default=1000.0,
                        help='Hard clamping threshold (for outlier rejection)')

    options = parser.parse_args()

    # 简单的检查
    if not options.scene.exists():
        _logger.error(f"Scene directory does not exist: {options.scene}")
        exit(1)
    if not options.encoder_path.exists():
        _logger.error(f"Encoder weights not found: {options.encoder_path}")
        exit(1)

    _logger.info(f"Starting Training on: {options.scene}")
    _logger.info(f"Device: {options.device} (Mapped to cuda:0 internally via CUDA_VISIBLE_DEVICES)")

    trainer = TrainerACE(options)
    trainer.train()