#!/usr/bin/env python3
# train_ace_dinov2.py
# Training script for ACE with DINOv2 encoder

import argparse
import logging
import os
import sys
from pathlib import Path
from distutils.util import strtobool

# Setup CUDA environment
def setup_cuda_environment():
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument('--device', type=str, default='cuda:0', help='Target device')
    pre_args, _ = pre_parser.parse_known_args()

    device_str = pre_args.device
    if 'cuda' in device_str and ':' in device_str:
        gpu_id = device_str.split(':')[-1]
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
        print(f"Info: Set CUDA_VISIBLE_DEVICES = {gpu_id}")
        print(f"Info: Inside PyTorch, this will be mapped to 'cuda:0'")

setup_cuda_environment()

import torch
from trainer_dinov2 import TrainerACEDINOv2

logging.basicConfig(level=logging.INFO)
_logger = logging.getLogger(__name__)


def _strtobool(x):
    return bool(strtobool(x))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train ACE with DINOv2 encoder',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Required arguments
    parser.add_argument('scene', type=Path,
                       help='Path to scene folder (e.g., datasets/7scenes_chess)')
    parser.add_argument('output_map', type=Path,
                       help='Output path for trained head network')

    # DINOv2 settings
    parser.add_argument('--dinov2_path', type=Path,
                       default=Path('/data/xwh/checkpoints/dinov2_vitl14_pretrain.pth'),
                       help='Path to DINOv2 pretrained weights')
    parser.add_argument('--freeze_backbone', type=_strtobool, default=True,
                       help='Freeze DINOv2 backbone during training')

    # Device settings
    parser.add_argument('--device', type=str, default='cuda:0',
                       help='Training device')

    # Network architecture
    parser.add_argument('--num_head_blocks', type=int, default=1,
                       help='Depth of regression head')
    parser.add_argument('--use_homogeneous', type=_strtobool, default=True,
                       help='Use homogeneous coordinates')

    # Training settings
    parser.add_argument('--training_buffer_size', type=int, default=8000000,
                       help='Training buffer size')
    parser.add_argument('--samples_per_image', type=int, default=1024,
                       help='Features sampled per image for buffer')
    parser.add_argument('--epochs', type=int, default=16,
                       help='Training epochs')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size (images per step). Keep small for DINOv2 ViT-L to avoid OOM (e.g. 4-16 on 24GB GPU)')
    parser.add_argument('--learning_rate_min', type=float, default=0.0001,
                       help='Minimum learning rate')
    parser.add_argument('--learning_rate_max', type=float, default=0.001,
                       help='Maximum learning rate')

    # Image settings
    parser.add_argument('--image_resolution', type=int, default=518,
                       help='Image height (must be multiple of 14)')

    # Data augmentation
    parser.add_argument('--use_aug', type=_strtobool, default=True,
                       help='Use data augmentation')
    parser.add_argument('--aug_rotation', type=int, default=15,
                       help='Max rotation angle for augmentation')
    parser.add_argument('--aug_scale', type=float, default=1.5,
                       help='Max scale factor for augmentation')

    # Loss settings
    parser.add_argument('--repro_loss_type', type=str, default='dyntanh',
                       choices=['l1', 'l2', 'smooth_l1', 'tanh', 'dyntanh'],
                       help='Reprojection loss type')
    parser.add_argument('--repro_loss_soft_clamp', type=float, default=50,
                       help='Soft clamping threshold')
    parser.add_argument('--repro_loss_soft_clamp_min', type=float, default=1,
                       help='Minimum soft clamping threshold')
    parser.add_argument('--repro_loss_schedule', type=str, default='circle',
                       choices=['circle', 'linear'],
                       help='Loss schedule type')

    # Precision
    parser.add_argument('--use_half', type=_strtobool, default=True,
                       help='Use half precision training')

    args = parser.parse_args()

    # Validate image resolution
    if args.image_resolution % 14 != 0:
        _logger.warning(f"Image resolution {args.image_resolution} is not multiple of 14, "
                       f"adjusting to {(args.image_resolution // 14) * 14}")
        args.image_resolution = (args.image_resolution // 14) * 14

    # Check DINOv2 weights exist
    if not args.dinov2_path.exists():
        _logger.error(f"DINOv2 weights not found at {args.dinov2_path}")
        sys.exit(1)

    # Create output directory
    args.output_map.parent.mkdir(parents=True, exist_ok=True)

    # Log configuration
    _logger.info("=" * 80)
    _logger.info("Training ACE with DINOv2 Encoder")
    _logger.info("=" * 80)
    _logger.info(f"Scene: {args.scene}")
    _logger.info(f"Output: {args.output_map}")
    _logger.info(f"DINOv2 weights: {args.dinov2_path}")
    _logger.info(f"Freeze backbone: {args.freeze_backbone}")
    _logger.info(f"Image resolution: {args.image_resolution}")
    _logger.info(f"Epochs: {args.epochs}")
    _logger.info(f"Batch size: {args.batch_size}")
    _logger.info(f"Device: {args.device}")
    _logger.info("=" * 80)

    # Create trainer and train
    trainer = TrainerACEDINOv2(args)
    trainer.train()
    trainer.save_model(args.output_map)

    _logger.info("Training completed successfully!")
