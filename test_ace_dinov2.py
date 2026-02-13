#!/usr/bin/env python3
# test_ace_dinov2.py
# Testing script for ACE with DINOv2 encoder

import argparse
import logging
import os
import sys
from pathlib import Path
from distutils.util import strtobool
import time

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

setup_cuda_environment()

import torch
import numpy as np
from torch.utils.data import DataLoader

from ace_network_dinov2 import Regressor
from dataset_dinov2 import CamLocDatasetDINOv2
import dsacstar

logging.basicConfig(level=logging.INFO)
_logger = logging.getLogger(__name__)


def _strtobool(x):
    return bool(strtobool(x))


def compute_pose_error(pred_pose, gt_pose):
    """Compute translation and rotation error."""
    # Translation error (meters)
    trans_error = torch.norm(pred_pose[:3, 3] - gt_pose[:3, 3]).item()

    # Rotation error (degrees)
    R_pred = pred_pose[:3, :3]
    R_gt = gt_pose[:3, :3]
    R_diff = torch.matmul(R_pred, R_gt.T)
    trace = torch.trace(R_diff)
    rot_error = torch.acos(torch.clamp((trace - 1) / 2, -1, 1)) * 180 / np.pi

    return trans_error, rot_error.item()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Test ACE with DINOv2 encoder',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Required arguments
    parser.add_argument('scene', type=Path,
                       help='Path to scene folder')
    parser.add_argument('head_network', type=Path,
                       help='Path to trained head network')

    # DINOv2 settings
    parser.add_argument('--dinov2_path', type=Path,
                       default=Path('/data/xwh/checkpoints/dinov2_vitl14_pretrain.pth'),
                       help='Path to DINOv2 pretrained weights')

    # Device settings
    parser.add_argument('--device', type=str, default='cuda:0',
                       help='Testing device')

    # Image settings
    parser.add_argument('--image_resolution', type=int, default=518,
                       help='Image height (must be multiple of 14)')

    # RANSAC settings
    parser.add_argument('--hypotheses', type=int, default=64,
                       help='Number of RANSAC hypotheses')
    parser.add_argument('--threshold', type=float, default=10,
                       help='Inlier threshold in pixels')
    parser.add_argument('--session', type=str, default='default',
                       help='Session name for output file')

    args = parser.parse_args()

    # Validate paths
    if not args.head_network.exists():
        _logger.error(f"Head network not found at {args.head_network}")
        sys.exit(1)
    if not args.dinov2_path.exists():
        _logger.error(f"DINOv2 weights not found at {args.dinov2_path}")
        sys.exit(1)

    # Validate image resolution
    if args.image_resolution % 14 != 0:
        args.image_resolution = (args.image_resolution // 14) * 14
        _logger.warning(f"Image resolution adjusted to {args.image_resolution}")

    device = torch.device('cuda:0')

    _logger.info("=" * 80)
    _logger.info("Testing ACE with DINOv2 Encoder")
    _logger.info("=" * 80)
    _logger.info(f"Scene: {args.scene}")
    _logger.info(f"Head network: {args.head_network}")
    _logger.info(f"DINOv2 weights: {args.dinov2_path}")
    _logger.info(f"Image resolution: {args.image_resolution}")
    _logger.info("=" * 80)

    # Load dataset
    test_dataset = CamLocDatasetDINOv2(
        root_dir=args.scene / "test",
        mode=0,
        use_half=False,
        image_height=args.image_resolution,
        augment=False
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4
    )

    _logger.info(f"Test dataset: {len(test_dataset)} images")

    # Load model
    _logger.info("Loading model...")
    regressor = Regressor.create_from_split_state_dict(
        dinov2_path=args.dinov2_path,
        head_state_dict=torch.load(args.head_network, map_location='cpu')
    )
    regressor = regressor.to(device)
    regressor.eval()
    _logger.info("Model loaded successfully")

    # Testing loop
    results = []
    trans_errors = []
    rot_errors = []
    inlier_counts = []
    processing_times = []

    _logger.info("Starting testing...")

    with torch.no_grad():
        for idx, batch in enumerate(test_dataloader):
            start_time = time.time()

            image = batch[0].to(device)
            gt_pose = batch[2][0]
            intrinsics = batch[4][0]

            # Predict scene coordinates
            scene_coords = regressor(image)
            scene_coords = scene_coords.squeeze(0).cpu()  # [3, H, W]

            # Run RANSAC
            _, _, H, W = image.shape

            # Create 2D pixel coordinates
            pixel_grid = torch.zeros(2, H, W)
            for y in range(H):
                for x in range(W):
                    pixel_grid[0, y, x] = x
                    pixel_grid[1, y, x] = y

            # Estimate pose using DSAC*
            try:
                pred_pose, inliers = dsacstar.forward_rgb(
                    scene_coords.unsqueeze(0),
                    pixel_grid.unsqueeze(0),
                    args.hypotheses,
                    args.threshold,
                    intrinsics.unsqueeze(0).float(),
                    1.0  # focal length scaling
                )
                pred_pose = pred_pose.squeeze(0)
                inlier_count = inliers.sum().item()
            except Exception as e:
                _logger.warning(f"RANSAC failed for image {idx}: {e}")
                pred_pose = torch.eye(4)
                inlier_count = 0

            # Compute errors
            trans_err, rot_err = compute_pose_error(pred_pose, gt_pose)

            processing_time = time.time() - start_time

            # Store results
            trans_errors.append(trans_err)
            rot_errors.append(rot_err)
            inlier_counts.append(inlier_count)
            processing_times.append(processing_time)

            if (idx + 1) % 10 == 0:
                _logger.info(f"Processed {idx + 1}/{len(test_dataset)} images")

    # Compute statistics
    trans_errors = np.array(trans_errors)
    rot_errors = np.array(rot_errors)
    inlier_counts = np.array(inlier_counts)
    processing_times = np.array(processing_times)

    _logger.info("=" * 80)
    _logger.info("Testing Results")
    _logger.info("=" * 80)
    _logger.info(f"Median translation error: {np.median(trans_errors):.3f} m")
    _logger.info(f"Median rotation error: {np.median(rot_errors):.3f} deg")
    _logger.info(f"Mean inlier count: {np.mean(inlier_counts):.1f}")
    _logger.info(f"Mean processing time: {np.mean(processing_times):.3f} s")

    # Accuracy at different thresholds
    for trans_th, rot_th in [(0.05, 5), (0.25, 2), (0.5, 5), (5.0, 10)]:
        acc = np.mean((trans_errors < trans_th) & (rot_errors < rot_th)) * 100
        _logger.info(f"Accuracy @ {trans_th}m, {rot_th}deg: {acc:.2f}%")

    _logger.info("=" * 80)
    _logger.info("Testing completed!")
