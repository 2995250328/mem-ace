#!/usr/bin/env python3
# test_ace_lmc.py
# Testing script for ACE with the original FCN encoder (ace_encoder_pretrained.pt).
# Mirrors test_ace_dinov2.py; only encoder/args are adapted for the FCN backbone.

import argparse
import logging
import math
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.amp import autocast
from torch.utils.data import DataLoader

import dsacstar
from ace_network_ace import RegressorACE
from dataset_dinov2 import CamLocDatasetDINOv2  # RGB dataset; ACEEncoder handles grayscale internally

_logger = logging.getLogger(__name__)


def run_evaluation(opt):
    """
    Run evaluation on a scene with a trained ACE-FCN head.

    opt: object with .scene, .network, .encoder_path, .device, .image_resolution, .session,
         and optionally .hypotheses, .threshold, .inlieralpha, .maxpixelerror, .render_visualization.
    Returns: dict with median_rErr, median_tErr, avg_time, pct25_5, pct10_5, pct5, pct2, pct1,
             total_frames, test_log_file, pose_log_file.
    """
    hypotheses = getattr(opt, 'hypotheses', 64)
    threshold = getattr(opt, 'threshold', 10)
    inlieralpha = getattr(opt, 'inlieralpha', 100)
    maxpixelerror = getattr(opt, 'maxpixelerror', 100)
    image_resolution = getattr(opt, 'image_resolution', 480)

    device = torch.device(opt.device)
    scene_path = Path(opt.scene)
    head_network_path = Path(opt.network)
    encoder_path = Path(opt.encoder_path)
    session = getattr(opt, 'session', '')

    if not head_network_path.exists():
        raise FileNotFoundError(f"Head network not found at {head_network_path}")
    if not encoder_path.exists():
        raise FileNotFoundError(f"ACE encoder weights not found at {encoder_path}")

    testset = CamLocDatasetDINOv2(
        scene_path / "test",
        mode=0,
        use_half=False,
        image_height=image_resolution,
        augment=False,
    )
    _logger.info("Test images found: %d", len(testset))
    testset_loader = DataLoader(testset, shuffle=False, num_workers=6)

    head_state_dict = torch.load(head_network_path, map_location="cpu")
    # LMC checkpoints store head under 'head_state_dict' key; vanilla checkpoints are the dict itself.
    if isinstance(head_state_dict, dict) and 'head_state_dict' in head_state_dict:
        head_state_dict = head_state_dict['head_state_dict']

    network = RegressorACE.create_from_split_state_dict(
        scene_path=scene_path,
        encoder_path=encoder_path,
    )
    # Load head weights from checkpoint.
    network.heads.load_state_dict(head_state_dict if 'res3_conv1.weight' in head_state_dict
                                  else {k.replace('heads.', ''): v
                                        for k, v in head_state_dict.items()
                                        if k.startswith('heads.')})
    network = network.to(device)
    network.eval()

    output_dir = head_network_path.parent
    scene_name = scene_path.name
    test_log_file = output_dir / f'test_{scene_name}_{session}.txt'
    pose_log_file = output_dir / f'poses_{scene_name}_{session}.txt'
    _logger.info("Saving test stats to: %s", test_log_file)
    _logger.info("Saving per-frame poses to: %s", pose_log_file)

    test_log = open(test_log_file, 'w', 1)
    pose_log = open(pose_log_file, 'w', 1)

    avg_batch_time = 0
    num_batches = 0
    rErrs = []
    tErrs = []
    pct25_5 = pct10_5 = pct5 = pct2 = pct1 = 0

    with torch.no_grad():
        for image_B1HW, _, gt_pose_B44, _, intrinsics_B33, _, _, filenames in testset_loader:
            batch_start_time = time.time()
            image_B1HW = image_B1HW.to(device, non_blocking=True)

            with autocast("cuda", enabled=True):
                scene_coordinates_B3HW = network(image_B1HW)
            scene_coordinates_B3HW = scene_coordinates_B3HW.float().cpu()

            if isinstance(filenames, str):
                filenames = (filenames,)
            for scene_coordinates_3HW, gt_pose_44, intrinsics_33, frame_path in zip(
                    scene_coordinates_B3HW, gt_pose_B44, intrinsics_B33, filenames):

                fx, fy = intrinsics_33[0, 0].item(), intrinsics_33[1, 1].item()
                focal_length = (fx + fy) / 2.0
                ppX = intrinsics_33[0, 2].item()
                ppY = intrinsics_33[1, 2].item()

                frame_name = Path(frame_path).name
                out_pose = torch.zeros((4, 4))

                inlier_count = dsacstar.forward_rgb(
                    scene_coordinates_3HW.unsqueeze(0),
                    out_pose,
                    hypotheses,
                    threshold,
                    focal_length,
                    ppX,
                    ppY,
                    inlieralpha,
                    maxpixelerror,
                    network.OUTPUT_SUBSAMPLE,
                )

                t_err = float(torch.norm(gt_pose_44[0:3, 3] - out_pose[0:3, 3]))
                gt_R = gt_pose_44[0:3, 0:3].numpy()
                out_R = out_pose[0:3, 0:3].numpy()
                r_err = np.linalg.norm(cv2.Rodrigues(np.matmul(out_R, np.transpose(gt_R)))[0]) * 180 / math.pi

                _logger.info("Rotation Error: %.2fdeg, Translation Error: %.1fcm", r_err, t_err * 100)

                rErrs.append(r_err)
                tErrs.append(t_err * 100)

                if r_err < 5 and t_err < 0.25:
                    pct25_5 += 1
                if r_err < 5 and t_err < 0.1:
                    pct10_5 += 1
                if r_err < 5 and t_err < 0.05:
                    pct5 += 1
                if r_err < 2 and t_err < 0.02:
                    pct2 += 1
                if r_err < 1 and t_err < 0.01:
                    pct1 += 1

                out_pose = out_pose.inverse()
                t = out_pose[0:3, 3]
                rot, _ = cv2.Rodrigues(out_pose[0:3, 0:3].numpy())
                angle = np.linalg.norm(rot)
                axis = rot / angle
                q_w = math.cos(angle * 0.5)
                q_xyz = math.sin(angle * 0.5) * axis
                pose_log.write(f"{frame_name} "
                               f"{q_w} {q_xyz[0].item()} {q_xyz[1].item()} {q_xyz[2].item()} "
                               f"{t[0]} {t[1]} {t[2]} "
                               f"{r_err} {t_err} {inlier_count}\n")

            avg_batch_time += time.time() - batch_start_time
            num_batches += 1

    total_frames = len(rErrs)
    assert total_frames == len(testset)

    tErrs.sort()
    rErrs.sort()
    median_idx = total_frames // 2
    median_rErr = rErrs[median_idx]
    median_tErr = tErrs[median_idx]
    avg_time = avg_batch_time / num_batches

    pct25_5 = pct25_5 / total_frames * 100
    pct10_5 = pct10_5 / total_frames * 100
    pct5 = pct5 / total_frames * 100
    pct2 = pct2 / total_frames * 100
    pct1 = pct1 / total_frames * 100

    _logger.info("===================================================")
    _logger.info("EVAL SUMMARY (ACE-FCN):")
    _logger.info("  Median Error: %.2f deg, %.2f cm", median_rErr, median_tErr)
    _logger.info("  25cm/5deg: %.2f%% | 10cm/5deg: %.2f%% | 5cm/5deg: %.2f%% | 2cm/2deg: %.2f%% | 1cm/1deg: %.2f%%",
                 pct25_5, pct10_5, pct5, pct2, pct1)
    _logger.info("  Avg time: %.2f ms | Frames: %d", avg_time * 1000, total_frames)
    _logger.info("===================================================")

    test_log.write(f"{median_rErr} {median_tErr} {avg_time}\n")
    test_log.close()
    pose_log.close()

    return {
        'median_rErr': median_rErr,
        'median_tErr': median_tErr,
        'avg_time': avg_time,
        'pct25_5': pct25_5,
        'pct10_5': pct10_5,
        'pct5': pct5,
        'pct2': pct2,
        'pct1': pct1,
        'total_frames': total_frames,
        'test_log_file': str(test_log_file),
        'pose_log_file': str(pose_log_file),
    }


def run_evaluation_lmc(opt):
    """Alias for run_evaluation; LMC and vanilla checkpoints use the same eval path for ACE-FCN."""
    return run_evaluation(opt)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(
        description='Test a trained ACE-FCN network on a specific scene.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('scene', type=Path, help='path to a scene directory')
    parser.add_argument('network', type=Path, help='path to a trained head weights file (.pt)')
    parser.add_argument('--encoder_path', type=Path, default=Path('ace_encoder_pretrained.pt'),
                        help='path to ACE FCN pretrained encoder weights')
    parser.add_argument('--session', '-sid', default='', help='custom session name for output files')
    parser.add_argument('--image_resolution', type=int, default=480, help='input image height')
    parser.add_argument('--device', type=str, default='cuda', help='device, e.g. cuda or cuda:0')
    parser.add_argument('--hypotheses', '-hyps', type=int, default=64, help='RANSAC iterations')
    parser.add_argument('--threshold', '-t', type=float, default=10, help='inlier threshold in pixels')
    parser.add_argument('--inlieralpha', '-ia', type=float, default=100)
    parser.add_argument('--maxpixelerror', '-maxerrr', type=float, default=100)

    opt = parser.parse_args()
    run_evaluation(opt)
