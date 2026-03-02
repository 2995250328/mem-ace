#!/usr/bin/env python3
# test_ace_dinov2_lmc.py
# Testing script for ACE DINOv2 with optional LMC.
# Auto-detects LMC vs vanilla checkpoint and adapts accordingly.

import argparse
import logging
import math
import os
import sys
import time
from pathlib import Path


def setup_cuda_environment():
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument('--device', type=str, default='cuda:0')
    pre_args, _ = pre_parser.parse_known_args()
    device_str = pre_args.device
    if 'cuda' in device_str and ':' in device_str:
        gpu_id = device_str.split(':')[-1]
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id


setup_cuda_environment()

import cv2
import numpy as np
import torch
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader

import dsacstar
from ace_network_dinov2 import Regressor
from dataset_dinov2 import CamLocDatasetDINOv2

logging.basicConfig(level=logging.INFO)
_logger = logging.getLogger(__name__)


def _strtobool(x):
    if isinstance(x, bool):
        return x
    v = str(x).strip().lower()
    if v in ('1', 'true', 'yes', 'on'):
        return True
    if v in ('0', 'false', 'no', 'off'):
        return False
    raise ValueError(f'Invalid boolean value: {x!r}')


def _is_lmc_checkpoint(checkpoint):
    """Check if a checkpoint contains LMC config."""
    return isinstance(checkpoint, dict) and 'lmc_config' in checkpoint


def run_evaluation_lmc(opt):
    """Run evaluation. Auto-detects LMC vs vanilla checkpoint.

    Returns dict with median_rErr, median_tErr, pct5, etc.
    """
    device = torch.device(opt.device)
    dinov2_path = Path(opt.dinov2_path)
    head_network_path = Path(opt.network)
    scene_path = Path(opt.scene)
    session = getattr(opt, 'session', '')
    image_resolution = getattr(opt, 'image_resolution', 518)
    hypotheses = getattr(opt, 'hypotheses', 64)
    threshold = getattr(opt, 'threshold', 10)
    inlieralpha = getattr(opt, 'inlieralpha', 100)
    maxpixelerror = getattr(opt, 'maxpixelerror', 100)

    if image_resolution % 14 != 0:
        image_resolution = (image_resolution // 14) * 14

    _logger.info("[Eval] device=%s  cuda_available=%s  cuda_device_count=%d",
                 device, torch.cuda.is_available(),
                 torch.cuda.device_count() if torch.cuda.is_available() else 0)

    # Load checkpoint
    checkpoint = torch.load(head_network_path, map_location='cpu')
    is_lmc = _is_lmc_checkpoint(checkpoint)

    if is_lmc:
        _logger.info("[LMC] Detected LMC checkpoint")
        lmc_config = checkpoint['lmc_config']
        head_state_dict = checkpoint['head_state_dict']
        memory_path = lmc_config.get('memory_path')
    else:
        _logger.info("[LMC] Vanilla checkpoint — delegating to standard eval")
        head_state_dict = checkpoint

    # Build regressor (encoder + head)
    network = Regressor.create_from_split_state_dict(
        dinov2_path=dinov2_path,
        head_state_dict=head_state_dict,
    )
    network = network.to(device)
    network.eval()
    # Verify the model actually landed on the requested device (catches silent CPU fallbacks).
    _actual_dev = next(network.parameters()).device
    _logger.info("[Eval] network moved to %s (requested %s)", _actual_dev, device)
    if device.type == 'cuda' and _actual_dev.type != 'cuda':
        _logger.error("[Eval] Network is on CPU despite requesting CUDA! "
                      "CUDA_VISIBLE_DEVICES=%s  cuda_available=%s",
                      os.environ.get('CUDA_VISIBLE_DEVICES', 'unset'), torch.cuda.is_available())

    # Build LMC modules if needed
    compressor = None
    fusion = None
    memory_dict = None
    compressor_out_cached = None  # single compression result, reused for all test frames

    if is_lmc and memory_path is not None:
        from ace_compressor import GeoLMC
        from ace_fusion import LMCFeatureFusion

        # Build compressor/fusion with same dims as training (per-layer feature_dim from memory)
        compress_dim = lmc_config.get('compress_dim', 1024)
        num_layers = lmc_config.get('num_layers', 1)
        backbone_feature_dim = 1024
        compressor = GeoLMC(
            input_dim=compress_dim,
            compress_dim=compress_dim,
            num_latent_tokens=lmc_config.get('num_latent_tokens', 64),
            mode=lmc_config.get('lmc_mode', 'global'),
            num_layers=num_layers,
            use_scale_token=lmc_config.get('use_scale_token', True),
            scale_token_dim=lmc_config.get('scale_token_dim', 1024),
            num_attn_layers=lmc_config.get('num_attn_layers', 2),
        ).to(device)
        compressor.load_state_dict(checkpoint['compressor_state_dict'])
        compressor.eval()

        fusion = LMCFeatureFusion(
            feature_dim=backbone_feature_dim,
            mode=lmc_config.get('lmc_mode', 'global'),
            query_feature_dim=backbone_feature_dim,
            memory_feature_dim=compress_dim,
        ).to(device)
        fusion.load_state_dict(checkpoint['fusion_state_dict'])
        fusion.eval()

        # Load memory
        _logger.info(f"[LMC] Loading memory from {memory_path}")
        raw = torch.load(str(memory_path), map_location='cpu')
        memory_dict = {}
        for k in ('pooled_points', 'pooled_features', 'scene_center'):
            v = raw[k]
            if not isinstance(v, torch.Tensor):
                v = torch.tensor(v)
            if k == 'scene_center' and v.dim() == 1:
                v = v.unsqueeze(0)
            elif v.dim() == 2:
                v = v.unsqueeze(0)
            memory_dict[k] = v.to(device)
        if 'all_scale_tokens' in raw and raw['all_scale_tokens'] is not None:
            st = raw['all_scale_tokens']
            if not isinstance(st, torch.Tensor):
                st = torch.tensor(st)
            if st.dim() == 2:
                st = st.unsqueeze(0)
            memory_dict['all_scale_tokens'] = st.to(device)

        # Compress memory once; reuse the same latent for all test frames (no per-frame compression).
        with torch.no_grad():
            compressor_out_cached = compressor(memory_dict)
        _logger.info("[LMC] Memory compressed once (cached for all test frames)")

    # Dataset
    testset = CamLocDatasetDINOv2(
        scene_path / "test", mode=0, use_half=False,
        image_height=image_resolution, augment=False)
    testset_loader = DataLoader(testset, shuffle=False, num_workers=6)
    _logger.info(f"Test images: {len(testset)}")

    # Output files
    output_dir = head_network_path.parent
    scene_name = scene_path.name
    test_log_file = output_dir / f'test_{scene_name}_{session}.txt'
    pose_log_file = output_dir / f'poses_{scene_name}_{session}.txt'
    test_log = open(test_log_file, 'w', 1)
    pose_log = open(pose_log_file, 'w', 1)

    if compressor is not None and compressor_out_cached is not None:
        _logger.info("[LMC] Using pre-computed compressed memory for all %d test images (no per-frame compression)", len(testset))

    avg_batch_time = 0
    num_batches = 0
    rErrs, tErrs = [], []
    pct25_5 = pct10_5 = pct5 = pct2 = pct1 = 0

    with torch.no_grad():
        for image_B1HW, _, gt_pose_B44, _, intrinsics_B33, _, _, filenames in testset_loader:
            batch_start_time = time.time()
            image_B1HW = image_B1HW.to(device, non_blocking=True)

            with autocast(enabled=True):
                features = network.get_features(image_B1HW)

                # Apply LMC fusion if active (reuse single cached compression for this batch)
                if fusion is not None and compressor_out_cached is not None:
                    B, C, H, W = features.shape
                    sc = memory_dict['scene_center']
                    if sc.shape[0] == 1 and B > 1:
                        sc = sc.expand(B, -1)
                    # Expand cached (1, K, ...) to (B, K, ...) when B > 1; no extra compressor forward
                    if B > 1:
                        if isinstance(compressor_out_cached, dict):
                            compressor_out_batch = {
                                k: v.expand(B, *v.shape[1:]) if v.dim() >= 2 and v.shape[0] == 1 else v
                                for k, v in compressor_out_cached.items()
                            }
                        else:
                            z, p = compressor_out_cached
                            z = z.expand(B, -1, -1) if z.shape[0] == 1 else z
                            p = p.expand(B, -1, -1) if p.shape[0] == 1 else p
                            compressor_out_batch = (z, p)
                    else:
                        compressor_out_batch = compressor_out_cached
                    query = features.permute(0, 2, 3, 1).reshape(B, H * W, C)
                    fused = fusion(query, compressor_out_batch, sc)
                    features = fused.reshape(B, H, W, C).permute(0, 3, 1, 2)

                scene_coordinates_B3HW = network.get_scene_coordinates(features)

            scene_coordinates_B3HW = scene_coordinates_B3HW.float().cpu()

            if isinstance(filenames, str):
                filenames = (filenames,)
            for scene_coordinates_3HW, gt_pose_44, intrinsics_33, frame_path in zip(
                    scene_coordinates_B3HW, gt_pose_B44, intrinsics_B33, filenames):

                focal_length = intrinsics_33[0, 0].item()
                ppX = intrinsics_33[0, 2].item()
                ppY = intrinsics_33[1, 2].item()
                frame_name = Path(frame_path).name
                out_pose = torch.zeros((4, 4))

                inlier_count = dsacstar.forward_rgb(
                    scene_coordinates_3HW.unsqueeze(0), out_pose,
                    hypotheses, threshold, focal_length, ppX, ppY,
                    inlieralpha, maxpixelerror, network.OUTPUT_SUBSAMPLE)

                t_err = float(torch.norm(gt_pose_44[0:3, 3] - out_pose[0:3, 3]))
                gt_R = gt_pose_44[0:3, 0:3].numpy()
                out_R = out_pose[0:3, 0:3].numpy()
                r_err = np.matmul(out_R, np.transpose(gt_R))
                r_err = cv2.Rodrigues(r_err)[0]
                r_err = np.linalg.norm(r_err) * 180 / math.pi

                rErrs.append(r_err)
                t_err_cm = t_err * 100
                tErrs.append(t_err_cm)

                if getattr(opt, 'log_per_frame', False):
                    _logger.info("  [Eval] %s  rErr=%.2f deg  tErr=%.2f cm", frame_name, r_err, t_err_cm)

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

                out_pose_inv = out_pose.inverse()
                t = out_pose_inv[0:3, 3]
                rot, _ = cv2.Rodrigues(out_pose_inv[0:3, 0:3].numpy())
                angle = np.linalg.norm(rot)
                axis = rot / angle
                q_w = math.cos(angle * 0.5)
                q_xyz = math.sin(angle * 0.5) * axis
                pose_log.write(
                    f"{frame_name} {q_w} {q_xyz[0].item()} {q_xyz[1].item()} "
                    f"{q_xyz[2].item()} {t[0]} {t[1]} {t[2]} "
                    f"{r_err} {t_err} {inlier_count}\n")

            avg_batch_time += time.time() - batch_start_time
            num_batches += 1

    total_frames = len(rErrs)
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

    _logger.info("=" * 50)
    _logger.info("EVAL SUMMARY (current errors):")
    _logger.info("  Median Error: %.2f deg, %.2f cm", median_rErr, median_tErr)
    _logger.info("  25cm/5deg: %.2f%% | 10cm/5deg: %.2f%% | 5cm/5deg: %.2f%% | 2cm/2deg: %.2f%% | 1cm/1deg: %.2f%%",
                 pct25_5, pct10_5, pct5, pct2, pct1)
    _logger.info("  Avg time: %.2f ms | Frames: %d", avg_time * 1000, total_frames)
    _logger.info("=" * 50)

    test_log.write(f"{median_rErr} {median_tErr} {avg_time}\n")
    test_log.close()
    pose_log.close()

    # 写入当前误差汇总，便于训练/复现时查看
    output_dir = Path(opt.network).parent
    scene_name = Path(opt.scene).name
    session = getattr(opt, 'session', '') or 'eval'
    eval_summary_file = output_dir / f"eval_summary_{scene_name}_{session}.txt"
    summary_lines = [
        f"# ACE DINOv2+LMC Eval Summary | {scene_name}",
        f"median_rotation_deg\t{median_rErr:.4f}",
        f"median_translation_cm\t{median_tErr:.4f}",
        f"accuracy_25cm5deg_pct\t{pct25_5:.2f}",
        f"accuracy_10cm5deg_pct\t{pct10_5:.2f}",
        f"accuracy_5cm5deg_pct\t{pct5:.2f}",
        f"accuracy_2cm2deg_pct\t{pct2:.2f}",
        f"accuracy_1cm1deg_pct\t{pct1:.2f}",
        f"avg_time_per_frame_ms\t{avg_time * 1000:.2f}",
        f"total_frames\t{total_frames}",
    ]
    eval_summary_file.write_text("\n".join(summary_lines) + "\n")
    _logger.info("Eval summary written to: %s", eval_summary_file)

    return {
        'median_rErr': median_rErr, 'median_tErr': median_tErr,
        'avg_time': avg_time,
        'pct25_5': pct25_5, 'pct10_5': pct10_5, 'pct5': pct5, 'pct2': pct2, 'pct1': pct1,
        'total_frames': total_frames,
        'test_log_file': str(test_log_file),
        'pose_log_file': str(pose_log_file),
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Test ACE DINOv2 (with optional LMC)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('scene', type=Path)
    parser.add_argument('network', type=Path, help='Path to checkpoint')
    parser.add_argument('--dinov2_path', type=Path,
                        default=Path('/data/xwh/checkpoints/dinov2_vitl14_pretrain.pth'))
    parser.add_argument('--session', '-sid', default='')
    parser.add_argument('--image_resolution', type=int, default=518)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--hypotheses', '-hyps', type=int, default=64)
    parser.add_argument('--threshold', '-t', type=float, default=10)
    parser.add_argument('--inlieralpha', '-ia', type=float, default=100)
    parser.add_argument('--maxpixelerror', '-maxerrr', type=float, default=100)
    parser.add_argument('--log_per_frame', type=_strtobool, default=False,
                        help='Print each frame’s rErr (deg) and tErr (cm) after evaluation.')

    opt = parser.parse_args()
    run_evaluation_lmc(opt)
