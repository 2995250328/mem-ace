"""Reusable evaluation function for ACE scene coordinate regression."""
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
from ace_network import Regressor
from dataset_origin import CamLocDataset

_logger = logging.getLogger(__name__)


def evaluate_scene(
    scene: Path,
    network_path: Path,
    encoder_path: Path,
    device: torch.device,
    image_resolution: int = 480,
    hypotheses: int = 64,
    threshold: float = 10,
    inlieralpha: float = 100,
    maxpixelerror: float = 100,
    session: str = '',
) -> dict:
    """Run RANSAC-based pose evaluation on the test split of a scene.

    Returns a dict with keys:
        median_rErr (deg), median_tErr (cm),
        pct25_5, pct10_5, pct5, pct2, pct1 (percentages),
        avg_time_ms, pose_log_file, test_log_file
    """
    scene = Path(scene)
    network_path = Path(network_path)
    encoder_path = Path(encoder_path)

    testset = CamLocDataset(scene / 'test', mode=0, image_height=image_resolution)
    _logger.info(f'Test images found: {len(testset)}')
    testset_loader = DataLoader(testset, shuffle=False, num_workers=6)

    encoder_state_dict = torch.load(encoder_path, map_location='cpu', weights_only=False)
    head_state_dict = torch.load(network_path, map_location='cpu', weights_only=False)
    network = Regressor.create_from_split_state_dict(encoder_state_dict, head_state_dict)
    network = network.to(device).eval()

    output_dir = network_path.parent
    scene_name = scene.name
    test_log_file = output_dir / f'test_{scene_name}_{session}.txt'
    pose_log_file = output_dir / f'poses_{scene_name}_{session}.txt'
    _logger.info(f'Saving test stats to: {test_log_file}')
    _logger.info(f'Saving per-frame poses to: {pose_log_file}')

    test_log = open(test_log_file, 'w', 1)
    pose_log = open(pose_log_file, 'w', 1)

    avg_batch_time = 0
    num_batches = 0
    rErrs, tErrs = [], []
    pct25_5 = pct10_5 = pct5 = pct2 = pct1 = 0

    with torch.no_grad():
        for image_B1HW, _, gt_pose_B44, _, intrinsics_B33, _, _, filenames in testset_loader:
            batch_start = time.time()
            image_B1HW = image_B1HW.to(device, non_blocking=True)

            with autocast('cuda', enabled=True):
                scene_coordinates_B3HW = network(image_B1HW)
            scene_coordinates_B3HW = scene_coordinates_B3HW.float().cpu()

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
                    hypotheses, threshold, focal_length, ppX, ppY,
                    inlieralpha, maxpixelerror, network.OUTPUT_SUBSAMPLE,
                )

                t_err = float(torch.norm(gt_pose_44[0:3, 3] - out_pose[0:3, 3]))
                gt_R = gt_pose_44[0:3, 0:3].numpy()
                out_R = out_pose[0:3, 0:3].numpy()
                r_err = cv2.Rodrigues(np.matmul(out_R, np.transpose(gt_R)))[0]
                r_err = float(np.linalg.norm(r_err) * 180 / math.pi)

                _logger.info(f'Rotation Error: {r_err:.2f}deg, Translation Error: {t_err * 100:.1f}cm')

                rErrs.append(r_err)
                tErrs.append(t_err * 100)

                if r_err < 5 and t_err < 0.25: pct25_5 += 1
                if r_err < 5 and t_err < 0.10: pct10_5 += 1
                if r_err < 5 and t_err < 0.05: pct5 += 1
                if r_err < 2 and t_err < 0.02: pct2 += 1
                if r_err < 1 and t_err < 0.01: pct1 += 1

                out_pose_inv = out_pose.inverse()
                t = out_pose_inv[0:3, 3]
                rot, _ = cv2.Rodrigues(out_pose_inv[0:3, 0:3].numpy())
                angle = np.linalg.norm(rot)
                axis = rot / angle
                q_w = math.cos(angle * 0.5)
                q_xyz = math.sin(angle * 0.5) * axis
                pose_log.write(
                    f"{frame_name} "
                    f"{q_w} {q_xyz[0].item()} {q_xyz[1].item()} {q_xyz[2].item()} "
                    f"{t[0]} {t[1]} {t[2]} "
                    f"{r_err} {t_err} {inlier_count}\n"
                )

            avg_batch_time += time.time() - batch_start
            num_batches += 1

    total_frames = len(rErrs)
    tErrs.sort(); rErrs.sort()
    median_rErr = rErrs[total_frames // 2]
    median_tErr = tErrs[total_frames // 2]
    avg_time = avg_batch_time / num_batches

    n = total_frames
    results = dict(
        median_rErr=median_rErr, median_tErr=median_tErr,
        pct25_5=pct25_5 / n * 100, pct10_5=pct10_5 / n * 100,
        pct5=pct5 / n * 100, pct2=pct2 / n * 100, pct1=pct1 / n * 100,
        avg_time_ms=avg_time * 1000,
        pose_log_file=pose_log_file, test_log_file=test_log_file,
    )

    _logger.info('===================================================')
    _logger.info('Test complete.')
    _logger.info(f"\t25cm/5deg: {results['pct25_5']:.1f}%")
    _logger.info(f"\t10cm/5deg: {results['pct10_5']:.1f}%")
    _logger.info(f"\t5cm/5deg:  {results['pct5']:.1f}%")
    _logger.info(f"\t2cm/2deg:  {results['pct2']:.1f}%")
    _logger.info(f"\t1cm/1deg:  {results['pct1']:.1f}%")
    _logger.info(f"Median Error: {median_rErr:.1f}deg, {median_tErr:.1f}cm")
    _logger.info(f"Avg. processing time: {avg_time * 1000:4.1f}ms")

    test_log.write(f"{median_rErr} {median_tErr} {avg_time}\n")
    test_log.close()
    pose_log.close()

    return results
