#!/usr/bin/env python3
# test_ace_dinov2.py
# Testing script for ACE with DINOv2 encoder.
# Structure and logic mirror test_ace.py; only encoder/dataset/args are adapted for DINOv2.

import argparse
import logging
import math
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.amp import autocast
from torch.utils.data import DataLoader

import dsacstar
from ace_network_dinov2 import Regressor
from dataset_dinov2 import CamLocDatasetDINOv2

import ace_vis_util as vutil
from ace_visualizer import ACEVisualizer

_logger = logging.getLogger(__name__)


def _sanitize_tag(text: str) -> str:
    """Filesystem-safe tag for eval run directory names (alnum, ._- only)."""
    s = "".join(c if c.isalnum() or c in "._-" else "_" for c in str(text))
    return s.strip("._-") or "run"


def _strtobool(x):
    """Parse common string values to bool (replacement for deprecated distutils.util.strtobool)."""
    if isinstance(x, bool):
        return x
    v = str(x).strip().lower()
    if v in ('1', 'true', 'yes', 'on'):
        return True
    if v in ('0', 'false', 'no', 'off'):
        return False
    raise ValueError(f"Invalid truth value: {x!r}")


def run_evaluation(opt):
    """
    Run evaluation on a scene with a trained head. Same logic as when running this script from CLI.
    opt: object with .scene, .network, .dinov2_path, .device, .image_resolution, .session,
         and optionally .hypotheses, .threshold, .inlieralpha, .maxpixelerror, .render_*.
    Returns: dict with median_rErr, median_tErr, avg_time, pct25_5, pct10_5, pct5, pct2, pct1,
             total_frames, test_log_file, pose_log_file.
    """
    # Optional args with defaults for when called from train script.
    hypotheses = getattr(opt, 'hypotheses', 64)
    threshold = getattr(opt, 'threshold', 10)
    inlieralpha = getattr(opt, 'inlieralpha', 100)
    maxpixelerror = getattr(opt, 'maxpixelerror', 100)
    render_visualization = getattr(opt, 'render_visualization', False)

    # Ensure image resolution is multiple of 14 for DINOv2.
    if opt.image_resolution % 14 != 0:
        opt.image_resolution = (opt.image_resolution // 14) * 14
        _logger.warning(f"Image resolution adjusted to {opt.image_resolution}")

    device = torch.device(opt.device)
    num_workers = 6

    scene_path = Path(opt.scene)
    head_network_path = Path(opt.network)
    dinov2_path = Path(opt.dinov2_path)
    session = opt.session

    if not head_network_path.exists():
        _logger.error(f"Head network not found at {head_network_path}")
        raise FileNotFoundError(f"Head network not found at {head_network_path}")
    if not dinov2_path.exists():
        _logger.error(f"DINOv2 weights not found at {dinov2_path}")
        raise FileNotFoundError(f"DINOv2 weights not found at {dinov2_path}")

    # Setup dataset.
    testset = CamLocDatasetDINOv2(
        scene_path / "test",
        mode=0,
        use_half=False,
        image_height=opt.image_resolution,
        augment=False,
    )
    _logger.info(f'Test images found: {len(testset)}')

    testset_loader = DataLoader(testset, shuffle=False, num_workers=num_workers)

    head_state_dict = torch.load(head_network_path, map_location="cpu")
    _logger.info(f"Loaded head weights from: {head_network_path}")
    _logger.info(f"Using DINOv2 encoder from: {dinov2_path}")

    network = Regressor.create_from_split_state_dict(
        dinov2_path=dinov2_path,
        head_state_dict=head_state_dict,
    )
    network = network.to(device)
    network.eval()

    output_dir = head_network_path.parent
    scene_name = scene_path.name
    test_log_file = output_dir / f'test_{scene_name}_{session}.txt'
    _logger.info(f"Saving test aggregate statistics to: {test_log_file}")
    pose_log_file = output_dir / f'poses_{scene_name}_{session}.txt'
    _logger.info(f"Saving per-frame poses and errors to: {pose_log_file}")

    test_log = open(test_log_file, 'w', 1)
    pose_log = open(pose_log_file, 'w', 1)

    avg_batch_time = 0
    num_batches = 0
    rErrs = []
    tErrs = []
    pct25_5 = 0
    pct10_5 = 0
    pct5 = 0
    pct2 = 0
    pct1 = 0

    if render_visualization:
        target_path = vutil.get_rendering_target_path(
            getattr(opt, 'render_target_path', Path('renderings')),
            opt.network)
        ace_visualizer = ACEVisualizer(target_path,
                                       getattr(opt, 'render_flipped_portrait', False),
                                       getattr(opt, 'render_map_depth_filter', 10),
                                       reloc_vis_error_threshold=getattr(opt, 'render_pose_error_threshold', 20))
        trainset = CamLocDatasetDINOv2(
            scene_path / "train",
            mode=0,
            use_half=False,
            image_height=opt.image_resolution,
            augment=False,
        )
        trainset_loader = DataLoader(trainset, shuffle=False, num_workers=num_workers)
        ace_visualizer.setup_reloc_visualisation(
            frame_count=len(testset),
            data_loader=trainset_loader,
            network=network,
            camera_z_offset=getattr(opt, 'render_camera_z_offset', 4),
            reloc_frame_skip=getattr(opt, 'render_frame_skip', 1))
    else:
        ace_visualizer = None

    with torch.no_grad():
        for image_B1HW, _, gt_pose_B44, _, intrinsics_B33, _, _, filenames in testset_loader:
            batch_start_time = time.time()
            image_B1HW = image_B1HW.to(device, non_blocking=True)

            with autocast("cuda", enabled=True):
                scene_coordinates_B3HW = network(image_B1HW)
            scene_coordinates_B3HW = scene_coordinates_B3HW.float().cpu()

            if isinstance(filenames, str):
                filenames = (filenames,)
            for frame_idx, (scene_coordinates_3HW, gt_pose_44, intrinsics_33, frame_path) in enumerate(
                    zip(scene_coordinates_B3HW, gt_pose_B44, intrinsics_B33, filenames)):

                # Single focal for dsacstar; use mean of fx/fy when they differ (e.g. indoor6).
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
                r_err = np.matmul(out_R, np.transpose(gt_R))
                r_err = cv2.Rodrigues(r_err)[0]
                r_err = np.linalg.norm(r_err) * 180 / math.pi

                _logger.info(f"Rotation Error: {r_err:.2f}deg, Translation Error: {t_err * 100:.1f}cm")

                if ace_visualizer is not None:
                    ace_visualizer.render_reloc_frame(
                        query_pose=gt_pose_44.numpy(),
                        query_file=frame_path,
                        est_pose=out_pose.numpy(),
                        est_error=max(r_err, t_err*100),
                        sparse_query=getattr(opt, 'render_sparse_queries', False))

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
    _logger.info("EVAL SUMMARY (current errors):")
    _logger.info("  Median Error: %.2f deg, %.2f cm", median_rErr, median_tErr)
    _logger.info("  25cm/5deg: %.2f%% | 10cm/5deg: %.2f%% | 5cm/5deg: %.2f%% | 2cm/2deg: %.2f%% | 1cm/1deg: %.2f%%",
                 pct25_5, pct10_5, pct5, pct2, pct1)
    _logger.info("  Avg time: %.2f ms | Frames: %d", avg_time * 1000, total_frames)
    _logger.info("===================================================")

    test_log.write(f"{median_rErr} {median_tErr} {avg_time}\n")
    test_log.close()
    pose_log.close()

    # 写入当前误差汇总，便于训练/复现时查看
    eval_summary_file = output_dir / f"eval_summary_{scene_name}_{session}.txt"
    summary_lines = [
        f"# DINOv2-ACE Eval Summary | {scene_name}",
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


if __name__ == '__main__':
    # Setup logging.
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(
        description='Test a trained DINOv2-ACE network on a specific scene.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('scene', type=Path,
                        help='path to a scene in the dataset folder, e.g. "datasets/Cambridge_GreatCourt"')

    parser.add_argument('network', type=Path, help='path to a network trained for the scene (just the head weights)')

    parser.add_argument('--dinov2_path', type=Path,
                        default=Path('/mnt/storage/xwh/checkpoints/dinov2_vitl14_pretrain.pth'),
                        help='path to DINOv2 pretrained encoder weights')

    parser.add_argument('--session', '-sid', default='',
                        help='custom session name appended to output files, '
                             'useful to separate different runs of a script')

    parser.add_argument('--image_resolution', type=int, default=518,
                        help='base image resolution (must be multiple of 14 for DINOv2)')

    parser.add_argument('--device', type=str, default='cuda',
                        help='device to run on, e.g. cuda or cuda:0')

    parser.add_argument('--eval_output_dir', type=Path, default=None,
                        help='override: save eval outputs here; default is <model_dir>/eval_results/<session>/')

    # DSACStar RANSAC parameters. Same as test_ace.py.
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

    # Params for the visualization. Same as test_ace.py.
    parser.add_argument('--render_visualization', type=_strtobool, default=False,
                        help='create a video of the mapping process')

    parser.add_argument('--render_target_path', type=Path, default='renderings',
                        help='target folder for renderings, visualizer will create a subfolder with the map name')

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

    opt = parser.parse_args()

    # Ensure image resolution is multiple of 14 for DINOv2.
    if opt.image_resolution % 14 != 0:
        opt.image_resolution = (opt.image_resolution // 14) * 14
        _logger.warning(f"Image resolution adjusted to {opt.image_resolution}")

    device = torch.device(opt.device)
    num_workers = 6

    scene_path = Path(opt.scene)
    head_network_path = Path(opt.network)
    dinov2_path = Path(opt.dinov2_path)
    session = opt.session

    if not head_network_path.exists():
        _logger.error(f"Head network not found at {head_network_path}")
        raise SystemExit(1)
    if not dinov2_path.exists():
        _logger.error(f"DINOv2 weights not found at {dinov2_path}")
        raise SystemExit(1)

    # Setup dataset. Same structure as test_ace.py, dataset swapped for DINOv2.
    testset = CamLocDatasetDINOv2(
        scene_path / "test",
        mode=0,
        use_half=False,
        image_height=opt.image_resolution,
        augment=False,
    )
    _logger.info(f'Test images found: {len(testset)}')

    # Setup dataloader. Batch size 1 by default, same as test_ace.py.
    testset_loader = DataLoader(testset, shuffle=False, num_workers=num_workers)

    # Load network weights. DINOv2: head only from file; encoder from dinov2_path.
    head_state_dict = torch.load(head_network_path, map_location="cpu")
    _logger.info(f"Loaded head weights from: {head_network_path}")
    _logger.info(f"Using DINOv2 encoder from: {dinov2_path}")

    # Create regressor. Same API as test_ace.py, different creation for DINOv2.
    network = Regressor.create_from_split_state_dict(
        dinov2_path=dinov2_path,
        head_state_dict=head_state_dict,
    )

    # Setup for evaluation.
    network = network.to(device)
    network.eval()

    # Output dir: <model_dir>/eval_results/<timestamp>_<scene>_<session>/ (like train_ace_dinov2_lmc run_id).
    scene_name = scene_path.name
    if getattr(opt, 'eval_output_dir', None):
        output_dir = Path(opt.eval_output_dir).resolve()
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_tag = _sanitize_tag(session) if session else "eval"
        eval_run_id = f"{timestamp}_{_sanitize_tag(scene_name)}_{session_tag}"
        eval_results_base = head_network_path.parent / "eval_results"
        output_dir = (eval_results_base / eval_run_id).resolve()
        eval_results_base.mkdir(parents=True, exist_ok=True)
        (eval_results_base / "last_eval_dir.txt").write_text(eval_run_id + "\n", encoding="utf-8")
    output_dir.mkdir(parents=True, exist_ok=True)
    _logger.info("Eval outputs saved to: %s", output_dir)
    test_log_file = output_dir / f'test_{scene_name}_{session}.txt'
    _logger.info(f"Saving test aggregate statistics to: {test_log_file}")
    pose_log_file = output_dir / f'poses_{scene_name}_{session}.txt'
    _logger.info(f"Saving per-frame poses and errors to: {pose_log_file}")

    # Setup output files.
    test_log = open(test_log_file, 'w', 1)
    pose_log = open(pose_log_file, 'w', 1)

    # Metrics of interest. Same as test_ace.py.
    avg_batch_time = 0
    num_batches = 0

    rErrs = []
    tErrs = []

    pct25_5 = 0
    pct10_5 = 0
    pct5 = 0
    pct2 = 0
    pct1 = 0

    # Generate video of training process. Same block as test_ace.py, dataset swapped for DINOv2.
    if opt.render_visualization:
        target_path = vutil.get_rendering_target_path(
            opt.render_target_path,
            opt.network)
        ace_visualizer = ACEVisualizer(target_path,
                                       opt.render_flipped_portrait,
                                       opt.render_map_depth_filter,
                                       reloc_vis_error_threshold=opt.render_pose_error_threshold)

        trainset = CamLocDatasetDINOv2(
            scene_path / "train",
            mode=0,
            use_half=False,
            image_height=opt.image_resolution,
            augment=False,
        )
        trainset_loader = DataLoader(trainset, shuffle=False, num_workers=num_workers)

        ace_visualizer.setup_reloc_visualisation(
            frame_count=len(testset),
            data_loader=trainset_loader,
            network=network,
            camera_z_offset=opt.render_camera_z_offset,
            reloc_frame_skip=opt.render_frame_skip)
    else:
        ace_visualizer = None

    # Testing loop. Same structure and variable names as test_ace.py.
    testing_start_time = time.time()
    with torch.no_grad():
        for image_B1HW, _, gt_pose_B44, _, intrinsics_B33, _, _, filenames in testset_loader:
            batch_start_time = time.time()
            batch_size = image_B1HW.shape[0]

            image_B1HW = image_B1HW.to(device, non_blocking=True)

            # Predict scene coordinates.
            with autocast("cuda", enabled=True):
                scene_coordinates_B3HW = network(image_B1HW)

            # We need them on the CPU to run RANSAC.
            scene_coordinates_B3HW = scene_coordinates_B3HW.float().cpu()

            # Each frame is processed independently. Ensure filenames is iterable (batch_size=1 yields a single string).
            if isinstance(filenames, str):
                filenames = (filenames,)
            for frame_idx, (scene_coordinates_3HW, gt_pose_44, intrinsics_33, frame_path) in enumerate(
                    zip(scene_coordinates_B3HW, gt_pose_B44, intrinsics_B33, filenames)):

                # Extract focal length and principal point from the intrinsics matrix.
                # Single focal for dsacstar; use mean of fx/fy when they differ (e.g. indoor6).
                fx, fy = intrinsics_33[0, 0].item(), intrinsics_33[1, 1].item()
                focal_length = (fx + fy) / 2.0
                ppX = intrinsics_33[0, 2].item()
                ppY = intrinsics_33[1, 2].item()

                # Remove path from file name
                frame_name = Path(frame_path).name

                # Allocate output variable.
                out_pose = torch.zeros((4, 4))

                # Compute the pose via RANSAC. Same call as test_ace.py.
                inlier_count = dsacstar.forward_rgb(
                    scene_coordinates_3HW.unsqueeze(0),
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

                # Calculate translation error.
                t_err = float(torch.norm(gt_pose_44[0:3, 3] - out_pose[0:3, 3]))

                # Rotation error.
                gt_R = gt_pose_44[0:3, 0:3].numpy()
                out_R = out_pose[0:3, 0:3].numpy()

                r_err = np.matmul(out_R, np.transpose(gt_R))
                # Compute angle-axis representation.
                r_err = cv2.Rodrigues(r_err)[0]
                # Extract the angle.
                r_err = np.linalg.norm(r_err) * 180 / math.pi

                _logger.info(f"Rotation Error: {r_err:.2f}deg, Translation Error: {t_err * 100:.1f}cm")

                if ace_visualizer is not None:
                    ace_visualizer.render_reloc_frame(
                        query_pose=gt_pose_44.numpy(),
                        query_file=frame_path,
                        est_pose=out_pose.numpy(),
                        est_error=max(r_err, t_err*100),
                        sparse_query=opt.render_sparse_queries)

                # Save the errors.
                rErrs.append(r_err)
                tErrs.append(t_err * 100)

                # Check various thresholds.
                if r_err < 5 and t_err < 0.25:  # 25cm/5deg
                    pct25_5 += 1
                if r_err < 5 and t_err < 0.1:  # 10cm/5deg
                    pct10_5 += 1
                if r_err < 5 and t_err < 0.05:  # 5cm/5deg
                    pct5 += 1
                if r_err < 2 and t_err < 0.02:  # 2cm/2deg
                    pct2 += 1
                if r_err < 1 and t_err < 0.01:  # 1cm/1deg
                    pct1 += 1

                # Write estimated pose to pose file (inverse).
                out_pose = out_pose.inverse()

                # Translation.
                t = out_pose[0:3, 3]

                # Rotation to axis angle.
                rot, _ = cv2.Rodrigues(out_pose[0:3, 0:3].numpy())
                angle = np.linalg.norm(rot)
                axis = rot / angle

                # Axis angle to quaternion.
                q_w = math.cos(angle * 0.5)
                q_xyz = math.sin(angle * 0.5) * axis

                # Write to output file. All in a single line.
                pose_log.write(f"{frame_name} "
                               f"{q_w} {q_xyz[0].item()} {q_xyz[1].item()} {q_xyz[2].item()} "
                               f"{t[0]} {t[1]} {t[2]} "
                               f"{r_err} {t_err} {inlier_count}\n")

            avg_batch_time += time.time() - batch_start_time
            num_batches += 1

    total_frames = len(rErrs)
    assert total_frames == len(testset)

    # Compute median errors.
    tErrs.sort()
    rErrs.sort()
    median_idx = total_frames // 2
    median_rErr = rErrs[median_idx]
    median_tErr = tErrs[median_idx]

    # Compute average time.
    avg_time = avg_batch_time / num_batches

    # Compute final metrics.
    pct25_5 = pct25_5 / total_frames * 100
    pct10_5 = pct10_5 / total_frames * 100
    pct5 = pct5 / total_frames * 100
    pct2 = pct2 / total_frames * 100
    pct1 = pct1 / total_frames * 100

    _logger.info("===================================================")
    _logger.info("Test complete.")

    _logger.info('Accuracy:')
    _logger.info(f'\t25cm/5deg: {pct25_5:.1f}%')
    _logger.info(f'\t10cm/5deg: {pct10_5:.1f}%')
    _logger.info(f'\t5cm/5deg: {pct5:.1f}%')
    _logger.info(f'\t2cm/2deg: {pct2:.1f}%')
    _logger.info(f'\t1cm/1deg: {pct1:.1f}%')

    _logger.info(f"Median Error: {median_rErr:.1f}deg, {median_tErr:.1f}cm")
    _logger.info(f"Avg. processing time: {avg_time * 1000:4.1f}ms")

    # Write to the test log file as well.
    test_log.write(f"{median_rErr} {median_tErr} {avg_time}\n")

    test_log.close()
    pose_log.close()
