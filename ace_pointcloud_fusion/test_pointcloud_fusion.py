#!/usr/bin/env python3
"""Evaluate an ACE/DINO + point-cloud fusion checkpoint."""

from __future__ import annotations

import argparse
import logging
import math
import os
import re
import sys
import time
from pathlib import Path

if hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(line_buffering=True)
    except Exception:
        pass


def _setup_cuda_environment_from_argv():
    if os.environ.get("CUDA_VISIBLE_DEVICES"):
        print(f"Info: preserving CUDA_VISIBLE_DEVICES = {os.environ['CUDA_VISIBLE_DEVICES']}", flush=True)
        return
    for idx, arg in enumerate(sys.argv):
        if arg == "--device" and idx + 1 < len(sys.argv):
            device_str = sys.argv[idx + 1]
            break
        if arg.startswith("--device="):
            device_str = arg.split("=", 1)[1]
            break
    else:
        device_str = "cuda:0"
    if "cuda" in device_str and ":" in device_str:
        gpu_id = device_str.split(":")[-1]
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
        print(f"Info: CUDA_VISIBLE_DEVICES = {gpu_id}", flush=True)


_setup_cuda_environment_from_argv()

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ace_fcn_lmc"))

import cv2  # noqa: E402
import dsacstar  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.amp import autocast  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from ace_util import get_pixel_grid  # noqa: E402
from dataset_dinov2 import CamLocDatasetDINOv2  # noqa: E402

from ace_pointcloud_fusion.ace_network_pointcloud import (  # noqa: E402
    RegressorACEPointCloudFusion,
    RegressorDINOPointCloudFusion,
)
from ace_pointcloud_fusion.utonia_encoder import load_feature_bank  # noqa: E402

_logger = logging.getLogger(__name__)


def _infer_head_config(head_state_dict):
    pattern = re.compile(r"^\d+c0\.weight$")
    num_head_blocks = sum(1 for key in head_state_dict if pattern.match(key))
    use_homogeneous = bool(head_state_dict["fc3.weight"].shape[0] == 4)
    in_channels = int(head_state_dict["res3_conv1.weight"].shape[1])
    return num_head_blocks, use_homogeneous, in_channels


def _make_world_rays(pixel_grid_2hw, pose_b44, intrinsics_inv_b33):
    bsz, _, height, width = pixel_grid_2hw.shape
    px_bhw2 = pixel_grid_2hw.permute(0, 2, 3, 1).reshape(bsz, height * width, 2)
    ones = torch.ones((bsz, height * width, 1), device=px_bhw2.device, dtype=px_bhw2.dtype)
    px_h = torch.cat([px_bhw2, ones], dim=-1)
    ray_cam = torch.matmul(px_h, intrinsics_inv_b33.transpose(1, 2))
    rot_c2w = pose_b44[:, :3, :3]
    ray_world = torch.matmul(ray_cam, rot_c2w.transpose(1, 2))
    ray_world = torch.nn.functional.normalize(ray_world, dim=-1, eps=1e-6)
    centers = pose_b44[:, None, :3, 3].expand(bsz, height * width, 3)
    return centers.reshape(-1, 3), ray_world.reshape(-1, 3)


def _quaternion_from_pose_inv(out_pose_inv):
    t = out_pose_inv[0:3, 3]
    rot, _ = cv2.Rodrigues(out_pose_inv[0:3, 0:3].numpy())
    angle = float(np.linalg.norm(rot))
    if angle < 1e-12:
        return t, 1.0, np.zeros((3, 1), dtype=np.float32)
    axis = rot / angle
    return t, math.cos(angle * 0.5), math.sin(angle * 0.5) * axis


def run_evaluation(opt):
    device = torch.device(opt.device)
    scene_path = Path(opt.scene)
    network_path = Path(opt.network)
    encoder_path = Path(opt.encoder_path)

    checkpoint = torch.load(network_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Not a point-cloud fusion checkpoint: {network_path}")
    model_type = checkpoint.get("model_type")
    image_backbone = checkpoint.get("image_backbone")
    if model_type == "ace_pointcloud_fusion":
        image_backbone = "ace_fcn"
    elif model_type == "dinov2_pointcloud_fusion":
        image_backbone = "dinov2"
    if image_backbone not in {"ace_fcn", "dinov2"}:
        raise ValueError(f"Unsupported point-cloud fusion checkpoint type: {model_type!r}")

    head_state = checkpoint["heads_state_dict"]
    adaptor_state = checkpoint["adaptor_state_dict"]
    point_feature_path = Path(getattr(opt, "point_feature_path", None) or checkpoint["point_feature_path"])
    point_feature_dim = int(checkpoint.get("point_feature_dim", 0) or torch.load(point_feature_path, map_location="cpu")["features"].shape[-1])
    num_head_blocks, use_homogeneous, image_feature_dim = _infer_head_config(head_state)

    if image_backbone == "dinov2":
        dinov2_path = Path(getattr(opt, "dinov2_path", None) or checkpoint["dinov2_path"])
        network = RegressorDINOPointCloudFusion.create_from_encoder(
            dinov2_path=dinov2_path,
            mean=torch.zeros(3),
            num_head_blocks=num_head_blocks,
            use_homogeneous=use_homogeneous,
            point_feature_dim=point_feature_dim,
            num_encoder_features=image_feature_dim,
            freeze_backbone=True,
            adaptor_hidden_dim=int(opt.adaptor_hidden_dim),
            adaptor_heads=int(opt.adaptor_heads),
            adaptor_dropout=0.0,
            max_context_points=int(opt.max_context_points),
        )
        _logger.info("Loaded DINOv2 point-fusion checkpoint. dinov2_path=%s", dinov2_path)
    else:
        network = RegressorACEPointCloudFusion.create_from_encoder(
            encoder_path=encoder_path,
            mean=torch.zeros(3),
            num_head_blocks=num_head_blocks,
            use_homogeneous=use_homogeneous,
            point_feature_dim=point_feature_dim,
            num_encoder_features=image_feature_dim,
            freeze_backbone=True,
            adaptor_hidden_dim=int(opt.adaptor_hidden_dim),
            adaptor_heads=int(opt.adaptor_heads),
            adaptor_dropout=0.0,
            max_context_points=int(opt.max_context_points),
        )
        _logger.info("Loaded ACE-FCN point-fusion checkpoint. encoder_path=%s", encoder_path)
    network.heads.load_state_dict(head_state)
    network.adaptor.load_state_dict(adaptor_state)
    network = network.to(device)
    network.eval()

    point_bank = load_feature_bank(point_feature_path, device=device)
    _logger.info("Loaded point feature bank: %s points=%d dim=%d",
                 point_feature_path, point_bank["points"].shape[0], point_bank["features"].shape[1])

    testset = CamLocDatasetDINOv2(
        scene_path / "test",
        mode=0,
        use_half=False,
        image_height=int(opt.image_resolution),
        augment=False,
    )
    test_loader = DataLoader(testset, shuffle=False, num_workers=int(opt.num_workers), batch_size=1)
    _logger.info("Test images found: %d", len(testset))

    output_dir = network_path.parent
    scene_name = scene_path.name
    session = getattr(opt, "session", "")
    test_log_file = output_dir / f"test_{scene_name}_{session}.txt"
    pose_log_file = output_dir / f"poses_{scene_name}_{session}.txt"
    _logger.info("Saving test stats to: %s", test_log_file)
    _logger.info("Saving per-frame poses to: %s", pose_log_file)

    pixel_grid_2hw_full = get_pixel_grid(network.OUTPUT_SUBSAMPLE).to(device)

    avg_batch_time = 0.0
    num_batches = 0
    r_errs = []
    t_errs = []
    pct25_5 = pct10_5 = pct5 = pct2 = pct1 = 0

    with test_log_file.open("w", encoding="utf-8", buffering=1) as test_log, \
            pose_log_file.open("w", encoding="utf-8", buffering=1) as pose_log, \
            torch.no_grad():
        for image_bchw, _, gt_pose_b44, _, intrinsics_b33, intrinsics_inv_b33, _, filenames in test_loader:
            batch_start = time.time()
            image_bchw = image_bchw.to(device, non_blocking=True)
            pose_b44 = gt_pose_b44.to(device, non_blocking=True)
            intrinsics_inv_b33 = intrinsics_inv_b33.to(device, non_blocking=True)

            with autocast(device_type=device.type, enabled=device.type == "cuda"):
                features = network.get_features(image_bchw)
                bsz, channels, height, width = features.shape
                pixel_grid_2hw = pixel_grid_2hw_full[:, :height, :width].unsqueeze(0).expand(bsz, 2, height, width)
                camera_centers, ray_dirs = _make_world_rays(pixel_grid_2hw, pose_b44, intrinsics_inv_b33)
                features_nc = features.permute(0, 2, 3, 1).reshape(-1, channels)
                fused_nc = network.fuse_buffer_features(features_nc, camera_centers, ray_dirs, point_bank)
                fused_bchw = fused_nc.reshape(bsz, height, width, channels).permute(0, 3, 1, 2)
                scene_coordinates_b3hw = network.get_scene_coordinates(fused_bchw)

            scene_coordinates_b3hw = scene_coordinates_b3hw.float().cpu()
            if isinstance(filenames, str):
                filenames = (filenames,)

            for scene_coordinates_3hw, gt_pose_44, intrinsics_33, frame_path in zip(
                    scene_coordinates_b3hw, gt_pose_b44, intrinsics_b33, filenames):
                fx, fy = intrinsics_33[0, 0].item(), intrinsics_33[1, 1].item()
                focal_length = (fx + fy) / 2.0
                pp_x = intrinsics_33[0, 2].item()
                pp_y = intrinsics_33[1, 2].item()

                out_pose = torch.zeros((4, 4))
                inlier_count = dsacstar.forward_rgb(
                    scene_coordinates_3hw.unsqueeze(0),
                    out_pose,
                    int(opt.hypotheses),
                    float(opt.threshold),
                    focal_length,
                    pp_x,
                    pp_y,
                    float(opt.inlieralpha),
                    float(opt.maxpixelerror),
                    network.OUTPUT_SUBSAMPLE,
                )

                t_err = float(torch.norm(gt_pose_44[0:3, 3] - out_pose[0:3, 3]))
                gt_r = gt_pose_44[0:3, 0:3].numpy()
                out_r = out_pose[0:3, 0:3].numpy()
                r_err = float(np.linalg.norm(cv2.Rodrigues(np.matmul(out_r, np.transpose(gt_r)))[0]) * 180 / math.pi)
                t_err_cm = t_err * 100.0

                r_errs.append(r_err)
                t_errs.append(t_err_cm)
                pct25_5 += int(r_err < 5 and t_err < 0.25)
                pct10_5 += int(r_err < 5 and t_err < 0.10)
                pct5 += int(r_err < 5 and t_err < 0.05)
                pct2 += int(r_err < 2 and t_err < 0.02)
                pct1 += int(r_err < 1 and t_err < 0.01)

                frame_name = Path(frame_path).name
                if bool(getattr(opt, "log_per_frame", False)):
                    _logger.info("%s  rErr=%.2fdeg  tErr=%.2fcm  inliers=%s",
                                 frame_name, r_err, t_err_cm, inlier_count)

                out_pose_inv = out_pose.inverse()
                t, q_w, q_xyz = _quaternion_from_pose_inv(out_pose_inv)
                pose_log.write(
                    f"{frame_name} {q_w} {q_xyz[0].item()} {q_xyz[1].item()} {q_xyz[2].item()} "
                    f"{t[0]} {t[1]} {t[2]} {r_err} {t_err} {inlier_count}\n"
                )

            avg_batch_time += time.time() - batch_start
            num_batches += 1

        total_frames = len(r_errs)
        if total_frames == 0:
            raise RuntimeError("No test frames were evaluated.")

        median_r = float(np.median(np.asarray(r_errs)))
        median_t = float(np.median(np.asarray(t_errs)))
        avg_time = avg_batch_time / max(num_batches, 1)
        pct25_5 = pct25_5 / total_frames * 100.0
        pct10_5 = pct10_5 / total_frames * 100.0
        pct5 = pct5 / total_frames * 100.0
        pct2 = pct2 / total_frames * 100.0
        pct1 = pct1 / total_frames * 100.0

        _logger.info("=" * 50)
        _logger.info("EVAL SUMMARY (ACE-FCN + PointFusion):")
        _logger.info("  Median: %.2f deg, %.2f cm", median_r, median_t)
        _logger.info("  25cm/5deg: %.2f%% | 10cm/5deg: %.2f%% | 5cm/5deg: %.2f%%", pct25_5, pct10_5, pct5)
        _logger.info("  2cm/2deg: %.2f%% | 1cm/1deg: %.2f%%", pct2, pct1)
        _logger.info("  Avg time: %.2f ms | Frames: %d", avg_time * 1000.0, total_frames)
        _logger.info("=" * 50)
        test_log.write(f"{median_r} {median_t} {avg_time}\n")

    summary_file = output_dir / f"eval_summary_{scene_name}_{session}.txt"
    summary_file.write_text(
        "\n".join([
            f"# ACE-FCN + PointFusion Eval | {scene_name}",
            f"median_rotation_deg\t{median_r:.4f}",
            f"median_translation_cm\t{median_t:.4f}",
            f"accuracy_25cm5deg_pct\t{pct25_5:.2f}",
            f"accuracy_10cm5deg_pct\t{pct10_5:.2f}",
            f"accuracy_5cm5deg_pct\t{pct5:.2f}",
            f"accuracy_2cm2deg_pct\t{pct2:.2f}",
            f"accuracy_1cm1deg_pct\t{pct1:.2f}",
            f"avg_time_per_frame_ms\t{avg_time * 1000.0:.2f}",
            f"total_frames\t{total_frames}",
        ]) + "\n",
        encoding="utf-8",
    )
    _logger.info("Eval summary written to: %s", summary_file)

    return {
        "median_rErr": median_r,
        "median_tErr": median_t,
        "avg_time": avg_time,
        "pct25_5": pct25_5,
        "pct10_5": pct10_5,
        "pct5": pct5,
        "pct2": pct2,
        "pct1": pct1,
        "total_frames": total_frames,
        "test_log_file": str(test_log_file),
        "pose_log_file": str(pose_log_file),
        "summary_file": str(summary_file),
    }


def get_parser():
    parser = argparse.ArgumentParser(
        description="Test ACE/DINO + point-cloud fusion checkpoint.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("scene", type=Path, help="Scene directory with test split.")
    parser.add_argument("network", type=Path, help="Point-fusion checkpoint.")
    parser.add_argument("--encoder_path", type=Path, default=Path("ace_encoder_pretrained.pt"))
    parser.add_argument(
        "--dinov2_path",
        type=Path,
        default=None,
        help="Override DINOv2 weights path for DINO point-fusion checkpoints.",
    )
    parser.add_argument("--point_feature_path", type=Path, default=None, help="Override point feature bank path in checkpoint.")
    parser.add_argument("--session", "-sid", default="pointfusion")
    parser.add_argument("--image_resolution", type=int, default=480)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--num_workers", type=int, default=6)
    parser.add_argument("--hypotheses", "-hyps", type=int, default=64)
    parser.add_argument("--threshold", "-t", type=float, default=10)
    parser.add_argument("--inlieralpha", "-ia", type=float, default=100)
    parser.add_argument("--maxpixelerror", "-maxerrr", type=float, default=100)
    parser.add_argument("--max_context_points", type=int, default=4096)
    parser.add_argument("--adaptor_hidden_dim", type=int, default=512)
    parser.add_argument("--adaptor_heads", type=int, default=8)
    parser.add_argument("--log_per_frame", action="store_true")
    return parser


def main():
    logging.basicConfig(level=logging.INFO)
    opt = get_parser().parse_args()
    run_evaluation(opt)


if __name__ == "__main__":
    main()
