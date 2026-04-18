#!/usr/bin/env python3
"""Evaluate an ACE head trained on EUPE features."""

from __future__ import annotations

import argparse
import logging
import math
import sys
import time
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.amp import autocast  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

import dsacstar  # noqa: E402
from ace_eupe.ace_network_eupe import BACKBONE_SPECS, Regressor, backbone_spec  # noqa: E402
from ace_eupe.dataset_eupe import CamLocDatasetEUPE  # noqa: E402

_logger = logging.getLogger(__name__)


def _sanitize_tag(text: str) -> str:
    s = "".join(c if c.isalnum() or c in "._-" else "_" for c in str(text))
    return s.strip("._-") or "run"


def _strtobool(x):
    if isinstance(x, bool):
        return x
    value = str(x).lower()
    if value in ("1", "true", "yes", "on"):
        return True
    if value in ("0", "false", "no", "off"):
        return False
    raise ValueError(f"Invalid truth value: {x!r}")


def run_evaluation(opt):
    hypotheses = getattr(opt, "hypotheses", 64)
    threshold = getattr(opt, "threshold", 10)
    inlieralpha = getattr(opt, "inlieralpha", 100)
    maxpixelerror = getattr(opt, "maxpixelerror", 100)

    output_subsample = int(backbone_spec(opt.eupe_model_name)["output_subsample"])
    if opt.image_resolution % output_subsample != 0:
        opt.image_resolution = max(output_subsample, (opt.image_resolution // output_subsample) * output_subsample)
        _logger.warning(
            "Image resolution adjusted to %s for %s stride %s",
            opt.image_resolution,
            opt.eupe_model_name,
            output_subsample,
        )

    device = torch.device(opt.device)
    scene_path = Path(opt.scene)
    head_network_path = Path(opt.network)
    eupe_root = Path(opt.eupe_root)
    eupe_checkpoint = Path(opt.eupe_checkpoint)
    session = opt.session

    if not head_network_path.exists():
        raise FileNotFoundError(f"Head network not found: {head_network_path}")
    if not eupe_checkpoint.exists():
        raise FileNotFoundError(f"EUPE checkpoint not found: {eupe_checkpoint}")

    testset = CamLocDatasetEUPE(
        scene_path / "test",
        mode=0,
        use_half=False,
        image_height=opt.image_resolution,
        output_subsample=output_subsample,
        augment=False,
    )
    _logger.info("Test images found: %s", len(testset))
    testset_loader = DataLoader(testset, shuffle=False, num_workers=getattr(opt, "num_workers", 6))

    head_state_dict = torch.load(head_network_path, map_location="cpu")
    network = Regressor.create_from_split_state_dict(
        eupe_root=eupe_root,
        eupe_checkpoint=eupe_checkpoint,
        eupe_model_name=opt.eupe_model_name,
        head_state_dict=head_state_dict,
    )
    network = network.to(device)
    network.eval()

    if getattr(opt, "eval_output_dir", None):
        output_dir = Path(opt.eval_output_dir).resolve()
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        eval_run_id = f"{timestamp}_{_sanitize_tag(scene_path.name)}_{_sanitize_tag(session) if session else 'eval'}"
        eval_results_base = head_network_path.parent / "eval_results"
        eval_results_base.mkdir(parents=True, exist_ok=True)
        (eval_results_base / "last_eval_dir.txt").write_text(eval_run_id + "\n", encoding="utf-8")
        output_dir = (eval_results_base / eval_run_id).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    test_log_file = output_dir / f"test_{scene_path.name}_{session}.txt"
    pose_log_file = output_dir / f"poses_{scene_path.name}_{session}.txt"

    avg_batch_time = 0.0
    num_batches = 0
    r_errs = []
    t_errs = []
    pct25_5 = pct10_5 = pct5 = pct2 = pct1 = 0

    with test_log_file.open("w", buffering=1, encoding="utf-8") as test_log, pose_log_file.open(
        "w", buffering=1, encoding="utf-8"
    ) as pose_log:
        with torch.no_grad():
            for image_bchw, _, gt_pose_b44, _, intrinsics_b33, _, _, filenames in testset_loader:
                batch_start_time = time.time()
                image_bchw = image_bchw.to(device, non_blocking=True)

                with autocast("cuda", enabled=device.type == "cuda"):
                    scene_coordinates_b3hw = network(image_bchw)
                scene_coordinates_b3hw = scene_coordinates_b3hw.float().cpu()

                if isinstance(filenames, str):
                    filenames = (filenames,)
                for scene_coordinates_3hw, gt_pose_44, intrinsics_33, frame_path in zip(
                    scene_coordinates_b3hw, gt_pose_b44, intrinsics_b33, filenames
                ):
                    fx, fy = intrinsics_33[0, 0].item(), intrinsics_33[1, 1].item()
                    focal_length = (fx + fy) / 2.0
                    pp_x = intrinsics_33[0, 2].item()
                    pp_y = intrinsics_33[1, 2].item()

                    out_pose = torch.zeros((4, 4))
                    inlier_count = dsacstar.forward_rgb(
                        scene_coordinates_3hw.unsqueeze(0),
                        out_pose,
                        hypotheses,
                        threshold,
                        focal_length,
                        pp_x,
                        pp_y,
                        inlieralpha,
                        maxpixelerror,
                        network.OUTPUT_SUBSAMPLE,
                    )

                    t_err = float(torch.norm(gt_pose_44[0:3, 3] - out_pose[0:3, 3]))
                    gt_r = gt_pose_44[0:3, 0:3].numpy()
                    out_r = out_pose[0:3, 0:3].numpy()
                    r_err = np.matmul(out_r, np.transpose(gt_r))
                    r_err = cv2.Rodrigues(r_err)[0]
                    r_err = np.linalg.norm(r_err) * 180 / math.pi

                    _logger.info("Rotation Error: %.2fdeg, Translation Error: %.1fcm", r_err, t_err * 100)

                    r_errs.append(r_err)
                    t_errs.append(t_err * 100)
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
                    translation = out_pose[0:3, 3]
                    rot, _ = cv2.Rodrigues(out_pose[0:3, 0:3].numpy())
                    angle = np.linalg.norm(rot)
                    axis = rot / angle if angle > 1e-12 else rot
                    q_w = math.cos(angle * 0.5)
                    q_xyz = math.sin(angle * 0.5) * axis
                    pose_log.write(
                        f"{Path(frame_path).name} "
                        f"{q_w} {q_xyz[0].item()} {q_xyz[1].item()} {q_xyz[2].item()} "
                        f"{translation[0]} {translation[1]} {translation[2]} "
                        f"{r_err} {t_err} {inlier_count}\n"
                    )

                avg_batch_time += time.time() - batch_start_time
                num_batches += 1

        total_frames = len(r_errs)
        if total_frames == 0:
            raise RuntimeError("No test frames were evaluated.")

        t_errs.sort()
        r_errs.sort()
        median_idx = total_frames // 2
        median_r_err = r_errs[median_idx]
        median_t_err = t_errs[median_idx]
        avg_time = avg_batch_time / max(num_batches, 1)

        pct25_5 = pct25_5 / total_frames * 100
        pct10_5 = pct10_5 / total_frames * 100
        pct5 = pct5 / total_frames * 100
        pct2 = pct2 / total_frames * 100
        pct1 = pct1 / total_frames * 100

        test_log.write(f"{median_r_err} {median_t_err} {avg_time}\n")

    eval_summary_file = output_dir / f"eval_summary_{scene_path.name}_{session}.txt"
    eval_summary_file.write_text(
        "\n".join(
            [
                f"# EUPE-ACE Eval Summary | {scene_path.name}",
                f"median_rotation_deg\t{median_r_err:.4f}",
                f"median_translation_cm\t{median_t_err:.4f}",
                f"accuracy_25cm5deg_pct\t{pct25_5:.2f}",
                f"accuracy_10cm5deg_pct\t{pct10_5:.2f}",
                f"accuracy_5cm5deg_pct\t{pct5:.2f}",
                f"accuracy_2cm2deg_pct\t{pct2:.2f}",
                f"accuracy_1cm1deg_pct\t{pct1:.2f}",
                f"avg_time_per_frame_ms\t{avg_time * 1000:.2f}",
                f"total_frames\t{total_frames}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    _logger.info("===================================================")
    _logger.info("EUPE EVAL SUMMARY")
    _logger.info("Median Error: %.2f deg, %.2f cm", median_r_err, median_t_err)
    _logger.info(
        "25cm/5deg: %.2f%% | 10cm/5deg: %.2f%% | 5cm/5deg: %.2f%% | 2cm/2deg: %.2f%% | 1cm/1deg: %.2f%%",
        pct25_5,
        pct10_5,
        pct5,
        pct2,
        pct1,
    )
    _logger.info("Eval summary written to: %s", eval_summary_file)

    return {
        "median_rErr": median_r_err,
        "median_tErr": median_t_err,
        "avg_time": avg_time,
        "total_frames": total_frames,
        "test_log_file": str(test_log_file),
        "pose_log_file": str(pose_log_file),
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    default_eupe_root = Path(__file__).resolve().parents[2] / "EUPE"
    parser = argparse.ArgumentParser(
        description="Test a trained EUPE-ACE head on a scene.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("scene", type=Path)
    parser.add_argument("network", type=Path, help="Path to trained head weights")
    parser.add_argument("--eupe_root", type=Path, default=default_eupe_root)
    parser.add_argument("--eupe_model_name", type=str, default="eupe_vitb16", choices=sorted(BACKBONE_SPECS))
    parser.add_argument("--eupe_checkpoint", type=Path, default=None)
    parser.add_argument("--session", "-sid", default="")
    parser.add_argument("--image_resolution", type=int, default=448)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--eval_output_dir", type=Path, default=None)
    parser.add_argument("--num_workers", type=int, default=6)

    parser.add_argument("--hypotheses", "-hyps", type=int, default=64)
    parser.add_argument("--threshold", "-t", type=float, default=10)
    parser.add_argument("--inlieralpha", "-ia", type=float, default=100)
    parser.add_argument("--maxpixelerror", "-maxerrr", type=float, default=100)
    parser.add_argument("--render_visualization", type=_strtobool, default=False, help="Reserved for CLI compatibility")

    opt = parser.parse_args()
    opt.eupe_root = Path(opt.eupe_root).expanduser().resolve()
    if opt.eupe_checkpoint is None:
        opt.eupe_checkpoint = opt.eupe_root / "checkpoints" / str(backbone_spec(opt.eupe_model_name)["checkpoint"])
    else:
        opt.eupe_checkpoint = Path(opt.eupe_checkpoint).expanduser().resolve()

    if opt.render_visualization:
        _logger.warning("EUPE test script currently ignores --render_visualization.")

    run_evaluation(opt)
