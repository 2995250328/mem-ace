#!/usr/bin/env python3
# train_ace_dinov2.py
# Training script for ACE with DINOv2 encoder

import argparse
import logging
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# Setup CUDA environment
def setup_cuda_environment():
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument('--device', type=str, default='cuda:3', help='Target device')
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
    """Convert string to boolean (replacement for distutils.util.strtobool)."""
    if isinstance(x, bool):
        return x
    x_lower = str(x).lower()
    if x_lower in ('yes', 'true', 't', 'y', '1', 'on'):
        return True
    elif x_lower in ('no', 'false', 'f', 'n', '0', 'off'):
        return False
    else:
        raise ValueError(f'Invalid boolean value: {x}')


def _sanitize_stem(p: Path) -> str:
    """文件名安全：只保留字母数字下划线横线。"""
    s = p.stem if hasattr(p, 'stem') else str(p)
    return "".join(c for c in s if c.isalnum() or c in "._-") or "model"


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train ACE with DINOv2 encoder',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Required arguments
    parser.add_argument('scene', type=Path,
                       help='Path to scene folder (e.g., /data/xwh/indoor6_ace/scene1/train)')
    parser.add_argument('output_map', type=Path,
                       help='Output filename suffix (e.g. best.pt). 实际路径为 <experiment_root>/<dataset>/<scene>/<scene>_dinov2_ep*_bs*_<suffix>.pt')

    # Output layout: 实验根目录，下按 数据集/场景 组织
    parser.add_argument('--experiment_root', type=Path, default=Path('output'),
                       help='实验输出根目录；输出为 <experiment_root>/<dataset>/<scene>/ 下按关键参数命名的 .pt')

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
    parser.add_argument('--num_head_blocks', type=int, default=4,
                       help='Depth of regression head')
    parser.add_argument('--use_homogeneous', type=_strtobool, default=True,
                       help='Use homogeneous coordinates')

    # Training settings
    parser.add_argument('--training_buffer_size', type=int, default=10000000,
                       help='Training buffer size')
    parser.add_argument('--buffer_batch_size', type=int, default=1,
                       help='Number of images per forward when filling buffer (1=original ACE one-by-one)')
    parser.add_argument('--buffer_image_width', type=int, default=None,
                       help='Fixed image width for buffer when buffer_batch_size>1 (default: 4:3 from height)')
    parser.add_argument('--samples_per_image', type=int, default=512,
                       help='Features sampled per image for buffer')
    parser.add_argument('--epochs', type=int, default=16,
                       help='Training epochs')
    parser.add_argument('--batch_size', type=int, default=5120,
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
    parser.add_argument('--repro_loss_hard_clamp', type=float, default=1000,
                       help='Hard clamp for reprojection error (px)')
    parser.add_argument('--depth_min', type=float, default=0.1, help='Min depth for valid prediction')
    parser.add_argument('--depth_max', type=float, default=1000, help='Max depth for valid prediction')
    parser.add_argument('--depth_target', type=float, default=10, help='Target depth for invalid proxy loss')

    # Precision
    parser.add_argument('--use_half', type=_strtobool, default=True,
                       help='Use half precision training')

    # Post-training evaluation
    parser.add_argument('--eval_after_train', type=_strtobool, default=True,
                       help='Run evaluation on test set after training and save results to eval_log.txt')
    parser.add_argument('--eval_session', type=str, default='post_train',
                       help='Session name for evaluation output files (test_*, poses_*, used when eval_after_train=True)')
    parser.add_argument('--post_train_eval_device', type=str, default='cuda:0',
                       help='Device for post-train eval subprocess. Default cuda:0 (fast); use cpu if eval OOM.')

    args = parser.parse_args()

    # 输出结构：<experiment_root>/<dataset>/<scene>/<scene>_dinov2_ep{epochs}_bs{batch_size}_{output_map_stem}.pt
    # scene 参数通常为 .../indoor6_ace/scene1/train，则 dataset=indoor6_ace, scene_name=scene1
    scene_path = Path(args.scene).resolve()
    if len(scene_path.parts) >= 2:
        # .../dataset_name/scene_name 或 .../dataset_name/scene_name/train
        if scene_path.name in ("train", "test", "val") and len(scene_path.parts) >= 3:
            dataset = scene_path.parent.parent.name
            scene_name = scene_path.parent.name
        else:
            dataset = scene_path.parent.name
            scene_name = scene_path.name
    else:
        dataset = "default"
        scene_name = scene_path.name or "scene"
    out_base = Path(args.experiment_root).resolve() / dataset / scene_name
    out_base.mkdir(parents=True, exist_ok=True)
    suffix = _sanitize_stem(args.output_map)
    pt_name = f"{scene_name}_dinov2_ep{args.epochs}_bs{args.batch_size}_{suffix}.pt"
    args.output_map = out_base / pt_name
    args._scene_display_name = scene_name  # 用于 eval 日志文件名（test_*_post_train.txt 等）

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

    # Create trainer and train (train() calls save_model internally, same as ace_trainer)
    trainer = TrainerACEDINOv2(args)
    trainer.train()

    _logger.info("Training completed successfully!")

    # Optional: run evaluation on test set and save results
    if args.eval_after_train:
        _logger.info("=" * 80)
        _logger.info("Running post-training evaluation on test set")
        _logger.info("=" * 80)
        try:
            # 父进程已 del trainer 并 empty_cache()，GPU 可被子进程使用；默认 cuda:0 以加速评测。
            del trainer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            eval_device = getattr(args, "post_train_eval_device", "cuda:0")
            if eval_device.startswith("cuda"):
                _logger.info("Running post-train evaluation in subprocess on %s (parent released GPU).", eval_device)
            else:
                _logger.info("Running post-train evaluation in subprocess on %s.", eval_device)

            output_dir = Path(args.output_map).parent
            scene_name = getattr(args, "_scene_display_name", None) or Path(args.scene).parent.name if args.scene.parts and args.scene.name in ("train", "test", "val") else Path(args.scene).name
            eval_session = getattr(args, "eval_session", "post_train")
            script_dir = Path(__file__).resolve().parent
            test_script = script_dir / "test_ace_dinov2.py"

            cmd = [
                sys.executable,
                str(test_script),
                str(args.scene),
                str(args.output_map),
                "--dinov2_path", str(args.dinov2_path),
                "--session", str(eval_session),
                "--image_resolution", str(args.image_resolution),
                "--device", eval_device,
            ]
            _logger.info("Eval command: %s", " ".join(cmd))
            result = subprocess.run(cmd, cwd=os.getcwd(), env=os.environ.copy())
            if result.returncode != 0:
                raise RuntimeError(f"test_ace_dinov2.py exited with code {result.returncode}")

            # 从评测脚本写出的 eval_summary 文件解析结果
            eval_summary_file = output_dir / f"eval_summary_{scene_name}_{eval_session}.txt"
            if not eval_summary_file.exists():
                raise FileNotFoundError(f"Eval summary not found: {eval_summary_file}")

            summary = {}
            for line in eval_summary_file.read_text(encoding="utf-8").strip().splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "\t" in line:
                    key, value = line.split("\t", 1)
                    summary[key.strip()] = value.strip()

            def _float(k, default=0.0):
                return float(summary.get(k, default))

            def _int(k, default=0):
                return int(summary.get(k, default))

            eval_result = {
                "median_rErr": _float("median_rotation_deg"),
                "median_tErr": _float("median_translation_cm"),
                "avg_time": _float("avg_time_per_frame_ms") / 1000.0,
                "pct25_5": _float("accuracy_25cm5deg_pct"),
                "pct10_5": _float("accuracy_10cm5deg_pct"),
                "pct5": _float("accuracy_5cm5deg_pct"),
                "pct2": _float("accuracy_2cm2deg_pct"),
                "pct1": _float("accuracy_1cm1deg_pct"),
                "total_frames": _int("total_frames"),
                "test_log_file": str(output_dir / f"test_{scene_name}_{eval_session}.txt"),
                "pose_log_file": str(output_dir / f"poses_{scene_name}_{eval_session}.txt"),
            }

            eval_log_path = output_dir / "eval_log.txt"
            with open(eval_log_path, "w", encoding="utf-8") as f:
                f.write("# DINOv2-ACE evaluation after training\n")
                f.write(f"# Generated: {datetime.now().isoformat()}\n")
                f.write(f"# Scene: {args.scene}\n")
                f.write(f"# Model: {args.output_map}\n")
                f.write(f"# Epochs: {args.epochs}  image_resolution: {args.image_resolution}\n")
                f.write("\n")
                f.write(f"median_rotation_deg\t{eval_result['median_rErr']:.4f}\n")
                f.write(f"median_translation_cm\t{eval_result['median_tErr']:.4f}\n")
                f.write(f"avg_time_per_frame_ms\t{eval_result['avg_time'] * 1000:.2f}\n")
                f.write(f"accuracy_25cm5deg_pct\t{eval_result['pct25_5']:.2f}\n")
                f.write(f"accuracy_10cm5deg_pct\t{eval_result['pct10_5']:.2f}\n")
                f.write(f"accuracy_5cm5deg_pct\t{eval_result['pct5']:.2f}\n")
                f.write(f"accuracy_2cm2deg_pct\t{eval_result['pct2']:.2f}\n")
                f.write(f"accuracy_1cm1deg_pct\t{eval_result['pct1']:.2f}\n")
                f.write(f"total_frames\t{eval_result['total_frames']}\n")
                f.write(f"test_log_file\t{eval_result['test_log_file']}\n")
                f.write(f"pose_log_file\t{eval_result['pose_log_file']}\n")

            _logger.info(f"Evaluation summary written to: {eval_log_path}")
            _logger.info("========== Post-train Eval (current errors) ==========")
            _logger.info(
                "  Median: %.2f deg, %.2f cm | 25cm/5deg: %.2f%% | 10cm/5deg: %.2f%% | 5cm/5deg: %.2f%% | "
                "2cm/2deg: %.2f%% | 1cm/1deg: %.2f%%",
                eval_result["median_rErr"], eval_result["median_tErr"],
                eval_result["pct25_5"], eval_result["pct10_5"], eval_result["pct5"],
                eval_result["pct2"], eval_result["pct1"],
            )
            _logger.info("  Avg time: %.2f ms | Frames: %d",
                         eval_result["avg_time"] * 1000, eval_result["total_frames"])
            _logger.info("=====================================================")
        except FileNotFoundError as e:
            _logger.warning(f"Post-training evaluation skipped (missing file): {e}")
        except Exception as e:
            _logger.warning(f"Post-training evaluation failed: {e}", exc_info=True)
