#!/usr/bin/env python3
"""Train an ACE scene-coordinate head using EUPE features."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def setup_cuda_environment():
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--device", type=str, default="cuda:0")
    pre_args, _ = pre_parser.parse_known_args()

    device_str = pre_args.device
    if "cuda" in device_str and ":" in device_str:
        gpu_id = device_str.split(":")[-1]
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
        print(f"Info: Set CUDA_VISIBLE_DEVICES = {gpu_id}")
        print("Info: Inside PyTorch, this will be mapped to 'cuda:0'")


setup_cuda_environment()

import torch  # noqa: E402

from ace_eupe.trainer_eupe import TrainerACEEUPE  # noqa: E402
from ace_eupe.ace_network_eupe import BACKBONE_SPECS, backbone_spec  # noqa: E402

logging.basicConfig(level=logging.INFO)
_logger = logging.getLogger(__name__)


def _strtobool(x):
    if isinstance(x, bool):
        return x
    value = str(x).lower()
    if value in ("yes", "true", "t", "y", "1", "on"):
        return True
    if value in ("no", "false", "f", "n", "0", "off"):
        return False
    raise ValueError(f"Invalid boolean value: {x}")


def _sanitize_stem(p: Path) -> str:
    s = p.stem if hasattr(p, "stem") else str(p)
    return "".join(c for c in s if c.isalnum() or c in "._-") or "model"


def _default_eupe_root() -> Path:
    return Path(os.environ.get("EUPE_ROOT", REPO_ROOT.parent / "EUPE"))


def _default_checkpoint(eupe_root: Path, model_name: str) -> Path:
    return eupe_root / "checkpoints" / str(backbone_spec(model_name)["checkpoint"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train ACE with an EUPE ViT or ConvNeXt backbone",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("scene", type=Path, help="Path to ACE scene root, or its train split")
    parser.add_argument("output_map", type=Path, help="Output filename suffix, e.g. best.pt")
    parser.add_argument("--experiment_root", type=Path, default=Path("output"))

    parser.add_argument("--eupe_root", type=Path, default=_default_eupe_root(), help="Path to the EUPE repository")
    parser.add_argument("--eupe_model_name", type=str, default="eupe_vitb16", choices=sorted(BACKBONE_SPECS))
    parser.add_argument(
        "--eupe_checkpoint",
        type=Path,
        default=None,
        help="Path to EUPE weights. Defaults from --eupe_model_name.",
    )
    parser.add_argument("--freeze_backbone", type=_strtobool, default=True)

    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--num_head_blocks", type=int, default=4)
    parser.add_argument("--use_homogeneous", type=_strtobool, default=True)

    parser.add_argument("--training_buffer_size", type=int, default=10000000)
    parser.add_argument("--buffer_batch_size", type=int, default=1)
    parser.add_argument("--num_data_loader_workers", type=int, default=12)
    parser.add_argument("--buffer_image_width", type=int, default=None)
    parser.add_argument("--buffer_sampling_replacement", type=_strtobool, default=True)
    parser.add_argument("--buffer_on_cpu", type=_strtobool, default=False)
    parser.add_argument("--samples_per_image", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=5120)
    parser.add_argument("--learning_rate_min", type=float, default=0.0001)
    parser.add_argument("--learning_rate_max", type=float, default=0.001)

    parser.add_argument("--image_resolution", type=int, default=448, help="Image height; must match the selected EUPE stride")
    parser.add_argument("--use_aug", type=_strtobool, default=True)
    parser.add_argument("--aug_rotation", type=int, default=15)
    parser.add_argument("--aug_scale", type=float, default=1.5)

    parser.add_argument("--repro_loss_type", type=str, default="dyntanh", choices=["l1", "l2", "smooth_l1", "tanh", "dyntanh"])
    parser.add_argument("--repro_loss_soft_clamp", type=float, default=50)
    parser.add_argument("--repro_loss_soft_clamp_min", type=float, default=1)
    parser.add_argument("--repro_loss_schedule", type=str, default="circle", choices=["circle", "linear"])
    parser.add_argument("--repro_loss_hard_clamp", type=float, default=1000)
    parser.add_argument("--depth_min", type=float, default=0.1)
    parser.add_argument("--depth_max", type=float, default=1000)
    parser.add_argument("--depth_target", type=float, default=10)

    parser.add_argument("--use_half", type=_strtobool, default=True)
    parser.add_argument("--eval_after_train", type=_strtobool, default=True)
    parser.add_argument("--eval_session", type=str, default="post_train")
    parser.add_argument("--post_train_eval_device", type=str, default="cuda:0")

    args = parser.parse_args()

    args.eupe_root = Path(args.eupe_root).expanduser().resolve()
    if args.eupe_checkpoint is None:
        args.eupe_checkpoint = _default_checkpoint(args.eupe_root, args.eupe_model_name)
    else:
        args.eupe_checkpoint = Path(args.eupe_checkpoint).expanduser().resolve()
    output_subsample = int(backbone_spec(args.eupe_model_name)["output_subsample"])

    scene_path = Path(args.scene).resolve()
    if scene_path.name in ("train", "test", "val") and len(scene_path.parts) >= 3:
        dataset = scene_path.parent.parent.name
        scene_name = scene_path.parent.name
    else:
        dataset = scene_path.parent.name
        scene_name = scene_path.name

    out_base = Path(args.experiment_root).resolve() / dataset / scene_name
    out_base.mkdir(parents=True, exist_ok=True)
    suffix = _sanitize_stem(args.output_map)
    model_tag = _sanitize_stem(Path(args.eupe_model_name))
    args.output_map = out_base / f"{scene_name}_{model_tag}_ep{args.epochs}_bs{args.batch_size}_{suffix}.pt"
    args._scene_display_name = scene_name

    if scene_path.name == "train":
        args.scene = scene_path.parent
    else:
        args.scene = scene_path

    if args.image_resolution % output_subsample != 0:
        adjusted = (args.image_resolution // output_subsample) * output_subsample
        if adjusted <= 0:
            adjusted = output_subsample
        _logger.warning(
            "Image resolution %s is not divisible by %s for %s; adjusting to %s",
            args.image_resolution,
            output_subsample,
            args.eupe_model_name,
            adjusted,
        )
        args.image_resolution = adjusted

    if not args.eupe_root.exists():
        _logger.error("EUPE repo not found at %s", args.eupe_root)
        sys.exit(1)
    if not args.eupe_checkpoint.exists():
        _logger.error("EUPE weights not found at %s", args.eupe_checkpoint)
        sys.exit(1)

    _logger.info("=" * 80)
    _logger.info("Training ACE with EUPE Encoder")
    _logger.info("=" * 80)
    _logger.info("Scene: %s", args.scene)
    _logger.info("Output: %s", args.output_map)
    _logger.info("EUPE repo: %s", args.eupe_root)
    _logger.info("EUPE model: %s", args.eupe_model_name)
    _logger.info("EUPE weights: %s", args.eupe_checkpoint)
    _logger.info("Freeze backbone: %s", args.freeze_backbone)
    _logger.info("Image resolution: %s", args.image_resolution)
    _logger.info("Epochs: %s", args.epochs)
    _logger.info("Batch size: %s", args.batch_size)
    _logger.info("Device: %s", args.device)
    _logger.info("=" * 80)

    trainer = TrainerACEEUPE(args)
    trainer.train()
    _logger.info("Training completed successfully.")

    if args.eval_after_train:
        try:
            del trainer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            eval_device = getattr(args, "post_train_eval_device", "cuda:0")
            eval_env = os.environ.copy()
            if eval_device.startswith("cuda"):
                gpu_id = eval_device.split(":")[-1]
                eval_env["CUDA_VISIBLE_DEVICES"] = gpu_id
                device_for_cmd = "cuda:0"
            else:
                device_for_cmd = eval_device

            cmd = [
                sys.executable,
                str(Path(__file__).resolve().parent / "test_ace_eupe.py"),
                str(args.scene),
                str(args.output_map),
                "--eupe_root",
                str(args.eupe_root),
                "--eupe_checkpoint",
                str(args.eupe_checkpoint),
                "--eupe_model_name",
                str(args.eupe_model_name),
                "--session",
                str(args.eval_session),
                "--image_resolution",
                str(args.image_resolution),
                "--device",
                device_for_cmd,
            ]
            _logger.info("Eval command: %s", " ".join(cmd))

            json_path = Path(args.output_map).parent / "post_train_eval_eupe.json"
            payload = {
                "cmd": cmd,
                "cwd": os.getcwd(),
                "env": {key: str(value) for key, value in eval_env.items()},
                "output_map": str(args.output_map),
                "scene": str(args.scene),
                "image_resolution": args.image_resolution,
            }
            json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

            launcher = REPO_ROOT / "scripts" / "run_post_train_eval.py"
            if not launcher.exists():
                raise FileNotFoundError(f"Launcher script not found: {launcher}")
            os.execv(sys.executable, [sys.executable, str(launcher), str(json_path)])
        except FileNotFoundError as exc:
            _logger.warning("Post-training evaluation skipped: %s", exc)
        except Exception as exc:
            _logger.warning("Post-training evaluation failed: %s", exc, exc_info=True)
