#!/usr/bin/env python3
"""Training entry point for ACE FCN + point-cloud feature fusion."""

from __future__ import annotations

import json
import logging
import os
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

if hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(line_buffering=True)
    except Exception:
        pass

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ace_fcn_lmc"))

def _setup_cuda_environment_from_argv():
    """Set CUDA_VISIBLE_DEVICES before importing torch, unless caller already set it."""
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

from ace_pointcloud_fusion.options_pointcloud_fusion import get_pointcloud_fusion_train_parser  # noqa: E402
from utils_lmc import _sanitize_tag  # noqa: E402

import torch  # noqa: E402

from ace_pointcloud_fusion.trainer_pointcloud_fusion import (  # noqa: E402
    TrainerACEPointCloudFusion,
    TrainerDINOPointCloudFusion,
)

logging.basicConfig(level=logging.INFO)
_logger = logging.getLogger(__name__)


def _jsonable_config(args):
    cfg = {}
    for key, value in vars(args).items():
        if isinstance(value, Path):
            cfg[key] = str(value)
        else:
            try:
                json.dumps(value)
                cfg[key] = value
            except TypeError:
                cfg[key] = str(value)
    return cfg


def _validate_args(args):
    if args.use_lmc:
        raise ValueError("ace_pointcloud_fusion is a separate path; run with --use_lmc False.")
    if args.image_backbone == "ace_fcn" and not args.encoder_path.exists():
        raise FileNotFoundError(f"ACE encoder weights not found: {args.encoder_path}")
    if args.image_backbone == "dinov2" and not args.dinov2_path.exists():
        raise FileNotFoundError(f"DINOv2 weights not found: {args.dinov2_path}")
    if not args.point_feature_path.exists():
        raise FileNotFoundError(f"Point feature bank not found: {args.point_feature_path}")
    if args.data_backend == "ace" and not (args.scene / "train").exists():
        raise FileNotFoundError(f"ACE backend expects training dir: {args.scene / 'train'}")
    if args.data_backend == "wai" and not (args.scene / "scene_meta.json").exists():
        raise FileNotFoundError(f"WAI backend expects scene_meta.json: {args.scene / 'scene_meta.json'}")


def _apply_runtime_defaults(args):
    if int(getattr(args, "buffer_batch_size", 1)) != 1:
        _logger.warning(
            "PointFusion prototype is forcing buffer_batch_size from %s to 1 "
            "to avoid batched-buffer geometry/order ambiguity.",
            args.buffer_batch_size,
        )
        args.buffer_batch_size = 1


def _build_run_dir(args):
    root = Path(args.experiment_root) if args.experiment_root else Path(__file__).parent / "04_evaluation"
    root = root.resolve()
    if args.experiment_subdir:
        root = root / _sanitize_tag(args.experiment_subdir)
    scene_tag = _sanitize_tag(Path(args.scene).name)
    run_tag = args.run_name
    if run_tag == "auto":
        run_tag = datetime.now().strftime("%Y%m%d_%H%M%S") + f"_{scene_tag}_{args.image_backbone}_pcfusion"
    run_dir = root / "ace_pointcloud_fusion" / _sanitize_tag(run_tag)
    run_dir.mkdir(parents=True, exist_ok=True)
    if getattr(args, "overwrite_run_dir", True) and args.run_name != "auto":
        for child in run_dir.iterdir():
            if child.is_file():
                child.unlink()
            else:
                shutil.rmtree(child)

    suffix = _sanitize_tag(Path(args.output_map).name)
    args.output_map = run_dir / f"best_{suffix}"
    args.run_dir = run_dir
    with (run_dir / "run_config.json").open("w", encoding="utf-8") as f:
        json.dump(_jsonable_config(args), f, indent=2, ensure_ascii=False)
    with (run_dir / "run_command.txt").open("w", encoding="utf-8") as f:
        f.write("python " + " ".join(sys.argv) + "\n")
    return run_dir


def _attach_log_file(run_dir: Path):
    path = run_dir / "training_full_log.txt"
    handler = logging.FileHandler(path, encoding="utf-8")
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s"))
    logging.getLogger().addHandler(handler)
    return path


def main():
    parser = get_pointcloud_fusion_train_parser()
    args = parser.parse_args()
    if getattr(args, "device", "").startswith("cuda:") and os.environ.get("CUDA_VISIBLE_DEVICES"):
        args.device = "cuda:0"

    _validate_args(args)
    _apply_runtime_defaults(args)
    run_dir = _build_run_dir(args)
    log_path = _attach_log_file(run_dir)
    _logger.info("=" * 80)
    _logger.info("ACE FCN + Point-Cloud Fusion Training")
    _logger.info("Image backbone : %s", args.image_backbone)
    _logger.info("Scene         : %s", args.scene)
    _logger.info("Encoder       : %s", args.encoder_path if args.image_backbone == "ace_fcn" else args.dinov2_path)
    _logger.info("Point features: %s", args.point_feature_path)
    _logger.info("Run dir       : %s", run_dir)
    _logger.info("Output        : %s", args.output_map)
    _logger.info("Full log      : %s", log_path)
    _logger.info("=" * 80)

    if args.eval_after_train:
        _logger.warning(
            "eval_after_train=True ignored in train entry; use "
            "python -m ace_pointcloud_fusion.test_pointcloud_fusion after training."
        )

    trainer_cls = TrainerDINOPointCloudFusion if args.image_backbone == "dinov2" else TrainerACEPointCloudFusion
    trainer = trainer_cls(args)
    trainer.train()
    summary = {
        "total_time_sec": time.time() - trainer.training_start,
        "output_map": str(args.output_map),
        "point_feature_path": str(args.point_feature_path),
    }
    with (run_dir / "training_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    _logger.info("Training completed.")


if __name__ == "__main__":
    main()
