#!/usr/bin/env python3
# train_ace_dinov2_lmc.py
# -----------------------------------------------------------------------------
# ACE DINOv2 + GeoLMC 训练入口。
#
# 运行模式：
#   1) Vanilla（不启用 LMC）
#      --use_lmc False
#
#   2) LMC（启用两阶段迭代训练）
#      --use_lmc True --memory_path <pooled_memory.pt>
#
# 关键约束：
#   - memory 文件仅支持 POOLED 格式（需包含 pooled_points / pooled_features）。
#   - 输入分辨率需要是 14 的倍数（DINOv2 patch size = 14）。
# -----------------------------------------------------------------------------

import argparse
import copy
import json
import logging
import numpy as np
import os
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# 让 stderr 行缓冲，便于在 tmux/非 TTY 下实时看到日志
if hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(line_buffering=True)
    except Exception:
        pass

print("train_ace_dinov2_lmc: loading options and CUDA env...", flush=True)
from options_dinov2_lmc import get_lmc_train_parser
from utils_lmc import (
    _format_buf_million,
    _sanitize_stem,
    _sanitize_tag,
    build_lmc_run_folder_config_tag,
    setup_cuda_environment,
)

setup_cuda_environment()

print("train_ace_dinov2_lmc: importing torch and LMC trainer...", flush=True)
import torch
from trainer_dinov2 import TrainerACEDINOv2
from trainer_dinov2_lmc import TrainerACEDINOv2LMC

print("train_ace_dinov2_lmc: ready, configuring logging.", flush=True)
logging.basicConfig(level=logging.INFO)
_logger = logging.getLogger(__name__)

TRAIN_PRESET_DEFAULTS = {
    "memory_compare_ace_g_v1": {
        "use_lmc": True,
        "lmc_flow": "ace_g",
        "lmc_memory_preflight_strict": True,
        "lmc_scene_center_max_distance": 4.0,
        "lmc_head_mean_max_shift": 4.0,
        "bse_denorm_to_world": True,
        "ace_g_fusion_in_s2": True,
        "ace_g_cross_iter_eval": True,
        "buffer_batch_size": 1,
        "samples_per_image": 384,
        "s1_batch_size": 16,
        "s1_use_buffer": True,
        "s1_loss_mode": "sample_per_image",
        "s1_buffer_refill_mode": "full",
        "best_metric": "pct5",
        "eval_deterministic": False,
        "post_train_eval_seeds": [1305],
        "post_train_hypotheses": 64,
        "experiment_root": (Path(__file__).parent / "04_evaluation" / "train_compare").resolve(),
        "experiment_subdir": "memory_pooled_vs_asb",
    },
    "ace_g_indoor6_4090_global_fixedzero_v1": {
        "use_lmc": True,
        "lmc_flow": "ace_g",
        "lmc_mode": "global",
        "lmc_auto_mode_by_visibility": False,
        "lmc_fps_start_policy": "farthest_from_center",
        "lmc_key_slice_idx": 2,
        "s1_loss_step_mode": "fixed_zero",
        "lmc_memory_preflight_strict": True,
        "lmc_scene_center_max_distance": 4.0,
        "lmc_head_mean_max_shift": 4.0,
        "bse_denorm_to_world": True,
        "ace_g_fusion_in_s2": True,
        "ace_g_cross_iter_eval": True,
        "training_buffer_size": 2560000,
        "buffer_size_final": 7680000,
        "buffer_batch_size": 1,
        "buffer_on_cpu": False,
        "buffer_on_cpu_final": True,
        "buffer_sample_valid_coords": True,
        "buffer_valid_coord_sample_ratio": 1.0,
        "buffer_valid_coord_neighbor_radius": 1,
        "buffer_valid_coord_neighbor_mode": "cross",
        "samples_per_image": 384,
        "batch_size": 10240,
        "s1_batch_size": 16,
        "s1_use_buffer": True,
        "s1_loss_mode": "sample_per_image",
        "s1_buffer_refill_mode": "full",
        "c1_aux_ref_loss_weight": 0.0,
        "best_metric": "pct5",
        "eval_deterministic": False,
        "post_train_eval_seeds": [1305, 2026, 4242, 7777, 9001],
        "post_train_hypotheses": 256,
        "experiment_root": (Path(__file__).parent / "04_evaluation" / "train_compare").resolve(),
        "experiment_subdir": "indoor6_full_baselines_4090_forceglobal_fixedzero",
    },
    "memory_compare_ace_g_v2": {
        "use_lmc": True,
        "lmc_flow": "ace_g",
        "lmc_memory_preflight_strict": True,
        "lmc_scene_center_max_distance": 4.0,
        "lmc_head_mean_max_shift": 4.0,
        "bse_denorm_to_world": True,
        "ace_g_fusion_in_s2": True,
        "ace_g_cross_iter_eval": True,
        "buffer_batch_size": 1,
        "samples_per_image": 384,
        "s1_batch_size": 16,
        "s1_use_buffer": True,
        "s1_loss_mode": "sample_per_image",
        "s1_buffer_refill_mode": "full",
        "best_metric": "composite",
        "eval_deterministic": True,
        "eval_dsacstar_seed": 1305,
        "eval_dsacstar_seed_per_frame": True,
        "post_train_eval_seeds": [1305, 2026, 4242],
        "post_train_hypotheses": 256,
        "experiment_root": (Path(__file__).parent / "04_evaluation" / "train_compare").resolve(),
        "experiment_subdir": "memory_pooled_vs_asb",
    },
}


def _normalize_eval_device_for_visible_cuda(device_str):
    """
    Keep eval on the logical CUDA device exposed inside this process.

    setup_cuda_environment() maps a physical --device like cuda:1 to
    CUDA_VISIBLE_DEVICES=1 before torch import. After that, the process sees
    that GPU as logical cuda:0, so passing cuda:1 to the in-process evaluator is
    an invalid device ordinal.
    """
    device_str = str(device_str or "cuda:0")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if device_str.startswith("cuda:") and visible:
        visible_ids = [x.strip() for x in visible.split(",") if x.strip()]
        if len(visible_ids) == 1 and device_str != "cuda:0":
            _logger.info(
                "Normalizing eval device %s -> cuda:0 because CUDA_VISIBLE_DEVICES=%s.",
                device_str,
                visible,
            )
            return "cuda:0"
    return device_str


def _collect_cli_flags(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    cli_flags = set()
    for token in argv:
        if token == "--":
            break
        if not token.startswith("--"):
            continue
        cli_flags.add(token.split("=", 1)[0])
    return cli_flags


def _apply_train_preset(args):
    preset = getattr(args, "train_preset", "none")
    if preset == "none":
        return

    preset_defaults = TRAIN_PRESET_DEFAULTS.get(preset)
    if preset_defaults is None:
        _logger.warning("[Train preset] unknown preset=%s, ignore.", preset)
        return

    cli_flags = _collect_cli_flags()
    applied = []
    skipped = []
    for key, value in preset_defaults.items():
        flag = f"--{key}"
        if flag in cli_flags:
            skipped.append(flag)
            continue
        setattr(args, key, copy.deepcopy(value))
        applied.append(f"{key}={value}")

    _logger.info(
        "[Train preset] %s applied defaults (CLI overrides respected): %s",
        preset,
        ", ".join(applied) if applied else "none",
    )
    if skipped:
        _logger.info("[Train preset] %s skipped explicit CLI flags: %s", preset, ", ".join(sorted(skipped)))


def _validate_args(args):
    if args.image_resolution % 14 != 0:
        args.image_resolution = (args.image_resolution // 14) * 14
        _logger.warning(
            "Image resolution adjusted to %s (must be multiple of 14)",
            args.image_resolution,
        )

    if not args.dinov2_path.exists():
        _logger.error("DINOv2 weights not found: %s", args.dinov2_path)
        sys.exit(1)

    if args.data_backend == 'wai':
        scene_meta_path = args.scene / "scene_meta.json"
        if not scene_meta_path.exists():
            _logger.error("WAI backend requires scene_meta.json at: %s", scene_meta_path)
            sys.exit(1)
        if not args.wai_repo_root.exists():
            _logger.error("WAI repo root not found: %s", args.wai_repo_root)
            sys.exit(1)
    else:
        train_dir = args.scene / "train"
        if not train_dir.exists():
            _logger.error("ACE backend expects training dir: %s", train_dir)
            sys.exit(1)

    if args.use_lmc and args.memory_path is None:
        _logger.error("--use_lmc requires --memory_path (path to POOLED .pt file)")
        sys.exit(1)
    if args.vanilla_iterations < 1:
        _logger.error("--vanilla_iterations must be >= 1")
        sys.exit(1)
    if args.s1_use_buffer and args.s1_loss_mode == "full_map":
        _logger.error("--s1_use_buffer=True 与 --s1_loss_mode=full_map 不兼容，请改为 sample_per_image 或 sample_pooled")
        sys.exit(1)
    if args.s1_buffer_refill_mode == "partial":
        keep_ratio = float(args.s1_buffer_keep_ratio)
        refill_ratio = args.s1_buffer_refill_ratio
        if keep_ratio <= 0.0 or keep_ratio >= 1.0:
            _logger.error("--s1_buffer_keep_ratio 在 partial 模式下必须位于 (0, 1)，当前为 %s", keep_ratio)
            sys.exit(1)
        if refill_ratio is not None:
            refill_ratio = float(refill_ratio)
            if refill_ratio <= 0.0 or refill_ratio >= 1.0:
                _logger.error("--s1_buffer_refill_ratio 在 partial 模式下必须位于 (0, 1)，当前为 %s", refill_ratio)
                sys.exit(1)
            if abs((keep_ratio + refill_ratio) - 1.0) > 1e-6:
                _logger.error(
                    "--s1_buffer_keep_ratio + --s1_buffer_refill_ratio 必须等于 1，当前为 %.6f",
                    keep_ratio + refill_ratio,
                )
                sys.exit(1)


def _apply_baseline_contract(args):
    # Baseline contract: when use_lmc=False, apply vanilla DINO ACE defaults so that
    # behavior matches train_ace_dinov2.py. Structural defaults (head blocks, LR) are
    # always applied; buffer/epochs/batch are only applied when still at parser default,
    # so explicit CLI (e.g. --training_buffer_size 8760000) is respected.
    if (not args.use_lmc) and args.apply_baseline_contract:
        _vanilla_defaults = {
            "num_head_blocks": 4,
            "learning_rate_max": 0.001,
            "s1_learning_rate_max": 0.001,
            "s2_learning_rate_max": 0.001,
            "training_buffer_size": 10000000,
            "buffer_batch_size": 1,
            "samples_per_image": 512,
            "epochs": 16,
            "batch_size": 5120,
        }
        _parser_defaults = {
            "num_head_blocks": 4,
            "learning_rate_max": 0.0001,
            "s1_learning_rate_max": 0.0001,
            "s2_learning_rate_max": 0.001,
            "training_buffer_size": 2560000,
            "buffer_batch_size": 10,
            "samples_per_image": 512,
            "epochs": 24,
            "batch_size": 5120,
        }
        applied = []
        for key, vanilla_val in _vanilla_defaults.items():
            cur = getattr(args, key)
            parser_default = _parser_defaults[key]
            if key in ("num_head_blocks", "learning_rate_max", "s1_learning_rate_max", "s2_learning_rate_max"):
                setattr(args, key, vanilla_val)
                applied.append(f"{key}={vanilla_val}")
            elif cur == parser_default:
                setattr(args, key, vanilla_val)
                applied.append(f"{key}={vanilla_val}")
            # else: user explicitly set a non-default value, keep it
        _logger.info(
            "[Baseline contract] use_lmc=False: applied vanilla defaults where not overridden: %s.",
            ", ".join(applied),
        )
    elif not args.use_lmc:
        _logger.info(
            "[Baseline contract] disabled (apply_baseline_contract=False). "
            "Will use CLI hyperparameters as-is for iterative no-LMC baseline."
        )


def _build_vanilla_iterative_args(args):
    """
    Build a vanilla baseline args object that matches train_ace_dinov2_iterative.py
    semantics instead of reusing LMC-specific defaults.
    """
    vanilla_args = copy.copy(args)
    vanilla_args.iterations = int(args.vanilla_iterations)
    vanilla_args.iter_buffer_size = int(args.training_buffer_size)
    vanilla_args.reset_optimizer_each_iter = False

    # LMC parser default is 10 for faster LMC buffer fill, but the validated vanilla
    # iterative baseline uses 1. If user did not explicitly override it, force parity.
    if getattr(vanilla_args, "buffer_batch_size", None) == 10:
        _logger.info(
            "[Vanilla parity] Overriding buffer_batch_size: 10 -> 1 "
            "to match train_ace_dinov2_iterative.py."
        )
        vanilla_args.buffer_batch_size = 1

    return vanilla_args


def _run_standard_eval(args, output_map, session):
    """
    Evaluate a vanilla checkpoint with the same evaluator used by train_ace_dinov2_iterative.py.
    """
    from test_ace_dinov2 import run_evaluation

    eval_device = _normalize_eval_device_for_visible_cuda(
        getattr(args, "post_train_eval_device", "cuda:0")
    )
    eval_opt = argparse.Namespace(
        scene=args.scene,
        network=output_map,
        dinov2_path=args.dinov2_path,
        device=eval_device,
        image_resolution=args.image_resolution,
        session=session,
        eval_deterministic=getattr(args, "eval_deterministic", False),
        dsacstar_seed=getattr(args, "eval_dsacstar_seed", 1305),
        dsacstar_seed_per_frame=getattr(args, "eval_dsacstar_seed_per_frame", True),
        eval_num_workers=getattr(args, "eval_num_workers", 6),
        render_visualization=False,
    )
    return run_evaluation(eval_opt)


def _score_standard_eval(result, metric):
    metric = (metric or "pct5").lower()
    if metric == "pct25_5":
        return float(result["pct25_5"])
    if metric == "pct10_5":
        return float(result["pct10_5"])
    if metric == "pct2":
        return float(result["pct2"])
    if metric == "pct1":
        return float(result["pct1"])
    if metric == "composite":
        return (
            float(result["pct5"])
            - 2.0 * float(result["median_tErr"])
            - 2.0 * float(result["median_rErr"])
        )
    if metric == "rt_error":
        return -float(result["median_rErr"]) - float(result["median_tErr"])
    if metric == "median":
        return -float(result["median_rErr"]) - float(result["median_tErr"])
    return float(result["pct5"])


def _write_vanilla_eval_summary(args, result):
    eval_log_path = args.run_dir / "post_train_eval.txt"
    with open(eval_log_path, "w", encoding="utf-8") as f:
        f.write(f"median_rotation_deg\t{result['median_rErr']:.4f}\n")
        f.write(f"median_translation_cm\t{result['median_tErr']:.4f}\n")
        f.write(f"accuracy_25cm5deg_pct\t{result['pct25_5']:.2f}\n")
        f.write(f"accuracy_10cm5deg_pct\t{result['pct10_5']:.2f}\n")
        f.write(f"accuracy_5cm5deg_pct\t{result['pct5']:.2f}\n")
        f.write(f"accuracy_2cm2deg_pct\t{result['pct2']:.2f}\n")
        f.write(f"accuracy_1cm1deg_pct\t{result['pct1']:.2f}\n")
        f.write(f"avg_time_per_frame_ms\t{result['avg_time'] * 1000:.2f}\n")
        f.write(f"total_frames\t{result['total_frames']}\n")
    _logger.info("Eval summary also written to: %s", eval_log_path)


def run_vanilla_iterative_baseline(args):
    """
    Run the no-LMC baseline via the validated TrainerACEDINOv2 path instead of the
    LMC trainer. This keeps the LMC entrypoint and output layout, but removes all
    residual LMC-side training/eval differences.
    """
    vanilla_args = _build_vanilla_iterative_args(args)
    iter_buffer_size = vanilla_args.iter_buffer_size

    trainer_args = copy.copy(vanilla_args)
    trainer_args.training_buffer_size = iter_buffer_size * vanilla_args.iterations

    trainer = TrainerACEDINOv2(trainer_args)
    trainer.training_start = time.time()

    pt_name = Path(vanilla_args.output_map).name
    out_base = Path(vanilla_args.output_map).parent
    best_score = -float("inf")
    best_iter = None

    _logger.info("=" * 80)
    _logger.info("[Vanilla parity] Running validated iterative baseline via TrainerACEDINOv2")
    _logger.info(
        "[Vanilla parity] iterations=%d iter_buffer_size=%d buffer_batch_size=%d epochs=%d",
        vanilla_args.iterations,
        iter_buffer_size,
        vanilla_args.buffer_batch_size,
        vanilla_args.epochs,
    )
    _logger.info("=" * 80)

    for iteration_idx in range(vanilla_args.iterations):
        _logger.info("=== Iteration %d/%d ===", iteration_idx + 1, vanilla_args.iterations)

        if iteration_idx > 0:
            trainer.reset_optimizer_scheduler(
                keep_optimizer_state=not vanilla_args.reset_optimizer_each_iter,
                buffer_size=iter_buffer_size,
            )

        buffer_start = time.time()
        trainer.create_training_buffer(buffer_size=iter_buffer_size)
        _logger.info("Filled training buffer in %.1fs.", time.time() - buffer_start)

        for trainer.epoch in range(vanilla_args.epochs):
            trainer.run_epoch()

        iter_ckpt = out_base / f"{vanilla_args.output_map.stem}.iter_{iteration_idx + 1:02d}.tmp.pt"
        trainer.save_model(iter_ckpt)

        eval_result = None
        score = -float("inf")
        if vanilla_args.eval_each_iteration:
            eval_result = _run_standard_eval(
                vanilla_args,
                iter_ckpt,
                session=f"iter_{iteration_idx + 1:02d}",
            )
            score = _score_standard_eval(eval_result, vanilla_args.best_metric)
            _logger.info(
                "[Vanilla parity] Iter %02d eval: median=%.2fdeg/%.2fcm pct5=%.2f pct25_5=%.2f",
                iteration_idx + 1,
                eval_result["median_rErr"],
                eval_result["median_tErr"],
                eval_result["pct5"],
                eval_result["pct25_5"],
            )
        else:
            score = float(iteration_idx + 1)

        if score > best_score:
            best_score = score
            best_iter = iteration_idx + 1
            os.replace(iter_ckpt, vanilla_args.output_map)
            _logger.info("[Vanilla parity] Updated best checkpoint at iter %d (score=%.4f)", best_iter, score)
        elif iter_ckpt.exists():
            if vanilla_args.keep_best_only:
                iter_ckpt.unlink()
            else:
                iter_ckpt.rename(out_base / f"{vanilla_args.output_map.stem}.iter_{iteration_idx + 1:02d}.pt")

    if not vanilla_args.output_map.exists():
        trainer.save_model(vanilla_args.output_map)

    if vanilla_args.eval_after_train:
        _logger.info("Running post-training evaluation with standard ACE evaluator...")
        result = _run_standard_eval(vanilla_args, vanilla_args.output_map, vanilla_args.eval_session)
        _logger.info("========== Post-train Eval (standard ACE evaluator) ==========")
        _logger.info(
            "  Median: %.2f deg, %.2f cm | 25cm/5deg: %.2f%% | 10cm/5deg: %.2f%% | 5cm/5deg: %.2f%% | "
            "2cm/2deg: %.2f%% | 1cm/1deg: %.2f%%",
            result["median_rErr"], result["median_tErr"],
            result["pct25_5"], result["pct10_5"], result["pct5"],
            result["pct2"], result["pct1"],
        )
        _logger.info("  Avg time: %.2f ms | Frames: %d", result["avg_time"] * 1000, result["total_frames"])
        _logger.info("==============================================================")
        _write_vanilla_eval_summary(vanilla_args, result)

    _logger.info(
        "Vanilla parity run completed. Total time: %.1fs | best_iter=%s | best_score=%.4f",
        time.time() - trainer.training_start,
        best_iter,
        best_score,
    )
    return trainer


def _apply_lmc_profile(args):
    """Apply profile defaults for LMC flow; explicit CLI overrides are respected."""
    profile = getattr(args, "lmc_profile", "legacy")

    def _finalize_buffer_sampling_default(default_value: bool):
        if getattr(args, "buffer_sampling_replacement", None) is None:
            args.buffer_sampling_replacement = default_value
            return True
        return False

    if not args.use_lmc:
        if profile != "legacy":
            _logger.info(
                "[LMC profile] use_lmc=False, ignore profile=%s and keep vanilla path.",
                profile,
            )
        _finalize_buffer_sampling_default(True)
        return

    if profile == "legacy":
        if getattr(args, "s2_repro_rewind_first_ratio", None) is None:
            args.s2_repro_rewind_first_ratio = 0.20
        if getattr(args, "s2_repro_rewind_later_ratio", None) is None:
            args.s2_repro_rewind_later_ratio = 0.08
        if getattr(args, "s2_lr_boost_first", None) is None:
            args.s2_lr_boost_first = 1.2
        if getattr(args, "s2_lr_boost_later", None) is None:
            args.s2_lr_boost_later = 1.0
        _finalize_buffer_sampling_default(True)
        _logger.info("[LMC profile] legacy: keep current behavior.")
        return

    if profile != "mapany_flow_v1":
        _logger.warning("[LMC profile] unknown profile=%s, fallback to legacy.", profile)
        if getattr(args, "s2_repro_rewind_first_ratio", None) is None:
            args.s2_repro_rewind_first_ratio = 0.20
        if getattr(args, "s2_repro_rewind_later_ratio", None) is None:
            args.s2_repro_rewind_later_ratio = 0.08
        if getattr(args, "s2_lr_boost_first", None) is None:
            args.s2_lr_boost_first = 1.2
        if getattr(args, "s2_lr_boost_later", None) is None:
            args.s2_lr_boost_later = 1.0
        _finalize_buffer_sampling_default(True)
        return

    # map-anything flow-aligned defaults. Only apply when still at parser default.
    profile_defaults = {
        "s1_lr_scale_later": 1.0,
    }
    parser_defaults = {
        "s1_lr_scale_later": 0.2,
        "s2_repro_rewind_first_ratio": 0.20,
        "s2_repro_rewind_later_ratio": 0.08,
        "s2_lr_boost_first": 1.2,
        "s2_lr_boost_later": 1.0,
    }

    applied = []
    for key, val in profile_defaults.items():
        cur = getattr(args, key)
        if cur == parser_defaults[key]:
            setattr(args, key, val)
            applied.append(f"{key}={val}")
    if getattr(args, "s2_repro_rewind_first_ratio", None) is None:
        args.s2_repro_rewind_first_ratio = 0.0
        applied.append("s2_repro_rewind_first_ratio=0.0")
    if getattr(args, "s2_repro_rewind_later_ratio", None) is None:
        args.s2_repro_rewind_later_ratio = 0.0
        applied.append("s2_repro_rewind_later_ratio=0.0")
    if getattr(args, "s2_lr_boost_first", None) is None:
        args.s2_lr_boost_first = 1.0
        applied.append("s2_lr_boost_first=1.0")
    if getattr(args, "s2_lr_boost_later", None) is None:
        args.s2_lr_boost_later = 1.0
        applied.append("s2_lr_boost_later=1.0")
    if _finalize_buffer_sampling_default(False):
        applied.append("buffer_sampling_replacement=False")
    _logger.info(
        "[LMC profile] mapany_flow_v1 applied defaults (if not overridden by CLI): %s",
        ", ".join(applied) if applied else "none",
    )


def _build_run_dir(args):
    if args.buffer_size_final is None:
        args.buffer_size_final = args.training_buffer_size * 3

    # 输出结构：<experiment_root>/<dataset>/<scene>/<function>/<run_id>/best_K*_it*_<suffix>.pt
    # dataset/scene 从 scene 路径解析，与 train_ace_dinov2 一致（如 .../indoor6_ace/scene1/train -> indoor6_ace, scene1）
    run_root = Path(args.experiment_root).resolve() if args.experiment_root is not None else (Path(__file__).parent / "04_evaluation").resolve()
    _subdir = getattr(args, "experiment_subdir", None)
    if _subdir is not None and str(_subdir).strip():
        run_root = run_root / _sanitize_tag(str(_subdir).strip())
    run_root.mkdir(parents=True, exist_ok=True)

    scene_path = Path(args.scene).resolve()
    if len(scene_path.parts) >= 2:
        if scene_path.name in ("train", "test", "val") and len(scene_path.parts) >= 3:
            dataset = _sanitize_tag(scene_path.parent.parent.name)
            scene = _sanitize_tag(scene_path.parent.name)
        else:
            dataset = _sanitize_tag(scene_path.parent.name)
            scene = _sanitize_tag(scene_path.name)
    else:
        dataset = "default"
        scene = _sanitize_tag(scene_path.name or "scene")

    scene_tag = scene
    mode_tag = _sanitize_tag(args.lmc_mode if args.use_lmc else "vanilla")
    sched_tag = _sanitize_tag(args.lmc_lr_scheduler_type if args.use_lmc else "onecycle")
    profile_tag = ""
    if args.use_lmc and getattr(args, "lmc_profile", "legacy") != "legacy":
        profile_tag = f"_pf-{_sanitize_tag(args.lmc_profile)}"
    buf_tag = _format_buf_million(args.training_buffer_size)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config_tag = build_lmc_run_folder_config_tag(args)
    if args.run_name == "auto":
        auto_run_name = f"{timestamp}_{config_tag}"
    else:
        auto_run_name = None
    # Always append config tag for isolation; auto already embeds it.
    if args.run_name == 'auto':
        run_name = auto_run_name
    else:
        run_name = f"{_sanitize_tag(args.run_name)}_{config_tag}"

    output_layout = getattr(args, 'output_layout', 'hierarchical')
    if output_layout == 'hierarchical':
        if not args.use_lmc:
            function = "dino_ace_baseline"
        elif getattr(args, 'lmc_flow', 'iterative') == 'ace_g':
            function = "dino_ace_lmc_ace_g"
        else:
            function = "dino_ace_lmc_s1s2"
        run_id = run_name
        run_dir = run_root / dataset / scene / function / run_id
    else:
        run_dir = run_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    if getattr(args, 'overwrite_run_dir', True) and args.run_name != 'auto':
        # 复用同一 run_name 时清空目录，避免混入上次结果
        for p in run_dir.iterdir():
            try:
                if p.is_file():
                    p.unlink()
                else:
                    shutil.rmtree(p)
            except OSError:
                pass

    out_suffix = _sanitize_stem(args.output_map)
    best_pt_name = f"best_K{args.num_latent_tokens}_it{args.lmc_iterations}_{out_suffix}.pt"
    args.output_map = run_dir / best_pt_name
    args.run_dir = run_dir
    return output_layout, run_dir


def _attach_full_log_file_handler(run_dir):
    # Save full console-equivalent training log to file (map-anything style).
    full_log_path = run_dir / "training_full_log.txt"
    root_logger = logging.getLogger()
    existing_paths = {
        getattr(h, 'baseFilename', None) for h in root_logger.handlers
        if hasattr(h, 'baseFilename')
    }
    if str(full_log_path) not in existing_paths:
        file_handler = logging.FileHandler(full_log_path, encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(logging.Formatter('%(asctime)s | %(levelname)s | %(name)s | %(message)s'))
        root_logger.addHandler(file_handler)
    return full_log_path


def _persist_run_metadata(args, run_dir):
    # Persist run metadata for reproducibility.
    with open(run_dir / "run_config.json", 'w', encoding='utf-8') as f:
        json.dump({k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}, f, indent=2, ensure_ascii=False)
    with open(run_dir / "run_command.txt", 'w', encoding='utf-8') as f:
        f.write("python " + " ".join(sys.argv) + "\n")


def _log_configuration_summary(args, output_layout, full_log_path):
    _logger.info("=" * 80)
    _logger.info("ACE DINOv2 Training%s", " + LMC" if args.use_lmc else "")
    _logger.info("=" * 80)
    _logger.info("Scene        : %s", args.scene)
    _logger.info("Data backend : %s", args.data_backend)
    if args.data_backend == 'wai':
        _logger.info("WAI repo     : %s", args.wai_repo_root)
        _logger.info("WAI image key: %s", args.wai_image_modality)
    _logger.info("Run Dir      : %s", args.run_dir)
    if output_layout == 'hierarchical':
        _logger.info("Output layout: hierarchical (dataset/scene/function/run_id)")
    _logger.info("Output       : %s", args.output_map)
    _logger.info("Full Log     : %s", full_log_path)
    _logger.info("Device       : %s", args.device)
    _logger.info("LMC          : %s", f"ON (mode={args.lmc_mode})" if args.use_lmc else "OFF")
    _logger.info("Train preset : %s", getattr(args, "train_preset", "none"))
    _logger.info("LMC profile  : %s", args.lmc_profile if args.use_lmc else "n/a")
    _logger.info("LMC flow     : %s", getattr(args, 'lmc_flow', 'iterative'))
    if not args.use_lmc and args.vanilla_iterations > 1:
        _logger.info("Vanilla iters: %d (iterative baseline mode)", args.vanilla_iterations)

    if args.use_lmc:
        _logger.info("Memory       : %s", args.memory_path)
        _logger.info(
            "Memory check : preflight=%s, strict=%s, center_tol=%.2f m, scene_strict=%s, center_strict=%s",
            args.lmc_memory_preflight,
            args.lmc_memory_preflight_strict,
            args.lmc_memory_preflight_center_tol,
            args.lmc_strict_scene_check,
            args.lmc_strict_center_check,
        )
        _logger.info("Tokens/Attn  : K=%d, attn_layers=%d", args.num_latent_tokens, args.num_attn_layers)
        _logger.info("Iterations   : %d (S1 warmup=%d, S1 later=%d)",
                     args.lmc_iterations, args.lmc_warmup_steps, args.lmc_train_steps)
        _logger.info(
            "Mode auto-sw  : enabled=%s, vis_thr=%.2f, fallback=%s, sample_pts=%d",
            args.lmc_auto_mode_by_visibility,
            args.lmc_visibility_front_ratio_threshold,
            args.lmc_visibility_fallback_mode,
            args.lmc_visibility_sample_points,
        )
        _logger.info("Head reset   : %s", args.head_reset_strategy)
        _logger.info(
            "LR max       : S1=%.2e, S2=%.2e (legacy learning_rate_max=%.2e)",
            args.s1_learning_rate_max, args.s2_learning_rate_max, args.learning_rate_max
        )
        _logger.info("S1 LR        : %s", args.lmc_lr_scheduler_type)
        _logger.info(
            "Flow knobs   : s1_lr_scale_later=%.2f, s2_rewind=(%.2f, %.2f), s2_boost=(%.2f, %.2f)",
            args.s1_lr_scale_later,
            args.s2_repro_rewind_first_ratio,
            args.s2_repro_rewind_later_ratio,
            args.s2_lr_boost_first,
            args.s2_lr_boost_later,
        )
        _logger.info(
            "S2 LR boost  : first=%.2f, later=%.2f | warmup_steps=%d",
            args.s2_lr_boost_first, args.s2_lr_boost_later, args.s2_lr_warmup_steps,
        )
        _logger.info(
            "S2 polish    : epochs=%d, head_lr=%.2e, fusion_lr_ratio=%.4f",
            args.s2_polish_epochs,
            args.s2_polish_head_lr,
            args.s2_polish_fusion_lr_ratio,
        )
        _logger.info(
            "S2 step rewind: first=%.2f, later=%.2f, tau=%.1f",
            args.s2_repro_rewind_first_ratio,
            args.s2_repro_rewind_later_ratio,
            args.s2_repro_rewind_tau,
        )
        _logger.info(
            "Buffer sample: replacement=%s, samples_per_image=%d",
            args.buffer_sampling_replacement,
            args.samples_per_image,
        )
        _logger.info(
            "Buffer sample valid coords: enabled=%s, ratio=%.2f, neighbor_radius=%d, neighbor_mode=%s",
            args.buffer_sample_valid_coords,
            args.buffer_valid_coord_sample_ratio,
            args.buffer_valid_coord_neighbor_radius,
            args.buffer_valid_coord_neighbor_mode,
        )
        _logger.info(
            "S1 early-stop: enabled=%s, min_updates=%d, patience=%d, rel_improve=%.4f, ema_beta=%.2f",
            args.s1_early_stop,
            args.s1_early_stop_min_updates,
            args.s1_early_stop_patience,
            args.s1_early_stop_rel_improve,
            args.s1_early_stop_ema_beta,
        )
        _logger.info("Head grid    : sampled features are reshaped to 16xW; use batch_size multiple of 16 to avoid trimming.")
        _logger.info("Buffer on CPU: %s (S2 显存不足时保持 True)", args.buffer_on_cpu)
        _logger.info("Final buffer on CPU: %s (buffer_size_final=%s)", args.buffer_on_cpu_final, args.buffer_size_final)
        _logger.info(
            "S1 data source: %s (s1_loss_mode=%s, refill_mode=%s)",
            "raw_buffer" if args.s1_use_buffer else "online_encoder",
            args.s1_loss_mode,
            args.s1_buffer_refill_mode,
        )
        if args.s1_use_buffer and args.s1_buffer_refill_mode == "partial":
            refill_ratio = (
                args.s1_buffer_refill_ratio
                if args.s1_buffer_refill_ratio is not None
                else (1.0 - args.s1_buffer_keep_ratio)
            )
            _logger.info(
                "S1 partial refill: keep_ratio=%.3f, refill_ratio=%.3f",
                args.s1_buffer_keep_ratio,
                refill_ratio,
            )

        # ACE-G specific parameters
        if getattr(args, 'lmc_flow', 'iterative') == 'ace_g':
            _logger.info(
                "ACE-G params : fusion_in_s2=%s, fusion_lr_ratio=%.4f, cross_iter_eval=%s",
                args.ace_g_fusion_in_s2,
                args.ace_g_fusion_lr_ratio,
                args.ace_g_cross_iter_eval,
            )
        _logger.info("PE normalize : %s", getattr(args, 'pe_normalize_input', False))

    _logger.info(
        "Eval policy  : each_iter=%s, keep_best_only=%s, best_metric=%s",
        args.eval_each_iteration,
        args.keep_best_only,
        args.best_metric,
    )
    _logger.info("=" * 80)


def setup_experiment(args):
    _apply_train_preset(args)
    _validate_args(args)
    _apply_baseline_contract(args)
    _apply_lmc_profile(args)
    output_layout, run_dir = _build_run_dir(args)
    full_log_path = _attach_full_log_file_handler(run_dir)
    _persist_run_metadata(args, run_dir)
    _log_configuration_summary(args, output_layout, full_log_path)
    return run_dir


def run_post_train_eval(args, trainer):
    if not args.eval_after_train:
        return
    _logger.info("Running post-training evaluation...")
    try:
        # 训练已在 trainer.train() 末尾释放 buffer；这里再释放 trainer 本身占用的显存，便于同进程 GPU eval
        del trainer
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        eval_device = _normalize_eval_device_for_visible_cuda(
            getattr(args, "post_train_eval_device", "cuda:0")
        )
        _logger.info("Freed trainer; running eval on %s.", eval_device)

        from test_ace_dinov2_lmc import run_evaluation_lmc
        # 使用 --post_train_eval_device，默认 cuda:0 以加速评测
        post_train_seeds = [int(s) for s in getattr(args, "post_train_eval_seeds", [1305])]
        post_train_hypotheses = int(getattr(args, "post_train_hypotheses", 64))
        seed_results = []
        for seed in post_train_seeds:
            session_name = args.eval_session
            if len(post_train_seeds) > 1:
                session_name = f"{args.eval_session}_seed{seed}"
            eval_opt = argparse.Namespace(
                scene=args.scene,
                network=args.output_map,
                dinov2_path=args.dinov2_path,
                device=eval_device,
                image_resolution=args.image_resolution,
                session=session_name,
                hypotheses=post_train_hypotheses,
                eval_deterministic=getattr(args, "eval_deterministic", False),
                dsacstar_seed=seed,
                dsacstar_seed_per_frame=getattr(args, "eval_dsacstar_seed_per_frame", True),
                eval_num_workers=getattr(args, "eval_num_workers", 6),
                render_visualization=False,
            )
            result = run_evaluation_lmc(eval_opt)
            result["seed"] = seed
            seed_results.append(result)

        result = dict(seed_results[0])
        if len(seed_results) > 1:
            agg_keys = [
                "median_rErr",
                "median_tErr",
                "pct25_5",
                "pct10_5",
                "pct5",
                "pct2",
                "pct1",
                "avg_time",
            ]
            for key in agg_keys:
                result[key] = float(np.median([r[key] for r in seed_results]))
            result["total_frames"] = int(seed_results[0]["total_frames"])

        _logger.info("========== Post-train Eval (aggregated) ==========")
        _logger.info(
            "  Median: %.2f deg, %.2f cm | 25cm/5deg: %.2f%% | 10cm/5deg: %.2f%% | 5cm/5deg: %.2f%% | "
            "2cm/2deg: %.2f%% | 1cm/1deg: %.2f%%",
            result['median_rErr'], result['median_tErr'],
            result['pct25_5'], result['pct10_5'], result['pct5'],
            result['pct2'], result['pct1'],
        )
        _logger.info("  Avg time: %.2f ms | Frames: %d",
                     result['avg_time'] * 1000, result['total_frames'])
        if len(seed_results) > 1:
            _logger.info(
                "  Seeds: %s | hypotheses=%d | aggregation=median",
                ",".join(str(r["seed"]) for r in seed_results),
                post_train_hypotheses,
            )
        _logger.info("=====================================================")
        # 写入 run_dir 便于与 training_full_log 一起查看
        semantic_fields = {}
        try:
            checkpoint = torch.load(args.output_map, map_location='cpu')
            lmc_config = checkpoint.get('lmc_config', {}) if isinstance(checkpoint, dict) else {}
            semantic_fields = {
                "requested_lmc_mode": lmc_config.get("requested_lmc_mode"),
                "effective_lmc_mode": lmc_config.get("effective_lmc_mode", lmc_config.get("lmc_mode")),
                "lmc_auto_mode_by_visibility": lmc_config.get("lmc_auto_mode_by_visibility"),
                "lmc_flow": lmc_config.get("lmc_flow"),
                "lmc_key_slice_idx": lmc_config.get("lmc_key_slice_idx"),
                "lmc_key_layer_label": lmc_config.get("lmc_key_layer_label"),
                "layers_idx": lmc_config.get("layers_idx"),
                "lmc_fps_start_policy": lmc_config.get("lmc_fps_start_policy"),
                "s1_loss_step_mode": lmc_config.get("s1_loss_step_mode"),
                "ace_g_fusion_in_s2": lmc_config.get("ace_g_fusion_in_s2"),
            }
        except Exception as e:
            _logger.warning("Could not read LMC semantics from checkpoint for post-train eval summary: %s", e)
        eval_log_path = args.run_dir / "post_train_eval.txt"
        with open(eval_log_path, "w", encoding="utf-8") as f:
            f.write(f"eval_deterministic\t{bool(getattr(args, 'eval_deterministic', False))}\n")
            f.write(f"aggregation\t{'median' if len(seed_results) > 1 else 'single'}\n")
            f.write(f"seeds\t{','.join(str(r['seed']) for r in seed_results)}\n")
            f.write(f"hypotheses\t{post_train_hypotheses}\n")
            f.write(f"median_rotation_deg\t{result['median_rErr']:.4f}\n")
            f.write(f"median_translation_cm\t{result['median_tErr']:.4f}\n")
            f.write(f"accuracy_25cm5deg_pct\t{result['pct25_5']:.2f}\n")
            f.write(f"accuracy_10cm5deg_pct\t{result['pct10_5']:.2f}\n")
            f.write(f"accuracy_5cm5deg_pct\t{result['pct5']:.2f}\n")
            f.write(f"accuracy_2cm2deg_pct\t{result['pct2']:.2f}\n")
            f.write(f"accuracy_1cm1deg_pct\t{result['pct1']:.2f}\n")
            f.write(f"avg_time_per_frame_ms\t{result['avg_time'] * 1000:.2f}\n")
            f.write(f"total_frames\t{result['total_frames']}\n")
            for key, value in semantic_fields.items():
                if isinstance(value, (list, tuple)):
                    value = ",".join(str(v) for v in value)
                f.write(f"{key}\t{value}\n")
        if len(seed_results) > 1:
            raw_eval_path = args.run_dir / "post_train_eval_seed_runs.txt"
            with open(raw_eval_path, "w", encoding="utf-8") as f:
                f.write("# seed-wise post-train evaluation results\n")
                f.write("seed\tmedian_rErr\tmedian_tErr\tpct25_5\tpct10_5\tpct5\tpct2\tpct1\tavg_time_ms\ttotal_frames\n")
                for seed_result in seed_results:
                    f.write(
                        f"{seed_result['seed']}\t{seed_result['median_rErr']:.4f}\t{seed_result['median_tErr']:.4f}\t"
                        f"{seed_result['pct25_5']:.2f}\t{seed_result['pct10_5']:.2f}\t{seed_result['pct5']:.2f}\t"
                        f"{seed_result['pct2']:.2f}\t{seed_result['pct1']:.2f}\t"
                        f"{seed_result['avg_time'] * 1000:.2f}\t{seed_result['total_frames']}\n"
                    )
            _logger.info("Seed-wise eval summary written to: %s", raw_eval_path)
        _logger.info("Eval summary also written to: %s", eval_log_path)
    except Exception as e:
        _logger.warning("Post-training eval failed: %s", e, exc_info=True)


def main():
    parser = get_lmc_train_parser()
    args = parser.parse_args()
    # setup_cuda_environment() 已按 --device 设置了 CUDA_VISIBLE_DEVICES，进程内仅见一块 GPU，逻辑设备为 cuda:0
    if getattr(args, 'device', '').startswith('cuda:') and os.environ.get('CUDA_VISIBLE_DEVICES'):
        args.device = 'cuda:0'
    setup_experiment(args)

    if not args.use_lmc:
        run_vanilla_iterative_baseline(args)
        _logger.info("Training completed.")
        return

    trainer = TrainerACEDINOv2LMC(args)
    trainer.train()
    _logger.info("Training completed.")

    run_post_train_eval(args, trainer)


if __name__ == '__main__':
    main()
