#!/usr/bin/env python3
# train_ace_lmc.py
# -----------------------------------------------------------------------------
# ACE FCN encoder + GeoLMC 训练入口。
# 与 train_ace_dinov2_lmc.py 逻辑完全一致，仅将 backbone 换为 ACE 原版 FCN encoder。
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
#   - 输入分辨率无 patch-size 约束（FCN encoder 支持任意分辨率）。
# -----------------------------------------------------------------------------

import argparse
import copy
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

print("train_ace_lmc: loading options and CUDA env...", flush=True)
from options_ace_lmc import get_lmc_train_parser
from result_manager import ResultManager
from utils_lmc import (
    _format_buf_million,
    _sanitize_stem,
    _sanitize_tag,
    setup_cuda_environment,
)

setup_cuda_environment()

print("train_ace_lmc: importing torch and trainers...", flush=True)
import torch
from trainer_ace_fcn import TrainerACEFCN, TrainerACEFCNLMC

print("train_ace_lmc: ready, configuring logging.", flush=True)
logging.basicConfig(level=logging.INFO)
_logger = logging.getLogger(__name__)


def _validate_args(args):
    if not args.encoder_path.exists():
        _logger.error("ACE encoder weights not found: %s", args.encoder_path)
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
        _logger.error("--s1_use_buffer=True 与 --s1_loss_mode=full_map 不兼容")
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
        _logger.info(
            "[Baseline contract] use_lmc=False: applied vanilla defaults where not overridden: %s.",
            ", ".join(applied),
        )
    elif not args.use_lmc:
        _logger.info(
            "[Baseline contract] disabled (apply_baseline_contract=False). "
            "Will use CLI hyperparameters as-is for iterative no-LMC baseline."
        )


def _build_run_dir(args):
    if args.buffer_size_final is None:
        args.buffer_size_final = args.training_buffer_size * 3

    result_mgr = ResultManager(args.experiment_root)
    scene_info = result_mgr.parse_scene_info(args.scene)

    # 构建算法标签
    algorithm = "ace_fcn_vanilla" if not args.use_lmc else "ace_fcn_lmc"

    # 构建配置标签
    mode_tag = _sanitize_tag(args.lmc_mode if args.use_lmc else "vanilla")
    sched_tag = _sanitize_tag(args.lmc_lr_scheduler_type if args.use_lmc else "onecycle")
    profile_tag = ""
    if args.use_lmc and getattr(args, "lmc_profile", "legacy") != "legacy":
        profile_tag = f"_pf-{_sanitize_tag(args.lmc_profile)}"
    buf_tag = _format_buf_million(args.training_buffer_size)

    if args.use_lmc:
        config_tag = (
            f"{mode_tag}_buf{buf_tag}M_K{args.num_latent_tokens}"
            f"_it{args.lmc_iterations}_bs{args.batch_size}_{sched_tag}{profile_tag}"
        )
    else:
        if args.vanilla_iterations > 1:
            config_tag = f"iter{args.vanilla_iterations}_buf{buf_tag}M_bs{args.batch_size}_ep{args.epochs}"
        else:
            config_tag = f"vanilla_buf{buf_tag}M_bs{args.batch_size}_ep{args.epochs}"

    # 使用结果管理器构建层级化目录
    run_dir = result_mgr.build_hierarchical_path(
        args.scene,
        algorithm,
        config_tag,
        timestamp=None,  # 使用当前时间戳
    )

    # 保存运行元数据
    result_mgr.save_run_metadata(run_dir, args, algorithm, scene_info)

    # 清空目录（如果需要）
    if getattr(args, 'overwrite_run_dir', True) and args.run_name != 'auto':
        for p in run_dir.iterdir():
            try:
                if p.is_file():
                    p.unlink()
                else:
                    shutil.rmtree(p)
            except OSError:
                pass

    out_suffix = _sanitize_stem(args.output_map)
    best_pt_name = f"best_K{args.num_latent_tokens}_it{args.lmc_iterations}_{out_suffix}.pt" if args.use_lmc else f"best_{out_suffix}.pt"
    args.output_map = run_dir / best_pt_name
    args.run_dir = run_dir
    args.result_manager = result_mgr

    return "hierarchical", run_dir


def _attach_full_log_file_handler(run_dir):
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
    with open(run_dir / "run_config.json", 'w', encoding='utf-8') as f:
        json.dump({k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}, f, indent=2, ensure_ascii=False)
    with open(run_dir / "run_command.txt", 'w', encoding='utf-8') as f:
        f.write("python " + " ".join(sys.argv) + "\n")


def _log_configuration_summary(args, output_layout, full_log_path):
    _logger.info("=" * 80)
    _logger.info("ACE FCN Training%s", " + LMC" if args.use_lmc else "")
    _logger.info("=" * 80)
    _logger.info("Scene        : %s", args.scene)
    _logger.info("Encoder      : %s", args.encoder_path)
    _logger.info("Run Dir      : %s", args.run_dir)
    _logger.info("Output       : %s", args.output_map)
    _logger.info("Full Log     : %s", full_log_path)
    _logger.info("Device       : %s", args.device)
    _logger.info("LMC          : %s", f"ON (mode={args.lmc_mode})" if args.use_lmc else "OFF")
    if not args.use_lmc and args.vanilla_iterations > 1:
        _logger.info("Vanilla iters: %d (iterative baseline mode)", args.vanilla_iterations)
    if args.use_lmc:
        _logger.info("Memory       : %s", args.memory_path)
        _logger.info("Tokens/Attn  : K=%d, attn_layers=%d", args.num_latent_tokens, args.num_attn_layers)
        _logger.info("Iterations   : %d", args.lmc_iterations)
        _logger.info("LR max       : S1=%.2e, S2=%.2e", args.s1_learning_rate_max, args.s2_learning_rate_max)
    _logger.info(
        "Eval policy  : each_iter=%s, keep_best_only=%s, best_metric=%s",
        args.eval_each_iteration, args.keep_best_only, args.best_metric,
    )
    _logger.info("=" * 80)


def setup_experiment(args):
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
        del trainer
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        from test_ace_lmc import run_evaluation_lmc
        eval_device = getattr(args, "post_train_eval_device", "cuda:0")
        eval_opt = argparse.Namespace(
            scene=args.scene,
            network=args.output_map,
            encoder_path=args.encoder_path,
            device=eval_device,
            image_resolution=args.image_resolution,
            session=args.eval_session,
            render_visualization=False,
        )
        result = run_evaluation_lmc(eval_opt)
        _logger.info("========== Post-train Eval ==========")
        _logger.info(
            "  Median: %.2f deg, %.2f cm | 25cm/5deg: %.2f%% | 10cm/5deg: %.2f%% | 5cm/5deg: %.2f%%",
            result['median_rErr'], result['median_tErr'],
            result['pct25_5'], result['pct10_5'], result['pct5'],
        )
        _logger.info("=====================================")

        # 使用结果管理器保存评估结果
        if hasattr(args, 'result_manager'):
            args.result_manager.save_evaluation_results(
                args.run_dir,
                result,
                session=args.eval_session,
            )

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
        _logger.info("Eval summary written to: %s", eval_log_path)
    except Exception as e:
        _logger.warning("Post-training eval failed: %s", e, exc_info=True)


def main():
    parser = get_lmc_train_parser()
    args = parser.parse_args()
    if getattr(args, 'device', '').startswith('cuda:') and os.environ.get('CUDA_VISIBLE_DEVICES'):
        args.device = 'cuda:0'
    setup_experiment(args)

    if not args.use_lmc:
        trainer = run_vanilla_iterative_baseline(args)
        _logger.info("Training completed.")
        run_post_train_eval(args, trainer)
    else:
        trainer = TrainerACEFCNLMC(args)
        trainer.train()
        _logger.info("Training completed.")
        run_post_train_eval(args, trainer)

    # 创建结果索引
    if hasattr(args, 'result_manager'):
        _logger.info("Creating results index...")
        args.result_manager.create_results_index()
        args.result_manager.print_results_summary()


if __name__ == '__main__':
    main()


def _build_vanilla_iterative_args(args):
    vanilla_args = copy.copy(args)
    vanilla_args.iterations = int(args.vanilla_iterations)
    vanilla_args.iter_buffer_size = int(args.training_buffer_size)
    vanilla_args.reset_optimizer_each_iter = False

    if getattr(vanilla_args, "buffer_batch_size", None) == 10:
        _logger.info(
            "[Vanilla parity] Overriding buffer_batch_size: 10 -> 1 "
            "to match ACE vanilla iterative baseline."
        )
        vanilla_args.buffer_batch_size = 1

    return vanilla_args


def _run_standard_eval(args, output_map, session):
    from test_ace_lmc import run_evaluation

    eval_device = getattr(args, "post_train_eval_device", "cuda:0")
    eval_opt = argparse.Namespace(
        scene=args.scene,
        network=output_map,
        encoder_path=args.encoder_path,
        device=eval_device,
        image_resolution=args.image_resolution,
        session=session,
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
    vanilla_args = _build_vanilla_iterative_args(args)
    iter_buffer_size = vanilla_args.iter_buffer_size

    trainer_args = copy.copy(vanilla_args)
    trainer_args.training_buffer_size = iter_buffer_size * vanilla_args.iterations

    trainer = TrainerACEFCN(trainer_args)
    trainer.training_start = time.time()

    out_base = Path(vanilla_args.output_map).parent
    best_score = -float("inf")
    best_iter = None

    _logger.info("=" * 80)
    _logger.info("[Vanilla] Running ACE-FCN iterative baseline via TrainerACEFCN")
    _logger.info(
        "[Vanilla] iterations=%d iter_buffer_size=%d buffer_batch_size=%d epochs=%d",
        vanilla_args.iterations, iter_buffer_size,
        vanilla_args.buffer_batch_size, vanilla_args.epochs,
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

        score = -float("inf")
        if vanilla_args.eval_each_iteration:
            eval_result = _run_standard_eval(
                vanilla_args, iter_ckpt, session=f"iter_{iteration_idx + 1:02d}",
            )
            score = _score_standard_eval(eval_result, vanilla_args.best_metric)
            _logger.info(
                "[Vanilla] Iter %02d eval: median=%.2fdeg/%.2fcm pct5=%.2f pct25_5=%.2f",
                iteration_idx + 1,
                eval_result["median_rErr"], eval_result["median_tErr"],
                eval_result["pct5"], eval_result["pct25_5"],
            )
        else:
            score = float(iteration_idx + 1)

        if score > best_score:
            best_score = score
            best_iter = iteration_idx + 1
            os.replace(iter_ckpt, vanilla_args.output_map)
            _logger.info("[Vanilla] Updated best checkpoint at iter %d (score=%.4f)", best_iter, score)
        elif iter_ckpt.exists():
            if vanilla_args.keep_best_only:
                iter_ckpt.unlink()
            else:
                iter_ckpt.rename(out_base / f"{vanilla_args.output_map.stem}.iter_{iteration_idx + 1:02d}.pt")

    if not vanilla_args.output_map.exists():
        trainer.save_model(vanilla_args.output_map)

    if vanilla_args.eval_after_train:
        _logger.info("Running post-training evaluation...")
        result = _run_standard_eval(vanilla_args, vanilla_args.output_map, vanilla_args.eval_session)
        _logger.info("========== Post-train Eval ==========")
        _logger.info(
            "  Median: %.2f deg, %.2f cm | 25cm/5deg: %.2f%% | 10cm/5deg: %.2f%% | 5cm/5deg: %.2f%%",
            result["median_rErr"], result["median_tErr"],
            result["pct25_5"], result["pct10_5"], result["pct5"],
        )
        _logger.info("=====================================")
        _write_vanilla_eval_summary(vanilla_args, result)

    _logger.info(
        "Vanilla run completed. Total time: %.1fs | best_iter=%s | best_score=%.4f",
        time.time() - trainer.training_start, best_iter, best_score,
    )

    # 保存训练摘要
    if hasattr(vanilla_args, 'result_manager'):
        training_stats = {
            'total_time': time.time() - trainer.training_start,
            'best_iter': best_iter,
            'best_score': best_score,
            'best_metric': vanilla_args.best_metric,
            'num_iterations': vanilla_args.vanilla_iterations,
            'final_checkpoint': str(vanilla_args.output_map),
        }
        vanilla_args.result_manager.save_training_summary(vanilla_args.run_dir, training_stats)

    return trainer


def _apply_lmc_profile(args):
    profile = getattr(args, "lmc_profile", "legacy")

    def _finalize_buffer_sampling_default(default_value: bool):
        if getattr(args, "buffer_sampling_replacement", None) is None:
            args.buffer_sampling_replacement = default_value
            return True
        return False

    if not args.use_lmc:
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

    profile_defaults = {"s1_lr_scale_later": 1.0}
    parser_defaults = {
        "s1_lr_scale_later": 0.2,
        "s2_repro_rewind_first_ratio": 0.20,
        "s2_repro_rewind_later_ratio": 0.08,
        "s2_lr_boost_first": 1.2,
        "s2_lr_boost_later": 1.0,
    }
    applied = []
    for key, val in profile_defaults.items():
        if getattr(args, key) == parser_defaults[key]:
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
        "[LMC profile] mapany_flow_v1 applied defaults: %s",
        ", ".join(applied) if applied else "none",
    )

