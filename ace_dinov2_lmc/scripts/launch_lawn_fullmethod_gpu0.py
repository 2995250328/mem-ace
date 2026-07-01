#!/usr/bin/env python3
"""Launch the Wayspots lawn final-method flow on GPU0."""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path


SCENE = "wayspots_lawn"
GPU = "0"

ROOT_DIR = Path(__file__).resolve().parents[2]
SCENE_ROOT = Path("/data/xwh/Wayspots") / SCENE
MEMORY_PATH = (
    Path("/data/xwh/ace_dinov2_lmc/04_evaluation/wayspots_ace_fcn_lmc_suite")
    / "20260530_181745"
    / "memory"
    / SCENE
    / "memory_ace_fcn_sparse_sp_r4.pt"
)
ACE_ENCODER_PATH = ROOT_DIR / "ace_encoder_pretrained.pt"
SUPERPOINT_WEIGHTS = ROOT_DIR / "ace_dinov2_lmc" / "superpoint_v1.pth"
GLACE_ROOT = Path("/home/xwh/project/glace")

RUN_ROOT = Path(
    "/data/xwh/ace_dinov2_lmc/04_evaluation/wayspots/stgs_guided_bigbuf"
    "/20260629_lawn_fullmethod_gpu0"
)
STAGE2_RUN_ROOT = Path(
    "/data/xwh/ace_dinov2_lmc/04_evaluation/wayspots/stage2/stgs_stage1_transfer"
    "/20260629_lawn_fullmethod_gpu0"
)

CONDA_ENV = os.environ.get("CONDA_ENV", "mapanything")
PYTHON_BIN = os.environ.get("PYTHON_BIN", "python")
STAMP = os.environ.get("STAMP", datetime.now().strftime("%Y%m%d_%H%M%S"))

LOG_DIR = RUN_ROOT / "logs"
PREPROCESS_ROOT = RUN_ROOT / "preprocess" / SCENE
WORKSPACE = PREPROCESS_ROOT / "superpoint_workspace"
MODEL_DIR = WORKSPACE / "triangulated_model"
SPARSE_DEPTH_SUBDIR = os.environ.get(
    "SPARSE_DEPTH_SUBDIR", f"sparse_depth_superpoint_strict_{STAMP}"
)
SPARSE_DEPTH_DIR = SCENE_ROOT / "train" / SPARSE_DEPTH_SUBDIR
SIDECAR_DIR = PREPROCESS_ROOT / "colmap_keyframe_channel_v1_sp_strict"
SIDECAR_PATH = SIDECAR_DIR / "keyframe_channel.npz"
SIDECAR_SUMMARY = SIDECAR_DIR / "summary.json"


def log(message: str) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}"
    print(line, flush=True)
    with (LOG_DIR / f"supervisor_{STAMP}.log").open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def run_logged(name: str, cmd: list[str], log_path: Path, env: dict[str, str]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log(f"start {name}; log={log_path}")
    with log_path.open("a", encoding="utf-8") as f:
        f.write(f"[{datetime.now()}] {name}\n")
        f.write(shlex.join(cmd) + "\n")
        f.flush()
        subprocess.run(
            cmd,
            cwd=ROOT_DIR,
            env=env,
            stdout=f,
            stderr=subprocess.STDOUT,
            check=True,
        )
    log(f"done {name}")


def require_path(path: Path, label: str, is_dir: bool = False) -> None:
    ok = path.is_dir() if is_dir else path.is_file()
    if not ok:
        raise FileNotFoundError(f"missing {label}: {path}")


def preflight() -> None:
    require_path(SCENE_ROOT / "train" / "rgb", "train/rgb", is_dir=True)
    require_path(SCENE_ROOT / "train" / "poses", "train/poses", is_dir=True)
    require_path(SCENE_ROOT / "train" / "calibration", "train/calibration", is_dir=True)
    require_path(SCENE_ROOT / "train" / "sparse_depth", "C1 sparse_depth", is_dir=True)
    require_path(SCENE_ROOT / "train" / "features.npy", "GLACE train features")
    require_path(SCENE_ROOT / "test" / "features.npy", "GLACE test features")
    require_path(MEMORY_PATH, "memory")
    require_path(ACE_ENCODER_PATH, "ACE encoder")
    require_path(SUPERPOINT_WEIGHTS, "SuperPoint weights")


def base_env(cuda_visible_devices: str = GPU) -> dict[str, str]:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
    env["PYTHONPATH"] = (
        "/data/xwh/SuperPointPretrainedNetwork:"
        + env.get("PYTHONPATH", "")
    ).rstrip(":")
    return env


def build_assets() -> None:
    PREPROCESS_ROOT.mkdir(parents=True, exist_ok=True)
    SIDECAR_DIR.mkdir(parents=True, exist_ok=True)

    if not (MODEL_DIR / "points3D.bin").is_file():
        cmd = [
            "conda",
            "run",
            "--no-capture-output",
            "-n",
            CONDA_ENV,
            PYTHON_BIN,
            "ace_dinov2_lmc/tools/wayspots_known_pose_sparse_depth.py",
            str(SCENE_ROOT),
            "--split",
            "train",
            "--workspace",
            str(WORKSPACE),
            "--output-subdir",
            SPARSE_DEPTH_SUBDIR,
            "--feature-backend",
            "superpoint",
            "--superpoint-weights",
            str(SUPERPOINT_WEIGHTS),
            "--superpoint-conf-thresh",
            "0.05",
            "--superpoint-max-keypoints",
            "1024",
            "--superpoint-nms-dist",
            "8",
            "--superpoint-nn-thresh",
            "0.60",
            "--matcher",
            "pairs",
            "--match-window",
            "20",
            "--num-threads",
            "4",
            "--min-triangulation-angle",
            "3.0",
            "--max-depth-m",
            "1000",
            "--use-gpu",
            "--gpu-index",
            "0",
            "--overwrite",
        ]
        run_logged(
            f"{SCENE} build_superpoint_workspace",
            cmd,
            LOG_DIR / f"{SCENE}_build_superpoint_workspace_{STAMP}.log",
            base_env(GPU),
        )
    else:
        log(f"reuse SuperPoint workspace: {MODEL_DIR}")

    if not SIDECAR_PATH.is_file():
        cmd = [
            "conda",
            "run",
            "--no-capture-output",
            "-n",
            CONDA_ENV,
            PYTHON_BIN,
            "ace_dinov2_lmc/tools/build_colmap_keyframe_channel.py",
            str(SCENE_ROOT),
            "--split",
            "train",
            "--model-dir",
            str(MODEL_DIR),
            "--output-dir",
            str(SIDECAR_DIR),
            "--image-resolution",
            "512",
            "--output-subsample",
            "8",
            "--round-image-multiple",
            "1",
            "--min-track-length",
            "4",
            "--max-reproj-error",
            "2.0",
            "--min-parallax-deg",
            "2.0",
            "--max-parallax-deg",
            "60.0",
            "--max-anchor-alignment-px",
            "2.0",
            "--sparse-depth-dir",
            str(SPARSE_DEPTH_DIR),
        ]
        run_logged(
            f"{SCENE} build_keyframe_channel",
            cmd,
            LOG_DIR / f"{SCENE}_build_keyframe_channel_{STAMP}.log",
            base_env(GPU),
        )
    else:
        log(f"reuse keyframe sidecar: {SIDECAR_PATH}")

    require_path(SIDECAR_SUMMARY, "keyframe sidecar summary")
    summary = json.loads(SIDECAR_SUMMARY.read_text(encoding="utf-8"))
    rows = int(summary.get("rows", 0))
    log(f"sidecar rows={rows}; sidecar={SIDECAR_PATH}")
    if rows <= 0:
        raise RuntimeError(f"empty keyframe sidecar: {SIDECAR_PATH}")


def common_stage1_args() -> list[str]:
    return [
        "--model_backend",
        "ace_fcn_lmc",
        "--data_backend",
        "ace",
        "--use_lmc",
        "True",
        "--lmc_flow",
        "ace_g",
        "--memory_path",
        str(MEMORY_PATH),
        "--use_scale_token",
        "False",
        "--ace_encoder_path",
        str(ACE_ENCODER_PATH),
        "--ace_lmc_global_head_mode",
        "none",
        "--lmc_iterations",
        "6",
        "--num_latent_tokens",
        "64",
        "--lmc_fusion_refinement_mode",
        "single",
        "--lmc_fusion_cascade_layers",
        "4",
        "--lmc_fusion_assembly_gamma_init",
        "0.0",
        "--lmc_fusion_reread_delta_alpha",
        "1.0",
        "--lmc_fusion_reread_scalar_gate",
        "False",
        "--lmc_fusion_reread_gate_init",
        "0.0",
        "--lmc_fusion_reread_post_norm",
        "True",
        "--lmc_fusion_reread_trust_region_ratio",
        "0.0",
        "--lmc_fusion_reread_temperature",
        "1.0",
        "--lmc_fusion_reread_common_scale",
        "1.0",
        "--lmc_fusion_reread_effective_ratio_cap",
        "0.0",
        "--lmc_fusion_reread_qknorm_eps",
        "1e-6",
        "--lmc_fusion_reread_qknorm_tau_init",
        "0.0",
        "--lmc_fusion_reread_layerscale_patch_init",
        "0.01",
        "--lmc_fusion_reread_layerscale_common_init",
        "0.0",
        "--lmc_fusion_reread_warmup_mode",
        "none",
        "--lmc_fusion_reread_warmup_iters",
        "0",
        "--lmc_fusion_reread_warmup_start",
        "0.0",
        "--lmc_fusion_reread_geo_lambda",
        "1.0",
        "--lmc_fusion_reread_geo_sigma",
        "1.0",
        "--lmc_fusion_reread_geo_sigma_mode",
        "fixed",
        "--lmc_fusion_reread_geo_sigma_beta",
        "1.0",
        "--lmc_fusion_reread_geo_sigma_min",
        "0.5",
        "--s1_loss_step_mode",
        "fixed_zero",
        "--s1_early_stop",
        "False",
        "--lmc_log_runtime_stats",
        "True",
        "--lmc_runtime_stats_interval",
        "100",
        "--lmc_runtime_stats_max_pixels",
        "4096",
        "--num_data_loader_workers",
        "12",
        "--eval_num_workers",
        "6",
        "--eval_deterministic",
        "True",
        "--eval_dsacstar_seed",
        "1305",
        "--eval_dsacstar_seed_per_frame",
        "True",
        "--iteration_eval_seed",
        "1305",
        "--iteration_eval_hypotheses",
        "256",
        "--ace_g_fusion_in_s2",
        "True",
        "--ace_g_cross_iter_eval",
        "True",
        "--s1_use_buffer",
        "True",
        "--s1_loss_mode",
        "sample_per_image",
        "--s1_buffer_refill_mode",
        "full",
        "--s1_last_iter_use_final_buffer",
        "True",
        "--image_resolution",
        "512",
        "--batch_size",
        "4096",
        "--training_buffer_size",
        "10000000",
        "--buffer_size_final",
        "10000000",
        "--buffer_on_cpu",
        "True",
        "--buffer_on_cpu_final",
        "True",
        "--samples_per_image",
        "512",
        "--buffer_sample_valid_coords",
        "True",
        "--buffer_valid_coord_sample_ratio",
        "1.0",
        "--buffer_valid_coord_neighbor_radius",
        "1",
        "--buffer_valid_coord_neighbor_mode",
        "cross",
        "--c1_aux_depth_root",
        str(SCENE_ROOT / "train" / "sparse_depth"),
        "--c1_aux_depth_kind",
        "sparse_depth",
        "--epochs",
        "24",
        "--post_train_eval_seeds",
        "1305",
        "2026",
        "4242",
        "--post_train_hypotheses",
        "256",
    ]


def stgs_args() -> list[str]:
    return [
        "--use_sfm_track_guided_sampling",
        "True",
        "--sfm_track_keyframe_channel_path",
        str(SIDECAR_PATH),
        "--sfm_track_guided_batch_size",
        "512",
        "--sfm_track_guided_sampling_strategy",
        "balanced_replace",
        "--sfm_track_guided_fraction",
        "0.10",
        "--sfm_track_guided_mode",
        "anchor_only",
        "--sfm_track_guided_main_loss_mode",
        "include",
        "--sfm_track_guided_source_target_mode",
        "patch_center",
        "--sfm_track_guided_aux_normalizer",
        "full_batch",
        "--sfm_track_anchor_self_weight",
        "0.5",
        "--sfm_track_anchor_use_alignment_weight",
        "True",
        "--sfm_track_inter_frame_weight",
        "0.0",
        "--sfm_track_inter_frame_dropout",
        "0.5",
        "--sfm_track_inter_frame_start_ratio",
        "0.2",
        "--sfm_track_inter_frame_decay_last_ratio",
        "0.3",
        "--sfm_track_inter_frame_max_px",
        "100.0",
    ]


def find_latest_checkpoint(root: Path, pattern: str) -> Path:
    matches = sorted(root.glob(pattern), key=lambda p: p.stat().st_mtime)
    if not matches:
        raise FileNotFoundError(f"no checkpoint under {root} matching {pattern}")
    return matches[-1]


def train_stage1() -> Path:
    tag = f"{SCENE}_anchor_v2_f010_w05_it6_buf10m_fullflow_{STAMP}"
    exp_root = RUN_ROOT / SCENE / "anchor_v2_f010_w05"
    exp_subdir = "stage1_local_stgs_anchor_bigbuf_it6"
    cmd = [
        "conda",
        "run",
        "--no-capture-output",
        "-n",
        CONDA_ENV,
        PYTHON_BIN,
        "ace_dinov2_lmc/train_ace_dinov2_lmc.py",
        str(SCENE_ROOT),
        f"{tag}.pt",
        "--run_name",
        tag,
        "--device",
        "cuda:0",
        "--post_train_eval_device",
        "cuda:0",
        "--experiment_root",
        str(exp_root),
        "--experiment_subdir",
        exp_subdir,
    ]
    cmd += common_stage1_args()
    cmd += stgs_args()
    run_logged(
        f"{SCENE} Stage1 anchor_v2_f010_w05",
        cmd,
        LOG_DIR / f"{tag}_gpu0.log",
        base_env(GPU),
    )
    ckpt = find_latest_checkpoint(exp_root / exp_subdir, "**/best_K64_it6_*.pt")
    log(f"Stage1 checkpoint: {ckpt}")
    return ckpt


def train_stage2(stage1_ckpt: Path) -> None:
    tag = f"{SCENE}_anchor_v2_f010_w05_stage2_r2_it10_buf10m_{STAMP}"
    exp_root = STAGE2_RUN_ROOT / SCENE / "anchor_v2_f010_w05"
    exp_subdir = "stage2_r2_bigbuf_from_anchor_v2_f010_w05_it10_buf10m_final12m_bs8192_pct50_5"
    cmd = [
        "conda",
        "run",
        "--no-capture-output",
        "-n",
        CONDA_ENV,
        PYTHON_BIN,
        "ace_dinov2_lmc/train_ace_dinov2_lmc.py",
        str(SCENE_ROOT),
        f"{tag}.pt",
        "--run_name",
        tag,
        "--model_backend",
        "ace_fcn_lmc",
        "--data_backend",
        "ace",
        "--post_train_eval_scene",
        str(SCENE_ROOT),
        "--use_lmc",
        "True",
        "--lmc_flow",
        "ace_g",
        "--memory_path",
        str(MEMORY_PATH),
        "--use_scale_token",
        "False",
        "--ace_encoder_path",
        str(ACE_ENCODER_PATH),
        "--ace_lmc_global_head_mode",
        "glace_concat",
        "--ace_lmc_local_checkpoint_path",
        str(stage1_ckpt),
        "--ace_lmc_freeze_local_stack",
        "True",
        "--ace_lmc_stage2_feature_source",
        "raw_backbone",
        "--ace_lmc_global_feature_mode",
        "glace",
        "--ace_lmc_global_gate_init",
        "1.0",
        "--ace_lmc_global_gate_learnable",
        "False",
        "--ace_lmc_global_normalize",
        "True",
        "--ace_lmc_global_noise_std",
        "0.1",
        "--glace_root",
        str(GLACE_ROOT),
        "--glace_feat_name",
        "features.npy",
        "--glace_head_channels",
        "512",
        "--glace_mlp_ratio",
        "1.0",
        "--num_head_blocks",
        "4",
        "--best_metric",
        "pct50_5",
        "--device",
        "cuda:0",
        "--post_train_eval_device",
        "cuda:0",
        "--experiment_root",
        str(exp_root),
        "--experiment_subdir",
        exp_subdir,
        "--lmc_iterations",
        "10",
        "--num_latent_tokens",
        "64",
        "--lmc_fusion_refinement_mode",
        "single",
        "--lmc_fusion_cascade_layers",
        "4",
        "--lmc_fusion_assembly_gamma_init",
        "0.0",
        "--lmc_train_steps",
        "600",
        "--lmc_warmup_steps",
        "2000",
        "--s1_early_stop",
        "False",
        "--num_data_loader_workers",
        "12",
        "--eval_num_workers",
        "6",
        "--eval_deterministic",
        "True",
        "--eval_dsacstar_seed",
        "1305",
        "--eval_dsacstar_seed_per_frame",
        "True",
        "--iteration_eval_seed",
        "1305",
        "--iteration_eval_hypotheses",
        "256",
        "--ace_g_fusion_in_s2",
        "True",
        "--ace_g_fusion_lr_ratio",
        "0.01",
        "--ace_g_cross_iter_eval",
        "False",
        "--s1_use_buffer",
        "True",
        "--s1_loss_mode",
        "sample_per_image",
        "--s1_buffer_refill_mode",
        "full",
        "--image_resolution",
        "512",
        "--batch_size",
        "8192",
        "--training_buffer_size",
        "10000000",
        "--buffer_size_final",
        "12000000",
        "--buffer_on_cpu",
        "True",
        "--buffer_on_cpu_final",
        "True",
        "--samples_per_image",
        "512",
        "--buffer_sample_valid_coords",
        "True",
        "--buffer_valid_coord_sample_ratio",
        "1.0",
        "--buffer_valid_coord_neighbor_radius",
        "1",
        "--buffer_valid_coord_neighbor_mode",
        "cross",
        "--c1_aux_depth_root",
        str(SCENE_ROOT / "train" / "sparse_depth"),
        "--c1_aux_depth_kind",
        "sparse_depth",
        "--post_train_eval_seeds",
        "1305",
        "2026",
        "4242",
        "--post_train_hypotheses",
        "256",
    ]
    run_logged(
        f"{SCENE} Stage2 anchor_v2_f010_w05",
        cmd,
        LOG_DIR / f"{tag}_gpu0.log",
        base_env(GPU),
    )


def write_plan() -> None:
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    STAGE2_RUN_ROOT.mkdir(parents=True, exist_ok=True)
    text = "\n".join(
        [
            "purpose: Wayspots lawn final-method full flow on GPU0",
            f"stamp: {STAMP}",
            f"scene: {SCENE}",
            f"gpu: {GPU}",
            f"run_root_stage1: {RUN_ROOT}",
            f"run_root_stage2: {STAGE2_RUN_ROOT}",
            f"sparse_depth_subdir: {SPARSE_DEPTH_SUBDIR}",
            f"sidecar: {SIDECAR_PATH}",
            "stage1: anchor_v2_f010_w05, it6, buf10m, batch4096",
            "stage2: r2 glace_concat, it10, buf10m/final12m, batch8192",
            "post_train_eval_seeds: 1305 2026 4242",
            "",
        ]
    )
    (RUN_ROOT / f"matrix_plan_{STAMP}.txt").write_text(text, encoding="utf-8")


def main() -> int:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    write_plan()
    log(f"plan: {RUN_ROOT / f'matrix_plan_{STAMP}.txt'}")
    log("GPU policy: CUDA_VISIBLE_DEVICES=0, train device=cuda:0")
    preflight()
    build_assets()
    stage1_ckpt = train_stage1()
    train_stage2(stage1_ckpt)
    log("lawn full-method flow complete")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        log(f"FAILED: {exc}")
        raise
