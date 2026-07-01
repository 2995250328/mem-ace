#!/usr/bin/env python3
"""Wait for GPU3 to finish the current cubes run, then launch inscription Stage2."""

from __future__ import annotations

import os
import shlex
import subprocess
import time
from datetime import datetime
from pathlib import Path


SCENE = "wayspots_inscription"
PHYSICAL_GPU = "3"
CONDA_ENV = os.environ.get("CONDA_ENV", "mapanything")
PYTHON_BIN = os.environ.get("PYTHON_BIN", "python")
WAIT_INTERVAL_SEC = int(os.environ.get("WAIT_INTERVAL_SEC", "120"))
STAMP = os.environ.get("STAMP", datetime.now().strftime("%Y%m%d_%H%M%S"))

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
GLACE_ROOT = Path("/home/xwh/project/glace")
SOURCE_STAGE1_ROOT = Path(
    "/data/xwh/ace_dinov2_lmc/04_evaluation/wayspots/stgs_guided_bigbuf"
    "/20260627_stage1_it6_buf10m_fullflow_extra_gpu23"
)
RUN_ROOT = Path(
    "/data/xwh/ace_dinov2_lmc/04_evaluation/wayspots/stage2/stgs_stage1_transfer"
    "/20260629_inscription_fullmethod_gpu3"
)
LOG_DIR = RUN_ROOT / "logs"


def log(message: str) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}"
    print(line, flush=True)
    with (LOG_DIR / f"supervisor_{STAMP}.log").open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def run_text(cmd: list[str]) -> str:
    proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    return proc.stdout


def cubes_cuda3_pids() -> list[str]:
    out = run_text(["pgrep", "-af", "wayspots_cubes"])
    pids: list[str] = []
    for line in out.splitlines():
        if "train_ace_dinov2_lmc.py" not in line:
            continue
        if "--device cuda:3" not in line and "--post_train_eval_device cuda:3" not in line:
            continue
        parts = line.split(maxsplit=1)
        if parts:
            pids.append(parts[0])
    return sorted(set(pids), key=int)


def wait_for_gpu3_cubes() -> None:
    while True:
        pids = cubes_cuda3_pids()
        if not pids:
            log("no wayspots_cubes cuda:3 training process found; launching inscription")
            return
        log(f"waiting for wayspots_cubes cuda:3 pids={','.join(pids)}")
        time.sleep(WAIT_INTERVAL_SEC)


def require_path(path: Path, label: str, is_dir: bool = False) -> None:
    ok = path.is_dir() if is_dir else path.is_file()
    if not ok:
        raise FileNotFoundError(f"missing {label}: {path}")


def find_stage1_checkpoint() -> Path:
    root = (
        SOURCE_STAGE1_ROOT
        / SCENE
        / "anchor_v2_f010_w05"
        / "stage1_local_stgs_anchor_bigbuf_it6"
    )
    matches = sorted(root.glob("**/best_K64_it6_*.pt"), key=lambda p: p.stat().st_mtime)
    if not matches:
        raise FileNotFoundError(f"missing Stage1 checkpoint under {root}")
    return matches[-1]


def preflight(stage1_ckpt: Path) -> None:
    require_path(SCENE_ROOT / "train", "train split", is_dir=True)
    require_path(SCENE_ROOT / "test", "test split", is_dir=True)
    require_path(SCENE_ROOT / "train" / "features.npy", "GLACE train features")
    require_path(SCENE_ROOT / "test" / "features.npy", "GLACE test features")
    require_path(SCENE_ROOT / "train" / "sparse_depth", "C1 sparse_depth", is_dir=True)
    require_path(MEMORY_PATH, "memory")
    require_path(ACE_ENCODER_PATH, "ACE encoder")
    require_path(stage1_ckpt, "Stage1 checkpoint")


def train_stage2(stage1_ckpt: Path) -> None:
    tag = f"{SCENE}_anchor_v2_f010_w05_stage2_r2_it10_buf10m_{STAMP}"
    exp_root = RUN_ROOT / SCENE / "anchor_v2_f010_w05"
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
        f"cuda:{PHYSICAL_GPU}",
        "--post_train_eval_device",
        f"cuda:{PHYSICAL_GPU}",
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

    env = os.environ.copy()
    env.pop("CUDA_VISIBLE_DEVICES", None)
    log_file = LOG_DIR / f"{tag}_physgpu{PHYSICAL_GPU}.log"
    log(f"start {SCENE} Stage2 on physical gpu{PHYSICAL_GPU}; log={log_file}")
    with log_file.open("a", encoding="utf-8") as f:
        f.write(f"[{datetime.now()}] physical_gpu={PHYSICAL_GPU} requested_device=cuda:{PHYSICAL_GPU}\n")
        f.write(f"stage1={stage1_ckpt}\n")
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
    log(f"done {SCENE} Stage2")


def write_plan(stage1_ckpt: Path) -> None:
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    text = "\n".join(
        [
            "purpose: finish Wayspots inscription full method by running Stage2 after GPU3 frees",
            f"stamp: {STAMP}",
            f"scene: {SCENE}",
            f"physical_gpu: {PHYSICAL_GPU}",
            f"requested_device: cuda:{PHYSICAL_GPU}",
            f"run_root: {RUN_ROOT}",
            f"source_stage1_root: {SOURCE_STAGE1_ROOT}",
            f"stage1_checkpoint: {stage1_ckpt}",
            "stage2: r2 glace_concat, it10, buf10m/final12m, batch8192",
            "post_train_eval_seeds: 1305 2026 4242",
            "",
        ]
    )
    (RUN_ROOT / f"matrix_plan_{STAMP}.txt").write_text(text, encoding="utf-8")
    log(f"plan: {RUN_ROOT / f'matrix_plan_{STAMP}.txt'}")


def main() -> int:
    stage1_ckpt = find_stage1_checkpoint()
    write_plan(stage1_ckpt)
    preflight(stage1_ckpt)
    log(f"GPU policy: request --device cuda:{PHYSICAL_GPU}; training entry maps it to process-local cuda:0")
    wait_for_gpu3_cubes()
    train_stage2(stage1_ckpt)
    log("inscription Stage2 flow complete")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        log(f"FAILED: {exc}")
        raise
