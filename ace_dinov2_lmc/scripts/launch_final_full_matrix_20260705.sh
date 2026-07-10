#!/usr/bin/env bash
set -euo pipefail

# Final full experiment queue for ACE-DINOv2-LMC / ACE-G.
#
# Scope:
#   - Indoor6 DINOv2 + MapAnything memory: Stage1 only.
#   - Indoor6 / Wayspots / Cambridge ACE-FCN memory: Stage1 + Stage2 GLACE concat.
#
# Final protocol defaults:
#   - value_only_raw fusion geometry.
#   - fixed global GeoLMC compression; automatic visibility fallback disabled.
#   - STGS/SFM inter-frame loss, w=0.05.
#   - 10 iterations, 10M buffer, final 10M buffer, post-train hypotheses 256.
#
# Usage from /home/xwh/project/ace_depth:
#   ACTION=preflight SIDECAR_RUN_ROOT=/data/... bash ace_dinov2_lmc/scripts/launch_final_full_matrix_20260705.sh
#   ACTION=launch    SIDECAR_RUN_ROOT=/data/... bash ace_dinov2_lmc/scripts/launch_final_full_matrix_20260705.sh
#
# Add GPUs later without interrupting running workers:
#   ACTION=worker WORKER_GPU=2 RUN_ROOT=<same-run-root> SIDECAR_RUN_ROOT=<same-sidecar-root> \
#     bash ace_dinov2_lmc/scripts/launch_final_full_matrix_20260705.sh

ROOT_DIR="${ROOT_DIR:-/home/xwh/project/ace_depth}"
REPO_ROOT="${REPO_ROOT:-${ROOT_DIR}/ace_dinov2_lmc}"
cd "${ROOT_DIR}"

ACTION="${ACTION:-preflight}"  # preflight | launch | worker | aggregate | status
WORKER_GPU="${WORKER_GPU:-}"
CONDA_ENV="${CONDA_ENV:-mapanything}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-/data/xwh/ace_dinov2_lmc/04_evaluation/shared/stage2/final_full/${STAMP}_all_value_raw_auto_ifw005_it10_buf10m_h256_gpu01_expandable}"
SESSION="${SESSION:-final_full_${STAMP}_gpu01}"
GPUS_STR="${GPUS_STR:-0 1}"
REWRITE_MATRIX="${REWRITE_MATRIX:-false}"
DRY_RUN="${DRY_RUN:-false}"
SKIP_EXISTING="${SKIP_EXISTING:-true}"
CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-false}"
SIDECAR_REQUIRED="${SIDECAR_REQUIRED:-true}"
MIN_DATA_FREE_GB="${MIN_DATA_FREE_GB:-150}"
ENFORCE_MIN_DATA_FREE_GB="${ENFORCE_MIN_DATA_FREE_GB:-false}"

LOG_DIR="${RUN_ROOT}/launcher_logs"
MATRIX_FILE="${RUN_ROOT}/matrix.tsv"
LOCK_FILE="${RUN_ROOT}/queue.lock"
MANIFEST_FILE="${RUN_ROOT}/manifest.json"
STATUS_LOG="${LOG_DIR}/queue_status.log"

WAYSPOTS_ROOT="${WAYSPOTS_ROOT:-/data/xwh/Wayspots}"
CAMBRIDGE_ROOT="${CAMBRIDGE_ROOT:-/data/xwh/Cambridge}"
INDOOR6_ACE_ROOT="${INDOOR6_ACE_ROOT:-/data/xwh/indoor6_ace}"
INDOOR6_WAI_ROOT="${INDOOR6_WAI_ROOT:-/data/xwh/mapanything-dataset/wai_data/indoor6}"
INDOOR6_COLMAP_ROOT="${INDOOR6_COLMAP_ROOT:-/data/xwh/indoor6/indoor6-colmap}"

GLACE_ROOT="${GLACE_ROOT:-/home/xwh/project/glace}"
ACE_ENCODER_PATH="${ACE_ENCODER_PATH:-${ROOT_DIR}/ace_encoder_pretrained.pt}"
PYTHON_ABS="${PYTHON_ABS:-/home/xwh/miniforge3/envs/mapanything/bin/python}"

SIDECAR_BASE="${SIDECAR_BASE:-/data/xwh/ace_dinov2_lmc/04_evaluation/shared/memory/stgs_sidecars}"
SIDECAR_RUN_ROOT="${SIDECAR_RUN_ROOT:-}"

INDOOR6_SCENES="${INDOOR6_SCENES:-scene1 scene2a scene3 scene4a scene5 scene6}"
CAMBRIDGE_SCENES="${CAMBRIDGE_SCENES:-Cambridge_GreatCourt Cambridge_KingsCollege Cambridge_OldHospital Cambridge_ShopFacade Cambridge_StMarysChurch}"
WAYSPOTS_SCENES_NORMAL="${WAYSPOTS_SCENES_NORMAL:-wayspots_bears wayspots_cubes wayspots_inscription wayspots_lawn wayspots_map wayspots_squarebench wayspots_tendrils wayspots_therock}"
WAYSPOTS_SCENES_HARD="${WAYSPOTS_SCENES_HARD:-wayspots_statue wayspots_wintersign}"

LMC_ITERATIONS="${LMC_ITERATIONS:-10}"
TRAINING_BUFFER_SIZE="${TRAINING_BUFFER_SIZE:-10000000}"
BUFFER_SIZE_FINAL="${BUFFER_SIZE_FINAL:-10000000}"
NUM_LATENT_TOKENS="${NUM_LATENT_TOKENS:-64}"
IMAGE_RESOLUTION_ACEFCN="${IMAGE_RESOLUTION_ACEFCN:-512}"
IMAGE_RESOLUTION_DINO="${IMAGE_RESOLUTION_DINO:-518}"
ACEFCN_BATCH_SIZE="${ACEFCN_BATCH_SIZE:-4096}"
DINO_BATCH_SIZE="${DINO_BATCH_SIZE:-10240}"
SAMPLES_PER_IMAGE="${SAMPLES_PER_IMAGE:-512}"
POST_TRAIN_HYPOTHESES="${POST_TRAIN_HYPOTHESES:-256}"
POST_TRAIN_EVAL_SEEDS_STR="${POST_TRAIN_EVAL_SEEDS_STR:-1305 2026 4242}"

LMC_FUSION_GEOMETRY_MODE="${LMC_FUSION_GEOMETRY_MODE:-value_only_raw}"
# Paper-facing protocol: every scene uses the global GeoLMC compressor.
LMC_AUTO_MODE_BY_VISIBILITY="False"

SFM_TRACK_INTER_FRAME_WEIGHT="${SFM_TRACK_INTER_FRAME_WEIGHT:-0.05}"
SFM_TRACK_GUIDED_BATCH_SIZE="${SFM_TRACK_GUIDED_BATCH_SIZE:-512}"
SFM_TRACK_GUIDED_FRACTION="${SFM_TRACK_GUIDED_FRACTION:-0.10}"
SFM_TRACK_INTER_FRAME_DROPOUT="${SFM_TRACK_INTER_FRAME_DROPOUT:-0.5}"
SFM_TRACK_INTER_FRAME_START_RATIO="${SFM_TRACK_INTER_FRAME_START_RATIO:-0.2}"
SFM_TRACK_INTER_FRAME_DECAY_LAST_RATIO="${SFM_TRACK_INTER_FRAME_DECAY_LAST_RATIO:-0.3}"
SFM_TRACK_INTER_FRAME_MAX_PX="${SFM_TRACK_INTER_FRAME_MAX_PX:-100.0}"
SFM_TRACK_ANCHOR_SELF_WEIGHT="${SFM_TRACK_ANCHOR_SELF_WEIGHT:-0.5}"

STAGE1_SUBDIR="${STAGE1_SUBDIR:-stage1_value_raw_auto_ifw005_it10_buf10m}"
STAGE2_SUBDIR="${STAGE2_SUBDIR:-stage2_glace_concat_value_raw_auto_ifw005_it10_buf10m}"
DINO_VARIANT_LABEL="${DINO_VARIANT_LABEL:-dino_ma_stage1_value_raw_auto_ifw005_it10_buf10m}"

mkdir -p "${RUN_ROOT}" "${LOG_DIR}" "${RUN_ROOT}/scripts"

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" | tee -a "${STATUS_LOG}"
}

bool_true() {
  case "${1,,}" in
    1|true|yes|y|on) return 0 ;;
    *) return 1 ;;
  esac
}

latest_sidecar_root() {
  find "${SIDECAR_BASE}" -maxdepth 1 -type d -name '*final_all_value_raw_auto_ifw005*' 2>/dev/null | sort | tail -n 1 || true
}

resolve_sidecar_root() {
  if [[ -z "${SIDECAR_RUN_ROOT}" ]]; then
    SIDECAR_RUN_ROOT="$(latest_sidecar_root)"
  fi
  if [[ -z "${SIDECAR_RUN_ROOT}" || ! -d "${SIDECAR_RUN_ROOT}" ]]; then
    echo "ERROR: SIDECAR_RUN_ROOT is required and must point to prepared sidecars." >&2
    echo "       Run prepare_final_full_sidecars_20260705.sh first, or pass SIDECAR_RUN_ROOT=/data/..." >&2
    exit 2
  fi
}

write_manifest() {
  "${PYTHON_ABS}" - "${MANIFEST_FILE}" "${RUN_ROOT}" "${SIDECAR_RUN_ROOT}" "${GPUS_STR}" <<'PYMANIFEST'
import json, sys, datetime
path, run_root, sidecar_root, gpus = sys.argv[1:5]
manifest = {
    "dataset": "indoor6_wayspots_cambridge",
    "track": "final_full",
    "method": "acefcn_glace_and_dino_ma_memory",
    "protocol": "value_only_raw_auto_global_local_stgs_inter_frame_w005_it10_buf10m",
    "created_at": datetime.datetime.now().isoformat(timespec="seconds"),
    "run_root": run_root,
    "sidecar_run_root": sidecar_root,
    "gpus_initial": gpus.split(),
    "metrics": {
        "indoor6": "Acc25 and median error",
        "wayspots": "Acc50 and Acc10",
        "cambridge": "median error only",
    },
    "notes": [
        "Wayspots statue/wintersign are queued last because DSAC is slow.",
        "Queue is dynamic; ACTION=worker can attach GPUs 2/3 later using the same RUN_ROOT.",
        "Metric aggregation must use metric-wise best across iter/cross/post-train rows.",
    ],
}
with open(path, "w", encoding="utf-8") as f:
    json.dump(manifest, f, ensure_ascii=False, indent=2)
PYMANIFEST
  cp "${REPO_ROOT}/scripts/launch_final_full_matrix_20260705.sh" "${RUN_ROOT}/scripts/" 2>/dev/null || true
  cp "${REPO_ROOT}/scripts/prepare_final_full_sidecars_20260705.sh" "${RUN_ROOT}/scripts/" 2>/dev/null || true
}

write_matrix() {
  resolve_sidecar_root
  if [[ -s "${MATRIX_FILE}" && "${REWRITE_MATRIX}" != "true" ]]; then
    log "reuse matrix: ${MATRIX_FILE}"
    return 0
  fi
  "${PYTHON_ABS}" - "${MATRIX_FILE}" "${RUN_ROOT}" "${SIDECAR_RUN_ROOT}" \
    "${INDOOR6_SCENES}" "${CAMBRIDGE_SCENES}" "${WAYSPOTS_SCENES_NORMAL}" "${WAYSPOTS_SCENES_HARD}" \
    "${INDOOR6_ACE_ROOT}" "${CAMBRIDGE_ROOT}" "${WAYSPOTS_ROOT}" \
    "${IMAGE_RESOLUTION_DINO}" "${IMAGE_RESOLUTION_ACEFCN}" <<'PYMATRIX'
import csv, sys
from pathlib import Path
(
    matrix_path, run_root, sidecar_root,
    indoor_s, cambridge_s, wayspots_normal_s, wayspots_hard_s,
    indoor_root, cambridge_root, wayspots_root,
    dino_res, ace_res,
) = sys.argv[1:13]
run_root = Path(run_root)
sidecar_root = Path(sidecar_root)
sidecar_rows = {}
sidecar_matrix = sidecar_root / "matrix.tsv"
if sidecar_matrix.is_file():
    with open(sidecar_matrix, newline="", encoding="utf-8") as f:
        sidecar_rows = {r["job_id"]: r for r in csv.DictReader(f, delimiter="\t")}

def split(s):
    return [x for x in s.split() if x]

def sidecar_info(sidecar_job_id, default_dir, default_sparse_rel=""):
    r = sidecar_rows.get(sidecar_job_id, {})
    sidecar_dir = r.get("sidecar_dir") or str(default_dir)
    sparse_subdir = r.get("sparse_depth_subdir") or ""
    sparse_rel = f"train/{sparse_subdir}" if sparse_subdir else default_sparse_rel
    return str(Path(sidecar_dir) / "keyframe_channel.npz"), sparse_rel

rows = []

def add(job_id, priority, dataset, variant, scene, scene_root, sidecar_job_id, default_sidecar_dir, sparse_rel, best_metric, run_subroot, note):
    sidecar_path, sparse_depth_rel = sidecar_info(sidecar_job_id, default_sidecar_dir, sparse_rel)
    rows.append({
        "job_id": job_id,
        "priority": str(priority),
        "dataset": dataset,
        "variant": variant,
        "scene": scene,
        "scene_root": scene_root,
        "sidecar_job_id": sidecar_job_id,
        "sidecar_path": sidecar_path,
        "sparse_depth_rel": sparse_depth_rel,
        "best_metric": best_metric,
        "run_subroot": str(run_root / run_subroot),
        "status": "pending",
        "claimed_by": "",
        "started_at": "",
        "finished_at": "",
        "exit_code": "",
        "log_file": "",
        "note": note,
    })

for scene in split(indoor_s):
    add(
        f"indoor6_dino_stage1_{scene}", 10, "indoor6", "dino_ma_stage1", scene,
        str(Path(indoor_root) / scene),
        f"indoor6_dino_{scene}",
        sidecar_root / "sidecars" / "indoor6" / "dino" / scene / "train" / f"colmap_keyframe_channel_v1_sp_strict_dino_r{dino_res}_wvar",
        "", "pct25_5", f"indoor6/dino_ma_stage1/{scene}",
        "Indoor6 DINOv2 + MapAnything memory, Stage1 only",
    )

for scene in split(indoor_s):
    add(
        f"indoor6_acefcn_stage12_{scene}", 20, "indoor6", "acefcn_stage12", scene,
        str(Path(indoor_root) / scene),
        f"indoor6_acefcn_{scene}",
        sidecar_root / "sidecars" / "indoor6" / "acefcn" / scene / "train" / f"colmap_keyframe_channel_v1_sp_strict_acefcn_r{ace_res}",
        "", "pct25_5", f"indoor6/acefcn_stage12/{scene}",
        "Indoor6 ACE-FCN memory Stage1 + GLACE concat Stage2",
    )

for scene in split(cambridge_s):
    add(
        f"cambridge_acefcn_stage12_{scene}", 30, "cambridge", "acefcn_stage12", scene,
        str(Path(cambridge_root) / scene),
        f"cambridge_acefcn_{scene}",
        sidecar_root / "sidecars" / "cambridge" / "acefcn" / scene / "train" / f"colmap_keyframe_channel_v1_sp_strict_acefcn_r{ace_res}",
        "train/sparse_depth", "median_error", f"cambridge/acefcn_stage12/{scene}",
        "Cambridge ACE-FCN memory Stage1 + GLACE concat Stage2; median error primary",
    )

for priority, scenes, hard_note in [(40, split(wayspots_normal_s), "normal Wayspots scene"), (90, split(wayspots_hard_s), "hard DSAC scene; intentionally queued last")]:
    for scene in scenes:
        add(
            f"wayspots_acefcn_stage12_{scene}", priority, "wayspots", "acefcn_stage12", scene,
            str(Path(wayspots_root) / scene),
            f"wayspots_acefcn_{scene}",
            sidecar_root / "sidecars" / "wayspots" / "acefcn" / scene / "train" / f"colmap_keyframe_channel_v1_sp_strict_acefcn_r{ace_res}",
            "train/sparse_depth_superpoint_strict_nms_r4", "pct50_5", f"wayspots/acefcn_stage12/{scene}",
            hard_note,
        )

fieldnames = [
    "job_id", "priority", "dataset", "variant", "scene", "scene_root", "sidecar_job_id",
    "sidecar_path", "sparse_depth_rel", "best_metric", "run_subroot", "status", "claimed_by",
    "started_at", "finished_at", "exit_code", "log_file", "note",
]
Path(matrix_path).parent.mkdir(parents=True, exist_ok=True)
with open(matrix_path, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t", lineterminator="\n")
    w.writeheader(); w.writerows(rows)
print(f"wrote {len(rows)} jobs to {matrix_path}")
PYMATRIX
}

check_data_space() {
  local free_gb
  free_gb=$(df -Pk /data | awk 'NR==2 {printf "%.0f", $4/1024/1024}')
  log "/data free: ${free_gb} GB; threshold=${MIN_DATA_FREE_GB} GB"
  if [[ "${free_gb}" -lt "${MIN_DATA_FREE_GB}" ]]; then
    if bool_true "${ENFORCE_MIN_DATA_FREE_GB}"; then
      echo "ERROR: /data free space below threshold." >&2
      exit 2
    fi
    log "WARN: /data space is low; do not launch full training unless space is cleaned."
  fi
}

preflight() {
  write_matrix
  write_manifest
  check_data_space
  "${PYTHON_ABS}" - "${MATRIX_FILE}" "${SIDECAR_REQUIRED}" \
    "${REPO_ROOT}" "${ACE_ENCODER_PATH}" "${GLACE_ROOT}" <<'PYPREFLIGHT'
import csv, json, sys
from pathlib import Path
matrix, sidecar_required, repo_root, ace_encoder, glace_root = sys.argv[1:6]
sidecar_required = sidecar_required.lower() in {"1", "true", "yes", "y", "on"}
errors, warnings = [], []
for path, label in [
    (Path(repo_root) / "scripts" / "launch_indoor6_dino_stgs_short_gpu01.sh", "Indoor6 DINO launcher"),
    (Path(repo_root) / "scripts" / "run_indoor6_ace_fcn_glace_lmc_all_dual_gpu.sh", "Indoor6 ACE-FCN launcher"),
    (Path(repo_root) / "scripts" / "run_wayspots_ace_fcn_lmc_suite.sh", "Wayspots launcher"),
    (Path(repo_root) / "scripts" / "launch_cambridge_single_stage12_gpu23.sh", "Cambridge launcher"),
    (Path(ace_encoder), "ACE encoder"),
    (Path(glace_root), "GLACE root"),
]:
    if not path.exists():
        errors.append(f"missing {label}: {path}")

with open(matrix, newline="", encoding="utf-8") as f:
    rows = list(csv.DictReader(f, delimiter="\t"))
for r in rows:
    scene_root = Path(r["scene_root"])
    if not (scene_root / "train" / "rgb").is_dir():
        errors.append(f"{r['job_id']}: missing train/rgb under {scene_root}")
    if r["variant"] == "acefcn_stage12" and not (scene_root / "test" / "rgb").is_dir():
        warnings.append(f"{r['job_id']}: missing test/rgb under {scene_root}")
    sidecar = Path(r["sidecar_path"])
    summary = sidecar.with_name("summary.json")
    if not sidecar.is_file():
        msg = f"{r['job_id']}: missing sidecar {sidecar}"
        (errors if sidecar_required else warnings).append(msg)
    elif summary.is_file():
        try:
            rows_n = int(json.loads(summary.read_text()).get("rows", 0))
            if rows_n <= 0:
                errors.append(f"{r['job_id']}: sidecar rows <= 0: {summary}")
        except Exception as exc:
            warnings.append(f"{r['job_id']}: cannot parse summary {summary}: {exc}")
    else:
        warnings.append(f"{r['job_id']}: sidecar summary missing: {summary}")

for w in warnings:
    print("WARN", w, file=sys.stderr)
if errors:
    for e in errors:
        print("ERROR", e, file=sys.stderr)
    raise SystemExit(2)
print(f"preflight ok: {len(rows)} jobs")
PYPREFLIGHT
}

claim_next_job() {
  "${PYTHON_ABS}" - "${MATRIX_FILE}" "${LOCK_FILE}" "${WORKER_GPU}" <<'PYCLAIM'
import csv, datetime, fcntl, os, sys
from pathlib import Path
matrix_path = Path(sys.argv[1]); lock_path = Path(sys.argv[2]); gpu = sys.argv[3]
lock_path.parent.mkdir(parents=True, exist_ok=True)
with open(lock_path, "w", encoding="utf-8") as lock:
    fcntl.flock(lock, fcntl.LOCK_EX)
    with open(matrix_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        rows = list(reader)
        fieldnames = reader.fieldnames or []
    pending = [r for r in rows if r.get("status") == "pending"]
    if not pending:
        print("NO_JOB=1")
        raise SystemExit(0)
    pending.sort(key=lambda r: (int(r["priority"]), r["job_id"]))
    job = pending[0]
    now = datetime.datetime.now().isoformat(timespec="seconds")
    for r in rows:
        if r["job_id"] == job["job_id"]:
            r["status"] = "running"
            r["claimed_by"] = f"gpu{gpu}:pid{os.getpid()}"
            r["started_at"] = now
            r["finished_at"] = ""
            r["exit_code"] = ""
            break
    tmp = matrix_path.with_suffix(".tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t", lineterminator="\n")
        w.writeheader(); w.writerows(rows)
    tmp.replace(matrix_path)
    for k, v in job.items():
        print(f"{k}={str(v)!r}")
    print("NO_JOB=0")
PYCLAIM
}

update_job() {
  local job_id="$1" status="$2" exit_code="$3" log_file="$4"
  "${PYTHON_ABS}" - "${MATRIX_FILE}" "${LOCK_FILE}" "${job_id}" "${status}" "${exit_code}" "${log_file}" <<'PYUPDATE'
import csv, datetime, fcntl, sys
from pathlib import Path
matrix_path = Path(sys.argv[1]); lock_path = Path(sys.argv[2])
job_id, status, exit_code, log_file = sys.argv[3:7]
with open(lock_path, "w", encoding="utf-8") as lock:
    fcntl.flock(lock, fcntl.LOCK_EX)
    with open(matrix_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        rows = list(reader)
        fieldnames = reader.fieldnames or []
    now = datetime.datetime.now().isoformat(timespec="seconds")
    for r in rows:
        if r["job_id"] == job_id:
            r["status"] = status
            r["finished_at"] = now
            r["exit_code"] = exit_code
            r["log_file"] = log_file
            break
    tmp = matrix_path.with_suffix(".tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t", lineterminator="\n")
        w.writeheader(); w.writerows(rows)
    tmp.replace(matrix_path)
PYUPDATE
}

acefcn_extra_args() {
  local best_metric="$1" sidecar_path="$2" scene_root="$3"
  printf '%s ' \
    --post_train_eval_scene "${scene_root}" \
    --best_metric "${best_metric}" \
    --lmc_fusion_geometry_mode "${LMC_FUSION_GEOMETRY_MODE}" \
    --lmc_auto_mode_by_visibility "${LMC_AUTO_MODE_BY_VISIBILITY}" \
    --use_sfm_track_guided_sampling True \
    --use_sfm_track_inter_frame_loss True \
    --sfm_track_keyframe_channel_path "${sidecar_path}" \
    --sfm_track_sidecar_mismatch_policy strict \
    --sfm_track_guided_batch_size "${SFM_TRACK_GUIDED_BATCH_SIZE}" \
    --sfm_track_guided_sampling_strategy balanced_replace \
    --sfm_track_guided_fraction "${SFM_TRACK_GUIDED_FRACTION}" \
    --sfm_track_guided_mode inter_frame \
    --sfm_track_guided_main_loss_mode include \
    --sfm_track_guided_source_target_mode patch_center \
    --sfm_track_guided_aux_normalizer full_batch \
    --sfm_track_anchor_self_weight "${SFM_TRACK_ANCHOR_SELF_WEIGHT}" \
    --sfm_track_anchor_use_alignment_weight True \
    --sfm_track_inter_frame_target_mode exact_target \
    --sfm_track_inter_frame_apply_to stage2_g \
    --sfm_track_inter_frame_weight "${SFM_TRACK_INTER_FRAME_WEIGHT}" \
    --sfm_track_inter_frame_dropout "${SFM_TRACK_INTER_FRAME_DROPOUT}" \
    --sfm_track_inter_frame_start_ratio "${SFM_TRACK_INTER_FRAME_START_RATIO}" \
    --sfm_track_inter_frame_decay_last_ratio "${SFM_TRACK_INTER_FRAME_DECAY_LAST_RATIO}" \
    --sfm_track_inter_frame_max_px "${SFM_TRACK_INTER_FRAME_MAX_PX}" \
    --sfm_track_max_anchor_patch_offset_px 2.0 \
    --sfm_track_max_target_patch_offset_px 0.0 \
    --sfm_track_min_alignment_weight 0.0 \
    --sfm_track_max_colmap_reproj_error_px 2.0 \
    --sfm_track_min_track_length 4
}

indoor6_gpu_scene_env() {
  local gpu="$1" scene="$2"
  case "${gpu}" in
    0) printf 'SCENES_GPU0_STR=%s SCENES_GPU1_STR= SCENES_GPU2_STR= SCENES_GPU3_STR=' "${scene}" ;;
    1) printf 'SCENES_GPU0_STR= SCENES_GPU1_STR=%s SCENES_GPU2_STR= SCENES_GPU3_STR=' "${scene}" ;;
    2) printf 'SCENES_GPU0_STR= SCENES_GPU1_STR= SCENES_GPU2_STR=%s SCENES_GPU3_STR=' "${scene}" ;;
    3) printf 'SCENES_GPU0_STR= SCENES_GPU1_STR= SCENES_GPU2_STR= SCENES_GPU3_STR=%s' "${scene}" ;;
    *) echo "ERROR: Indoor6 ACE-FCN launcher supports physical GPU 0/1/2/3 only, got ${gpu}" >&2; return 2 ;;
  esac
}

run_job() {
  local job_id="$1" dataset="$2" variant="$3" scene="$4" scene_root="$5" sidecar_path="$6" sparse_depth_rel="$7" best_metric="$8" run_subroot="$9"
  local log_file="${LOG_DIR}/${job_id}_gpu${WORKER_GPU}.log"
  mkdir -p "${run_subroot}" "$(dirname "${log_file}")"
  log "start ${job_id} on gpu${WORKER_GPU}; log=${log_file}"

  if [[ ! -s "${sidecar_path}" ]]; then
    echo "ERROR: missing sidecar for ${job_id}: ${sidecar_path}" | tee -a "${log_file}"
    return 2
  fi

  if bool_true "${DRY_RUN}"; then
    {
      echo "DRY_RUN job_id=${job_id}"
      echo "dataset=${dataset} variant=${variant} scene=${scene} gpu=${WORKER_GPU}"
      echo "scene_root=${scene_root}"
      echo "sidecar_path=${sidecar_path}"
      echo "sparse_depth_rel=${sparse_depth_rel}"
      echo "run_subroot=${run_subroot}"
    } | tee -a "${log_file}"
    return 0
  fi

  case "${variant}" in
    dino_ma_stage1)
      env \
        ACE_DATA_ROOT="/data/xwh" \
        ACE_ROOT="${INDOOR6_ACE_ROOT}" \
        WAI_ROOT="${INDOOR6_WAI_ROOT}" \
        COLMAP_ROOT="${INDOOR6_COLMAP_ROOT}" \
        PYTHON_BIN="${PYTHON_ABS}" \
        SCENES_STR="${scene}" \
        GPUS_STR="${WORKER_GPU}" \
        RUN_ROOT="${run_subroot}" \
        SIDECAR_ROOT="${SIDECAR_RUN_ROOT}/sidecars/indoor6/dino" \
        VARIANT_LABEL="${DINO_VARIANT_LABEL}" \
        TRAIN_BATCH_SIZE="${DINO_BATCH_SIZE}" \
        LMC_ITERATIONS="${LMC_ITERATIONS}" \
        TRAINING_BUFFER_SIZE="${TRAINING_BUFFER_SIZE}" \
        BUFFER_SIZE_FINAL="${BUFFER_SIZE_FINAL}" \
        IMAGE_RESOLUTION="${IMAGE_RESOLUTION_DINO}" \
        NUM_LATENT_TOKENS="${NUM_LATENT_TOKENS}" \
        BEST_METRIC="${best_metric}" \
        POST_TRAIN_SEEDS_STR="${POST_TRAIN_EVAL_SEEDS_STR}" \
        POST_TRAIN_HYPOTHESES="${POST_TRAIN_HYPOTHESES}" \
        USE_STGS_GUIDED_SAMPLING=True \
        USE_STGS_INTER_FRAME_LOSS=True \
        SFM_TRACK_GUIDED_MODE=inter_frame \
        SFM_TRACK_GUIDED_FRACTION="${SFM_TRACK_GUIDED_FRACTION}" \
        SFM_TRACK_GUIDED_BATCH_SIZE="${SFM_TRACK_GUIDED_BATCH_SIZE}" \
        SFM_TRACK_INTER_FRAME_APPLY_TO=stage2_g \
        SFM_TRACK_INTER_FRAME_WEIGHT="${SFM_TRACK_INTER_FRAME_WEIGHT}" \
        SFM_TRACK_INTER_FRAME_DROPOUT="${SFM_TRACK_INTER_FRAME_DROPOUT}" \
        SFM_TRACK_INTER_FRAME_START_RATIO="${SFM_TRACK_INTER_FRAME_START_RATIO}" \
        SFM_TRACK_INTER_FRAME_DECAY_LAST_RATIO="${SFM_TRACK_INTER_FRAME_DECAY_LAST_RATIO}" \
        SFM_TRACK_INTER_FRAME_MAX_PX="${SFM_TRACK_INTER_FRAME_MAX_PX}" \
        SFM_TRACK_ANCHOR_SELF_WEIGHT="${SFM_TRACK_ANCHOR_SELF_WEIGHT}" \
        LMC_AUTO_MODE_BY_VISIBILITY="${LMC_AUTO_MODE_BY_VISIBILITY}" \
        LMC_FUSION_GEOMETRY_MODE="${LMC_FUSION_GEOMETRY_MODE}" \
        REBUILD_SIDECAR=False \
        DRY_RUN=False \
        bash "${REPO_ROOT}/scripts/launch_indoor6_dino_stgs_short_gpu01.sh" > "${log_file}" 2>&1
      ;;
    acefcn_stage12)
      local extra_args
      extra_args="$(acefcn_extra_args "${best_metric}" "${sidecar_path}" "${scene_root}")"
      case "${dataset}" in
        indoor6)
          local scene_env
          scene_env="$(indoor6_gpu_scene_env "${WORKER_GPU}" "${scene}")"
          # shellcheck disable=SC2086
          env \
            ${scene_env} \
            RUN_ROOT="${run_subroot}" \
            ACE_ROOT="${INDOOR6_ACE_ROOT}" \
            WAI_ROOT="${INDOOR6_WAI_ROOT}" \
            GLACE_ROOT="${GLACE_ROOT}" \
            ACE_ENCODER_PATH="${ACE_ENCODER_PATH}" \
            METHODS="features memory stage1 stage2" \
            PYTHON_BIN=python \
            CONDA_ENV="${CONDA_ENV}" \
            DRY_RUN=False \
            SKIP_EXISTING="${SKIP_EXISTING}" \
            CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR}" \
            LMC_ITERATIONS="${LMC_ITERATIONS}" \
            NUM_LATENT_TOKENS="${NUM_LATENT_TOKENS}" \
            IMAGE_RESOLUTION="${IMAGE_RESOLUTION_ACEFCN}" \
            MEMORY_IMAGE_RESOLUTION="${IMAGE_RESOLUTION_ACEFCN}" \
            BATCH_SIZE="${ACEFCN_BATCH_SIZE}" \
            TRAINING_BUFFER_SIZE="${TRAINING_BUFFER_SIZE}" \
            BUFFER_SIZE_FINAL="${BUFFER_SIZE_FINAL}" \
            SAMPLES_PER_IMAGE="${SAMPLES_PER_IMAGE}" \
            POST_TRAIN_HYPOTHESES="${POST_TRAIN_HYPOTHESES}" \
            POST_TRAIN_EVAL_SEEDS="${POST_TRAIN_EVAL_SEEDS_STR}" \
            STAGE1_SUBDIR="${STAGE1_SUBDIR}" \
            STAGE2_SUBDIR="${STAGE2_SUBDIR}" \
            LMC_FUSION_GEOMETRY_MODE="${LMC_FUSION_GEOMETRY_MODE}" \
            EXTRA_TRAIN_ARGS="${extra_args}" \
            bash "${REPO_ROOT}/scripts/run_indoor6_ace_fcn_glace_lmc_all_dual_gpu.sh" > "${log_file}" 2>&1
          ;;
        wayspots)
          env \
            RUN_ROOT="${run_subroot}" \
            WAYSPOTS_ROOT="${WAYSPOTS_ROOT}" \
            GLACE_ROOT="${GLACE_ROOT}" \
            ACE_ENCODER_PATH="${ACE_ENCODER_PATH}" \
            SCENES="${scene}" \
            METHODS="memory stage1 stage2" \
            GPU_0="${WORKER_GPU}" \
            GPU_1="${WORKER_GPU}" \
            PYTHON_BIN=python \
            CONDA_ENV="${CONDA_ENV}" \
            DRY_RUN=False \
            SKIP_EXISTING="${SKIP_EXISTING}" \
            CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR}" \
            COORD_SOURCE=sparse_depth \
            DEPTH_REL_DIR="${sparse_depth_rel}" \
            LMC_ITERATIONS="${LMC_ITERATIONS}" \
            NUM_LATENT_TOKENS="${NUM_LATENT_TOKENS}" \
            IMAGE_RESOLUTION="${IMAGE_RESOLUTION_ACEFCN}" \
            MEMORY_IMAGE_RESOLUTION="${IMAGE_RESOLUTION_ACEFCN}" \
            BATCH_SIZE="${ACEFCN_BATCH_SIZE}" \
            TRAINING_BUFFER_SIZE="${TRAINING_BUFFER_SIZE}" \
            BUFFER_SIZE_FINAL="${BUFFER_SIZE_FINAL}" \
            SAMPLES_PER_IMAGE="${SAMPLES_PER_IMAGE}" \
            POST_TRAIN_HYPOTHESES="${POST_TRAIN_HYPOTHESES}" \
            POST_TRAIN_EVAL_SEEDS="${POST_TRAIN_EVAL_SEEDS_STR}" \
            STAGE1_SUBDIR="${STAGE1_SUBDIR}" \
            STAGE2_SUBDIR="${STAGE2_SUBDIR}" \
            LMC_FUSION_GEOMETRY_MODE="${LMC_FUSION_GEOMETRY_MODE}" \
            EXTRA_TRAIN_ARGS="${extra_args}" \
            bash "${REPO_ROOT}/scripts/run_wayspots_ace_fcn_lmc_suite.sh" > "${log_file}" 2>&1
          ;;
        cambridge)
          env \
            RUN_ROOT="${run_subroot}" \
            CAMBRIDGE_ROOT="${CAMBRIDGE_ROOT}" \
            GLACE_ROOT="${GLACE_ROOT}" \
            ACE_ENCODER_PATH="${ACE_ENCODER_PATH}" \
            SCENES="${scene}" \
            METHODS="memory stage1 stage2" \
            GPU_0="${WORKER_GPU}" \
            GPU_1="${WORKER_GPU}" \
            PYTHON_BIN=python \
            CONDA_ENV="${CONDA_ENV}" \
            DRY_RUN=False \
            SKIP_EXISTING="${SKIP_EXISTING}" \
            CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR}" \
            COORD_SOURCE=sparse_depth \
            DEPTH_REL_DIR="${sparse_depth_rel}" \
            LMC_ITERATIONS="${LMC_ITERATIONS}" \
            NUM_LATENT_TOKENS="${NUM_LATENT_TOKENS}" \
            IMAGE_RESOLUTION="${IMAGE_RESOLUTION_ACEFCN}" \
            MEMORY_IMAGE_RESOLUTION="${IMAGE_RESOLUTION_ACEFCN}" \
            BATCH_SIZE="${ACEFCN_BATCH_SIZE}" \
            TRAINING_BUFFER_SIZE="${TRAINING_BUFFER_SIZE}" \
            BUFFER_SIZE_FINAL="${BUFFER_SIZE_FINAL}" \
            SAMPLES_PER_IMAGE="${SAMPLES_PER_IMAGE}" \
            POST_TRAIN_HYPOTHESES="${POST_TRAIN_HYPOTHESES}" \
            POST_TRAIN_EVAL_SEEDS="${POST_TRAIN_EVAL_SEEDS_STR}" \
            STAGE1_SUBDIR="${STAGE1_SUBDIR}" \
            STAGE2_SUBDIR="${STAGE2_SUBDIR}" \
            EXTRA_TRAIN_ARGS="${extra_args}" \
            bash "${REPO_ROOT}/scripts/launch_cambridge_single_stage12_gpu23.sh" > "${log_file}" 2>&1
          ;;
        *)
          echo "ERROR: unsupported acefcn dataset=${dataset}" | tee -a "${log_file}"
          return 2
          ;;
      esac
      ;;
    *)
      echo "ERROR: unsupported variant=${variant}" | tee -a "${log_file}"
      return 2
      ;;
  esac
}

worker_loop() {
  if [[ -z "${WORKER_GPU}" ]]; then
    echo "ERROR: ACTION=worker requires WORKER_GPU" >&2
    exit 2
  fi
  write_matrix
  log "worker gpu${WORKER_GPU} started"
  while true; do
    local claim no_job
    claim="$(claim_next_job)"
    eval "${claim}"
    no_job="${NO_JOB:-0}"
    if [[ "${no_job}" == "1" ]]; then
      log "worker gpu${WORKER_GPU}: no pending jobs"
      break
    fi
    local code=0 status="done" log_file="${LOG_DIR}/${job_id}_gpu${WORKER_GPU}.log"
    set +e
    run_job "${job_id}" "${dataset}" "${variant}" "${scene}" "${scene_root}" "${sidecar_path}" "${sparse_depth_rel}" "${best_metric}" "${run_subroot}"
    code=$?
    set -e
    if [[ "${code}" -ne 0 ]]; then
      status="failed"
      log "failed ${job_id} code=${code}; log=${log_file}"
    else
      log "done ${job_id}; log=${log_file}"
    fi
    update_job "${job_id}" "${status}" "${code}" "${log_file}"
  done
}

launch_tmux() {
  preflight
  bash "${REPO_ROOT}/scripts/agent_safe_snapshot.sh" run_provenance "final_full_matrix_${STAMP}" "${RUN_ROOT}" >/dev/null 2>&1 || true
  if tmux has-session -t "${SESSION}" 2>/dev/null; then
    echo "ERROR: tmux session already exists: ${SESSION}" >&2
    exit 2
  fi
  local first=1 gpu
  for gpu in ${GPUS_STR}; do
    local cmd="cd ${ROOT_DIR}; ACTION=worker WORKER_GPU=${gpu} RUN_ROOT='${RUN_ROOT}' SIDECAR_RUN_ROOT='${SIDECAR_RUN_ROOT}' bash ${REPO_ROOT}/scripts/launch_final_full_matrix_20260705.sh 2>&1 | tee -a '${LOG_DIR}/worker_gpu${gpu}.log'"
    if [[ "${first}" -eq 1 ]]; then
      tmux new-session -d -s "${SESSION}" -n "gpu${gpu}" "${cmd}"
      first=0
    else
      tmux new-window -t "${SESSION}" -n "gpu${gpu}" "${cmd}"
    fi
  done
  log "launched tmux session ${SESSION}; run_root=${RUN_ROOT}"
}

aggregate_results() {
  bash "${REPO_ROOT}/scripts/aggregate_lmc_metricwise_best.sh" "${RUN_ROOT}" | tee "${RUN_ROOT}/metricwise_best.aggregate.log" || true
}

print_status() {
  if [[ -s "${MATRIX_FILE}" ]]; then
    "${PYTHON_ABS}" - "${MATRIX_FILE}" <<'PYSTATUS'
import csv, sys, collections
with open(sys.argv[1], newline="", encoding="utf-8") as f:
    rows = list(csv.DictReader(f, delimiter="\t"))
print("status counts:")
for k, v in collections.Counter(r["status"] for r in rows).items():
    print(f"  {k}: {v}")
print("running:")
for r in rows:
    if r["status"] == "running":
        print(f"  {r['job_id']} {r['claimed_by']} {r['started_at']}")
PYSTATUS
  else
    echo "matrix not found: ${MATRIX_FILE}"
  fi
}

case "${ACTION}" in
  preflight)
    preflight
    ;;
  launch)
    launch_tmux
    ;;
  worker)
    worker_loop
    ;;
  aggregate)
    aggregate_results
    ;;
  status)
    print_status
    ;;
  *)
    echo "ERROR: unknown ACTION=${ACTION}" >&2
    exit 2
    ;;
esac
