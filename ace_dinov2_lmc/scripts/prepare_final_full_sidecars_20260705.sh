#!/usr/bin/env bash
set -euo pipefail

# Prepare STGS keyframe_channel sidecars for the final full experiment.
#
# Usage from /home/xwh/project/ace_depth:
#   ACTION=preflight bash ace_dinov2_lmc/scripts/prepare_final_full_sidecars_20260705.sh
#   ACTION=launch    bash ace_dinov2_lmc/scripts/prepare_final_full_sidecars_20260705.sh
#   ACTION=worker WORKER_GPU=2 RUN_ROOT=<same-root> bash ace_dinov2_lmc/scripts/prepare_final_full_sidecars_20260705.sh
#
# The queue is dynamic: extra workers can be attached later without interrupting
# currently running workers.

ROOT_DIR="${ROOT_DIR:-/home/xwh/project/ace_depth}"
REPO_ROOT="${REPO_ROOT:-${ROOT_DIR}/ace_dinov2_lmc}"
cd "${ROOT_DIR}"

ACTION="${ACTION:-preflight}"  # preflight | launch | worker | inventory
WORKER_GPU="${WORKER_GPU:-}"
CONDA_ENV="${CONDA_ENV:-mapanything}"
PYTHON_BIN="${PYTHON_BIN:-python}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-/data/xwh/ace_dinov2_lmc/04_evaluation/shared/memory/stgs_sidecars/${STAMP}_final_all_value_raw_auto_ifw005_gpu01_expandable}"
SESSION="${SESSION:-final_sidecars_${STAMP}_gpu01}"
GPUS_STR="${GPUS_STR:-0 1}"
REWRITE_MATRIX="${REWRITE_MATRIX:-false}"
REBUILD_SIDECAR="${REBUILD_SIDECAR:-false}"
REBUILD_WORKSPACE="${REBUILD_WORKSPACE:-false}"
DRY_RUN="${DRY_RUN:-false}"

LOG_DIR="${RUN_ROOT}/logs"
MATRIX_FILE="${RUN_ROOT}/matrix.tsv"
LOCK_FILE="${RUN_ROOT}/queue.lock"
INVENTORY_FILE="${RUN_ROOT}/sidecar_inventory.tsv"
MANIFEST_FILE="${RUN_ROOT}/manifest.json"

SUPERPOINT_WEIGHTS="${SUPERPOINT_WEIGHTS:-${REPO_ROOT}/superpoint_v1.pth}"
SUPERPOINT_PYTHONPATH="${SUPERPOINT_PYTHONPATH:-/data/xwh/SuperPointPretrainedNetwork}"

WAYSPOTS_ROOT="${WAYSPOTS_ROOT:-/data/xwh/Wayspots}"
CAMBRIDGE_ROOT="${CAMBRIDGE_ROOT:-/data/xwh/Cambridge}"
INDOOR6_ACE_ROOT="${INDOOR6_ACE_ROOT:-/data/xwh/indoor6_ace}"
INDOOR6_WAI_ROOT="${INDOOR6_WAI_ROOT:-/data/xwh/mapanything-dataset/wai_data/indoor6}"
INDOOR6_COLMAP_ROOT="${INDOOR6_COLMAP_ROOT:-/data/xwh/indoor6/indoor6-colmap}"

ACEFCN_IMAGE_RESOLUTION="${ACEFCN_IMAGE_RESOLUTION:-512}"
DINO_IMAGE_RESOLUTION="${DINO_IMAGE_RESOLUTION:-518}"
DINO_IMAGE_WIDTH="${DINO_IMAGE_WIDTH:-none}"

WAYSPOTS_SCENES_NORMAL="${WAYSPOTS_SCENES_NORMAL:-wayspots_bears wayspots_cubes wayspots_inscription wayspots_lawn wayspots_map wayspots_squarebench wayspots_tendrils wayspots_therock}"
WAYSPOTS_SCENES_HARD="${WAYSPOTS_SCENES_HARD:-wayspots_statue wayspots_wintersign}"
CAMBRIDGE_SCENES="${CAMBRIDGE_SCENES:-Cambridge_GreatCourt Cambridge_KingsCollege Cambridge_OldHospital Cambridge_ShopFacade Cambridge_StMarysChurch}"
INDOOR6_SCENES="${INDOOR6_SCENES:-scene1 scene2a scene3 scene4a scene5 scene6}"

mkdir -p "${RUN_ROOT}" "${LOG_DIR}" "${RUN_ROOT}/sidecars" "${RUN_ROOT}/workspaces" "${RUN_ROOT}/summaries" "${RUN_ROOT}/scripts"

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

bool_true() {
  case "${1,,}" in
    1|true|yes|y|on) return 0 ;;
    *) return 1 ;;
  esac
}

write_manifest() {
  python - "${MANIFEST_FILE}" "${RUN_ROOT}" "${GPUS_STR}" <<'PYMANIFEST'
import json, sys, datetime
path, run_root, gpus = sys.argv[1:4]
manifest = {
    "dataset": "shared",
    "track": "memory",
    "method": "stgs_sidecars",
    "scope": "indoor6_wayspots_cambridge_all",
    "protocol": "final_value_raw_auto_ifw005_sidecars",
    "gpu_tag": "expandable",
    "status": "planned",
    "created_at": datetime.datetime.now().isoformat(timespec="seconds"),
    "run_root": run_root,
    "gpus_initial": gpus.split(),
    "notes": [
        "Wayspots statue/wintersign are listed last because DSAC/eval is slow.",
        "Sidecars are built before final training; rows must be > 0.",
    ],
}
with open(path, "w", encoding="utf-8") as f:
    json.dump(manifest, f, ensure_ascii=False, indent=2)
PYMANIFEST
  cp "${REPO_ROOT}/scripts/prepare_final_full_sidecars_20260705.sh" "${RUN_ROOT}/scripts/" 2>/dev/null || true
}

append_job() {
  local job_id="$1" priority="$2" dataset="$3" backend="$4" scene="$5" scene_root="$6" model_dir="$7" workspace="$8" sidecar_dir="$9" sparse_depth_dir="${10}" sparse_depth_subdir="${11}" note="${12}"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\tpending\t\t\t\t\t\t%s\n" \
    "${job_id}" "${priority}" "${dataset}" "${backend}" "${scene}" "${scene_root}" "${model_dir}" "${workspace}" "${sidecar_dir}" "${sparse_depth_dir}" "${sparse_depth_subdir}" "${note}" >> "${MATRIX_FILE}"
}

write_matrix() {
  if [[ -s "${MATRIX_FILE}" && "${REWRITE_MATRIX}" != "true" ]]; then
    log "reuse matrix: ${MATRIX_FILE}"
    return 0
  fi
  printf "job_id\tpriority\tdataset\tbackend\tscene\tscene_root\tmodel_dir\tworkspace\tsidecar_dir\tsparse_depth_dir\tsparse_depth_subdir\tstatus\tclaimed_by\tstarted_at\tfinished_at\texit_code\trows\tnote\n" > "${MATRIX_FILE}"

  local scene scene_root model_dir workspace sidecar_dir sparse_depth_dir sparse_depth_subdir

  for scene in ${INDOOR6_SCENES}; do
    scene_root="${INDOOR6_ACE_ROOT}/${scene}"
    model_dir="${INDOOR6_COLMAP_ROOT}/${scene}-tr/sparse/0"
    sparse_depth_dir=""
    sidecar_dir="${RUN_ROOT}/sidecars/indoor6/dino/${scene}/train/colmap_keyframe_channel_v1_sp_strict_dino_r${DINO_IMAGE_RESOLUTION}_wvar"
    append_job "indoor6_dino_${scene}" 10 "indoor6" "dino" "${scene}" "${scene_root}" "${model_dir}" "" "${sidecar_dir}" "${sparse_depth_dir}" "" "dinov2 variable-width sidecar"
  done

  for scene in ${INDOOR6_SCENES}; do
    scene_root="${INDOOR6_ACE_ROOT}/${scene}"
    model_dir="${INDOOR6_COLMAP_ROOT}/${scene}-tr/sparse/0"
    sparse_depth_dir=""
    sidecar_dir="${RUN_ROOT}/sidecars/indoor6/acefcn/${scene}/train/colmap_keyframe_channel_v1_sp_strict_acefcn_r${ACEFCN_IMAGE_RESOLUTION}"
    append_job "indoor6_acefcn_${scene}" 20 "indoor6" "acefcn_colmap" "${scene}" "${scene_root}" "${model_dir}" "" "${sidecar_dir}" "${sparse_depth_dir}" "" "acefcn sidecar from train-only colmap"
  done

  for scene in ${CAMBRIDGE_SCENES}; do
    scene_root="${CAMBRIDGE_ROOT}/${scene}"
    workspace="${RUN_ROOT}/workspaces/cambridge/${scene}/superpoint_workspace"
    model_dir="${workspace}/triangulated_model"
    sparse_depth_subdir="sparse_depth_superpoint_stgs_final_ifw005_${STAMP}"
    sparse_depth_dir="${scene_root}/train/${sparse_depth_subdir}"
    sidecar_dir="${RUN_ROOT}/sidecars/cambridge/acefcn/${scene}/train/colmap_keyframe_channel_v1_sp_strict_acefcn_r${ACEFCN_IMAGE_RESOLUTION}"
    append_job "cambridge_acefcn_${scene}" 30 "cambridge" "acefcn_superpoint" "${scene}" "${scene_root}" "${model_dir}" "${workspace}" "${sidecar_dir}" "${sparse_depth_dir}" "${sparse_depth_subdir}" "acefcn sidecar via known-pose SuperPoint"
  done

  for scene in ${WAYSPOTS_SCENES_NORMAL}; do
    scene_root="${WAYSPOTS_ROOT}/${scene}"
    workspace="${RUN_ROOT}/workspaces/wayspots/${scene}/superpoint_workspace"
    model_dir="${workspace}/triangulated_model"
    sparse_depth_subdir="sparse_depth_superpoint_stgs_final_ifw005_${STAMP}"
    sparse_depth_dir="${scene_root}/train/${sparse_depth_subdir}"
    sidecar_dir="${RUN_ROOT}/sidecars/wayspots/acefcn/${scene}/train/colmap_keyframe_channel_v1_sp_strict_acefcn_r${ACEFCN_IMAGE_RESOLUTION}"
    append_job "wayspots_acefcn_${scene}" 40 "wayspots" "acefcn_superpoint" "${scene}" "${scene_root}" "${model_dir}" "${workspace}" "${sidecar_dir}" "${sparse_depth_dir}" "${sparse_depth_subdir}" "normal wayspots scene"
  done

  for scene in ${WAYSPOTS_SCENES_HARD}; do
    scene_root="${WAYSPOTS_ROOT}/${scene}"
    workspace="${RUN_ROOT}/workspaces/wayspots/${scene}/superpoint_workspace"
    model_dir="${workspace}/triangulated_model"
    sparse_depth_subdir="sparse_depth_superpoint_stgs_final_ifw005_${STAMP}"
    sparse_depth_dir="${scene_root}/train/${sparse_depth_subdir}"
    sidecar_dir="${RUN_ROOT}/sidecars/wayspots/acefcn/${scene}/train/colmap_keyframe_channel_v1_sp_strict_acefcn_r${ACEFCN_IMAGE_RESOLUTION}"
    append_job "wayspots_acefcn_${scene}" 90 "wayspots" "acefcn_superpoint" "${scene}" "${scene_root}" "${model_dir}" "${workspace}" "${sidecar_dir}" "${sparse_depth_dir}" "${sparse_depth_subdir}" "hard DSAC scene; intentionally last"
  done

  log "wrote matrix: ${MATRIX_FILE}"
}

preflight_inputs() {
  local failed=0
  python - "${MATRIX_FILE}" "${SUPERPOINT_WEIGHTS}" "${SUPERPOINT_PYTHONPATH}" <<'PYPREFLIGHT' || failed=1
import csv, sys
from pathlib import Path
matrix, sp_weights, sp_path = sys.argv[1:4]
errors = []
if not Path(sp_weights).is_file():
    errors.append(f"missing superpoint weights: {sp_weights}")
if not Path(sp_path).is_dir():
    errors.append(f"missing SuperPoint repo: {sp_path}")
with open(matrix, newline="", encoding="utf-8") as f:
    for r in csv.DictReader(f, delimiter="\t"):
        scene_root = Path(r["scene_root"])
        if not (scene_root / "train" / "rgb").is_dir():
            errors.append(f"{r['job_id']}: missing train/rgb under {scene_root}")
        if r["backend"] in {"dino", "acefcn_colmap"}:
            if not (Path(r["model_dir"]) / "points3D.bin").is_file():
                errors.append(f"{r['job_id']}: missing COLMAP points3D.bin under {r['model_dir']}")
        if r["sparse_depth_dir"] and r["backend"] in {"dino", "acefcn_colmap"} and not Path(r["sparse_depth_dir"]).is_dir():
            print(f"WARN {r['job_id']}: sparse depth dir missing: {r['sparse_depth_dir']}", file=sys.stderr)
if errors:
    for e in errors:
        print("ERROR", e, file=sys.stderr)
    raise SystemExit(2)
print("preflight ok")
PYPREFLIGHT
  return "${failed}"
}

claim_next_job() {
  python - "${MATRIX_FILE}" "${LOCK_FILE}" "${WORKER_GPU}" <<'PYCLAIM'
import csv, datetime, fcntl, os, sys
from pathlib import Path
matrix_path = Path(sys.argv[1])
lock_path = Path(sys.argv[2])
gpu = sys.argv[3]
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
            r["exit_code"] = ""
            break
    tmp = matrix_path.with_suffix(".tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t", lineterminator="\n")
        w.writeheader(); w.writerows(rows)
    tmp.replace(matrix_path)
    for k, v in job.items():
        safe = str(v).replace("'", "'\"'\"'")
        print(f"{k}='{safe}'")
    print("NO_JOB=0")
PYCLAIM
}

update_job() {
  local job_id="$1" status="$2" exit_code="$3" rows_value="$4"
  python - "${MATRIX_FILE}" "${LOCK_FILE}" "${job_id}" "${status}" "${exit_code}" "${rows_value}" <<'PYUPDATE'
import csv, datetime, fcntl, sys
from pathlib import Path
matrix_path = Path(sys.argv[1]); lock_path = Path(sys.argv[2])
job_id, status, exit_code, rows_value = sys.argv[3:7]
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
            r["rows"] = rows_value
            break
    tmp = matrix_path.with_suffix(".tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t", lineterminator="\n")
        w.writeheader(); w.writerows(rows)
    tmp.replace(matrix_path)
PYUPDATE
}

sidecar_rows() {
  local sidecar_dir="$1"
  python - "${sidecar_dir}" <<'PYROWS'
import json, sys
from pathlib import Path
p = Path(sys.argv[1]) / "summary.json"
if not p.is_file():
    print("0"); raise SystemExit(0)
try:
    print(int(json.loads(p.read_text()).get("rows", 0) or 0))
except Exception:
    print("0")
PYROWS
}

build_superpoint_workspace_if_needed() {
  local scene_root="$1" workspace="$2" sparse_depth_subdir="$3" gpu="$4" log_file="$5"
  local model_dir="${workspace}/triangulated_model"
  if [[ -f "${model_dir}/points3D.bin" && "${REBUILD_WORKSPACE}" != "true" ]]; then
    log "reuse workspace: ${model_dir}"
    return 0
  fi
  mkdir -p "${workspace}"
  if bool_true "${DRY_RUN}"; then
    log "DRY_RUN build SuperPoint workspace scene=${scene_root} workspace=${workspace}"
    return 0
  fi
  env CUDA_VISIBLE_DEVICES="${gpu}" PYTHONPATH="${SUPERPOINT_PYTHONPATH}:${PYTHONPATH:-}" \
    conda run --no-capture-output -n "${CONDA_ENV}" "${PYTHON_BIN}" ace_dinov2_lmc/tools/wayspots_known_pose_sparse_depth.py \
      "${scene_root}" \
      --split train \
      --workspace "${workspace}" \
      --output-subdir "${sparse_depth_subdir}" \
      --feature-backend superpoint \
      --superpoint-weights "${SUPERPOINT_WEIGHTS}" \
      --superpoint-conf-thresh 0.05 \
      --superpoint-max-keypoints 1024 \
      --superpoint-nms-dist 8 \
      --superpoint-nn-thresh 0.60 \
      --matcher pairs \
      --match-window 20 \
      --num-threads 4 \
      --min-triangulation-angle 3.0 \
      --max-depth-m 1000 \
      --use-gpu \
      --gpu-index 0 \
      --overwrite >> "${log_file}" 2>&1
}

build_acefcn_sidecar() {
  local scene_root="$1" model_dir="$2" sidecar_dir="$3" sparse_depth_dir="$4" gpu="$5" log_file="$6"
  local sidecar="${sidecar_dir}/keyframe_channel.npz"
  if [[ -s "${sidecar}" && "${REBUILD_SIDECAR}" != "true" ]]; then
    log "reuse sidecar: ${sidecar}"
    return 0
  fi
  mkdir -p "${sidecar_dir}"
  if bool_true "${DRY_RUN}"; then
    log "DRY_RUN build ACE-FCN sidecar scene=${scene_root} model=${model_dir} out=${sidecar_dir}"
    return 0
  fi
  local sparse_args=()
  if [[ -n "${sparse_depth_dir}" && -d "${sparse_depth_dir}" ]]; then
    sparse_args=(--sparse-depth-dir "${sparse_depth_dir}")
  fi
  conda run --no-capture-output -n "${CONDA_ENV}" "${PYTHON_BIN}" ace_dinov2_lmc/tools/build_colmap_keyframe_channel.py \
    "${scene_root}" \
    --split train \
    --model-dir "${model_dir}" \
    --output-dir "${sidecar_dir}" \
    --target-backbone ace_fcn \
    --image-resolution "${ACEFCN_IMAGE_RESOLUTION}" \
    --min-track-length 4 \
    --max-reproj-error 2.0 \
    --min-parallax-deg 2.0 \
    --max-parallax-deg 60.0 \
    --max-anchor-alignment-px 2.0 \
    --max-target-alignment-px 0.0 \
    --alignment-sigma-px 2.0 \
    --image-match-mode auto \
    "${sparse_args[@]}" >> "${log_file}" 2>&1
}

build_dino_sidecar() {
  local scene_root="$1" model_dir="$2" sidecar_dir="$3" sparse_depth_dir="$4" gpu="$5" log_file="$6"
  local sidecar="${sidecar_dir}/keyframe_channel.npz"
  if [[ -s "${sidecar}" && "${REBUILD_SIDECAR}" != "true" ]]; then
    log "reuse sidecar: ${sidecar}"
    return 0
  fi
  mkdir -p "${sidecar_dir}"
  if bool_true "${DRY_RUN}"; then
    log "DRY_RUN build DINO sidecar scene=${scene_root} model=${model_dir} out=${sidecar_dir}"
    return 0
  fi
  IMAGE_RESOLUTION="${DINO_IMAGE_RESOLUTION}" \
  IMAGE_WIDTH="${DINO_IMAGE_WIDTH}" \
  SPARSE_DEPTH_DIR="${sparse_depth_dir:-/__ace_lmc_no_sparse_depth_filter__}" \
  MIN_TRACK_LENGTH=4 \
  MAX_REPROJ_ERROR=2.0 \
  MIN_PARALLAX_DEG=2.0 \
  MAX_PARALLAX_DEG=60.0 \
  MAX_ANCHOR_ALIGNMENT_PX=3.5 \
  MAX_TARGET_ALIGNMENT_PX=3.5 \
  IMAGE_MATCH_MODE=auto \
  bash "${REPO_ROOT}/scripts/build_dino_stgs_keyframe_channel.sh" \
    "${scene_root}" "${model_dir}" "${sidecar_dir}" >> "${log_file}" 2>&1
}

run_job() {
  local job_id="$1" dataset="$2" backend="$3" scene="$4" scene_root="$5" model_dir="$6" workspace="$7" sidecar_dir="$8" sparse_depth_dir="$9" sparse_depth_subdir="${10}" gpu="${11}"
  local log_file="${LOG_DIR}/${job_id}_gpu${gpu}.log"
  mkdir -p "$(dirname "${log_file}")"
  {
    printf '[%s] job_id=%s dataset=%s backend=%s scene=%s gpu=%s\n' "$(date)" "${job_id}" "${dataset}" "${backend}" "${scene}" "${gpu}"
    printf 'scene_root=%s\nmodel_dir=%s\nworkspace=%s\nsidecar_dir=%s\nsparse_depth_dir=%s\n' "${scene_root}" "${model_dir}" "${workspace}" "${sidecar_dir}" "${sparse_depth_dir}"
  } >> "${log_file}"

  case "${backend}" in
    acefcn_superpoint)
      build_superpoint_workspace_if_needed "${scene_root}" "${workspace}" "${sparse_depth_subdir}" "${gpu}" "${log_file}"
      build_acefcn_sidecar "${scene_root}" "${model_dir}" "${sidecar_dir}" "${sparse_depth_dir}" "${gpu}" "${log_file}"
      ;;
    acefcn_colmap)
      build_acefcn_sidecar "${scene_root}" "${model_dir}" "${sidecar_dir}" "${sparse_depth_dir}" "${gpu}" "${log_file}"
      ;;
    dino)
      build_dino_sidecar "${scene_root}" "${model_dir}" "${sidecar_dir}" "${sparse_depth_dir}" "${gpu}" "${log_file}"
      ;;
    *)
      echo "ERROR: unknown backend=${backend}" >> "${log_file}"
      return 2
      ;;
  esac

  local rows
  rows="$(sidecar_rows "${sidecar_dir}")"
  if [[ "${rows}" -le 0 && "${DRY_RUN}" != "true" ]]; then
    echo "ERROR: sidecar rows <= 0 for ${job_id}; sidecar_dir=${sidecar_dir}" >> "${log_file}"
    return 3
  fi
  echo "rows=${rows}" >> "${log_file}"
}

write_inventory() {
  python - "${MATRIX_FILE}" "${INVENTORY_FILE}" <<'PYINV'
import csv, json, sys
from pathlib import Path
matrix, out = map(Path, sys.argv[1:3])
fields = [
    "job_id", "priority", "dataset", "backend", "scene", "status", "sidecar_path",
    "summary_path", "rows", "image_name_matches", "image_pose_matches",
    "num_colmap_images_matched", "num_train_frames", "output_subsample",
    "image_resolution", "image_width", "target_backbone", "model_dir", "sparse_depth_dir",
]
with open(matrix, newline="", encoding="utf-8") as f:
    rows = list(csv.DictReader(f, delimiter="\t"))
out.parent.mkdir(parents=True, exist_ok=True)
with open(out, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=fields, delimiter="\t", lineterminator="\n")
    w.writeheader()
    for r in rows:
        sidecar = Path(r["sidecar_dir"]) / "keyframe_channel.npz"
        summary = Path(r["sidecar_dir"]) / "summary.json"
        data = {}
        if summary.is_file():
            try:
                data = json.loads(summary.read_text())
            except Exception as exc:
                data = {"summary_error": str(exc)}
        w.writerow({
            "job_id": r["job_id"],
            "priority": r["priority"],
            "dataset": r["dataset"],
            "backend": r["backend"],
            "scene": r["scene"],
            "status": r["status"],
            "sidecar_path": str(sidecar),
            "summary_path": str(summary),
            "rows": data.get("rows", r.get("rows", "")),
            "image_name_matches": data.get("image_name_matches", ""),
            "image_pose_matches": data.get("image_pose_matches", ""),
            "num_colmap_images_matched": data.get("num_colmap_images_matched", ""),
            "num_train_frames": data.get("num_train_frames", ""),
            "output_subsample": data.get("output_subsample", ""),
            "image_resolution": data.get("image_resolution", ""),
            "image_width": data.get("image_width", ""),
            "target_backbone": data.get("target_backbone", ""),
            "model_dir": data.get("model_dir", r.get("model_dir", "")),
            "sparse_depth_dir": data.get("sparse_depth_dir", r.get("sparse_depth_dir", "")),
        })
print(out)
PYINV
}

run_worker() {
  if [[ -z "${WORKER_GPU}" ]]; then
    echo "ERROR: WORKER_GPU is required for ACTION=worker" >&2
    exit 2
  fi
  log "worker start gpu=${WORKER_GPU} run_root=${RUN_ROOT}"
  while true; do
    local claim
    claim="$(claim_next_job)"
    eval "${claim}"
    if [[ "${NO_JOB}" == "1" ]]; then
      log "worker gpu=${WORKER_GPU}: no pending jobs"
      break
    fi
    log "worker gpu=${WORKER_GPU}: claimed ${job_id} (${dataset}/${backend}/${scene})"
    set +e
    run_job "${job_id}" "${dataset}" "${backend}" "${scene}" "${scene_root}" "${model_dir}" "${workspace}" "${sidecar_dir}" "${sparse_depth_dir}" "${sparse_depth_subdir}" "${WORKER_GPU}"
    local ec=$?
    set -e
    local rows_value="0"
    rows_value="$(sidecar_rows "${sidecar_dir}")"
    if [[ "${ec}" -eq 0 ]]; then
      update_job "${job_id}" "ok" "0" "${rows_value}"
      log "worker gpu=${WORKER_GPU}: ok ${job_id} rows=${rows_value}"
    else
      update_job "${job_id}" "failed" "${ec}" "${rows_value}"
      log "worker gpu=${WORKER_GPU}: failed ${job_id} exit=${ec} rows=${rows_value}"
    fi
    write_inventory >/dev/null || true
  done
  write_inventory >/dev/null || true
}

launch_tmux() {
  local script_path="${REPO_ROOT}/scripts/prepare_final_full_sidecars_20260705.sh"
  read -r -a GPUS <<< "${GPUS_STR}"
  if tmux has-session -t "${SESSION}" 2>/dev/null; then
    echo "ERROR: tmux session already exists: ${SESSION}" >&2
    exit 2
  fi
  local first=1 gpu
  for gpu in "${GPUS[@]}"; do
    local cmd="cd '${ROOT_DIR}' && ACTION=worker WORKER_GPU='${gpu}' RUN_ROOT='${RUN_ROOT}' STAMP='${STAMP}' SESSION='${SESSION}' bash '${script_path}' 2>&1 | tee '${LOG_DIR}/worker_gpu${gpu}.tmux.log'"
    if [[ "${first}" -eq 1 ]]; then
      tmux new-session -d -s "${SESSION}" -n "gpu${gpu}" "${cmd}"
      first=0
    else
      tmux new-window -t "${SESSION}" -n "gpu${gpu}" "${cmd}"
    fi
  done
  log "tmux launched session=${SESSION} run_root=${RUN_ROOT}"
}

case "${ACTION}" in
  preflight)
    write_manifest
    write_matrix
    preflight_inputs
    write_inventory >/dev/null || true
    log "preflight done run_root=${RUN_ROOT}"
    ;;
  launch)
    write_manifest
    write_matrix
    preflight_inputs
    launch_tmux
    ;;
  worker)
    write_manifest
    write_matrix
    run_worker
    ;;
  inventory)
    write_matrix
    write_inventory
    ;;
  *)
    echo "ERROR: unsupported ACTION=${ACTION}; use preflight | launch | worker | inventory" >&2
    exit 2
    ;;
esac
