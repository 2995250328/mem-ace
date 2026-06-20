#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="/home/xwh/project/ace_depth"
PYTHON_ENV="${PYTHON_ENV:-mapanything}"
RUN_ROOT="${RUN_ROOT:-/data/xwh/ace_dinov2_lmc/04_evaluation/pmrf_deterministic_reeval_20260619_gpu0123}"
HYPOTHESES="${HYPOTHESES:-256}"
EVAL_WORKERS="${EVAL_WORKERS:-6}"
OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
OMP_DYNAMIC="${OMP_DYNAMIC:-FALSE}"
SEEDS=(1305 2026 4242)
BEARS_SCENE="/data/xwh/Wayspots/wayspots_bears"
SQ_SCENE="/data/xwh/Wayspots/wayspots_squarebench"
BEARS_SINGLE="/data/xwh/ace_dinov2_lmc/04_evaluation/weak_ffn_control_bears_20260618_0019/single_repeat/stage1_local_ace_memory_it12/extracted/wayspots_bears/dino_ace_lmc_ace_g/20260618_002135_aceg_fS2cie_global_res512_buf2.8M_F7.6M_K64_it12_ep24_bs4096_spi_s1buf_sp512_onecycle_improved/best_K64_it12_ace_fcn_local_stage1.pt"
BEARS_FFN="/data/xwh/ace_dinov2_lmc/04_evaluation/weak_ffn_control_bears_20260618_0019/weak_ffn_alpha010/stage1_local_ace_memory_it12/extracted/wayspots_bears/dino_ace_lmc_ace_g/20260618_023348_aceg_fS2cie_global_res512_buf2.8M_F7.6M_K64_it12_ep24_bs4096_spi_s1buf_sp512_onecycle_improved/best_K64_it12_ace_fcn_local_stage1.pt"
BEARS_PMRF="/data/xwh/ace_dinov2_lmc/04_evaluation/weak_ffn_control_bears_20260618_0019/pmrf_nonorm_alpha010/stage1_local_ace_memory_it12/extracted/wayspots_bears/dino_ace_lmc_ace_g/20260618_002135_aceg_fS2cie_global_res512_buf2.8M_F7.6M_K64_it12_ep24_bs4096_spi_s1buf_sp512_onecycle_improved/best_K64_it12_ace_fcn_local_stage1.pt"
BEARS_TR05="/data/xwh/ace_dinov2_lmc/04_evaluation/pmrf_trust_region_matrix_20260619_gpu02/pmrf_tr05/stage1_local_ace_memory_it12/extracted/wayspots_bears/dino_ace_lmc_ace_g/20260619_112747_aceg_fS2cie_global_res512_buf2.8M_F7.6M_K64_it12_ep24_bs4096_spi_s1buf_sp512_onecycle_improved/best_K64_it12_ace_fcn_local_stage1.pt"
BEARS_TR025="/data/xwh/ace_dinov2_lmc/04_evaluation/pmrf_trust_region_matrix_20260619_gpu02/pmrf_tr025/stage1_local_ace_memory_it12/extracted/wayspots_bears/dino_ace_lmc_ace_g/20260619_132850_aceg_fS2cie_global_res512_buf2.8M_F7.6M_K64_it12_ep24_bs4096_spi_s1buf_sp512_onecycle_improved/best_K64_it12_ace_fcn_local_stage1.pt"
SQ_SINGLE="/data/xwh/ace_dinov2_lmc/04_evaluation/weak_ffn_control_sq_20260618_1105/single_repeat/stage1_local_ace_memory_it12/extracted/wayspots_squarebench/dino_ace_lmc_ace_g/20260618_110527_aceg_fS2cie_global_res512_buf2.8M_F7.6M_K64_it12_ep24_bs4096_spi_s1buf_sp512_onecycle_improved/best_K64_it12_ace_fcn_local_stage1.pt"
SQ_FFN="/data/xwh/ace_dinov2_lmc/04_evaluation/weak_ffn_control_sq_20260618_1105/weak_ffn_alpha010/stage1_local_ace_memory_it12/extracted/wayspots_squarebench/dino_ace_lmc_ace_g/20260618_130604_aceg_fS2cie_global_res512_buf2.8M_F7.6M_K64_it12_ep24_bs4096_spi_s1buf_sp512_onecycle_improved/best_K64_it12_ace_fcn_local_stage1.pt"
SQ_PMRF="/data/xwh/ace_dinov2_lmc/04_evaluation/weak_ffn_control_sq_20260618_1105/pmrf_nonorm_alpha010/stage1_local_ace_memory_it12/extracted/wayspots_squarebench/dino_ace_lmc_ace_g/20260618_110527_aceg_fS2cie_global_res512_buf2.8M_F7.6M_K64_it12_ep24_bs4096_spi_s1buf_sp512_onecycle_improved/best_K64_it12_ace_fcn_local_stage1.pt"
SQ_TR05="/data/xwh/ace_dinov2_lmc/04_evaluation/pmrf_trust_region_matrix_20260619_gpu02/pmrf_tr05/stage1_local_ace_memory_it12/extracted/wayspots_squarebench/dino_ace_lmc_ace_g/20260619_112747_aceg_fS2cie_global_res512_buf2.8M_F7.6M_K64_it12_ep24_bs4096_spi_s1buf_sp512_onecycle_improved/best_K64_it12_ace_fcn_local_stage1.pt"
SQ_TR025="/data/xwh/ace_dinov2_lmc/04_evaluation/pmrf_trust_region_matrix_20260619_gpu02/pmrf_tr025/stage1_local_ace_memory_it12/extracted/wayspots_squarebench/dino_ace_lmc_ace_g/20260619_132850_aceg_fS2cie_global_res512_buf2.8M_F7.6M_K64_it12_ep24_bs4096_spi_s1buf_sp512_onecycle_improved/best_K64_it12_ace_fcn_local_stage1.pt"
mkdir -p "${RUN_ROOT}"
STATUS_FILE="${RUN_ROOT}/status.tsv"
[[ -f "${STATUS_FILE}" ]] || printf 'timestamp\tscene\tmethod\tseed\tstatus\texit_code\tdevice\toutput_dir\n' > "${STATUS_FILE}"
run_job() {
    local scene_name="$1" scene_path="$2" method="$3" checkpoint="$4" device="$5"
    [[ -f "${checkpoint}" ]] || { echo "Missing checkpoint: ${checkpoint}" >&2; return 2; }
    for seed in "${SEEDS[@]}"; do
        local output_dir="${RUN_ROOT}/${scene_name}/${method}/seed${seed}"
        local summary="${output_dir}/eval_summary_${scene_name}_det_seed${seed}.txt"
        local log="${output_dir}/run.log"
        mkdir -p "${output_dir}"
        [[ ! -s "${summary}" ]] || { echo "[skip] ${scene_name} ${method} seed=${seed}"; continue; }
        echo "[eval] ${scene_name} ${method} seed=${seed} device=${device}"
        set +e
        OMP_NUM_THREADS="${OMP_NUM_THREADS}" OMP_DYNAMIC="${OMP_DYNAMIC}" \
        conda run --no-capture-output -n "${PYTHON_ENV}" python ace_dinov2_lmc/test_ace_dinov2_lmc.py \
            "${scene_path}" "${checkpoint}" --device "${device}" --output_dir "${output_dir}" \
            --session "det_seed${seed}" --image_resolution 512 --hypotheses "${HYPOTHESES}" \
            --eval_deterministic True --dsacstar_seed "${seed}" --dsacstar_seed_per_frame True \
            --eval_num_workers "${EVAL_WORKERS}" > "${log}" 2>&1
        local rc=$?
        set -e
        local status=ok
        [[ ${rc} -eq 0 ]] || status=failed
        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$(date --iso-8601=seconds)" \
            "${scene_name}" "${method}" "${seed}" "${status}" "${rc}" "${device}" "${output_dir}" >> "${STATUS_FILE}"
        [[ ${rc} -eq 0 ]] || return "${rc}"
    done
}
worker0() { run_job wayspots_bears "${BEARS_SCENE}" single "${BEARS_SINGLE}" cuda:0; run_job wayspots_bears "${BEARS_SCENE}" pmrf "${BEARS_PMRF}" cuda:0; run_job wayspots_bears "${BEARS_SCENE}" tr025 "${BEARS_TR025}" cuda:0; }
worker1() { run_job wayspots_bears "${BEARS_SCENE}" weak_ffn "${BEARS_FFN}" cuda:1; run_job wayspots_bears "${BEARS_SCENE}" tr05 "${BEARS_TR05}" cuda:1; }
worker2() { run_job wayspots_squarebench "${SQ_SCENE}" single "${SQ_SINGLE}" cuda:2; run_job wayspots_squarebench "${SQ_SCENE}" pmrf "${SQ_PMRF}" cuda:2; run_job wayspots_squarebench "${SQ_SCENE}" tr025 "${SQ_TR025}" cuda:2; }
worker3() { run_job wayspots_squarebench "${SQ_SCENE}" weak_ffn "${SQ_FFN}" cuda:3; run_job wayspots_squarebench "${SQ_SCENE}" tr05 "${SQ_TR05}" cuda:3; }
cd "${PROJECT_ROOT}"
worker0 & p0=$!
worker1 & p1=$!
worker2 & p2=$!
worker3 & p3=$!
wait "${p0}"; wait "${p1}"; wait "${p2}"; wait "${p3}"
echo "Deterministic PMRF reevaluation complete: ${RUN_ROOT}"
