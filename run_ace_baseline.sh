#!/bin/bash
# run_ace_baseline.sh
# 训练 + 测试 + 结构化保存日志
# 用法: bash run_ace_baseline.sh <scene_path> <dataset_tag> <scene_tag> [device]
# 示例: bash run_ace_baseline.sh datasets/7scenes_chess 7Scenes chess cuda:0

set -e

SCENE=${1:?"Usage: $0 <scene_path> <dataset_tag> <scene_tag> [device]"}
DATASET_TAG=${2:?"Usage: $0 <scene_path> <dataset_tag> <scene_tag> [device]"}
SCENE_TAG=${3:?"Usage: $0 <scene_path> <dataset_tag> <scene_tag> [device]"}
DEVICE=${4:-cuda:0}

# ── 关键参数（改这里即可）──────────────────────────────────────────
NUM_HEAD_BLOCKS=4
TRAINING_BUFFER_SIZE=8000000
SAMPLES_PER_IMAGE=1024
BATCH_SIZE=5120
EPOCHS=24
LR_MIN=0.0005
LR_MAX=0.005
IMAGE_RESOLUTION=480
USE_HALF=True
USE_HOMOGENEOUS=True
USE_AUG=True
AUG_ROTATION=15
AUG_SCALE=1.5
REPRO_LOSS_TYPE=dyntanh
REPRO_LOSS_SCHEDULE=circle
REPRO_LOSS_HARD_CLAMP=1000
REPRO_LOSS_SOFT_CLAMP=50
REPRO_LOSS_SOFT_CLAMP_MIN=1
# ──────────────────────────────────────────────────────────────────

ALGORITHM="ace_fcn"
CONFIG_TAG="buf${TRAINING_BUFFER_SIZE}_ep${EPOCHS}_bs${BATCH_SIZE}_spi${SAMPLES_PER_IMAGE}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

RUN_DIR="output/${DATASET_TAG}/${SCENE_TAG}/${ALGORITHM}/${TIMESTAMP}_${CONFIG_TAG}"
mkdir -p "${RUN_DIR}"

MAP_FILE="${RUN_DIR}/model.pt"
TRAIN_LOG="${RUN_DIR}/training_log.txt"
TEST_LOG="${RUN_DIR}/test_log.txt"
POSES_FILE="${RUN_DIR}/poses.txt"
CONFIG_FILE="${RUN_DIR}/run_config.json"
CMD_FILE="${RUN_DIR}/run_command.txt"

# 保存配置
cat > "${CONFIG_FILE}" <<EOF
{
  "scene": "${SCENE}",
  "dataset_tag": "${DATASET_TAG}",
  "scene_tag": "${SCENE_TAG}",
  "algorithm": "${ALGORITHM}",
  "device": "${DEVICE}",
  "num_head_blocks": ${NUM_HEAD_BLOCKS},
  "training_buffer_size": ${TRAINING_BUFFER_SIZE},
  "samples_per_image": ${SAMPLES_PER_IMAGE},
  "batch_size": ${BATCH_SIZE},
  "epochs": ${EPOCHS},
  "learning_rate_min": ${LR_MIN},
  "learning_rate_max": ${LR_MAX},
  "image_resolution": ${IMAGE_RESOLUTION},
  "use_half": ${USE_HALF},
  "use_homogeneous": ${USE_HOMOGENEOUS},
  "use_aug": ${USE_AUG},
  "aug_rotation": ${AUG_ROTATION},
  "aug_scale": ${AUG_SCALE},
  "repro_loss_type": "${REPRO_LOSS_TYPE}",
  "repro_loss_schedule": "${REPRO_LOSS_SCHEDULE}",
  "repro_loss_hard_clamp": ${REPRO_LOSS_HARD_CLAMP},
  "repro_loss_soft_clamp": ${REPRO_LOSS_SOFT_CLAMP},
  "repro_loss_soft_clamp_min": ${REPRO_LOSS_SOFT_CLAMP_MIN},
  "run_dir": "${RUN_DIR}",
  "timestamp": "${TIMESTAMP}"
}
EOF

TRAIN_CMD="python train_ace.py ${SCENE} ${MAP_FILE} \
    --device ${DEVICE} \
    --num_head_blocks ${NUM_HEAD_BLOCKS} \
    --training_buffer_size ${TRAINING_BUFFER_SIZE} \
    --samples_per_image ${SAMPLES_PER_IMAGE} \
    --batch_size ${BATCH_SIZE} \
    --epochs ${EPOCHS} \
    --learning_rate_min ${LR_MIN} \
    --learning_rate_max ${LR_MAX} \
    --image_resolution ${IMAGE_RESOLUTION} \
    --use_half ${USE_HALF} \
    --use_homogeneous ${USE_HOMOGENEOUS} \
    --use_aug ${USE_AUG} \
    --aug_rotation ${AUG_ROTATION} \
    --aug_scale ${AUG_SCALE} \
    --repro_loss_type ${REPRO_LOSS_TYPE} \
    --repro_loss_schedule ${REPRO_LOSS_SCHEDULE} \
    --repro_loss_hard_clamp ${REPRO_LOSS_HARD_CLAMP} \
    --repro_loss_soft_clamp ${REPRO_LOSS_SOFT_CLAMP} \
    --repro_loss_soft_clamp_min ${REPRO_LOSS_SOFT_CLAMP_MIN}"

TEST_CMD="python test_ace.py ${SCENE} ${MAP_FILE} \
    --device ${DEVICE} \
    --session baseline"

echo "${TRAIN_CMD}" > "${CMD_FILE}"
echo "${TEST_CMD}" >> "${CMD_FILE}"

echo "[$(date)] Run dir: ${RUN_DIR}"
echo "[$(date)] Starting training..."

# 训练
eval "${TRAIN_CMD}" 2>&1 | tee "${TRAIN_LOG}"
TRAIN_EXIT=${PIPESTATUS[0]}

if [ ${TRAIN_EXIT} -ne 0 ]; then
    echo "[$(date)] Training FAILED (exit ${TRAIN_EXIT})" | tee -a "${TRAIN_LOG}"
    exit ${TRAIN_EXIT}
fi

echo "[$(date)] Training done. Starting test..."

# 测试（poses/test 文件由 test_ace.py 自动写入 map 所在目录，即 RUN_DIR）
eval "${TEST_CMD}" 2>&1 | tee "${TEST_LOG}"
TEST_EXIT=${PIPESTATUS[0]}

if [ ${TEST_EXIT} -ne 0 ]; then
    echo "[$(date)] Test FAILED (exit ${TEST_EXIT})" | tee -a "${TEST_LOG}"
    exit ${TEST_EXIT}
fi

# 从 test_ace.py 生成的 test_*.txt 提取摘要
SCENE_NAME=$(basename "${SCENE}")
TEST_RESULT="${RUN_DIR}/test_${SCENE_NAME}_baseline.txt"
if [ -f "${TEST_RESULT}" ]; then
    cp "${TEST_RESULT}" "${RUN_DIR}/eval_summary.txt"
fi

echo "[$(date)] Done. Results in: ${RUN_DIR}"
