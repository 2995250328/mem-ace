#!/usr/bin/env bash
# 将本地 ace_depth 代码同步到远程（排除数据、权重、日志等；单文件最大 20MB）
# 适用：Linux 服务器或 WSL（在 WSL 里若从 Windows 盘同步，可把 ACE_ROOT 设为 /mnt/d/学习资料/科研相关/ace_depth）
#
# 用法：
#   ./scripts/rsync_ace_depth_upload.sh           # 正式上传
#   ./scripts/rsync_ace_depth_upload.sh --dry-run # 仅列出将传输的文件
#
# 环境变量（可选）：
#   REMOTE_HOST   默认 10.13.73.88
#   REMOTE_PORT   默认 6247（非 22 时必须设置，与 sshd 监听端口一致）
#   REMOTE_USER   默认 xwh
#   REMOTE_BASE   默认 /home/xwh/project（远程上 ace_depth 所在父目录）
#   ACE_ROOT      默认 本脚本所在仓库的 ace_depth 根目录

set -euo pipefail

DRY_RUN=()
if [[ "${1:-}" == "--dry-run" ]] || [[ "${1:-}" == "-n" ]]; then
  DRY_RUN=(-n)
fi

REMOTE_HOST="${REMOTE_HOST:-10.13.73.88}"
REMOTE_PORT="${REMOTE_PORT:-6247}"
REMOTE_USER="${REMOTE_USER:-xwh}"
REMOTE_BASE="${REMOTE_BASE:-/home/xwh/project}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ACE_ROOT="${ACE_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"

REMOTE_DEST="${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_BASE}/ace_depth/"

echo "=== rsync upload ==="
echo "  本地: ${ACE_ROOT}/"
echo "  远程: ${REMOTE_DEST}"
echo "  SSH:  -p ${REMOTE_PORT}"
[[ ${#DRY_RUN[@]} -gt 0 ]] && echo "  模式: dry-run（不会真正传输）"
echo ""

rsync -avz "${DRY_RUN[@]}" --progress --prune-empty-dirs \
  -e "ssh -p ${REMOTE_PORT}" \
  --max-size=20m \
  --exclude='.git/' \
  --exclude='__pycache__/' \
  --exclude='.pytest_cache/' \
  --exclude='.mypy_cache/' \
  --exclude='.ruff_cache/' \
  --exclude='.ipynb_checkpoints/' \
  --exclude='output/' \
  --exclude='outputs/' \
  --exclude='runs/' \
  --exclude='wandb/' \
  --exclude='logs/' \
  --exclude='checkpoints/' \
  --exclude='weights/' \
  --exclude='pretrained/' \
  --exclude='data/' \
  --exclude='datasets/' \
  --exclude='*.pt' \
  --exclude='*.pth' \
  --exclude='*.ckpt' \
  --exclude='*.safetensors' \
  --exclude='*.bin' \
  --exclude='*.onnx' \
  --exclude='*.npy' \
  --exclude='*.npz' \
  --exclude='*.pkl' \
  --exclude='*.h5' \
  --exclude='*.tar' \
  --exclude='*.tar.gz' \
  --exclude='*.zip' \
  "${ACE_ROOT}/" \
  "${REMOTE_DEST}"

echo ""
echo "=== 完成 ==="
