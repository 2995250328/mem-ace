#!/usr/bin/env bash
# 将 ace_depth 项目 + Cursor skills/plans/项目配置 同步到 10.13.73.92:6247，用于在新设备交接
# 用法：./scripts/sync_to_remote_10.13.73.92.sh [user]
# 默认 user 为 xwh；若需密码，执行时会提示输入。

set -e
REMOTE_HOST="10.13.73.92"
REMOTE_PORT="6247"
REMOTE_PATH="/home/xwh/project"
USER="${1:-xwh}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ACE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PROJECT_ROOT="$(cd "$ACE_ROOT/.." && pwd)"
PLAN_SOURCE="${HOME}/.cursor/plans/迭代baseline对比规划_036c67e7.plan.md"
CURSOR_HOME="${HOME}/.cursor"

echo "=== 同步到 ${USER}@${REMOTE_HOST}:${REMOTE_PORT} ==="
echo ""

# 1. 同步整个 ace_depth（含日志，排除权重）
echo ">>> 1/8 ace_depth（排除 *.pt / *.pth）"
rsync -avz -e "ssh -p ${REMOTE_PORT}" \
  --exclude='**/*.pt' \
  --exclude='**/*.pth' \
  "$ACE_ROOT/" "${USER}@${REMOTE_HOST}:${REMOTE_PATH}/ace_depth/"

# 2. 同步规划文档到 ace_depth/docs/
if [[ -f "$PLAN_SOURCE" ]]; then
  echo ""
  echo ">>> 2/8 规划文档到 ace_depth/docs/"
  rsync -avz -e "ssh -p ${REMOTE_PORT}" \
    "$PLAN_SOURCE" "${USER}@${REMOTE_HOST}:${REMOTE_PATH}/ace_depth/docs/"
else
  echo ">>> 2/8 跳过规划文档（未找到 $PLAN_SOURCE）"
fi

# 3. 同步 Cursor 全局 skills 与 plans
echo ""
echo ">>> 3/8 Cursor 全局: skills + plans"
if [[ -d "$CURSOR_HOME/skills" ]]; then
  rsync -avz -e "ssh -p ${REMOTE_PORT}" \
    "$CURSOR_HOME/skills/" "${USER}@${REMOTE_HOST}:${HOME}/.cursor/skills/"
fi
if [[ -d "$CURSOR_HOME/plans" ]]; then
  rsync -avz -e "ssh -p ${REMOTE_PORT}" \
    "$CURSOR_HOME/plans/" "${USER}@${REMOTE_HOST}:${HOME}/.cursor/plans/"
fi

# 4. 同步 .claude/skills（Claude Code 等使用的 skills）
echo ""
echo ">>> 4/8 .claude/skills"
if [[ -d "${HOME}/.claude/skills" ]]; then
  rsync -avz -e "ssh -p ${REMOTE_PORT}" \
    "${HOME}/.claude/skills/" "${USER}@${REMOTE_HOST}:${HOME}/.claude/skills/"
fi

# 5. 同步项目级 .cursor（rules、skills 等）
echo ""
echo ">>> 5/8 项目 .cursor（rules、skills、hooks）"
if [[ -d "$PROJECT_ROOT/.cursor" ]]; then
  rsync -avz -e "ssh -p ${REMOTE_PORT}" \
    "$PROJECT_ROOT/.cursor/" "${USER}@${REMOTE_HOST}:${REMOTE_PATH}/.cursor/"
fi

# 6. 同步 Cursor 其他配置：plugins、skills-cursor、MCP（scc 等相关）
echo ""
echo ">>> 6/8 Cursor 其他: plugins、skills-cursor、mcp.json"
if [[ -d "$CURSOR_HOME/plugins" ]]; then
  rsync -avz -e "ssh -p ${REMOTE_PORT}" \
    "$CURSOR_HOME/plugins/" "${USER}@${REMOTE_HOST}:${HOME}/.cursor/plugins/"
fi
if [[ -d "$CURSOR_HOME/skills-cursor" ]]; then
  rsync -avz -e "ssh -p ${REMOTE_PORT}" \
    "$CURSOR_HOME/skills-cursor/" "${USER}@${REMOTE_HOST}:${HOME}/.cursor/skills-cursor/"
fi
if [[ -f "$CURSOR_HOME/mcp.json" ]]; then
  rsync -avz -e "ssh -p ${REMOTE_PORT}" \
    "$CURSOR_HOME/mcp.json" "${USER}@${REMOTE_HOST}:${HOME}/.cursor/"
fi

# 7. 同步 SSH 配置（若存在，便于新设备连回本机或它机）
echo ""
echo ">>> 7/8 SSH 配置（若存在）"
if [[ -f "${HOME}/.ssh/config" ]]; then
  rsync -avz -e "ssh -p ${REMOTE_PORT}" \
    "${HOME}/.ssh/config" "${USER}@${REMOTE_HOST}:${HOME}/.ssh/config"
else
  echo "    跳过（无 ~/.ssh/config）"
fi

# 8. 同步 cc-launcher.sh（scc 命令）及 bashrc 配置片段
echo ""
echo ">>> 8/8 cc-launcher.sh（scc）"
if [[ -f "${HOME}/cc-launcher.sh" ]]; then
  rsync -avz -e "ssh -p ${REMOTE_PORT}" \
    "${HOME}/cc-launcher.sh" "${USER}@${REMOTE_HOST}:${HOME}/"
  echo "    bashrc 配置片段见: ${REMOTE_PATH}/ace_depth/docs/bashrc_scc_cc_launcher_snippet.txt"
else
  echo "    跳过（无 ~/cc-launcher.sh）"
fi

echo ""
echo "=== 同步完成 ==="
echo "  ace_depth:       ${REMOTE_PATH}/ace_depth/"
echo "  交接+规划:       ${REMOTE_PATH}/ace_depth/docs/"
echo "  Cursor skills:   ${HOME}/.cursor/skills/"
echo "  Cursor plans:   ${HOME}/.cursor/plans/"
echo "  .claude skills: ${HOME}/.claude/skills/"
echo "  项目 .cursor:   ${REMOTE_PATH}/.cursor/"
echo "  Cursor plugins: ${HOME}/.cursor/plugins/"
echo "  skills-cursor:  ${HOME}/.cursor/skills-cursor/"
echo "  MCP:            ${HOME}/.cursor/mcp.json"
echo "  SSH config:     ${HOME}/.ssh/config（若存在）"
echo "  cc-launcher.sh: ${HOME}/cc-launcher.sh（scc 命令）"
echo "  bashrc 片段:    ${REMOTE_PATH}/ace_depth/docs/bashrc_scc_cc_launcher_snippet.txt"
