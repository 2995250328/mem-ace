#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

CLAUDE_SRC="$REPO_ROOT/.claude/skills"
CODEX_SRC="$REPO_ROOT/.codex_skills"
CURSOR_SRC="$REPO_ROOT/.cursor"

CLAUDE_DST="${HOME}/.claude/skills"
CODEX_DST="${HOME}/.codex/skills"
CURSOR_DST="${HOME}/.cursor"
CURSOR_SKILLS_DST="${HOME}/.cursor/skills-cursor"

PREFIX="ace-dinov2-lmc"
EXTERNAL_SKILLS_ROOT="/home/xwh/project/skills_research/superpowers/skills"
EXTERNAL_SHARED_ROOT="/home/xwh/project/ace_depth/.agents/skills"

link_skill_dir() {
  local src_root="$1"
  local dst_root="$2"
  local entry
  mkdir -p "$dst_root"
  for entry in "$src_root"/*; do
    [ -d "$entry" ] || continue
    local base
    base="$(basename "$entry")"
    ln -sfn "$entry" "$dst_root/${PREFIX}-${base}"
    printf '[linked] %s -> %s\n' "$dst_root/${PREFIX}-${base}" "$entry"
  done
}

printf 'Repo root: %s\n' "$REPO_ROOT"

printf '\n== Claude Code skills ==\n'
link_skill_dir "$CLAUDE_SRC" "$CLAUDE_DST"

printf '\n== Codex skills ==\n'
link_skill_dir "$CODEX_SRC" "$CODEX_DST"

printf '\n== Cursor MCP config ==\n'
mkdir -p "$CURSOR_DST"
ln -sfn "$CURSOR_SRC/mcp.json" "$CURSOR_DST/mcp.json"
printf '[linked] %s -> %s\n' "$CURSOR_DST/mcp.json" "$CURSOR_SRC/mcp.json"

printf '\n== External Skills ==\n'
mkdir -p "${HOME}/.agents/skills" "$CLAUDE_DST" "$CODEX_DST" "$CURSOR_SKILLS_DST"

if [ -d "$EXTERNAL_SKILLS_ROOT" ]; then
  ln -sfn "$EXTERNAL_SKILLS_ROOT" "${HOME}/.agents/skills/superpowers"
  printf '[linked] %s -> %s\n' "${HOME}/.agents/skills/superpowers" "$EXTERNAL_SKILLS_ROOT"
  for entry in "$EXTERNAL_SKILLS_ROOT"/*; do
    [ -d "$entry" ] || continue
    base="$(basename "$entry")"
    ln -sfn "$entry" "$CLAUDE_DST/superpowers-${base}"
    ln -sfn "$entry" "$CODEX_DST/superpowers-${base}"
    ln -sfn "$entry" "$CURSOR_SKILLS_DST/superpowers-${base}"
    printf '[linked] superpowers-%s into Claude/Codex/Cursor\n' "$base"
  done
fi

for base in karpathy grill-me; do
  if [ -d "$EXTERNAL_SHARED_ROOT/$base" ]; then
    ln -sfn "$EXTERNAL_SHARED_ROOT/$base" "$CLAUDE_DST/$base"
    ln -sfn "$EXTERNAL_SHARED_ROOT/$base" "$CODEX_DST/$base"
    ln -sfn "$EXTERNAL_SHARED_ROOT/$base" "$CURSOR_SKILLS_DST/$base"
    printf '[linked] %s into Claude/Codex/Cursor\n' "$base"
  fi
done

printf '\nDone.\n'
printf 'Claude skills dir: %s\n' "$CLAUDE_DST"
printf 'Codex skills dir: %s\n' "$CODEX_DST"
printf 'Cursor config: %s\n' "$CURSOR_DST/mcp.json"
printf 'Cursor skills dir: %s\n' "$CURSOR_SKILLS_DST"
