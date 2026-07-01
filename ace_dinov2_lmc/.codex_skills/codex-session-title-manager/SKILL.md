---
name: codex-session-title-manager
description: Rename local Codex conversation titles shown by `codex resume`, especially when multiple CODEX_HOME account profiles share local sessions. Use when the user asks to rename, retitle, list, or safely edit Codex resume conversation names/titles across `.codex`, `.codex-acc*`, or shared-session setups.
---

# Codex Session Title Manager

Use this skill to safely change the local title shown by `codex resume`.

The title is stored in each account profile's SQLite state DB, usually in:

```text
~/.codex*/state_*.sqlite table threads columns title, preview
```

Do not edit rollout JSONL transcripts for a title rename. Do not change `first_user_message`.

## Workflow

1. Identify the target thread:
   - Prefer `CODEX_THREAD_ID` for the current conversation.
   - Use `--thread-id <id>` when editing another conversation.

2. Inspect first:

```bash
python .codex_skills/codex-session-title-manager/scripts/rename_codex_session_title.py --list
```

3. Dry-run the change:

```bash
python .codex_skills/codex-session-title-manager/scripts/rename_codex_session_title.py \
  --title "最终全面训练"
```

4. Apply only after the target rows look correct:

```bash
python .codex_skills/codex-session-title-manager/scripts/rename_codex_session_title.py \
  --title "最终全面训练" \
  --apply
```

The script backs up every touched `state_*.sqlite` plus `-wal`/`-shm` sidecars before writing.

## Multi-account setups

By default the script scans `~/.codex`, `~/.codex-acc*`, and the active `CODEX_HOME`.

Use `--account-root` one or more times to restrict the edit:

```bash
python .codex_skills/codex-session-title-manager/scripts/rename_codex_session_title.py \
  --thread-id 019ee0b4-eae5-7773-99de-2601dda98247 \
  --title "最终全面训练" \
  --account-root /home/xwh/.codex-acc1 \
  --account-root /home/xwh/.codex-acc2 \
  --apply
```

## Safety rules

- Always inspect or dry-run before applying unless the user explicitly gave the exact title and target.
- Keep the transcript JSONL unchanged.
- Keep `first_user_message` unchanged.
- Update both `title` and `preview`; this makes the resume picker display cleanly.
- If no rows match, stop and report that the requested thread was not found.
- If a restore is needed, copy the backed-up SQLite files from the printed backup directory.
