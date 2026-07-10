#!/usr/bin/env bash
set -euo pipefail

# Monitor the final full experiment queue in a terminal.
#
# Usage from /home/xwh/project/ace_depth:
#   bash ace_dinov2_lmc/scripts/monitor_final_full_matrix_20260705.sh
#
# Optional:
#   INTERVAL=60 bash ace_dinov2_lmc/scripts/monitor_final_full_matrix_20260705.sh
#   ONCE=true  bash ace_dinov2_lmc/scripts/monitor_final_full_matrix_20260705.sh
#   RUN_ROOT=/data/... bash ace_dinov2_lmc/scripts/monitor_final_full_matrix_20260705.sh

ROOT_DIR="${ROOT_DIR:-/home/xwh/project/ace_depth}"
RUN_ROOT="${RUN_ROOT:-/data/xwh/ace_dinov2_lmc/04_evaluation/shared/stage2/final_full/20260705_final_full_all_value_raw_auto_ifw005_it10_buf10m_h256_gpu01_expandable}"
SESSION="${SESSION:-final_full_20260705_final_full_gpu01}"
INTERVAL="${INTERVAL:-30}"
ONCE="${ONCE:-false}"
PYTHON_BIN="${PYTHON_BIN:-/home/xwh/miniforge3/envs/mapanything/bin/python}"

MATRIX="${RUN_ROOT}/matrix.tsv"
LOG_DIR="${RUN_ROOT}/launcher_logs"

cd "${ROOT_DIR}"

bool_true() {
  case "${1:-}" in
    1|true|True|TRUE|yes|YES|y|Y) return 0 ;;
    *) return 1 ;;
  esac
}

print_once() {
  clear 2>/dev/null || true
  echo "==== ACE final full matrix monitor ===="
  date '+time: %F %T'
  echo "run_root: ${RUN_ROOT}"
  echo "matrix  : ${MATRIX}"
  echo

  if [[ ! -s "${MATRIX}" ]]; then
    echo "ERROR: matrix not found or empty: ${MATRIX}"
    return 1
  fi

  "${PYTHON_BIN}" - "${MATRIX}" "${LOG_DIR}" <<'PY'
import csv
import os
import re
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

matrix_path = Path(sys.argv[1])
log_dir = Path(sys.argv[2])
rows = list(csv.DictReader(matrix_path.open(newline=""), delimiter="\t"))

def parse_time(s):
    if not s:
        return None
    try:
        return datetime.fromisoformat(s)
    except Exception:
        return None

def elapsed(start, end=None):
    st = parse_time(start)
    if st is None:
        return "-"
    ed = parse_time(end) or datetime.now()
    sec = max(0, int((ed - st).total_seconds()))
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m"
    return f"{m}m{s:02d}s"

def short_scene(job_id):
    for prefix in (
        "indoor6_dino_stage1_",
        "indoor6_acefcn_stage12_",
        "cambridge_acefcn_stage12_",
        "wayspots_acefcn_stage12_",
    ):
        if job_id.startswith(prefix):
            return job_id[len(prefix):]
    return job_id

def log_candidates(row):
    job = row.get("job_id", "")
    claimed = row.get("claimed_by", "")
    gpu = None
    m = re.search(r"gpu(\d+)", claimed)
    if m:
        gpu = m.group(1)
    paths = []
    lp = row.get("log_path") or ""
    if lp:
        paths.append(Path(lp))
    if gpu is not None:
        paths.append(log_dir / f"{job}_gpu{gpu}.log")
    paths.extend(sorted(log_dir.glob(f"{job}_gpu*.log")))
    out = []
    seen = set()
    for p in paths:
        sp = str(p)
        if sp not in seen:
            seen.add(sp)
            out.append(p)
    return out

def tail_text(path, max_bytes=24000):
    try:
        with path.open("rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - max_bytes))
            return f.read().decode("utf-8", errors="replace")
    except Exception:
        return ""

def follow_nested_train_log(text):
    if not text:
        return None
    patterns = [
        r"log=(/[^ \n\r]+/train_[^ \n\r]+\.log)",
        r"\[train start\].*?log=(/[^ \n\r]+\.log)",
    ]
    for pat in patterns:
        ms = list(re.finditer(pat, text))
        if ms:
            p = Path(ms[-1].group(1))
            if p.is_file():
                return p
    return None

def stage_from_text(text):
    if not text:
        return "-"
    patterns = [
        (r"post.?train.*seed|eval_summary_.*post_train|Post-train", "post-train eval"),
        (r"Testing|Evaluating|eval_summary|DSAC|hypotheses", "eval"),
        (r"\[ACE-G\] Iteration\s+(\d+)/(\d+)", "iter"),
        (r"\[S2", "S2/head+fusion"),
        (r"\[S1", "S1/compressor"),
        (r"Buffer:\s+([0-9]+)%", "buffer"),
        (r"Buffer will be allocated|Starting creation of the training buffer", "buffer build"),
        (r"\[train start\]", "starting"),
    ]
    for pat, label in patterns:
        ms = list(re.finditer(pat, text, re.IGNORECASE))
        if not ms:
            continue
        m = ms[-1]
        if label == "iter":
            return f"iter {m.group(1)}/{m.group(2)}"
        if label == "buffer":
            return f"buffer {m.group(1)}%"
        return label
    return "running"

def buffer_progress(text):
    ms = list(re.finditer(r"Buffer:\s+([0-9]+)%.*?([0-9.]+[kM]?)/([0-9.]+[kM]?)", text))
    if not ms:
        ms = list(re.finditer(r"Buffer:\s+([0-9]+)%", text))
    if not ms:
        return ""
    m = ms[-1]
    if len(m.groups()) >= 3:
        return f"{m.group(1)}% ({m.group(2)}/{m.group(3)})"
    return f"{m.group(1)}%"

counts = Counter(r.get("status", "") for r in rows)
total = len(rows)
done = counts.get("done", 0)
failed = counts.get("failed", 0)
running = counts.get("running", 0)
pending = counts.get("pending", 0)
print(f"queue: total={total} done={done} running={running} pending={pending} failed={failed}")

completed = [r for r in rows if r.get("status") == "done"]
failed_rows = [r for r in rows if r.get("status") == "failed"]
running_rows = [r for r in rows if r.get("status") == "running"]
pending_rows = [r for r in rows if r.get("status") == "pending"]
completed.sort(key=lambda r: r.get("finished_at", ""))
pending_rows.sort(key=lambda r: (int(r.get("priority") or 999), r.get("job_id", "")))

print("\nRUNNING")
if not running_rows:
    print("  none")
else:
    for r in running_rows:
        text = ""
        log_path = "-"
        for p in log_candidates(r):
            if p.is_file():
                log_path = str(p)
                text = tail_text(p)
                nested = follow_nested_train_log(text)
                if nested is not None:
                    log_path = str(nested)
                    text = tail_text(nested)
                break
        stage = stage_from_text(text)
        progress = buffer_progress(text)
        extra = f", {progress}" if progress else ""
        print(f"  {r.get('job_id')} | {r.get('claimed_by')} | {elapsed(r.get('started_at'))} | {stage}{extra}")
        print(f"    log: {log_path}")

print("\nDONE recent")
if not completed:
    print("  none")
else:
    for r in completed[-12:]:
        print(f"  {r.get('finished_at')} | {elapsed(r.get('started_at'), r.get('finished_at'))} | {r.get('job_id')}")

print("\nFAILED")
if not failed_rows:
    print("  none")
else:
    for r in failed_rows:
        print(f"  {r.get('job_id')} | exit={r.get('exit_code')} | log={r.get('log_path')}")

print("\nNEXT pending")
if not pending_rows:
    print("  none")
else:
    for r in pending_rows[:12]:
        print(f"  p{r.get('priority')} | {r.get('job_id')}")
    if len(pending_rows) > 12:
        print(f"  ... {len(pending_rows) - 12} more")
PY

  echo
  echo "TMUX"
  if tmux has-session -t "${SESSION}" 2>/dev/null; then
    tmux list-windows -t "${SESSION}" 2>/dev/null || true
  else
    echo "  no tmux session: ${SESSION}"
  fi

  echo
  echo "GPU"
  nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null || true
  echo "-- compute apps --"
  nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader,nounits 2>/dev/null || true

  echo
  echo "DISK"
  df -h /data 2>/dev/null || true

  echo
  echo "tips:"
  echo "  tmux attach -t ${SESSION}"
  echo "  ACTION=status RUN_ROOT='${RUN_ROOT}' bash ace_dinov2_lmc/scripts/launch_final_full_matrix_20260705.sh"
}

while true; do
  print_once
  if bool_true "${ONCE}"; then
    break
  fi
  sleep "${INTERVAL}"
done
