#!/usr/bin/env python3
"""
Collect post_train_eval.txt files from output/ into ace_fcn_lmc/04_evaluation/logs/
for use with /research-eval ace_fcn_lmc.
"""
import shutil
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent  # project root
OUTPUT = ROOT / "output"
LOGS_DIR = Path(__file__).parent / "logs"
LOGS_DIR.mkdir(exist_ok=True)

algorithms = {"ace_fcn_vanilla", "ace_fcn_lmc_iter", "ace_fcn_lmc_aceg"}
collected = 0

for eval_file in OUTPUT.rglob("post_train_eval.txt"):
    # Check if this belongs to an ace_fcn run
    parts = eval_file.parts
    if not any(a in parts for a in algorithms):
        continue
    # Build a unique name: dataset_scene_algorithm_runid.txt
    run_dir = eval_file.parent
    try:
        # run_dir = output/{dataset}/{scene}/{algorithm}/{run_id}
        run_id = run_dir.name
        algorithm = run_dir.parent.name
        scene = run_dir.parent.parent.name
        dataset = run_dir.parent.parent.parent.name
        dest_name = f"{dataset}_{scene}_{algorithm}_{run_id}_post_train_eval.txt"
    except Exception:
        dest_name = f"{run_dir.name}_post_train_eval.txt"
    dest = LOGS_DIR / dest_name
    shutil.copy2(eval_file, dest)
    print(f"Copied: {eval_file.relative_to(ROOT)} -> logs/{dest_name}")
    collected += 1

print(f"\nCollected {collected} eval file(s) into {LOGS_DIR}")
