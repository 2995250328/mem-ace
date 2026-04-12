#!/usr/bin/env python3
"""
Post-train evaluation launcher. Run as a separate process so the training process
can exit and release GPU memory before test_ace_dinov2.py uses the same GPU.
Invoked via os.execv() from train_ace_dinov2.py after training; this script
does not import torch, so the replaced process releases all GPU memory.
"""
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def main():
    if len(sys.argv) != 2:
        print("Usage: run_post_train_eval.py <post_train_eval.json>", file=sys.stderr)
        sys.exit(2)
    json_path = Path(sys.argv[1])
    if not json_path.exists():
        print(f"JSON not found: {json_path}", file=sys.stderr)
        sys.exit(2)

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    cmd = data["cmd"]
    cwd = data.get("cwd") or "."
    env = os.environ.copy()
    env.update(data.get("env") or {})

    result = subprocess.run(cmd, cwd=cwd, env=env)
    if result.returncode != 0:
        sys.exit(result.returncode)

    scene_name = data["scene_name"]
    eval_session = data.get("eval_session") or "eval"
    output_dir = Path(data["output_dir"])
    eval_results_base = output_dir / "eval_results"
    # test_ace_dinov2.py writes last_eval_dir.txt with the run subdir (timestamp_scene_session)
    last_eval_file = eval_results_base / "last_eval_dir.txt"
    if data.get("eval_output_dir"):
        eval_output_dir = Path(data["eval_output_dir"])
    elif last_eval_file.exists():
        eval_run_id = last_eval_file.read_text(encoding="utf-8").strip()
        eval_output_dir = eval_results_base / eval_run_id
    else:
        eval_output_dir = eval_results_base / eval_session
    eval_summary_file = eval_output_dir / f"eval_summary_{scene_name}_{eval_session}.txt"
    if not eval_summary_file.exists():
        print(f"Eval summary not found: {eval_summary_file}", file=sys.stderr)
        sys.exit(1)

    summary = {}
    for line in eval_summary_file.read_text(encoding="utf-8").strip().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "\t" in line:
            key, value = line.split("\t", 1)
            summary[key.strip()] = value.strip()

    def _float(k, default=0.0):
        return float(summary.get(k, default))

    def _int(k, default=0):
        return int(summary.get(k, default))

    eval_result = {
        "median_rErr": _float("median_rotation_deg"),
        "median_tErr": _float("median_translation_cm"),
        "avg_time": _float("avg_time_per_frame_ms") / 1000.0,
        "pct25_5": _float("accuracy_25cm5deg_pct"),
        "pct10_5": _float("accuracy_10cm5deg_pct"),
        "pct5": _float("accuracy_5cm5deg_pct"),
        "pct2": _float("accuracy_2cm2deg_pct"),
        "pct1": _float("accuracy_1cm1deg_pct"),
        "total_frames": _int("total_frames"),
        "test_log_file": str(eval_output_dir / f"test_{scene_name}_{eval_session}.txt"),
        "pose_log_file": str(eval_output_dir / f"poses_{scene_name}_{eval_session}.txt"),
    }

    eval_log_path = eval_output_dir / "eval_log.txt"
    with open(eval_log_path, "w", encoding="utf-8") as f:
        f.write("# DINOv2-ACE evaluation after training\n")
        f.write(f"# Generated: {datetime.now().isoformat()}\n")
        f.write(f"# Scene: {data['scene']}\n")
        f.write(f"# Model: {data['output_map']}\n")
        f.write(f"# Epochs: {data['epochs']}  image_resolution: {data['image_resolution']}\n")
        f.write("\n")
        f.write(f"median_rotation_deg\t{eval_result['median_rErr']:.4f}\n")
        f.write(f"median_translation_cm\t{eval_result['median_tErr']:.4f}\n")
        f.write(f"avg_time_per_frame_ms\t{eval_result['avg_time'] * 1000:.2f}\n")
        f.write(f"accuracy_25cm5deg_pct\t{eval_result['pct25_5']:.2f}\n")
        f.write(f"accuracy_10cm5deg_pct\t{eval_result['pct10_5']:.2f}\n")
        f.write(f"accuracy_5cm5deg_pct\t{eval_result['pct5']:.2f}\n")
        f.write(f"accuracy_2cm2deg_pct\t{eval_result['pct2']:.2f}\n")
        f.write(f"accuracy_1cm1deg_pct\t{eval_result['pct1']:.2f}\n")
        f.write(f"total_frames\t{eval_result['total_frames']}\n")
        f.write(f"test_log_file\t{eval_result['test_log_file']}\n")
        f.write(f"pose_log_file\t{eval_result['pose_log_file']}\n")

    print(f"Evaluation summary written to: {eval_log_path}")
    print(
        "  Median: %.2f deg, %.2f cm | 5cm/5deg: %.2f%% | Avg time: %.2f ms | Frames: %d"
        % (
            eval_result["median_rErr"],
            eval_result["median_tErr"],
            eval_result["pct5"],
            eval_result["avg_time"] * 1000,
            eval_result["total_frames"],
        )
    )


if __name__ == "__main__":
    main()
