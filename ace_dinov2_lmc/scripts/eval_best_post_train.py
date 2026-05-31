#!/usr/bin/env python3
"""Run multi-seed post-train eval for an existing checkpoint."""

import argparse
import sys
import time
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
PKG_ROOT = REPO_ROOT / "ace_dinov2_lmc"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(PKG_ROOT))

from test_ace_dinov2_lmc import run_evaluation_lmc  # noqa: E402


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("scene", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--data_backend", default="ace", choices=["ace", "wai"])
    parser.add_argument("--image_resolution", type=int, default=512)
    parser.add_argument("--hypotheses", type=int, default=256)
    parser.add_argument("--seeds", nargs="+", type=int, default=[1305, 2026, 4242])
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--ace_encoder_path", type=Path, default=Path("/home/xwh/project/ace_depth/ace_encoder_pretrained.pt"))
    parser.add_argument("--glace_root", type=Path, default=Path("/home/xwh/project/glace"))
    parser.add_argument("--glace_encoder_path", type=Path, default=Path("/home/xwh/project/glace/ace_encoder_pretrained.pt"))
    parser.add_argument("--glace_feat_name", default="features.npy")
    parser.add_argument("--eval_num_workers", type=int, default=6)
    parser.add_argument("--eval_deterministic", action="store_true")
    parser.add_argument("--dsacstar_seed_per_frame", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main():
    args = _parse_args()
    run_dir = args.output_dir or args.checkpoint.parent
    run_dir.mkdir(parents=True, exist_ok=True)

    results = []
    start = time.time()
    for seed in args.seeds:
        opt = argparse.Namespace(
            scene=args.scene,
            network=args.checkpoint,
            data_backend=args.data_backend,
            wai_repo_root=None,
            wai_image_modality="image",
            dinov2_path=Path("/home/xwh/data/checkpoints/dinov2_vitl14_pretrain.pth"),
            ace_encoder_path=args.ace_encoder_path,
            glace_root=args.glace_root,
            glace_encoder_path=args.glace_encoder_path,
            glace_feat_name=args.glace_feat_name,
            device=args.device,
            image_resolution=args.image_resolution,
            session=f"post_train_seed{seed}",
            hypotheses=args.hypotheses,
            threshold=10,
            inlieralpha=100,
            maxpixelerror=100,
            eval_deterministic=args.eval_deterministic,
            dsacstar_seed=seed,
            dsacstar_seed_per_frame=args.dsacstar_seed_per_frame,
            eval_num_workers=args.eval_num_workers,
            render_visualization=False,
            log_per_frame=False,
            lmc_log_runtime_stats=False,
            lmc_runtime_stats_interval=100,
            lmc_runtime_stats_max_pixels=4096,
            output_dir=run_dir,
            ensemble_networks=None,
        )
        result = run_evaluation_lmc(opt)
        result["seed"] = seed
        results.append(result)

    aggregate = dict(results[0])
    if len(results) > 1:
        for key in ["median_rErr", "median_tErr", "pct25_5", "pct10_5", "pct5", "pct2", "pct1", "avg_time"]:
            aggregate[key] = float(np.median([r[key] for r in results]))
        aggregate["total_frames"] = int(results[0]["total_frames"])

    summary_path = run_dir / "post_train_eval.txt"
    with summary_path.open("w", encoding="utf-8") as f:
        f.write("eval_type\tpost_train\n")
        f.write(f"eval_deterministic\t{bool(args.eval_deterministic)}\n")
        f.write(f"aggregation\t{'median' if len(results) > 1 else 'single'}\n")
        f.write(f"seeds\t{','.join(str(r['seed']) for r in results)}\n")
        f.write(f"post_train_eval_seeds\t{','.join(str(r['seed']) for r in results)}\n")
        f.write(f"hypotheses\t{args.hypotheses}\n")
        f.write(f"post_train_hypotheses\t{args.hypotheses}\n")
        f.write(f"dsacstar_seed_per_frame\t{bool(args.dsacstar_seed_per_frame)}\n")
        f.write(f"median_rotation_deg\t{aggregate['median_rErr']:.4f}\n")
        f.write(f"median_translation_cm\t{aggregate['median_tErr']:.4f}\n")
        f.write(f"accuracy_25cm5deg_pct\t{aggregate['pct25_5']:.2f}\n")
        f.write(f"accuracy_10cm5deg_pct\t{aggregate['pct10_5']:.2f}\n")
        f.write(f"accuracy_5cm5deg_pct\t{aggregate['pct5']:.2f}\n")
        f.write(f"accuracy_2cm2deg_pct\t{aggregate['pct2']:.2f}\n")
        f.write(f"accuracy_1cm1deg_pct\t{aggregate['pct1']:.2f}\n")
        f.write(f"avg_time_per_frame_ms\t{aggregate['avg_time'] * 1000:.2f}\n")
        f.write(f"total_frames\t{aggregate['total_frames']}\n")
        f.write(f"checkpoint\t{args.checkpoint}\n")

    if len(results) > 1:
        seed_path = run_dir / "post_train_eval_seed_runs.txt"
        with seed_path.open("w", encoding="utf-8") as f:
            f.write("# seed-wise post-train evaluation results\n")
            f.write("seed\tmedian_rErr\tmedian_tErr\tpct25_5\tpct10_5\tpct5\tpct2\tpct1\tavg_time_ms\ttotal_frames\n")
            for result in results:
                f.write(
                    f"{result['seed']}\t{result['median_rErr']:.4f}\t{result['median_tErr']:.4f}\t"
                    f"{result['pct25_5']:.2f}\t{result['pct10_5']:.2f}\t{result['pct5']:.2f}\t"
                    f"{result['pct2']:.2f}\t{result['pct1']:.2f}\t"
                    f"{result['avg_time'] * 1000:.2f}\t{result['total_frames']}\n"
                )
    else:
        seed_path = None

    print("========== Post-train Eval (aggregated) ==========")
    print(
        f"Median: {aggregate['median_rErr']:.2f} deg, {aggregate['median_tErr']:.2f} cm | "
        f"25cm/5deg: {aggregate['pct25_5']:.2f}% | 10cm/5deg: {aggregate['pct10_5']:.2f}% | "
        f"5cm/5deg: {aggregate['pct5']:.2f}% | 2cm/2deg: {aggregate['pct2']:.2f}% | "
        f"1cm/1deg: {aggregate['pct1']:.2f}%"
    )
    print(f"Avg time: {aggregate['avg_time'] * 1000:.2f} ms | Frames: {aggregate['total_frames']}")
    print(f"Seeds: {','.join(str(r['seed']) for r in results)} | hypotheses={args.hypotheses}")
    print(f"Eval summary written to: {summary_path}")
    if seed_path is not None:
        print(f"Seed-wise eval summary written to: {seed_path}")
    print(f"Elapsed: {time.time() - start:.1f}s")


if __name__ == "__main__":
    main()
