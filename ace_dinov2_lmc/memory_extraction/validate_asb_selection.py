"""Validate Anchor-Support selection without running memory inference.

This script checks the ASB full-pool behavior that matters for S1 stability:
`candidate_pool_ratio <= 0` must not become reference-local through adaptive
ref-distance/supportability filters, and the final selection should preserve
scene-level camera coverage.

Run from the parent project root, for example:

    conda run -n mapanything_new python -m ace_dinov2_lmc.memory_extraction.validate_asb_selection
    conda run -n mapanything_new python -m ace_dinov2_lmc.memory_extraction.validate_asb_selection \
        --dataset_path /path/to/wai_root --scene_name <scene> --n_memory 40 \
        --output /tmp/asb_selection_validation.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from .run_memory_extraction import (
    _extract_centers_from_wai_scene_meta,
    _load_wai_pairwise_covisibility,
    _selection_coverage_summary,
    anchor_support_select_views,
    fps_select_views,
)


def _compact_summary(summary: Dict[str, Any]) -> Dict[str, Any]:
    coverage = summary["coverage"]
    ref_distance = summary["ref_distance"]
    grid = summary.get("pca_grid_occupancy", {})
    strong = summary.get("strong_covis_graph", {})
    return {
        "num_scene_views": int(summary["num_scene_views"]),
        "num_selected_views": int(summary["num_selected_views"]),
        "coverage_mean_m": float(coverage["mean"]),
        "coverage_p95_m": float(coverage["p95"]),
        "coverage_max_m": float(coverage["max"]),
        "ref_distance_mean_m": float(ref_distance["mean"]),
        "ref_distance_p90_m": float(ref_distance["p90"]),
        "ref_distance_max_m": float(ref_distance["max"]),
        "grid_occupancy_ratio": float(grid.get("occupancy_ratio", 0.0)),
        "strong_covis_components": int(strong.get("num_components", 0)),
        "strong_covis_isolated": int(strong.get("isolated_count", 0)),
    }


def _print_compact_table(rows: Dict[str, Dict[str, Any]]) -> None:
    columns = [
        "name",
        "views",
        "cov_mean",
        "cov_p95",
        "cov_max",
        "ref_mean",
        "ref_p90",
        "ref_max",
        "grid_occ",
        "components",
        "isolated",
    ]
    print(" ".join(f"{c:>12}" for c in columns), flush=True)
    for name, row in rows.items():
        values = [
            name,
            str(row["num_selected_views"]),
            f"{row['coverage_mean_m']:.4f}",
            f"{row['coverage_p95_m']:.4f}",
            f"{row['coverage_max_m']:.4f}",
            f"{row['ref_distance_mean_m']:.4f}",
            f"{row['ref_distance_p90_m']:.4f}",
            f"{row['ref_distance_max_m']:.4f}",
            f"{row['grid_occupancy_ratio']:.3f}",
            str(row["strong_covis_components"]),
            str(row["strong_covis_isolated"]),
        ]
        print(" ".join(f"{v:>12}" for v in values), flush=True)


def _make_synthetic_scene(n_views: int = 60) -> tuple[np.ndarray, np.ndarray, int]:
    centers = np.zeros((n_views, 3), dtype=np.float64)
    t = np.linspace(-10.0, 10.0, n_views)
    centers[:, 0] = t
    centers[:, 1] = np.sin(np.linspace(0.0, 6.0, n_views))

    # Local covisibility only. A reference-distance hard filter would keep the
    # selection near the root and fail the max_ref_dist assertion below.
    idx = np.arange(n_views)
    covis = np.zeros((n_views, n_views), dtype=np.float64)
    covis[np.abs(idx[:, None] - idx[None, :]) <= 2] = 1e-3
    np.fill_diagonal(covis, 0.0)

    root = n_views // 2
    return centers, covis, root


def run_synthetic_check(n_memory: int, ref_dist_limit: float) -> Dict[str, Any]:
    centers, covis, root = _make_synthetic_scene()
    selected = anchor_support_select_views(
        centers,
        n_memory,
        covis,
        initial_index=root,
        candidate_pool_ratio=0.0,
        adaptive=True,
        adaptive_scene_stats=False,
        adaptive_view_count=False,
        ref_dist_max_limit=ref_dist_limit,
        ref_dist_mean_limit=max(ref_dist_limit * 0.5, 1e-6),
        target_coverage_mean=0.5,
        target_coverage_max=1.0,
        log_tag="ValidateASB:Synthetic",
    )
    summary = _selection_coverage_summary(centers, selected, ref_idx=root, covisibility=covis)
    compact = _compact_summary(summary)
    compact["selected_indices"] = [int(i) for i in selected]

    if len(selected) != int(n_memory):
        raise AssertionError(f"ASB selected {len(selected)} views, expected {n_memory}")
    if compact["ref_distance_max_m"] <= float(ref_dist_limit) * 1.5:
        raise AssertionError(
            "ASB fullpool still looks reference-local: "
            f"ref_distance_max={compact['ref_distance_max_m']:.4f}m, "
            f"limit={float(ref_dist_limit):.4f}m"
        )
    if compact["coverage_max_m"] > 2.0:
        raise AssertionError(
            "ASB fullpool failed synthetic coverage check: "
            f"coverage_max={compact['coverage_max_m']:.4f}m"
        )
    return compact


def run_wai_scene_check(args: argparse.Namespace) -> Dict[str, Any]:
    centers = _extract_centers_from_wai_scene_meta(args.dataset_path, args.scene_name)
    if centers is None or len(centers) == 0:
        raise RuntimeError(
            f"Could not load camera centers from {Path(args.dataset_path) / args.scene_name / 'scene_meta.json'}"
        )

    covis = _load_wai_pairwise_covisibility(args.dataset_path, args.scene_name, n_expected=len(centers))
    if covis is None:
        raise RuntimeError("ASB validation needs WAI covisibility under <scene>/covisibility/v0/*.npy")

    scene_center = centers.mean(axis=0)
    ref_idx = int(np.argmin(np.linalg.norm(centers - scene_center.reshape(1, 3), axis=1)))
    fps_selected = fps_select_views(centers, args.n_memory, force_include=[ref_idx])
    asb_selected = anchor_support_select_views(
        centers,
        args.n_memory,
        covis,
        initial_index=ref_idx,
        alpha=args.alpha,
        eps=args.eps,
        tau=args.tau,
        ref_lambda=args.ref_lambda,
        anchor_count=args.anchor_count,
        support_per_anchor=args.support_per_anchor,
        support_tau=args.support_tau,
        support_min_neighbors=args.support_min_neighbors,
        safe_dist_to_ref=args.safe_dist_to_ref,
        far_view_budget=args.far_view_budget,
        far_anchor_budget=args.far_anchor_budget,
        coverage_beta=args.coverage_beta,
        candidate_pool_ratio=args.candidate_pool_ratio,
        adaptive=args.adaptive,
        adaptive_scene_stats=not args.disable_adaptive_scene_stats,
        adaptive_view_count=args.adaptive_view_count,
        adaptive_min_views=args.adaptive_min_views,
        ref_dist_max_limit=args.ref_dist_max_limit,
        ref_dist_mean_limit=args.ref_dist_mean_limit,
        min_coverage_gain=args.min_coverage_gain,
        gain_patience=args.gain_patience,
        target_coverage_mean=args.target_coverage_mean,
        target_coverage_max=args.target_coverage_max,
        log_tag="ValidateASB:WAI",
    )

    fps_summary = _selection_coverage_summary(centers, fps_selected, ref_idx=ref_idx, covisibility=covis)
    asb_summary = _selection_coverage_summary(centers, asb_selected, ref_idx=ref_idx, covisibility=covis)
    fps_compact = _compact_summary(fps_summary)
    asb_compact = _compact_summary(asb_summary)

    if len(asb_selected) != int(args.n_memory) and not args.adaptive_view_count:
        raise AssertionError(f"ASB selected {len(asb_selected)} views, expected {args.n_memory}")

    cov_ratio = asb_compact["coverage_max_m"] / max(fps_compact["coverage_max_m"], 1e-9)
    if args.max_coverage_max_ratio > 0 and cov_ratio > args.max_coverage_max_ratio:
        raise AssertionError(
            "ASB coverage tail is too much worse than FPS: "
            f"ratio={cov_ratio:.3f}, limit={args.max_coverage_max_ratio:.3f}"
        )

    return {
        "scene": args.scene_name,
        "dataset_path": args.dataset_path,
        "reference_index": ref_idx,
        "n_memory": int(args.n_memory),
        "asb_selected_indices": [int(i) for i in asb_selected],
        "fps_selected_indices": [int(i) for i in fps_selected],
        "compact": {
            "asb_fullpool": asb_compact,
            "fps": fps_compact,
        },
        "full": {
            "asb_fullpool": asb_summary,
            "fps": fps_summary,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_path", type=str, default=None)
    parser.add_argument("--scene_name", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--n_memory", type=int, default=40)
    parser.add_argument("--synthetic_ref_dist_limit", type=float, default=2.0)
    parser.add_argument("--skip_synthetic", action="store_true")

    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--eps", type=float, default=1e-6)
    parser.add_argument("--tau", type=float, default=-1.0)
    parser.add_argument("--ref_lambda", type=float, default=0.0)
    parser.add_argument("--anchor_count", type=int, default=0)
    parser.add_argument("--support_per_anchor", type=int, default=3)
    parser.add_argument("--support_tau", type=float, default=1e-4)
    parser.add_argument("--support_min_neighbors", type=int, default=3)
    parser.add_argument("--safe_dist_to_ref", type=float, default=3.0)
    parser.add_argument("--far_view_budget", type=int, default=8)
    parser.add_argument("--far_anchor_budget", type=int, default=2)
    parser.add_argument("--coverage_beta", type=float, default=1.0)
    parser.add_argument("--candidate_pool_ratio", type=float, default=0.0)

    parser.add_argument("--adaptive", action="store_true")
    parser.add_argument("--adaptive_view_count", action="store_true")
    parser.add_argument("--disable_adaptive_scene_stats", action="store_true")
    parser.add_argument("--adaptive_min_views", type=int, default=28)
    parser.add_argument("--ref_dist_max_limit", type=float, default=4.2)
    parser.add_argument("--ref_dist_mean_limit", type=float, default=2.7)
    parser.add_argument("--min_coverage_gain", type=float, default=0.01)
    parser.add_argument("--gain_patience", type=int, default=2)
    parser.add_argument("--target_coverage_mean", type=float, default=0.55)
    parser.add_argument("--target_coverage_max", type=float, default=2.0)
    parser.add_argument(
        "--max_coverage_max_ratio",
        type=float,
        default=1.5,
        help="Fail if ASB coverage_max / FPS coverage_max exceeds this value; <=0 disables.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report: Dict[str, Any] = {}

    if not args.skip_synthetic:
        synthetic = run_synthetic_check(
            n_memory=min(int(args.n_memory), 12),
            ref_dist_limit=float(args.synthetic_ref_dist_limit),
        )
        report["synthetic"] = synthetic
        print("[ValidateASB] Synthetic check passed.", flush=True)

    if args.dataset_path and args.scene_name:
        wai_report = run_wai_scene_check(args)
        report["wai_scene"] = wai_report
        _print_compact_table(wai_report["compact"])
        print("[ValidateASB] WAI scene check passed.", flush=True)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"[ValidateASB] Report saved: {out_path}", flush=True)


if __name__ == "__main__":
    main()
