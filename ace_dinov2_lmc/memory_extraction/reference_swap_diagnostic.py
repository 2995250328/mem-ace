#!/usr/bin/env python3
"""Q1a reference-swap diagnostic for reference-consistent memory construction."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from .run_memory_extraction import (
    MapAnythingExtractor,
    _error_distribution_summary,
    _extract_centers_from_dataset_items,
    _extract_centers_from_flat_wai_dataset,
    _extract_centers_from_pose_dir,
    _extract_centers_from_wai_scene_meta,
    _load_wai_pairwise_covisibility,
    canonicalize_wai_view_mode,
    convert_ace_tuple_to_dict,
    detect_dataset_type,
    load_dataset,
    prepare_batch_input,
    print_pose_prediction_vs_gt,
    select_memory_views,
)


def _strtobool(x: Any) -> bool:
    if isinstance(x, bool):
        return x
    v = str(x).strip().lower()
    if v in ("1", "true", "yes", "on"):
        return True
    if v in ("0", "false", "no", "off"):
        return False
    raise ValueError(f"Invalid boolean value: {x!r}")


def _first_scalar_str(x: Any) -> Optional[str]:
    if isinstance(x, (list, tuple)):
        x = x[0] if x else None
    if x is None:
        return None
    return str(x)


def _view_source_name(view: Any) -> str:
    if not isinstance(view, dict):
        return "<non-dict-view>"
    for key in ("instance", "image_path", "rgb_path", "filename", "path"):
        val = _first_scalar_str(view.get(key))
        if val:
            return val
    return "<unknown>"


def _build_flat_view_lookup(dataset) -> Dict[Tuple[str, str], int]:
    lookup: Dict[Tuple[str, str], int] = {}
    flat_view_list = getattr(dataset, "flat_view_list", None)
    if not isinstance(flat_view_list, list):
        return lookup
    for flat_i, item in enumerate(flat_view_list):
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            lookup[(str(item[0]), str(item[1]))] = int(flat_i)
    return lookup


def _lookup_actual_flat_idx(view: Any, flat_view_lookup: Dict[Tuple[str, str], int], scene_name: str) -> Optional[int]:
    if not isinstance(view, dict) or not flat_view_lookup:
        return None
    label = _first_scalar_str(view.get("label", view.get("scene_name", scene_name))) or str(scene_name)
    source = _view_source_name(view)
    frame = os.path.basename(source)
    candidates = [frame]
    stem, ext = os.path.splitext(frame)
    if ext:
        candidates.append(stem)
    for cand in candidates:
        idx = flat_view_lookup.get((label, cand))
        if idx is not None:
            return int(idx)
    return None


def _dataset_num_views(args: argparse.Namespace) -> int:
    if args.dataset_loader == "wai" and (
        args.wai_view_mode == "original_multiview"
        or (args.wai_view_mode == "anchor_support" and args.covis_native_group_asb)
    ):
        return int(args.n_memory)
    return 1


def _resolve_scene_name(args: argparse.Namespace) -> str:
    if args.scene_name:
        return str(args.scene_name)
    scene_name = os.path.basename(str(args.dataset_path).rstrip("/"))
    if args.dataset_loader == "ace" and scene_name in ("train", "test", "val"):
        parent = os.path.basename(os.path.dirname(str(args.dataset_path).rstrip("/")))
        if parent:
            scene_name = parent
    return scene_name


def _resolve_output_path(args: argparse.Namespace, scene_name: str) -> Path:
    if args.output_path:
        return Path(args.output_path)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_dir = Path(__file__).resolve().parent / "04_evaluation" / "reference_swap" / scene_name
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"reference_swap_report_{stamp}.json"


def _load_selected_views(
    dataset,
    memory_indices: Sequence[int],
    *,
    probe_memory_views: int,
    device: torch.device,
    dataset_loader: str,
    scene_name: str,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, torch.Tensor]], List[torch.Tensor], List[Dict[str, Any]]]:
    raw_batches: List[Dict[str, Any]] = []
    memory_views: List[Dict[str, torch.Tensor]] = []
    memory_gt_poses: List[torch.Tensor] = []
    loaded_view_records: List[Dict[str, Any]] = []
    flat_view_lookup = _build_flat_view_lookup(dataset)

    for outer_i, idx in enumerate(memory_indices):
        if len(raw_batches) >= probe_memory_views:
            break
        raw_data = dataset[idx]

        if dataset_loader == "ace" and isinstance(raw_data, tuple):
            views = [convert_ace_tuple_to_dict(raw_data, getattr(dataset, "root_dir", ""))]
        elif isinstance(raw_data, list):
            views = raw_data
        else:
            views = [raw_data]

        for inner_j, view in enumerate(views):
            if len(raw_batches) >= probe_memory_views:
                break
            actual_flat_idx = _lookup_actual_flat_idx(view, flat_view_lookup, scene_name)
            record = {
                "view_idx": len(raw_batches),
                "outer_index": int(outer_i),
                "inner_index": int(inner_j),
                "selected_outer_fps_idx": int(idx),
                "actual_flat_idx": int(actual_flat_idx) if actual_flat_idx is not None else None,
                "source": _view_source_name(view),
                "label": _first_scalar_str(view.get("label")) if isinstance(view, dict) else None,
            }
            loaded_view_records.append(record)
            raw_batches.append(view)
            processed = prepare_batch_input(view, device)
            memory_views.append(processed)
            if "camera_poses" in processed:
                memory_gt_poses.append(processed["camera_poses"])
            else:
                raise ValueError(f"View {record['view_idx']} has no camera_poses; Q1a requires GT poses.")

    return raw_batches, memory_views, memory_gt_poses, loaded_view_records


def _reorder_for_reference(
    reference_view_id: int,
    memory_views: Sequence[Dict[str, torch.Tensor]],
    memory_gt_poses: Sequence[torch.Tensor],
    loaded_view_records: Sequence[Dict[str, Any]],
) -> Tuple[List[Dict[str, torch.Tensor]], List[torch.Tensor], List[Dict[str, Any]]]:
    ref_pos = None
    for i, rec in enumerate(loaded_view_records):
        actual_idx = rec.get("actual_flat_idx")
        if actual_idx == reference_view_id:
            ref_pos = int(i)
            break
    if ref_pos is None:
        raise ValueError(f"Reference view id {reference_view_id} is not in the loaded selected set.")

    order = [ref_pos] + [i for i in range(len(loaded_view_records)) if i != ref_pos]
    re_views = [memory_views[i] for i in order]
    re_gt = [memory_gt_poses[i] for i in order]
    re_records: List[Dict[str, Any]] = []
    for new_i, old_i in enumerate(order):
        rec = dict(loaded_view_records[old_i])
        rec["view_idx_original"] = int(old_i)
        rec["view_idx_reordered"] = int(new_i)
        rec["is_reference"] = bool(new_i == 0)
        re_records.append(rec)
    return re_views, re_gt, re_records


def _summarize_metric(values: Sequence[float]) -> Dict[str, Any]:
    arr = np.asarray(list(values), dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"count": 0, "mean": None, "median": None, "q50": None, "q90": None, "max": None}
    return {
        "count": int(arr.size),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "q50": float(np.percentile(arr, 50)),
        "q90": float(np.percentile(arr, 90)),
        "max": float(np.max(arr)),
    }


def _json_safe(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return _json_safe(value.detach().cpu().item())
        return _json_safe(value.detach().cpu().tolist())
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        val = float(value)
        return val if np.isfinite(val) else None
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    return value


def _reference_candidate_stats(
    selected_view_ids: Sequence[int],
    centers: np.ndarray,
    covisibility: Optional[np.ndarray],
) -> List[Dict[str, Any]]:
    if covisibility is None or centers is None:
        return [{"view_id": int(v)} for v in selected_view_ids]

    covis = np.asarray(covisibility)
    covis_strength = np.maximum(covis, covis.T).astype(np.float64, copy=False)
    np.fill_diagonal(covis_strength, 0.0)
    selected = np.asarray([int(v) for v in selected_view_ids], dtype=np.int64)
    scene_center = np.asarray(centers, dtype=np.float64).mean(axis=0)
    rows: List[Dict[str, Any]] = []
    for idx in selected:
        row = covis_strength[int(idx)]
        selected_row = row[selected]
        selected_nonself = selected_row[selected != int(idx)]
        selected_positive = selected_nonself[np.isfinite(selected_nonself) & (selected_nonself > 0)]
        global_positive = row[np.isfinite(row) & (row > 0)]
        rows.append({
            "view_id": int(idx),
            "center_dist": float(np.linalg.norm(centers[int(idx)] - scene_center)),
            "selected_positive_links": int(selected_positive.size),
            "selected_positive_mean": float(selected_positive.mean()) if selected_positive.size else 0.0,
            "global_positive_links": int(global_positive.size),
            "global_positive_mean": float(global_positive.mean()) if global_positive.size else 0.0,
        })
    return rows


def _choose_reference_candidates(
    selected_view_ids: Sequence[int],
    centers: Optional[np.ndarray],
    covisibility: Optional[np.ndarray],
    *,
    explicit_reference_ids: Optional[Sequence[int]],
    num_candidates: int,
) -> List[int]:
    selected = [int(v) for v in selected_view_ids]
    if not selected:
        return []

    if explicit_reference_ids:
        explicit = [int(v) for v in explicit_reference_ids]
        missing = [v for v in explicit if v not in selected]
        if missing:
            preview = ", ".join(str(v) for v in selected[:10])
            raise ValueError(
                "Explicit reference ids must belong to the current selected set. "
                f"Missing: {missing}. Current selected set has {len(selected)} views; "
                f"first views: [{preview}]"
            )
        ordered = []
        seen = set()
        for v in explicit:
            if v not in seen:
                ordered.append(v)
                seen.add(v)
        return ordered

    current_ref = int(selected[0])
    if centers is None or covisibility is None:
        return selected[: max(2, min(int(num_candidates), len(selected)))]

    rows = _reference_candidate_stats(selected, centers, covisibility)
    rows_sorted = sorted(
        rows,
        key=lambda r: (
            -int(r["selected_positive_links"]),
            -float(r["selected_positive_mean"]),
            -int(r["global_positive_links"]),
            -float(r["global_positive_mean"]),
            float(r["center_dist"]),
        ),
    )
    ordered = [current_ref]
    seen = {current_ref}
    for row in rows_sorted:
        view_id = int(row["view_id"])
        if view_id in seen:
            continue
        ordered.append(view_id)
        seen.add(view_id)
        if len(ordered) >= max(2, int(num_candidates)):
            break
    return ordered


def _compare_reference_runs(
    reports: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    pairwise: List[Dict[str, Any]] = []
    for i in range(len(reports)):
        for j in range(i + 1, len(reports)):
            rep_a = reports[i]
            rep_b = reports[j]
            ref_a = int(rep_a["reference_view_id"])
            ref_b = int(rep_b["reference_view_id"])
            per_view_a = {int(v["view_id"]): v for v in rep_a["per_view"]}
            per_view_b = {int(v["view_id"]): v for v in rep_b["per_view"]}
            common = sorted(set(per_view_a).intersection(per_view_b) - {ref_a, ref_b})
            trans_diffs = []
            rot_diffs = []
            for view_id in common:
                trans_diffs.append(abs(float(per_view_a[view_id]["trans_error_m"]) - float(per_view_b[view_id]["trans_error_m"])))
                rot_diffs.append(abs(float(per_view_a[view_id]["rot_error_deg"]) - float(per_view_b[view_id]["rot_error_deg"])))
            pairwise.append({
                "reference_a": ref_a,
                "reference_b": ref_b,
                "common_non_reference_views": [int(v) for v in common],
                "translation_abs_diff": _summarize_metric(trans_diffs),
                "rotation_abs_diff": _summarize_metric(rot_diffs),
            })
    return pairwise


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Q1a reference-swap diagnostic on one selected memory set.")
    parser.add_argument("dataset_path", type=str, help="Dataset root or scene path.")
    parser.add_argument("--dataset_type", type=str, default="auto")
    parser.add_argument("--scene_name", type=str, default=None)
    parser.add_argument("--dataset_loader", type=str, default="wai", choices=["wai", "ace"])
    parser.add_argument("--n_memory", type=int, default=40)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--wai_view_mode", type=str, default="anchor_support")
    parser.add_argument("--dataset_transform", type=str, default="imgnorm")
    parser.add_argument("--dataset_data_norm_type", type=str, default="dinov2")
    parser.add_argument("--dataset_aug_crop", type=int, default=0)
    parser.add_argument("--anchor_support_alpha", type=float, default=1.0)
    parser.add_argument("--anchor_support_eps", type=float, default=1e-6)
    parser.add_argument("--anchor_support_tau", type=float, default=-1.0)
    parser.add_argument("--covis_ref_lambda", type=float, default=0.0)
    parser.add_argument("--covis_anchor_count", type=int, default=0)
    parser.add_argument("--covis_support_per_anchor", type=int, default=3)
    parser.add_argument("--covis_support_tau", type=float, default=1e-4)
    parser.add_argument("--covis_support_min_neighbors", type=int, default=3)
    parser.add_argument("--covis_safe_dist_to_ref", type=float, default=3.0)
    parser.add_argument("--covis_far_view_budget", type=int, default=8)
    parser.add_argument("--covis_far_anchor_budget", type=int, default=2)
    parser.add_argument("--covis_coverage_beta", type=float, default=1.0)
    parser.add_argument("--covis_candidate_pool_ratio", type=float, default=1.0)
    parser.add_argument("--covis_ma_safe_asb", type=_strtobool, default=False)
    parser.add_argument("--covis_ma_safe_pool_ratio", type=float, default=3.0)
    parser.add_argument("--covis_native_group_asb", type=_strtobool, default=False)
    parser.add_argument("--covis_native_anchor_candidates", type=int, default=64)
    parser.add_argument("--covis_adaptive_asb", type=_strtobool, default=False)
    parser.add_argument("--covis_disable_adaptive_scene_stats", type=_strtobool, default=False)
    parser.add_argument("--covis_adaptive_view_count", type=_strtobool, default=False)
    parser.add_argument("--covis_adaptive_min_views", type=int, default=28)
    parser.add_argument("--covis_ref_dist_max_limit", type=float, default=4.2)
    parser.add_argument("--covis_ref_dist_mean_limit", type=float, default=2.7)
    parser.add_argument("--covis_min_coverage_gain", type=float, default=0.01)
    parser.add_argument("--covis_gain_patience", type=int, default=2)
    parser.add_argument("--covis_target_coverage_mean", type=float, default=0.55)
    parser.add_argument("--covis_target_coverage_max", type=float, default=2.0)
    parser.add_argument("--pose_eval_translation_ok_m", type=float, default=0.1)
    parser.add_argument("--model_config", type=str, default="default")
    parser.add_argument("--model_checkpoint", type=str, default="/mnt/storage/xwh/checkpoints/facebook_map-anything.pth")
    parser.add_argument("--dinov2_checkpoint", type=str, default="/home/xwh/project/ace_depth/checkpoints/dinov2_vitl14_pretrain.pth")
    parser.add_argument("--num_reference_candidates", type=int, default=4)
    parser.add_argument(
        "--reference_indices",
        type=str,
        default=None,
        help="Comma-separated actual flat view ids from the selected set. If omitted, auto-pick candidates.",
    )
    parser.add_argument("--output_path", type=str, default=None)
    return parser


def run_reference_swap_diagnostic(args: argparse.Namespace) -> Path:
    scene_name = _resolve_scene_name(args)
    dataset_type = args.dataset_type
    if dataset_type == "auto":
        dataset_type = detect_dataset_type(args.dataset_path)
    output_path = _resolve_output_path(args, scene_name)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    dataset_num_views = _dataset_num_views(args)
    wai_view_mode = canonicalize_wai_view_mode(args.wai_view_mode)

    print(f"[Q1a] Dataset: {args.dataset_path}")
    print(f"[Q1a] Scene: {scene_name} (type={dataset_type}, loader={args.dataset_loader})")
    print(f"[Q1a] Selection mode: {wai_view_mode}, n_memory={args.n_memory}, dataset_num_views={dataset_num_views}")
    print(f"[Q1a] Output: {output_path}")

    dataset = load_dataset(
        args.dataset_path,
        dataset_type=dataset_type,
        scene_name=scene_name,
        n_views=dataset_num_views,
        dataset_loader=args.dataset_loader,
        dataset_transform=args.dataset_transform,
        dataset_data_norm_type=args.dataset_data_norm_type,
        dataset_aug_crop=args.dataset_aug_crop,
    )

    memory_indices, scene_center, memory_index_groups = select_memory_views(
        dataset,
        int(args.n_memory),
        dataset_path=args.dataset_path,
        scene_name=scene_name,
        selection_mode=wai_view_mode,
        covis_alpha=args.anchor_support_alpha,
        covis_eps=args.anchor_support_eps,
        covis_tau=args.anchor_support_tau,
        covis_ref_lambda=args.covis_ref_lambda,
        covis_anchor_count=args.covis_anchor_count,
        covis_support_per_anchor=args.covis_support_per_anchor,
        covis_support_tau=args.covis_support_tau,
        covis_support_min_neighbors=args.covis_support_min_neighbors,
        covis_safe_dist_to_ref=args.covis_safe_dist_to_ref,
        covis_far_view_budget=args.covis_far_view_budget,
        covis_far_anchor_budget=args.covis_far_anchor_budget,
        covis_coverage_beta=args.covis_coverage_beta,
        covis_candidate_pool_ratio=args.covis_candidate_pool_ratio,
        covis_ma_safe_asb=bool(args.covis_ma_safe_asb),
        covis_ma_safe_pool_ratio=args.covis_ma_safe_pool_ratio,
        covis_native_group_asb=bool(args.covis_native_group_asb),
        covis_native_anchor_candidates=args.covis_native_anchor_candidates,
        covis_adaptive_asb=bool(args.covis_adaptive_asb),
        covis_adaptive_scene_stats=not bool(args.covis_disable_adaptive_scene_stats),
        covis_adaptive_view_count=bool(args.covis_adaptive_view_count),
        covis_adaptive_allow_early_stop=True,
        covis_adaptive_min_views=args.covis_adaptive_min_views,
        covis_ref_dist_max_limit=args.covis_ref_dist_max_limit,
        covis_ref_dist_mean_limit=args.covis_ref_dist_mean_limit,
        covis_min_coverage_gain=args.covis_min_coverage_gain,
        covis_gain_patience=args.covis_gain_patience,
        covis_target_coverage_mean=args.covis_target_coverage_mean,
        covis_target_coverage_max=args.covis_target_coverage_max,
    )

    raw_batches, memory_views, memory_gt_poses, loaded_view_records = _load_selected_views(
        dataset,
        memory_indices,
        probe_memory_views=len(memory_indices),
        device=device,
        dataset_loader=args.dataset_loader,
        scene_name=scene_name,
    )
    del raw_batches  # Q1a only needs model-ready views and GT poses.

    actual_loaded_view_ids = [
        int(rec["actual_flat_idx"]) if rec.get("actual_flat_idx") is not None else int(rec["selected_outer_fps_idx"])
        for rec in loaded_view_records
    ]
    if len(memory_views) != len(memory_gt_poses) or len(memory_views) != len(loaded_view_records):
        raise ValueError(
            f"Loaded view cardinality mismatch: views={len(memory_views)}, gt={len(memory_gt_poses)}, "
            f"records={len(loaded_view_records)}"
        )

    centers = _extract_centers_from_flat_wai_dataset(dataset)
    if centers is None:
        centers = _extract_centers_from_wai_scene_meta(args.dataset_path, scene_name)
    if centers is None:
        centers = _extract_centers_from_pose_dir(args.dataset_path, scene_name=scene_name)
    if centers is None:
        centers = _extract_centers_from_dataset_items(dataset)
    covisibility = _load_wai_pairwise_covisibility(args.dataset_path, scene_name, n_expected=len(centers)) if centers is not None else None

    explicit_reference_ids = None
    if args.reference_indices:
        explicit_reference_ids = [int(tok.strip()) for tok in str(args.reference_indices).split(",") if tok.strip()]
    reference_candidates = _choose_reference_candidates(
        actual_loaded_view_ids,
        centers,
        covisibility,
        explicit_reference_ids=explicit_reference_ids,
        num_candidates=max(2, int(args.num_reference_candidates)),
    )

    extractor = MapAnythingExtractor(
        "mapanything",
        args.model_config,
        args.model_checkpoint,
        device=str(device),
        dinov2_checkpoint=args.dinov2_checkpoint,
    )

    reference_reports: List[Dict[str, Any]] = []
    for ref_view_id in reference_candidates:
        reordered_views, reordered_gt, reordered_records = _reorder_for_reference(
            ref_view_id,
            memory_views,
            memory_gt_poses,
            loaded_view_records,
        )
        print(f"[Q1a] Running infer with reference view {ref_view_id} ({len(reordered_views)} views)")
        with torch.no_grad():
            predictions = extractor.model.infer(
                reordered_views,
                memory_efficient_inference=True,
                use_amp=False,
                ignore_depth_inputs=False,
                ignore_pose_inputs=False,
                ignore_calibration_inputs=False,
            )
        pose_ok, trans_errors, rot_errors = print_pose_prediction_vs_gt(
            reordered_gt,
            predictions,
            translation_ok_m=float(args.pose_eval_translation_ok_m),
            return_errors=True,
        )
        per_view = []
        for rec, t_err, r_err in zip(reordered_records, trans_errors, rot_errors):
            per_view.append({
                "view_id": int(rec["actual_flat_idx"]) if rec.get("actual_flat_idx") is not None else int(rec["selected_outer_fps_idx"]),
                "source": rec.get("source"),
                "batch_position": int(rec["view_idx_reordered"]),
                "is_reference": bool(rec["is_reference"]),
                "trans_error_m": float(t_err),
                "rot_error_deg": float(r_err),
            })
        reference_reports.append({
            "reference_view_id": int(ref_view_id),
            "pose_eval_ok": bool(pose_ok),
            "translation_summary": _summarize_metric(trans_errors),
            "rotation_summary": _summarize_metric(rot_errors),
            "translation_distribution": _error_distribution_summary(trans_errors),
            "rotation_distribution": _error_distribution_summary(rot_errors),
            "per_view": per_view,
        })
        if hasattr(extractor.model, "get_info_sharing_intermediate_features"):
            extractor.model.get_info_sharing_intermediate_features(clear=True)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    pairwise = _compare_reference_runs(reference_reports)
    candidate_stats = _reference_candidate_stats(actual_loaded_view_ids, centers, covisibility)

    report = {
        "schema_version": "reference_swap_diagnostic_v1",
        "q1_phase": "Q1a",
        "scene_name": scene_name,
        "dataset_path": str(args.dataset_path),
        "dataset_type": dataset_type,
        "dataset_loader": args.dataset_loader,
        "selection": {
            "mode": wai_view_mode,
            "selected_view_ids": [int(v) for v in actual_loaded_view_ids],
            "selected_outer_indices": [int(v) for v in memory_indices],
            "memory_index_groups": [[int(v) for v in group] for group in memory_index_groups],
            "initial_reference_index": int(actual_loaded_view_ids[0]) if actual_loaded_view_ids else None,
            "scene_center": scene_center.tolist() if isinstance(scene_center, np.ndarray) else None,
        },
        "reference_candidates": [int(v) for v in reference_candidates],
        "reference_candidate_stats": candidate_stats,
        "pose_eval_translation_ok_m": float(args.pose_eval_translation_ok_m),
        "reference_runs": reference_reports,
        "pairwise_reference_drift": pairwise,
        "config": {
            "dataset_transform": args.dataset_transform,
            "dataset_data_norm_type": args.dataset_data_norm_type,
            "dataset_aug_crop": int(args.dataset_aug_crop),
            "anchor_support_alpha": float(args.anchor_support_alpha),
            "anchor_support_eps": float(args.anchor_support_eps),
            "anchor_support_tau": float(args.anchor_support_tau),
            "covis_ref_lambda": float(args.covis_ref_lambda),
            "covis_anchor_count": int(args.covis_anchor_count),
            "covis_support_per_anchor": int(args.covis_support_per_anchor),
            "covis_support_tau": float(args.covis_support_tau),
            "covis_support_min_neighbors": int(args.covis_support_min_neighbors),
            "covis_safe_dist_to_ref": float(args.covis_safe_dist_to_ref),
            "covis_far_view_budget": int(args.covis_far_view_budget),
            "covis_far_anchor_budget": int(args.covis_far_anchor_budget),
            "covis_coverage_beta": float(args.covis_coverage_beta),
            "covis_candidate_pool_ratio": float(args.covis_candidate_pool_ratio),
            "covis_ma_safe_asb": bool(args.covis_ma_safe_asb),
            "covis_ma_safe_pool_ratio": float(args.covis_ma_safe_pool_ratio),
            "covis_native_group_asb": bool(args.covis_native_group_asb),
            "covis_native_anchor_candidates": int(args.covis_native_anchor_candidates),
            "covis_adaptive_asb": bool(args.covis_adaptive_asb),
            "covis_adaptive_scene_stats": not bool(args.covis_disable_adaptive_scene_stats),
            "covis_adaptive_view_count": bool(args.covis_adaptive_view_count),
            "covis_adaptive_min_views": int(args.covis_adaptive_min_views),
            "covis_ref_dist_max_limit": float(args.covis_ref_dist_max_limit),
            "covis_ref_dist_mean_limit": float(args.covis_ref_dist_mean_limit),
            "covis_min_coverage_gain": float(args.covis_min_coverage_gain),
            "covis_gain_patience": int(args.covis_gain_patience),
            "covis_target_coverage_mean": float(args.covis_target_coverage_mean),
            "covis_target_coverage_max": float(args.covis_target_coverage_max),
        },
    }

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(_json_safe(report), f, ensure_ascii=False, indent=2)
    print(f"[Q1a] Report saved: {output_path}")
    return output_path


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    run_reference_swap_diagnostic(args)


if __name__ == "__main__":
    main()
