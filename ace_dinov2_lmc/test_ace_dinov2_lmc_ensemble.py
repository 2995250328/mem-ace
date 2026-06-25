#!/usr/bin/env python3
# Joint evaluation for multiple ACE DINOv2 + LMC checkpoints.
# First version: run each checkpoint independently on every query frame and
# select the final pose by DSAC inlier count.

import argparse
import json
import logging
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))


def setup_cuda_environment():
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--device", type=str, default="cuda:0")
    pre_args, _ = pre_parser.parse_known_args()
    device_str = pre_args.device
    if "cuda" in device_str and ":" in device_str:
        os.environ["CUDA_VISIBLE_DEVICES"] = device_str.split(":")[-1]


setup_cuda_environment()

import cv2
import numpy as np
import torch
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader

import dsacstar
from ace_network_dinov2 import Regressor
from dataset_dinov2 import CamLocDatasetDINOv2

logging.basicConfig(level=logging.INFO)
_logger = logging.getLogger(__name__)
DATA_ROOT = Path(os.environ.get("ACE_DATA_ROOT", "/home/xwh/data"))


def _torch_load_trusted_checkpoint(path, *, map_location="cpu"):
    """Load a local project checkpoint while making pickle semantics explicit."""
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _to_tensor(value: Any, *, dtype=torch.float32):
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().to(dtype=dtype)
    return torch.as_tensor(value, dtype=dtype)


def _build_reference_eval_state(lmc_config: Dict[str, Any], bank_data: Dict[str, Any] | None) -> Dict[str, Any]:
    contract_mode = str(lmc_config.get("memory_contract_mode", "C0")).upper()
    state: Dict[str, Any] = {
        "enabled": contract_mode == "C1",
        "contract_mode": contract_mode,
        "output_space": str(
            lmc_config.get(
                "reference_output_space",
                lmc_config.get("memory_training_target", "points_world"),
            )
        ),
        "conditioning_reference": None,
        "normalization_ref": None,
    }
    if contract_mode != "C1":
        return state

    conditioning_reference = lmc_config.get("conditioning_reference")
    if (
        (not isinstance(conditioning_reference, dict) or conditioning_reference.get("T_ref_c2w_world") is None)
        and isinstance(bank_data, dict)
    ):
        conditioning_reference = bank_data.get("conditioning_reference")
    if not isinstance(conditioning_reference, dict):
        raise ValueError("[Eval] contract_mode=C1 requires conditioning_reference in checkpoint or memory file.")

    normalization_ref = lmc_config.get("normalization_ref")
    if (
        (not isinstance(normalization_ref, dict) or normalization_ref.get("mu_ref") is None)
        and isinstance(bank_data, dict)
    ):
        normalization_ref = bank_data.get("normalization_ref")
    if not isinstance(normalization_ref, dict):
        normalization_ref = {}

    T_ref_c2w_world = _to_tensor(conditioning_reference.get("T_ref_c2w_world"))
    if T_ref_c2w_world is None or tuple(T_ref_c2w_world.shape) != (4, 4):
        raise ValueError("[Eval] C1 checkpoint is missing a valid T_ref_c2w_world recovery transform.")

    mu_ref = _to_tensor(normalization_ref.get("mu_ref"))
    sigma_ref = normalization_ref.get("sigma_ref")
    ref_norm_alpha = lmc_config.get("c1_ref_norm_alpha", normalization_ref.get("alpha", 1.0))
    if sigma_ref is not None:
        sigma_ref = float(torch.as_tensor(sigma_ref, dtype=torch.float32).item())
    ref_norm_alpha = float(torch.as_tensor(ref_norm_alpha, dtype=torch.float32).item())

    state["conditioning_reference"] = {
        "reference_index": conditioning_reference.get("reference_index"),
        "T_ref_c2w_world": T_ref_c2w_world,
        "T_world_to_ref": _to_tensor(conditioning_reference.get("T_world_to_ref")),
    }
    state["normalization_ref"] = {
        "mu_ref": mu_ref,
        "sigma_ref": sigma_ref,
        "alpha": ref_norm_alpha,
    }
    return state


def _recover_pred_scene_to_world(scene_coordinates_B3HW: torch.Tensor, ref_state: Dict[str, Any]) -> torch.Tensor:
    if not ref_state.get("enabled", False):
        return scene_coordinates_B3HW

    pred_world = scene_coordinates_B3HW
    if ref_state.get("output_space") == "points_ref_norm":
        mu_ref = ref_state["normalization_ref"].get("mu_ref")
        sigma_ref = ref_state["normalization_ref"].get("sigma_ref")
        ref_norm_alpha = ref_state["normalization_ref"].get("alpha", 1.0)
        if mu_ref is None or sigma_ref is None:
            raise ValueError("[Eval] output_space=points_ref_norm but normalization_ref is incomplete.")
        pred_world = pred_world * (float(sigma_ref) / float(ref_norm_alpha)) + mu_ref.view(1, 3, 1, 1)

    T_ref_c2w_world = ref_state["conditioning_reference"]["T_ref_c2w_world"].to(
        dtype=pred_world.dtype, device=pred_world.device
    )
    R_ref = T_ref_c2w_world[:3, :3]
    C_ref = T_ref_c2w_world[:3, 3]
    pred_world = torch.einsum("ij,bjhw->bihw", R_ref, pred_world) + C_ref.view(1, 3, 1, 1)
    return pred_world


def _strtobool(x):
    if isinstance(x, bool):
        return x
    v = str(x).strip().lower()
    if v in ("1", "true", "yes", "on"):
        return True
    if v in ("0", "false", "no", "off"):
        return False
    raise ValueError(f"Invalid boolean value: {x!r}")


def _is_lmc_checkpoint(checkpoint):
    return isinstance(checkpoint, dict) and "lmc_config" in checkpoint


@dataclass
class EvalBundle:
    checkpoint_path: Path
    tag: str
    network: Any
    is_lmc: bool
    lmc_config: Optional[Dict[str, Any]]
    fusion: Optional[Any]
    memory_dict: Optional[Dict[str, torch.Tensor]]
    compressor_out_cached: Optional[Any]
    reference_eval_state: Dict[str, Any]


def _safe_checkpoint_tag(path: Path) -> str:
    stem = path.stem
    return "".join(c if (c.isalnum() or c in "._-") else "_" for c in stem).strip("._-") or "ckpt"


def _unsqueeze0_to_device(value: Any, device: torch.device):
    if value is None:
        return None
    if not isinstance(value, torch.Tensor):
        value = torch.tensor(value, device=device)
    if value.dim() in (1, 2):
        return value.unsqueeze(0)
    return value


def _load_bundle(
    checkpoint_path: Path,
    *,
    dinov2_path: Path,
    device: torch.device,
) -> EvalBundle:
    checkpoint = _torch_load_trusted_checkpoint(checkpoint_path, map_location="cpu")
    is_lmc = _is_lmc_checkpoint(checkpoint)

    if is_lmc:
        lmc_config = checkpoint["lmc_config"]
        head_state_dict = checkpoint["head_state_dict"]
        memory_path = lmc_config.get("memory_path")
    else:
        lmc_config = None
        head_state_dict = checkpoint
        memory_path = None

    network = Regressor.create_from_split_state_dict(
        dinov2_path=dinov2_path,
        head_state_dict=head_state_dict,
    )
    network = network.to(device)
    network.eval()

    fusion = None
    memory_dict = None
    compressor_out_cached = None
    reference_eval_state = {"enabled": False, "contract_mode": "C0", "output_space": "points_world"}

    if is_lmc and memory_path is not None:
        from ace_compressor import GeoLMC
        from ace_fusion import LMCFeatureFusion
        from utils_lmc import load_memory_features

        compress_dim = lmc_config.get("compress_dim", 1024)
        num_layers = lmc_config.get("num_layers", 1)
        backbone_feature_dim = 1024

        compressor = GeoLMC(
            input_dim=compress_dim,
            compress_dim=compress_dim,
            num_latent_tokens=lmc_config.get("num_latent_tokens", 64),
            mode=lmc_config.get("lmc_mode", "global"),
            num_layers=num_layers,
            use_scale_token=lmc_config.get("use_scale_token", True),
            scale_token_dim=lmc_config.get("scale_token_dim", 1024),
            num_attn_layers=lmc_config.get("num_attn_layers", 2),
            pe_normalize_input=lmc_config.get("pe_normalize_input", False),
            pe_scale_mode=lmc_config.get(
                "lmc_compressor_pe_scale_mode",
                "std" if lmc_config.get("pe_normalize_input", False) else "raw",
            ),
            pe_scene_scale=lmc_config.get(
                "lmc_compressor_pe_scene_scale",
                lmc_config.get("lmc_fusion_scene_scale", 1.0),
            ),
            fps_start_policy=lmc_config.get("lmc_fps_start_policy", "farthest_from_center"),
            key_slice_idx=lmc_config.get("lmc_key_slice_idx", None),
            key_feature_mode=lmc_config.get("lmc_key_feature_mode", "slice"),
            feature_hierarchy_mode=lmc_config.get(
                "lmc_feature_hierarchy_mode", "selected_key_concat_value"
            ),
            level_merge_mode=lmc_config.get("lmc_level_merge_mode", "softmax_gate"),
            level_merge_init=lmc_config.get("lmc_level_merge_init", "uniform"),
            level_proj_shared=lmc_config.get("lmc_level_proj_shared", False),
            level_cross_attn_shared=lmc_config.get("lmc_level_cross_attn_shared", True),
            level_gate_entropy_weight=lmc_config.get("lmc_level_gate_entropy_weight", 0.0),
            level_token_gate=lmc_config.get("lmc_level_token_gate", False),
            level_anchor_residual_gamma_init=lmc_config.get("lmc_level_anchor_residual_gamma_init", 0.0),
            geo_bias_mode=lmc_config.get("geo_bias_mode", "legacy"),
            geo_bias_rbf_scales=lmc_config.get("geo_bias_rbf_scales", [0.25, 0.5, 1.0, 2.0, 4.0]),
            geo_bias_rbf_alpha_init=lmc_config.get("geo_bias_rbf_alpha_init", 0.0),
            geo_bias_rbf_learn_weights=lmc_config.get("geo_bias_rbf_learn_weights", True),
            geo_bias_rbf_per_head=lmc_config.get("geo_bias_rbf_per_head", False),
            pos_encoding_mode=lmc_config.get("pos_encoding_mode", "fourier_legacy"),
            pos_fourier_v2_scales=lmc_config.get("pos_fourier_v2_scales", [1.0, 2.0, 4.0, 8.0, 16.0]),
            pos_fourier_coord_norm=lmc_config.get("pos_fourier_coord_norm", "scene_radius"),
            pos_fourier_radius=lmc_config.get("pos_fourier_radius", 4.0),
            pos_fourier_learnable_scale=lmc_config.get("pos_fourier_learnable_scale", False),
            pos_fourier_residual_gate_init=lmc_config.get("pos_fourier_residual_gate_init", 0.0),
            point_rope_coord_norm=lmc_config.get("point_rope_coord_norm", "scene_radius"),
            point_rope_radius=lmc_config.get("point_rope_radius", 4.0),
            point_rope_radius_policy=lmc_config.get("point_rope_radius_policy", "fixed"),
            point_rope_mixed_memory_ratio=lmc_config.get("point_rope_mixed_memory_ratio", 0.5),
            point_rope_seed_pe=lmc_config.get("point_rope_seed_pe", "fourier_legacy"),
            point_rope_base=lmc_config.get("point_rope_base", 10000.0),
            point_rope_axes=lmc_config.get("point_rope_axes", "xyz_split"),
            point_rope_apply_to=lmc_config.get("point_rope_apply_to", "qk"),
            geo_bias_crpb_dim=lmc_config.get("geo_bias_crpb_dim", 32),
            geo_bias_crpb_input=lmc_config.get("geo_bias_crpb_input", "delta_dist_log"),
            geo_bias_crpb_radius=lmc_config.get("geo_bias_crpb_radius", 4.0),
            geo_bias_crpb_per_head=lmc_config.get("geo_bias_crpb_per_head", False),
            geo_bias_crpb_zero_init=lmc_config.get("geo_bias_crpb_zero_init", True),
        ).to(device)
        compressor.load_state_dict(checkpoint["compressor_state_dict"])
        compressor.eval()

        fusion = LMCFeatureFusion(
            feature_dim=backbone_feature_dim,
            mode=lmc_config.get("lmc_mode", "global"),
            query_feature_dim=backbone_feature_dim,
            memory_feature_dim=compress_dim,
            fusion_geometry_mode=lmc_config.get("lmc_fusion_geometry_mode", "value_only_raw"),
            fusion_scene_scale=lmc_config.get("lmc_fusion_scene_scale", 1.0),
            fusion_key_geo_init=lmc_config.get("lmc_fusion_key_geo_init", 0.0),
            fusion_refinement_mode=lmc_config.get("lmc_fusion_refinement_mode", "single"),
            fusion_cascade_layers=lmc_config.get("lmc_fusion_cascade_layers", 4),
            fusion_assembly_mode=lmc_config.get("lmc_fusion_assembly_mode", "concat_mlp"),
            fusion_assembly_gamma_init=lmc_config.get("lmc_fusion_assembly_gamma_init", 0.0),
            fusion_coord_prior_scale_init=lmc_config.get("lmc_fusion_coord_prior_scale_init", 0.10),
        ).to(device)
        fusion.load_state_dict(checkpoint["fusion_state_dict"])
        fusion.eval()

        _logger.info("[Ensemble] Loading memory for %s from %s", checkpoint_path.name, memory_path)
        c1_ref_norm_alpha = float(lmc_config.get("c1_ref_norm_alpha", 1.0) or 1.0)
        bank_data = load_memory_features(str(memory_path), device, c1_ref_norm_alpha=c1_ref_norm_alpha)
        reference_eval_state = _build_reference_eval_state(lmc_config, bank_data)

        memory_dict = {
            "pooled_points": _unsqueeze0_to_device(bank_data["pooled_points"], device),
            "pooled_features": _unsqueeze0_to_device(bank_data["pooled_features"], device),
            "scene_center": _unsqueeze0_to_device(bank_data.get("scene_center"), device),
        }
        if bank_data.get("all_scale_tokens") is not None:
            memory_dict["all_scale_tokens"] = _unsqueeze0_to_device(bank_data["all_scale_tokens"], device)

        with torch.no_grad():
            compressor_out_cached = compressor(memory_dict)

    return EvalBundle(
        checkpoint_path=checkpoint_path,
        tag=_safe_checkpoint_tag(checkpoint_path),
        network=network,
        is_lmc=is_lmc,
        lmc_config=lmc_config,
        fusion=fusion,
        memory_dict=memory_dict,
        compressor_out_cached=compressor_out_cached,
        reference_eval_state=reference_eval_state,
    )


def _predict_scene_coordinates(
    bundle: EvalBundle,
    base_features: torch.Tensor,
) -> torch.Tensor:
    features = base_features
    if bundle.fusion is not None and bundle.compressor_out_cached is not None:
        B, C, H, W = features.shape
        sc = bundle.memory_dict["scene_center"]
        if sc.shape[0] == 1 and B > 1:
            sc = sc.expand(B, -1)
        if B > 1:
            if isinstance(bundle.compressor_out_cached, dict):
                compressor_out_batch = {
                    k: v.expand(B, *v.shape[1:]) if v.dim() >= 2 and v.shape[0] == 1 else v
                    for k, v in bundle.compressor_out_cached.items()
                }
            else:
                z, p = bundle.compressor_out_cached
                z = z.expand(B, -1, -1) if z.shape[0] == 1 else z
                p = p.expand(B, -1, -1) if p.shape[0] == 1 else p
                compressor_out_batch = (z, p)
        else:
            compressor_out_batch = bundle.compressor_out_cached
        query = features.permute(0, 2, 3, 1).reshape(B, H * W, C)
        fused = bundle.fusion(query, compressor_out_batch, sc)
        features = fused.reshape(B, H, W, C).permute(0, 3, 1, 2)

    scene_coordinates_B3HW = bundle.network.get_scene_coordinates(features).float().cpu()
    if bundle.is_lmc and bundle.lmc_config is not None:
        if bundle.reference_eval_state.get("enabled", False):
            scene_coordinates_B3HW = _recover_pred_scene_to_world(
                scene_coordinates_B3HW, bundle.reference_eval_state
            )
        else:
            norm_mu = bundle.lmc_config.get("normalization_mu")
            norm_sigma = bundle.lmc_config.get("normalization_sigma")
            if norm_mu is not None and norm_sigma is not None:
                mu_t = torch.tensor(norm_mu, dtype=torch.float32).view(1, 3, 1, 1)
                scene_coordinates_B3HW = scene_coordinates_B3HW * float(norm_sigma) + mu_t
    return scene_coordinates_B3HW


def _evaluate_candidate(
    *,
    bundle: EvalBundle,
    scene_coordinates_3HW: torch.Tensor,
    gt_pose_44: torch.Tensor,
    intrinsics_33: torch.Tensor,
    hypotheses: int,
    threshold: float,
    inlieralpha: float,
    maxpixelerror: float,
) -> Dict[str, Any]:
    fx = intrinsics_33[0, 0].item()
    fy = intrinsics_33[1, 1].item()
    focal_length = (fx + fy) / 2.0
    ppX = intrinsics_33[0, 2].item()
    ppY = intrinsics_33[1, 2].item()

    out_pose = torch.zeros((4, 4))
    inlier_count = int(
        dsacstar.forward_rgb(
            scene_coordinates_3HW.unsqueeze(0),
            out_pose,
            hypotheses,
            threshold,
            focal_length,
            ppX,
            ppY,
            inlieralpha,
            maxpixelerror,
            bundle.network.OUTPUT_SUBSAMPLE,
        )
    )

    t_err = float(torch.norm(gt_pose_44[0:3, 3] - out_pose[0:3, 3]))
    gt_R = gt_pose_44[0:3, 0:3].numpy()
    out_R = out_pose[0:3, 0:3].numpy()
    r_err = np.matmul(out_R, np.transpose(gt_R))
    r_err = cv2.Rodrigues(r_err)[0]
    r_err = float(np.linalg.norm(r_err) * 180 / math.pi)

    out_pose_inv = out_pose.inverse()
    t = out_pose_inv[0:3, 3]
    rot, _ = cv2.Rodrigues(out_pose_inv[0:3, 0:3].numpy())
    angle = float(np.linalg.norm(rot))
    axis = rot / max(angle, 1e-12)
    q_w = math.cos(angle * 0.5)
    q_xyz = math.sin(angle * 0.5) * axis

    return {
        "tag": bundle.tag,
        "checkpoint_path": str(bundle.checkpoint_path),
        "inlier_count": inlier_count,
        "r_err_deg": r_err,
        "t_err_cm": t_err * 100.0,
        "pose": out_pose,
        "pose_inv_qt": [
            float(q_w),
            float(q_xyz[0].item()),
            float(q_xyz[1].item()),
            float(q_xyz[2].item()),
            float(t[0].item()),
            float(t[1].item()),
            float(t[2].item()),
        ],
    }


def _select_candidate(candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
    # First version: select by DSAC inlier count only; ties keep source order.
    best_idx = 0
    best_inliers = candidates[0]["inlier_count"]
    for idx, cand in enumerate(candidates[1:], start=1):
        if cand["inlier_count"] > best_inliers:
            best_idx = idx
            best_inliers = cand["inlier_count"]
    chosen = dict(candidates[best_idx])
    chosen["selected_index"] = best_idx
    return chosen


def run_ensemble_evaluation(opt):
    device = torch.device(opt.device)
    dinov2_path = Path(opt.dinov2_path)
    scene_path = Path(opt.scene)
    checkpoint_paths = [Path(p) for p in opt.networks]
    output_dir = Path(opt.output_dir) if opt.output_dir else checkpoint_paths[0].parent
    output_dir.mkdir(parents=True, exist_ok=True)

    image_resolution = int(getattr(opt, "image_resolution", 518))
    if image_resolution % 14 != 0:
        image_resolution = (image_resolution // 14) * 14

    bundles = [
        _load_bundle(ckpt, dinov2_path=dinov2_path, device=device)
        for ckpt in checkpoint_paths
    ]
    _logger.info(
        "[Ensemble] Loaded %d checkpoints: %s",
        len(bundles),
        ", ".join(bundle.tag for bundle in bundles),
    )

    testset = CamLocDatasetDINOv2(
        scene_path / "test",
        mode=0,
        use_half=False,
        image_height=image_resolution,
        augment=False,
    )
    testset_loader = DataLoader(testset, shuffle=False, num_workers=6)
    _logger.info("[Ensemble] Test images: %d", len(testset))

    session = getattr(opt, "session", "ensemble")
    scene_name = scene_path.name
    test_log_file = output_dir / f"test_{scene_name}_{session}.txt"
    pose_log_file = output_dir / f"poses_{scene_name}_{session}.txt"
    selection_log_file = output_dir / f"ensemble_selection_{scene_name}_{session}.jsonl"
    test_log = open(test_log_file, "w", 1)
    pose_log = open(pose_log_file, "w", 1)
    selection_log = open(selection_log_file, "w", 1, encoding="utf-8")

    avg_batch_time = 0.0
    num_batches = 0
    rErrs: List[float] = []
    tErrs: List[float] = []
    pct25_5 = pct10_5 = pct5 = pct2 = pct1 = 0
    selected_counter = {bundle.tag: 0 for bundle in bundles}

    with torch.no_grad():
        for image_B1HW, _, gt_pose_B44, _, intrinsics_B33, _, _, filenames in testset_loader:
            batch_start_time = time.time()
            image_B1HW = image_B1HW.to(device, non_blocking=True)

            with autocast(enabled=True):
                base_features = bundles[0].network.get_features(image_B1HW)
                predicted_scene_coords = [
                    _predict_scene_coordinates(bundle, base_features)
                    for bundle in bundles
                ]

            if isinstance(filenames, str):
                filenames = (filenames,)

            for frame_idx, (gt_pose_44, intrinsics_33, frame_path) in enumerate(
                zip(gt_pose_B44, intrinsics_B33, filenames)
            ):
                frame_name = Path(frame_path).name
                candidates = [
                    _evaluate_candidate(
                        bundle=bundle,
                        scene_coordinates_3HW=coords[frame_idx],
                        gt_pose_44=gt_pose_44,
                        intrinsics_33=intrinsics_33,
                        hypotheses=int(opt.hypotheses),
                        threshold=float(opt.threshold),
                        inlieralpha=float(opt.inlieralpha),
                        maxpixelerror=float(opt.maxpixelerror),
                    )
                    for bundle, coords in zip(bundles, predicted_scene_coords)
                ]
                chosen = _select_candidate(candidates)
                selected_counter[chosen["tag"]] += 1

                r_err = float(chosen["r_err_deg"])
                t_err_cm = float(chosen["t_err_cm"])
                rErrs.append(r_err)
                tErrs.append(t_err_cm)

                if r_err < 5 and t_err_cm < 25.0:
                    pct25_5 += 1
                if r_err < 5 and t_err_cm < 10.0:
                    pct10_5 += 1
                if r_err < 5 and t_err_cm < 5.0:
                    pct5 += 1
                if r_err < 2 and t_err_cm < 2.0:
                    pct2 += 1
                if r_err < 1 and t_err_cm < 1.0:
                    pct1 += 1

                if getattr(opt, "log_per_frame", False):
                    _logger.info(
                        "[Ensemble] %s -> %s | inliers=%d | rErr=%.2f deg | tErr=%.2f cm",
                        frame_name,
                        chosen["tag"],
                        chosen["inlier_count"],
                        r_err,
                        t_err_cm,
                    )

                q_w, q_x, q_y, q_z, t_x, t_y, t_z = chosen["pose_inv_qt"]
                pose_log.write(
                    f"{frame_name} {q_w} {q_x} {q_y} {q_z} "
                    f"{t_x} {t_y} {t_z} {r_err} {t_err_cm / 100.0} {chosen['inlier_count']} {chosen['tag']}\n"
                )

                selection_log.write(
                    json.dumps(
                        {
                            "frame": frame_name,
                            "selected_tag": chosen["tag"],
                            "selected_inlier_count": chosen["inlier_count"],
                            "candidates": [
                                {
                                    "tag": cand["tag"],
                                    "checkpoint_path": cand["checkpoint_path"],
                                    "inlier_count": cand["inlier_count"],
                                }
                                for cand in candidates
                            ],
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

            avg_batch_time += time.time() - batch_start_time
            num_batches += 1

    total_frames = len(rErrs)
    tErrs_sorted = sorted(tErrs)
    rErrs_sorted = sorted(rErrs)
    median_idx = total_frames // 2
    median_rErr = rErrs_sorted[median_idx]
    median_tErr = tErrs_sorted[median_idx]
    avg_time = avg_batch_time / max(num_batches, 1)

    pct25_5 = pct25_5 / total_frames * 100.0
    pct10_5 = pct10_5 / total_frames * 100.0
    pct5 = pct5 / total_frames * 100.0
    pct2 = pct2 / total_frames * 100.0
    pct1 = pct1 / total_frames * 100.0

    _logger.info("=" * 50)
    _logger.info("ENSEMBLE EVAL SUMMARY:")
    _logger.info("  Median Error: %.2f deg, %.2f cm", median_rErr, median_tErr)
    _logger.info(
        "  25cm/5deg: %.2f%% | 10cm/5deg: %.2f%% | 5cm/5deg: %.2f%% | 2cm/2deg: %.2f%% | 1cm/1deg: %.2f%%",
        pct25_5,
        pct10_5,
        pct5,
        pct2,
        pct1,
    )
    _logger.info("  Avg time: %.2f ms | Frames: %d", avg_time * 1000.0, total_frames)
    _logger.info("  Selected counts: %s", selected_counter)
    _logger.info("=" * 50)

    test_log.write(f"{median_rErr} {median_tErr} {avg_time}\n")
    test_log.close()
    pose_log.close()
    selection_log.close()

    eval_summary_file = output_dir / f"eval_summary_{scene_name}_{session}.txt"
    eval_summary_file.write_text(
        "\n".join(
            [
                f"# ACE DINOv2+LMC Ensemble Eval | {scene_name}",
                f"median_rotation_deg\t{median_rErr:.4f}",
                f"median_translation_cm\t{median_tErr:.4f}",
                f"accuracy_25cm5deg_pct\t{pct25_5:.2f}",
                f"accuracy_10cm5deg_pct\t{pct10_5:.2f}",
                f"accuracy_5cm5deg_pct\t{pct5:.2f}",
                f"accuracy_2cm2deg_pct\t{pct2:.2f}",
                f"accuracy_1cm1deg_pct\t{pct1:.2f}",
                f"avg_time_per_frame_ms\t{avg_time * 1000.0:.2f}",
                f"total_frames\t{total_frames}",
                f"selected_counter_json\t{json.dumps(selected_counter, ensure_ascii=False, sort_keys=True)}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    _logger.info("Eval summary written to: %s", eval_summary_file)

    return {
        "median_rErr": median_rErr,
        "median_tErr": median_tErr,
        "avg_time": avg_time,
        "pct25_5": pct25_5,
        "pct10_5": pct10_5,
        "pct5": pct5,
        "pct2": pct2,
        "pct1": pct1,
        "total_frames": total_frames,
        "selected_counter": selected_counter,
        "test_log_file": str(test_log_file),
        "pose_log_file": str(pose_log_file),
        "selection_log_file": str(selection_log_file),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Joint evaluation for multiple ACE DINOv2 + LMC checkpoints.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("scene", type=Path)
    parser.add_argument("networks", nargs="+", type=Path, help="Two or more checkpoints to ensemble.")
    parser.add_argument("--dinov2_path", type=Path, default=DATA_ROOT / "checkpoints" / "dinov2_vitl14_pretrain.pth")
    parser.add_argument("--output_dir", type=Path, default=None, help="Directory for joint-eval outputs.")
    parser.add_argument("--session", "-sid", default="ensemble")
    parser.add_argument("--image_resolution", type=int, default=518)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--hypotheses", "-hyps", type=int, default=64)
    parser.add_argument("--threshold", "-t", type=float, default=10)
    parser.add_argument("--inlieralpha", "-ia", type=float, default=100)
    parser.add_argument("--maxpixelerror", "-maxerrr", type=float, default=100)
    parser.add_argument("--log_per_frame", type=_strtobool, default=False)

    opt = parser.parse_args()
    if len(opt.networks) < 2:
        raise ValueError("Joint evaluation needs at least two checkpoints.")
    run_ensemble_evaluation(opt)
