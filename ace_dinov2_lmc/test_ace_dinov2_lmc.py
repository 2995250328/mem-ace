#!/usr/bin/env python3
# test_ace_dinov2_lmc.py
# Testing script for ACE DINOv2 with optional LMC.
# Auto-detects LMC vs vanilla checkpoint and adapts accordingly.

import argparse
import logging
import math
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict

sys.path.insert(0, str(Path(__file__).parent.parent))


def setup_cuda_environment():
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument('--device', type=str, default='cuda:0')
    pre_args, _ = pre_parser.parse_known_args()
    device_str = pre_args.device
    if 'cuda' in device_str and ':' in device_str:
        gpu_id = device_str.split(':')[-1]
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id


setup_cuda_environment()

import cv2
import numpy as np
import torch
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader

import dsacstar
from ace_network_dinov2 import Head, Regressor
from dataset_dinov2 import CamLocDatasetDINOv2
from dataset_wai_dinov2 import CamLocDatasetWAIDINOv2
from ace_network_ace import RegressorACE
from dataset_ace_fcn_lmc import CamLocDatasetACEFCNLMC
from ace_lmc_global_film import ACEGlobalFiLMHead
from ace_lmc_global_residual import ACEGlobalResidualHead
from glace_backend import (
    GLACEDecoderFeatureResidualAdapter,
    build_glace_camloc_dataset,
    create_glace_regressor_from_split_state_dict,
)

logging.basicConfig(level=logging.INFO)
_logger = logging.getLogger(__name__)
DATA_ROOT = Path(os.environ.get("ACE_DATA_ROOT", "/home/xwh/data"))
_WARNED_DSACSTAR_SEEDING = False


def _torch_load_trusted_checkpoint(path, *, map_location='cpu'):
    """Load a local project checkpoint while making pickle semantics explicit."""
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _normalize_device_for_visible_cuda(device_str):
    """Map a physical cuda:N request to logical cuda:0 after CUDA_VISIBLE_DEVICES narrowing."""
    device_str = str(device_str or "cuda:0")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if device_str.startswith("cuda:") and visible:
        visible_ids = [x.strip() for x in visible.split(",") if x.strip()]
        if len(visible_ids) == 1 and device_str != "cuda:0":
            _logger.info(
                "[Eval] Normalizing device %s -> cuda:0 because CUDA_VISIBLE_DEVICES=%s.",
                device_str,
                visible,
            )
            return "cuda:0"
    return device_str


def _configure_eval_determinism(seed: int):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _seed_eval_worker(worker_id: int):
    worker_seed = (torch.initial_seed() + worker_id) % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def _maybe_set_dsacstar_seed(seed: int) -> bool:
    global _WARNED_DSACSTAR_SEEDING
    if hasattr(dsacstar, 'set_seed'):
        dsacstar.set_seed(int(seed))
        return True
    if not _WARNED_DSACSTAR_SEEDING:
        _logger.warning(
            "[Eval] dsacstar extension does not expose set_seed(); RANSAC is still stochastic until ../dsacstar is rebuilt."
        )
        _WARNED_DSACSTAR_SEEDING = True
    return False


def _to_tensor(value: Any, *, dtype=torch.float32):
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().to(dtype=dtype)
    return torch.as_tensor(value, dtype=dtype)


def _build_reference_eval_state(lmc_config: Dict[str, Any], bank_data: Dict[str, Any] | None) -> Dict[str, Any]:
    """Resolve C1 recovery metadata from checkpoint config, with memory-file fallback."""
    contract_mode = str(lmc_config.get("memory_contract_mode", "C0")).upper()
    state: Dict[str, Any] = {
        "enabled": contract_mode == "C1",
        "contract_mode": contract_mode,
        "output_space": str(lmc_config.get("reference_output_space", lmc_config.get("memory_training_target", "points_world"))),
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
    """Convert C1 outputs from reference space back to dataset world coordinates."""
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
    if v in ('1', 'true', 'yes', 'on'):
        return True
    if v in ('0', 'false', 'no', 'off'):
        return False
    raise ValueError(f'Invalid boolean value: {x!r}')


def _is_lmc_checkpoint(checkpoint):
    """Check if a checkpoint contains LMC config."""
    return isinstance(checkpoint, dict) and 'lmc_config' in checkpoint


def _format_lmc_runtime_extra_stats(stats: Dict[str, Any]) -> str:
    base_keys = {
        "attn_entropy_mean", "attn_entropy_p10", "attn_entropy_p50", "attn_entropy_p90",
        "effective_token_count", "avg_max_attention", "token_usage_min", "token_usage_max",
        "token_usage_top5", "raw_feature_norm", "attention_out_norm", "fused_feature_norm",
        "num_queries_used", "num_tokens", "fusion_geometry_mode", "fusion_scene_scale",
        "key_geo_scale", "memory_p_norm_std", "memory_p_norm_absmax", "memory_p_norm_finite",
    }
    fields = []
    for key in sorted(k for k in stats.keys() if k not in base_keys):
        value = stats.get(key)
        if isinstance(value, bool):
            fields.append(f"{key}={value}")
        elif isinstance(value, int):
            fields.append(f"{key}={value}")
        elif isinstance(value, float):
            fields.append(f"{key}={value:.6f}")
        elif isinstance(value, str):
            fields.append(f"{key}={value}")
    return " ".join(fields)


def _log_fusion_runtime_stats(stage_tag: str, call_idx: int, stats: Dict[str, Any]):
    if not stats:
        return
    top5 = stats.get("token_usage_top5", [])
    top5_str = ",".join(f"{float(v):.4f}" for v in top5)
    extra_str = _format_lmc_runtime_extra_stats(stats)
    _logger.info(
        "[LMC-EvalRuntime][%s] call=%d entropy_mean=%.4f p10=%.4f p50=%.4f p90=%.4f "
        "effective_tokens=%.2f avg_max=%.4f usage_min=%.5f usage_max=%.5f top5=[%s] "
        "raw_norm=%.4f attn_out_norm=%.4f fused_norm=%.4f queries=%d tokens=%d "
        "fusion_mode=%s scene_scale=%.6f key_geo_scale=%.6f p_norm_std=%.4f "
        "p_norm_absmax=%.4f p_norm_finite=%s extra={%s}",
        stage_tag,
        int(call_idx),
        float(stats.get("attn_entropy_mean", 0.0)),
        float(stats.get("attn_entropy_p10", 0.0)),
        float(stats.get("attn_entropy_p50", 0.0)),
        float(stats.get("attn_entropy_p90", 0.0)),
        float(stats.get("effective_token_count", 0.0)),
        float(stats.get("avg_max_attention", 0.0)),
        float(stats.get("token_usage_min", 0.0)),
        float(stats.get("token_usage_max", 0.0)),
        top5_str,
        float(stats.get("raw_feature_norm", 0.0)),
        float(stats.get("attention_out_norm", 0.0)),
        float(stats.get("fused_feature_norm", 0.0)),
        int(stats.get("num_queries_used", 0)),
        int(stats.get("num_tokens", 0)),
        str(stats.get("fusion_geometry_mode", "n/a")),
        float(stats.get("fusion_scene_scale", 0.0)),
        float(stats.get("key_geo_scale", 0.0)),
        float(stats.get("memory_p_norm_std", 0.0)),
        float(stats.get("memory_p_norm_absmax", 0.0)),
        str(stats.get("memory_p_norm_finite", "n/a")),
        extra_str,
    )


def run_evaluation_lmc(opt):
    """Run evaluation. Auto-detects LMC vs vanilla checkpoint.

    Returns dict with median_rErr, median_tErr, pct5, etc.
    """
    device = torch.device(_normalize_device_for_visible_cuda(opt.device))
    dinov2_path = Path(opt.dinov2_path)
    head_network_path = Path(opt.network)
    scene_path = Path(opt.scene)
    session = getattr(opt, 'session', '')
    image_resolution = getattr(opt, 'image_resolution', 518)
    hypotheses = getattr(opt, 'hypotheses', 64)
    threshold = getattr(opt, 'threshold', 10)
    inlieralpha = getattr(opt, 'inlieralpha', 100)
    maxpixelerror = getattr(opt, 'maxpixelerror', 100)
    eval_deterministic = bool(getattr(opt, 'eval_deterministic', False))
    dsacstar_seed = int(getattr(opt, 'dsacstar_seed', 1305))
    dsacstar_seed_per_frame = bool(getattr(opt, 'dsacstar_seed_per_frame', True))
    eval_num_workers = int(getattr(opt, 'eval_num_workers', 6))
    lmc_log_runtime_stats = bool(getattr(opt, 'lmc_log_runtime_stats', False))
    lmc_runtime_stats_interval = max(1, int(getattr(opt, 'lmc_runtime_stats_interval', 100)))
    lmc_runtime_stats_max_pixels = max(1, int(getattr(opt, 'lmc_runtime_stats_max_pixels', 4096)))

    if eval_deterministic:
        _configure_eval_determinism(dsacstar_seed)

    # The exact stride is known after checkpoint inspection; keep DINOv2's
    # legacy default here and adjust again once model_backend is known.

    _logger.info(
        "[Eval] mode=%s dsacstar_seed=%s per_frame_seed=%s num_workers=%d",
        "deterministic" if eval_deterministic else "legacy-random",
        dsacstar_seed if eval_deterministic else "disabled",
        dsacstar_seed_per_frame if eval_deterministic else "disabled",
        eval_num_workers,
    )
    _logger.info("[Eval] device=%s  cuda_available=%s  cuda_device_count=%d",
                 device, torch.cuda.is_available(),
                 torch.cuda.device_count() if torch.cuda.is_available() else 0)

    # Load checkpoint
    checkpoint = _torch_load_trusted_checkpoint(head_network_path, map_location='cpu')
    is_lmc = _is_lmc_checkpoint(checkpoint)
    lmc_config = None

    if is_lmc:
        _logger.info("[LMC] Detected LMC checkpoint")
        lmc_config = checkpoint['lmc_config']
        head_state_dict = checkpoint['head_state_dict']
        memory_path = lmc_config.get('memory_path')
        model_backend = str(lmc_config.get('model_backend', 'ace_dinov2'))
    else:
        model_backend = 'ace_dinov2'
        _logger.info("[LMC] Vanilla checkpoint — delegating to standard eval")
        head_state_dict = checkpoint

    ace_lmc_global_head_mode = str(lmc_config.get('ace_lmc_global_head_mode', 'none')) if is_lmc else 'none'
    ace_lmc_global_feature_mode = str(lmc_config.get('ace_lmc_global_feature_mode', 'glace')) if is_lmc else 'glace'
    ace_lmc_global_gate_raw = (
        lmc_config.get('final_ace_lmc_global_gate', None)
        if is_lmc else 1.0
    )
    if ace_lmc_global_gate_raw is None:
        ace_lmc_global_gate_raw = lmc_config.get('ace_lmc_global_gate_init', 1.0)
    ace_lmc_global_gate_eval = float(ace_lmc_global_gate_raw)
    ace_lmc_random_global_seed = int(lmc_config.get('ace_lmc_random_global_seed', 20260531)) if is_lmc else 20260531
    if model_backend != 'ace_fcn_lmc' and ace_lmc_global_head_mode != 'none':
        raise ValueError('ace_lmc_global_head_mode is only valid for model_backend=ace_fcn_lmc.')
    output_subsample = 8 if model_backend == 'ace_fcn_lmc' else 14
    if image_resolution % output_subsample != 0:
        image_resolution = (image_resolution // output_subsample) * output_subsample
        _logger.warning("Image resolution adjusted to %s (must be multiple of %d)", image_resolution, output_subsample)

    # Build regressor (encoder + head)
    if model_backend == 'glace_lmc':
        if not is_lmc:
            raise ValueError('[Eval] GLACE backend is only supported through LMC checkpoints in this entrypoint.')
        glace_root = Path(getattr(opt, 'glace_root', None) or lmc_config.get('glace_root') or '/home/xwh/project/glace')
        glace_encoder_path = Path(
            getattr(opt, 'glace_encoder_path', None) or lmc_config.get('glace_encoder_path') or '/home/xwh/project/glace/ace_encoder_pretrained.pt'
        )
        network = create_glace_regressor_from_split_state_dict(
            glace_root=glace_root,
            encoder_path=glace_encoder_path,
            head_state_dict=head_state_dict,
            map_location='cpu',
        )
    elif model_backend == 'ace_fcn_lmc':
        if not is_lmc:
            raise ValueError('[Eval] ACE-FCN backend is only supported through LMC checkpoints in this entrypoint.')
        ace_encoder_path = Path(
            getattr(opt, 'ace_encoder_path', None) or lmc_config.get('ace_encoder_path') or '/home/xwh/project/ace_depth/ace_encoder_pretrained.pt'
        )
        local_dim = int(lmc_config.get('encoder_feature_dim', 512))
        in_channels = int(head_state_dict['res3_conv1.weight'].shape[1])
        pattern_count = sum(1 for key in head_state_dict.keys() if re.match(r'^\d+c0\.weight$', key))
        use_homogeneous = head_state_dict['fc3.weight'].shape[0] == 4
        network = RegressorACE.create_from_encoder(
            encoder_path=ace_encoder_path,
            mean=torch.zeros((3,)),
            num_head_blocks=pattern_count,
            use_homogeneous=use_homogeneous,
            num_encoder_features=local_dim,
            freeze_backbone=True,
        )
        network.heads = Head(torch.zeros((3,)), pattern_count, use_homogeneous, in_channels=in_channels)
        if ace_lmc_global_head_mode == 'glace_film':
            global_dim = int(lmc_config.get('glace_global_feat_dim', lmc_config.get('ace_lmc_global_feature_dim', 0)) or 0)
            if global_dim <= 0:
                global_dim = int(lmc_config.get('ace_lmc_final_head_dim', 0)) - local_dim
            if global_dim <= 0:
                raise ValueError('[Eval] glace_film checkpoint is missing global feature dimension.')
            network.heads = ACEGlobalFiLMHead(
                torch.zeros((3,)),
                pattern_count,
                use_homogeneous,
                in_channels=in_channels,
                global_dim=global_dim,
                gate_init=float(lmc_config.get('ace_lmc_global_gate_init', 0.0)),
                gate_max=float(lmc_config.get('ace_lmc_global_gate_max', 1.0)),
            )
        if ace_lmc_global_head_mode == 'glace_residual' and checkpoint.get('ace_lmc_global_residual_base_head_state_dict') is not None:
            head_state_dict = checkpoint['ace_lmc_global_residual_base_head_state_dict']
            _logger.info('[Eval] glace_residual: using stored Stage1 base head for local prediction.')
        network.heads.load_state_dict(head_state_dict)
        network.ace_lmc_local_feature_dim = local_dim
        network.ace_lmc_final_head_dim = in_channels
    else:
        network = Regressor.create_from_split_state_dict(
            dinov2_path=dinov2_path,
            head_state_dict=head_state_dict,
        )
    network = network.to(device)
    network.eval()
    # Verify the model actually landed on the requested device (catches silent CPU fallbacks).
    _actual_dev = next(network.parameters()).device
    _logger.info("[Eval] network moved to %s (requested %s)", _actual_dev, device)
    if device.type == 'cuda' and _actual_dev.type != 'cuda':
        _logger.error("[Eval] Network is on CPU despite requesting CUDA! "
                      "CUDA_VISIBLE_DEVICES=%s  cuda_available=%s",
                      os.environ.get('CUDA_VISIBLE_DEVICES', 'unset'), torch.cuda.is_available())

    # Build LMC modules if needed
    compressor = None
    fusion = None
    glace_residual_adapter = None
    ace_lmc_global_residual_head = None
    if model_backend == 'ace_fcn_lmc' and ace_lmc_global_head_mode == 'glace_residual':
        residual_state = checkpoint.get('ace_lmc_global_residual_state_dict')
        if residual_state is None:
            raise ValueError('[Eval] glace_residual checkpoint is missing ace_lmc_global_residual_state_dict.')
        residual_blocks = sum(1 for key in residual_state.keys() if re.match(r'^delta_head\.\d+c0\.weight$', key))
        residual_in_channels = int(residual_state['gate.weight'].shape[1])
        ace_lmc_global_residual_head = ACEGlobalResidualHead(
            mean=torch.zeros((3,)),
            num_head_blocks=residual_blocks,
            in_channels=residual_in_channels,
            gate_init=float(lmc_config.get('ace_lmc_global_gate_init', 0.001)),
            gate_max=float(lmc_config.get('ace_lmc_global_gate_max', 0.1)),
            delta_max_m=float(lmc_config.get('ace_lmc_global_residual_delta_max_m', 1.0)),
        ).to(device)
        ace_lmc_global_residual_head.load_state_dict(residual_state, strict=True)
        ace_lmc_global_residual_head.eval()
        _logger.info(
            '[Eval] Loaded ACE-LMC global residual head: in_channels=%d blocks=%d gate_max=%.6f delta_max_m=%.3f.',
            residual_in_channels, residual_blocks, float(lmc_config.get('ace_lmc_global_gate_max', 0.1)),
            float(lmc_config.get('ace_lmc_global_residual_delta_max_m', 1.0)),
        )
    memory_dict = None
    bank_data = None
    reference_eval_state = {"enabled": False, "contract_mode": "C0", "output_space": "points_world"}
    compressor_out_cached = None  # single compression result, reused for all test frames
    glace_lmc_fusion_target = str(
        lmc_config.get('effective_lmc_fusion_target', lmc_config.get('lmc_fusion_target', 'decoder'))
    ) if lmc_config is not None else 'decoder'
    local_residual_mode = str(lmc_config.get('local_residual_mode', 'none')) if lmc_config is not None else 'none'
    local_residual_alpha = float(lmc_config.get('local_residual_alpha', 1.0)) if lmc_config is not None else 1.0
    local_residual_alpha_init = float(lmc_config.get('local_residual_alpha_init', 0.001)) if lmc_config is not None else 0.001
    local_residual_alpha_max = float(lmc_config.get('local_residual_alpha_max', local_residual_alpha)) if lmc_config is not None else local_residual_alpha
    local_residual_alpha_warmup_steps = int(lmc_config.get('local_residual_alpha_warmup_steps', 0)) if lmc_config is not None else 0
    final_local_residual_alpha = None
    if lmc_config is not None and lmc_config.get('final_local_residual_alpha') is not None:
        final_local_residual_alpha = float(lmc_config.get('final_local_residual_alpha'))
    if local_residual_mode not in ('none', 'fixed_alpha', 'learned_alpha'):
        raise ValueError(f"Unsupported local_residual_mode={local_residual_mode!r}")
    if not math.isfinite(local_residual_alpha) or local_residual_alpha < 0.0 or local_residual_alpha > 1.0:
        raise ValueError(f"local_residual_alpha must be in [0,1], got {local_residual_alpha}")
    if not math.isfinite(local_residual_alpha_max) or local_residual_alpha_max < 0.0 or local_residual_alpha_max > 1.0:
        raise ValueError(f"local_residual_alpha_max must be in [0,1], got {local_residual_alpha_max}")
    if local_residual_mode == 'learned_alpha':
        if final_local_residual_alpha is None:
            raise ValueError("learned_alpha checkpoint is missing lmc_config['final_local_residual_alpha']; cannot evaluate deterministically.")
        if not math.isfinite(final_local_residual_alpha) or final_local_residual_alpha < 0.0 or final_local_residual_alpha > 1.0:
            raise ValueError(f"final_local_residual_alpha must be in [0,1], got {final_local_residual_alpha}")
    local_residual_alpha_eval = final_local_residual_alpha if local_residual_mode == 'learned_alpha' else local_residual_alpha
    glace_eval_local_dims_logged = False

    if is_lmc and memory_path is not None:
        from ace_compressor import GeoLMC
        from ace_fusion import LMCFeatureFusion

        # Build compressor/fusion with same dims as training (per-layer feature_dim from memory)
        compress_dim = lmc_config.get('compress_dim', 1024)
        num_layers = lmc_config.get('num_layers', 1)
        backbone_feature_dim = lmc_config.get('backbone_feature_dim', 1024)
        compressor = GeoLMC(
            input_dim=compress_dim,
            compress_dim=compress_dim,
            num_latent_tokens=lmc_config.get('num_latent_tokens', 64),
            num_fine=lmc_config.get('num_fine', 128),
            num_coarse=lmc_config.get('num_coarse', 16),
            mode=lmc_config.get('lmc_mode', 'global'),
            num_layers=num_layers,
            geo_sigma=lmc_config.get('geo_sigma', 0.5),
            use_scale_token=lmc_config.get('use_scale_token', True),
            scale_token_dim=lmc_config.get('scale_token_dim', 1024),
            num_attn_layers=lmc_config.get('num_attn_layers', 2),
            pe_normalize_input=lmc_config.get('pe_normalize_input', False),
            pe_scale_mode=lmc_config.get(
                'lmc_compressor_pe_scale_mode',
                'std' if lmc_config.get('pe_normalize_input', False) else 'raw',
            ),
            pe_scene_scale=lmc_config.get(
                'lmc_compressor_pe_scene_scale',
                lmc_config.get('lmc_fusion_scene_scale', 1.0),
            ),
            fps_start_policy=lmc_config.get('lmc_fps_start_policy', 'farthest_from_center'),
            key_slice_idx=lmc_config.get('lmc_key_slice_idx', None),
            key_feature_mode=lmc_config.get('lmc_key_feature_mode', 'slice'),
            feature_hierarchy_mode=lmc_config.get(
                'lmc_feature_hierarchy_mode', 'selected_key_concat_value'
            ),
            level_merge_mode=lmc_config.get('lmc_level_merge_mode', 'softmax_gate'),
            level_merge_init=lmc_config.get('lmc_level_merge_init', 'uniform'),
            level_proj_shared=lmc_config.get('lmc_level_proj_shared', False),
            level_cross_attn_shared=lmc_config.get('lmc_level_cross_attn_shared', True),
            level_gate_entropy_weight=lmc_config.get('lmc_level_gate_entropy_weight', 0.0),
            level_token_gate=lmc_config.get('lmc_level_token_gate', False),
            level_anchor_residual_gamma_init=lmc_config.get('lmc_level_anchor_residual_gamma_init', 0.0),
            geo_bias_mode=lmc_config.get('geo_bias_mode', 'legacy'),
            geo_bias_rbf_scales=lmc_config.get('geo_bias_rbf_scales', [0.25, 0.5, 1.0, 2.0, 4.0]),
            geo_bias_rbf_alpha_init=lmc_config.get('geo_bias_rbf_alpha_init', 0.0),
            geo_bias_rbf_learn_weights=lmc_config.get('geo_bias_rbf_learn_weights', True),
            geo_bias_rbf_per_head=lmc_config.get('geo_bias_rbf_per_head', False),
            pos_encoding_mode=lmc_config.get('pos_encoding_mode', 'fourier_legacy'),
            pos_fourier_v2_scales=lmc_config.get('pos_fourier_v2_scales', [1.0, 2.0, 4.0, 8.0, 16.0]),
            pos_fourier_coord_norm=lmc_config.get('pos_fourier_coord_norm', 'scene_radius'),
            pos_fourier_radius=lmc_config.get('pos_fourier_radius', 4.0),
            pos_fourier_learnable_scale=lmc_config.get('pos_fourier_learnable_scale', False),
            pos_fourier_residual_gate_init=lmc_config.get('pos_fourier_residual_gate_init', 0.0),
            point_rope_coord_norm=lmc_config.get('point_rope_coord_norm', 'scene_radius'),
            point_rope_radius=lmc_config.get('point_rope_radius', 4.0),
            point_rope_radius_policy=lmc_config.get('point_rope_radius_policy', 'fixed'),
            point_rope_mixed_memory_ratio=lmc_config.get('point_rope_mixed_memory_ratio', 0.5),
            point_rope_seed_pe=lmc_config.get('point_rope_seed_pe', 'fourier_legacy'),
            point_rope_base=lmc_config.get('point_rope_base', 10000.0),
            point_rope_axes=lmc_config.get('point_rope_axes', 'xyz_split'),
            point_rope_apply_to=lmc_config.get('point_rope_apply_to', 'qk'),
            geo_bias_crpb_dim=lmc_config.get('geo_bias_crpb_dim', 32),
            geo_bias_crpb_input=lmc_config.get('geo_bias_crpb_input', 'delta_dist_log'),
            geo_bias_crpb_radius=lmc_config.get('geo_bias_crpb_radius', 4.0),
            geo_bias_crpb_per_head=lmc_config.get('geo_bias_crpb_per_head', False),
            geo_bias_crpb_zero_init=lmc_config.get('geo_bias_crpb_zero_init', True),
        ).to(device)
        compressor.load_state_dict(checkpoint['compressor_state_dict'])
        compressor.eval()
        compressor.collect_runtime_stats = lmc_log_runtime_stats

        fusion = LMCFeatureFusion(
            feature_dim=backbone_feature_dim,
            mode=lmc_config.get('lmc_mode', 'global'),
            query_feature_dim=backbone_feature_dim,
            memory_feature_dim=compress_dim,
            fusion_geometry_mode=lmc_config.get('lmc_fusion_geometry_mode', 'value_only_raw'),
            fusion_scene_scale=lmc_config.get('lmc_fusion_scene_scale', 1.0),
            fusion_key_geo_init=lmc_config.get('lmc_fusion_key_geo_init', 0.0),
            fusion_refinement_mode=lmc_config.get('lmc_fusion_refinement_mode', 'single'),
            fusion_cascade_layers=lmc_config.get('lmc_fusion_cascade_layers', 4),
            fusion_assembly_mode=lmc_config.get('lmc_fusion_assembly_mode', 'concat_mlp'),
            fusion_assembly_gamma_init=lmc_config.get('lmc_fusion_assembly_gamma_init', 0.0),
            fusion_reread_delta_alpha=lmc_config.get('lmc_fusion_reread_delta_alpha', 1.0),
            fusion_reread_scalar_gate=lmc_config.get('lmc_fusion_reread_scalar_gate', False),
            fusion_reread_gate_init=lmc_config.get('lmc_fusion_reread_gate_init', 0.0),
            fusion_reread_post_norm=lmc_config.get('lmc_fusion_reread_post_norm', True),
            fusion_reread_trust_region_ratio=lmc_config.get('lmc_fusion_reread_trust_region_ratio', 0.0),
            fusion_reread_temperature=lmc_config.get('lmc_fusion_reread_temperature', 1.0),
            fusion_reread_geo_lambda=lmc_config.get('lmc_fusion_reread_geo_lambda', 1.0),
            fusion_reread_geo_sigma=lmc_config.get('lmc_fusion_reread_geo_sigma', 1.0),
            fusion_reread_geo_sigma_mode=lmc_config.get('lmc_fusion_reread_geo_sigma_mode', 'fixed'),
            fusion_reread_geo_sigma_beta=lmc_config.get('lmc_fusion_reread_geo_sigma_beta', 1.0),
            fusion_reread_geo_sigma_min=lmc_config.get('lmc_fusion_reread_geo_sigma_min', 0.5),
        ).to(device)
        fusion.load_state_dict(checkpoint['fusion_state_dict'])
        fusion.eval()
        if model_backend == 'glace_lmc':
            glace_residual_adapter = GLACEDecoderFeatureResidualAdapter(
                residual_gate_init=float(lmc_config.get('glace_residual_gate_init', 0.0) or 0.0),
                mode=str(lmc_config.get('glace_residual_mode', 'decoder_delta_tanh_scalar') or 'decoder_delta_tanh_scalar'),
                global_dim=int(lmc_config.get('glace_residual_global_dim', lmc_config.get('glace_global_feat_dim', 0)) or 0),
            ).to(device)
            adapter_state = checkpoint.get('glace_residual_adapter_state_dict')
            if adapter_state is not None:
                glace_residual_adapter.load_state_dict(adapter_state, strict=True)
            glace_residual_adapter.eval()

        # Load memory (supports both pooled and BSE formats)
        from utils_lmc import load_memory_features

        _logger.info(f"[LMC] Loading memory from {memory_path}")
        c1_ref_norm_alpha = float(lmc_config.get("c1_ref_norm_alpha", 1.0) or 1.0)
        bank_data = load_memory_features(str(memory_path), device, c1_ref_norm_alpha=c1_ref_norm_alpha)
        reference_eval_state = _build_reference_eval_state(lmc_config, bank_data)
        if reference_eval_state.get("enabled", False):
            _logger.info(
                "[LMC] Eval C1 recovery enabled: output_space=%s, reference_index=%s",
                reference_eval_state["output_space"],
                reference_eval_state["conditioning_reference"].get("reference_index"),
            )

        def _unsqueeze0(t):
            if t is None:
                return None
            if not isinstance(t, torch.Tensor):
                t = torch.tensor(t, device=device)
            if t.dim() == 1:
                return t.unsqueeze(0)
            if t.dim() == 2:
                return t.unsqueeze(0)
            return t

        memory_dict = {
            'pooled_points': _unsqueeze0(bank_data['pooled_points']),
            'pooled_features': _unsqueeze0(bank_data['pooled_features']),
            'scene_center': _unsqueeze0(bank_data.get('scene_center')),
        }
        if bank_data.get('all_scale_tokens') is not None:
            memory_dict['all_scale_tokens'] = _unsqueeze0(bank_data['all_scale_tokens'])

        # Compress memory once; reuse the same latent for all test frames (no per-frame compression).
        with torch.no_grad():
            compressor_out_cached = compressor(memory_dict)
        level_stats = getattr(compressor, "last_levelwise_runtime_stats", None)
        if isinstance(level_stats, dict):
            lmc_config["lmc_level_merge_weights"] = level_stats.get("lmc_level_merge_weights")
            lmc_config["lmc_level_gate_entropy"] = level_stats.get("lmc_level_gate_entropy")
            if lmc_log_runtime_stats:
                _logger.info(
                    "[LMC-Runtime][EvalCompress] level_merge_weights=%s level_gate_entropy=%.4f "
                    "per_level_latent_norm_mean=%s per_level_attention_entropy_mean=%s",
                    level_stats.get("final_level_merge_weights"),
                    float(level_stats.get("final_level_gate_entropy", 0.0)),
                    level_stats.get("per_level_latent_norm_mean", []),
                    level_stats.get("per_level_attention_entropy_mean", []),
                )
        _logger.info("[LMC] Memory compressed once (cached for all test frames)")

    # Dataset
    data_backend = getattr(opt, "data_backend", None)
    if data_backend is None:
        data_backend = lmc_config.get("data_backend", "ace") if is_lmc else "ace"
    data_backend = str(data_backend)
    if data_backend == "wai":
        testset = CamLocDatasetWAIDINOv2(
            scene_path,
            mode=0,
            use_half=False,
            image_height=image_resolution,
            augment=False,
            wai_repo_root=getattr(opt, "wai_repo_root", lmc_config.get("wai_repo_root", None)),
            wai_image_modality=getattr(opt, "wai_image_modality", lmc_config.get("wai_image_modality", "image")),
        )
    elif model_backend == 'glace_lmc':
        if is_lmc and lmc_config is not None:
            glace_root = Path(getattr(opt, 'glace_root', None) or lmc_config.get('glace_root') or '/home/xwh/project/glace')
            feat_name = str(getattr(opt, 'glace_feat_name', None) or lmc_config.get('glace_feat_name', 'features.npy'))
        else:
            glace_root = Path(getattr(opt, 'glace_root', None) or '/home/xwh/project/glace')
            feat_name = str(getattr(opt, 'glace_feat_name', 'features.npy'))
        testset = build_glace_camloc_dataset(
            glace_root=glace_root,
            root_dir=scene_path / 'test',
            mode=0,
            augment=False,
            aug_rotation=0.0,
            aug_scale_max=1.0,
            aug_scale_min=1.0,
            image_height=image_resolution,
            use_half=False,
            feat_name=feat_name,
        )
    elif model_backend == 'ace_fcn_lmc':
        feat_name = str(getattr(opt, 'glace_feat_name', None) or lmc_config.get('glace_feat_name', 'features.npy'))
        testset = CamLocDatasetACEFCNLMC(
            scene_path / 'test',
            mode=0,
            augment=False,
            aug_rotation=0.0,
            aug_scale_max=1.0,
            aug_scale_min=1.0,
            image_height=image_resolution,
            use_half=False,
            feat_name=feat_name if ace_lmc_global_head_mode in ('glace_concat', 'glace_residual', 'glace_film') else None,
        )
    else:
        testset = CamLocDatasetDINOv2(
            scene_path / "test", mode=0, use_half=False,
            image_height=image_resolution, augment=False)
    loader_generator = None
    worker_init_fn = None
    if eval_deterministic:
        loader_generator = torch.Generator()
        loader_generator.manual_seed(dsacstar_seed)
        worker_init_fn = _seed_eval_worker
    testset_loader = DataLoader(
        testset,
        shuffle=False,
        num_workers=eval_num_workers,
        worker_init_fn=worker_init_fn,
        generator=loader_generator,
    )
    _logger.info(f"Test images: {len(testset)}")

    # Output files
    output_dir = Path(getattr(opt, 'output_dir', head_network_path.parent) or head_network_path.parent)
    output_dir.mkdir(parents=True, exist_ok=True)
    scene_name = scene_path.name
    test_log_file = output_dir / f'test_{scene_name}_{session}.txt'
    pose_log_file = output_dir / f'poses_{scene_name}_{session}.txt'
    test_log = open(test_log_file, 'w', 1)
    pose_log = open(pose_log_file, 'w', 1)

    if compressor is not None and compressor_out_cached is not None:
        _logger.info("[LMC] Using pre-computed compressed memory for all %d test images (no per-frame compression)", len(testset))

    ace_lmc_eval_random_global_cache = {}

    def _apply_ace_lmc_eval_global_policy(global_feat_BC):
        if model_backend != 'ace_fcn_lmc' or ace_lmc_global_head_mode not in ('glace_concat', 'glace_residual', 'glace_film'):
            return global_feat_BC
        mode = ace_lmc_global_feature_mode
        if mode == 'glace':
            out = global_feat_BC
        elif mode == 'zero':
            out = torch.zeros_like(global_feat_BC)
        elif mode == 'random':
            key = (int(global_feat_BC.shape[1]), str(global_feat_BC.device), str(global_feat_BC.dtype))
            if key not in ace_lmc_eval_random_global_cache:
                gen = torch.Generator(device='cpu').manual_seed(ace_lmc_random_global_seed)
                random_vec = torch.randn((global_feat_BC.shape[1],), generator=gen, dtype=torch.float32)
                ace_lmc_eval_random_global_cache[key] = random_vec.to(
                    device=global_feat_BC.device,
                    dtype=global_feat_BC.dtype,
                ).view(1, -1)
            out = ace_lmc_eval_random_global_cache[key].expand_as(global_feat_BC)
        else:
            raise ValueError(f"Unsupported ace_lmc_global_feature_mode={mode!r}")
        if ace_lmc_global_head_mode in ('glace_residual', 'glace_film'):
            return out
        return out * torch.tensor(ace_lmc_global_gate_eval, device=global_feat_BC.device, dtype=global_feat_BC.dtype)

    avg_batch_time = 0
    num_batches = 0
    rErrs, tErrs = [], []
    pct25_5 = pct10_5 = pct5 = pct2 = pct1 = 0
    frame_idx = 0
    lmc_runtime_stats_calls = 0

    with torch.no_grad():
        for batch in testset_loader:
            if model_backend == 'glace_lmc' or (model_backend == 'ace_fcn_lmc' and ace_lmc_global_head_mode in ('glace_concat', 'glace_residual', 'glace_film')):
                image_B1HW, _, gt_pose_B44, _, intrinsics_B33, _, _, filenames, global_feat_BC, _ = batch
                global_feat_BC = global_feat_BC.to(device, non_blocking=True)
            else:
                image_B1HW, _, gt_pose_B44, _, intrinsics_B33, _, _, filenames = batch
                global_feat_BC = None
            batch_start_time = time.time()
            image_B1HW = image_B1HW.to(device, non_blocking=True)

            with autocast(enabled=True):
                local_features = network.get_features(image_B1HW)
                if model_backend == 'glace_lmc':
                    if global_feat_BC.dtype != local_features.dtype:
                        global_feat_BC = global_feat_BC.to(dtype=local_features.dtype)
                    features = torch.cat(
                        (
                            global_feat_BC[..., None, None].expand(-1, -1, local_features.shape[2], local_features.shape[3]),
                            local_features,
                        ),
                        dim=1,
                    )
                else:
                    features = local_features
                base_features = features

                # Apply LMC fusion if active (reuse single cached compression for this batch)
                if fusion is not None and compressor_out_cached is not None:
                    B, C, H, W = features.shape
                    sc = memory_dict['scene_center']
                    if sc.shape[0] == 1 and B > 1:
                        sc = sc.expand(B, -1)
                    # Expand cached (1, K, ...) to (B, K, ...) when B > 1; no extra compressor forward
                    if B > 1:
                        if isinstance(compressor_out_cached, dict):
                            compressor_out_batch = {
                                k: v.expand(B, *v.shape[1:]) if v.dim() >= 2 and v.shape[0] == 1 else v
                                for k, v in compressor_out_cached.items()
                            }
                        else:
                            z, p = compressor_out_cached
                            z = z.expand(B, -1, -1) if z.shape[0] == 1 else z
                            p = p.expand(B, -1, -1) if p.shape[0] == 1 else p
                            compressor_out_batch = (z, p)
                    else:
                        compressor_out_batch = compressor_out_cached
                    lmc_runtime_stats_calls += 1
                    collect_stats = lmc_log_runtime_stats and (
                        lmc_runtime_stats_calls == 1
                        or lmc_runtime_stats_calls % lmc_runtime_stats_interval == 0
                    )
                    if model_backend == 'glace_lmc' and glace_lmc_fusion_target == 'local':
                        global_dim = int(lmc_config.get('glace_global_feat_dim', lmc_config.get('glace_residual_global_dim', 0)) or 0)
                        if global_dim <= 0 or C <= global_dim:
                            raise ValueError(
                                f"[EvalFusion] invalid GLACE local fusion split: C={C}, global_dim={global_dim}"
                            )
                        global_part = features[:, :global_dim]
                        local_part = features[:, global_dim:]
                        local_C = local_part.shape[1]
                        if not glace_eval_local_dims_logged:
                            _logger.info(
                                "[GLACE-LMC] Eval local-only fusion: global_dim=%d local_dim=%d head_dim=%d local_residual=%s alpha_eval=%.6f alpha_target=%.6f alpha_init=%.6f alpha_max=%.6f warmup_steps=%d",
                                global_dim, local_C, C, local_residual_mode, local_residual_alpha_eval,
                                local_residual_alpha, local_residual_alpha_init, local_residual_alpha_max,
                                local_residual_alpha_warmup_steps,
                            )
                            glace_eval_local_dims_logged = True
                        query = local_part.permute(0, 2, 3, 1).reshape(B, H * W, local_C)
                        fused = fusion(
                            query,
                            compressor_out_batch,
                            sc,
                            return_stats=collect_stats,
                            stats_max_pixels=lmc_runtime_stats_max_pixels,
                        )
                        if collect_stats:
                            fused, stats = fused
                            _log_fusion_runtime_stats("EvalFusionLocal", lmc_runtime_stats_calls, stats)
                        fused_local = fused.reshape(B, H, W, local_C).permute(0, 3, 1, 2)
                        if local_residual_mode == 'none':
                            local_out = fused_local
                        elif local_residual_mode in ('fixed_alpha', 'learned_alpha'):
                            local_out = local_part + local_residual_alpha_eval * (fused_local - local_part)
                        else:
                            raise ValueError(f"Unsupported local_residual_mode={local_residual_mode!r}")
                        features = torch.cat((global_part, local_out), dim=1)
                    else:
                        query = features.permute(0, 2, 3, 1).reshape(B, H * W, C)
                        fused = fusion(
                            query,
                            compressor_out_batch,
                            sc,
                            return_stats=collect_stats,
                            stats_max_pixels=lmc_runtime_stats_max_pixels,
                        )
                        if collect_stats:
                            fused, stats = fused
                            _log_fusion_runtime_stats("EvalFusion", lmc_runtime_stats_calls, stats)
                        features = fused.reshape(B, H, W, C).permute(0, 3, 1, 2)

                if model_backend == 'glace_lmc':
                    if glace_residual_adapter is not None and glace_lmc_fusion_target != 'local':
                        B, C, H, W = features.shape
                        base_bC = base_features.permute(0, 2, 3, 1).reshape(B * H * W, C)
                        cand_bC = features.permute(0, 2, 3, 1).reshape(B * H * W, C)
                        mixed_bC = glace_residual_adapter(base_bC, cand_bC)
                        features = mixed_bC.view(B, H, W, C).permute(0, 3, 1, 2)
                    head_features = features
                    scene_coordinates_B3HW = network.get_scene_coordinates(head_features)
                elif model_backend == 'ace_fcn_lmc' and ace_lmc_global_head_mode == 'glace_concat':
                    if global_feat_BC is None:
                        raise ValueError('[Eval] ace_fcn_lmc/glace_concat requires global features in the dataset batch.')
                    if global_feat_BC.dtype != features.dtype:
                        global_feat_BC = global_feat_BC.to(dtype=features.dtype)
                    global_feat_BC = _apply_ace_lmc_eval_global_policy(global_feat_BC)
                    head_features = torch.cat(
                        (
                            global_feat_BC[..., None, None].expand(-1, -1, features.shape[2], features.shape[3]),
                            features,
                        ),
                        dim=1,
                    )
                    scene_coordinates_B3HW = network.get_scene_coordinates(head_features)
                elif model_backend == 'ace_fcn_lmc' and ace_lmc_global_head_mode == 'glace_film':
                    if global_feat_BC is None:
                        raise ValueError('[Eval] ace_fcn_lmc/glace_film requires global features in the dataset batch.')
                    if global_feat_BC.dtype != features.dtype:
                        global_feat_BC = global_feat_BC.to(dtype=features.dtype)
                    global_feat_BC = _apply_ace_lmc_eval_global_policy(global_feat_BC)
                    scene_coordinates_B3HW = network.heads(features, global_feat_BC)
                elif model_backend == 'ace_fcn_lmc' and ace_lmc_global_head_mode == 'glace_residual':
                    if global_feat_BC is None:
                        raise ValueError('[Eval] ace_fcn_lmc/glace_residual requires global features in the dataset batch.')
                    if ace_lmc_global_residual_head is None:
                        raise ValueError('[Eval] glace_residual mode requires loaded residual head.')
                    if global_feat_BC.dtype != features.dtype:
                        global_feat_BC = global_feat_BC.to(dtype=features.dtype)
                    global_feat_BC = _apply_ace_lmc_eval_global_policy(global_feat_BC)
                    local_pred = network.get_scene_coordinates(features)
                    residual_input = torch.cat(
                        (
                            global_feat_BC[..., None, None].expand(-1, -1, features.shape[2], features.shape[3]),
                            features,
                        ),
                        dim=1,
                    )
                    scene_coordinates_B3HW, _, _ = ace_lmc_global_residual_head(local_pred, residual_input)
                else:
                    head_features = features
                    scene_coordinates_B3HW = network.get_scene_coordinates(head_features)

            scene_coordinates_B3HW = scene_coordinates_B3HW.float().cpu()

            # De-normalize predictions if full-pipeline normalization was used during training
            if is_lmc and lmc_config is not None:
                if reference_eval_state.get("enabled", False):
                    scene_coordinates_B3HW = _recover_pred_scene_to_world(scene_coordinates_B3HW, reference_eval_state)
                else:
                    norm_mu = lmc_config.get('normalization_mu')
                    norm_sigma = lmc_config.get('normalization_sigma')
                    if norm_mu is not None and norm_sigma is not None:
                        mu_t = torch.tensor(norm_mu, dtype=torch.float32).view(1, 3, 1, 1)
                        sigma_t = float(norm_sigma)
                        scene_coordinates_B3HW = scene_coordinates_B3HW * sigma_t + mu_t
                        _logger.debug(
                            "[LMC] De-normalized scene coords: sigma=%.4f, mu=(%.3f,%.3f,%.3f)",
                            sigma_t, float(mu_t[0, 0, 0, 0]), float(mu_t[0, 1, 0, 0]), float(mu_t[0, 2, 0, 0]),
                        )

            if isinstance(filenames, str):
                filenames = (filenames,)
            for scene_coordinates_3HW, gt_pose_44, intrinsics_33, frame_path in zip(
                    scene_coordinates_B3HW, gt_pose_B44, intrinsics_B33, filenames):

                focal_length = intrinsics_33[0, 0].item()
                ppX = intrinsics_33[0, 2].item()
                ppY = intrinsics_33[1, 2].item()
                frame_name = Path(frame_path).name
                out_pose = torch.zeros((4, 4))
                if eval_deterministic:
                    frame_seed = dsacstar_seed + frame_idx if dsacstar_seed_per_frame else dsacstar_seed
                    _maybe_set_dsacstar_seed(frame_seed)

                inlier_count = dsacstar.forward_rgb(
                    scene_coordinates_3HW.unsqueeze(0), out_pose,
                    hypotheses, threshold, focal_length, ppX, ppY,
                    inlieralpha, maxpixelerror, network.OUTPUT_SUBSAMPLE)

                t_err = float(torch.norm(gt_pose_44[0:3, 3] - out_pose[0:3, 3]))
                gt_R = gt_pose_44[0:3, 0:3].numpy()
                out_R = out_pose[0:3, 0:3].numpy()
                r_err = np.matmul(out_R, np.transpose(gt_R))
                r_err = cv2.Rodrigues(r_err)[0]
                r_err = np.linalg.norm(r_err) * 180 / math.pi

                rErrs.append(r_err)
                t_err_cm = t_err * 100
                tErrs.append(t_err_cm)

                if getattr(opt, 'log_per_frame', False):
                    _logger.info("  [Eval] %s  rErr=%.2f deg  tErr=%.2f cm", frame_name, r_err, t_err_cm)

                if r_err < 5 and t_err < 0.25:
                    pct25_5 += 1
                if r_err < 5 and t_err < 0.1:
                    pct10_5 += 1
                if r_err < 5 and t_err < 0.05:
                    pct5 += 1
                if r_err < 2 and t_err < 0.02:
                    pct2 += 1
                if r_err < 1 and t_err < 0.01:
                    pct1 += 1

                out_pose_inv = out_pose.inverse()
                t = out_pose_inv[0:3, 3]
                rot, _ = cv2.Rodrigues(out_pose_inv[0:3, 0:3].numpy())
                angle = np.linalg.norm(rot)
                axis = rot / angle
                q_w = math.cos(angle * 0.5)
                q_xyz = math.sin(angle * 0.5) * axis
                pose_log.write(
                    f"{frame_name} {q_w} {q_xyz[0].item()} {q_xyz[1].item()} "
                    f"{q_xyz[2].item()} {t[0]} {t[1]} {t[2]} "
                    f"{r_err} {t_err} {inlier_count}\n")
                frame_idx += 1

            avg_batch_time += time.time() - batch_start_time
            num_batches += 1

    total_frames = len(rErrs)
    tErrs.sort()
    rErrs.sort()
    median_idx = total_frames // 2
    median_rErr = rErrs[median_idx]
    median_tErr = tErrs[median_idx]
    avg_time = avg_batch_time / num_batches

    pct25_5 = pct25_5 / total_frames * 100
    pct10_5 = pct10_5 / total_frames * 100
    pct5 = pct5 / total_frames * 100
    pct2 = pct2 / total_frames * 100
    pct1 = pct1 / total_frames * 100

    _logger.info("=" * 50)
    _logger.info("EVAL SUMMARY (current errors):")
    _logger.info("  Median Error: %.2f deg, %.2f cm", median_rErr, median_tErr)
    _logger.info("  25cm/5deg: %.2f%% | 10cm/5deg: %.2f%% | 5cm/5deg: %.2f%% | 2cm/2deg: %.2f%% | 1cm/1deg: %.2f%%",
                 pct25_5, pct10_5, pct5, pct2, pct1)
    _logger.info("  Avg time: %.2f ms | Frames: %d", avg_time * 1000, total_frames)
    _logger.info("=" * 50)

    test_log.write(f"{median_rErr} {median_tErr} {avg_time}\n")
    test_log.close()
    pose_log.close()

    # 写入当前误差汇总，便于训练/复现时查看
    output_dir = Path(getattr(opt, 'output_dir', Path(opt.network).parent) or Path(opt.network).parent)
    output_dir.mkdir(parents=True, exist_ok=True)
    scene_name = Path(opt.scene).name
    session = getattr(opt, 'session', '') or 'eval'
    eval_summary_file = output_dir / f"eval_summary_{scene_name}_{session}.txt"
    summary_lines = [
        f"# ACE DINOv2+LMC Eval Summary | {scene_name}",
        f"median_rotation_deg\t{median_rErr:.4f}",
        f"median_translation_cm\t{median_tErr:.4f}",
        f"accuracy_25cm5deg_pct\t{pct25_5:.2f}",
        f"accuracy_10cm5deg_pct\t{pct10_5:.2f}",
        f"accuracy_5cm5deg_pct\t{pct5:.2f}",
        f"accuracy_2cm2deg_pct\t{pct2:.2f}",
        f"accuracy_1cm1deg_pct\t{pct1:.2f}",
        f"avg_time_per_frame_ms\t{avg_time * 1000:.2f}",
        f"total_frames\t{total_frames}",
    ]
    if is_lmc and lmc_config is not None:
        semantic_fields = {
            "requested_lmc_mode": lmc_config.get("requested_lmc_mode"),
            "effective_lmc_mode": lmc_config.get("effective_lmc_mode", lmc_config.get("lmc_mode")),
            "lmc_auto_mode_by_visibility": lmc_config.get("lmc_auto_mode_by_visibility"),
            "lmc_flow": lmc_config.get("lmc_flow"),
            "model_backend": lmc_config.get("model_backend"),
            "ace_encoder_path": lmc_config.get("ace_encoder_path"),
            "ace_lmc_global_head_mode": lmc_config.get("ace_lmc_global_head_mode"),
            "ace_lmc_local_checkpoint_path": lmc_config.get("ace_lmc_local_checkpoint_path"),
            "ace_lmc_freeze_local_stack": lmc_config.get("ace_lmc_freeze_local_stack"),
            "ace_lmc_global_feature_mode": lmc_config.get("ace_lmc_global_feature_mode"),
            "ace_lmc_global_gate_init": lmc_config.get("ace_lmc_global_gate_init"),
            "ace_lmc_global_gate_learnable": lmc_config.get("ace_lmc_global_gate_learnable"),
            "ace_lmc_global_gate_max": lmc_config.get("ace_lmc_global_gate_max"),
            "ace_lmc_global_residual_gate_l1_weight": lmc_config.get("ace_lmc_global_residual_gate_l1_weight"),
            "ace_lmc_global_residual_delta_max_m": lmc_config.get("ace_lmc_global_residual_delta_max_m"),
            "final_ace_lmc_global_gate": lmc_config.get("final_ace_lmc_global_gate"),
            "final_ace_lmc_global_residual_gate_mean": lmc_config.get("final_ace_lmc_global_residual_gate_mean"),
            "final_ace_lmc_global_residual_gate_max": lmc_config.get("final_ace_lmc_global_residual_gate_max"),
            "final_ace_lmc_global_residual_gated_delta_l2": lmc_config.get("final_ace_lmc_global_residual_gated_delta_l2"),
            "ace_lmc_final_head_dim": lmc_config.get("ace_lmc_final_head_dim"),
            "glace_global_feat_dim": lmc_config.get("glace_global_feat_dim"),
            "lmc_key_slice_idx": lmc_config.get("lmc_key_slice_idx"),
            "lmc_key_layer_label": lmc_config.get("lmc_key_layer_label"),
            "lmc_key_feature_mode": lmc_config.get("lmc_key_feature_mode"),
            "lmc_key_mix_weights": lmc_config.get("lmc_key_mix_weights"),
            "lmc_feature_hierarchy_mode": lmc_config.get(
                "lmc_feature_hierarchy_mode", "selected_key_concat_value"
            ),
            "lmc_level_merge_mode": lmc_config.get("lmc_level_merge_mode", "softmax_gate"),
            "lmc_level_merge_init": lmc_config.get("lmc_level_merge_init", "uniform"),
            "lmc_level_proj_shared": lmc_config.get("lmc_level_proj_shared", False),
            "lmc_level_cross_attn_shared": lmc_config.get("lmc_level_cross_attn_shared", True),
            "lmc_level_gate_entropy_weight": lmc_config.get("lmc_level_gate_entropy_weight", 0.0),
            "lmc_level_token_gate": lmc_config.get("lmc_level_token_gate", False),
            "lmc_level_anchor_residual_gamma_init": lmc_config.get(
                "lmc_level_anchor_residual_gamma_init", 0.0
            ),
            "final_lmc_level_anchor_residual_gamma": lmc_config.get(
                "final_lmc_level_anchor_residual_gamma"
            ),
            "lmc_level_merge_weights": lmc_config.get("lmc_level_merge_weights"),
            "lmc_level_gate_entropy": lmc_config.get("lmc_level_gate_entropy"),
            "geo_bias_mode": lmc_config.get("geo_bias_mode", "legacy"),
            "geo_bias_rbf_scales": lmc_config.get("geo_bias_rbf_scales"),
            "geo_bias_rbf_alpha_init": lmc_config.get("geo_bias_rbf_alpha_init", 0.0),
            "geo_bias_rbf_learn_weights": lmc_config.get("geo_bias_rbf_learn_weights", True),
            "geo_bias_rbf_per_head": lmc_config.get("geo_bias_rbf_per_head", False),
            "final_geo_bias_rbf_alpha": lmc_config.get("final_geo_bias_rbf_alpha"),
            "final_geo_bias_rbf_weights": lmc_config.get("final_geo_bias_rbf_weights"),
            "pos_encoding_mode": lmc_config.get("pos_encoding_mode", "fourier_legacy"),
            "pos_fourier_v2_scales": lmc_config.get("pos_fourier_v2_scales"),
            "pos_fourier_coord_norm": lmc_config.get("pos_fourier_coord_norm", "scene_radius"),
            "pos_fourier_radius": lmc_config.get("pos_fourier_radius", 4.0),
            "pos_fourier_learnable_scale": lmc_config.get("pos_fourier_learnable_scale", False),
            "pos_fourier_residual_gate_init": lmc_config.get("pos_fourier_residual_gate_init", 0.0),
            "final_pos_fourier_residual_gate": lmc_config.get("final_pos_fourier_residual_gate"),
            "point_rope_coord_norm": lmc_config.get("point_rope_coord_norm", "scene_radius"),
            "point_rope_radius": lmc_config.get("point_rope_radius", 4.0),
            "point_rope_radius_policy": lmc_config.get("point_rope_radius_policy", "fixed"),
            "point_rope_mixed_memory_ratio": lmc_config.get("point_rope_mixed_memory_ratio", 0.5),
            "point_rope_seed_pe": lmc_config.get("point_rope_seed_pe", "fourier_legacy"),
            "point_rope_base": lmc_config.get("point_rope_base", 10000.0),
            "point_rope_axes": lmc_config.get("point_rope_axes", "xyz_split"),
            "point_rope_apply_to": lmc_config.get("point_rope_apply_to", "qk"),
            "geo_bias_crpb_dim": lmc_config.get("geo_bias_crpb_dim", 32),
            "geo_bias_crpb_input": lmc_config.get("geo_bias_crpb_input", "delta_dist_log"),
            "geo_bias_crpb_radius": lmc_config.get("geo_bias_crpb_radius", 4.0),
            "geo_bias_crpb_per_head": lmc_config.get("geo_bias_crpb_per_head", False),
            "geo_bias_crpb_zero_init": lmc_config.get("geo_bias_crpb_zero_init", True),
            "layers_idx": lmc_config.get("layers_idx"),
            "lmc_fps_start_policy": lmc_config.get("lmc_fps_start_policy"),
            "lmc_compressor_pe_scale_mode": lmc_config.get("lmc_compressor_pe_scale_mode"),
            "lmc_compressor_pe_scene_scale": lmc_config.get("lmc_compressor_pe_scene_scale"),
            "s1_loss_step_mode": lmc_config.get("s1_loss_step_mode"),
            "ace_g_fusion_in_s2": lmc_config.get("ace_g_fusion_in_s2"),
            "lmc_fusion_geometry_mode": lmc_config.get("lmc_fusion_geometry_mode"),
            "lmc_fusion_key_geo_init": lmc_config.get("lmc_fusion_key_geo_init"),
            "lmc_fusion_scene_scale": lmc_config.get("lmc_fusion_scene_scale"),
            "lmc_fusion_scene_scale_source": lmc_config.get("lmc_fusion_scene_scale_source"),
            "lmc_fusion_refinement_mode": lmc_config.get("lmc_fusion_refinement_mode", "single"),
            "lmc_fusion_cascade_layers": lmc_config.get("lmc_fusion_cascade_layers", 4),
            "lmc_fusion_assembly_mode": lmc_config.get("lmc_fusion_assembly_mode", "concat_mlp"),
            "lmc_fusion_assembly_gamma_init": lmc_config.get("lmc_fusion_assembly_gamma_init", 0.0),
            "final_lmc_fusion_assembly_gamma": lmc_config.get("final_lmc_fusion_assembly_gamma"),
            "lmc_fusion_reread_delta_alpha": lmc_config.get("lmc_fusion_reread_delta_alpha", 1.0),
            "lmc_fusion_reread_scalar_gate": lmc_config.get("lmc_fusion_reread_scalar_gate", False),
            "lmc_fusion_reread_gate_init": lmc_config.get("lmc_fusion_reread_gate_init", 0.0),
            "lmc_fusion_reread_post_norm": lmc_config.get("lmc_fusion_reread_post_norm", True),
            "lmc_fusion_reread_trust_region_ratio": lmc_config.get("lmc_fusion_reread_trust_region_ratio", 0.0),
            "lmc_fusion_reread_temperature": lmc_config.get("lmc_fusion_reread_temperature", 1.0),
            "lmc_fusion_reread_geo_lambda": lmc_config.get("lmc_fusion_reread_geo_lambda", 1.0),
            "lmc_fusion_reread_geo_sigma": lmc_config.get("lmc_fusion_reread_geo_sigma", 1.0),
            "lmc_fusion_reread_geo_sigma_mode": lmc_config.get("lmc_fusion_reread_geo_sigma_mode", "fixed"),
            "lmc_fusion_reread_geo_sigma_beta": lmc_config.get("lmc_fusion_reread_geo_sigma_beta", 1.0),
            "lmc_fusion_reread_geo_sigma_min": lmc_config.get("lmc_fusion_reread_geo_sigma_min", 0.5),
            "final_lmc_fusion_reread_gate": lmc_config.get("final_lmc_fusion_reread_gate"),
            "final_lmc_fusion_reread_gate_logit": lmc_config.get("final_lmc_fusion_reread_gate_logit"),
            "local_residual_mode": lmc_config.get("local_residual_mode", "none"),
            "local_residual_alpha": lmc_config.get("local_residual_alpha", 1.0),
            "local_residual_alpha_init": lmc_config.get("local_residual_alpha_init", 0.001),
            "local_residual_alpha_max": lmc_config.get("local_residual_alpha_max", lmc_config.get("local_residual_alpha", 1.0)),
            "local_residual_alpha_warmup_steps": lmc_config.get("local_residual_alpha_warmup_steps", 0),
            "final_local_residual_alpha": lmc_config.get("final_local_residual_alpha"),
            "final_local_residual_alpha_logit": lmc_config.get("final_local_residual_alpha_logit"),
            "glace_freeze_base_network": lmc_config.get("glace_freeze_base_network"),
            "glace_freeze_encoder": lmc_config.get("glace_freeze_encoder"),
            "glace_freeze_head": lmc_config.get("glace_freeze_head"),
            "lmc_log_runtime_stats": lmc_log_runtime_stats,
            "lmc_runtime_stats_interval": lmc_runtime_stats_interval,
            "lmc_runtime_stats_max_pixels": lmc_runtime_stats_max_pixels,
        }
        for key, value in semantic_fields.items():
            if isinstance(value, (list, tuple)):
                value = ",".join(str(v) for v in value)
            summary_lines.append(f"{key}\t{value}")
    eval_summary_file.write_text("\n".join(summary_lines) + "\n")
    _logger.info("Eval summary written to: %s", eval_summary_file)

    return {
        'median_rErr': median_rErr, 'median_tErr': median_tErr,
        'avg_time': avg_time,
        'pct25_5': pct25_5, 'pct10_5': pct10_5, 'pct5': pct5, 'pct2': pct2, 'pct1': pct1,
        'total_frames': total_frames,
        'test_log_file': str(test_log_file),
        'pose_log_file': str(pose_log_file),
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Test ACE DINOv2 (with optional LMC or multi-checkpoint ensemble)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('scene', type=Path)
    parser.add_argument('network', type=Path, help='Path to checkpoint')
    parser.add_argument('--dinov2_path', type=Path,
                        default=DATA_ROOT / 'checkpoints' / 'dinov2_vitl14_pretrain.pth')
    parser.add_argument('--ace_encoder_path', type=Path, default=Path('/home/xwh/project/ace_depth/ace_encoder_pretrained.pt'),
                        help='ACE FCN encoder checkpoint path for ace_fcn_lmc checkpoint reconstruction.')
    parser.add_argument('--glace_root', type=Path, default=Path('/home/xwh/project/glace'),
                        help='GLACE repo root for glace_lmc checkpoint reconstruction.')
    parser.add_argument('--glace_encoder_path', type=Path, default=Path('/home/xwh/project/glace/ace_encoder_pretrained.pt'),
                        help='GLACE encoder checkpoint path for glace_lmc checkpoint reconstruction.')
    parser.add_argument('--glace_feat_name', type=str, default='features.npy',
                        help='GLACE per-split global feature filename (ACE-format only).')
    parser.add_argument('--ensemble_networks', nargs='*', type=Path, default=None,
                        help='Additional checkpoints for joint evaluation. When set, the primary positional '
                             '`network` plus these checkpoints are evaluated per-frame and the final pose is '
                             'selected by DSAC inlier count.')
    parser.add_argument('--output_dir', type=Path, default=None,
                        help='Optional directory for eval outputs. For ensemble mode this is the joint-eval '
                             'output directory; for single-checkpoint mode it overrides the checkpoint parent.')
    parser.add_argument('--session', '-sid', default='')
    parser.add_argument('--image_resolution', type=int, default=518)
    parser.add_argument('--data_backend', type=str, default=None, choices=['ace', 'wai'],
                        help='Evaluation dataset backend. None uses checkpoint lmc_config or ace fallback.')
    parser.add_argument('--wai_repo_root', type=Path, default=None,
                        help='map-anything repo root for WAI eval.')
    parser.add_argument('--wai_image_modality', type=str, default='image',
                        help='WAI image modality key.')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--hypotheses', '-hyps', type=int, default=64)
    parser.add_argument('--threshold', '-t', type=float, default=10)
    parser.add_argument('--inlieralpha', '-ia', type=float, default=100)
    parser.add_argument('--maxpixelerror', '-maxerrr', type=float, default=100)
    parser.add_argument('--eval_deterministic', type=_strtobool, default=False)
    parser.add_argument('--dsacstar_seed', type=int, default=1305)
    parser.add_argument('--dsacstar_seed_per_frame', type=_strtobool, default=True)
    parser.add_argument('--eval_num_workers', type=int, default=6)
    parser.add_argument('--log_per_frame', type=_strtobool, default=False,
                        help='Print each frame’s rErr (deg) and tErr (cm) after evaluation.')
    parser.add_argument('--lmc_log_runtime_stats', type=_strtobool, default=False,
                        help='Log diagnostic-only LMC fusion attention/runtime stats during eval.')
    parser.add_argument('--lmc_runtime_stats_interval', type=int, default=100,
                        help='Fusion call interval for LMC runtime stats when enabled.')
    parser.add_argument('--lmc_runtime_stats_max_pixels', type=int, default=4096,
                        help='Maximum query pixels sampled for LMC runtime attention stats.')

    opt = parser.parse_args()
    if opt.ensemble_networks:
        from test_ace_dinov2_lmc_ensemble import run_ensemble_evaluation

        opt.networks = [opt.network, *opt.ensemble_networks]
        if not opt.session:
            opt.session = 'ensemble'
        run_ensemble_evaluation(opt)
    else:
        run_evaluation_lmc(opt)
