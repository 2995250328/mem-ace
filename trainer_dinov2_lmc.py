# trainer_dinov2_lmc.py
# Two-stage iterative trainer with Latent Memory Compression.
# Extends TrainerACEDINOv2 — when use_lmc=False, degrades to vanilla DINO ACE.
# Memory loading mirrors map-anything/tasks/ace/utils.py load_memory_features.

import gc
import logging
import math
import os
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict

import torch
import torch.optim as optim
import torchvision.transforms.functional as TF
from torch.amp import autocast
from torch.utils.data import DataLoader, sampler

from ace_util import to_homogeneous
from trainer_dinov2 import TrainerACEDINOv2, set_seed
from ace_compressor import GeoLMC
from ace_fusion import LMCFeatureFusion
from ace_loss import ReproLoss

_logger = logging.getLogger(__name__)


def estimate_memory_front_visibility(
    pooled_points: torch.Tensor,
    all_poses: torch.Tensor,
    max_points: int = 4096,
) -> Dict[str, float] | None:
    """Estimate how many memory points lie in front of cameras (z>0 in camera frame).

    A very low ratio means global memory tokens are weakly co-visible across views,
    which often destabilizes global-mode LMC on wide-baseline scenes.
    """
    if pooled_points is None or all_poses is None:
        return None
    if not isinstance(pooled_points, torch.Tensor) or not isinstance(all_poses, torch.Tensor):
        return None
    if pooled_points.numel() == 0 or all_poses.numel() == 0:
        return None

    pts = pooled_points.detach().float().cpu()
    poses = all_poses.detach().float().cpu()
    if poses.ndim == 4 and poses.shape[1:] == (1, 4, 4):
        poses = poses[:, 0]
    if poses.ndim != 3 or poses.shape[1:] != (4, 4):
        return None

    n_pts = pts.shape[0]
    if n_pts > max_points:
        perm = torch.randperm(n_pts)[:max_points]
        pts = pts[perm]
        n_pts = pts.shape[0]

    pts_h = torch.cat([pts, torch.ones(n_pts, 1)], dim=1).t()  # (4, N)
    ratios = []
    for p in poses:
        try:
            w2c = torch.linalg.inv(p)
        except RuntimeError:
            continue
        cam = (w2c @ pts_h)[:3]
        z = cam[2]
        ratios.append(float((z > 0).float().mean().item()))

    if len(ratios) == 0:
        return None
    return {
        "mean_front_ratio": float(sum(ratios) / len(ratios)),
        "min_front_ratio": float(min(ratios)),
        "median_front_ratio": float(torch.tensor(ratios).median().item()),
        "max_front_ratio": float(max(ratios)),
        "num_points_sampled": int(n_pts),
        "num_poses": int(len(ratios)),
    }


def load_memory_features(path: str, device: torch.device) -> Dict[str, Any]:
    """
    Load memory tensors from disk (same contract as map-anything load_memory_features).
    Supports POOLED memory bank format; unknown format raises with available keys.
    """
    payload = torch.load(path, map_location=device, weights_only=False)

    def safe_to_device(key: str):
        val = payload.get(key)
        if isinstance(val, torch.Tensor):
            return val.to(device)
        return val

    if "pooled_points" in payload and "pooled_features" in payload:
        _logger.info("[LMC] Detected POOLED memory bank at %s", path)
        def _to_tensor(x):
            if isinstance(x, torch.Tensor):
                return x.to(device)
            return torch.tensor(x, device=device)
        return {
            "type": "pooled",
            "pooled_points": _to_tensor(payload["pooled_points"]),
            "pooled_features": _to_tensor(payload["pooled_features"]),
            "pooled_colors": safe_to_device("pooled_colors"),
            "all_poses": safe_to_device("all_poses"),
            "all_intrinsics": safe_to_device("all_intrinsics"),
            "all_scale_tokens": safe_to_device("all_scale_tokens"),
            "scene_center": safe_to_device("scene_center"),
            "ref_pose": safe_to_device("ref_pose"),
            "patch_stride": payload.get("patch_stride", 14.0),
            "voxel_size": payload.get("voxel_size", 0.05),
            "original_views": payload.get("original_views", 0),
            "scene": payload.get("scene", "unknown"),
            "layers_idx": payload.get("layers_idx", []),
        }
    if "intermediate" in payload or "final" in payload:
        raise ValueError(
            f"Memory file at {path} is FULL intermediate format. "
            "ace_depth LMC only supports POOLED memory bank (pooled_points, pooled_features)."
        )
    available = list(payload.keys()) if isinstance(payload, dict) else "Not a dict"
    raise ValueError(f"Unknown memory file format at {path}. Keys: {available}")


def _normalize_scene_tag(scene: str | Path | None) -> str:
    """Normalize scene identifiers for robust memory-vs-train matching."""
    if scene is None:
        return ""
    s = str(scene).strip()
    if not s:
        return ""
    p = Path(s)
    name = p.name
    if name in ("train", "val", "test"):
        name = p.parent.name
    if name.endswith(("_train", "_val", "_test")):
        name = name.rsplit("_", 1)[0]
    return name


class TrainerACEDINOv2LMC(TrainerACEDINOv2):
    """ACE DINOv2 trainer with optional Latent Memory Compression.

    When ``use_lmc`` is False (or memory_path is None), the trainer
    degrades to the vanilla single-stage TrainerACEDINOv2.
    """

    def __init__(self, options):
        # Determine if LMC is active *before* parent __init__
        self.use_lmc = getattr(options, 'use_lmc', False)
        memory_path = getattr(options, 'memory_path', None)
        if memory_path is None:
            self.use_lmc = False

        # Parent builds: dataset, regressor, optimizer, scheduler, loss, buffer
        super().__init__(options)

        # Common iteration/eval policy (used by both LMC and vanilla-iterative mode)
        self.vanilla_iterations = max(1, int(getattr(options, 'vanilla_iterations', 1)))
        self.eval_each_iteration = getattr(options, 'eval_each_iteration', True)
        self.keep_best_only = getattr(options, 'keep_best_only', True)
        self.best_metric = getattr(options, 'best_metric', 'pct5')
        self.best_score = -float('inf')
        self.best_iter = -1
        self.best_eval = None
        self.step_log_path = self.options.output_map.parent / "training_log.txt"
        self.eval_log_path = self.options.output_map.parent / f"{self.options.output_map.stem}_eval_log.txt"
        self.training_log_path = self.options.output_map.parent / f"{self.options.output_map.stem}_training_log.txt"

        if not self.use_lmc:
            _logger.info(
                "[LMC] Disabled — running vanilla DINO ACE%s.",
                f" with iterative baseline ({self.vanilla_iterations} iters)" if self.vanilla_iterations > 1 else "",
            )
            return

        # --- Load pre-saved memory (same as map-anything train_ace + load_memory_features) ---
        _logger.info("[LMC] Loading memory from %s", memory_path)
        bank_data = load_memory_features(str(memory_path), self.device)

        # Build memory_dict with batch dim, like map-anything train_ace.py
        def _unsqueeze0(t):
            if t is None:
                return None
            if not isinstance(t, torch.Tensor):
                t = torch.tensor(t, device=self.device)
            if t.dim() == 1:
                return t.unsqueeze(0)
            if t.dim() == 2:
                return t.unsqueeze(0)
            return t

        scene_center = bank_data.get("scene_center")
        if scene_center is None:
            pts = bank_data["pooled_points"]
            scene_center = pts.mean(dim=0)
            _logger.info("[LMC] scene_center missing in file, using mean(pooled_points)")
        self._validate_memory_scene_consistency(bank_data, scene_center)

        self.memory_dict = {
            "pooled_points": _unsqueeze0(bank_data["pooled_points"]),
            "pooled_features": _unsqueeze0(bank_data["pooled_features"]),
            "scene_center": _unsqueeze0(scene_center),
            "all_scale_tokens": _unsqueeze0(bank_data.get("all_scale_tokens")) if bank_data.get("all_scale_tokens") is not None else None,
        }

        N_mem = self.memory_dict["pooled_points"].shape[1]
        pooled_features_dim = self.memory_dict["pooled_features"].shape[-1]
        _logger.info("[LMC] Memory loaded: %d points, pooled_features_dim=%d", N_mem, pooled_features_dim)

        # --- LMC config (same as map-anything train_ace: num_layers from layers_idx, feature_dim per layer) ---
        requested_lmc_mode = getattr(options, 'lmc_mode', 'global')
        lmc_mode = requested_lmc_mode
        vis_stats = estimate_memory_front_visibility(
            bank_data["pooled_points"],
            bank_data.get("all_poses"),
            max_points=int(getattr(options, "lmc_visibility_sample_points", 4096)),
        )
        self.memory_visibility_stats = vis_stats
        if vis_stats is not None:
            _logger.info(
                "[LMC] Memory front-visibility: mean=%.3f, median=%.3f, min=%.3f, max=%.3f "
                "(sampled_points=%d, poses=%d)",
                vis_stats["mean_front_ratio"],
                vis_stats["median_front_ratio"],
                vis_stats["min_front_ratio"],
                vis_stats["max_front_ratio"],
                vis_stats["num_points_sampled"],
                vis_stats["num_poses"],
            )
        auto_mode = bool(getattr(options, "lmc_auto_mode_by_visibility", True))
        fallback_mode = str(getattr(options, "lmc_visibility_fallback_mode", "local"))
        vis_thr = float(getattr(options, "lmc_visibility_front_ratio_threshold", 0.85))
        if (
            auto_mode
            and requested_lmc_mode == "global"
            and fallback_mode in ("local", "hierarchical")
            and vis_stats is not None
            and vis_stats["mean_front_ratio"] < vis_thr
        ):
            _logger.warning(
                "[LMC] Global mode auto-fallback triggered: mean_front_ratio=%.3f < %.3f. "
                "Switching lmc_mode: %s -> %s",
                vis_stats["mean_front_ratio"],
                vis_thr,
                requested_lmc_mode,
                fallback_mode,
            )
            lmc_mode = fallback_mode
        num_latent_tokens = getattr(options, 'num_latent_tokens', 64)
        num_attn_layers = getattr(options, 'num_attn_layers', 2)
        use_scale_token = getattr(options, 'use_scale_token', True)
        geo_sigma = getattr(options, 'geo_sigma', 0.5)
        num_fine = getattr(options, 'num_fine', 128)
        num_coarse = getattr(options, 'num_coarse', 16)

        layers_idx = bank_data.get("layers_idx") or []
        if isinstance(layers_idx, (list, tuple)) and len(layers_idx) > 0:
            num_layers = len(layers_idx)
            _logger.info("[LMC] Detected num_layers=%d from layers_idx", num_layers)
        else:
            num_layers = 4
            _logger.info("[LMC] No layers_idx, using default num_layers=%d", num_layers)

        feature_dim = pooled_features_dim // num_layers
        if feature_dim * num_layers != pooled_features_dim:
            _logger.warning(
                "[LMC] pooled_features_dim=%d not divisible by num_layers=%d; using feature_dim=%d",
                pooled_features_dim, num_layers, feature_dim)
        _logger.info("[LMC] feature_dim (per layer)=%d", feature_dim)

        scale_token_dim = 1024
        if bank_data.get("all_scale_tokens") is not None:
            st = bank_data["all_scale_tokens"]
            if isinstance(st, torch.Tensor) and st.numel() > 0:
                scale_token_dim = st.shape[-1]
                _logger.info("[LMC] scale_token_dim=%d from all_scale_tokens", scale_token_dim)

        backbone_feature_dim = 1024

        self.lmc_config = {
            'use_lmc': True,
            'lmc_mode': lmc_mode,
            'num_latent_tokens': num_latent_tokens,
            'num_attn_layers': num_attn_layers,
            'use_scale_token': use_scale_token,
            'compress_dim': feature_dim,
            'num_layers': num_layers,
            'scale_token_dim': scale_token_dim,
            'memory_path': str(memory_path),
        }

        # --- Build compressor (same as map-anything: input_dim/compress_dim = per-layer feature_dim) ---
        self.compressor = GeoLMC(
            input_dim=feature_dim,
            compress_dim=feature_dim,
            num_latent_tokens=num_latent_tokens,
            num_fine=num_fine,
            num_coarse=num_coarse,
            num_layers=num_layers,
            geo_sigma=geo_sigma,
            mode=lmc_mode,
            use_scale_token=use_scale_token,
            scale_token_dim=scale_token_dim,
            num_attn_layers=num_attn_layers,
        ).to(self.device)

        # --- Build fusion (query=backbone 1024, memory=compressor output feature_dim) ---
        self.fusion = LMCFeatureFusion(
            feature_dim=backbone_feature_dim,
            mode=lmc_mode,
            query_feature_dim=backbone_feature_dim,
            memory_feature_dim=feature_dim,
        ).to(self.device)

        # --- LMC training params ---
        self.lmc_iterations = getattr(options, 'lmc_iterations', 15)
        self.lmc_train_steps = getattr(options, 'lmc_train_steps', 2000)
        self.lmc_warmup_steps = getattr(options, 'lmc_warmup_steps', 5000)
        self.buffer_size_final = getattr(
            options, 'buffer_size_final',
            self.options.training_buffer_size * 3)
        legacy_lr_max = float(getattr(options, 'learning_rate_max', 1e-4))
        self.s1_learning_rate_max = float(getattr(options, 's1_learning_rate_max', legacy_lr_max))
        self.s2_learning_rate_max = float(getattr(options, 's2_learning_rate_max', legacy_lr_max))
        self.head_reset_strategy = getattr(
            options, 'head_reset_strategy', 'first_only')
        self.head_lr_multiplier_s2 = getattr(
            options, 'head_lr_multiplier_s2', 1.5)
        self.s2_repro_rewind_first_ratio = getattr(options, 's2_repro_rewind_first_ratio', 0.20)
        self.s2_repro_rewind_later_ratio = getattr(options, 's2_repro_rewind_later_ratio', 0.08)
        self.s2_repro_rewind_tau = getattr(options, 's2_repro_rewind_tau', 300.0)
        self.s2_lr_boost_first = getattr(options, 's2_lr_boost_first', 1.5)
        self.s2_lr_boost_later = getattr(options, 's2_lr_boost_later', 1.2)
        self.s2_lr_warmup_steps = getattr(options, 's2_lr_warmup_steps', 0)
        self.s1_early_stop_enable = getattr(options, 's1_early_stop_enable', True)
        self.s1_early_stop_min_steps = getattr(options, 's1_early_stop_min_steps', 400)
        self.s1_early_stop_patience = getattr(options, 's1_early_stop_patience', 200)
        self.s1_early_stop_min_delta = getattr(options, 's1_early_stop_min_delta', 1.0)

        # Iteration-level evaluation and checkpoint policy
        self.eval_each_iteration = getattr(options, 'eval_each_iteration', True)
        self.keep_best_only = getattr(options, 'keep_best_only', True)
        self.best_metric = getattr(options, 'best_metric', 'pct5')
        self.best_score = -float('inf')
        self.best_iter = -1
        self.best_eval = None
        self.step_log_path = self.options.output_map.parent / "training_log.txt"
        self.eval_log_path = self.options.output_map.parent / f"{self.options.output_map.stem}_eval_log.txt"
        self.training_log_path = self.options.output_map.parent / f"{self.options.output_map.stem}_training_log.txt"

        # ReproLoss 使用 S2 总步数作为 total_iterations，S1 传 0（早期权重大），S2 传 step_eff
        buf_per_it = self.options.training_buffer_size // self.options.batch_size
        buf_final = self.buffer_size_final // self.options.batch_size
        total_s2_steps = (self.lmc_iterations - 1) * self.options.epochs * buf_per_it + self.options.epochs * buf_final
        total_s2_steps = max(total_s2_steps, 1)
        self.repro_loss = ReproLoss(
            total_iterations=total_s2_steps,
            soft_clamp=self.options.repro_loss_soft_clamp,
            soft_clamp_min=self.options.repro_loss_soft_clamp_min,
            type=self.options.repro_loss_type,
            circle_schedule=(self.options.repro_loss_schedule == 'circle'),
        )
        self.global_s2_step = 0
        self.local_s2_step = 0
        self.current_lmc_iter = 0
        self.steps_per_s2_phase = 0
        self.s2_rewind_amount = 0.0
        self.optimizer_head = None
        self.scheduler_head = None

        # Rebuild optimizer to include compressor + fusion params (used only when not in LMC; S1/S2 use their own)
        self._rebuild_optimizer()

        _logger.info(f"[LMC] mode={lmc_mode}, tokens={num_latent_tokens}, "
                     f"attn_layers={num_attn_layers}, iterations={self.lmc_iterations}")

    def _validate_memory_scene_consistency(self, bank_data: Dict[str, Any], scene_center: torch.Tensor):
        """Fail fast when memory and training scene geometry look inconsistent."""
        strict_scene = bool(getattr(self.options, "lmc_strict_scene_check", True))
        strict_center = bool(getattr(self.options, "lmc_strict_center_check", True))
        max_center_dist = float(getattr(self.options, "lmc_scene_center_max_distance", 3.0))

        train_scene_norm = _normalize_scene_tag(getattr(self.options, "scene", None))
        mem_scene_raw = bank_data.get("scene")
        mem_scene_norm = _normalize_scene_tag(mem_scene_raw)
        if mem_scene_norm and mem_scene_norm.lower() != "unknown":
            if train_scene_norm and mem_scene_norm != train_scene_norm:
                # Allow if one tag is substring of the other (e.g. heads vs pgt_7scenes_heads)
                allowed = (
                    mem_scene_norm in train_scene_norm or train_scene_norm in mem_scene_norm
                )
                if not allowed:
                    msg = (
                        f"[LMC] Memory scene mismatch: memory.scene={mem_scene_raw!r} "
                        f"(normalized={mem_scene_norm!r}) vs train scene={self.options.scene!s} "
                        f"(normalized={train_scene_norm!r})."
                    )
                    if strict_scene:
                        raise ValueError(msg)
                    _logger.warning(msg)
        else:
            _logger.warning("[LMC] memory.scene is missing/unknown; skip strict scene-name match.")

        dataset_center = self.dataset.mean_cam_center.detach().float().view(-1).cpu()
        mem_center = scene_center.detach().float().view(-1).cpu()
        if dataset_center.numel() >= 3 and mem_center.numel() >= 3:
            center_dist = float(torch.linalg.norm(mem_center[:3] - dataset_center[:3]).item())
            _logger.info(
                "[LMC] Center consistency: ||memory.scene_center - dataset.mean_cam_center|| = %.4f m "
                "(threshold=%.4f m)",
                center_dist,
                max_center_dist,
            )
            if center_dist > max_center_dist:
                msg = (
                    f"[LMC] Memory center mismatch is too large ({center_dist:.4f} m > {max_center_dist:.4f} m). "
                    "Likely using memory from a different scene or coordinate system."
                )
                if strict_center:
                    raise ValueError(msg)
                _logger.warning(msg)

        all_poses = bank_data.get("all_poses")
        if isinstance(all_poses, torch.Tensor) and all_poses.numel() > 0:
            poses = all_poses.detach().float().cpu()
            if poses.ndim == 4 and poses.shape[1] == 1:
                poses = poses[:, 0]
            centers = None
            if poses.ndim == 3 and poses.shape[1:] == (4, 4):
                centers = poses[:, :3, 3]
            elif poses.ndim == 3 and poses.shape[1:] == (3, 4):
                centers = poses[:, :3, 3]
            if centers is not None and centers.numel() > 0:
                mem_center_mean = centers.mean(dim=0)
                mean_dist = float(torch.linalg.norm(mem_center_mean - dataset_center[:3]).item())
                cmin = centers.min(dim=0)[0]
                cmax = centers.max(dim=0)[0]
                _logger.info(
                    "[LMC] all_poses stats: center_mean=(%.3f, %.3f, %.3f), "
                    "range_x=[%.3f, %.3f], range_y=[%.3f, %.3f], range_z=[%.3f, %.3f], "
                    "mean_center_dist_to_dataset=%.4f m",
                    float(mem_center_mean[0]), float(mem_center_mean[1]), float(mem_center_mean[2]),
                    float(cmin[0]), float(cmax[0]),
                    float(cmin[1]), float(cmax[1]),
                    float(cmin[2]), float(cmax[2]),
                    mean_dist,
                )

    # ------------------------------------------------------------------
    # Optimizer rebuild (includes compressor + fusion + head)
    # ------------------------------------------------------------------

    def _rebuild_optimizer(self):
        """Rebuild optimizer & scheduler to cover compressor + fusion + head."""
        params = (
            list(self.compressor.parameters()) +
            list(self.fusion.parameters()) +
            list(self.regressor.heads.parameters())
        )
        self.optimizer = optim.AdamW(params, lr=self.options.learning_rate_min)

        total_s2_steps = (self.lmc_iterations *
                          self.options.training_buffer_size //
                          self.options.batch_size)
        total_s1_steps = (self.lmc_warmup_steps +
                          (self.lmc_iterations - 1) * self.lmc_train_steps)
        total_steps = max(total_s1_steps + total_s2_steps, 1)

        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=max(self.s1_learning_rate_max, self.s2_learning_rate_max),
            total_steps=total_steps,
            cycle_momentum=False,
        )

    # ------------------------------------------------------------------
    # Head reset
    # ------------------------------------------------------------------

    def _align_head_mean_to_scene_center(self):
        """After head reset in LMC path: set head.mean buffer to memory scene_center so coordinate anchor is consistent."""
        if not self.use_lmc or not hasattr(self, 'memory_dict'):
            return
        sc = self.memory_dict.get('scene_center')
        if sc is None:
            return
        sc = sc.detach().to(self.device, dtype=self.regressor.heads.mean.dtype)
        if sc.dim() == 1:
            sc = sc.unsqueeze(0)
        target = sc.view(-1)[:3].detach().float().cpu()
        dataset_center = self.dataset.mean_cam_center.detach().float().view(-1).cpu()[:3]
        shift = float(torch.linalg.norm(target - dataset_center).item())
        max_shift = float(getattr(self.options, "lmc_head_mean_max_shift", 3.0))
        strict_shift = bool(getattr(self.options, "lmc_strict_head_mean_alignment", True))
        if shift > max_shift:
            msg = (
                f"[LMC] Refusing to align head.mean to memory scene_center: "
                f"distance to dataset mean is {shift:.4f} m > {max_shift:.4f} m."
            )
            if strict_shift:
                raise ValueError(msg)
            _logger.warning(msg)
            return
        self.regressor.heads.mean.copy_(sc.view(1, 3, 1, 1))
        _logger.info(
            "[LMC] Head mean aligned to memory scene_center (distance to dataset mean: %.4f m).",
            shift,
        )

    def _maybe_reset_head(self, iteration_idx):
        """Apply head reset strategy."""
        strategy = self.head_reset_strategy
        if strategy == 'none':
            return
        if strategy == 'first_only' and iteration_idx != 0:
            return
        if strategy == 'every' or (strategy == 'first_only' and iteration_idx == 0):
            _logger.info(f"[LMC] Resetting head (strategy={strategy}, iter={iteration_idx})")
            for m in self.regressor.heads.modules():
                if hasattr(m, 'reset_parameters'):
                    m.reset_parameters()
            self._align_head_mean_to_scene_center()
            return
        if strategy == 'output_only':
            _logger.info(f"[LMC] Resetting head output layer (iter={iteration_idx})")
            for name, m in self.regressor.heads.named_modules():
                if 'fc3' in name and hasattr(m, 'reset_parameters'):
                    m.reset_parameters()
            self._align_head_mean_to_scene_center()

    # ------------------------------------------------------------------
    # Compress memory (used before each buffer fill)
    # ------------------------------------------------------------------

    def _compress_memory(self):
        """Run GeoLMC on the loaded memory_dict. Returns compressor output."""
        self.compressor.eval()
        with torch.no_grad():
            out = self.compressor(self.memory_dict)
        self.compressor.train()
        return out

    # ------------------------------------------------------------------
    # Fuse features with memory
    # ------------------------------------------------------------------

    def _fuse_features(self, features_BCHW, compressor_out):
        """Fuse encoder features with compressed memory tokens.

        features_BCHW: (B, C, H, W) from DINOv2 encoder
        compressor_out: from _compress_memory(), batch size 1 (single scene memory)
        Returns: fused features with same shape (B, C, H, W)
        """
        B, C, H, W = features_BCHW.shape
        scene_center = self.memory_dict['scene_center']
        if scene_center.shape[0] == 1 and B > 1:
            scene_center = scene_center.expand(B, -1)

        # Expand compressor_out to query batch size when B > 1 (memory is single-scene, batch=1)
        if B > 1:
            if isinstance(compressor_out, dict):
                compressor_out = {
                    k: v.expand(B, *v.shape[1:]) if v.dim() >= 2 and v.shape[0] == 1 else v
                    for k, v in compressor_out.items()
                }
            else:
                latent_z, latent_p = compressor_out
                if latent_z.shape[0] == 1:
                    latent_z = latent_z.expand(B, -1, -1)
                if latent_p.shape[0] == 1:
                    latent_p = latent_p.expand(B, -1, -1)
                compressor_out = (latent_z, latent_p)

        query = features_BCHW.permute(0, 2, 3, 1).reshape(B, H * W, C)
        fused = self.fusion(query, compressor_out, scene_center)
        return fused.reshape(B, H, W, C).permute(0, 3, 1, 2)

    # ------------------------------------------------------------------
    # Override: create_training_buffer (adds fusion step)
    # ------------------------------------------------------------------

    def create_training_buffer(self, buffer_size_override=None):
        """Fill buffer. When LMC is active, fuse features with memory first.
        If buffer_on_cpu is True (default for LMC), buffer is moved to CPU after fill to avoid GPU OOM.
        """
        if not self.use_lmc:
            return super().create_training_buffer()

        # S2 buffer collection should be deterministic: disable dropout etc.
        comp_was_training = self.compressor.training
        fusion_was_training = self.fusion.training
        self.compressor.eval()
        self.fusion.eval()

        # Compress memory once for this buffer fill
        compressor_out = self._compress_memory()

        # Temporarily override buffer size if requested (for final iteration)
        orig_buf_size = self.options.training_buffer_size
        if buffer_size_override is not None:
            self.options.training_buffer_size = buffer_size_override

        # We need to intercept the feature extraction to add fusion.
        # Strategy: monkey-patch regressor.get_features temporarily.
        original_get_features = self.regressor.get_features

        def fused_get_features(images):
            raw_feats = original_get_features(images)
            return self._fuse_features(raw_feats, compressor_out)

        self.regressor.get_features = fused_get_features
        try:
            super().create_training_buffer()
        finally:
            self.regressor.get_features = original_get_features
            self.options.training_buffer_size = orig_buf_size
            if comp_was_training:
                self.compressor.train()
            if fusion_was_training:
                self.fusion.train()

        # LMC: keep buffer on CPU to avoid GPU OOM (map-anything style: only batch on GPU)
        buffer_on_cpu = getattr(self.options, "buffer_on_cpu", True)
        if buffer_on_cpu and self.training_buffer is not None:
            for k in list(self.training_buffer.keys()):
                self.training_buffer[k] = self.training_buffer[k].cpu()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            _logger.info("[LMC] Buffer moved to CPU (buffer_on_cpu=True) to save GPU memory.")

    # ------------------------------------------------------------------
    # S1: Train compressor for a few steps
    # ------------------------------------------------------------------

    # Sanity thresholds for full_map (align with map-anything tasks/ace/loss_utils.py)
    SANITY_PIXEL_ERR = 50000.0
    SANITY_COORD_VAL = 10000.0

    def _s1_compute_loss_full_map(self, fused_feats_BCHW, gt_pose_inv_B34, K_B33, invK_B33, image_mask_B1HW, s1_step=0):
        """Full E2E: head on (B,C,H,W), repro loss on spatial positions (optionally subsampled).
        Aligns with map-anything:
        - Only pixels inside image_mask contribute (same as buffer only storing valid_mask pixels).
        - Nuclear (extreme outlier) points excluded from both repro and invalid loss.
        - Valid points: ReproLoss with dyntanh (s1_step for schedule); invalid: depth-pull loss.
        - Optional s1_full_map_max_points: subsample valid points to avoid hard points dominating.
        """
        B, C, H, W = fused_feats_BCHW.shape
        N = B * H * W
        # Restrict to image mask (align with map-anything: only train on pixels with valid GT)
        mask_flat = image_mask_B1HW.flatten()

        pred_scene_B3HW = self.regressor.get_scene_coordinates(fused_feats_BCHW)
        pred_scene_N31 = pred_scene_B3HW.permute(0, 2, 3, 1).reshape(N, 3).unsqueeze(-1).float()
        pred_scene_N41 = to_homogeneous(pred_scene_N31)

        gt_pose_inv_N34 = gt_pose_inv_B34.unsqueeze(1).unsqueeze(1).expand(B, H, W, 3, 4).reshape(N, 3, 4)
        K_N33 = K_B33.unsqueeze(1).unsqueeze(1).expand(B, H, W, 3, 3).reshape(N, 3, 3)
        invK_N33 = invK_B33.unsqueeze(1).unsqueeze(1).expand(B, H, W, 3, 3).reshape(N, 3, 3)

        pred_cam_N31 = torch.bmm(gt_pose_inv_N34, pred_scene_N41)
        pred_px_N31 = torch.bmm(K_N33, pred_cam_N31)
        pred_px_N31[:, 2].clamp_(min=self.options.depth_min)
        pred_px_N21 = pred_px_N31[:, :2] / pred_px_N31[:, 2, None]

        pixel_grid_B2HW = self.pixel_grid_2HW[:, :H, :W].clone().unsqueeze(0).expand(B, 2, H, W)
        target_px_N2 = pixel_grid_B2HW.permute(0, 2, 3, 1).reshape(N, 2)

        repro_b2 = pred_px_N21.squeeze() - target_px_N2
        repro_l1_b1 = torch.norm(repro_b2, dim=1, keepdim=True, p=1)
        repro_l2_b1 = torch.norm(repro_b2, dim=1, keepdim=True, p=2)

        # Nuclear mask (extreme outliers): exclude from both repro and invalid loss (align with loss_utils.py)
        repro_flat = repro_l1_b1.flatten()
        coords_max = torch.abs(pred_scene_N31).view(N, -1).max(dim=1)[0]
        nuclear_mask = (
            (repro_flat > self.SANITY_PIXEL_ERR)
            | (coords_max > self.SANITY_COORD_VAL)
            | (~torch.isfinite(repro_flat))
            | (~torch.isfinite(pred_cam_N31).all(dim=1).flatten())
        ).flatten()

        # Base invalid: depth / repro threshold / nonfinite (same as before)
        finite_repro = torch.isfinite(repro_l1_b1).flatten()
        finite_cam = torch.isfinite(pred_cam_N31).all(dim=1).flatten()
        invalid_min_depth = (pred_cam_N31[:, 2] < self.options.depth_min).flatten()
        invalid_repro = (repro_l1_b1 > self.options.repro_loss_hard_clamp).flatten()
        invalid_max_depth = (pred_cam_N31[:, 2] > self.options.depth_max).flatten()
        invalid_nonfinite = ~(finite_repro & finite_cam)
        base_invalid_mask = (invalid_min_depth | invalid_repro | invalid_max_depth | invalid_nonfinite).flatten()

        # Valid = inside image_mask, not base_invalid, not nuclear (align with map-anything buffer)
        valid_flat = mask_flat & (~base_invalid_mask) & (~nuclear_mask)
        invalid_mask = mask_flat & base_invalid_mask & (~nuclear_mask)

        # Optional subsampling: cap number of valid points for repro loss (reduce hard-point dominance)
        max_points = int(getattr(self.options, "s1_full_map_max_points", 0))
        if max_points > 0 and valid_flat.sum() > max_points:
            valid_idx = torch.where(valid_flat)[0]
            perm = torch.randperm(valid_idx.numel(), device=valid_idx.device, generator=getattr(self, "sampling_generator", None))
            chosen = valid_idx[perm[:max_points]]
            valid_repro_err = repro_l1_b1[chosen]
        else:
            valid_repro_err = repro_l1_b1[valid_flat]
        if valid_repro_err.numel() > 0:
            loss_valid = self.repro_loss.compute(valid_repro_err, s1_step)
        else:
            loss_valid = torch.zeros((), device=fused_feats_BCHW.device, dtype=fused_feats_BCHW.dtype)

        pixel_grid_crop_N31 = to_homogeneous(target_px_N2.unsqueeze(2))
        target_cam_N31 = self.options.depth_target * torch.bmm(invK_N33, pixel_grid_crop_N31)
        invalid_abs = torch.abs(target_cam_N31 - pred_cam_N31)
        invalid_abs = torch.nan_to_num(invalid_abs, nan=0.0, posinf=1e4, neginf=1e4)
        # Index by invalid_mask instead of expand to (N,3,1) to save GPU memory (avoids 6+ GiB for large N)
        loss_invalid = invalid_abs[invalid_mask].sum()

        loss = (loss_valid + loss_invalid) / N
        fraction_valid = float(valid_flat.sum() / N)
        finite_l1 = repro_l1_b1[torch.isfinite(repro_l1_b1).flatten()]
        finite_l2 = repro_l2_b1[torch.isfinite(repro_l2_b1).flatten()]
        nuclear_cnt = int(nuclear_mask.sum().item())
        stats = {
            "fraction_valid": fraction_valid,
            "pxerr_l1": float(finite_l1.mean().item()) if finite_l1.numel() > 0 else float("nan"),
            "pxerr_l2": float(finite_l2.mean().item()) if finite_l2.numel() > 0 else float("nan"),
            "valid_pxerr_l1": float(valid_repro_err.mean().item()) if valid_repro_err.numel() > 0 else float("nan"),
            "nonfinite_ratio": float(invalid_nonfinite.float().mean().item()),
            "nuclear_cnt": nuclear_cnt,
        }
        return loss, stats

    def _s1_compute_loss_from_features(self, features_bC, target_px_b2, gt_inv_poses_b34, Ks_b33, invKs_b33):
        """Compute ace_depth reprojection loss from sampled fused features (same formula as TrainerACEDINOv2.training_step)."""
        batch_size = features_bC.shape[0]
        channels = features_bC.shape[1]

        # Keep same head input reshaping logic as ace_depth training_step.
        h, w = 16, batch_size // 16
        if h * w != batch_size:
            batch_size = h * w
            if batch_size <= 0:
                return None, None
            features_bC = features_bC[:batch_size]
            target_px_b2 = target_px_b2[:batch_size]
            gt_inv_poses_b34 = gt_inv_poses_b34[:batch_size]
            Ks_b33 = Ks_b33[:batch_size]
            invKs_b33 = invKs_b33[:batch_size]

        features_bCHW = features_bC.view(1, h, w, channels).permute(0, 3, 1, 2)
        pred_scene_coords_b3HW = self.regressor.get_scene_coordinates(features_bCHW)

        pred_scene_coords_b31 = pred_scene_coords_b3HW.permute(0, 2, 3, 1).flatten(0, 2).unsqueeze(-1).float()
        pred_scene_coords_b41 = to_homogeneous(pred_scene_coords_b31)
        pred_cam_coords_b31 = torch.bmm(gt_inv_poses_b34, pred_scene_coords_b41)
        pred_px_b31 = torch.bmm(Ks_b33, pred_cam_coords_b31)
        pred_px_b31[:, 2].clamp_(min=self.options.depth_min)
        pred_px_b21 = pred_px_b31[:, :2] / pred_px_b31[:, 2, None]

        reprojection_error_b2 = pred_px_b21.squeeze() - target_px_b2
        reprojection_error_l1_b1 = torch.norm(reprojection_error_b2, dim=1, keepdim=True, p=1)
        reprojection_error_l2_b1 = torch.norm(reprojection_error_b2, dim=1, keepdim=True, p=2)

        finite_repro_b1 = torch.isfinite(reprojection_error_l1_b1)
        # Keep mask shape as (N, 1), matching reprojection_error_l1_b1.
        finite_cam_b1 = torch.isfinite(pred_cam_coords_b31).all(dim=1).all(dim=1, keepdim=True)
        invalid_min_depth_b1 = pred_cam_coords_b31[:, 2] < self.options.depth_min
        invalid_repro_b1 = reprojection_error_l1_b1 > self.options.repro_loss_hard_clamp
        invalid_max_depth_b1 = pred_cam_coords_b31[:, 2] > self.options.depth_max
        invalid_nonfinite_b1 = ~(finite_repro_b1 & finite_cam_b1)
        invalid_mask_b1 = invalid_min_depth_b1 | invalid_repro_b1 | invalid_max_depth_b1 | invalid_nonfinite_b1
        valid_mask_b1 = ~invalid_mask_b1

        valid_reprojection_error_b1 = reprojection_error_l1_b1[valid_mask_b1]
        # S1 使用早期 schedule（传 0），保持较大 repro 权重
        iter_for_loss = 0
        if valid_reprojection_error_b1.numel() > 0:
            loss_valid = self.repro_loss.compute(valid_reprojection_error_b1, iter_for_loss)
        else:
            loss_valid = torch.zeros((), device=features_bC.device, dtype=features_bC.dtype)

        pixel_grid_crop_b31 = to_homogeneous(target_px_b2.unsqueeze(2))
        target_camera_coords_b31 = self.options.depth_target * torch.bmm(invKs_b33, pixel_grid_crop_b31)
        invalid_mask_b11 = invalid_mask_b1.unsqueeze(2)
        invalid_abs = torch.abs(target_camera_coords_b31 - pred_cam_coords_b31)
        invalid_abs = torch.nan_to_num(invalid_abs, nan=0.0, posinf=1e4, neginf=1e4)
        loss_invalid = invalid_abs.masked_select(invalid_mask_b11).sum()

        loss = (loss_valid + loss_invalid) / batch_size
        fraction_valid = float(valid_mask_b1.sum() / batch_size)
        finite_l1 = reprojection_error_l1_b1[torch.isfinite(reprojection_error_l1_b1)]
        finite_l2 = reprojection_error_l2_b1[torch.isfinite(reprojection_error_l2_b1)]
        stats = {
            "fraction_valid": fraction_valid,
            "pxerr_l1": float(finite_l1.mean().item()) if finite_l1.numel() > 0 else float("nan"),
            "pxerr_l2": float(finite_l2.mean().item()) if finite_l2.numel() > 0 else float("nan"),
            "valid_pxerr_l1": float(valid_reprojection_error_b1.mean().item()) if valid_reprojection_error_b1.numel() > 0 else float("nan"),
            "nonfinite_ratio": float(invalid_nonfinite_b1.float().mean().item()),
        }
        return loss, stats

    def _build_s1_dataloader(self):
        """Build S1 dataloader with fixed-size images for batched forward passes."""
        # s1_batch_size: default 8 to avoid OOM (full_map and sample_per_image both use encoder on full batch;
        # DINOv2 attention ~(B*N)^2 causes fragmentation OOM after many steps on 24GB GPUs).
        effective_batch = max(1, getattr(self.options, 's1_batch_size', 8))
        buffer_image_width = getattr(self.options, 'buffer_image_width', None)
        if buffer_image_width is None:
            buffer_image_width = (self.options.image_resolution * 4 // 3 + 13) // 14 * 14

        s1_dataset = self._build_train_dataset(
            image_width=buffer_image_width,
            augment=False,
            aug_rotation=0,
            aug_scale_max=1.0,
            aug_scale_min=1.0,
        )

        batch_sampler = sampler.BatchSampler(
            sampler.RandomSampler(s1_dataset, generator=self.batch_generator),
            batch_size=effective_batch,
            drop_last=False,
        )

        return DataLoader(
            dataset=s1_dataset,
            batch_sampler=batch_sampler,
            generator=self.loader_generator,
            pin_memory=True,
            num_workers=self.num_data_loader_workers,
            persistent_workers=self.num_data_loader_workers > 0,
        )

    def _build_s1_scheduler(self, optimizer_s1, target_steps, iteration_idx, base_lr=None):
        """Build S1 LR scheduler (map-anything style). base_lr overrides learning_rate_max when set."""
        if base_lr is None:
            base_lr = self.s1_learning_rate_max
        sched_type = getattr(self.options, 'lmc_lr_scheduler_type', 'auto')
        if sched_type == 'auto':
            sched_type = 'warmup_plateau_cosine' if iteration_idx == 0 else 'warmup_cosine'
            _logger.info("  [S1 LR] auto -> %s (iter %d)", sched_type, iteration_idx)
        # iter>0 时 onecycle 爬坡容易把已训好的 compressor 推飞，改用 warmup_cosine 更稳
        if sched_type == 'onecycle_improved' and iteration_idx > 0:
            sched_type = 'warmup_cosine'
            _logger.info("  [S1 LR] iter>0: onecycle_improved -> warmup_cosine (gentler ramp)")
        min_lr_ratio = getattr(self.options, 'lmc_min_lr_ratio', 0.01)
        min_lr = base_lr * min_lr_ratio

        if sched_type == 'warmup_cosine':
            warmup_ratio = getattr(self.options, 'lmc_warmup_ratio', 0.2 if iteration_idx > 0 else 0.1)
            warmup_steps = int(target_steps * warmup_ratio)
            cosine_steps = max(1, target_steps - warmup_steps)

            def lr_lambda(step):
                if step < warmup_steps:
                    return (step / warmup_steps) if warmup_steps > 0 else 1.0
                progress = min(1.0, (step - warmup_steps) / cosine_steps)
                cos_f = 0.5 * (1 + math.cos(math.pi * progress))
                return min_lr_ratio + (1 - min_lr_ratio) * cos_f

            return optim.lr_scheduler.LambdaLR(optimizer_s1, lr_lambda)
        if sched_type == 'warmup_plateau_cosine':
            warmup_ratio = getattr(self.options, 'lmc_warmup_ratio', 0.1)
            plateau_ratio = getattr(self.options, 'lmc_plateau_ratio', 0.35)
            warmup_steps = int(target_steps * warmup_ratio)
            plateau_steps = int(target_steps * plateau_ratio)
            cosine_steps = max(1, target_steps - warmup_steps - plateau_steps)

            def lr_lambda(step):
                if step < warmup_steps:
                    return (step / warmup_steps) if warmup_steps > 0 else 1.0
                if step < warmup_steps + plateau_steps:
                    return 1.0
                progress = min(1.0, (step - warmup_steps - plateau_steps) / cosine_steps)
                cos_f = 0.5 * (1 + math.cos(math.pi * progress))
                return min_lr_ratio + (1 - min_lr_ratio) * cos_f

            return optim.lr_scheduler.LambdaLR(optimizer_s1, lr_lambda)
        if sched_type == 'onecycle_improved':
            pct = getattr(self.options, 'lmc_lr_pct_start', 0.3)
            div = getattr(self.options, 'lmc_lr_div_factor', 25.0)
            final_div = getattr(self.options, 'lmc_lr_final_div_factor', 10000.0)
            return optim.lr_scheduler.OneCycleLR(
                optimizer_s1, max_lr=base_lr, total_steps=target_steps,
                pct_start=pct, div_factor=div, final_div_factor=final_div, anneal_strategy='cos',
            )
        # onecycle_legacy
        pct = getattr(self.options, 'lmc_lr_pct_start', 0.4)
        return optim.lr_scheduler.OneCycleLR(
            optimizer_s1, max_lr=base_lr, total_steps=target_steps, pct_start=pct,
        )

    def _train_compressor_steps(self, iteration_idx, n_steps):
        """Stage 1: full-supervised reprojection training (compressor + fusion + head), aligned with map-anything logic."""
        self.compressor.train()
        self.fusion.train()
        self.regressor.heads.train()

        # Backbone remains frozen / inference-only in S1.
        self.regressor.encoder.eval()

        # From iter 1 onward use a lower S1 LR to avoid destabilizing the already-trained compressor/fusion.
        s1_lr_scale_later = float(getattr(self.options, 's1_lr_scale_later', 0.4))
        base_lr_s1 = self.s1_learning_rate_max * (s1_lr_scale_later if iteration_idx > 0 else 1.0)
        if iteration_idx > 0:
            _logger.info("  [S1 LR] iter>0: base_lr scaled by %.2f -> %.2e", s1_lr_scale_later, base_lr_s1)

        comp_optimizer = optim.AdamW([
            {'params': self.compressor.parameters()},
            {'params': self.fusion.parameters()},
            {'params': self.regressor.heads.parameters(), 'lr': base_lr_s1 * 0.1},
        ], lr=base_lr_s1)

        sched_lmc = self._build_s1_scheduler(comp_optimizer, n_steps, iteration_idx, base_lr=base_lr_s1)
        s1_loader = self._build_s1_dataloader()
        s1_loss_mode = getattr(self.options, 's1_loss_mode', 'full_map')
        s1_bs = max(1, getattr(self.options, 's1_batch_size', 8))
        _logger.info(
            "  [S1] loss_mode=%s, dataloader batch=%d, target_updates=%d (strict update-count mode)",
            s1_loss_mode,
            s1_bs,
            n_steps,
        )
        if s1_bs >= 8:
            _logger.info(
                "  [S1] Tip: if OOM after many steps, use --s1_batch_size 1 (or 2) and/or --s1_empty_cache_interval 30"
            )
        # S1 OOM after many steps (both full_map and sample_per_image): encoder runs on the same B images,
        # DINOv2 attention needs ~(B*N)^2; after many steps the allocator cache fragments and the same
        # allocation can fail. Use --s1_batch_size 1 (or 2) and/or --s1_empty_cache_interval 30.

        def cycle(iterable):
            while True:
                for x in iterable:
                    yield x

        s1_iterator = iter(cycle(s1_loader))
        log_interval = 10
        skipped_mask = 0
        skipped_sample = 0
        skipped_nonfinite = 0
        consecutive_unstable = 0
        max_consecutive_unstable = int(getattr(self.options, 's1_max_consecutive_unstable', 300))
        s1_lr_cap = base_lr_s1
        # Floor bound to base_lr_s1 only so floor <= cap <= base_lr_s1 (avoid learning_rate_min > base_lr_s1).
        s1_lr_floor_ratio = float(getattr(self.options, 's1_lr_floor_ratio', 0.05))
        s1_lr_floor = base_lr_s1 * s1_lr_floor_ratio
        update_step = 0
        raw_step = 0
        max_attempts = max(n_steps * 20, n_steps + 1)
        s1_early_stop = getattr(self.options, 's1_early_stop', True)
        s1_early_stop_min_updates = max(1, int(getattr(self.options, 's1_early_stop_min_updates', 400)))
        s1_early_stop_patience = max(1, int(getattr(self.options, 's1_early_stop_patience', 180)))
        s1_early_stop_rel_improve = float(getattr(self.options, 's1_early_stop_rel_improve', 0.01))
        s1_early_stop_ema_beta = min(0.999, max(0.0, float(getattr(self.options, 's1_early_stop_ema_beta', 0.90))))
        ema_px_err = None
        best_ema_px_err = float('inf')
        no_improve_updates = 0
        if s1_early_stop:
            _logger.info(
                "  [S1] early-stop ON (min_updates=%d, patience=%d, rel_improve=%.4f, ema_beta=%.2f)",
                s1_early_stop_min_updates, s1_early_stop_patience, s1_early_stop_rel_improve, s1_early_stop_ema_beta
            )

        while update_step < n_steps:
            raw_step += 1
            if raw_step > max_attempts:
                _logger.warning(
                    "  [S1] reached max attempts (%d) before target updates (%d). updates=%d",
                    max_attempts, n_steps, update_step
                )
                break
            batch = next(s1_iterator)
            image_BCHW, image_mask_B1HW, _, gt_pose_inv_B44, intrinsics_B33, intrinsics_inv_B33, _, _ = batch

            image_BCHW = image_BCHW.to(self.device, non_blocking=True)
            image_mask_B1HW = image_mask_B1HW.to(self.device, non_blocking=True)
            gt_pose_inv_B44 = gt_pose_inv_B44.to(self.device, non_blocking=True)
            intrinsics_B33 = intrinsics_B33.to(self.device, non_blocking=True)
            intrinsics_inv_B33 = intrinsics_inv_B33.to(self.device, non_blocking=True)

            if gt_pose_inv_B44.shape[1] == 4:
                gt_pose_inv_B34 = gt_pose_inv_B44[:, :3, :]
            else:
                gt_pose_inv_B34 = gt_pose_inv_B44

            if image_BCHW.dtype == torch.float16:
                image_BCHW = image_BCHW.float()

            with autocast("cuda", enabled=self.options.use_half):
                with torch.no_grad():
                    raw_feats = self.regressor.get_features(image_BCHW)

                comp_out = self.compressor(self.memory_dict)
                fused_feats = self._fuse_features(raw_feats, comp_out)

                B, C, H, W = fused_feats.shape
                image_mask_B1HW = TF.resize(image_mask_B1HW, [H, W], interpolation=TF.InterpolationMode.NEAREST)
                image_mask_B1HW = image_mask_B1HW.bool()

                if image_mask_B1HW.sum() == 0:
                    skipped_mask += 1
                    continue

                s1_loss_mode = getattr(self.options, 's1_loss_mode', 'full_map')

                if s1_loss_mode == 'full_map':
                    s1_step = iteration_idx * n_steps + update_step
                    loss, s1_stats = self._s1_compute_loss_full_map(
                        fused_feats, gt_pose_inv_B34, intrinsics_B33, intrinsics_inv_B33, image_mask_B1HW,
                        s1_step=s1_step,
                    )
                else:
                    pixel_positions_B2HW = self.pixel_grid_2HW[:, :H, :W].clone().unsqueeze(0).expand(B, 2, H, W)
                    gt_pose_inv = gt_pose_inv_B34.unsqueeze(1).expand(B, H * W, 3, 4).reshape(-1, 3, 4)
                    intrinsics = intrinsics_B33.unsqueeze(1).expand(B, H * W, 3, 3).reshape(-1, 3, 3)
                    intrinsics_inv = intrinsics_inv_B33.unsqueeze(1).expand(B, H * W, 3, 3).reshape(-1, 3, 3)

                    def normalize_shape(tensor_in):
                        return tensor_in.transpose(0, 1).flatten(1).transpose(0, 1)

                    batch_data = {
                        'features': normalize_shape(fused_feats),
                        'target_px': normalize_shape(pixel_positions_B2HW),
                        'gt_poses_inv': gt_pose_inv,
                        'intrinsics': intrinsics,
                        'intrinsics_inv': intrinsics_inv,
                    }

                    image_mask_N1 = normalize_shape(image_mask_B1HW.float())
                    n_per_image = self.options.samples_per_image

                    if s1_loss_mode == 'sample_per_image':
                        per_image_indices = []
                        for b in range(B):
                            valid_flat = torch.where(image_mask_B1HW[b].flatten())[0]
                            n_sample = min(n_per_image, valid_flat.numel())
                            if n_sample < 16:
                                continue
                            perm = torch.randperm(valid_flat.numel(), device=valid_flat.device, generator=self.sampling_generator)
                            idx_b = valid_flat[perm[:n_sample]]
                            global_idx = b * (H * W) + idx_b
                            per_image_indices.append(global_idx)
                        if len(per_image_indices) == 0:
                            skipped_sample += 1
                            continue
                        sample_idxs = torch.cat(per_image_indices, dim=0)
                    else:
                        features_to_select = min(n_per_image * B, image_mask_N1.numel())
                        if features_to_select < 16:
                            skipped_sample += 1
                            continue
                        sample_idxs = torch.multinomial(
                            image_mask_N1.view(-1),
                            features_to_select,
                            replacement=True,
                            generator=self.sampling_generator,
                        )

                    for k in batch_data:
                        batch_data[k] = batch_data[k][sample_idxs]

                    loss, s1_stats = self._s1_compute_loss_from_features(
                        batch_data['features'].contiguous(),
                        batch_data['target_px'].contiguous(),
                        batch_data['gt_poses_inv'].contiguous(),
                        batch_data['intrinsics'].contiguous(),
                        batch_data['intrinsics_inv'].contiguous(),
                    )

            if loss is None:
                continue

            # Stability guard: if non-finite ratio spikes, back off LR cap and skip this noisy step.
            if s1_stats["nonfinite_ratio"] > 0.10:
                skipped_nonfinite += 1
                consecutive_unstable += 1
                s1_lr_cap = max(s1_lr_floor, s1_lr_cap * 0.7)
                for pg in comp_optimizer.param_groups:
                    pg["lr"] = min(pg["lr"], s1_lr_cap)
                comp_optimizer.zero_grad(set_to_none=True)
                if consecutive_unstable >= max_consecutive_unstable:
                    _logger.warning(
                        "  [S1] stopping early: %d consecutive unstable batches (nonFinite=%.2f%%). updates=%d/%d.",
                        consecutive_unstable, s1_stats["nonfinite_ratio"] * 100.0, update_step, n_steps
                    )
                    break
                if update_step % log_interval == 0 or consecutive_unstable <= 3:
                    _logger.warning(
                        "  [S1] unstable batch at update %d/%d, attempt=%d (nonFinite=%.2f%%, consecutive=%d). lr_cap -> %.2e",
                        update_step + 1, n_steps, raw_step, s1_stats["nonfinite_ratio"] * 100.0, consecutive_unstable, s1_lr_cap
                    )
                continue

            if not torch.isfinite(loss):
                skipped_nonfinite += 1
                consecutive_unstable += 1
                s1_lr_cap = max(s1_lr_floor, s1_lr_cap * 0.7)
                for pg in comp_optimizer.param_groups:
                    pg["lr"] = min(pg["lr"], s1_lr_cap)
                comp_optimizer.zero_grad(set_to_none=True)
                if consecutive_unstable >= max_consecutive_unstable:
                    _logger.warning(
                        "  [S1] stopping early: %d consecutive unstable/non-finite. updates=%d/%d.",
                        consecutive_unstable, update_step, n_steps
                    )
                    break
                _logger.warning(
                    "  [S1] non-finite loss at update %d/%d, attempt=%d, skip update. lr_cap -> %.2e",
                    update_step + 1, n_steps, raw_step, s1_lr_cap
                )
                continue

            consecutive_unstable = 0
            comp_optimizer.zero_grad(set_to_none=True)
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(comp_optimizer)
            torch.nn.utils.clip_grad_norm_(
                list(self.compressor.parameters()) +
                list(self.fusion.parameters()) +
                list(self.regressor.heads.parameters()),
                max_norm=1.0
            )
            self.scaler.step(comp_optimizer)
            self.scaler.update()
            sched_lmc.step()
            # Enforce cap after scheduler step (important for OneCycle-like schedules).
            for pg in comp_optimizer.param_groups:
                pg["lr"] = min(pg["lr"], s1_lr_cap)

            update_step += 1
            # Mitigate CUDA fragmentation: both full_map and sample_per_image use same encoder batch;
            # DINOv2 attention ~(B*N)^2. After many steps cache fragments -> same alloc can OOM.
            s1_empty_cache_interval = int(getattr(self.options, "s1_empty_cache_interval", 30))
            if s1_empty_cache_interval > 0 and update_step % s1_empty_cache_interval == 0:
                torch.cuda.empty_cache()
            self.iteration += 1
            px_err = s1_stats["pxerr_l2"] if math.isfinite(s1_stats["pxerr_l2"]) else s1_stats["pxerr_l1"]
            if not math.isfinite(px_err):
                px_err = 0.0

            # Optional early stop: terminate S1 plateau to avoid wasting updates.
            if s1_early_stop:
                if ema_px_err is None:
                    ema_px_err = px_err
                else:
                    ema_px_err = s1_early_stop_ema_beta * ema_px_err + (1.0 - s1_early_stop_ema_beta) * px_err

                if update_step >= s1_early_stop_min_updates:
                    if ema_px_err < best_ema_px_err * (1.0 - s1_early_stop_rel_improve):
                        best_ema_px_err = ema_px_err
                        no_improve_updates = 0
                    else:
                        no_improve_updates += 1

                    if no_improve_updates >= s1_early_stop_patience:
                        _logger.info(
                            "  [S1] early-stop at update %d/%d (ema_pxErr=%.2f, best=%.2f, patience=%d)",
                            update_step, n_steps, ema_px_err, best_ema_px_err, s1_early_stop_patience
                        )
                        break

            if update_step % log_interval == 0 or update_step == 1 or update_step == n_steps:
                self._append_step_log(
                    iter_idx=iteration_idx,
                    step=self.iteration,
                    stage="S1",
                    loss=float(loss.item()),
                    px_err=float(px_err),
                    lr=float(comp_optimizer.param_groups[0]["lr"]),
                    mode="reproj",
                    med3d=-1.0,
                )
                _logger.info(
                    "  [S1] update %d/%d (attempt=%d), bs=%d, loss=%.4f, valid=%.1f%%, pxErrL1=%.2f, pxErrL2=%.2f, validPxErrL1=%.2f, nonFinite=%.2f%%, lr=%.2e",
                    update_step, n_steps, raw_step, image_BCHW.shape[0], loss.item(),
                    s1_stats["fraction_valid"] * 100.0, s1_stats["pxerr_l1"], s1_stats["pxerr_l2"],
                    s1_stats["valid_pxerr_l1"], s1_stats["nonfinite_ratio"] * 100.0, comp_optimizer.param_groups[0]["lr"],
                )

        if skipped_mask > 0 or skipped_sample > 0 or skipped_nonfinite > 0:
            _logger.warning(
                "  [S1] skipped batches: empty_mask=%d, few_samples=%d, nonfinite_loss=%d (total steps=%d)",
                skipped_mask, skipped_sample, skipped_nonfinite, n_steps
            )

    # ------------------------------------------------------------------
    # S2: Head-only optimizer and schedule (per phase)
    # ------------------------------------------------------------------

    def _setup_s2_optimizer_and_schedule(self, iteration_idx, is_last):
        """Build head-only optimizer and scheduler for this S2 phase; set rewind and local step."""
        current_buffer_size = self.buffer_size_final if is_last else self.options.training_buffer_size
        self.steps_per_s2_phase = self.options.epochs * (current_buffer_size // self.options.batch_size)
        self.steps_per_s2_phase = max(self.steps_per_s2_phase, 1)

        rewind_ratio = self.s2_repro_rewind_first_ratio if iteration_idx == 0 else self.s2_repro_rewind_later_ratio
        self.s2_rewind_amount = rewind_ratio * self.steps_per_s2_phase
        self.current_lmc_iter = iteration_idx
        self.local_s2_step = 0

        base_lr = self.s2_learning_rate_max * self.head_lr_multiplier_s2
        boost = self.s2_lr_boost_first if iteration_idx == 0 else self.s2_lr_boost_later
        head_lr = base_lr * boost
        self.optimizer_head = optim.AdamW(self.regressor.heads.parameters(), lr=head_lr)
        warmup_ratio = (self.s2_lr_warmup_steps / self.steps_per_s2_phase) if self.s2_lr_warmup_steps else 0.1
        warmup_ratio = min(0.5, max(0.0, warmup_ratio))
        self.scheduler_head = optim.lr_scheduler.OneCycleLR(
            self.optimizer_head,
            max_lr=head_lr,
            total_steps=self.steps_per_s2_phase,
            pct_start=warmup_ratio,
            anneal_strategy='cos',
        )
        _logger.info(
            "[S2] head lr=%.2e (boost=%.2f), steps=%d, rewind=%.1f (tau=%.0f)",
            head_lr, boost, self.steps_per_s2_phase, self.s2_rewind_amount, self.s2_repro_rewind_tau,
        )

    # ------------------------------------------------------------------
    # Iteration eval / best-checkpoint helpers
    # ------------------------------------------------------------------

    def _write_train_header(self):
        with open(self.step_log_path, 'w', encoding='utf-8') as f:
            f.write("Timestamp   Iter      Step  Stage               Loss       PxErr          LR    3D_Med  Mode        \n")
        with open(self.training_log_path, 'w', encoding='utf-8') as f:
            f.write("iter,s1_steps,s2_epochs,buffer_size,is_best,score,pct25_5,pct5,median_tErr_cm,median_rErr_deg,elapsed_s\n")
        with open(self.eval_log_path, 'w', encoding='utf-8') as f:
            f.write("# Iteration evaluation log\n")
            f.write(f"best_metric={self.best_metric}, keep_best_only={self.keep_best_only}\n")

    def _append_step_log(self, iter_idx, step, stage, loss, px_err, lr, mode, med3d=-1.0):
        timestamp = time.strftime("%H:%M:%S", time.localtime())
        line = (
            f"{timestamp:<10} "
            f"{iter_idx:>6d} "
            f"{step:>9d}  "
            f"{stage:<16} "
            f"{loss:>12.6f} "
            f"{px_err:>10.4f} "
            f"{lr:>10.2e} "
            f"{med3d:>9.3f}  "
            f"{mode:<12}\n"
        )
        with open(self.step_log_path, 'a', encoding='utf-8') as f:
            f.write(line)

    def _evaluate_checkpoint(self, ckpt_path, iter_idx):
        from test_ace_dinov2_lmc import run_evaluation_lmc
        # Always use 'cuda:0': after setup_cuda_environment() sets CUDA_VISIBLE_DEVICES to the
        # physical GPU index, that GPU is always visible as logical cuda:0 within this process.
        # str(self.device) returns 'cuda' (no index) which can behave unexpectedly in eval.
        _eval_d = self.device
        if _eval_d.type == 'cuda':
            eval_device = 'cuda:0'
        else:
            eval_device = str(_eval_d)
        _logger.info("[Eval] iter %d: ckpt=%s  eval_device=%s", iter_idx + 1, ckpt_path, eval_device)
        eval_opt = SimpleNamespace(
            scene=self.options.scene,
            network=ckpt_path,
            dinov2_path=self.options.dinov2_path,
            device=eval_device,
            image_resolution=self.options.image_resolution,
            session=f"iter_{iter_idx+1:02d}",
            hypotheses=64,
            threshold=10,
            inlieralpha=100,
            maxpixelerror=100,
            log_per_frame=True,
        )
        return run_evaluation_lmc(eval_opt)

    def _score_eval(self, eval_result):
        if not eval_result:
            return -float('inf')
        if self.best_metric == 'pct5':
            return float(eval_result.get('pct5', 0.0))
        if self.best_metric == 'pct10_5':
            return float(eval_result.get('pct10_5', 0.0))
        if self.best_metric == 'rt_error':
            # Lower error is better; convert to a score where larger is better.
            med_t = float(eval_result.get('median_tErr', 1e9))  # cm
            med_r = float(eval_result.get('median_rErr', 1e9))  # deg
            return -(med_t + med_r)
        return float(eval_result.get('pct5', 0.0)) - 1e-3 * float(eval_result.get('median_tErr', 0.0)) - 1e-4 * float(eval_result.get('median_rErr', 0.0))

    def _log_iteration_summary(self, it, s1_steps, is_last, is_best, score, eval_result, elapsed_s):
        buffer_size = self.buffer_size_final if is_last else self.options.training_buffer_size
        pct25_5 = eval_result.get('pct25_5', 0.0) if eval_result else 0.0
        pct5 = eval_result.get('pct5', 0.0) if eval_result else 0.0
        med_t = eval_result.get('median_tErr', 0.0) if eval_result else 0.0
        med_r = eval_result.get('median_rErr', 0.0) if eval_result else 0.0
        with open(self.training_log_path, 'a', encoding='utf-8') as f:
            f.write(f"{it+1},{s1_steps},{self.options.epochs},{buffer_size},{int(is_best)},{score:.6f},{pct25_5:.4f},{pct5:.4f},{med_t:.4f},{med_r:.4f},{elapsed_s:.2f}\n")
        with open(self.eval_log_path, 'a', encoding='utf-8') as f:
            tag = "BEST" if is_best else "-"
            f.write(
                f"iter={it+1:02d} tag={tag} score={score:.4f} "
                f"pct25_5={pct25_5:.2f} pct5={pct5:.2f} median={med_r:.2f}deg/{med_t:.2f}cm elapsed={elapsed_s:.1f}s\n"
            )

    def _write_best_checkpoint_meta(self, best_iter, best_score, eval_result):
        """Write best_checkpoint_meta.json for reproducibility and comparison."""
        if not self.use_lmc:
            return
        meta = {
            "best_iter": int(best_iter),
            "best_score": float(best_score),
            "best_checkpoint_path": str(self.options.output_map),
        }
        if eval_result:
            meta["pct25_5"] = float(eval_result.get("pct25_5", 0.0))
            meta["pct5"] = float(eval_result.get("pct5", 0.0))
            meta["median_tErr"] = float(eval_result.get("median_tErr", 0.0))
            meta["median_rErr"] = float(eval_result.get("median_rErr", 0.0))
        import json
        path = self.options.output_map.parent / "best_checkpoint_meta.json"
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(meta, f, indent=2)

    def _reset_vanilla_optimizer_scheduler(self):
        """Reset optimizer/scheduler so each vanilla iteration uses the same training schedule."""
        self.optimizer = optim.AdamW(self.regressor.parameters(), lr=self.options.learning_rate_min)
        steps_per_epoch = self.options.training_buffer_size // self.options.batch_size
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=self.options.learning_rate_max,
            epochs=self.options.epochs,
            steps_per_epoch=steps_per_epoch,
            cycle_momentum=False,
        )
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.options.use_half)
        self.iteration = 0

    def _train_vanilla_iterations(self):
        """Iterative vanilla baseline: repeat buffer+train+eval with no LMC modules."""
        self.training_start = time.time()
        self._write_train_header()
        best_ckpt_exists = False

        for it in range(self.vanilla_iterations):
            iter_start = time.time()
            _logger.info(f"\n{'='*60}")
            _logger.info(f"[Vanilla-Iter] Iteration {it+1}/{self.vanilla_iterations}")
            _logger.info(f"{'='*60}")

            self._reset_vanilla_optimizer_scheduler()

            _logger.info("[Vanilla-Iter] Filling buffer (size=%d)", self.options.training_buffer_size)
            self.create_training_buffer()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            _logger.info("[Vanilla-Iter] Training for %d epochs", self.options.epochs)
            for self.epoch in range(self.options.epochs):
                self.run_epoch()

            iter_ckpt = self.options.output_map.parent / f"{self.options.output_map.stem}.iter_{it+1:02d}.tmp.pt"
            self.save_model(iter_ckpt)

            eval_result = None
            score = -float('inf')
            if self.eval_each_iteration:
                try:
                    eval_result = self._evaluate_checkpoint(iter_ckpt, it)
                    score = self._score_eval(eval_result)
                except Exception as e:
                    _logger.warning("[Eval] vanilla iteration %d failed: %s", it + 1, e, exc_info=True)
            else:
                score = float(it + 1)

            is_best = (score > self.best_score)
            if is_best:
                self.best_score = score
                self.best_iter = it + 1
                self.best_eval = eval_result
                os.replace(iter_ckpt, self.options.output_map)
                best_ckpt_exists = True
                _logger.info("[Best] Updated best vanilla checkpoint at iter %d (score=%.4f)", it + 1, score)
            else:
                if iter_ckpt.exists():
                    if self.keep_best_only:
                        iter_ckpt.unlink()
                    else:
                        iter_ckpt.rename(self.options.output_map.parent / f"{self.options.output_map.stem}.iter_{it+1:02d}.pt")

            elapsed_s = time.time() - iter_start
            self._log_iteration_summary(
                it=it,
                s1_steps=0,
                is_last=False,
                is_best=is_best,
                score=score,
                eval_result=eval_result,
                elapsed_s=elapsed_s,
            )

        if not best_ckpt_exists:
            self.save_model(self.options.output_map)

        self._free_training_gpu_memory()
        _logger.info(
            "Done vanilla-iterative. Total time: %.1fs | best_iter=%s | best_score=%.4f",
            time.time() - self.training_start,
            self.best_iter,
            self.best_score,
        )
        _logger.info("Logs: %s | %s | %s", self.step_log_path, self.training_log_path, self.eval_log_path)

    # ------------------------------------------------------------------
    # Override: train (two-stage multi-iteration)
    # ------------------------------------------------------------------

    def train(self):
        """Two-stage iterative training with iteration-level eval and best-checkpoint policy."""
        if not self.use_lmc:
            if self.vanilla_iterations <= 1:
                return super().train()
            return self._train_vanilla_iterations()

        self.training_start = time.time()
        self._write_train_header()
        best_ckpt_exists = False

        for it in range(self.lmc_iterations):
            iter_start = time.time()
            is_last = (it == self.lmc_iterations - 1)
            _logger.info(f"\n{'='*60}")
            _logger.info(f"[LMC] Iteration {it+1}/{self.lmc_iterations}"
                         f"{' (FINAL)' if is_last else ''}")
            _logger.info(f"{'='*60}")

            # --- Stage 1 ---
            s1_steps = self.lmc_warmup_steps if it == 0 else self.lmc_train_steps
            _logger.info(f"[S1] Training compressor for {s1_steps} steps")
            self._train_compressor_steps(it, s1_steps)

            # S1 NaN guard: if compressor weights are all NaN (fp16 overflow / unstable S1),
            # skip S2 for this iteration to avoid wasting time with all-NaN features.
            _s1_nan = any(
                p.data.isnan().any().item()
                for p in list(self.compressor.parameters()) + list(self.fusion.parameters())
            )
            if _s1_nan:
                _logger.error(
                    "[LMC] iter %d: compressor/fusion has NaN weights after S1 — "
                    "skipping S2. Likely cause: fp16 overflow in attention. "
                    "Retry with --use_half False to use fp32 throughout.",
                    it + 1,
                )
                continue

            # --- Head reset before Stage 2 ---
            self._maybe_reset_head(it)

            # --- Stage 2 ---
            buf_size = self.buffer_size_final if is_last else None
            _logger.info(f"[S2] Filling buffer"
                         f" (size={'FINAL ' + str(self.buffer_size_final) if is_last else 'default'})")
            self.create_training_buffer(buffer_size_override=buf_size)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            self._setup_s2_optimizer_and_schedule(it, is_last)
            _logger.info(f"[S2] Training head for {self.options.epochs} epochs")
            for self.epoch in range(self.options.epochs):
                self.run_epoch()

            # --- Iteration checkpoint + eval ---
            iter_ckpt = self.options.output_map.parent / f"{self.options.output_map.stem}.iter_{it+1:02d}.tmp.pt"
            self.save_model(iter_ckpt)
            eval_result = None
            score = -float('inf')
            if self.eval_each_iteration:
                try:
                    eval_result = self._evaluate_checkpoint(iter_ckpt, it)
                    score = self._score_eval(eval_result)
                except Exception as e:
                    _logger.warning("[Eval] iteration %d failed: %s", it + 1, e, exc_info=True)
            else:
                score = float(it + 1)

            is_best = (score > self.best_score)
            if is_best:
                self.best_score = score
                self.best_iter = it + 1
                self.best_eval = eval_result
                os.replace(iter_ckpt, self.options.output_map)
                best_ckpt_exists = True
                _logger.info("[Best] Updated best checkpoint at iter %d (score=%.4f)", it + 1, score)
                self._write_best_checkpoint_meta(it + 1, score, eval_result)
            else:
                if iter_ckpt.exists():
                    if self.keep_best_only:
                        iter_ckpt.unlink()
                    else:
                        iter_ckpt.rename(self.options.output_map.parent / f"{self.options.output_map.stem}.iter_{it+1:02d}.pt")

            elapsed_s = time.time() - iter_start
            self._log_iteration_summary(it, s1_steps, is_last, is_best, score, eval_result, elapsed_s)

        if not best_ckpt_exists:
            self.save_model(self.options.output_map)

        # 训练结束后立即释放 buffer 等大块显存，便于同进程内后续用 GPU 跑 eval
        self._free_training_gpu_memory()

        _logger.info("Done. Total time: %.1fs | best_iter=%s | best_score=%.4f", time.time() - self.training_start, self.best_iter, self.best_score)
        _logger.info("Logs: %s | %s | %s", self.step_log_path, self.training_log_path, self.eval_log_path)

    def _free_training_gpu_memory(self):
        """Release training buffer and other large GPU tensors so eval can run in the same process."""
        if self.training_buffer is not None:
            for k in list(self.training_buffer.keys()):
                del self.training_buffer[k]
            self.training_buffer = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        _logger.info("Freed training buffer and cleared GPU cache for post-train eval.")

    # ------------------------------------------------------------------
    # Override: run_epoch / training_step (S2: head-only, step_eff for ReproLoss)
    # ------------------------------------------------------------------

    def run_epoch(self):
        """Use actual buffer size (e.g. buffer_size_final on last iter); step iteration and S2 counters.
        When buffer is on CPU (buffer_on_cpu=True), each batch is moved to GPU here to avoid OOM.
        """
        if not self.use_lmc or self.optimizer_head is None:
            return super().run_epoch()
        torch.backends.cudnn.benchmark = True
        buf = self.training_buffer
        buffer_len = buf['features'].shape[0]
        buf_device = buf['features'].device
        # Randperm on same device as buffer so indexing is cheap (no cross-device)
        random_indices = torch.randperm(buffer_len, generator=self.training_generator, device=buf_device)
        for batch_start in range(0, buffer_len, self.options.batch_size):
            batch_end = batch_start + self.options.batch_size
            if batch_end > buffer_len:
                continue
            random_batch_indices = random_indices[batch_start:batch_end]
            # Slice on buffer device, then move to training device if buffer is on CPU
            def _to_dev(t):
                out = t.contiguous()
                if out.device != self.device:
                    out = out.to(self.device, non_blocking=True)
                return out
            self.training_step(
                _to_dev(buf['features'][random_batch_indices]),
                _to_dev(buf['target_px'][random_batch_indices]),
                _to_dev(buf['gt_poses_inv'][random_batch_indices]),
                _to_dev(buf['intrinsics'][random_batch_indices]),
                _to_dev(buf['intrinsics_inv'][random_batch_indices]),
            )
            self.iteration += 1
            self.global_s2_step += 1
            self.local_s2_step += 1

    def training_step(self, features_bC, target_px_b2, gt_inv_poses_b34, Ks_b33, invKs_b33):
        """When LMC S2: use step_eff for ReproLoss and head-only optimizer/scheduler."""
        if not self.use_lmc or self.optimizer_head is None:
            return super().training_step(features_bC, target_px_b2, gt_inv_poses_b34, Ks_b33, invKs_b33)

        batch_size = features_bC.shape[0]
        channels = features_bC.shape[1]
        h, w = 16, batch_size // 16
        if h * w != batch_size:
            batch_size = h * w
            if batch_size <= 0:
                return None
            features_bC = features_bC[:batch_size]
            target_px_b2 = target_px_b2[:batch_size]
            gt_inv_poses_b34 = gt_inv_poses_b34[:batch_size]
            Ks_b33 = Ks_b33[:batch_size]
            invKs_b33 = invKs_b33[:batch_size]
        features_bCHW = features_bC.view(1, h, w, channels).permute(0, 3, 1, 2)

        # step_eff: 每轮 S2 开始时小幅回拨，再随 local_s2_step 指数恢复至全局轨道
        rewind = self.s2_rewind_amount * math.exp(-self.local_s2_step / max(1e-6, self.s2_repro_rewind_tau))
        step_eff = max(0, self.global_s2_step - rewind)
        step_eff = min(step_eff, self.repro_loss.total_iterations - 1)

        with autocast("cuda", enabled=self.options.use_half):
            pred_scene_coords_b3HW = self.regressor.get_scene_coordinates(features_bCHW)

        pred_scene_coords_b31 = pred_scene_coords_b3HW.permute(0, 2, 3, 1).flatten(0, 2).unsqueeze(-1).float()
        pred_scene_coords_b41 = to_homogeneous(pred_scene_coords_b31)
        pred_cam_coords_b31 = torch.bmm(gt_inv_poses_b34, pred_scene_coords_b41)
        pred_px_b31 = torch.bmm(Ks_b33, pred_cam_coords_b31)
        pred_px_b31[:, 2].clamp_(min=self.options.depth_min)
        pred_px_b21 = pred_px_b31[:, :2] / pred_px_b31[:, 2, None]

        reprojection_error_b2 = pred_px_b21.squeeze() - target_px_b2
        reprojection_error_b1 = torch.norm(reprojection_error_b2, dim=1, keepdim=True, p=1)

        invalid_min_depth_b1 = (pred_cam_coords_b31[:, 2] < self.options.depth_min).squeeze(-1)
        invalid_repro_b1 = (reprojection_error_b1 > self.options.repro_loss_hard_clamp).squeeze(-1)
        invalid_max_depth_b1 = (pred_cam_coords_b31[:, 2] > self.options.depth_max).squeeze(-1)
        finite_repro_b1 = torch.isfinite(reprojection_error_b1).squeeze(-1)
        finite_cam_b1 = torch.isfinite(pred_cam_coords_b31).all(dim=1).squeeze(-1)
        invalid_mask_b1 = invalid_min_depth_b1 | invalid_repro_b1 | invalid_max_depth_b1 | (~finite_repro_b1) | (~finite_cam_b1)
        # Ensure strict 1D mask (B,) to avoid accidental broadcast to (B,B)
        n_batch = int(reprojection_error_b1.shape[0])
        invalid_mask_b1 = invalid_mask_b1.reshape(-1)
        if invalid_mask_b1.numel() != n_batch:
            _logger.warning(
                "[S2] invalid_mask_b1 numel mismatch: got=%d expected=%d; slicing to expected length.",
                int(invalid_mask_b1.numel()), n_batch,
            )
            invalid_mask_b1 = invalid_mask_b1[:n_batch]
        valid_mask_b1 = ~invalid_mask_b1

        valid_reprojection_error_b1 = reprojection_error_b1[valid_mask_b1]
        if valid_reprojection_error_b1.numel() > 0:
            loss_valid = self.repro_loss.compute(valid_reprojection_error_b1, int(step_eff))
            if not isinstance(loss_valid, torch.Tensor):
                loss_valid = torch.tensor(loss_valid, device=features_bC.device, dtype=features_bC.dtype)
        else:
            loss_valid = torch.zeros((), device=features_bC.device, dtype=features_bC.dtype)

        pixel_grid_crop_b31 = to_homogeneous(target_px_b2.unsqueeze(2))
        target_camera_coords_b31 = self.options.depth_target * torch.bmm(invKs_b33, pixel_grid_crop_b31)
        # Match (B, 3, 1) camera-coordinate tensor shape via broadcastable mask.
        invalid_mask_b11 = invalid_mask_b1.reshape(n_batch, 1, 1)
        loss_invalid = torch.abs(target_camera_coords_b31 - pred_cam_coords_b31).masked_select(invalid_mask_b11).sum()
        loss_invalid = torch.nan_to_num(loss_invalid, nan=0.0, posinf=0.0, neginf=0.0)
        loss = (loss_valid + loss_invalid) / batch_size

        # Guard NaN/Inf: head reset or bad buffer can make loss non-finite; skip update and log
        loss_for_log = float(loss.item()) if torch.isfinite(loss).all() else -1.0  # step log 用，-1 表示被置零
        if not torch.isfinite(loss).all():
            loss = torch.zeros((), device=loss.device, dtype=loss.dtype)
            _logger.debug(
                "S2 step %d: non-finite loss (valid_frac=%.2f), zeroing loss and skipping effective update.",
                self.iteration, float(valid_mask_b1.sum() / batch_size),
            )

        self.optimizer_head.zero_grad(set_to_none=True)
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer_head)
        self.scaler.update()
        self.scheduler_head.step()

        if self.iteration % self.iterations_output == 0:
            time_since_start = time.time() - self.training_start
            fraction_valid = float(valid_mask_b1.sum() / batch_size)
            # Log pxErr without masking nan/inf as 0: report mean of finite values and naninf count.
            finite_pxerr = torch.isfinite(reprojection_error_b1)
            pxerr_naninf_count = int((~finite_pxerr).sum().item())
            if finite_pxerr.any():
                px_err_finite = float(reprojection_error_b1[finite_pxerr].mean().item())
                px_err_for_log = px_err_finite
            else:
                px_err_finite = float('nan')
                px_err_for_log = -1.0  # Sentinel so we don't hide instability as 0
            lr = float(self.optimizer_head.param_groups[0]["lr"])
            self._append_step_log(
                iter_idx=self.current_lmc_iter,
                step=self.iteration,
                stage="S2",
                loss=loss_for_log,
                px_err=px_err_for_log,
                lr=lr,
                mode="reproj",
                med3d=-1.0,
            )
            _logger.info(
                'Iteration: {:6d} | S2 global={:6d} local={:6d} step_eff={:.0f} | Epoch {:03d}|{:03d}, Loss: {:.4f}, Valid: {:.1f}%, pxErr_finite: {:.2f}, pxerr_naninf: {:d}, Time: {:.2f}s'.format(
                    self.iteration,
                    self.global_s2_step,
                    self.local_s2_step,
                    step_eff,
                    self.epoch,
                    self.options.epochs,
                    loss_for_log if loss_for_log >= 0 else 0.0,
                    fraction_valid * 100,
                    px_err_finite if math.isfinite(px_err_finite) else -1.0,
                    pxerr_naninf_count,
                    time_since_start,
                )
            )
        return loss

    # ------------------------------------------------------------------
    # Override: save_model (includes LMC weights + config)
    # ------------------------------------------------------------------

    def save_model(self, output_path):
        """Save checkpoint. When LMC is active, includes compressor + fusion."""
        if not self.use_lmc:
            return super().save_model(output_path)

        head_sd = self.regressor.heads.state_dict()
        for k in list(head_sd.keys()):
            head_sd[k] = head_sd[k].half()

        checkpoint = {
            'head_state_dict': head_sd,
            'compressor_state_dict': self.compressor.state_dict(),
            'fusion_state_dict': self.fusion.state_dict(),
            'mean_cam_center': self.dataset.mean_cam_center,
            'lmc_config': self.lmc_config,
        }
        torch.save(checkpoint, output_path)
        _logger.info(f"Saved LMC checkpoint to: {output_path}")
