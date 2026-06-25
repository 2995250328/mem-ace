# trainer_dinov2_lmc.py
# Two-stage iterative trainer with Latent Memory Compression.
# Extends TrainerACEDINOv2 — when use_lmc=False, degrades to vanilla DINO ACE.
# Memory loading mirrors map-anything/tasks/ace/utils.py load_memory_features.

import gc
import copy
import json
import logging
import math
import os
import random
import sys
import time
from pathlib import Path
from types import MethodType, SimpleNamespace
from typing import Any, Dict, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import torch.nn.functional as Fnn
import torch.optim as optim
import torchvision.transforms.functional as TF
from skimage import io as skio
from skimage.transform import rotate, resize
from torch.amp import autocast
from torch.utils.data import DataLoader, sampler
from tqdm import tqdm

from ace_network_dinov2 import Head, Regressor
from ace_util import to_homogeneous
from trainer_dinov2 import TrainerACEDINOv2, set_seed
from ace_compressor import GeoLMC
from ace_fusion import LMCFeatureFusion
from ace_loss import ReproLoss
from ace_lmc_global_film import ACEGlobalFiLMHead
from ace_lmc_global_residual import ACEGlobalResidualHead
from ace_network_ace import RegressorACE
from dataset_ace_fcn_lmc import CamLocDatasetACEFCNLMC
from relative_depth_distillation import RelativeDepthDistiller
from utils_lmc import (
    _normalize_scene_tag,
    estimate_memory_front_visibility,
    load_memory_features,
    preflight_memory_features,
)

from glace_backend import (
    GLACEDecoderFeatureResidualAdapter,
    build_glace_camloc_dataset,
    create_glace_regressor_from_encoder,
    create_glace_regressor_from_split_state_dict,
    get_glace_head_class,
)

_logger = logging.getLogger(__name__)


class TrainerACEDINOv2LMC(TrainerACEDINOv2):
    """ACE DINOv2 trainer with optional Latent Memory Compression.

    When ``use_lmc`` is False (or memory_path is None), the trainer
    degrades to the vanilla single-stage TrainerACEDINOv2.
    """

    BUFFER_SCHEMA_SPECS = {
        "fused_buffer": {
            "description": "S2 iterative fused feature buffer",
            "fields": {
                "features": {"rank": 2, "shape_suffix": ("feature_dim",), "dtype": "feature"},
                "target_px": {"rank": 2, "shape_suffix": (2,), "dtype": torch.float32},
                "gt_poses_inv": {"rank": 3, "shape_suffix": (3, 4), "dtype": torch.float32},
                "intrinsics": {"rank": 3, "shape_suffix": (3, 3), "dtype": torch.float32},
                "intrinsics_inv": {"rank": 3, "shape_suffix": (3, 3), "dtype": torch.float32},
                "gt_scene_coords_world": {"rank": 2, "shape_suffix": (3,), "dtype": torch.float32},
                "gt_scene_coords_valid": {"rank": 2, "shape_suffix": (1,), "dtype": torch.bool},
            },
        },
        "raw_buffer": {
            "description": "ACE-G and S1 raw backbone feature buffer",
            "fields": {
                "features": {"rank": 2, "shape_suffix": ("feature_dim",), "dtype": "feature"},
                "target_px": {"rank": 2, "shape_suffix": (2,), "dtype": torch.float32},
                "gt_poses_inv": {"rank": 3, "shape_suffix": (3, 4), "dtype": torch.float32},
                "intrinsics": {"rank": 3, "shape_suffix": (3, 3), "dtype": torch.float32},
                "intrinsics_inv": {"rank": 3, "shape_suffix": (3, 3), "dtype": torch.float32},
                "gt_scene_coords_world": {"rank": 2, "shape_suffix": (3,), "dtype": torch.float32},
                "gt_scene_coords_valid": {"rank": 2, "shape_suffix": (1,), "dtype": torch.bool},
            },
        },
        "fused_buffer_glace": {
            "description": "S2 iterative fused feature buffer for GLACE backend",
            "fields": {
                "features": {"rank": 2, "shape_suffix": ("feature_dim",), "dtype": "feature"},
                "target_px": {"rank": 2, "shape_suffix": (2,), "dtype": torch.float32},
                "gt_poses_inv": {"rank": 3, "shape_suffix": (3, 4), "dtype": torch.float32},
                "intrinsics": {"rank": 3, "shape_suffix": (3, 3), "dtype": torch.float32},
                "intrinsics_inv": {"rank": 3, "shape_suffix": (3, 3), "dtype": torch.float32},
                "gt_scene_coords_world": {"rank": 2, "shape_suffix": (3,), "dtype": torch.float32},
                "gt_scene_coords_valid": {"rank": 2, "shape_suffix": (1,), "dtype": torch.bool},
                "img_idx": {"rank": 1, "shape_suffix": (), "dtype": torch.int64},
            },
        },
        "raw_buffer_glace": {
            "description": "ACE-G and S1 raw backbone feature buffer for GLACE backend",
            "fields": {
                "features": {"rank": 2, "shape_suffix": ("feature_dim",), "dtype": "feature"},
                "target_px": {"rank": 2, "shape_suffix": (2,), "dtype": torch.float32},
                "gt_poses_inv": {"rank": 3, "shape_suffix": (3, 4), "dtype": torch.float32},
                "intrinsics": {"rank": 3, "shape_suffix": (3, 3), "dtype": torch.float32},
                "intrinsics_inv": {"rank": 3, "shape_suffix": (3, 3), "dtype": torch.float32},
                "gt_scene_coords_world": {"rank": 2, "shape_suffix": (3,), "dtype": torch.float32},
                "gt_scene_coords_valid": {"rank": 2, "shape_suffix": (1,), "dtype": torch.bool},
                "img_idx": {"rank": 1, "shape_suffix": (), "dtype": torch.int64},
            },
        },
    }

    @staticmethod
    def _tensor_to_config_value(value: Any):
        """Convert tensors inside nested metadata to checkpoint-friendly Python values."""
        if isinstance(value, torch.Tensor):
            if value.numel() == 1:
                return float(value.detach().cpu().item())
            return value.detach().cpu().tolist()
        if isinstance(value, dict):
            return {k: TrainerACEDINOv2LMC._tensor_to_config_value(v) for k, v in value.items()}
        if isinstance(value, list):
            return [TrainerACEDINOv2LMC._tensor_to_config_value(v) for v in value]
        if isinstance(value, tuple):
            return [TrainerACEDINOv2LMC._tensor_to_config_value(v) for v in value]
        return value

    def _resolve_lmc_geometry_scene_scale(self, needs_scene_scale: bool):
        """Resolve the fixed scene scale used by PE/fusion geometry normalization."""
        source = str(getattr(self.options, "lmc_fusion_scene_scale_source", "memory_points_p95"))
        if not needs_scene_scale:
            return 1.0, "unused_raw_geometry"

        if source == "fixed":
            value = getattr(self.options, "lmc_fusion_scene_scale_value", None)
            if value is None:
                raise ValueError("[LMC-Geometry] fixed scene scale requires --lmc_fusion_scene_scale_value.")
            scale = float(value)
        elif source == "memory_points_p95":
            pooled_points = self.memory_dict["pooled_points"].detach().float()
            scene_center = self.memory_dict["scene_center"].detach().float()
            if scene_center.ndim == 1:
                scene_center = scene_center.unsqueeze(0)
            if scene_center.ndim == 2:
                scene_center = scene_center.unsqueeze(1)
            centered = pooled_points - scene_center
            radius = torch.linalg.norm(centered, dim=-1).flatten()
            if radius.numel() == 0:
                raise ValueError("[LMC-Geometry] Cannot compute scene_scale from empty pooled_points.")
            scale = float(torch.quantile(radius, 0.95).item())
        else:
            raise ValueError(f"[LMC-Geometry] Unsupported lmc_fusion_scene_scale_source={source!r}")

        if (not torch.isfinite(torch.tensor(scale)).item()) or scale <= 1e-6:
            raise ValueError(f"[LMC-Geometry] Invalid scene_scale={scale!r} from source={source!r}")
        return scale, source

    @staticmethod
    def _capture_module_training_modes(*modules):
        """Capture exact train/eval flags for modules that may be toggled temporarily."""
        snapshot = []
        seen = set()
        for module in modules:
            if module is None:
                continue
            module_id = id(module)
            if module_id in seen:
                continue
            seen.add(module_id)
            snapshot.append((module, bool(module.training)))
        return snapshot

    @staticmethod
    def _restore_module_training_modes(snapshot):
        """Restore module train/eval flags captured by _capture_module_training_modes."""
        for module, was_training in snapshot:
            module.train(was_training)

    @staticmethod
    def _count_trainable_params(module):
        if module is None:
            return 0
        return sum(p.numel() for p in module.parameters() if p.requires_grad)

    @staticmethod
    def _count_total_params(module):
        if module is None:
            return 0
        return sum(p.numel() for p in module.parameters())

    @staticmethod
    def _optimizer_contains_module_params(optimizer, module):
        if optimizer is None or module is None:
            return False
        module_param_ids = {id(p) for p in module.parameters()}
        for group in optimizer.param_groups:
            for param in group.get("params", []):
                if id(param) in module_param_ids:
                    return True
        return False

    @staticmethod
    def _optimizer_group_param_count(group):
        return sum(param.numel() for param in group.get("params", []))

    def _log_optimizer_groups(self, stage_tag: str, optimizer):
        if optimizer is None:
            return
        for group_idx, group in enumerate(optimizer.param_groups):
            params = group.get("params", [])
            group_name = group.get("name", f"group{group_idx}")
            _logger.info(
                "[%s] optimizer group %d (%s): lr=%.3e tensors=%d params=%d",
                stage_tag,
                group_idx,
                group_name,
                float(group.get("lr", 0.0)),
                len(params),
                self._optimizer_group_param_count(group),
            )

    def _validate_glace_head_optimizer_membership(self, stage_tag: str, optimizer, *, expected_trainable: bool):
        if not self._is_glace_backend():
            return
        head = getattr(self.regressor, "heads", None)
        head_trainable_params = self._count_trainable_params(head)
        head_in_optimizer = self._optimizer_contains_module_params(optimizer, head)
        if expected_trainable and (head_trainable_params <= 0 or not head_in_optimizer):
            raise RuntimeError(
                f"[{stage_tag}] GLACE head was expected to be trainable, but "
                f"trainable_params={head_trainable_params} optimizer_contains_head={head_in_optimizer}."
            )
        if (not expected_trainable) and head_in_optimizer and head_trainable_params > 0:
            raise RuntimeError(
                f"[{stage_tag}] GLACE head was expected to be frozen, but trainable head params are in the optimizer."
            )

    def _get_training_generator(self, device: Optional[torch.device] = None) -> torch.Generator:
        """Return the training RNG that matches the target tensor device."""
        target_device = torch.device(device) if device is not None else self.device
        if target_device.type == "cuda":
            if not hasattr(self, "_training_generator_cuda") or self._training_generator_cuda is None:
                self._training_generator_cuda = torch.Generator(device=target_device).manual_seed(self.base_seed + 8191)
            return self._training_generator_cuda
        if not hasattr(self, "_training_generator_cpu") or self._training_generator_cpu is None:
            self._training_generator_cpu = torch.Generator().manual_seed(self.base_seed + 8191)
        return self._training_generator_cpu

    def _build_reference_contract_state(
        self,
        *,
        memory_contract_mode: str,
        reference_index: Optional[int],
        conditioning_reference: Dict[str, Any],
        normalization_ref: Dict[str, Any],
        scene_center_world: Optional[torch.Tensor],
        scene_center_ref: Optional[torch.Tensor],
        scene_center_ref_norm: Optional[torch.Tensor],
    ) -> Dict[str, Any]:
        """Prepare the runtime contract that interprets head outputs."""
        contract_mode = str(memory_contract_mode or "C0").upper()
        state: Dict[str, Any] = {
            "enabled": contract_mode == "C1",
            "contract_mode": contract_mode,
            "reference_index": reference_index,
            "output_space": "points_world",
            "conditioning_reference": {},
            "normalization_ref": {},
            "head_mean": None,
        }
        if contract_mode != "C1":
            return state

        T_ref_c2w_world = conditioning_reference.get("T_ref_c2w_world")
        if T_ref_c2w_world is None:
            raise ValueError("[LMC] contract_mode=C1 requires conditioning_reference.T_ref_c2w_world.")
        if not isinstance(T_ref_c2w_world, torch.Tensor):
            T_ref_c2w_world = torch.as_tensor(T_ref_c2w_world, dtype=torch.float32, device=self.device)
        else:
            T_ref_c2w_world = T_ref_c2w_world.to(self.device).float()
        if tuple(T_ref_c2w_world.shape) != (4, 4):
            raise ValueError(
                f"[LMC] conditioning_reference.T_ref_c2w_world has invalid shape "
                f"{tuple(T_ref_c2w_world.shape)}, expected (4, 4)."
            )

        T_world_to_ref = conditioning_reference.get("T_world_to_ref")
        if T_world_to_ref is not None:
            if not isinstance(T_world_to_ref, torch.Tensor):
                T_world_to_ref = torch.as_tensor(T_world_to_ref, dtype=torch.float32, device=self.device)
            else:
                T_world_to_ref = T_world_to_ref.to(self.device).float()

        mu_ref = normalization_ref.get("mu_ref")
        sigma_ref = normalization_ref.get("sigma_ref")
        ref_norm_alpha = float(normalization_ref.get("alpha", 1.0) or 1.0)
        has_ref_norm = mu_ref is not None and sigma_ref is not None
        if has_ref_norm:
            if not isinstance(mu_ref, torch.Tensor):
                mu_ref = torch.as_tensor(mu_ref, dtype=torch.float32, device=self.device)
            else:
                mu_ref = mu_ref.to(self.device).float()
            sigma_ref = float(torch.as_tensor(sigma_ref, dtype=torch.float32).item())
            if (
                not math.isfinite(sigma_ref)
                or sigma_ref <= 1e-12
                or not math.isfinite(ref_norm_alpha)
                or ref_norm_alpha <= 0
            ):
                has_ref_norm = False

        if has_ref_norm:
            output_space = "points_ref_norm"
            if isinstance(scene_center_ref_norm, torch.Tensor):
                head_mean = scene_center_ref_norm.to(self.device).float().view(-1)[:3]
            else:
                head_mean = torch.zeros(3, device=self.device, dtype=torch.float32)
        else:
            output_space = "points_ref"
            if isinstance(scene_center_ref, torch.Tensor):
                head_mean = scene_center_ref.to(self.device).float().view(-1)[:3]
            elif isinstance(mu_ref, torch.Tensor):
                head_mean = mu_ref
            else:
                head_mean = torch.zeros(3, device=self.device, dtype=torch.float32)

        state.update({
            "output_space": output_space,
            "conditioning_reference": {
                "reference_index": reference_index,
                "T_ref_c2w_world": T_ref_c2w_world,
                "T_world_to_ref": T_world_to_ref,
            },
            "normalization_ref": {
                "mu_ref": mu_ref if has_ref_norm else None,
                "sigma_ref": sigma_ref if has_ref_norm else None,
                "alpha": ref_norm_alpha if has_ref_norm else 1.0,
            },
            "head_mean": head_mean.detach().clone().float(),
        })
        return state

    def _recover_pred_scene_to_training_world(self, pred_scene_B3HW: torch.Tensor) -> torch.Tensor:
        """Map head outputs to the world coordinate space used by the reprojection loss."""
        contract = getattr(self, "reference_contract_state", None)
        if not contract or not contract.get("enabled", False):
            return pred_scene_B3HW

        pred_world = pred_scene_B3HW
        if contract["output_space"] == "points_ref_norm":
            mu_ref = contract["normalization_ref"]["mu_ref"].to(pred_world.device, dtype=pred_world.dtype)
            sigma_ref = float(contract["normalization_ref"]["sigma_ref"])
            ref_norm_alpha = float(contract["normalization_ref"].get("alpha", 1.0) or 1.0)
            pred_world = pred_world * (sigma_ref / ref_norm_alpha) + mu_ref.view(1, 3, 1, 1)

        T_ref_c2w_world = contract["conditioning_reference"]["T_ref_c2w_world"].to(
            pred_world.device, dtype=pred_world.dtype
        )
        R_ref = T_ref_c2w_world[:3, :3]
        C_ref = T_ref_c2w_world[:3, 3]
        pred_world = torch.einsum("ij,bjhw->bihw", R_ref, pred_world) + C_ref.view(1, 3, 1, 1)

        # Reprojection still operates in the trainer's current world space.
        if self.coord_sigma is not None and self.coord_mu is not None:
            mu = self.coord_mu.to(pred_world.device, dtype=pred_world.dtype).view(1, 3, 1, 1)
            pred_world = (pred_world - mu) / float(self.coord_sigma)
        return pred_world

    def _recover_pred_scene_to_reference(self, pred_scene_B3HW: torch.Tensor) -> Optional[torch.Tensor]:
        """Recover C1 head outputs to raw reference-frame coordinates."""
        contract = getattr(self, "reference_contract_state", None)
        if not contract or not contract.get("enabled", False):
            return None

        pred_ref = pred_scene_B3HW
        if contract["output_space"] == "points_ref_norm":
            mu_ref = contract["normalization_ref"]["mu_ref"]
            sigma_ref = contract["normalization_ref"]["sigma_ref"]
            ref_norm_alpha = float(contract["normalization_ref"].get("alpha", 1.0) or 1.0)
            if mu_ref is None or sigma_ref is None:
                return None
            mu_ref = mu_ref.to(pred_ref.device, dtype=pred_ref.dtype)
            pred_ref = pred_ref * (float(sigma_ref) / ref_norm_alpha) + mu_ref.view(1, 3, 1, 1)
        return pred_ref

    def _world_to_reference(self, points_world_N3: torch.Tensor) -> Optional[torch.Tensor]:
        """Map raw world scene coordinates to raw reference-frame coordinates."""
        contract = getattr(self, "reference_contract_state", None)
        if not contract or not contract.get("enabled", False):
            return None

        T_world_to_ref = contract["conditioning_reference"].get("T_world_to_ref")
        if T_world_to_ref is None:
            T_ref_c2w_world = contract["conditioning_reference"]["T_ref_c2w_world"]
            R_ref = T_ref_c2w_world[:3, :3]
            C_ref = T_ref_c2w_world[:3, 3]
            return torch.einsum("ij,nj->ni", R_ref.transpose(0, 1), points_world_N3 - C_ref.view(1, 3))

        T_world_to_ref = T_world_to_ref.to(points_world_N3.device, dtype=points_world_N3.dtype)
        R = T_world_to_ref[:3, :3]
        t = T_world_to_ref[:3, 3]
        return torch.einsum("ij,nj->ni", R, points_world_N3) + t.view(1, 3)

    def _compute_c1_aux_ref_loss(
        self,
        pred_scene_B3HW: torch.Tensor,
        gt_scene_coords_world_N3: Optional[torch.Tensor],
        gt_scene_coords_valid_N1: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Auxiliary metric loss in raw reference-frame coordinates.

        Callers pass predictions after _recover_pred_scene_to_training_world(),
        because the main ACE reprojection loss uses that same coordinate space.
        """
        weight = float(getattr(self.options, "c1_aux_ref_loss_weight", 0.0))
        if weight <= 0.0 or gt_scene_coords_world_N3 is None:
            return pred_scene_B3HW.new_zeros(())

        pred_world_B3HW = pred_scene_B3HW
        if self.coord_sigma is not None and self.coord_mu is not None:
            mu = self.coord_mu.to(pred_world_B3HW.device, dtype=pred_world_B3HW.dtype).view(1, 3, 1, 1)
            pred_world_B3HW = pred_world_B3HW * float(self.coord_sigma) + mu

        pred_world_N3 = pred_world_B3HW.permute(0, 2, 3, 1).reshape(-1, 3).float()
        pred_ref_N3 = self._world_to_reference(pred_world_N3)
        if pred_ref_N3 is None:
            return pred_scene_B3HW.new_zeros(())
        gt_world_N3 = gt_scene_coords_world_N3.reshape(-1, 3).to(pred_ref_N3.device, dtype=pred_ref_N3.dtype)
        gt_ref_N3 = self._world_to_reference(gt_world_N3)
        if gt_ref_N3 is None:
            return pred_scene_B3HW.new_zeros(())

        valid_mask = torch.isfinite(pred_ref_N3).all(dim=1) & torch.isfinite(gt_ref_N3).all(dim=1)
        valid_mask &= (gt_world_N3.abs().sum(dim=1) > 0)
        if gt_scene_coords_valid_N1 is not None:
            valid_mask &= gt_scene_coords_valid_N1.reshape(-1).to(valid_mask.device).bool()
        if not bool(valid_mask.any().item()):
            return pred_scene_B3HW.new_zeros(())

        aux_loss = Fnn.smooth_l1_loss(
            pred_ref_N3[valid_mask],
            gt_ref_N3[valid_mask],
            reduction="mean",
            beta=1.0,
        )
        return aux_loss * weight

    def _relative_depth_enabled_for_stage(self, stage_tag: str) -> bool:
        if not bool(getattr(self.options, "use_relative_depth_loss", False)):
            return False
        apply_to = str(getattr(self.options, "relative_depth_apply_to", "stage2") or "stage2").lower()
        stage = str(stage_tag).upper()
        if apply_to == "all":
            return stage in {"S2", "S2-G"}
        if apply_to == "stage2_g":
            return stage == "S2-G"
        return stage in {"S2", "S2-G"}

    def _relative_depth_image_path(self, image_idx: int) -> Optional[Path]:
        rgb_files = getattr(self.dataset, "rgb_files", None)
        if rgb_files is None:
            return None
        if image_idx < 0 or image_idx >= len(rgb_files):
            return None
        return Path(rgb_files[image_idx])

    def _resolve_batch_image_indices(self, raw_img_idx, *, device=None) -> Optional[torch.Tensor]:
        if raw_img_idx is None:
            return None
        if torch.is_tensor(raw_img_idx):
            return raw_img_idx.to(device or self.device, non_blocking=True).long()
        if not hasattr(self, "_image_path_to_dataset_index"):
            rgb_files = list(getattr(self.dataset, "rgb_files", []) or [])
            self._image_path_to_dataset_index = {str(Path(path)): int(i) for i, path in enumerate(rgb_files)}
        mapping = self._image_path_to_dataset_index
        if isinstance(raw_img_idx, (str, Path)):
            raw_items = [raw_img_idx]
        else:
            raw_items = list(raw_img_idx)
        indices = []
        missing = []
        for item in raw_items:
            key = str(Path(item))
            idx = mapping.get(key)
            if idx is None:
                missing.append(key)
            else:
                indices.append(idx)
        if missing:
            preview = ", ".join(missing[:3])
            raise ValueError(f"[Buffer] could not map image path(s) to dataset indices: {preview}")
        return torch.tensor(indices, dtype=torch.int64, device=device or self.device)

    def _init_relative_depth_distiller(self) -> None:
        if not bool(getattr(self.options, "use_relative_depth_loss", False)):
            return
        checkpoint = getattr(self.options, "relative_depth_teacher_checkpoint", None)
        if checkpoint is None:
            raise ValueError(
                "--relative_depth_teacher_checkpoint is required when --use_relative_depth_loss True."
            )
        self.relative_depth_distiller = RelativeDepthDistiller(
            teacher=str(getattr(self.options, "relative_depth_teacher", "depth_anything_v2_online")),
            teacher_checkpoint=Path(checkpoint),
            teacher_encoder=str(getattr(self.options, "relative_depth_teacher_encoder", "vitb")),
            teacher_input_size=int(getattr(self.options, "relative_depth_teacher_input_size", 518)),
            image_height=int(getattr(self.options, "image_resolution", 512)),
            image_path_getter=self._relative_depth_image_path,
            device=self.device,
            pair_weight=float(getattr(self.options, "relative_depth_pair_weight", 0.5)),
            max_samples=int(getattr(self.options, "relative_depth_max_samples", 1024)),
            max_pairs=int(getattr(self.options, "relative_depth_max_pairs", 4096)),
            min_points=int(getattr(self.options, "relative_depth_min_points", 16)),
            depth_min=1e-3,
            depth_max=1000.0,
            cache_size=int(getattr(self.options, "relative_depth_teacher_cache_size", 256)),
        )
        _logger.info(
            "[RelDepth] enabled apply_to=%s weight=%.4f phase_start_ratio=%.3f lmc_start_iter=%d lmc_start_ratio=%.3f pair_weight=%.3f",
            str(getattr(self.options, "relative_depth_apply_to", "stage2")),
            float(getattr(self.options, "relative_depth_loss_weight", 0.05)),
            float(getattr(self.options, "relative_depth_start_ratio", 0.3)),
            int(getattr(self.options, "relative_depth_start_lmc_iteration", 0)),
            float(getattr(self.options, "relative_depth_start_lmc_ratio", 0.0)),
            float(getattr(self.options, "relative_depth_pair_weight", 0.5)),
        )

    def _relative_depth_weight(self) -> float:
        base_weight = float(getattr(self.options, "relative_depth_loss_weight", 0.0) or 0.0)
        if base_weight <= 0.0:
            return 0.0

        current_lmc_iter = int(getattr(self, "current_lmc_iteration", 0))
        start_lmc_iter = max(0, int(getattr(self.options, "relative_depth_start_lmc_iteration", 0) or 0))
        if current_lmc_iter < start_lmc_iter:
            return 0.0

        start_lmc_ratio = float(getattr(self.options, "relative_depth_start_lmc_ratio", 0.0) or 0.0)
        start_lmc_ratio = min(max(start_lmc_ratio, 0.0), 0.99)
        if start_lmc_ratio > 0.0:
            lmc_progress = float(current_lmc_iter) / float(max(1, int(getattr(self, "lmc_iterations", 1))))
            if lmc_progress < start_lmc_ratio:
                return 0.0

        start_ratio = float(getattr(self.options, "relative_depth_start_ratio", 0.0) or 0.0)
        start_ratio = min(max(start_ratio, 0.0), 0.99)
        phase = float(self.local_s2_step) / float(max(1, self.steps_per_s2_phase))
        if phase < start_ratio:
            return 0.0
        if start_ratio <= 0.0:
            return base_weight
        ramp = min(1.0, max(0.0, (phase - start_ratio) / max(1e-6, 1.0 - start_ratio)))
        return base_weight * ramp

    def _build_relative_depth_image_dataloader(self):
        effective_batch = max(1, int(getattr(self.options, "relative_depth_image_batch_size", 1)))
        if self._is_ace_fcn_backend() and effective_batch != 1:
            _logger.info("[RelDepth] Forcing image batch size to 1 for variable-width ACE images.")
            effective_batch = 1
        buffer_image_width = getattr(self.options, "buffer_image_width", None)
        if buffer_image_width is None:
            buffer_image_width = (self.options.image_resolution * 4 // 3 + 13) // 14 * 14
        depth_dataset = self._build_train_dataset(
            image_width=buffer_image_width,
            augment=False,
            aug_rotation=0,
            aug_scale_max=1.0,
            aug_scale_min=1.0,
        )
        batch_sampler = sampler.BatchSampler(
            sampler.RandomSampler(depth_dataset, generator=self.batch_generator),
            batch_size=effective_batch,
            drop_last=False,
        )
        return DataLoader(
            dataset=depth_dataset,
            batch_sampler=batch_sampler,
            generator=self.loader_generator,
            pin_memory=True,
            num_workers=self.num_data_loader_workers,
            persistent_workers=self.num_data_loader_workers > 0,
        )

    def _next_relative_depth_image_batch(self, stage_tag: str):
        if self.relative_depth_distiller is None or not self._relative_depth_enabled_for_stage(stage_tag):
            return None
        if self._relative_depth_weight() <= 0.0:
            return None
        interval = int(getattr(self.options, "relative_depth_image_step_interval", 10))
        if interval <= 0:
            raise ValueError("--relative_depth_image_step_interval must be > 0.")
        if (int(self.local_s2_step) + 1) % interval != 0:
            return None
        if self._relative_depth_image_loader is None:
            self._relative_depth_image_loader = self._build_relative_depth_image_dataloader()
            self._relative_depth_image_iterator = iter(self._relative_depth_image_loader)
            _logger.info(
                "[RelDepth] image-level supervision active: stage=%s interval=%d batch=%d",
                stage_tag,
                interval,
                int(getattr(self.options, "relative_depth_image_batch_size", 1)),
            )
        try:
            return next(self._relative_depth_image_iterator)
        except StopIteration:
            self._relative_depth_image_iterator = iter(self._relative_depth_image_loader)
            return next(self._relative_depth_image_iterator)

    def _compute_relative_depth_reprojection_gate(
        self,
        *,
        pred_cam_N31: torch.Tensor,
        intrinsics_B33: torch.Tensor,
        valid_mask_B1HW: torch.Tensor,
        H: int,
        W: int,
    ) -> Dict[str, Any]:
        """Detached same-view reprojection diagnostics for image-level relative-depth loss."""
        B = int(intrinsics_B33.shape[0])
        device = pred_cam_N31.device
        pixel_grid_B2HW = self.pixel_grid_2HW[:, :H, :W].to(device=device).unsqueeze(0).expand(B, 2, H, W)
        target_px_N2 = pixel_grid_B2HW.permute(0, 2, 3, 1).reshape(B * H * W, 2).float()
        Ks_N33 = intrinsics_B33.float().unsqueeze(1).expand(B, H * W, 3, 3).reshape(B * H * W, 3, 3)

        with torch.no_grad():
            pred_cam = pred_cam_N31.detach().float()
            pred_px_N31 = torch.bmm(Ks_N33, pred_cam)
            z_proj_N1 = pred_px_N31[:, 2, 0:1].clamp_min(float(getattr(self.options, "depth_min", 1e-3)))
            pred_px_N2 = pred_px_N31[:, :2, 0] / z_proj_N1
            repro_N2 = pred_px_N2 - target_px_N2
            repro_l1_N = torch.norm(repro_N2, dim=1, p=1)
            repro_l2_N = torch.norm(repro_N2, dim=1, p=2)

            z_N = pred_cam[:, 2, 0]
            finite_cam_N = torch.isfinite(pred_cam).all(dim=1).flatten()
            finite_repro_N = torch.isfinite(repro_l2_N)
            depth_valid_N = (
                finite_cam_N
                & torch.isfinite(z_N)
                & (z_N > float(getattr(self.options, "depth_min", 1e-3)))
                & (z_N < float(getattr(self.options, "depth_max", 1000.0)))
            )
            base_ray_valid_N = finite_repro_N & depth_valid_N
            threshold = max(0.0, float(getattr(self.options, "relative_depth_reprojection_threshold_px", 4.0)))
            hard_ray_valid_N = base_ray_valid_N & (repro_l2_N < threshold)

            sigma = max(1e-6, float(getattr(self.options, "relative_depth_reprojection_sigma_px", 4.0)))
            ray_weight_N = torch.exp(-torch.square(repro_l2_N / sigma))
            ray_weight_N = torch.where(base_ray_valid_N, ray_weight_N, torch.zeros_like(ray_weight_N))

            image_valid_BHW = valid_mask_B1HW.bool().view(B, H, W)
            image_valid_N = image_valid_BHW.reshape(-1)
            finite_eval_N = image_valid_N & finite_repro_N
            finite_l1 = repro_l1_N[finite_eval_N]
            finite_l2 = repro_l2_N[finite_eval_N]
            image_valid_count = int(image_valid_N.sum().item())
            ray_count = int((image_valid_N & hard_ray_valid_N).sum().item())
            depth_invalid_count = int((image_valid_N & ~depth_valid_N).sum().item())

            stats = {
                "reproj_l1": float(finite_l1.mean().cpu().item()) if finite_l1.numel() > 0 else 0.0,
                "reproj_l2": float(finite_l2.mean().cpu().item()) if finite_l2.numel() > 0 else 0.0,
                "reproj_p90": float(torch.quantile(finite_l2.float(), 0.9).cpu().item()) if finite_l2.numel() > 0 else 0.0,
                "pre_ray_coverage": float(ray_count) / float(max(1, image_valid_count)),
                "depth_invalid": float(depth_invalid_count) / float(max(1, image_valid_count)),
            }

        return {
            "hard_ray_valid_BHW": hard_ray_valid_N.view(B, H, W).detach(),
            "ray_weight_BHW": ray_weight_N.view(B, H, W).detach(),
            "stats": stats,
        }

    def _compute_image_relative_depth_loss(self, *, stage_tag: str, image_batch):
        param = next(self.regressor.heads.parameters())
        zero = param.new_zeros(())
        if image_batch is None:
            return zero, {"enabled": 0.0, "weight": 0.0, "loss_raw": 0.0}
        weight = self._relative_depth_weight()
        if weight <= 0.0:
            return zero, {"enabled": 1.0, "weight": 0.0, "loss_raw": 0.0}

        image_BCHW, image_mask_B1HW, _, gt_pose_inv_B44, intrinsics_B33, *_ = image_batch
        raw_img_idx = image_batch[-1]
        image_BCHW = image_BCHW.to(self.device, non_blocking=True)
        image_mask_B1HW = image_mask_B1HW.to(self.device, non_blocking=True)
        gt_pose_inv_B44 = gt_pose_inv_B44.to(self.device, non_blocking=True).float()
        intrinsics_B33 = intrinsics_B33.to(self.device, non_blocking=True).float()
        img_idx_B = self._resolve_batch_image_indices(raw_img_idx, device=self.device)
        if img_idx_B is None:
            raise ValueError("[RelDepth] image-level batch is missing image identity/path metadata.")
        if image_BCHW.dtype == torch.float16:
            image_BCHW = image_BCHW.float()
        gt_pose_inv_B34 = gt_pose_inv_B44[:, :3, :] if gt_pose_inv_B44.shape[1] == 4 else gt_pose_inv_B44

        with autocast("cuda", enabled=self.options.use_half):
            with torch.no_grad():
                local_features_BCHW = self.regressor.get_features(image_BCHW)
            if self._is_glace_backend():
                raw_features_BCHW = self._build_glace_decoder_feature_maps(
                    local_features_BCHW,
                    img_idx_B,
                    stage_tag=f"{stage_tag}-RelDepth",
                )
            else:
                raw_features_BCHW = local_features_BCHW

            compressor_out = self._s2_compressor_out
            if compressor_out is None:
                compressor_out = self._compress_memory()
            if stage_tag == "S2-G" and not bool(getattr(self.options, "ace_g_fusion_in_s2", False)):
                with torch.no_grad():
                    fused_BCHW, base_BCHW = self._fuse_lmc_features_for_head(
                        raw_features_BCHW,
                        compressor_out,
                        stage_tag=f"{stage_tag}-RelDepth",
                    )
            else:
                fused_BCHW, base_BCHW = self._fuse_lmc_features_for_head(
                    raw_features_BCHW,
                    compressor_out,
                    stage_tag=f"{stage_tag}-RelDepth",
                )
            head_features_BCHW = self._mix_glace_decoder_feature_maps(
                base_BCHW,
                fused_BCHW,
                stage_tag=f"{stage_tag}-RelDepth",
            )
            B, C, H, W = head_features_BCHW.shape
            head_features_bC = head_features_BCHW.permute(0, 2, 3, 1).reshape(B * H * W, C)
            if self._uses_ace_lmc_global_residual_head():
                pred_scene_B3HW, _, _, _ = self._predict_ace_lmc_global_residual_coords(
                    head_features_bC,
                    img_idx_B[:, None].expand(B, H * W).reshape(-1),
                    H,
                    W,
                    stage_tag=f"{stage_tag}-RelDepth",
                )
            elif self._uses_ace_lmc_global_film_head():
                pred_scene_B3HW, _ = self._predict_ace_lmc_global_film_coords(
                    head_features_bC,
                    img_idx_B[:, None].expand(B, H * W).reshape(-1),
                    H,
                    W,
                    stage_tag=f"{stage_tag}-RelDepth",
                )
            else:
                if self._uses_ace_lmc_concat_head():
                    head_features_BCHW = self._append_ace_lmc_global_to_feature_maps(
                        head_features_BCHW,
                        img_idx_B,
                        stage_tag=f"{stage_tag}-RelDepth",
                    )
                pred_scene_B3HW = self.regressor.get_scene_coordinates(head_features_BCHW)

        pred_scene_B3HW = self._recover_pred_scene_to_training_world(pred_scene_B3HW).float()
        pred_scene_N31 = pred_scene_B3HW.permute(0, 2, 3, 1).reshape(B * H * W, 3, 1)
        gt_pose_inv_N34 = gt_pose_inv_B34[:, None, None].expand(B, H, W, 3, 4).reshape(B * H * W, 3, 4)
        pred_cam_N31 = torch.bmm(gt_pose_inv_N34, to_homogeneous(pred_scene_N31))
        student_depth_BHW = pred_cam_N31[:, 2, 0].view(B, H, W)
        valid_mask_B1HW = TF.resize(
            image_mask_B1HW.float(),
            [H, W],
            interpolation=TF.InterpolationMode.NEAREST,
        ).bool()

        gate_mode = str(getattr(self.options, "relative_depth_reprojection_gate", "none") or "none").lower()
        log_reprojection = bool(getattr(self.options, "relative_depth_log_reprojection_stats", False))
        ray_valid_mask_BHW = None
        ray_weight_BHW = None
        repro_stats: Dict[str, float] = {}
        if gate_mode != "none" or log_reprojection:
            repro_info = self._compute_relative_depth_reprojection_gate(
                pred_cam_N31=pred_cam_N31,
                intrinsics_B33=intrinsics_B33,
                valid_mask_B1HW=valid_mask_B1HW,
                H=H,
                W=W,
            )
            repro_stats = dict(repro_info.get("stats", {}))
            if gate_mode == "hard":
                ray_valid_mask_BHW = repro_info["hard_ray_valid_BHW"]
            elif gate_mode == "soft":
                # First-step plumbing only: soft weights are logged/passed through,
                # but ImageRelativeDepthLoss keeps legacy loss weighting unchanged.
                ray_weight_BHW = repro_info["ray_weight_BHW"]
            elif gate_mode != "none":
                raise ValueError(f"Unsupported --relative_depth_reprojection_gate={gate_mode!r}")

        raw_loss, stats = self.relative_depth_distiller.forward_image(
            student_depth_BHW=student_depth_BHW,
            valid_mask_B1HW=valid_mask_B1HW,
            img_idx_B=img_idx_B,
            generator=self._training_generator_cuda,
            ray_valid_mask_BHW=ray_valid_mask_BHW,
            ray_weight_BHW=ray_weight_BHW,
            student_space=str(getattr(self.options, "relative_depth_student_space", "neg_z_legacy")),
            min_ray_coverage=(
                float(getattr(self.options, "relative_depth_min_ray_coverage", 0.05))
                if gate_mode == "hard"
                else 0.0
            ),
        )
        stats = dict(stats)
        stats.update(repro_stats)
        stats["weight"] = float(weight)
        if not bool(torch.isfinite(raw_loss).all().item()):
            if not self._relative_depth_warned_nonfinite:
                _logger.warning("[RelDepth] non-finite image loss at %s; skipped.", stage_tag)
                self._relative_depth_warned_nonfinite = True
            stats["loss_raw"] = -1.0
            return zero, stats
        if not self._relative_depth_logged_active and float(stats.get("groups", 0.0)) > 0.0:
            _logger.info(
                "[RelDepth] image-level active at %s: raw=%.6f weight=%.6f valid_points=%d images=%d missing=%d",
                stage_tag,
                float(stats.get("loss_raw", 0.0)),
                float(weight),
                int(float(stats.get("valid_points", 0.0))),
                int(float(stats.get("groups", 0.0))),
                int(float(stats.get("missing_images", 0.0))),
            )
            self._relative_depth_logged_active = True
        return raw_loss * weight, stats

    @staticmethod
    def _format_relative_depth_stats(stats: Optional[Dict[str, float]]) -> str:
        if not stats or float(stats.get("enabled", 0.0)) <= 0.0:
            return ""
        text = (
            f", relD={float(stats.get('loss_raw', 0.0)):.4f}"
            f", relW={float(stats.get('weight', 0.0)):.4f}"
            f", relPts={int(float(stats.get('valid_points', 0.0)))}"
            f", relGrp={int(float(stats.get('groups', 0.0)))}"
        )
        if "image_coverage" in stats:
            text += f", relImgCov={float(stats.get('image_coverage', 0.0)):.3f}"
        if "ray_coverage" in stats:
            text += f", relRayCov={float(stats.get('ray_coverage', 0.0)):.3f}"
        if "reproj_l1" in stats:
            text += f", relReprojL1={float(stats.get('reproj_l1', 0.0)):.2f}"
        if "reproj_l2" in stats:
            text += f", relReprojL2={float(stats.get('reproj_l2', 0.0)):.2f}"
        if "reproj_p90" in stats:
            text += f", relReprojP90={float(stats.get('reproj_p90', 0.0)):.2f}"
        if "depth_invalid" in stats:
            text += f", relDepthInv={float(stats.get('depth_invalid', 0.0)):.3f}"
        if "gate_skipped" in stats and float(stats.get('gate_skipped', 0.0)) > 0.0:
            text += f", relGateSkip={int(float(stats.get('gate_skipped', 0.0)))}"
        return text

    def _extract_gt_scene_coords_from_batch(self, batch, image_BCHW: Optional[torch.Tensor] = None):
        """Return GT world scene coords from dataset batch when the backend provides them."""
        if not isinstance(batch, (list, tuple)):
            return None
        for item in reversed(batch):
            if not torch.is_tensor(item):
                continue
            if image_BCHW is not None and item is image_BCHW:
                continue
            if item.ndim == 4 and item.shape[1] == 3:
                return item
        return None

    def _scene_coords_valid_mask(self, coords_B3HW: Optional[torch.Tensor], size_hw=None, device=None):
        if coords_B3HW is None:
            if size_hw is None:
                return None
            H, W = size_hw
            return torch.zeros((1, 1, H, W), dtype=torch.bool, device=device or self.device)
        valid_B1HW = torch.isfinite(coords_B3HW).all(dim=1, keepdim=True)
        valid_B1HW &= coords_B3HW.abs().sum(dim=1, keepdim=True) > 0
        if size_hw is not None and tuple(valid_B1HW.shape[-2:]) != tuple(size_hw):
            valid_B1HW = Fnn.interpolate(valid_B1HW.float(), size=size_hw, mode="nearest") > 0.5
        return valid_B1HW

    def _expand_valid_coord_sampling_mask(self, coord_valid_B1HW, image_mask_B1HW=None):
        """Expand valid-depth patches to nearby DINO patches for buffer sampling only."""
        if coord_valid_B1HW is None:
            return None
        radius = int(getattr(self.options, "buffer_valid_coord_neighbor_radius", 1))
        mode = str(getattr(self.options, "buffer_valid_coord_neighbor_mode", "cross") or "cross").lower()
        valid = coord_valid_B1HW.bool()
        if radius <= 0 or mode == "none" or valid.numel() == 0 or not bool(valid.any().item()):
            expanded = valid
        elif mode == "square":
            kernel_size = radius * 2 + 1
            expanded = Fnn.max_pool2d(
                valid.float(),
                kernel_size=kernel_size,
                stride=1,
                padding=radius,
            ) > 0.5
        elif mode == "cross":
            offsets = []
            for dy in range(-radius, radius + 1):
                for dx in range(-radius, radius + 1):
                    if abs(dy) + abs(dx) <= radius:
                        offsets.append((dy, dx))
            kernel = torch.zeros(
                (1, 1, radius * 2 + 1, radius * 2 + 1),
                dtype=valid.dtype if valid.is_floating_point() else torch.float32,
                device=valid.device,
            )
            for dy, dx in offsets:
                kernel[0, 0, dy + radius, dx + radius] = 1.0
            expanded = Fnn.conv2d(valid.float(), kernel, padding=radius) > 0.5
        else:
            raise ValueError(
                f"Unsupported buffer_valid_coord_neighbor_mode={mode!r}; expected none/cross/square."
            )

        if image_mask_B1HW is not None:
            expanded = expanded & image_mask_B1HW.bool()

        seed_count = int((valid & image_mask_B1HW.bool()).sum().item()) if image_mask_B1HW is not None else int(valid.sum().item())
        expanded_count = int(expanded.sum().item())
        self._buffer_sample_valid_coord_seed_available = (
            int(getattr(self, "_buffer_sample_valid_coord_seed_available", 0)) + seed_count
        )
        self._buffer_sample_valid_coord_roi_available = (
            int(getattr(self, "_buffer_sample_valid_coord_roi_available", 0)) + expanded_count
        )
        self._buffer_sample_valid_coord_neighbor_available = (
            int(getattr(self, "_buffer_sample_valid_coord_neighbor_available", 0)) + max(0, expanded_count - seed_count)
        )
        return expanded

    def _sample_buffer_indices(self, image_mask_N1, coord_valid_N1, features_to_select, use_replacement):
        image_weights = image_mask_N1.view(-1).float()
        sample_generator = self._get_training_generator(image_weights.device)
        if features_to_select <= 0:
            return None
        if float(image_weights.sum().item()) <= 0.0:
            return None

        prefer_valid_coords = bool(getattr(self.options, "buffer_sample_valid_coords", True))
        aux_weight = float(getattr(self.options, "c1_aux_ref_loss_weight", 0.0))
        prefer_valid_coords = prefer_valid_coords or aux_weight > 0.0
        coord_weights = coord_valid_N1.view(-1).float() if coord_valid_N1 is not None else None
        if not prefer_valid_coords or coord_weights is None or float(coord_weights.sum().item()) <= 0.0:
            self._buffer_sample_random_selected = (
                int(getattr(self, "_buffer_sample_random_selected", 0)) + int(features_to_select)
            )
            return torch.multinomial(
                image_weights,
                features_to_select,
                replacement=use_replacement,
                generator=sample_generator,
            )

        valid_coord_idx = torch.where((coord_weights > 0) & (image_weights > 0))[0]
        if valid_coord_idx.numel() <= 0:
            self._buffer_sample_random_selected = (
                int(getattr(self, "_buffer_sample_random_selected", 0)) + int(features_to_select)
            )
            return torch.multinomial(
                image_weights,
                features_to_select,
                replacement=use_replacement,
                generator=sample_generator,
            )

        valid_ratio = float(getattr(self.options, "buffer_valid_coord_sample_ratio", 1.0))
        valid_ratio = min(1.0, max(0.0, valid_ratio))
        n_valid_target = features_to_select if valid_ratio >= 1.0 else int(math.ceil(features_to_select * valid_ratio))
        n_aux = min(int(valid_coord_idx.numel()), features_to_select, max(0, n_valid_target))
        pieces = []
        if n_aux > 0:
            coord_generator = self._get_training_generator(valid_coord_idx.device)
            perm = torch.randperm(valid_coord_idx.numel(), device=valid_coord_idx.device, generator=coord_generator)
            pieces.append(valid_coord_idx[perm[:n_aux]])
            self._buffer_sample_valid_coord_selected = int(getattr(self, "_buffer_sample_valid_coord_selected", 0)) + n_aux
        self._buffer_sample_valid_coord_available = (
            int(getattr(self, "_buffer_sample_valid_coord_available", 0)) + int(valid_coord_idx.numel())
        )
        n_random = features_to_select - n_aux
        if n_random > 0:
            random_weights = image_weights
            if not use_replacement and n_aux > 0:
                random_weights = image_weights.clone()
                random_weights[pieces[0]] = 0.0
            if float(random_weights.sum().item()) <= 0.0:
                return pieces[0] if pieces else None
            pieces.append(torch.multinomial(
                random_weights,
                n_random,
                replacement=use_replacement,
                generator=sample_generator,
            ))
            self._buffer_sample_random_selected = int(getattr(self, "_buffer_sample_random_selected", 0)) + n_random
        return torch.cat(pieces, dim=0) if len(pieces) > 1 else pieces[0]

    def _infer_c1_aux_depth_dir(self, train_root: Path) -> Optional[Path]:
        explicit_root = getattr(self.options, "c1_aux_depth_root", None)
        depth_kind = str(getattr(self.options, "c1_aux_depth_kind", "gt_depth") or "gt_depth")
        if getattr(self.options, "data_backend", "ace") == "wai":
            return self._resolve_wai_aux_depth_dir(train_root, explicit_root, depth_kind)
        if explicit_root is not None and str(explicit_root):
            explicit_root = Path(explicit_root)
            if explicit_root.name in {"gt_depth", "colmap_depth", "sparse_depth"}:
                return explicit_root
            if depth_kind == "sparse_depth":
                # RIO10 sparse-depth root stores scans as sceneXX_seqXX_seqXX_YY/sparse_depth.
                scene_name = Path(train_root).parent.name
                sparse_scan_name = self._derive_rio10_sparse_scan_name(scene_name)
                candidates = [
                    explicit_root / sparse_scan_name / "sparse_depth",
                    explicit_root / scene_name / "sparse_depth",
                    explicit_root / "sparse_depth",
                ]
                for candidate in candidates:
                    if candidate.exists():
                        return candidate
                return candidates[0]
            return explicit_root / depth_kind

        scene_name = train_root.parent.name
        data_root = Path(os.environ.get("ACE_DATA_ROOT", "/home/xwh/data"))
        candidates = [
            data_root / "mapanything-dataset" / "wai_data" / "indoor6" / f"{scene_name}_train" / depth_kind,
            data_root / "mapanything-dataset" / "wai_data" / "indoor6" / scene_name / depth_kind,
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return candidates[0]

    @staticmethod
    def _load_wai_depth_npy(path: Path) -> np.ndarray:
        path = Path(path)
        if not path.exists():
            alt = None
            if path.suffix.lower() == ".npy":
                alt = path.with_suffix(".npz")
            elif path.suffix.lower() == ".npz":
                alt = path.with_suffix(".npy")
            if alt is not None and alt.exists():
                path = alt

        if path.suffix.lower() == ".npy":
            depth = np.load(path, allow_pickle=False).astype(np.float64, copy=False)
        elif path.suffix.lower() == ".npz":
            depth = np.load(path, allow_pickle=False)["arr_0"].astype(np.float64, copy=False)
        else:
            depth_raw = skio.imread(path)
            depth = np.asarray(depth_raw).astype(np.float64, copy=False)
            if path.suffix.lower() in {".png", ".tif", ".tiff"} and np.issubdtype(np.asarray(depth_raw).dtype, np.integer):
                # RIO10 sparse_depth PNGs are stored as millimeters. ACE reprojection expects meters.
                depth = depth / 1000.0
        if depth.ndim == 3:
            depth = np.squeeze(depth)
        if depth.ndim != 2:
            raise ValueError(f"Unsupported aux depth shape {depth.shape} in {path}")
        depth = np.where(np.isfinite(depth), depth, 0.0)
        return depth

    @staticmethod
    def _sample_patch_depth_nearest_valid(
        depth: np.ndarray,
        stride: int,
        coords_h: int,
        coords_w: int,
        *,
        depth_min: float = 1e-6,
        depth_max: float = 1000.0,
    ):
        """Return one representative valid depth pixel per patch, preferring the patch center."""
        patch_depth = np.zeros((coords_h, coords_w), dtype=np.float64)
        patch_px = np.zeros((coords_h, coords_w), dtype=np.float64)
        patch_py = np.zeros((coords_h, coords_w), dtype=np.float64)
        patch_valid = np.zeros((coords_h, coords_w), dtype=bool)

        image_h, image_w = depth.shape
        half = int(stride) // 2
        for gy in range(coords_h):
            cy = min(gy * stride + half, image_h - 1)
            y0 = gy * stride
            y1 = min((gy + 1) * stride, image_h)
            for gx in range(coords_w):
                cx = min(gx * stride + half, image_w - 1)
                x0 = gx * stride
                x1 = min((gx + 1) * stride, image_w)

                center_depth = depth[cy, cx]
                if np.isfinite(center_depth) and depth_min <= center_depth <= depth_max:
                    patch_depth[gy, gx] = center_depth
                    patch_px[gy, gx] = cx
                    patch_py[gy, gx] = cy
                    patch_valid[gy, gx] = True
                    continue

                window = depth[y0:y1, x0:x1]
                valid = np.isfinite(window) & (window >= depth_min) & (window <= depth_max)
                if not np.any(valid):
                    continue
                yy, xx = np.where(valid)
                abs_y = yy + y0
                abs_x = xx + x0
                best = int(np.argmin((abs_y - cy) ** 2 + (abs_x - cx) ** 2))
                py = int(abs_y[best])
                px = int(abs_x[best])
                patch_depth[gy, gx] = depth[py, px]
                patch_px[gy, gx] = px
                patch_py[gy, gx] = py
                patch_valid[gy, gx] = True
        return patch_depth, patch_px, patch_py, patch_valid

    @staticmethod
    def _depth_to_patch_scene_coords(
        depth: np.ndarray,
        pose: torch.Tensor,
        image_hw,
        focal_length,
        centre_point,
        output_subsample=None,
        depth_min: float = 1e-6,
        depth_max: float = 1000.0,
    ) -> torch.Tensor:
        """Project resized metric depth to patch-level raw world scene coordinates."""
        image_h, image_w = image_hw
        if tuple(depth.shape) != (image_h, image_w):
            depth = resize(
                depth,
                (image_h, image_w),
                order=0,
                preserve_range=True,
                anti_aliasing=False,
            )

        stride = int(output_subsample or Regressor.OUTPUT_SUBSAMPLE)
        coords_h = math.ceil(image_h / stride)
        coords_w = math.ceil(image_w / stride)
        depth_patch, px, py, valid = TrainerACEDINOv2LMC._sample_patch_depth_nearest_valid(
            depth,
            stride,
            coords_h,
            coords_w,
            depth_min=depth_min,
            depth_max=depth_max,
        )

        coords = torch.zeros((3, coords_h, coords_w), dtype=torch.float32)
        if not np.any(valid):
            return coords

        if centre_point:
            fx = float(focal_length[0])
            fy = float(focal_length[1])
            cx = float(centre_point[0])
            cy = float(centre_point[1])
        else:
            fx = float(focal_length)
            fy = float(focal_length)
            cx = image_w / 2.0
            cy = image_h / 2.0

        eye = np.ones((4, depth_patch.shape[0], depth_patch.shape[1]), dtype=np.float64)
        eye[0] = ((px - cx) / fx) * depth_patch
        eye[1] = ((py - cy) / fy) * depth_patch
        eye[2] = depth_patch
        eye[:, ~valid] = 0.0

        scene = np.matmul(pose.numpy(), eye.reshape(4, -1)).reshape(4, depth_patch.shape[0], depth_patch.shape[1])
        coords[:, : scene.shape[1], : scene.shape[2]] = torch.from_numpy(scene[:3]).float()
        return coords

    @staticmethod
    def _frame_intrinsics_from_meta(frame: Dict[str, Any]) -> np.ndarray:
        return np.asarray(
            [
                [float(frame["fl_x"]), 0.0, float(frame["cx"])],
                [0.0, float(frame["fl_y"]), float(frame["cy"])],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

    @staticmethod
    def _pose_key(pose: np.ndarray, decimals: int = 5):
        return tuple(np.round(pose.reshape(-1), decimals=decimals).tolist())

    def _build_aux_depth_map_from_scene_meta(self, dataset, depth_dir: Path):
        scene_root = depth_dir.parent
        meta_path = scene_root / "scene_meta.json"
        if not meta_path.exists():
            return None

        depth_kind = str(getattr(self.options, "c1_aux_depth_kind", depth_dir.name) or depth_dir.name)
        with meta_path.open("r", encoding="utf-8") as f:
            frames = json.load(f).get("frames", [])
        if not frames:
            return None

        pose_to_frames: Dict[tuple, list[Dict[str, Any]]] = {}
        meta_poses = []
        for frame in frames:
            if "transform_matrix" not in frame or depth_kind not in frame:
                continue
            pose = np.asarray(frame["transform_matrix"], dtype=np.float64)
            if pose.shape != (4, 4):
                continue
            pose_to_frames.setdefault(self._pose_key(pose), []).append(frame)
            meta_poses.append((pose.reshape(-1), frame))

        if not meta_poses:
            return None

        depth_paths = [None] * len(getattr(dataset, "rgb_files", []))
        matched = 0
        fallback_matched = 0
        bad_intrinsics = 0
        missing_depth = 0
        used_frame_names = set()
        meta_pose_mat = np.stack([p for p, _ in meta_poses], axis=0)

        rgb_files = list(getattr(dataset, "rgb_files", []))
        calibration_files = list(getattr(dataset, "calibration_files", []))
        pose_files = getattr(dataset, "pose_files", None)
        pose_values = getattr(dataset, "pose_values", None)
        pose_files = list(pose_files) if pose_files is not None else None

        for real_idx in range(len(rgb_files)):
            if real_idx >= len(calibration_files):
                continue
            if pose_files is not None and real_idx < len(pose_files):
                ace_pose = np.loadtxt(pose_files[real_idx]).astype(np.float64)
            elif pose_values is not None and real_idx < len(pose_values):
                ace_pose = np.asarray(pose_values[real_idx], dtype=np.float64)
            else:
                continue
            ace_k = np.loadtxt(calibration_files[real_idx]).astype(np.float64)
            if ace_pose.shape != (4, 4) or ace_k.shape != (3, 3):
                continue

            candidates = pose_to_frames.get(self._pose_key(ace_pose), [])
            chosen = None
            if candidates:
                chosen = min(
                    candidates,
                    key=lambda fr: float(np.max(np.abs(self._frame_intrinsics_from_meta(fr) - ace_k))),
                )
            else:
                # Rare truncation/rounding mismatch: fall back to nearest pose with a strict tolerance.
                pose_diffs = np.max(np.abs(meta_pose_mat - ace_pose.reshape(1, -1)), axis=1)
                best_idx = int(np.argmin(pose_diffs))
                if float(pose_diffs[best_idx]) <= 1e-4:
                    chosen = meta_poses[best_idx][1]
                    fallback_matched += 1

            if chosen is None:
                continue

            k_diff = float(np.max(np.abs(self._frame_intrinsics_from_meta(chosen) - ace_k)))
            if k_diff > 1e-2:
                bad_intrinsics += 1
                continue

            rel_depth = Path(chosen[depth_kind])
            depth_path = scene_root / rel_depth
            if not depth_path.exists():
                missing_depth += 1
                continue
            depth_paths[real_idx] = depth_path
            matched += 1
            used_frame_names.add(str(chosen.get("frame_name", rel_depth.stem)))

        _logger.info(
            "[LMC] C1 aux depth scene_meta alignment: meta=%s, matched=%d/%d, fallback_pose=%d, "
            "bad_intrinsics=%d, missing_depth=%d, unique_meta_frames=%d",
            meta_path,
            matched,
            len(depth_paths),
            fallback_matched,
            bad_intrinsics,
            missing_depth,
            len(used_frame_names),
        )
        if matched == 0:
            raise ValueError(
                f"[LMC] C1 aux depth scene_meta alignment found no pose/calibration matches: {meta_path}"
            )
        return depth_paths

    @staticmethod
    def _derive_rio10_sparse_scan_name(scene_name: str) -> str:
        base = str(scene_name)
        for suffix in ("_train", "_test", "_val"):
            if base.endswith(suffix):
                base = base[:-len(suffix)]
                break
        parts = base.split("_")
        if len(parts) >= 3 and parts[0].startswith("scene") and parts[1].startswith("seq"):
            return f"{parts[0]}_{parts[1]}_{parts[1]}_{parts[2]}"
        return base

    def _resolve_wai_aux_depth_dir(self, scene_root: Path, explicit_root, depth_kind: str) -> Path:
        scene_root = Path(scene_root)
        if explicit_root is None or not str(explicit_root):
            return scene_root / depth_kind

        root = Path(explicit_root)
        if root.name == depth_kind:
            return root
        if (root / "scene_meta.json").exists():
            return root / depth_kind
        if (root / depth_kind).exists() and not any((root / child).exists() for child in ("logs", "metadata")):
            return root / depth_kind
        if depth_kind == "sparse_depth" and (root / "sparse_depth").exists():
            return root / "sparse_depth"

        sparse_scan_name = self._derive_rio10_sparse_scan_name(scene_root.name)
        candidates = [
            root / sparse_scan_name / depth_kind,
            root / sparse_scan_name / "sparse_depth",
            root / scene_root.name / depth_kind,
            root / depth_kind,
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return candidates[0]

    @staticmethod
    def _wai_frame_sparse_depth_stems(frame_name: str):
        stems = []
        text = str(frame_name)
        if "frame-" in text:
            stems.append(text[text.index("frame-"):])
        stems.append(Path(text).stem)
        deduped = []
        for item in stems:
            if item and item not in deduped:
                deduped.append(item)
        return deduped

    def _build_wai_depth_paths_by_index(self, dataset, scene_root: Path, depth_dir: Path):
        depth_kind = str(getattr(self.options, "c1_aux_depth_kind", depth_dir.name) or depth_dir.name)
        frames = getattr(dataset, "scene_meta", {}).get("frames", [])
        frame_meta = {str(f.get("frame_name")): f for f in frames if "frame_name" in f}
        depth_paths = [None] * len(getattr(dataset, "frame_names", []))
        matched = 0
        for real_idx, frame_name in enumerate(getattr(dataset, "frame_names", [])):
            candidates = []
            meta = frame_meta.get(str(frame_name))
            if isinstance(meta, dict) and depth_kind in meta:
                candidates.append(Path(scene_root) / str(meta[depth_kind]))
            for stem in self._wai_frame_sparse_depth_stems(frame_name):
                candidates.extend([
                    depth_dir / f"{stem}.stable.depth.png",
                    depth_dir / f"{stem}.depth.png",
                    depth_dir / f"{stem}.png",
                    depth_dir / f"{stem}.npy",
                    depth_dir / f"{stem}.npz",
                    depth_dir / f"{stem}.exr",
                ])
            chosen = next((p for p in candidates if p.exists()), None)
            if chosen is not None:
                depth_paths[real_idx] = chosen
                matched += 1
        _logger.info(
            "[LMC] WAI aux depth alignment: scene=%s depth_dir=%s matched=%d/%d depth_kind=%s",
            scene_root, depth_dir, matched, len(depth_paths), depth_kind,
        )
        if matched == 0:
            raise ValueError(f"[LMC] WAI aux depth found no matching frames: scene={scene_root}, depth_dir={depth_dir}")
        return depth_paths

    def _attach_wai_aux_depth_dataset(self, dataset, scene_root: Path, depth_dir: Path):
        if not depth_dir.exists():
            raise FileNotFoundError(
                f"[LMC] WAI aux depth dir is missing: {depth_dir}. "
                "For RIO10 sparse depth, pass --c1_aux_depth_root /data/xwh/RIO10_sparse_depth "
                "--c1_aux_depth_kind sparse_depth."
            )
        depth_paths_by_index = self._build_wai_depth_paths_by_index(dataset, scene_root, depth_dir)
        original_get_single_item = dataset._get_single_item

        def _get_single_item_with_aux_depth(ds, idx, image_height):
            image, image_mask, pose, pose_inv, intrinsics, intrinsics_inv, _coords, rgb_path = original_get_single_item(idx, image_height)
            real_idx = int(ds.valid_file_indices[int(idx)])
            depth_path = depth_paths_by_index[real_idx] if real_idx < len(depth_paths_by_index) else None
            image_hw = (int(image.shape[1]), int(image.shape[2]))
            if depth_path is None:
                depth = np.zeros(image_hw, dtype=np.float64)
            else:
                depth = TrainerACEDINOv2LMC._load_wai_depth_npy(depth_path)
            coords = TrainerACEDINOv2LMC._depth_to_patch_scene_coords(
                depth,
                pose,
                image_hw=image_hw,
                focal_length=[float(intrinsics[0, 0]), float(intrinsics[1, 1])],
                centre_point=[float(intrinsics[0, 2]), float(intrinsics[1, 2])],
                output_subsample=int(getattr(self.regressor, 'OUTPUT_SUBSAMPLE', Regressor.OUTPUT_SUBSAMPLE)),
                depth_min=float(getattr(self.options, 'c1_aux_depth_min', 1e-6)),
                depth_max=float(getattr(self.options, 'c1_aux_depth_max', 1000.0)),
            )
            return image, image_mask, pose, pose_inv, intrinsics, intrinsics_inv, coords, rgb_path

        dataset._lmc_original_get_single_item = original_get_single_item
        dataset._lmc_aux_depth_dir = depth_dir
        dataset._lmc_aux_depth_missing_count = sum(path is None for path in depth_paths_by_index)
        dataset._get_single_item = MethodType(_get_single_item_with_aux_depth, dataset)
        return dataset

    def _attach_ace_aux_depth_dataset(self, dataset, train_root: Path, depth_dir: Path):
        if not depth_dir.exists():
            raise FileNotFoundError(
                f"[LMC] C1 aux_ref requested but aux depth dir is missing: {depth_dir}. "
                "Set --c1_aux_depth_root or provide ACE train/depth."
            )

        rgb_files = getattr(dataset, "rgb_files", [])
        depth_paths_by_index = self._build_aux_depth_map_from_scene_meta(dataset, depth_dir)
        if depth_paths_by_index is None:
            def _depth_match_key(path: Path) -> str:
                stem = path.stem.replace("image-", "")
                for suffix in (".stable.depth", ".depth"):
                    if stem.endswith(suffix):
                        stem = stem[:-len(suffix)]
                        break
                return stem

            depth_files = {
                _depth_match_key(p): p
                for p in depth_dir.iterdir()
                if p.is_file() and p.suffix.lower() in {".npy", ".npz", ".png", ".tif", ".tiff", ".exr"}
            }
            rgb_ids = {Path(p).stem for p in rgb_files}
            matched = sorted(rgb_ids & set(depth_files))
            missing = sorted(rgb_ids - set(depth_files))
            extra = sorted(set(depth_files) - rgb_ids)
            if len(matched) == 0:
                raise ValueError(
                    f"[LMC] C1 aux depth has no overlap with ACE RGB frames: rgb_dir={train_root / 'rgb'}, depth_dir={depth_dir}"
                )
            depth_paths_by_index = [depth_files.get(Path(p).stem) for p in rgb_files]
            _logger.warning(
                "[LMC] C1 aux depth fell back to filename alignment: dir=%s, matched=%d/%d, missing=%d, extra=%d. "
                "Use a WAI scene_meta.json whenever possible.",
                depth_dir,
                len(matched),
                len(rgb_ids),
                len(missing),
                len(extra),
            )
            if missing:
                _logger.warning("[LMC] C1 aux depth missing first frames: %s", missing[:20])

        original_get_single_item = dataset._get_single_item

        def _get_single_item_with_aux_depth(ds, idx, image_height):
            idx = int(idx)
            real_idx = int(ds.valid_file_indices[idx]) if hasattr(ds, "valid_file_indices") else idx
            depth_path = depth_paths_by_index[real_idx] if real_idx < len(depth_paths_by_index) else None

            # Preserve the backend dataset behavior exactly (GLACE global features,
            # image resizing, augmentation, intrinsics, pose handling, etc.). We only
            # replace the coords slot with sparse-depth-derived scene coordinates.
            item = list(original_get_single_item(idx, image_height))
            if len(item) < 7:
                raise ValueError(f"[LMC] ACE aux depth wrapper expected dataset item with coords slot, got len={len(item)}")

            image = item[0]
            pose = item[2]
            intrinsics = item[4]
            if not torch.is_tensor(image) or image.ndim != 3:
                raise ValueError(f"[LMC] ACE aux depth wrapper expected image CHW tensor, got {type(image)} shape={getattr(image, 'shape', None)}")
            image_hw = (int(image.shape[1]), int(image.shape[2]))

            if depth_path is None:
                depth = np.zeros(image_hw, dtype=np.float64)
            else:
                depth = TrainerACEDINOv2LMC._load_wai_depth_npy(depth_path)
                if tuple(depth.shape) != tuple(image_hw):
                    depth = resize(
                        depth,
                        image_hw,
                        order=0,
                        preserve_range=True,
                        anti_aliasing=False,
                    )

            focal_length = [float(intrinsics[0, 0]), float(intrinsics[1, 1])]
            centre_point = [float(intrinsics[0, 2]), float(intrinsics[1, 2])]
            coords = TrainerACEDINOv2LMC._depth_to_patch_scene_coords(
                depth,
                pose,
                image_hw=image_hw,
                focal_length=focal_length,
                centre_point=centre_point,
                output_subsample=int(getattr(self.regressor, 'OUTPUT_SUBSAMPLE', Regressor.OUTPUT_SUBSAMPLE)),
                depth_min=float(getattr(self.options, 'c1_aux_depth_min', 1e-6)),
                depth_max=float(getattr(self.options, 'c1_aux_depth_max', 1000.0)),
            )
            item[6] = coords
            return tuple(item)
        dataset._lmc_original_get_single_item = original_get_single_item
        dataset._lmc_aux_depth_dir = depth_dir
        dataset._lmc_aux_depth_missing_count = sum(path is None for path in depth_paths_by_index)
        dataset._get_single_item = MethodType(_get_single_item_with_aux_depth, dataset)
        return dataset

    def _warn_missing_aux_ref_coords_once(self):
        if float(getattr(self.options, "c1_aux_ref_loss_weight", 0.0)) <= 0.0:
            return
        raise ValueError(
            "[LMC] --c1_aux_ref_loss_weight > 0 but this dataset batch has no GT scene coords. "
            "Auxiliary coordinate supervision requires patch-level scene coordinates from depth. "
            "For indoor6_ace exports, provide train/depth or use a dataset backend that returns coords."
        )

    def _is_glace_backend(self):
        return str(getattr(self.options, "model_backend", "ace_dinov2")) == "glace_lmc"

    def _is_ace_fcn_backend(self):
        return str(getattr(self.options, "model_backend", "ace_dinov2")) == "ace_fcn_lmc"

    def _ace_lmc_global_head_mode(self):
        return str(getattr(self.options, "ace_lmc_global_head_mode", "none"))

    def _uses_ace_lmc_concat_head(self):
        return bool(self._is_ace_fcn_backend() and self._ace_lmc_global_head_mode() == "glace_concat")

    def _uses_ace_lmc_global_residual_head(self):
        return bool(self._is_ace_fcn_backend() and self._ace_lmc_global_head_mode() == "glace_residual")

    def _uses_ace_lmc_global_film_head(self):
        return bool(self._is_ace_fcn_backend() and self._ace_lmc_global_head_mode() == "glace_film")

    def _uses_ace_lmc_global_head(self):
        return bool(
            self._uses_ace_lmc_concat_head()
            or self._uses_ace_lmc_global_residual_head()
            or self._uses_ace_lmc_global_film_head()
        )

    def _uses_image_global_features(self):
        return bool(self._is_glace_backend() or self._uses_ace_lmc_global_head())

    def _ace_lmc_stage2_feature_source(self):
        return str(getattr(self.options, "ace_lmc_stage2_feature_source", "raw_backbone") or "raw_backbone").lower()

    def _ace_lmc_stage2_uses_fused_buffer(self):
        return bool(
            self.lmc_flow == 'ace_g'
            and (self._uses_ace_lmc_concat_head() or self._uses_ace_lmc_global_residual_head())
            and self._ace_lmc_stage2_feature_source() == 'stage1_fused'
        )

    def _ace_lmc_global_normalize(self):
        return bool(getattr(self.options, "ace_lmc_global_normalize", False))

    def _ace_lmc_global_noise_std(self):
        return max(0.0, float(getattr(self.options, "ace_lmc_global_noise_std", 0.0) or 0.0))

    def _needs_image_indices_in_buffer(self):
        return self._uses_image_global_features()

    def _glace_head_freeze_active(self, iteration_idx):
        if not self._is_glace_backend():
            return False
        freeze_iters = max(0, int(getattr(self.options, "glace_head_freeze_iters", 0)))
        return int(iteration_idx) < freeze_iters

    def _set_glace_head_trainable(self, trainable, *, stage_tag):
        if not self._is_glace_backend():
            return
        if self._glace_freeze_head():
            trainable = False
        trainable = bool(trainable)
        for param in self.regressor.heads.parameters():
            param.requires_grad_(trainable)
        self.regressor.heads.train(trainable)
        _logger.info("[%s] GLACE head trainable=%s.", stage_tag, trainable)

    # Legacy compatibility: when glace_freeze_head is unset, this still freezes the head.
    def _glace_freeze_base_network(self):
        return bool(self._is_glace_backend() and getattr(self.options, "glace_freeze_base_network", True))

    def _glace_freeze_encoder(self):
        return bool(self._is_glace_backend() and getattr(self.options, "glace_freeze_encoder", True))

    def _glace_freeze_head(self):
        if not self._is_glace_backend():
            return False
        explicit = getattr(self.options, "glace_freeze_head", None)
        if explicit is None:
            return self._glace_freeze_base_network()
        return bool(explicit)

    def _glace_residual_lr_ratio(self, fallback_ratio):
        ratio = float(getattr(self.options, "glace_residual_lr_ratio", -1.0))
        if ratio < 0:
            return float(fallback_ratio)
        return ratio

    @staticmethod
    def _logit_from_probability(prob):
        prob = min(max(float(prob), 1e-6), 1.0 - 1e-6)
        return math.log(prob / (1.0 - prob))

    def _local_residual_warmup_scale(self):
        warmup_steps = max(0, int(getattr(self, "local_residual_alpha_warmup_steps", 0)))
        if warmup_steps <= 0:
            return 1.0
        return min(1.0, max(0.0, float(getattr(self, "iteration", 0)) / float(warmup_steps)))

    def _current_local_residual_alpha_tensor(self, *, device, dtype):
        mode = str(getattr(self, "local_residual_mode", "none"))
        if mode == "none":
            return torch.ones((), device=device, dtype=dtype)
        warmup = self._local_residual_warmup_scale()
        if mode == "fixed_alpha":
            return torch.tensor(
                float(getattr(self, "local_residual_alpha", 1.0)) * warmup,
                device=device,
                dtype=dtype,
            )
        if mode == "learned_alpha":
            logit = getattr(self, "local_residual_alpha_logit", None)
            if logit is None:
                raise RuntimeError("local_residual_mode=learned_alpha but local_residual_alpha_logit is not initialized.")
            alpha_max = float(getattr(self, "local_residual_alpha_max", getattr(self, "local_residual_alpha", 1.0)))
            return torch.sigmoid(logit.to(device=device, dtype=dtype)) * alpha_max * warmup
        raise ValueError(f"Unsupported local_residual_mode={mode!r}")

    def _current_local_residual_alpha_value(self):
        alpha = self._current_local_residual_alpha_tensor(device=self.device, dtype=torch.float32)
        return float(alpha.detach().cpu().item())

    def _local_residual_gate_params(self):
        param = getattr(self, "local_residual_alpha_logit", None)
        return [param] if isinstance(param, torch.nn.Parameter) else []

    def _glace_residual_adapter_params(self):
        adapter = getattr(self, "glace_residual_adapter", None)
        return list(adapter.parameters()) if isinstance(adapter, torch.nn.Module) else []

    def _log_glace_lmc_config_diagnostics(self):
        if not self._is_glace_backend():
            return
        _logger.info(
            "[GLACE-LMC config] freeze_base_network=%s freeze_encoder=%s freeze_head=%s "
            "lmc_fusion_target=%s effective_lmc_fusion_target=%s local_residual_mode=%s "
            "local_residual_alpha=%.4f alpha_init=%.6f alpha_max=%.6f alpha_warmup=%d ace_g_fusion_in_s2=%s",
            self._glace_freeze_base_network(),
            self._glace_freeze_encoder(),
            self._glace_freeze_head(),
            getattr(self, "requested_lmc_fusion_target", getattr(self.options, "lmc_fusion_target", "decoder")),
            getattr(self, "effective_lmc_fusion_target", getattr(self.options, "lmc_fusion_target", "decoder")),
            getattr(self, "local_residual_mode", getattr(self.options, "local_residual_mode", "none")),
            float(getattr(self, "local_residual_alpha", getattr(self.options, "local_residual_alpha", 1.0))),
            float(getattr(self, "local_residual_alpha_init", getattr(self.options, "local_residual_alpha_init", 0.001))),
            float(getattr(self, "local_residual_alpha_max", getattr(self.options, "local_residual_alpha_max", None) or getattr(self.options, "local_residual_alpha", 1.0))),
            int(getattr(self, "local_residual_alpha_warmup_steps", getattr(self.options, "local_residual_alpha_warmup_steps", 0))),
            bool(getattr(self.options, "ace_g_fusion_in_s2", False)),
        )

    def _select_glace_delta_local_features(self, base_features_bC, head_features_bC):
        target = str(getattr(self, "effective_lmc_fusion_target", "decoder"))
        global_dim = int(getattr(self, "glace_global_feat_dim", 0) or 0)
        encoder_dim = int(getattr(self, "lmc_config", {}).get(
            "encoder_feature_dim",
            getattr(getattr(self.regressor, "encoder", None), "feature_dim", 0) or 0,
        ))
        if target == "decoder":
            if global_dim > 0 and base_features_bC.shape[1] > global_dim and head_features_bC.shape[1] > global_dim:
                return base_features_bC[:, global_dim:], head_features_bC[:, global_dim:], "decoder_local_slice"
            return base_features_bC, head_features_bC, "decoder_full_fallback"
        if target == "local":
            if base_features_bC.shape[1] == encoder_dim and head_features_bC.shape[1] == encoder_dim:
                return base_features_bC, head_features_bC, "local_query"
            if global_dim > 0 and base_features_bC.shape[1] > global_dim and head_features_bC.shape[1] > global_dim:
                return base_features_bC[:, global_dim:], head_features_bC[:, global_dim:], "local_head_input_slice"
            return base_features_bC, head_features_bC, "local_query_fallback"
        raise ValueError(f"Unsupported effective_lmc_fusion_target={target!r}")

    def _should_log_glace_pixel_diag(self):
        if not self._is_glace_backend():
            return False
        interval = int(getattr(self.options, "glace_pixel_diag_interval", 100))
        if interval <= 0:
            return False
        if not getattr(self, "_logged_glace_pixel_diag_once", False):
            return True
        return int(getattr(self, "iteration", 0)) % interval == 0

    @staticmethod
    def _glace_pixel_bucket_stats(base_err, fused_err, mask):
        mask = mask & torch.isfinite(base_err) & torch.isfinite(fused_err)
        count = int(mask.sum().item())
        if count <= 0:
            return count, float("nan"), float("nan")
        delta = fused_err[mask] - base_err[mask]
        improved_ratio = float((delta < 0).float().mean().item())
        damaged_ratio = float((delta > 0).float().mean().item())
        return count, improved_ratio, damaged_ratio

    def _compute_glace_base_vs_fused_pixel_diag(
        self,
        *,
        features_bCHW,
        fused_reprojection_error_b1,
        target_px_b2,
        gt_inv_poses_b34,
        Ks_b33,
        invKs_b33,
        step_eff,
        normalizer,
    ):
        if not self._should_log_glace_pixel_diag():
            return None
        with torch.no_grad():
            with autocast("cuda", enabled=self.options.use_half):
                base_pred_b3HW = self.regressor.get_scene_coordinates(features_bCHW)
            base_pred_b3HW = self._recover_pred_scene_to_training_world(base_pred_b3HW)
            base_pred_b31 = base_pred_b3HW.permute(0, 2, 3, 1).flatten(0, 2).unsqueeze(-1).float()
            base_contract = self._compute_reprojection_invalid_loss_contract(
                base_pred_b31,
                target_px_b2,
                gt_inv_poses_b34,
                Ks_b33,
                invKs_b33,
                step_eff=step_eff,
                normalizer=normalizer,
                invalid_max_delta=self._loss_invalid_max_delta,
                invalid_posinf=0.0,
                invalid_neginf=0.0,
            )
            base_err = base_contract["reprojection_error_l1"].detach().float().flatten()
            fused_err = fused_reprojection_error_b1.detach().float().flatten()
            base_valid = base_contract["valid_mask"].detach().flatten().bool()
            compare_mask = base_valid & torch.isfinite(base_err) & torch.isfinite(fused_err)
            count = int(compare_mask.sum().item())
            if count <= 0:
                diag = {
                    "count": 0,
                    "mean_delta": float("nan"),
                    "median_delta": float("nan"),
                    "improved_ratio": float("nan"),
                    "worsened_ratio": float("nan"),
                    "easy_count": 0,
                    "easy_damaged_ratio": float("nan"),
                    "medium_count": 0,
                    "medium_improved_ratio": float("nan"),
                    "hard_count": 0,
                    "hard_improved_ratio": float("nan"),
                }
            else:
                delta = fused_err[compare_mask] - base_err[compare_mask]
                easy_mask = compare_mask & (base_err < 5.0)
                medium_mask = compare_mask & (base_err >= 5.0) & (base_err < 20.0)
                hard_mask = compare_mask & (base_err >= 20.0)
                easy_count, _, easy_damaged_ratio = self._glace_pixel_bucket_stats(base_err, fused_err, easy_mask)
                medium_count, medium_improved_ratio, _ = self._glace_pixel_bucket_stats(base_err, fused_err, medium_mask)
                hard_count, hard_improved_ratio, _ = self._glace_pixel_bucket_stats(base_err, fused_err, hard_mask)
                diag = {
                    "count": count,
                    "mean_delta": float(delta.mean().item()),
                    "median_delta": float(delta.median().item()),
                    "improved_ratio": float((delta < 0).float().mean().item()),
                    "worsened_ratio": float((delta > 0).float().mean().item()),
                    "easy_count": easy_count,
                    "easy_damaged_ratio": easy_damaged_ratio,
                    "medium_count": medium_count,
                    "medium_improved_ratio": medium_improved_ratio,
                    "hard_count": hard_count,
                    "hard_improved_ratio": hard_improved_ratio,
                }
            self._logged_glace_pixel_diag_once = True
            self._last_glace_pixel_diag = diag
            return diag

    def _log_glace_pixel_diag(self, diag):
        if diag is None:
            return
        _logger.info(
            "[GLACE-LMC pixel diag] n=%d mean_delta=%.3fpx median_delta=%.3fpx improved=%.2f%% worsened=%.2f%% "
            "easy_n=%d easy_damaged=%.2f%% medium_n=%d medium_improved=%.2f%% hard_n=%d hard_improved=%.2f%%",
            int(diag.get("count", 0)),
            float(diag.get("mean_delta", float("nan"))),
            float(diag.get("median_delta", float("nan"))),
            100.0 * float(diag.get("improved_ratio", float("nan"))),
            100.0 * float(diag.get("worsened_ratio", float("nan"))),
            int(diag.get("easy_count", 0)),
            100.0 * float(diag.get("easy_damaged_ratio", float("nan"))),
            int(diag.get("medium_count", 0)),
            100.0 * float(diag.get("medium_improved_ratio", float("nan"))),
            int(diag.get("hard_count", 0)),
            100.0 * float(diag.get("hard_improved_ratio", float("nan"))),
        )

    def _buffer_schema_name(self, base_name):
        if self._needs_image_indices_in_buffer():
            return f"{base_name}_glace"
        return base_name

    def _glace_decoder_feature_dim(self):
        if not self._is_glace_backend():
            return int(getattr(getattr(self.regressor, "encoder", None), "feature_dim", 0) or 0)
        decoder_dim = int(getattr(self.regressor, "decoder_dim", 0) or 0)
        if decoder_dim <= 0:
            encoder_dim = int(getattr(self.regressor, "feature_dim", 0))
            decoder_dim = encoder_dim + int(getattr(self, "glace_global_feat_dim", 0))
        return decoder_dim

    def _buffer_feature_dim(self):
        return self._glace_decoder_feature_dim() if self._is_glace_backend() else int(getattr(self.regressor, "feature_dim", 0))

    def _get_image_global_features(self, img_idx_B, *, device, dtype):
        if img_idx_B is None:
            raise ValueError("Image-level global features require img_idx.")
        if not hasattr(self, "global_feats") or self.global_feats is None:
            raise ValueError("Image-level global features are not initialized.")
        idx = img_idx_B.to(self.global_feats.device, non_blocking=True).long().view(-1)
        global_features_BC = self.global_feats[idx]
        if global_features_BC.device != device:
            global_features_BC = global_features_BC.to(device, non_blocking=True)
        if global_features_BC.dtype != dtype:
            global_features_BC = global_features_BC.to(dtype=dtype)
        return global_features_BC

    def _ace_lmc_global_feature_mode(self):
        return str(getattr(self.options, "ace_lmc_global_feature_mode", "glace")).lower()

    def _prepare_ace_lmc_global_feature_bank(self, global_feats):
        mode = self._ace_lmc_global_feature_mode()
        if mode == "glace":
            return global_feats
        if mode == "zero":
            return torch.zeros_like(global_feats)
        if mode == "random":
            seed = int(getattr(self.options, "ace_lmc_random_global_seed", 20260531))
            gen = torch.Generator(device="cpu").manual_seed(seed)
            random_vec = torch.randn((global_feats.shape[1],), generator=gen, dtype=torch.float32)
            return random_vec.to(device=global_feats.device, dtype=global_feats.dtype).view(1, -1).expand_as(global_feats).clone()
        raise ValueError(f"Unsupported ace_lmc_global_feature_mode={mode!r}")

    @staticmethod
    def _logit_from_unit_interval(value):
        value = min(max(float(value), 1e-6), 1.0 - 1e-6)
        return math.log(value / (1.0 - value))

    def _ace_lmc_global_gate_max(self):
        return max(0.0, float(getattr(self.options, "ace_lmc_global_gate_max", 0.0) or 0.0))

    def _init_ace_lmc_global_gate(self):
        self.ace_lmc_global_gate = None
        self.ace_lmc_global_gate_value = float(getattr(self.options, "ace_lmc_global_gate_init", 1.0))
        if not self._uses_ace_lmc_global_head() or self._uses_ace_lmc_global_film_head():
            return
        if bool(getattr(self.options, "ace_lmc_global_gate_learnable", False)):
            gate_max = self._ace_lmc_global_gate_max()
            init_value = self.ace_lmc_global_gate_value
            if gate_max > 0.0:
                init_value = self._logit_from_unit_interval(init_value / gate_max)
            self.ace_lmc_global_gate = torch.nn.Parameter(
                torch.tensor(init_value, device=self.device, dtype=torch.float32)
            )

    def _current_ace_lmc_global_gate_tensor(self, *, device, dtype):
        param = getattr(self, "ace_lmc_global_gate", None)
        gate_max = self._ace_lmc_global_gate_max()
        if isinstance(param, torch.nn.Parameter):
            param = param.to(device=device, dtype=dtype)
            if gate_max > 0.0:
                return torch.sigmoid(param) * torch.tensor(gate_max, device=device, dtype=dtype)
            return param
        value = float(getattr(self, "ace_lmc_global_gate_value", 1.0))
        if gate_max > 0.0:
            value = min(max(value, 0.0), gate_max)
        return torch.tensor(value, device=device, dtype=dtype)

    def _current_ace_lmc_global_gate_value(self):
        return float(self._current_ace_lmc_global_gate_tensor(device=self.device, dtype=torch.float32).detach().cpu().item())

    def _ace_lmc_global_gate_params(self):
        if self._uses_ace_lmc_global_film_head():
            return []
        param = getattr(self, "ace_lmc_global_gate", None)
        return [param] if isinstance(param, torch.nn.Parameter) else []

    def _ace_lmc_global_gate_l1_weight(self):
        if not self._uses_ace_lmc_global_head() or self._uses_ace_lmc_global_film_head():
            return 0.0
        return max(0.0, float(getattr(self.options, "ace_lmc_global_gate_l1_weight", 0.0) or 0.0))

    def _compute_ace_lmc_global_gate_l1_loss(self):
        weight = self._ace_lmc_global_gate_l1_weight()
        if weight <= 0.0:
            return torch.zeros((), device=self.device, dtype=torch.float32), {"enabled": False}
        gate = self._current_ace_lmc_global_gate_tensor(device=self.device, dtype=torch.float32)
        loss = gate.abs() * weight
        return loss, {
            "enabled": True,
            "gate": float(gate.detach().cpu().item()),
            "weight": float(weight),
            "loss": float(loss.detach().cpu().item()),
        }

    @staticmethod
    def _format_ace_lmc_global_gate_l1_stats(stats):
        if not isinstance(stats, dict) or not stats.get("enabled", False):
            return ""
        return (
            f", gScalar={float(stats.get('gate', 0.0)):.5f}"
            f", gScalarL1={float(stats.get('loss', 0.0)):.6f}"
        )

    def _init_ace_lmc_global_residual_head(self):
        self.ace_lmc_global_residual_head = None
        if not self._uses_ace_lmc_global_residual_head():
            return
        local_dim = int(getattr(self.regressor, "ace_lmc_local_feature_dim", getattr(self.options, "ace_encoder_features", 512)))
        global_dim = int(getattr(self, "glace_global_feat_dim", 0) or 0)
        if global_dim <= 0:
            raise ValueError("[ACE-FCN-LMC] glace_residual requires loaded global features.")
        xyz_condition_mode = str(getattr(self.options, "ace_lmc_global_residual_xyz_condition_mode", "fourier_adaln") or "fourier_adaln").lower()
        if not bool(getattr(self.options, "ace_lmc_global_residual_use_xyz_condition", True)):
            xyz_condition_mode = "none"
        self.ace_lmc_global_residual_head = ACEGlobalResidualHead(
            mean=self.dataset.mean_cam_center,
            num_head_blocks=self.options.num_head_blocks,
            in_channels=local_dim + global_dim,
            gate_init=float(getattr(self.options, "ace_lmc_global_gate_init", 0.001)),
            gate_max=float(getattr(self.options, "ace_lmc_global_gate_max", 0.1)),
            delta_max_m=float(getattr(self.options, "ace_lmc_global_residual_delta_max_m", 1.0)),
            xyz_condition_mode=xyz_condition_mode,
            xyz_condition_scale=float(getattr(self.options, "ace_lmc_global_residual_xyz_condition_scale", 10.0)),
        ).to(self.device)
        _logger.info(
            "[ACE-FCN-LMC] Global residual head initialized: local_dim=%d global_dim=%d gate_init=%.6f gate_max=%.6f delta_max_m=%.3f gate_l1=%.6f xyz_condition=%s xyz_scale=%.3f.",
            local_dim,
            global_dim,
            float(getattr(self.options, "ace_lmc_global_gate_init", 0.001)),
            float(getattr(self.options, "ace_lmc_global_gate_max", 0.1)),
            float(getattr(self.options, "ace_lmc_global_residual_delta_max_m", 1.0)),
            float(getattr(self.options, "ace_lmc_global_residual_gate_l1_weight", 0.0) or 0.0),
            xyz_condition_mode,
            float(getattr(self.options, "ace_lmc_global_residual_xyz_condition_scale", 10.0)),
        )

    def _ace_lmc_global_residual_params(self):
        module = getattr(self, "ace_lmc_global_residual_head", None)
        return list(module.parameters()) if module is not None else []

    def _ace_lmc_global_residual_gate_l1_weight(self):
        if not self._uses_ace_lmc_global_residual_head():
            return 0.0
        return max(0.0, float(getattr(self.options, "ace_lmc_global_residual_gate_l1_weight", 0.0) or 0.0))

    def _ace_lmc_global_residual_bad_gate_weight(self):
        if not self._uses_ace_lmc_global_residual_head():
            return 0.0
        return max(0.0, float(getattr(self.options, "ace_lmc_global_residual_bad_gate_weight", 0.0) or 0.0))

    def _format_ace_lmc_global_residual_stats(self, stats):
        if not isinstance(stats, dict) or not stats.get("enabled", False):
            return ""
        return (
            f", gGate={float(stats.get('gate_mean', 0.0)):.5f}"
            f", gGateMax={float(stats.get('gate_max', 0.0)):.5f}"
            f", gDelta={float(stats.get('delta_l2', 0.0)):.5f}"
            f", gStep={float(stats.get('gated_delta_l2', 0.0)):.5f}"
            f", gL1={float(stats.get('gate_l1_loss', 0.0)):.5f}"
            f", xyzAdaS={float(stats.get('xyz_adaln_scale_abs', 0.0)):.5f}"
            f", xyzAdaB={float(stats.get('xyz_adaln_shift_abs', 0.0)):.5f}"
        )

    def _predict_ace_lmc_global_residual_coords(self, local_head_features_bC, img_idx_b1, h, w, *, stage_tag):
        module = getattr(self, "ace_lmc_global_residual_head", None)
        if module is None:
            raise ValueError(f"[{stage_tag}] glace_residual mode requires ace_lmc_global_residual_head.")
        local_BCHW = local_head_features_bC.view(1, h, w, local_head_features_bC.shape[1]).permute(0, 3, 1, 2)
        global_bG = self._get_image_global_features(
            img_idx_b1,
            device=local_head_features_bC.device,
            dtype=local_head_features_bC.dtype,
        )
        global_bG = self._prepare_ace_lmc_global_for_head(global_bG, stage_tag=stage_tag)
        if global_bG.shape[0] != local_head_features_bC.shape[0]:
            raise ValueError(
                f"[{stage_tag}] Global/local row mismatch: global={tuple(global_bG.shape)} "
                f"local={tuple(local_head_features_bC.shape)}"
            )
        global_BGHW = global_bG.view(1, h, w, global_bG.shape[1]).permute(0, 3, 1, 2)
        residual_input_BCHW = torch.cat((global_BGHW, local_BCHW), dim=1)
        base_head = self._ace_lmc_global_residual_base_head()
        if base_head is None:
            raise ValueError(f"[{stage_tag}] glace_residual mode requires a frozen Stage1 base head.")
        with torch.no_grad():
            local_pred_B3HW = base_head(local_BCHW)
        pred_B3HW, gate_B1HW, _ = module(local_pred_B3HW, residual_input_BCHW)
        gate_l1_weight = self._ace_lmc_global_residual_gate_l1_weight()
        gate_l1_loss = gate_B1HW.float().mean() * gate_l1_weight if gate_l1_weight > 0.0 else gate_B1HW.new_zeros(())
        stats = {"enabled": True, "gate_l1_loss": float(gate_l1_loss.detach().cpu().item())}
        stats.update(getattr(module, "last_stats", {}) or {})
        return pred_B3HW, gate_l1_loss, stats, gate_B1HW

    def _format_ace_lmc_global_film_stats(self, stats):
        if not isinstance(stats, dict) or not stats.get("enabled", False):
            return ""
        return (
            f", filmGate={float(stats.get('gate', 0.0)):.5f}"
            f", filmGamma={float(stats.get('gamma_abs', 0.0)):.5f}"
            f", filmBeta={float(stats.get('beta_abs', 0.0)):.5f}"
            f", filmStep={float(stats.get('step_abs', 0.0)):.5f}"
        )

    def _predict_ace_lmc_global_film_coords(self, local_head_features_bC, img_idx_b1, h, w, *, stage_tag):
        if not self._uses_ace_lmc_global_film_head():
            raise ValueError(f"[{stage_tag}] glace_film mode is not active.")
        local_BCHW = local_head_features_bC.view(1, h, w, local_head_features_bC.shape[1]).permute(0, 3, 1, 2)
        global_bG = self._get_image_global_features(
            img_idx_b1,
            device=local_head_features_bC.device,
            dtype=local_head_features_bC.dtype,
        )
        if global_bG.shape[0] != local_head_features_bC.shape[0]:
            raise ValueError(
                f"[{stage_tag}] Global/local row mismatch: global={tuple(global_bG.shape)} "
                f"local={tuple(local_head_features_bC.shape)}"
            )
        global_BC = global_bG.view(1, h, w, global_bG.shape[1])[:, 0, 0, :].contiguous()
        pred_B3HW = self.regressor.heads(local_BCHW, global_BC)
        return pred_B3HW, dict(getattr(self.regressor.heads, "last_stats", {}) or {})

    def _ace_lmc_stage2_consistency_base_weight(self):
        if not self._uses_ace_lmc_global_head():
            return 0.0
        return max(0.0, float(getattr(self.options, "ace_lmc_stage2_consistency_weight", 0.0) or 0.0))

    def _ace_lmc_stage2_consistency_weight(self):
        weight = self._ace_lmc_stage2_consistency_base_weight()
        if weight <= 0.0:
            return 0.0
        warmup_steps = max(0, int(getattr(self.options, "ace_lmc_stage2_consistency_warmup_steps", 0) or 0))
        if warmup_steps <= 0:
            return weight
        scale = min(1.0, max(0.0, float(getattr(self, "local_s2_step", 0)) / float(warmup_steps)))
        return weight * scale

    def _uses_ace_lmc_stage2_consistency(self):
        return bool(self._ace_lmc_stage2_consistency_base_weight() > 0.0)

    def _ace_lmc_stage2_guard_base_weight(self):
        if not self._uses_ace_lmc_global_head():
            return 0.0
        return max(0.0, float(getattr(self.options, "ace_lmc_stage2_guard_weight", 0.0) or 0.0))

    def _ace_lmc_stage2_guard_weight(self):
        weight = self._ace_lmc_stage2_guard_base_weight()
        if weight <= 0.0:
            return 0.0
        warmup_steps = max(0, int(getattr(self.options, "ace_lmc_stage2_guard_warmup_steps", 0) or 0))
        if warmup_steps <= 0:
            return weight
        scale = min(1.0, max(0.0, float(getattr(self, "local_s2_step", 0)) / float(warmup_steps)))
        return weight * scale

    def _uses_ace_lmc_stage2_guard(self):
        return bool(self._ace_lmc_stage2_guard_base_weight() > 0.0)

    def _uses_ace_lmc_stage2_teacher(self):
        return bool(
            self._uses_ace_lmc_global_residual_head()
            or self._uses_ace_lmc_stage2_consistency()
            or self._uses_ace_lmc_stage2_guard()
        )

    def _ace_lmc_global_residual_base_head(self):
        teacher_head = getattr(self, "ace_lmc_stage2_teacher_head", None)
        if teacher_head is not None:
            return teacher_head
        return getattr(self.regressor, "heads", None)

    def _predict_ace_lmc_stage2_teacher_coords(self, local_head_features_bC, reference_pred_scene_coords_b3HW):
        teacher_head = getattr(self, "ace_lmc_stage2_teacher_head", None)
        if teacher_head is None:
            return None
        if local_head_features_bC is None or local_head_features_bC.dim() != 2:
            raise ValueError("Stage2 teacher expects local head features [N,C].")
        if reference_pred_scene_coords_b3HW.dim() != 4 or reference_pred_scene_coords_b3HW.shape[0] != 1:
            raise ValueError("Stage2 teacher currently expects packed single-grid predictions [1,3,H,W].")

        _, _, H, W = reference_pred_scene_coords_b3HW.shape
        N = H * W
        if int(local_head_features_bC.shape[0]) != N:
            raise ValueError(
                f"Stage2 teacher feature/prediction mismatch: features={tuple(local_head_features_bC.shape)} HxW={H}x{W}."
            )
        local_BCHW = local_head_features_bC.view(1, H, W, local_head_features_bC.shape[1]).permute(0, 3, 1, 2)
        with torch.no_grad():
            with autocast("cuda", enabled=self.options.use_half):
                teacher_pred_B3HW = teacher_head(local_BCHW)
            teacher_pred_B3HW = self._recover_pred_scene_to_training_world(teacher_pred_B3HW).detach()
        return teacher_pred_B3HW

    def _compute_ace_lmc_stage2_consistency_loss(self, local_head_features_bC, student_pred_scene_coords_b3HW):
        zero = student_pred_scene_coords_b3HW.new_zeros(())
        stats = {"enabled": False, "loss": 0.0, "weight": 0.0, "points": 0}
        weight = self._ace_lmc_stage2_consistency_weight()
        if weight <= 0.0 or getattr(self, "ace_lmc_stage2_teacher_head", None) is None:
            return zero, stats
        teacher_pred_B3HW = self._predict_ace_lmc_stage2_teacher_coords(
            local_head_features_bC,
            student_pred_scene_coords_b3HW,
        )
        if teacher_pred_B3HW is None:
            return zero, stats

        _, _, H, W = student_pred_scene_coords_b3HW.shape
        N = H * W
        student_N3 = student_pred_scene_coords_b3HW.permute(0, 2, 3, 1).reshape(N, 3).float()
        teacher_N3 = teacher_pred_B3HW.permute(0, 2, 3, 1).reshape(N, 3).to(
            device=student_N3.device,
            dtype=student_N3.dtype,
        )
        valid_N = torch.isfinite(student_N3).all(dim=1) & torch.isfinite(teacher_N3).all(dim=1)
        if not valid_N.any():
            return zero, {"enabled": True, "loss": 0.0, "weight": float(weight), "points": 0}

        student_valid = student_N3[valid_N]
        teacher_valid = teacher_N3[valid_N]
        limit = int(getattr(self.options, "ace_lmc_stage2_consistency_sample_limit", 0) or 0)
        if limit > 0 and student_valid.shape[0] > limit:
            perm = torch.randperm(student_valid.shape[0], device=student_valid.device)[:limit]
            student_valid = student_valid[perm]
            teacher_valid = teacher_valid[perm]

        loss_mode = str(getattr(self.options, "ace_lmc_stage2_consistency_loss", "smooth_l1")).lower()
        if loss_mode == "l1":
            raw_loss = Fnn.l1_loss(student_valid, teacher_valid)
        elif loss_mode == "l2":
            raw_loss = Fnn.mse_loss(student_valid, teacher_valid)
        elif loss_mode == "smooth_l1":
            raw_loss = Fnn.smooth_l1_loss(student_valid, teacher_valid)
        else:
            raise ValueError(f"Unsupported ace_lmc_stage2_consistency_loss={loss_mode!r}")
        loss = raw_loss * float(weight)
        return loss, {
            "enabled": True,
            "loss": float(raw_loss.detach().cpu().item()),
            "weight": float(weight),
            "points": int(student_valid.shape[0]),
        }

    def _compute_ace_lmc_stage2_reprojection_guard_loss(
        self,
        *,
        local_head_features_bC,
        student_pred_scene_coords_b3HW,
        student_reprojection_error_N1,
        target_px_N2,
        gt_inv_poses_N34,
        Ks_N33,
        invKs_N33,
        step_eff,
        normalizer,
        student_gate_B1HW=None,
    ):
        zero = student_pred_scene_coords_b3HW.new_zeros(())
        stats = {"enabled": False, "loss": 0.0, "weight": 0.0, "points": 0, "teacher_px": -1.0, "bad_gate_loss": 0.0, "bad_gate_weight": 0.0}
        weight = self._ace_lmc_stage2_guard_weight()
        if weight <= 0.0 or getattr(self, "ace_lmc_stage2_teacher_head", None) is None:
            return zero, stats
        teacher_pred_B3HW = self._predict_ace_lmc_stage2_teacher_coords(
            local_head_features_bC,
            student_pred_scene_coords_b3HW,
        )
        if teacher_pred_B3HW is None:
            return zero, stats

        teacher_pred_N31 = teacher_pred_B3HW.permute(0, 2, 3, 1).flatten(0, 2).unsqueeze(-1).float()
        teacher_contract = self._compute_reprojection_invalid_loss_contract(
            teacher_pred_N31,
            target_px_N2,
            gt_inv_poses_N34,
            Ks_N33,
            invKs_N33,
            step_eff=step_eff,
            normalizer=normalizer,
            invalid_max_delta=self._loss_invalid_max_delta,
            invalid_posinf=0.0,
            invalid_neginf=0.0,
        )
        student_err = student_reprojection_error_N1.float().flatten()
        teacher_err = teacher_contract["reprojection_error_l1"].detach().float().flatten()
        teacher_valid = teacher_contract["valid_mask"].detach().flatten().bool()
        guard_mask = torch.isfinite(student_err) & torch.isfinite(teacher_err) & teacher_valid
        if not guard_mask.any():
            return zero, {"enabled": True, "loss": 0.0, "weight": float(weight), "points": 0, "teacher_px": -1.0}

        margin = max(0.0, float(getattr(self.options, "ace_lmc_stage2_guard_margin_px", 0.25) or 0.0))
        max_px = float(getattr(self.options, "ace_lmc_stage2_guard_max_px", 100.0) or 0.0)
        guard_terms = Fnn.relu(student_err[guard_mask] - teacher_err[guard_mask] - margin)
        if max_px > 0.0:
            guard_terms = guard_terms.clamp(max=max_px)
        raw_guard = guard_terms.mean()
        loss = raw_guard * float(weight)
        bad_gate_weight = self._ace_lmc_global_residual_bad_gate_weight()
        bad_gate_loss = raw_guard.new_zeros(())
        if bad_gate_weight > 0.0 and student_gate_B1HW is not None:
            gate_flat = student_gate_B1HW.float().flatten()
            if gate_flat.shape[0] == student_err.shape[0]:
                bad_gap = Fnn.relu(student_err[guard_mask].detach() - teacher_err[guard_mask] - margin)
                if max_px > 0.0:
                    bad_gap = bad_gap.clamp(max=max_px)
                # Penalize opening the gate exactly where the global residual is worse than the local teacher.
                bad_gate_loss = (gate_flat[guard_mask] * bad_gap).mean() * float(bad_gate_weight)
                loss = loss + bad_gate_loss
        return loss, {
            "enabled": True,
            "loss": float(raw_guard.detach().cpu().item()),
            "weight": float(weight),
            "points": int(guard_mask.sum().item()),
            "teacher_px": float(teacher_err[guard_mask].mean().detach().cpu().item()),
            "bad_gate_loss": float(bad_gate_loss.detach().cpu().item()),
            "bad_gate_weight": float(bad_gate_weight),
        }

    @staticmethod
    def _format_ace_lmc_stage2_consistency_stats(stats):
        if not isinstance(stats, dict) or not stats.get("enabled", False):
            return ""
        return (
            f", cons={float(stats.get('loss', 0.0)):.5f}"
            f", consW={float(stats.get('weight', 0.0)):.4f}"
            f", consN={int(stats.get('points', 0))}"
        )

    @staticmethod
    def _format_ace_lmc_stage2_guard_stats(stats):
        if not isinstance(stats, dict) or not stats.get("enabled", False):
            return ""
        return (
            f", guardPx={float(stats.get('loss', 0.0)):.3f}"
            f", guardW={float(stats.get('weight', 0.0)):.4f}"
            f", guardN={int(stats.get('points', 0))}"
            f", teacherPx={float(stats.get('teacher_px', -1.0)):.2f}"
            f", badGate={float(stats.get('bad_gate_loss', 0.0)):.5f}"
        )

    def _prepare_ace_lmc_global_for_head(self, global_features, *, stage_tag):
        if not self._uses_ace_lmc_global_head():
            return global_features
        out = global_features
        noise_std = self._ace_lmc_global_noise_std()
        if noise_std > 0.0 and self.optimizer_head is not None:
            out = out + torch.empty_like(out).normal_(
                mean=0.0,
                std=noise_std,
                generator=getattr(self, '_ace_lmc_global_noise_generator', None),
            )
        if self._ace_lmc_global_normalize():
            orig_dtype = out.dtype
            out = Fnn.normalize(out.float(), dim=1, eps=1e-12).to(dtype=orig_dtype)
        return out

    def _apply_ace_lmc_global_gate(self, global_features):
        if not self._uses_ace_lmc_global_head():
            return global_features
        gate = self._current_ace_lmc_global_gate_tensor(device=global_features.device, dtype=global_features.dtype)
        return global_features * gate

    def _get_glace_global_features(self, img_idx_B, *, device, dtype):
        if not self._is_glace_backend():
            raise ValueError("GLACE global features are only available for model_backend=glace_lmc.")
        return self._get_image_global_features(img_idx_B, device=device, dtype=dtype)

    def _append_ace_lmc_global_to_features(self, local_features_bC, img_idx_b1, *, stage_tag):
        if not self._uses_ace_lmc_concat_head():
            return local_features_bC
        global_bG = self._get_image_global_features(
            img_idx_b1,
            device=local_features_bC.device,
            dtype=local_features_bC.dtype,
        )
        if global_bG.shape[0] != local_features_bC.shape[0]:
            raise ValueError(
                f"[{stage_tag}] Global/local row mismatch: global={tuple(global_bG.shape)} "
                f"local={tuple(local_features_bC.shape)}"
            )
        global_bG = self._prepare_ace_lmc_global_for_head(global_bG, stage_tag=stage_tag)
        global_bG = self._apply_ace_lmc_global_gate(global_bG)
        return torch.cat((global_bG, local_features_bC), dim=1)

    def _append_ace_lmc_global_to_feature_maps(self, local_BCHW, img_idx_B, *, stage_tag):
        if not self._uses_ace_lmc_concat_head():
            return local_BCHW
        if local_BCHW.dim() != 4:
            raise ValueError(f"[{stage_tag}] Expected local feature map [B,C,H,W], got {tuple(local_BCHW.shape)}.")
        global_BC = self._get_image_global_features(
            img_idx_B,
            device=local_BCHW.device,
            dtype=local_BCHW.dtype,
        )
        B, _, H, W = local_BCHW.shape
        if global_BC.shape[0] != B:
            raise ValueError(
                f"[{stage_tag}] Global/local batch mismatch: global={tuple(global_BC.shape)} "
                f"local={tuple(local_BCHW.shape)}"
            )
        global_BC = self._prepare_ace_lmc_global_for_head(global_BC, stage_tag=stage_tag)
        global_BC = self._apply_ace_lmc_global_gate(global_BC)
        global_map = global_BC[:, :, None, None].expand(-1, -1, H, W)
        return torch.cat((global_map, local_BCHW), dim=1)

    def _split_glace_decoder_feature_maps(self, decoder_BCHW, *, stage_tag):
        if not self._is_glace_backend():
            return None, decoder_BCHW
        global_dim = int(getattr(self, "glace_global_feat_dim", 0) or 0)
        if global_dim <= 0:
            raise ValueError(f"[{stage_tag}] GLACE decoder split requires global_dim > 0.")
        if decoder_BCHW.dim() != 4:
            raise ValueError(f"[{stage_tag}] Expected decoder feature map [B,C,H,W], got {tuple(decoder_BCHW.shape)}.")
        if decoder_BCHW.shape[1] <= global_dim:
            raise ValueError(
                f"[{stage_tag}] GLACE decoder feature dim must exceed global_dim, "
                f"got C={decoder_BCHW.shape[1]} global_dim={global_dim}."
            )
        global_map_BCHW = decoder_BCHW[:, :global_dim]
        local_BCHW = decoder_BCHW[:, global_dim:]
        global_BC = global_map_BCHW[:, :, 0, 0].contiguous()
        return global_BC, local_BCHW

    def _build_glace_head_input(self, global_BC, local_BCHW):
        if not self._is_glace_backend():
            return local_BCHW
        if global_BC is None:
            raise ValueError("GLACE head input requires global features.")
        if local_BCHW.dim() != 4:
            raise ValueError(f"Expected local feature map [B,C,H,W], got {tuple(local_BCHW.shape)}.")
        if global_BC.dim() != 2:
            raise ValueError(f"Expected global features [B,Cg], got {tuple(global_BC.shape)}.")
        B, _, H, W = local_BCHW.shape
        if global_BC.shape[0] != B:
            raise ValueError(f"GLACE global/local batch mismatch: global_B={global_BC.shape[0]} local_B={B}.")
        if global_BC.device != local_BCHW.device:
            global_BC = global_BC.to(local_BCHW.device, non_blocking=True)
        if global_BC.dtype != local_BCHW.dtype:
            global_BC = global_BC.to(dtype=local_BCHW.dtype)
        global_BCHW = global_BC[..., None, None].expand(-1, -1, H, W)
        return torch.cat((global_BCHW, local_BCHW), dim=1)

    def _mix_glace_decoder_features(self, base_decoder_features_bC, candidate_decoder_features_bC, *, stage_tag):
        if not self._is_glace_backend():
            return candidate_decoder_features_bC
        if base_decoder_features_bC is None:
            return candidate_decoder_features_bC
        if getattr(self, "glace_residual_adapter", None) is None:
            return candidate_decoder_features_bC
        if tuple(base_decoder_features_bC.shape) != tuple(candidate_decoder_features_bC.shape):
            raise ValueError(
                f"[{stage_tag}] GLACE decoder feature shape mismatch: "
                f"base={tuple(base_decoder_features_bC.shape)} candidate={tuple(candidate_decoder_features_bC.shape)}"
            )
        return self.glace_residual_adapter(base_decoder_features_bC, candidate_decoder_features_bC)

    def _mix_glace_decoder_feature_maps(self, base_decoder_features_BCHW, candidate_decoder_features_BCHW, *, stage_tag):
        if not self._is_glace_backend():
            return candidate_decoder_features_BCHW
        if base_decoder_features_BCHW is None:
            return candidate_decoder_features_BCHW
        if tuple(base_decoder_features_BCHW.shape) != tuple(candidate_decoder_features_BCHW.shape):
            raise ValueError(
                f"[{stage_tag}] GLACE decoder feature-map shape mismatch: "
                f"base={tuple(base_decoder_features_BCHW.shape)} candidate={tuple(candidate_decoder_features_BCHW.shape)}"
            )
        B, C, H, W = candidate_decoder_features_BCHW.shape
        base_bC = base_decoder_features_BCHW.permute(0, 2, 3, 1).reshape(B * H * W, C)
        cand_bC = candidate_decoder_features_BCHW.permute(0, 2, 3, 1).reshape(B * H * W, C)
        mixed_bC = self._mix_glace_decoder_features(base_bC, cand_bC, stage_tag=stage_tag)
        return mixed_bC.view(B, H, W, C).permute(0, 3, 1, 2)

    def _build_glace_decoder_feature_maps(self, local_features_BCHW, img_idx_B, *, stage_tag):
        if not self._is_glace_backend():
            return local_features_BCHW
        if img_idx_B is None:
            raise ValueError(f"[{stage_tag}] GLACE backend requires img_idx to assemble decoder features.")
        global_features_BC = self._get_glace_global_features(
            img_idx_B,
            device=local_features_BCHW.device,
            dtype=local_features_BCHW.dtype,
        )
        return self._build_glace_head_input(global_features_BC, local_features_BCHW)

    def _build_glace_decoder_features(self, local_features_bC, img_idx_b1, *, stage_tag):
        if not self._is_glace_backend():
            return local_features_bC
        if img_idx_b1 is None:
            raise ValueError(f"[{stage_tag}] GLACE backend requires img_idx to assemble decoder features.")
        if not hasattr(self, "global_feats") or self.global_feats is None:
            raise ValueError(f"[{stage_tag}] GLACE backend global_feats are not initialized.")
        idx = img_idx_b1.to(self.global_feats.device, non_blocking=True).long().view(-1)
        global_features_bC = self.global_feats[idx]
        if global_features_bC.device != local_features_bC.device:
            global_features_bC = global_features_bC.to(local_features_bC.device, non_blocking=True)
        if global_features_bC.dtype != local_features_bC.dtype:
            global_features_bC = global_features_bC.to(dtype=local_features_bC.dtype)
        return torch.cat((global_features_bC, local_features_bC), dim=1)

    def _build_train_dataset(self, image_width=None, augment=False, aug_rotation=0, aug_scale_max=1.0, aug_scale_min=1.0):
        if self._is_glace_backend():
            backend = getattr(self.options, "data_backend", "ace")
            if backend != "ace":
                raise ValueError("[GLACE-LMC] model_backend=glace_lmc currently supports only data_backend=ace.")
            train_root = self._get_train_root()
            dataset = build_glace_camloc_dataset(
                glace_root=self.options.glace_root,
                root_dir=train_root,
                mode=0,
                augment=augment,
                aug_rotation=aug_rotation,
                aug_scale_max=aug_scale_max,
                aug_scale_min=aug_scale_min,
                image_height=self.options.image_resolution,
                use_half=self.options.use_half,
                num_clusters=None,
                cluster_idx=None,
                feat_name=str(getattr(self.options, "glace_feat_name", "features.npy")),
            )
        elif self._is_ace_fcn_backend():
            backend = getattr(self.options, "data_backend", "ace")
            if backend != "ace":
                raise ValueError("[ACE-FCN-LMC] model_backend=ace_fcn_lmc currently supports only data_backend=ace.")
            dataset = CamLocDatasetACEFCNLMC(
                root_dir=self._get_train_root(),
                mode=0,
                augment=augment,
                aug_rotation=aug_rotation,
                aug_scale_max=aug_scale_max,
                aug_scale_min=aug_scale_min,
                image_height=self.options.image_resolution,
                use_half=self.options.use_half,
                num_clusters=None,
                cluster_idx=None,
                feat_name=str(getattr(self.options, "glace_feat_name", "features.npy")) if self._uses_ace_lmc_global_head() else None,
            )
        else:
            dataset = super()._build_train_dataset(
                image_width=image_width,
                augment=augment,
                aug_rotation=aug_rotation,
                aug_scale_max=aug_scale_max,
                aug_scale_min=aug_scale_min,
            )
        aux_ref_required = float(getattr(self.options, "c1_aux_ref_loss_weight", 0.0)) > 0.0
        valid_coord_sampling = bool(getattr(self.options, "buffer_sample_valid_coords", True))
        if (aux_ref_required or valid_coord_sampling) and hasattr(dataset, "init"):
            train_root = self._get_train_root()
            backend = getattr(self.options, "data_backend", "ace")
            if backend == "ace":
                ace_depth_dir = train_root / "depth"
                if ace_depth_dir.exists():
                    dataset.init = True
                    dataset.sparse = False
                    dataset.eye = False
                    dataset.coord_files = sorted(ace_depth_dir.iterdir())
                    if getattr(dataset, "prediction_grid", None) is None:
                        dataset.prediction_grid = dataset._create_prediction_grid()
                else:
                    depth_dir = self._infer_c1_aux_depth_dir(train_root)
                    if aux_ref_required or depth_dir.exists():
                        dataset = self._attach_ace_aux_depth_dataset(dataset, train_root, depth_dir)
                    elif not getattr(self, "_warned_missing_valid_coord_sampling_depth", False):
                        _logger.warning(
                            "[LMC] --buffer_sample_valid_coords=True but no aux depth dir found at %s; "
                            "falling back to image-mask random sampling.",
                            depth_dir,
                        )
                        self._warned_missing_valid_coord_sampling_depth = True
            elif backend == "wai":
                depth_dir = self._infer_c1_aux_depth_dir(train_root)
                if aux_ref_required or depth_dir.exists():
                    dataset = self._attach_wai_aux_depth_dataset(dataset, train_root, depth_dir)
                elif not getattr(self, "_warned_missing_valid_coord_sampling_depth", False):
                    _logger.warning(
                        "[LMC] --buffer_sample_valid_coords=True but no WAI aux depth dir found at %s; "
                        "falling back to image-mask random sampling.",
                        depth_dir,
                    )
                    self._warned_missing_valid_coord_sampling_depth = True
        return dataset

    def _create_regressor(self):
        if self._is_ace_fcn_backend():
            encoder_dim = int(getattr(self.options, "ace_encoder_features", 512))
            regressor = RegressorACE.create_from_encoder(
                encoder_path=self.options.ace_encoder_path,
                mean=self.dataset.mean_cam_center,
                num_head_blocks=self.options.num_head_blocks,
                use_homogeneous=self.options.use_homogeneous,
                num_encoder_features=encoder_dim,
                freeze_backbone=self.options.freeze_backbone,
            )
            if self._uses_ace_lmc_concat_head():
                global_feat_dim = int(getattr(self.dataset, "global_feat_dim", 0))
                if global_feat_dim <= 0:
                    raise ValueError("[ACE-FCN-LMC] glace_concat requires non-empty GLACE global features.")
                final_head_dim = encoder_dim + global_feat_dim
                glace_head_cls = get_glace_head_class(getattr(self.options, "glace_root", Path('/home/xwh/project/glace')))
                head_channels = int(getattr(self.options, "glace_head_channels", 512))
                mlp_ratio = float(getattr(self.options, "glace_mlp_ratio", 1.0))
                regressor.heads = glace_head_cls(
                    self.dataset.mean_cam_center,
                    self.options.num_head_blocks,
                    self.options.use_homogeneous,
                    in_channels=final_head_dim,
                    head_channels=head_channels,
                    mlp_ratio=mlp_ratio,
                )
                regressor.ace_lmc_local_feature_dim = encoder_dim
                regressor.ace_lmc_final_head_dim = final_head_dim
                regressor.ace_lmc_head_impl = 'glace_head'
                _logger.info(
                    "Loaded ACE FCN encoder from %s and created GLACE-concat head: local_dim=%d global_dim=%d final_dim=%d head_channels=%d mlp_ratio=%.3f",
                    self.options.ace_encoder_path,
                    encoder_dim,
                    global_feat_dim,
                    final_head_dim,
                    head_channels,
                    mlp_ratio,
                )
            elif self._uses_ace_lmc_global_residual_head():
                global_feat_dim = int(getattr(self.dataset, "global_feat_dim", 0))
                if global_feat_dim <= 0:
                    raise ValueError("[ACE-FCN-LMC] glace_residual requires non-empty GLACE global features.")
                regressor.ace_lmc_local_feature_dim = encoder_dim
                regressor.ace_lmc_final_head_dim = encoder_dim
                regressor.ace_lmc_head_impl = 'ace_global_residual'
                _logger.info(
                    "Loaded ACE FCN encoder from %s and kept local Stage1 head for GLACE residual mode: local_dim=%d global_dim=%d",
                    self.options.ace_encoder_path,
                    encoder_dim,
                    global_feat_dim,
                )
            elif self._uses_ace_lmc_global_film_head():
                global_feat_dim = int(getattr(self.dataset, "global_feat_dim", 0))
                if global_feat_dim <= 0:
                    raise ValueError("[ACE-FCN-LMC] glace_film requires non-empty GLACE global features.")
                regressor.heads = ACEGlobalFiLMHead(
                    self.dataset.mean_cam_center,
                    self.options.num_head_blocks,
                    self.options.use_homogeneous,
                    in_channels=encoder_dim,
                    global_dim=global_feat_dim,
                    gate_init=float(getattr(self.options, "ace_lmc_global_gate_init", 0.0)),
                    gate_max=float(getattr(self.options, "ace_lmc_global_gate_max", 1.0)),
                )
                regressor.ace_lmc_local_feature_dim = encoder_dim
                regressor.ace_lmc_final_head_dim = encoder_dim
                _logger.info(
                    "Loaded ACE FCN encoder from %s and created GLACE-FiLM head: local_dim=%d global_dim=%d gate_init=%.6f gate_max=%.6f",
                    self.options.ace_encoder_path,
                    encoder_dim,
                    global_feat_dim,
                    float(getattr(self.options, "ace_lmc_global_gate_init", 0.0)),
                    float(getattr(self.options, "ace_lmc_global_gate_max", 1.0)),
                )
            else:
                _logger.info("Loaded ACE FCN encoder from: %s", self.options.ace_encoder_path)
            return regressor
        if not self._is_glace_backend():
            return super()._create_regressor()
        global_feat_dim = int(getattr(self.dataset, "global_feat_dim", 0))
        init_head_path = getattr(self.options, "glace_init_head_path", None)
        if init_head_path is not None:
            init_head_path = Path(init_head_path)
        # If a vanilla GLACE head checkpoint is supplied, reuse it as the
        # starting point instead of rebuilding a fresh head from scratch.
        if init_head_path is not None and str(init_head_path) and init_head_path.exists():
            head_state_dict = self._torch_load_trusted_checkpoint(init_head_path, map_location="cpu")
            regressor = create_glace_regressor_from_split_state_dict(
                glace_root=self.options.glace_root,
                encoder_path=self.options.glace_encoder_path,
                head_state_dict=head_state_dict,
                map_location="cpu",
            )
            _logger.info(
                "Loaded GLACE encoder from %s and initialized scene head from %s (global_feat_dim=%d, feat_name=%s)",
                self.options.glace_encoder_path,
                init_head_path,
                global_feat_dim,
                getattr(self.options, "glace_feat_name", "features.npy"),
            )
            return regressor
        if init_head_path is not None and str(init_head_path):
            raise FileNotFoundError(f"GLACE init head checkpoint not found: {init_head_path}")

        regressor = create_glace_regressor_from_encoder(
            glace_root=self.options.glace_root,
            encoder_path=self.options.glace_encoder_path,
            mean=self.dataset.mean_cam_center,
            num_head_blocks=self.options.num_head_blocks,
            use_homogeneous=self.options.use_homogeneous,
            global_feat_dim=global_feat_dim,
            head_channels=int(getattr(self.options, "glace_head_channels", 512)),
            mlp_ratio=float(getattr(self.options, "glace_mlp_ratio", 1.0)),
            map_location="cpu",
        )
        _logger.info(
            "Loaded GLACE encoder from: %s (global_feat_dim=%d, feat_name=%s)",
            self.options.glace_encoder_path,
            global_feat_dim,
            getattr(self.options, "glace_feat_name", "features.npy"),
        )
        return regressor

    def __init__(self, options):
        # Determine if LMC is active *before* parent __init__
        self.model_backend = str(getattr(options, 'model_backend', 'ace_dinov2'))
        self.use_lmc = getattr(options, 'use_lmc', False)
        memory_path = getattr(options, 'memory_path', None)
        if memory_path is None:
            self.use_lmc = False
        if self.model_backend in ('glace_lmc', 'ace_fcn_lmc') and not self.use_lmc:
            raise ValueError(f'[{self.model_backend}] requires --use_lmc True and a valid --memory_path.')
        if self.model_backend in ('glace_lmc', 'ace_fcn_lmc') and str(getattr(options, 'data_backend', 'ace')) != 'ace':
            raise ValueError(f'[{self.model_backend}] currently supports only data_backend=ace.')
        if self.model_backend == 'glace_lmc' and str(getattr(options, 'lmc_flow', 'iterative')) != 'ace_g':
            raise ValueError('[GLACE-LMC] full-decoder fusion currently requires --lmc_flow ace_g.')
        if self.model_backend == 'ace_fcn_lmc' and str(getattr(options, 'lmc_flow', 'iterative')) != 'ace_g':
            raise ValueError('[ACE-FCN-LMC] v1 requires --lmc_flow ace_g.')
        if self.model_backend == 'ace_fcn_lmc' and not bool(getattr(options, 'freeze_backbone', True)):
            raise ValueError('[ACE-FCN-LMC] v1 keeps the ACE FCN encoder frozen; use --freeze_backbone True.')
        ace_lmc_global_head_mode = str(getattr(options, 'ace_lmc_global_head_mode', 'none'))
        if self.model_backend != 'ace_fcn_lmc' and ace_lmc_global_head_mode != 'none':
            raise ValueError('--ace_lmc_global_head_mode is only supported for model_backend=ace_fcn_lmc.')
        if self.model_backend == 'ace_fcn_lmc' and ace_lmc_global_head_mode != 'none' and getattr(options, 'ace_lmc_local_checkpoint_path', None) is None:
            raise ValueError('[ACE-FCN-LMC] Stage2 global modes require --ace_lmc_local_checkpoint_path from stage 1.')
        if ace_lmc_global_head_mode == 'glace_residual' and float(getattr(options, 'ace_lmc_global_gate_max', 0.0) or 0.0) <= 0.0:
            raise ValueError('[ACE-FCN-LMC] glace_residual requires --ace_lmc_global_gate_max > 0; otherwise the residual gate is permanently zero.')
        if self.model_backend == 'glace_lmc' and str(getattr(options, 'glace_freeze_encoder', True)).lower() in ('false', '0', 'no', 'off'):
            raise ValueError(
                'glace_freeze_encoder=False is not supported yet: encoder params are not included in S1/S2 optimizers.'
            )
        ace_lmc_stage2_feature_source = str(getattr(options, 'ace_lmc_stage2_feature_source', 'raw_backbone') or 'raw_backbone').lower()
        if ace_lmc_stage2_feature_source not in ('raw_backbone', 'stage1_fused'):
            raise ValueError(f"Unsupported ace_lmc_stage2_feature_source={ace_lmc_stage2_feature_source!r}")
        if ace_lmc_stage2_feature_source == 'stage1_fused':
            if self.model_backend != 'ace_fcn_lmc' or ace_lmc_global_head_mode not in ('glace_concat', 'glace_residual'):
                raise ValueError('--ace_lmc_stage2_feature_source stage1_fused requires model_backend=ace_fcn_lmc and --ace_lmc_global_head_mode glace_concat or glace_residual.')
            if not bool(getattr(options, 'ace_lmc_freeze_local_stack', True)):
                raise ValueError('--ace_lmc_stage2_feature_source stage1_fused requires --ace_lmc_freeze_local_stack True.')
            if bool(getattr(options, 'ace_g_fusion_in_s2', False)):
                raise ValueError('--ace_lmc_stage2_feature_source stage1_fused is a frozen-backbone GLACE flow; use --ace_g_fusion_in_s2 False.')
        if float(getattr(options, 'ace_lmc_global_noise_std', 0.0) or 0.0) < 0.0:
            raise ValueError('--ace_lmc_global_noise_std must be >= 0.')

        # Parent builds: dataset, regressor, optimizer, scheduler, loss, buffer
        super().__init__(options)
        self.global_feats = None
        self.glace_global_feat_dim = 0
        self.ace_lmc_global_gate = None
        self.ace_lmc_global_gate_value = float(getattr(options, 'ace_lmc_global_gate_init', 1.0))
        self.ace_lmc_global_residual_head = None
        self.ace_lmc_stage2_teacher_head = None
        self.glace_residual_adapter = None
        self.glace_reference_regressor = None
        if self._is_glace_backend():
            self.glace_global_feat_dim = int(getattr(self.dataset, 'global_feat_dim', 0))
            self.global_feats = torch.tensor(
                self.dataset.global_feats,
                dtype=(torch.float32, torch.float16)[self.options.use_half],
                device=self.device,
            )
            glace_residual_mode = str(getattr(self.options, 'glace_residual_mode', 'local_delta_tanh_scalar'))
            glace_residual_gate_init = float(getattr(self.options, 'glace_residual_gate_init', 0.0))
            self.glace_residual_adapter = GLACEDecoderFeatureResidualAdapter(
                residual_gate_init=glace_residual_gate_init,
                mode=glace_residual_mode,
                global_dim=self.glace_global_feat_dim,
            ).to(self.device)
            _logger.info('[GLACE-LMC] Loaded %d global features with dim=%d.', int(self.global_feats.shape[0]), self.glace_global_feat_dim)
            _logger.info(
                '[GLACE-LMC] Decoder residual adapter initialized: mode=%s, init=%.4f, gain=%.4f, global_dim=%d.',
                glace_residual_mode,
                glace_residual_gate_init,
                float(self.glace_residual_adapter.residual_gain().detach().cpu().item()),
                self.glace_global_feat_dim,
            )
            # Anti-regression adds a small penalty when S2 fused output is worse
            # than the frozen reference regressor on the same batch.
            self.glace_antiregression_weight = max(0.0, float(getattr(options, 'glace_antiregression_weight', 0.0)))
            self.glace_antiregression_margin_px = max(0.0, float(getattr(options, 'glace_antiregression_margin_px', 0.25)))
            self.glace_antiregression_max_px = max(0.0, float(getattr(options, 'glace_antiregression_max_px', 100.0)))
            if self.glace_antiregression_weight > 0.0:
                self.glace_reference_regressor = copy.deepcopy(self.regressor).to(self.device).eval()
                for param in self.glace_reference_regressor.parameters():
                    param.requires_grad_(False)
                _logger.info(
                    '[GLACE-LMC] Anti-regression enabled: weight=%.4f margin=%.3fpx max=%.1fpx '
                    '(fixed frozen GLACE reference).',
                    self.glace_antiregression_weight,
                    self.glace_antiregression_margin_px,
                    self.glace_antiregression_max_px,
                )
            if self._glace_freeze_encoder():
                self.regressor.encoder.eval()
                for param in self.regressor.encoder.parameters():
                    param.requires_grad_(False)
            else:
                for param in self.regressor.encoder.parameters():
                    param.requires_grad_(True)
            self._set_glace_head_trainable(not self._glace_freeze_head(), stage_tag="GLACE-LMC")
            _logger.info(
                '[GLACE-LMC] Freeze policy: legacy_base=%s encoder_frozen=%s head_frozen=%s.',
                self._glace_freeze_base_network(),
                self._glace_freeze_encoder(),
                self._glace_freeze_head(),
            )
        if self._uses_ace_lmc_global_head():
            self.glace_global_feat_dim = int(getattr(self.dataset, 'global_feat_dim', 0))
            self.global_feats = torch.tensor(
                self.dataset.global_feats,
                dtype=(torch.float32, torch.float16)[self.options.use_half],
                device=self.device,
            )
            self.global_feats = self._prepare_ace_lmc_global_feature_bank(self.global_feats)
            self._init_ace_lmc_global_gate()
            self._init_ace_lmc_global_residual_head()
            _logger.info(
                '[ACE-FCN-LMC] Loaded %d GLACE global features with dim=%d for final head mode=%s: feature_mode=%s gate_init=%.6f gate_max=%.6f learnable=%s gate_current=%.6f normalize=%s noise_std=%.4f stage2_feature_source=%s.',
                int(self.global_feats.shape[0]),
                self.glace_global_feat_dim,
                self._ace_lmc_global_head_mode(),
                self._ace_lmc_global_feature_mode(),
                float(getattr(self.options, 'ace_lmc_global_gate_init', 1.0)),
                self._ace_lmc_global_gate_max(),
                bool(getattr(self.options, 'ace_lmc_global_gate_learnable', False)),
                self._current_ace_lmc_global_gate_value(),
                self._ace_lmc_global_normalize(),
                self._ace_lmc_global_noise_std(),
                self._ace_lmc_stage2_feature_source(),
            )
        self._ace_lmc_global_noise_generator = torch.Generator(device=self.device if self.device.type == 'cuda' else 'cpu').manual_seed(self.base_seed + 24601)
        self._training_generator_cpu = torch.Generator().manual_seed(self.base_seed + 8191)
        self._training_generator_cuda = None
        if torch.cuda.is_available():
            self._training_generator_cuda = torch.Generator(device=self.device).manual_seed(self.base_seed + 8191)

        # Common iteration/eval policy (used by both LMC and vanilla-iterative mode)
        self.vanilla_iterations = max(1, int(getattr(options, 'vanilla_iterations', 1)))
        self.lmc_flow = str(getattr(options, 'lmc_flow', 'iterative'))
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
        _bse_denorm = bool(getattr(options, "bse_denorm_to_world", False))
        c1_ref_norm_alpha = float(getattr(options, "c1_ref_norm_alpha", 1.0))
        bank_data = load_memory_features(
            str(memory_path),
            self.device,
            bse_denorm_to_world=_bse_denorm,
            c1_ref_norm_alpha=c1_ref_norm_alpha,
        )
        if bool(getattr(options, "lmc_memory_preflight", True)):
            preflight_report = preflight_memory_features(
                bank_data,
                strict=bool(getattr(options, "lmc_memory_preflight_strict", False)),
                expect_scale_tokens=bool(getattr(options, "use_scale_token", True)),
                scene_center_tol=float(getattr(options, "lmc_memory_preflight_center_tol", 1.0)),
            )
            self.memory_preflight_report = preflight_report
            _logger.info(
                "[LMC preflight] done: ok=%s, issues=%d",
                preflight_report["ok"],
                len(preflight_report["issues"]),
            )
        else:
            self.memory_preflight_report = None

        conditioning_reference = bank_data.get("conditioning_reference")
        if not isinstance(conditioning_reference, dict):
            conditioning_reference = {}
        normalization_ref = bank_data.get("normalization_ref")
        if not isinstance(normalization_ref, dict):
            normalization_ref = {}
        memory_contract_mode = str(bank_data.get("contract_mode", "C0")).upper()
        memory_mode = str(bank_data.get("mode", "single_forward"))
        scene_center_world = bank_data.get("scene_center_world")
        if scene_center_world is None:
            scene_center_world = bank_data.get("scene_center")
        if scene_center_world is None:
            scene_center_world = bank_data.get("scene_center_cam")
        scene_center_ref = bank_data.get("scene_center_ref")
        scene_center_ref_norm = bank_data.get("scene_center_ref_norm")
        reference_index = bank_data.get("reference_index", conditioning_reference.get("reference_index"))
        try:
            reference_index = int(reference_index) if reference_index is not None else None
        except (TypeError, ValueError):
            reference_index = None

        self.reference_contract_state = self._build_reference_contract_state(
            memory_contract_mode=memory_contract_mode,
            reference_index=reference_index,
            conditioning_reference=conditioning_reference,
            normalization_ref=normalization_ref,
            scene_center_world=scene_center_world,
            scene_center_ref=scene_center_ref,
            scene_center_ref_norm=scene_center_ref_norm,
        )
        self.memory_contract_info = {
            "memory_mode": memory_mode,
            "contract_mode": memory_contract_mode,
            "reference_index": reference_index,
            "has_points_ref": bank_data.get("points_ref") is not None,
            "has_points_ref_norm": bank_data.get("points_ref_norm") is not None,
            "has_conditioning_reference": bool(conditioning_reference),
            "has_normalization_ref": isinstance(bank_data.get("normalization_ref"), dict),
            "training_target": self.reference_contract_state["output_space"],
            "has_scene_center_world": scene_center_world is not None,
            "has_scene_center_ref": scene_center_ref is not None,
            "has_scene_center_ref_norm": scene_center_ref_norm is not None,
        }
        _logger.info(
            "[LMC] Memory contract: mode=%s, contract=%s, reference_index=%s, "
            "points_ref=%s, points_ref_norm=%s, output_space=%s",
            self.memory_contract_info["memory_mode"],
            self.memory_contract_info["contract_mode"],
            self.memory_contract_info["reference_index"],
            self.memory_contract_info["has_points_ref"],
            self.memory_contract_info["has_points_ref_norm"],
            self.memory_contract_info["training_target"],
        )
        if self.reference_contract_state.get("enabled", False):
            _logger.info(
                "[LMC] C1 recovery enabled: output_space=%s, reference_index=%s",
                self.reference_contract_state["output_space"],
                self.reference_contract_state["reference_index"],
            )

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

        scene_center = bank_data.get("scene_center_contract")
        if scene_center is None:
            if self.reference_contract_state.get("enabled", False):
                if self.reference_contract_state["output_space"] == "points_ref_norm":
                    scene_center = scene_center_ref_norm
                else:
                    scene_center = scene_center_ref
            if scene_center is None:
                scene_center = scene_center_world
        if scene_center is None:
            pts = bank_data["pooled_points"]
            scene_center = pts.mean(dim=0)
            _logger.info("[LMC] scene_center missing in file, using mean(pooled_points)")
        scene_center_world_for_check = scene_center_world if scene_center_world is not None else scene_center
        self._validate_memory_scene_consistency(bank_data, scene_center_world_for_check)

        self.memory_dict = {
            "pooled_points": _unsqueeze0(bank_data["pooled_points"]),
            "pooled_features": _unsqueeze0(bank_data["pooled_features"]),
            "scene_center": _unsqueeze0(scene_center),
            "scene_center_world": _unsqueeze0(scene_center_world) if scene_center_world is not None else None,
            "scene_center_ref": _unsqueeze0(scene_center_ref) if scene_center_ref is not None else None,
            "scene_center_ref_norm": _unsqueeze0(scene_center_ref_norm) if scene_center_ref_norm is not None else None,
            "all_scale_tokens": _unsqueeze0(bank_data.get("all_scale_tokens")) if bank_data.get("all_scale_tokens") is not None else None,
        }
        for opt_key in (
            "points_world",
            "points_norm",
            "ray_dirs",
            "ray_dirs_mean",
            "ray_dirs_dominant",
            "ray_dirs_first",
            "plucker_rays",
            "cluster_sizes",
            "view_camera_centers",
            "view_camera_rotations",
            "view_camera_intrinsics",
            "view_plucker_main_rays",
        ):
            opt_val = bank_data.get(opt_key)
            if opt_val is not None:
                if opt_key in ("view_camera_rotations", "view_camera_intrinsics") and isinstance(opt_val, torch.Tensor) and opt_val.dim() == 3:
                    self.memory_dict[opt_key] = opt_val.unsqueeze(0)
                else:
                    self.memory_dict[opt_key] = _unsqueeze0(opt_val)

        N_mem = self.memory_dict["pooled_points"].shape[1]
        pooled_features_dim = self.memory_dict["pooled_features"].shape[-1]
        _logger.info("[LMC] Memory loaded: %d points, pooled_features_dim=%d", N_mem, pooled_features_dim)
        ace_lmc_allow_mismatched_memory = bool(getattr(options, 'ace_lmc_allow_mismatched_memory', False))
        if self._is_ace_fcn_backend():
            feature_source = str(bank_data.get("feature_source") or "unknown")
            output_subsample_meta = bank_data.get("output_subsample", bank_data.get("patch_stride", None))
            coord_source = str(bank_data.get("coord_source") or "unknown")
            if not ace_lmc_allow_mismatched_memory:
                if feature_source != "ace_fcn":
                    raise ValueError(
                        "[ACE-FCN-LMC] memory feature_source must be 'ace_fcn'. "
                        f"Got {feature_source!r}. Extract a matching memory with "
                        "ace_dinov2_lmc.memory_extraction.extract_memory_ace_fcn, or pass "
                        "--ace_lmc_allow_mismatched_memory True for the explicit MapAnything/DINO/BSE ablation."
                    )
                if int(pooled_features_dim) != 512:
                    raise ValueError(f"[ACE-FCN-LMC] expected pooled_features dim=512, got {pooled_features_dim}.")
                if output_subsample_meta is not None and int(float(output_subsample_meta)) != 8:
                    raise ValueError(
                        f"[ACE-FCN-LMC] expected memory output_subsample/patch_stride=8, got {output_subsample_meta!r}."
                    )
                sanity = bank_data.get("sanity_report")
                if isinstance(sanity, dict) and sanity.get("hard_pass") is False:
                    raise ValueError(f"[ACE-FCN-LMC] memory sanity_report hard_pass=False: {sanity.get('hard_fail_reasons')}")
                _logger.info(
                    "[ACE-FCN-LMC] Memory contract OK: feature_source=%s dim=%d stride=%s coord_source=%s",
                    feature_source,
                    int(pooled_features_dim),
                    output_subsample_meta,
                    coord_source,
                )
            else:
                _logger.warning(
                    "[ACE-FCN-LMC] MISMATCHED MEMORY ABLATION enabled: query_source=ace_fcn memory_feature_source=%s dim=%d stride=%s coord_source=%s. "
                    "This run is expected to test feature-space mismatch, not the main ACE-FCN memory path.",
                    feature_source,
                    int(pooled_features_dim),
                    output_subsample_meta,
                    coord_source,
                )

        # --- LMC config (same as map-anything train_ace: num_layers from layers_idx, feature_dim per layer) ---
        requested_lmc_mode = getattr(options, 'lmc_mode', 'global')
        lmc_mode = requested_lmc_mode
        low_pose_thr = float(getattr(options, "lmc_visibility_low_pose_front_ratio_threshold", 0.30))
        vis_stats = estimate_memory_front_visibility(
            bank_data["pooled_points"],
            bank_data.get("all_poses"),
            max_points=int(getattr(options, "lmc_visibility_sample_points", 4096)),
            low_pose_front_ratio_threshold=low_pose_thr,
        )
        self.memory_visibility_stats = vis_stats
        if vis_stats is not None:
            _logger.info(
                "[LMC] Memory front-visibility: mean=%.3f, median=%.3f, min=%.3f, max=%.3f, "
                "low_pose(<%.2f)=%.3f, route_score=%.3f (sampled_points=%d, poses=%d)",
                vis_stats["mean_front_ratio"],
                vis_stats["median_front_ratio"],
                vis_stats["min_front_ratio"],
                vis_stats["max_front_ratio"],
                vis_stats["low_pose_front_ratio_threshold"],
                vis_stats["low_pose_front_ratio_fraction"],
                vis_stats["global_visibility_route_score"],
                vis_stats["num_points_sampled"],
                vis_stats["num_poses"],
            )
        auto_mode = bool(getattr(options, "lmc_auto_mode_by_visibility", False))
        fallback_mode = str(getattr(options, "lmc_visibility_fallback_mode", "local"))
        route_metric = str(getattr(options, "lmc_visibility_route_metric", "gvcs") or "gvcs").lower()
        route_score_thr = float(getattr(options, "lmc_visibility_route_score_threshold", 0.20))
        legacy_mean_thr = float(getattr(options, "lmc_visibility_front_ratio_threshold", 0.85))
        should_fallback = False
        fallback_msg = ""
        if auto_mode and requested_lmc_mode == "global" and fallback_mode in ("local", "hierarchical") and vis_stats is not None:
            if route_metric == "legacy_mean":
                should_fallback = vis_stats["mean_front_ratio"] < legacy_mean_thr
                fallback_msg = (
                    "mean_front_ratio=%.3f < %.3f. Switching lmc_mode: %s -> %s"
                    % (vis_stats["mean_front_ratio"], legacy_mean_thr, requested_lmc_mode, fallback_mode)
                )
            else:
                should_fallback = vis_stats["global_visibility_route_score"] < route_score_thr
                fallback_msg = (
                    "visibility_route_score=%.3f < %.3f "
                    "(median_front_ratio=%.3f, low_pose_fraction=%.3f at front_ratio<%.2f). "
                    "Switching lmc_mode: %s -> %s"
                    % (
                        vis_stats["global_visibility_route_score"], route_score_thr,
                        vis_stats["median_front_ratio"], vis_stats["low_pose_front_ratio_fraction"],
                        vis_stats["low_pose_front_ratio_threshold"], requested_lmc_mode, fallback_mode,
                    )
                )
        if should_fallback:
            _logger.warning(
                "[LMC] Global mode auto-fallback triggered (%s): %s",
                route_metric,
                fallback_msg,
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

        feature_dim = self._resolve_lmc_feature_dim(
            pooled_features_dim,
            num_layers,
            layers_idx,
            memory_path,
        )
        _logger.info("[LMC] feature_dim (per layer)=%d", feature_dim)

        lmc_key_slice_idx = getattr(options, 'lmc_key_slice_idx', None)
        if num_layers <= 1:
            resolved_key_slice_idx = 0
        elif lmc_key_slice_idx is None:
            resolved_key_slice_idx = min(2, num_layers - 1)
        else:
            resolved_key_slice_idx = int(lmc_key_slice_idx)
        if resolved_key_slice_idx < 0 or resolved_key_slice_idx >= num_layers:
            raise ValueError(
                f"[LMC] lmc_key_slice_idx={resolved_key_slice_idx} out of range for num_layers={num_layers}."
            )
        key_layer_label = None
        if isinstance(layers_idx, (list, tuple)) and len(layers_idx) > resolved_key_slice_idx:
            key_layer_label = layers_idx[resolved_key_slice_idx]
        lmc_key_feature_mode = str(getattr(options, 'lmc_key_feature_mode', 'slice'))
        if lmc_key_feature_mode not in ('slice', 'scalar_mix'):
            raise ValueError(f"Unsupported lmc_key_feature_mode={lmc_key_feature_mode!r}")
        lmc_feature_hierarchy_mode = str(
            getattr(options, 'lmc_feature_hierarchy_mode', 'selected_key_concat_value')
        )
        if lmc_feature_hierarchy_mode not in (
            'selected_key_concat_value',
            'levelwise_latent_merge',
            'levelwise_anchor_residual',
        ):
            raise ValueError(f"Unsupported lmc_feature_hierarchy_mode={lmc_feature_hierarchy_mode!r}")
        lmc_level_merge_mode = str(getattr(options, 'lmc_level_merge_mode', 'softmax_gate'))
        lmc_level_merge_init = str(getattr(options, 'lmc_level_merge_init', 'uniform'))
        lmc_level_proj_shared = bool(getattr(options, 'lmc_level_proj_shared', False))
        lmc_level_cross_attn_shared = bool(getattr(options, 'lmc_level_cross_attn_shared', True))
        lmc_level_gate_entropy_weight = float(getattr(options, 'lmc_level_gate_entropy_weight', 0.0))
        lmc_level_token_gate = bool(getattr(options, 'lmc_level_token_gate', False))
        lmc_level_anchor_residual_gamma_init = float(
            getattr(options, 'lmc_level_anchor_residual_gamma_init', 0.0)
        )
        lmc_geo_bias_mode = str(getattr(options, 'lmc_geo_bias_mode', 'legacy'))
        if lmc_geo_bias_mode not in ('legacy', 'rbf_residual', 'crpb'):
            raise ValueError(f"Unsupported lmc_geo_bias_mode={lmc_geo_bias_mode!r}")
        lmc_geo_bias_rbf_scales = [float(v) for v in getattr(options, 'lmc_geo_bias_rbf_scales', [0.25, 0.5, 1.0, 2.0, 4.0])]
        if len(lmc_geo_bias_rbf_scales) == 0 or any(v <= 0.0 for v in lmc_geo_bias_rbf_scales):
            raise ValueError(f"Invalid lmc_geo_bias_rbf_scales={lmc_geo_bias_rbf_scales!r}")
        lmc_geo_bias_rbf_alpha_init = float(getattr(options, 'lmc_geo_bias_rbf_alpha_init', 0.0))
        lmc_geo_bias_rbf_learn_weights = bool(getattr(options, 'lmc_geo_bias_rbf_learn_weights', True))
        lmc_geo_bias_rbf_per_head = bool(getattr(options, 'lmc_geo_bias_rbf_per_head', False))
        lmc_pos_encoding_mode = str(getattr(options, 'lmc_pos_encoding_mode', 'fourier_legacy'))
        if lmc_pos_encoding_mode not in ('fourier_legacy', 'fourier_v2', 'point_rope'):
            raise ValueError(f"Unsupported lmc_pos_encoding_mode={lmc_pos_encoding_mode!r}")
        lmc_pos_fourier_v2_scales = [float(v) for v in getattr(options, 'lmc_pos_fourier_v2_scales', [1.0, 2.0, 4.0, 8.0, 16.0])]
        if len(lmc_pos_fourier_v2_scales) == 0 or any(v <= 0.0 for v in lmc_pos_fourier_v2_scales):
            raise ValueError(f"Invalid lmc_pos_fourier_v2_scales={lmc_pos_fourier_v2_scales!r}")
        lmc_pos_fourier_coord_norm = str(getattr(options, 'lmc_pos_fourier_coord_norm', 'scene_radius'))
        if lmc_pos_fourier_coord_norm != 'scene_radius':
            raise ValueError(f"Unsupported lmc_pos_fourier_coord_norm={lmc_pos_fourier_coord_norm!r}")
        lmc_pos_fourier_radius = float(getattr(options, 'lmc_pos_fourier_radius', 4.0))
        if lmc_pos_fourier_radius <= 0.0:
            raise ValueError(f"lmc_pos_fourier_radius must be > 0, got {lmc_pos_fourier_radius!r}")
        lmc_pos_fourier_learnable_scale = bool(getattr(options, 'lmc_pos_fourier_learnable_scale', False))
        lmc_pos_fourier_residual_gate_init = float(getattr(options, 'lmc_pos_fourier_residual_gate_init', 0.0))
        lmc_point_rope_coord_norm = str(getattr(options, 'lmc_point_rope_coord_norm', 'scene_radius'))
        if lmc_point_rope_coord_norm != 'scene_radius':
            raise ValueError(f"Unsupported lmc_point_rope_coord_norm={lmc_point_rope_coord_norm!r}")
        lmc_point_rope_radius = float(getattr(options, 'lmc_point_rope_radius', 4.0))
        lmc_point_rope_radius_policy = str(getattr(options, 'lmc_point_rope_radius_policy', 'fixed'))
        if lmc_point_rope_radius_policy not in ('fixed', 'memory_p95', 'mixed_fixed_memory_p95'):
            raise ValueError(f"Unsupported lmc_point_rope_radius_policy={lmc_point_rope_radius_policy!r}")
        lmc_point_rope_mixed_memory_ratio = float(getattr(options, 'lmc_point_rope_mixed_memory_ratio', 0.5))
        if not 0.0 <= lmc_point_rope_mixed_memory_ratio <= 1.0:
            raise ValueError(
                f"lmc_point_rope_mixed_memory_ratio must be in [0,1], got {lmc_point_rope_mixed_memory_ratio!r}"
            )
        lmc_point_rope_seed_pe = str(getattr(options, 'lmc_point_rope_seed_pe', 'fourier_legacy'))
        if lmc_point_rope_seed_pe not in ('fourier_legacy', 'sincos_deterministic'):
            raise ValueError(f"Unsupported lmc_point_rope_seed_pe={lmc_point_rope_seed_pe!r}")
        lmc_point_rope_base = float(getattr(options, 'lmc_point_rope_base', 10000.0))
        lmc_point_rope_axes = str(getattr(options, 'lmc_point_rope_axes', 'xyz_split'))
        lmc_point_rope_apply_to = str(getattr(options, 'lmc_point_rope_apply_to', 'qk'))
        lmc_geo_bias_crpb_dim = int(getattr(options, 'lmc_geo_bias_crpb_dim', 32))
        lmc_geo_bias_crpb_input = str(getattr(options, 'lmc_geo_bias_crpb_input', 'delta_dist_log'))
        lmc_geo_bias_crpb_radius = float(getattr(options, 'lmc_geo_bias_crpb_radius', 4.0))
        lmc_geo_bias_crpb_per_head = bool(getattr(options, 'lmc_geo_bias_crpb_per_head', False))
        lmc_geo_bias_crpb_zero_init = bool(getattr(options, 'lmc_geo_bias_crpb_zero_init', True))
        if lmc_feature_hierarchy_mode in ('levelwise_latent_merge', 'levelwise_anchor_residual'):
            if lmc_mode not in ('global', 'local'):
                raise ValueError(
                    f"{lmc_feature_hierarchy_mode} currently supports only global/local LMC modes, "
                    f"got {lmc_mode!r}."
                )
            if lmc_level_merge_mode != 'softmax_gate':
                raise ValueError(f"Unsupported lmc_level_merge_mode={lmc_level_merge_mode!r}")
            if lmc_level_merge_init not in ('uniform', 'key_slice_bias'):
                raise ValueError(f"Unsupported lmc_level_merge_init={lmc_level_merge_init!r}")
            if lmc_level_gate_entropy_weight != 0.0:
                raise ValueError("lmc_level_gate_entropy_weight must be 0.0 for current levelwise modes.")
            if lmc_level_token_gate:
                raise ValueError("lmc_level_token_gate must be False for current levelwise modes.")
        _logger.info(
            "[LMC] Mode contract: requested=%s effective=%s auto_by_visibility=%s",
            requested_lmc_mode,
            lmc_mode,
            auto_mode,
        )
        _logger.info(
            "[LMC] Key feature: mode=%s key_slice_idx=%d key_layer_label=%s layers_idx=%s",
            lmc_key_feature_mode,
            resolved_key_slice_idx,
            key_layer_label,
            layers_idx if layers_idx else "n/a",
        )
        _logger.info(
            "[LMC] Feature hierarchy: mode=%s merge=%s init=%s proj_shared=%s cross_attn_shared=%s",
            lmc_feature_hierarchy_mode,
            lmc_level_merge_mode,
            lmc_level_merge_init,
            lmc_level_proj_shared,
            lmc_level_cross_attn_shared,
        )
        _logger.info(
            "[LMC] Geo/PE: geo_bias_mode=%s pos_encoding_mode=%s",
            lmc_geo_bias_mode,
            lmc_pos_encoding_mode,
        )

        scale_token_dim = 1024
        if bank_data.get("all_scale_tokens") is not None:
            st = bank_data["all_scale_tokens"]
            if isinstance(st, torch.Tensor) and st.numel() > 0:
                scale_token_dim = st.shape[-1]
                _logger.info("[LMC] scale_token_dim=%d from all_scale_tokens", scale_token_dim)

        encoder_feature_dim = int(getattr(self.regressor.encoder, 'feature_dim', getattr(self.regressor, 'feature_dim', 1024)))
        decoder_feature_dim = self._glace_decoder_feature_dim() if self._is_glace_backend() else encoder_feature_dim
        requested_lmc_fusion_target = str(getattr(options, 'lmc_fusion_target', 'decoder'))
        if requested_lmc_fusion_target not in ('decoder', 'local'):
            raise ValueError(f"Unsupported lmc_fusion_target={requested_lmc_fusion_target!r}")
        if requested_lmc_fusion_target == 'local' and not self._is_glace_backend():
            raise ValueError("--lmc_fusion_target=local is only supported for model_backend=glace_lmc.")
        effective_lmc_fusion_target = requested_lmc_fusion_target
        fusion_query_dim = decoder_feature_dim if effective_lmc_fusion_target == 'decoder' else encoder_feature_dim
        backbone_feature_dim = fusion_query_dim
        self.requested_lmc_fusion_target = requested_lmc_fusion_target
        self.effective_lmc_fusion_target = effective_lmc_fusion_target
        self.lmc_fusion_target = effective_lmc_fusion_target
        local_residual_mode = str(getattr(options, 'local_residual_mode', 'none'))
        if local_residual_mode not in ('none', 'fixed_alpha', 'learned_alpha'):
            raise ValueError(f"Unsupported local_residual_mode={local_residual_mode!r}")
        local_residual_alpha = float(getattr(options, 'local_residual_alpha', 1.0))
        if not math.isfinite(local_residual_alpha) or local_residual_alpha < 0.0 or local_residual_alpha > 1.0:
            raise ValueError(f"--local_residual_alpha must be in [0,1], got {local_residual_alpha}")
        local_residual_alpha_max_arg = getattr(options, 'local_residual_alpha_max', None)
        local_residual_alpha_max = local_residual_alpha if local_residual_alpha_max_arg is None else float(local_residual_alpha_max_arg)
        if not math.isfinite(local_residual_alpha_max) or local_residual_alpha_max < 0.0 or local_residual_alpha_max > 1.0:
            raise ValueError(f"--local_residual_alpha_max must be in [0,1], got {local_residual_alpha_max}")
        local_residual_alpha_init = float(getattr(options, 'local_residual_alpha_init', 0.001))
        if not math.isfinite(local_residual_alpha_init) or local_residual_alpha_init < 0.0:
            raise ValueError(f"--local_residual_alpha_init must be finite and >=0, got {local_residual_alpha_init}")
        local_residual_alpha_warmup_steps = max(0, int(getattr(options, 'local_residual_alpha_warmup_steps', 0)))
        if local_residual_mode == 'learned_alpha':
            if local_residual_alpha_max <= 0.0:
                raise ValueError("--local_residual_alpha_max must be >0 for learned_alpha.")
            if local_residual_alpha_init > local_residual_alpha_max:
                raise ValueError(
                    f"--local_residual_alpha_init ({local_residual_alpha_init}) must be <= alpha_max ({local_residual_alpha_max})."
                )
        self.local_residual_mode = local_residual_mode
        self.local_residual_alpha = local_residual_alpha
        self.local_residual_alpha_init = local_residual_alpha_init
        self.local_residual_alpha_max = local_residual_alpha_max
        self.local_residual_alpha_warmup_steps = local_residual_alpha_warmup_steps
        self.local_residual_alpha_logit = None
        if self._is_glace_backend() and local_residual_mode == 'learned_alpha':
            init_prob = local_residual_alpha_init / max(local_residual_alpha_max, 1e-12)
            self.local_residual_alpha_logit = torch.nn.Parameter(
                torch.tensor(self._logit_from_probability(init_prob), dtype=torch.float32, device=self.device)
            )

        fusion_query_desc = 'decoder_global_local'
        if self._is_glace_backend() and effective_lmc_fusion_target == 'local':
            fusion_query_desc = 'local_only_global_bypass'
            if local_residual_mode == 'fixed_alpha':
                fusion_query_desc = 'local_only_global_bypass_fixed_alpha'
            elif local_residual_mode == 'learned_alpha':
                fusion_query_desc = 'local_only_global_bypass_learned_alpha'
        elif not self._is_glace_backend():
            fusion_query_desc = 'encoder'
        _logger.info(
            "[LMC] encoder_feature_dim=%d decoder_feature_dim=%d fusion_query_dim=%d fusion_target=%s requested=%s query=%s local_residual=%s alpha=%.4f",
            encoder_feature_dim,
            decoder_feature_dim,
            fusion_query_dim,
            effective_lmc_fusion_target,
            requested_lmc_fusion_target,
            fusion_query_desc,
            local_residual_mode,
            local_residual_alpha,
        )

        # --- Full-pipeline normalization: store mu/sigma for config ---
        norm_mu = bank_data.get("normalization_mu")   # [3] or None
        norm_sigma = bank_data.get("normalization_sigma")  # scalar or None

        if norm_mu is None or norm_sigma is None:
            # Non-BSE or pooled format — no normalization, use world coords
            self.coord_sigma = None
            self.coord_mu = None
            self._depth_min_eff = self.options.depth_min
            self._depth_max_eff = self.options.depth_max
            self._depth_target_eff = self.options.depth_target
            _logger.info("[LMC] No normalization metadata — using raw world coordinates and depth thresholds.")
        else:
            self.coord_sigma = float(norm_sigma.detach().cpu().item())
            self.coord_mu = norm_mu.detach().cpu()  # [3]
            self._depth_min_eff = self.options.depth_min / self.coord_sigma
            self._depth_max_eff = self.options.depth_max / self.coord_sigma
            self._depth_target_eff = self.options.depth_target / self.coord_sigma
            # Override options so all loss functions use scaled thresholds automatically
            self.options.depth_min = self._depth_min_eff
            self.options.depth_max = self._depth_max_eff
            self.options.depth_target = self._depth_target_eff
            _logger.info(
                "[LMC] Normalization active: sigma=%.4f, mu=(%.3f, %.3f, %.3f), "
                "depth_min_eff=%.4f, depth_max_eff=%.4f, depth_target_eff=%.4f (scaled by 1/sigma)",
                self.coord_sigma,
                float(self.coord_mu[0]), float(self.coord_mu[1]), float(self.coord_mu[2]),
                self._depth_min_eff, self._depth_max_eff, self._depth_target_eff,
            )

        pe_normalize_input = bool(getattr(options, 'pe_normalize_input', False))
        requested_compressor_pe_scale_mode = getattr(options, 'lmc_compressor_pe_scale_mode', None)
        if requested_compressor_pe_scale_mode is None:
            compressor_pe_scale_mode = 'std' if pe_normalize_input else 'raw'
        else:
            compressor_pe_scale_mode = str(requested_compressor_pe_scale_mode)
        if compressor_pe_scale_mode not in ('raw', 'std', 'scene_scale'):
            raise ValueError(f"Unsupported lmc_compressor_pe_scale_mode={compressor_pe_scale_mode!r}")
        lmc_fps_start_policy = str(getattr(options, 'lmc_fps_start_policy', 'farthest_from_center'))
        requested_s1_loss_step_mode = getattr(options, 's1_loss_step_mode', None)
        lmc_profile_for_s1_step = str(getattr(options, 'lmc_profile', 'legacy'))
        if requested_s1_loss_step_mode is None:
            s1_loss_step_mode = 'global_monotonic' if lmc_profile_for_s1_step == 'mapany_flow_v1' else 'fixed_zero'
        else:
            s1_loss_step_mode = str(requested_s1_loss_step_mode)
        if s1_loss_step_mode not in ('fixed_zero', 'per_iter', 'global_monotonic'):
            raise ValueError(f"Unsupported s1_loss_step_mode={s1_loss_step_mode!r}")
        _logger.info(
            "[S1] sampled loss step mode: requested=%s resolved=%s",
            requested_s1_loss_step_mode if requested_s1_loss_step_mode is not None else "legacy_default",
            s1_loss_step_mode,
        )
        self.lmc_log_runtime_stats = bool(getattr(options, 'lmc_log_runtime_stats', False))
        self.lmc_runtime_stats_interval = max(1, int(getattr(options, 'lmc_runtime_stats_interval', 100)))
        self.lmc_runtime_stats_max_pixels = max(1, int(getattr(options, 'lmc_runtime_stats_max_pixels', 4096)))
        self._lmc_runtime_stats_calls = 0
        lmc_fusion_geometry_mode = str(getattr(options, 'lmc_fusion_geometry_mode', 'value_only_raw'))
        if lmc_fusion_geometry_mode not in ('value_only_raw', 'value_only_norm', 'geokey_norm'):
            raise ValueError(f"Unsupported lmc_fusion_geometry_mode={lmc_fusion_geometry_mode!r}")
        lmc_fusion_key_geo_init = float(getattr(options, 'lmc_fusion_key_geo_init', 0.0))
        lmc_fusion_refinement_mode = str(getattr(options, 'lmc_fusion_refinement_mode', 'single'))
        if lmc_fusion_refinement_mode not in (
            'single', 'single_qknorm_layerscale',
            'cascade_internal', 'progressive_reread', 'centered_reread',
            'centered_reread_qknorm_layerscale', 'geometry_reread_lite',
            'adapter_ffn', 'weak_residual_ffn', 'v3_adapter_control',
            'v3_dual_refine', 'coord_prior_v1',
        ):
            raise ValueError(f"Unsupported lmc_fusion_refinement_mode={lmc_fusion_refinement_mode!r}")
        lmc_fusion_cascade_layers = int(getattr(options, 'lmc_fusion_cascade_layers', 4))
        if lmc_fusion_cascade_layers < 1:
            raise ValueError(f"lmc_fusion_cascade_layers must be >= 1, got {lmc_fusion_cascade_layers}")
        lmc_fusion_assembly_mode = str(getattr(options, 'lmc_fusion_assembly_mode', 'concat_mlp'))
        if lmc_fusion_assembly_mode not in ('concat_mlp',):
            raise ValueError(f"Unsupported lmc_fusion_assembly_mode={lmc_fusion_assembly_mode!r}")
        lmc_fusion_assembly_gamma_init = float(getattr(options, 'lmc_fusion_assembly_gamma_init', 0.0))
        lmc_fusion_reread_delta_alpha = float(getattr(options, 'lmc_fusion_reread_delta_alpha', 1.0))
        if lmc_fusion_reread_delta_alpha < 0.0:
            raise ValueError(
                f"lmc_fusion_reread_delta_alpha must be >= 0, got {lmc_fusion_reread_delta_alpha}"
            )
        lmc_fusion_reread_scalar_gate = bool(getattr(options, 'lmc_fusion_reread_scalar_gate', False))
        lmc_fusion_reread_gate_init = float(getattr(options, 'lmc_fusion_reread_gate_init', 0.0))
        lmc_fusion_reread_post_norm = bool(getattr(options, 'lmc_fusion_reread_post_norm', True))
        lmc_fusion_reread_trust_region_ratio = float(getattr(options, 'lmc_fusion_reread_trust_region_ratio', 0.0))
        lmc_fusion_reread_temperature = float(getattr(options, 'lmc_fusion_reread_temperature', 1.0))
        lmc_fusion_reread_common_scale = float(getattr(options, 'lmc_fusion_reread_common_scale', 1.0))
        lmc_fusion_reread_effective_ratio_cap = float(
            getattr(options, 'lmc_fusion_reread_effective_ratio_cap', 0.0)
        )
        lmc_fusion_reread_qknorm_eps = float(getattr(options, 'lmc_fusion_reread_qknorm_eps', 1e-6))
        lmc_fusion_reread_qknorm_tau_init = float(
            getattr(options, 'lmc_fusion_reread_qknorm_tau_init', 0.0)
        )
        lmc_fusion_reread_layerscale_patch_init = float(
            getattr(options, 'lmc_fusion_reread_layerscale_patch_init', 0.01)
        )
        lmc_fusion_reread_layerscale_common_init = float(
            getattr(options, 'lmc_fusion_reread_layerscale_common_init', 0.0)
        )
        lmc_fusion_dual_memory_layerscale_patch_init = float(
            getattr(options, 'lmc_fusion_dual_memory_layerscale_patch_init', 0.005)
        )
        lmc_fusion_dual_memory_layerscale_common_init = float(
            getattr(options, 'lmc_fusion_dual_memory_layerscale_common_init', 0.0)
        )
        lmc_fusion_single_qknorm_eps = float(getattr(options, 'lmc_fusion_single_qknorm_eps', 1e-6))
        lmc_fusion_single_qknorm_tau_init = float(
            getattr(options, 'lmc_fusion_single_qknorm_tau_init', 0.0)
        )
        lmc_fusion_single_layerscale_init = float(
            getattr(options, 'lmc_fusion_single_layerscale_init', 1.0)
        )
        lmc_fusion_coord_prior_scale_init = float(
            getattr(options, 'lmc_fusion_coord_prior_scale_init', 0.10)
        )
        lmc_fusion_reread_warmup_mode = str(getattr(options, 'lmc_fusion_reread_warmup_mode', 'none'))
        lmc_fusion_reread_warmup_iters = int(getattr(options, 'lmc_fusion_reread_warmup_iters', 0))
        lmc_fusion_reread_warmup_start = float(getattr(options, 'lmc_fusion_reread_warmup_start', 0.0))
        lmc_fusion_reread_geo_lambda = float(getattr(options, 'lmc_fusion_reread_geo_lambda', 1.0))
        lmc_fusion_reread_geo_sigma = float(getattr(options, 'lmc_fusion_reread_geo_sigma', 1.0))
        lmc_fusion_reread_geo_sigma_mode = str(getattr(options, 'lmc_fusion_reread_geo_sigma_mode', 'fixed'))
        lmc_fusion_reread_geo_sigma_beta = float(getattr(options, 'lmc_fusion_reread_geo_sigma_beta', 1.0))
        lmc_fusion_reread_geo_sigma_min = float(getattr(options, 'lmc_fusion_reread_geo_sigma_min', 0.5))
        if lmc_fusion_reread_trust_region_ratio < 0.0:
            raise ValueError(
                f"lmc_fusion_reread_trust_region_ratio must be >= 0, got {lmc_fusion_reread_trust_region_ratio}"
            )
        if lmc_fusion_reread_temperature <= 0.0:
            raise ValueError(
                f"lmc_fusion_reread_temperature must be > 0, got {lmc_fusion_reread_temperature}"
            )
        if lmc_fusion_reread_common_scale < 0.0:
            raise ValueError(
                f"lmc_fusion_reread_common_scale must be >= 0, got {lmc_fusion_reread_common_scale}"
            )
        if lmc_fusion_reread_effective_ratio_cap < 0.0:
            raise ValueError(
                "lmc_fusion_reread_effective_ratio_cap must be >= 0, got "
                f"{lmc_fusion_reread_effective_ratio_cap}"
            )
        if lmc_fusion_reread_qknorm_eps <= 0.0:
            raise ValueError(
                f"lmc_fusion_reread_qknorm_eps must be > 0, got {lmc_fusion_reread_qknorm_eps}"
            )
        if lmc_fusion_reread_qknorm_tau_init < 0.0:
            raise ValueError(
                "lmc_fusion_reread_qknorm_tau_init must be >= 0; use 0 for sqrt(head_dim), got "
                f"{lmc_fusion_reread_qknorm_tau_init}"
            )
        if lmc_fusion_dual_memory_layerscale_patch_init < 0.0:
            raise ValueError(
                "lmc_fusion_dual_memory_layerscale_patch_init must be >= 0, got "
                f"{lmc_fusion_dual_memory_layerscale_patch_init}"
            )
        if lmc_fusion_dual_memory_layerscale_common_init < 0.0:
            raise ValueError(
                "lmc_fusion_dual_memory_layerscale_common_init must be >= 0, got "
                f"{lmc_fusion_dual_memory_layerscale_common_init}"
            )
        if lmc_fusion_single_qknorm_eps <= 0.0:
            raise ValueError(
                f"lmc_fusion_single_qknorm_eps must be > 0, got {lmc_fusion_single_qknorm_eps}"
            )
        if lmc_fusion_single_qknorm_tau_init < 0.0:
            raise ValueError(
                "lmc_fusion_single_qknorm_tau_init must be >= 0; use 0 for sqrt(head_dim), got "
                f"{lmc_fusion_single_qknorm_tau_init}"
            )
        if lmc_fusion_coord_prior_scale_init < 0.0:
            raise ValueError(
                "lmc_fusion_coord_prior_scale_init must be >= 0, got "
                f"{lmc_fusion_coord_prior_scale_init}"
            )
        if lmc_fusion_reread_warmup_mode not in ('none', 'linear', 'cosine'):
            raise ValueError(
                "lmc_fusion_reread_warmup_mode must be one of none/linear/cosine, got "
                f"{lmc_fusion_reread_warmup_mode!r}"
            )
        if lmc_fusion_reread_warmup_iters < 0:
            raise ValueError(
                f"lmc_fusion_reread_warmup_iters must be >= 0, got {lmc_fusion_reread_warmup_iters}"
            )
        if not 0.0 <= lmc_fusion_reread_warmup_start <= 1.0:
            raise ValueError(
                "lmc_fusion_reread_warmup_start must be in [0,1], got "
                f"{lmc_fusion_reread_warmup_start}"
            )
        if lmc_fusion_reread_geo_lambda < 0.0:
            raise ValueError(f"lmc_fusion_reread_geo_lambda must be >= 0, got {lmc_fusion_reread_geo_lambda}")
        if lmc_fusion_reread_geo_sigma <= 0.0:
            raise ValueError(f"lmc_fusion_reread_geo_sigma must be > 0, got {lmc_fusion_reread_geo_sigma}")
        if lmc_fusion_reread_geo_sigma_mode not in ('fixed', 'adaptive_spread'):
            raise ValueError(
                f"lmc_fusion_reread_geo_sigma_mode must be fixed or adaptive_spread, got {lmc_fusion_reread_geo_sigma_mode}"
            )
        if lmc_fusion_reread_geo_sigma_beta <= 0.0:
            raise ValueError(
                f"lmc_fusion_reread_geo_sigma_beta must be > 0, got {lmc_fusion_reread_geo_sigma_beta}"
            )
        if lmc_fusion_reread_geo_sigma_min <= 0.0:
            raise ValueError(
                f"lmc_fusion_reread_geo_sigma_min must be > 0, got {lmc_fusion_reread_geo_sigma_min}"
            )
        if lmc_fusion_refinement_mode != 'single' and lmc_mode == 'hierarchical':
            raise ValueError('Fusion refinement is only supported for non-hierarchical LMC modes.')
        needs_scene_scale = (
            lmc_fusion_geometry_mode != 'value_only_raw'
            or compressor_pe_scale_mode == 'scene_scale'
        )
        lmc_geometry_scene_scale, lmc_geometry_scene_scale_source = self._resolve_lmc_geometry_scene_scale(
            needs_scene_scale
        )
        _logger.info(
            "[LMC-Geometry] fusion_mode=%s compressor_pe_scale=%s scene_scale=%.6f source=%s key_geo_init=%.6f",
            lmc_fusion_geometry_mode,
            compressor_pe_scale_mode,
            lmc_geometry_scene_scale,
            lmc_geometry_scene_scale_source,
            lmc_fusion_key_geo_init,
        )
        _logger.info(
            "[LMC-Fusion] refinement_mode=%s cascade_layers=%d assembly_mode=%s gamma_init=%.6f single_qknorm_eps=%.2e single_qknorm_tau_init=%.6f single_layerscale_init=%.6f coord_prior_scale_init=%.6f reread_alpha=%.6f scalar_gate=%s gate_init=%.6f post_norm=%s trust_ratio=%.6f temperature=%.6f common_scale=%.6f effective_ratio_cap=%.6f qknorm_eps=%.2e qknorm_tau_init=%.6f layerscale_patch_init=%.6f layerscale_common_init=%.6f dual_memory_patch_init=%.6f dual_memory_common_init=%.6f warmup=%s/%d/start%.3f geo_lambda=%.6f geo_sigma=%.6f geo_sigma_mode=%s geo_sigma_beta=%.6f geo_sigma_min=%.6f",
            lmc_fusion_refinement_mode,
            lmc_fusion_cascade_layers,
            lmc_fusion_assembly_mode,
            lmc_fusion_assembly_gamma_init,
            lmc_fusion_single_qknorm_eps,
            lmc_fusion_single_qknorm_tau_init,
            lmc_fusion_single_layerscale_init,
            lmc_fusion_coord_prior_scale_init,
            lmc_fusion_reread_delta_alpha,
            lmc_fusion_reread_scalar_gate,
            lmc_fusion_reread_gate_init,
            lmc_fusion_reread_post_norm,
            lmc_fusion_reread_trust_region_ratio,
            lmc_fusion_reread_temperature,
            lmc_fusion_reread_common_scale,
            lmc_fusion_reread_effective_ratio_cap,
            lmc_fusion_reread_qknorm_eps,
            lmc_fusion_reread_qknorm_tau_init,
            lmc_fusion_reread_layerscale_patch_init,
            lmc_fusion_reread_layerscale_common_init,
            lmc_fusion_dual_memory_layerscale_patch_init,
            lmc_fusion_dual_memory_layerscale_common_init,
            lmc_fusion_reread_warmup_mode,
            lmc_fusion_reread_warmup_iters,
            lmc_fusion_reread_warmup_start,
            lmc_fusion_reread_geo_lambda,
            lmc_fusion_reread_geo_sigma,
            lmc_fusion_reread_geo_sigma_mode,
            lmc_fusion_reread_geo_sigma_beta,
            lmc_fusion_reread_geo_sigma_min,
        )

        self.lmc_config = {
            'use_lmc': True,
            'lmc_flow': str(getattr(options, 'lmc_flow', 'iterative')),
            'lmc_mode': lmc_mode,
            'requested_lmc_mode': requested_lmc_mode,
            'effective_lmc_mode': lmc_mode,
            'lmc_auto_mode_by_visibility': bool(getattr(options, 'lmc_auto_mode_by_visibility', False)),
            'lmc_visibility_route_metric': str(getattr(options, 'lmc_visibility_route_metric', 'gvcs')),
            'lmc_visibility_route_score_threshold': float(getattr(options, 'lmc_visibility_route_score_threshold', 0.20)),
            'lmc_visibility_front_ratio_threshold': float(getattr(options, 'lmc_visibility_front_ratio_threshold', 0.85)),
            'num_latent_tokens': num_latent_tokens,
            'num_fine': num_fine,
            'num_coarse': num_coarse,
            'num_attn_layers': num_attn_layers,
            'use_scale_token': use_scale_token,
            'compress_dim': feature_dim,
            'num_layers': num_layers,
            'layers_idx': self._tensor_to_config_value(layers_idx),
            'lmc_key_slice_idx': resolved_key_slice_idx,
            'lmc_key_layer_label': self._tensor_to_config_value(key_layer_label),
            'lmc_key_feature_mode': lmc_key_feature_mode,
            'lmc_feature_hierarchy_mode': lmc_feature_hierarchy_mode,
            'lmc_level_merge_mode': lmc_level_merge_mode,
            'lmc_level_merge_init': lmc_level_merge_init,
            'lmc_level_proj_shared': lmc_level_proj_shared,
            'lmc_level_cross_attn_shared': lmc_level_cross_attn_shared,
            'lmc_level_gate_entropy_weight': lmc_level_gate_entropy_weight,
            'lmc_level_token_gate': lmc_level_token_gate,
            'lmc_level_anchor_residual_gamma_init': lmc_level_anchor_residual_gamma_init,
            'final_lmc_level_anchor_residual_gamma': None,
            'lmc_level_merge_weights': None,
            'lmc_level_gate_entropy': None,
            'geo_bias_mode': lmc_geo_bias_mode,
            'geo_bias_rbf_scales': lmc_geo_bias_rbf_scales,
            'geo_bias_rbf_alpha_init': lmc_geo_bias_rbf_alpha_init,
            'geo_bias_rbf_learn_weights': lmc_geo_bias_rbf_learn_weights,
            'geo_bias_rbf_per_head': lmc_geo_bias_rbf_per_head,
            'final_geo_bias_rbf_alpha': None,
            'final_geo_bias_rbf_weights': None,
            'attention_bias_stats': None,
            'pos_encoding_mode': lmc_pos_encoding_mode,
            'pos_fourier_v2_scales': lmc_pos_fourier_v2_scales,
            'pos_fourier_coord_norm': lmc_pos_fourier_coord_norm,
            'pos_fourier_radius': lmc_pos_fourier_radius,
            'pos_fourier_learnable_scale': lmc_pos_fourier_learnable_scale,
            'pos_fourier_residual_gate_init': lmc_pos_fourier_residual_gate_init,
            'final_pos_fourier_residual_gate': None,
            'point_rope_coord_norm': lmc_point_rope_coord_norm,
            'point_rope_radius': lmc_point_rope_radius,
            'point_rope_radius_policy': lmc_point_rope_radius_policy,
            'point_rope_mixed_memory_ratio': lmc_point_rope_mixed_memory_ratio,
            'point_rope_seed_pe': lmc_point_rope_seed_pe,
            'point_rope_base': lmc_point_rope_base,
            'point_rope_axes': lmc_point_rope_axes,
            'point_rope_apply_to': lmc_point_rope_apply_to,
            'geo_bias_crpb_dim': lmc_geo_bias_crpb_dim,
            'geo_bias_crpb_input': lmc_geo_bias_crpb_input,
            'geo_bias_crpb_radius': lmc_geo_bias_crpb_radius,
            'geo_bias_crpb_per_head': lmc_geo_bias_crpb_per_head,
            'geo_bias_crpb_zero_init': lmc_geo_bias_crpb_zero_init,
            'geo_sigma': geo_sigma,
            'pe_normalize_input': pe_normalize_input,
            'lmc_compressor_pe_scale_mode': compressor_pe_scale_mode,
            'lmc_compressor_pe_scene_scale': lmc_geometry_scene_scale,
            'lmc_fps_start_policy': lmc_fps_start_policy,
            's1_loss_step_mode': s1_loss_step_mode,
            'requested_s1_loss_step_mode': requested_s1_loss_step_mode,
            'lmc_log_runtime_stats': self.lmc_log_runtime_stats,
            'lmc_runtime_stats_interval': self.lmc_runtime_stats_interval,
            'lmc_runtime_stats_max_pixels': self.lmc_runtime_stats_max_pixels,
            'lmc_fusion_geometry_mode': lmc_fusion_geometry_mode,
            'lmc_fusion_key_geo_init': lmc_fusion_key_geo_init,
            'lmc_fusion_scene_scale': lmc_geometry_scene_scale,
            'lmc_fusion_scene_scale_source': lmc_geometry_scene_scale_source,
            'lmc_fusion_refinement_mode': lmc_fusion_refinement_mode,
            'lmc_fusion_single_qknorm_eps': lmc_fusion_single_qknorm_eps,
            'lmc_fusion_single_qknorm_tau_init': lmc_fusion_single_qknorm_tau_init,
            'lmc_fusion_single_layerscale_init': lmc_fusion_single_layerscale_init,
            'lmc_fusion_coord_prior_scale_init': lmc_fusion_coord_prior_scale_init,
            'lmc_fusion_cascade_layers': lmc_fusion_cascade_layers,
            'lmc_fusion_assembly_mode': lmc_fusion_assembly_mode,
            'lmc_fusion_assembly_gamma_init': lmc_fusion_assembly_gamma_init,
            'final_lmc_fusion_assembly_gamma': None,
            'lmc_fusion_reread_delta_alpha': lmc_fusion_reread_delta_alpha,
            'lmc_fusion_reread_scalar_gate': lmc_fusion_reread_scalar_gate,
            'lmc_fusion_reread_gate_init': lmc_fusion_reread_gate_init,
            'lmc_fusion_reread_post_norm': lmc_fusion_reread_post_norm,
            'lmc_fusion_reread_trust_region_ratio': lmc_fusion_reread_trust_region_ratio,
            'lmc_fusion_reread_temperature': lmc_fusion_reread_temperature,
            'lmc_fusion_reread_common_scale': lmc_fusion_reread_common_scale,
            'lmc_fusion_reread_effective_ratio_cap': lmc_fusion_reread_effective_ratio_cap,
            'lmc_fusion_reread_qknorm_eps': lmc_fusion_reread_qknorm_eps,
            'lmc_fusion_reread_qknorm_tau_init': lmc_fusion_reread_qknorm_tau_init,
            'lmc_fusion_reread_layerscale_patch_init': lmc_fusion_reread_layerscale_patch_init,
            'lmc_fusion_reread_layerscale_common_init': lmc_fusion_reread_layerscale_common_init,
            'lmc_fusion_dual_memory_layerscale_patch_init': lmc_fusion_dual_memory_layerscale_patch_init,
            'lmc_fusion_dual_memory_layerscale_common_init': lmc_fusion_dual_memory_layerscale_common_init,
            'lmc_fusion_reread_warmup_mode': lmc_fusion_reread_warmup_mode,
            'lmc_fusion_reread_warmup_iters': lmc_fusion_reread_warmup_iters,
            'lmc_fusion_reread_warmup_start': lmc_fusion_reread_warmup_start,
            'lmc_fusion_reread_geo_lambda': lmc_fusion_reread_geo_lambda,
            'lmc_fusion_reread_geo_sigma': lmc_fusion_reread_geo_sigma,
            'lmc_fusion_reread_geo_sigma_mode': lmc_fusion_reread_geo_sigma_mode,
            'lmc_fusion_reread_geo_sigma_beta': lmc_fusion_reread_geo_sigma_beta,
            'lmc_fusion_reread_geo_sigma_min': lmc_fusion_reread_geo_sigma_min,
            'final_lmc_fusion_reread_gate': None,
            'final_lmc_fusion_reread_gate_logit': None,
            'final_lmc_fusion_reread_gamma_patch_mean': None,
            'final_lmc_fusion_reread_gamma_patch_absmax': None,
            'final_lmc_fusion_reread_gamma_common_mean': None,
            'final_lmc_fusion_reread_gamma_common_absmax': None,
            'final_lmc_fusion_dual_memory_gamma_patch_mean': None,
            'final_lmc_fusion_dual_memory_gamma_patch_absmax': None,
            'final_lmc_fusion_dual_memory_gamma_common_mean': None,
            'final_lmc_fusion_dual_memory_gamma_common_absmax': None,
            'final_lmc_fusion_coord_prior_gamma': None,
            'ace_g_fusion_in_s2': bool(getattr(options, 'ace_g_fusion_in_s2', False)),
            'lmc_fusion_target': effective_lmc_fusion_target,
            'requested_lmc_fusion_target': requested_lmc_fusion_target,
            'effective_lmc_fusion_target': effective_lmc_fusion_target,
            'fusion_query_dim': fusion_query_dim,
            'local_residual_mode': local_residual_mode,
            'local_residual_alpha': local_residual_alpha,
            'local_residual_alpha_init': local_residual_alpha_init,
            'local_residual_alpha_max': local_residual_alpha_max,
            'local_residual_alpha_warmup_steps': local_residual_alpha_warmup_steps,
            'final_local_residual_alpha': None,
            'backbone_feature_dim': backbone_feature_dim,
            'encoder_feature_dim': encoder_feature_dim,
            'memory_feature_dim': feature_dim,
            'scale_token_dim': scale_token_dim,
            'memory_path': str(memory_path),
            'model_backend': self.model_backend,
            'ace_encoder_path': str(getattr(options, 'ace_encoder_path', '')),
            'ace_lmc_global_head_mode': self._ace_lmc_global_head_mode() if self._is_ace_fcn_backend() else 'none',
            'ace_lmc_local_checkpoint_path': str(getattr(options, 'ace_lmc_local_checkpoint_path', '') or ''),
            'ace_lmc_freeze_local_stack': bool(getattr(options, 'ace_lmc_freeze_local_stack', True)),
            'ace_lmc_global_feature_mode': str(getattr(options, 'ace_lmc_global_feature_mode', 'glace')),
            'ace_lmc_stage2_feature_source': str(getattr(options, 'ace_lmc_stage2_feature_source', 'raw_backbone')),
            'ace_lmc_global_normalize': bool(getattr(options, 'ace_lmc_global_normalize', False)),
            'ace_lmc_global_noise_std': float(getattr(options, 'ace_lmc_global_noise_std', 0.0)),
            'ace_lmc_global_feature_dim': int(getattr(self, 'glace_global_feat_dim', getattr(self.dataset, 'global_feat_dim', 0)) or 0),
            'ace_lmc_global_gate_init': float(getattr(options, 'ace_lmc_global_gate_init', 1.0)),
            'ace_lmc_global_gate_learnable': bool(getattr(options, 'ace_lmc_global_gate_learnable', False)),
            'ace_lmc_global_gate_max': float(getattr(options, 'ace_lmc_global_gate_max', 0.0)),
            'ace_lmc_global_gate_l1_weight': float(getattr(options, 'ace_lmc_global_gate_l1_weight', 0.0)),
            'ace_lmc_global_residual_gate_l1_weight': float(getattr(options, 'ace_lmc_global_residual_gate_l1_weight', 0.0)),
            'ace_lmc_global_residual_delta_max_m': float(getattr(options, 'ace_lmc_global_residual_delta_max_m', 1.0)),
            'ace_lmc_global_residual_use_xyz_condition': bool(getattr(options, 'ace_lmc_global_residual_use_xyz_condition', True)),
            'ace_lmc_global_residual_xyz_condition_mode': (
                str(getattr(options, 'ace_lmc_global_residual_xyz_condition_mode', 'fourier_adaln') or 'fourier_adaln').lower()
                if bool(getattr(options, 'ace_lmc_global_residual_use_xyz_condition', True)) else 'none'
            ),
            'ace_lmc_global_residual_xyz_condition_scale': float(getattr(options, 'ace_lmc_global_residual_xyz_condition_scale', 10.0)),
            'ace_lmc_global_residual_bad_gate_weight': float(getattr(options, 'ace_lmc_global_residual_bad_gate_weight', 0.0)),
            'ace_lmc_random_global_seed': int(getattr(options, 'ace_lmc_random_global_seed', 20260531)),
            'ace_lmc_stage2_consistency_weight': float(getattr(options, 'ace_lmc_stage2_consistency_weight', 0.0)),
            'ace_lmc_stage2_consistency_loss': str(getattr(options, 'ace_lmc_stage2_consistency_loss', 'smooth_l1')),
            'ace_lmc_stage2_consistency_warmup_steps': int(getattr(options, 'ace_lmc_stage2_consistency_warmup_steps', 0)),
            'ace_lmc_stage2_consistency_sample_limit': int(getattr(options, 'ace_lmc_stage2_consistency_sample_limit', 0)),
            'ace_lmc_stage2_guard_weight': float(getattr(options, 'ace_lmc_stage2_guard_weight', 0.0)),
            'ace_lmc_stage2_guard_margin_px': float(getattr(options, 'ace_lmc_stage2_guard_margin_px', 0.25)),
            'ace_lmc_stage2_guard_max_px': float(getattr(options, 'ace_lmc_stage2_guard_max_px', 100.0)),
            'ace_lmc_stage2_guard_warmup_steps': int(getattr(options, 'ace_lmc_stage2_guard_warmup_steps', 0)),
            'final_ace_lmc_global_gate': None,
            'ace_lmc_allow_mismatched_memory': bool(getattr(options, 'ace_lmc_allow_mismatched_memory', False)),
            'ace_lmc_memory_feature_source': str(bank_data.get('feature_source') or 'unknown'),
            'ace_lmc_memory_output_subsample': bank_data.get('output_subsample', bank_data.get('patch_stride', None)),
            'ace_lmc_memory_coord_source': str(bank_data.get('coord_source') or 'unknown'),
            'ace_lmc_final_head_dim': int(getattr(self.regressor, 'ace_lmc_final_head_dim', backbone_feature_dim)),
            'ace_lmc_head_impl': str(getattr(self.regressor, 'ace_lmc_head_impl', 'ace_head')),
            'data_backend': str(getattr(options, 'data_backend', 'ace')),
            'wai_repo_root': str(getattr(options, 'wai_repo_root', '')),
            'wai_image_modality': str(getattr(options, 'wai_image_modality', 'image')),
            'glace_root': str(getattr(options, 'glace_root', '')),
            'glace_encoder_path': str(getattr(options, 'glace_encoder_path', '')),
            'glace_init_head_path': str(getattr(options, 'glace_init_head_path', '') or ''),
            'glace_feat_name': str(getattr(options, 'glace_feat_name', 'features.npy')),
            'glace_global_feat_dim': int(getattr(self, 'glace_global_feat_dim', 0)),
            'glace_head_channels': int(getattr(options, 'glace_head_channels', 512)),
            'glace_mlp_ratio': float(getattr(options, 'glace_mlp_ratio', 1.0)),
            'glace_fusion_query': fusion_query_desc if self._is_glace_backend() else '',
            'glace_residual_mode': str(getattr(options, 'glace_residual_mode', 'local_delta_tanh_scalar')) if self._is_glace_backend() else '',
            'glace_residual_gate_init': float(getattr(options, 'glace_residual_gate_init', 0.0)) if self._is_glace_backend() else 0.0,
            'glace_residual_global_dim': int(getattr(self, 'glace_global_feat_dim', 0)) if self._is_glace_backend() else 0,
            'glace_head_freeze_iters': int(getattr(options, 'glace_head_freeze_iters', 0)) if self._is_glace_backend() else 0,
            'glace_residual_lr_ratio': float(getattr(options, 'glace_residual_lr_ratio', -1.0)) if self._is_glace_backend() else -1.0,
            'glace_antiregression_weight': float(getattr(options, 'glace_antiregression_weight', 0.0)) if self._is_glace_backend() else 0.0,
            'glace_antiregression_margin_px': float(getattr(options, 'glace_antiregression_margin_px', 0.25)) if self._is_glace_backend() else 0.0,
            'glace_antiregression_max_px': float(getattr(options, 'glace_antiregression_max_px', 100.0)) if self._is_glace_backend() else 0.0,
            'glace_pixel_diag_interval': int(getattr(options, 'glace_pixel_diag_interval', 100)) if self._is_glace_backend() else 0,
            'glace_freeze_base_network': bool(getattr(options, 'glace_freeze_base_network', True)) if self._is_glace_backend() else False,
            'glace_freeze_encoder': self._glace_freeze_encoder() if self._is_glace_backend() else False,
            'glace_freeze_head': self._glace_freeze_head() if self._is_glace_backend() else False,
            # Normalization metadata for test-time de-normalization
            'normalization_mu': self.coord_mu.cpu().tolist() if self.coord_mu is not None else None,
            'normalization_sigma': self.coord_sigma if self.coord_sigma is not None else None,
            'memory_mode': self.memory_contract_info['memory_mode'],
            'memory_contract_mode': self.memory_contract_info['contract_mode'],
            'memory_reference_index': self.memory_contract_info['reference_index'],
            'memory_training_target': self.memory_contract_info['training_target'],
            'reference_output_space': self.reference_contract_state['output_space'],
            'c1_ref_norm_alpha': float(self.reference_contract_state.get('normalization_ref', {}).get('alpha', 1.0) or 1.0),
            'c1_aux_ref_loss_weight': float(getattr(self.options, 'c1_aux_ref_loss_weight', 0.0)),
            'c1_aux_ref_sample_ratio': float(getattr(self.options, 'c1_aux_ref_sample_ratio', 0.5)),
            'use_relative_depth_loss': bool(getattr(self.options, 'use_relative_depth_loss', False)),
            'relative_depth_apply_to': str(getattr(self.options, 'relative_depth_apply_to', 'stage2')),
            'relative_depth_teacher': str(getattr(self.options, 'relative_depth_teacher', 'depth_anything_v2_online')),
            'relative_depth_teacher_encoder': str(getattr(self.options, 'relative_depth_teacher_encoder', 'vitb')),
            'relative_depth_teacher_checkpoint': str(getattr(self.options, 'relative_depth_teacher_checkpoint', '') or ''),
            'relative_depth_teacher_input_size': int(getattr(self.options, 'relative_depth_teacher_input_size', 518)),
            'relative_depth_loss_weight': float(getattr(self.options, 'relative_depth_loss_weight', 0.05)),
            'relative_depth_start_ratio': float(getattr(self.options, 'relative_depth_start_ratio', 0.3)),
            'relative_depth_pair_weight': float(getattr(self.options, 'relative_depth_pair_weight', 0.5)),
            'relative_depth_max_samples': int(getattr(self.options, 'relative_depth_max_samples', 1024)),
            'relative_depth_max_pairs': int(getattr(self.options, 'relative_depth_max_pairs', 4096)),
            'relative_depth_min_points': int(getattr(self.options, 'relative_depth_min_points', 16)),
            'relative_depth_image_step_interval': int(getattr(self.options, 'relative_depth_image_step_interval', 10)),
            'relative_depth_image_batch_size': int(getattr(self.options, 'relative_depth_image_batch_size', 1)),
            'memory_has_points_ref': self.memory_contract_info['has_points_ref'],
            'memory_has_points_ref_norm': self.memory_contract_info['has_points_ref_norm'],
            'memory_has_scene_center_world': self.memory_contract_info['has_scene_center_world'],
            'memory_has_scene_center_ref': self.memory_contract_info['has_scene_center_ref'],
            'memory_has_scene_center_ref_norm': self.memory_contract_info['has_scene_center_ref_norm'],
            'conditioning_reference': self._tensor_to_config_value(
                self.reference_contract_state.get('conditioning_reference')
            ),
            'normalization_ref': self._tensor_to_config_value(
                self.reference_contract_state.get('normalization_ref')
            ),
        }
        if self.reference_contract_state.get("enabled", False):
            self._align_head_mean_to_scene_center()

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
            pe_normalize_input=pe_normalize_input,
            pe_scale_mode=compressor_pe_scale_mode,
            pe_scene_scale=lmc_geometry_scene_scale,
            fps_start_policy=lmc_fps_start_policy,
            key_slice_idx=resolved_key_slice_idx,
            key_feature_mode=lmc_key_feature_mode,
            feature_hierarchy_mode=lmc_feature_hierarchy_mode,
            level_merge_mode=lmc_level_merge_mode,
            level_merge_init=lmc_level_merge_init,
            level_proj_shared=lmc_level_proj_shared,
            level_cross_attn_shared=lmc_level_cross_attn_shared,
            level_gate_entropy_weight=lmc_level_gate_entropy_weight,
            level_token_gate=lmc_level_token_gate,
            level_anchor_residual_gamma_init=lmc_level_anchor_residual_gamma_init,
            geo_bias_mode=lmc_geo_bias_mode,
            geo_bias_rbf_scales=lmc_geo_bias_rbf_scales,
            geo_bias_rbf_alpha_init=lmc_geo_bias_rbf_alpha_init,
            geo_bias_rbf_learn_weights=lmc_geo_bias_rbf_learn_weights,
            geo_bias_rbf_per_head=lmc_geo_bias_rbf_per_head,
            pos_encoding_mode=lmc_pos_encoding_mode,
            pos_fourier_v2_scales=lmc_pos_fourier_v2_scales,
            pos_fourier_coord_norm=lmc_pos_fourier_coord_norm,
            pos_fourier_radius=lmc_pos_fourier_radius,
            pos_fourier_learnable_scale=lmc_pos_fourier_learnable_scale,
            pos_fourier_residual_gate_init=lmc_pos_fourier_residual_gate_init,
            point_rope_coord_norm=lmc_point_rope_coord_norm,
            point_rope_radius=lmc_point_rope_radius,
            point_rope_radius_policy=lmc_point_rope_radius_policy,
            point_rope_mixed_memory_ratio=lmc_point_rope_mixed_memory_ratio,
            point_rope_seed_pe=lmc_point_rope_seed_pe,
            point_rope_base=lmc_point_rope_base,
            point_rope_axes=lmc_point_rope_axes,
            point_rope_apply_to=lmc_point_rope_apply_to,
            geo_bias_crpb_dim=lmc_geo_bias_crpb_dim,
            geo_bias_crpb_input=lmc_geo_bias_crpb_input,
            geo_bias_crpb_radius=lmc_geo_bias_crpb_radius,
            geo_bias_crpb_per_head=lmc_geo_bias_crpb_per_head,
            geo_bias_crpb_zero_init=lmc_geo_bias_crpb_zero_init,
        ).to(self.device)
        self.compressor.collect_runtime_stats = self.lmc_log_runtime_stats

        # --- Build fusion (query dimension follows --lmc_fusion_target, memory=compressor output feature_dim) ---
        self.fusion = LMCFeatureFusion(
            feature_dim=backbone_feature_dim,
            mode=lmc_mode,
            query_feature_dim=backbone_feature_dim,
            memory_feature_dim=feature_dim,
            fusion_geometry_mode=lmc_fusion_geometry_mode,
            fusion_scene_scale=lmc_geometry_scene_scale,
            fusion_key_geo_init=lmc_fusion_key_geo_init,
            fusion_refinement_mode=lmc_fusion_refinement_mode,
            fusion_cascade_layers=lmc_fusion_cascade_layers,
            fusion_assembly_mode=lmc_fusion_assembly_mode,
            fusion_assembly_gamma_init=lmc_fusion_assembly_gamma_init,
            fusion_reread_delta_alpha=lmc_fusion_reread_delta_alpha,
            fusion_reread_scalar_gate=lmc_fusion_reread_scalar_gate,
            fusion_reread_gate_init=lmc_fusion_reread_gate_init,
            fusion_reread_post_norm=lmc_fusion_reread_post_norm,
            fusion_reread_trust_region_ratio=lmc_fusion_reread_trust_region_ratio,
            fusion_reread_temperature=lmc_fusion_reread_temperature,
            fusion_reread_common_scale=lmc_fusion_reread_common_scale,
            fusion_reread_effective_ratio_cap=lmc_fusion_reread_effective_ratio_cap,
            fusion_reread_qknorm_eps=lmc_fusion_reread_qknorm_eps,
            fusion_reread_qknorm_tau_init=lmc_fusion_reread_qknorm_tau_init,
            fusion_reread_layerscale_patch_init=lmc_fusion_reread_layerscale_patch_init,
            fusion_reread_layerscale_common_init=lmc_fusion_reread_layerscale_common_init,
            fusion_dual_memory_layerscale_patch_init=lmc_fusion_dual_memory_layerscale_patch_init,
            fusion_dual_memory_layerscale_common_init=lmc_fusion_dual_memory_layerscale_common_init,
            fusion_single_qknorm_eps=lmc_fusion_single_qknorm_eps,
            fusion_single_qknorm_tau_init=lmc_fusion_single_qknorm_tau_init,
            fusion_single_layerscale_init=lmc_fusion_single_layerscale_init,
            fusion_coord_prior_scale_init=lmc_fusion_coord_prior_scale_init,
            fusion_reread_warmup_mode=lmc_fusion_reread_warmup_mode,
            fusion_reread_warmup_iters=lmc_fusion_reread_warmup_iters,
            fusion_reread_warmup_start=lmc_fusion_reread_warmup_start,
            fusion_reread_geo_lambda=lmc_fusion_reread_geo_lambda,
            fusion_reread_geo_sigma=lmc_fusion_reread_geo_sigma,
            fusion_reread_geo_sigma_mode=lmc_fusion_reread_geo_sigma_mode,
            fusion_reread_geo_sigma_beta=lmc_fusion_reread_geo_sigma_beta,
            fusion_reread_geo_sigma_min=lmc_fusion_reread_geo_sigma_min,
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
        self.s2_lr_boost_first = getattr(options, 's2_lr_boost_first', 1.2)
        self.s2_lr_boost_later = getattr(options, 's2_lr_boost_later', 1.0)
        self.s2_lr_warmup_steps = getattr(options, 's2_lr_warmup_steps', 0)
        self.s2_polish_epochs = max(0, int(getattr(options, 's2_polish_epochs', 0)))
        self.s2_polish_head_lr = float(getattr(options, 's2_polish_head_lr', 1e-4))
        self.s2_polish_fusion_lr_ratio = max(0.0, float(getattr(options, 's2_polish_fusion_lr_ratio', 0.005)))
        self.lmc_profile = str(getattr(options, 'lmc_profile', 'legacy'))
        self.mapany_flow_profile = (self.lmc_profile == 'mapany_flow_v1')
        self.s1_loss_step_mode = s1_loss_step_mode
        self.s1_use_buffer = bool(getattr(options, 's1_use_buffer', False))
        self.s1_buffer_refill_mode = str(getattr(options, 's1_buffer_refill_mode', 'full'))
        if self.s1_use_buffer and str(getattr(options, 's1_loss_mode', 'full_map')) == 'full_map':
            raise ValueError("S1 buffer mode requires s1_loss_mode != 'full_map'.")

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

        # ReproLoss time axis:
        # - legacy: keep S2-only timeline + step rewind behavior.
        # - mapany_flow_v1: use monotonic global step across S1+S2 (map-anything style).
        buf_per_it = self.options.training_buffer_size // self.options.batch_size
        buf_final = self.buffer_size_final // self.options.batch_size
        total_s2_steps = (self.lmc_iterations - 1) * self.options.epochs * buf_per_it + self.options.epochs * buf_final
        total_s2_steps = max(total_s2_steps, 1)
        total_s1_steps = self.lmc_warmup_steps + max(0, self.lmc_iterations - 1) * self.lmc_train_steps
        if self.mapany_flow_profile:
            repro_total_iterations = max(total_s1_steps + total_s2_steps, 1)
            self.repro_step_mode = "global_monotonic"
        else:
            repro_total_iterations = total_s2_steps
            # Determine step_eff mode: 'auto' chooses 'per_iter' when ace_g_fusion_in_s2=True
            _ace_g_fusion_in_s2 = (
                str(getattr(options, 'lmc_flow', 'ace')) == 'ace_g'
                and bool(getattr(options, 'ace_g_fusion_in_s2', False))
            )
            _s2_step_eff_mode = str(getattr(options, 's2_step_eff_mode', 'auto'))
            if _s2_step_eff_mode == 'auto':
                _s2_step_eff_mode = 'per_iter' if _ace_g_fusion_in_s2 else 'legacy_rewind'
            self.repro_step_mode = _s2_step_eff_mode
        self.repro_loss = ReproLoss(
            total_iterations=repro_total_iterations,
            soft_clamp=self.options.repro_loss_soft_clamp,
            soft_clamp_min=self.options.repro_loss_soft_clamp_min,
            type=self.options.repro_loss_type,
            circle_schedule=(self.options.repro_loss_schedule == 'circle'),
        )
        self.global_s2_step = 0
        self.local_s2_step = 0
        self.current_lmc_iteration = 0
        self.current_lmc_iter = 0
        self.steps_per_s2_phase = 0
        self.s2_rewind_amount = 0.0
        self.optimizer_head = None
        self.scheduler_head = None
        self._s2_compressor_out = None  # Cached compressor output for ACE-G S2
        self._s2_update_applied_last = True
        # S2 stability options (see options_dinov2_lmc.py for docs)
        self._loss_invalid_max_delta = float(getattr(options, 'loss_invalid_max_delta', 1000.0))
        self._s2_grad_clip_max_norm = float(getattr(options, 's2_grad_clip_max_norm', 1.0))
        self._s2_nan_guard_enabled = bool(getattr(options, 's2_nan_guard', True))
        self._s2_nan_guard_patience = max(1, int(getattr(options, 's2_nan_guard_patience', 2)))
        self._s2_nan_guard_naninf_ratio = min(1.0, max(0.0, float(getattr(options, 's2_nan_guard_naninf_ratio', 0.99))))
        self._s2_nan_guard_bad_steps = 0
        self._s2_abort_current_iteration = False
        self._abort_lmc_training = False
        self.relative_depth_distiller = None
        self._relative_depth_warned_no_img_idx = False
        self._relative_depth_warned_nonfinite = False
        self._relative_depth_logged_active = False
        self._relative_depth_image_loader = None
        self._relative_depth_image_iterator = None
        self._init_relative_depth_distiller()

        self._load_ace_lmc_local_checkpoint_if_requested()

        # Rebuild optimizer to include compressor + fusion params (used only when not in LMC; S1/S2 use their own)
        self._rebuild_optimizer()
        self._log_glace_lmc_config_diagnostics()
        self._log_stage_trainability("LMC-init")
        self._log_optimizer_groups("LMC-init", self.optimizer)

        _logger.info(
            f"[LMC] mode={lmc_mode}, tokens={num_latent_tokens}, "
            f"attn_layers={num_attn_layers}, iterations={self.lmc_iterations}, "
            f"profile={self.lmc_profile}, repro_step_mode={self.repro_step_mode}, "
            f"repro_total_iterations={self.repro_loss.total_iterations}, "
            f"loss_invalid_max_delta={self._loss_invalid_max_delta}, "
            f"s2_grad_clip={self._s2_grad_clip_max_norm}"
        )
        self._load_resume_checkpoint_if_requested()

    def _resolve_buffer_schema_dim(self, dim_spec):
        if dim_spec == "feature_dim":
            return int(self._buffer_feature_dim())
        return int(dim_spec)

    def _expected_training_buffer_feature_dtype(self):
        return torch.float16 if bool(getattr(self.options, "use_half", False)) else torch.float32

    def _current_buffer_should_be_on_cpu(self):
        if bool(getattr(self.options, "buffer_on_cpu", True)):
            return True
        return bool(getattr(self, "_force_current_buffer_on_cpu", False))

    def _expected_training_buffer_device_type(self):
        current_buffer_on_cpu = getattr(self, "_current_training_buffer_on_cpu", None)
        if current_buffer_on_cpu is None:
            current_buffer_on_cpu = self._current_buffer_should_be_on_cpu()
        if bool(current_buffer_on_cpu):
            return "cpu"
        return self.device.type

    def _get_s1_early_stop_cfg(self):
        return {
            "enabled": bool(getattr(self.options, "s1_early_stop", True)),
            "min_updates": max(1, int(getattr(self.options, "s1_early_stop_min_updates", 400))),
            "patience": max(1, int(getattr(self.options, "s1_early_stop_patience", 180))),
            "rel_improve": float(getattr(self.options, "s1_early_stop_rel_improve", 0.01)),
            "ema_beta": min(0.999, max(0.0, float(getattr(self.options, "s1_early_stop_ema_beta", 0.90)))),
        }

    @staticmethod
    def _resolve_lmc_feature_dim(pooled_features_dim, num_layers, layers_idx, memory_path):
        feature_dim = int(pooled_features_dim) // int(num_layers)
        if feature_dim * int(num_layers) != int(pooled_features_dim):
            raise ValueError(
                "[LMC] pooled_features_dim is not divisible by num_layers; "
                f"pooled_features_dim={pooled_features_dim}, num_layers={num_layers}, "
                f"layers_idx={layers_idx if layers_idx else 'n/a'}, memory_path={memory_path}"
            )
        return feature_dim

    def _pack_feature_rows_for_head(self, stage_tag, features_bC, *aligned_tensors, grid_h=16):
        """Pack sampled feature rows into the fixed H x W grid expected by the 1x1 ACE head."""
        batch_size = int(features_bC.shape[0])
        h = int(grid_h)
        if h <= 0:
            raise ValueError(f"grid_h must be positive, got {grid_h}.")
        w = batch_size // h
        trimmed_batch = h * w
        if trimmed_batch <= 0:
            return None, None, None, None, (None,) * len(aligned_tensors)
        if trimmed_batch != batch_size:
            trim_count = batch_size - trimmed_batch
            if not hasattr(self, "_head_grid_trim_stats"):
                self._head_grid_trim_stats = {}
            stats = self._head_grid_trim_stats.setdefault(
                stage_tag,
                {"trim_events": 0, "trimmed_rows": 0},
            )
            stats["trim_events"] += 1
            stats["trimmed_rows"] += int(trim_count)
            if not hasattr(self, "_logged_grid_trim_stages"):
                self._logged_grid_trim_stages = set()
            if stage_tag not in self._logged_grid_trim_stages:
                _logger.warning(
                    "[%s] batch_size=%d is not divisible by %d; trimming %d tail samples to %d (=%dx%d). "
                    "cumulative_trim_events=%d cumulative_trimmed_rows=%d",
                    stage_tag,
                    batch_size,
                    h,
                    trim_count,
                    trimmed_batch,
                    h,
                    w,
                    stats["trim_events"],
                    stats["trimmed_rows"],
                )
                self._logged_grid_trim_stages.add(stage_tag)
            features_bC = features_bC[:trimmed_batch]
            aligned_tensors = tuple(t[:trimmed_batch] for t in aligned_tensors)
        return features_bC, trimmed_batch, h, w, aligned_tensors

    def _trim_batch_for_head_grid(self, stage_tag, features_bC, *aligned_tensors):
        """Compatibility wrapper for older call sites; prefer _pack_feature_rows_for_head."""
        return self._pack_feature_rows_for_head(stage_tag, features_bC, *aligned_tensors)

    def _validate_training_buffer_schema(self, buffer_dict, schema_name, expected_size=None):
        schema = self.BUFFER_SCHEMA_SPECS.get(schema_name)
        if schema is None:
            raise ValueError(f"Unknown training buffer schema: {schema_name!r}")
        if not isinstance(buffer_dict, dict):
            raise ValueError(f"{schema_name} must be a dict, got {type(buffer_dict).__name__}.")

        field_specs = schema["fields"]
        missing_keys = [key for key in field_specs if key not in buffer_dict]
        if missing_keys:
            raise ValueError(f"{schema_name} missing keys: {missing_keys}")

        expected_device_type = self._expected_training_buffer_device_type()
        batch_size = None
        for key, spec in field_specs.items():
            tensor = buffer_dict[key]
            if not isinstance(tensor, torch.Tensor):
                raise ValueError(f"{schema_name}.{key} must be a torch.Tensor, got {type(tensor).__name__}.")
            if tensor.dim() != spec["rank"]:
                raise ValueError(
                    f"{schema_name}.{key} has wrong rank: got {tensor.dim()}, expected {spec['rank']}."
                )

            current_batch = int(tensor.shape[0])
            if batch_size is None:
                batch_size = current_batch
            elif current_batch != batch_size:
                raise ValueError(
                    f"{schema_name}.{key} has inconsistent batch size: got {current_batch}, expected {batch_size}."
                )

            expected_shape = tuple(self._resolve_buffer_schema_dim(dim) for dim in spec["shape_suffix"])
            actual_shape = tuple(int(dim) for dim in tensor.shape[1:])
            if actual_shape != expected_shape:
                raise ValueError(
                    f"{schema_name}.{key} has wrong shape suffix: got {actual_shape}, expected {expected_shape}."
                )

            expected_dtype = (
                self._expected_training_buffer_feature_dtype()
                if spec["dtype"] == "feature"
                else spec["dtype"]
            )
            if tensor.dtype != expected_dtype:
                raise ValueError(
                    f"{schema_name}.{key} has wrong dtype: got {tensor.dtype}, expected {expected_dtype}."
                )
            if tensor.device.type != expected_device_type:
                raise ValueError(
                    f"{schema_name}.{key} has wrong device: got {tensor.device.type}, expected {expected_device_type}."
                )

        if batch_size is None:
            raise ValueError(f"{schema_name} is empty.")
        if expected_size is not None and batch_size != int(expected_size):
            raise ValueError(
                f"{schema_name} has wrong batch size: got {batch_size}, expected {int(expected_size)}."
            )
        return batch_size

    def _resolve_s1_partial_refill_counts(self, total_size):
        total_size = int(total_size)
        if total_size <= 0:
            raise ValueError(f"total_size must be > 0, got {total_size}.")

        keep_ratio = float(getattr(self.options, "s1_buffer_keep_ratio", 0.5))
        refill_ratio = getattr(self.options, "s1_buffer_refill_ratio", None)
        if keep_ratio < 0.0 or keep_ratio > 1.0:
            raise ValueError(f"s1_buffer_keep_ratio must be in [0, 1], got {keep_ratio}.")
        if refill_ratio is not None:
            refill_ratio = float(refill_ratio)
            if refill_ratio < 0.0 or refill_ratio > 1.0:
                raise ValueError(f"s1_buffer_refill_ratio must be in [0, 1], got {refill_ratio}.")
            if not math.isclose(keep_ratio + refill_ratio, 1.0, rel_tol=0.0, abs_tol=1e-6):
                raise ValueError(
                    f"s1_buffer_keep_ratio + s1_buffer_refill_ratio must sum to 1, got {keep_ratio + refill_ratio:.6f}."
                )
        else:
            refill_ratio = 1.0 - keep_ratio

        keep_count = int(round(total_size * keep_ratio))
        keep_count = min(max(keep_count, 0), total_size)
        refill_count = total_size - keep_count
        if keep_count <= 0 or refill_count <= 0:
            raise ValueError(
                f"partial refill requires both keep/refill counts > 0, got keep={keep_count}, refill={refill_count}."
            )
        return keep_count, refill_count

    def _slice_training_buffer_rows(self, buffer_dict, schema_name, row_indices):
        self._validate_training_buffer_schema(
            buffer_dict=buffer_dict,
            schema_name=schema_name,
            expected_size=buffer_dict["features"].shape[0],
        )
        row_indices = row_indices.to(buffer_dict["features"].device)
        return {
            key: value[row_indices].contiguous()
            for key, value in buffer_dict.items()
        }

    def _merge_training_buffers(self, schema_name, buffers, expected_size=None):
        buffers = [buf for buf in buffers if buf is not None]
        if not buffers:
            raise ValueError("buffers must contain at least one buffer dict.")

        merged = {}
        expected_keys = tuple(self.BUFFER_SCHEMA_SPECS[schema_name]["fields"].keys())
        for buf in buffers:
            self._validate_training_buffer_schema(
                buffer_dict=buf,
                schema_name=schema_name,
                expected_size=buf["features"].shape[0],
            )
        for key in expected_keys:
            merged[key] = torch.cat([buf[key] for buf in buffers], dim=0).contiguous()
        self._validate_training_buffer_schema(
            buffer_dict=merged,
            schema_name=schema_name,
            expected_size=expected_size,
        )
        return merged

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

        # Skip only when the provided center itself is the synthetic zero anchor.
        if (
            hasattr(self, 'coord_sigma')
            and self.coord_sigma is not None
            and mem_center.numel() >= 3
            and float(torch.linalg.norm(mem_center[:3]).item()) <= 1e-6
        ):
            _logger.info(
                "[LMC] Skipping center distance check (normalized coordinates, scene_center=0). "
                "dataset.mean_cam_center=(%.3f, %.3f, %.3f)",
                float(dataset_center[0]), float(dataset_center[1]), float(dataset_center[2]),
            )
        elif dataset_center.numel() >= 3 and mem_center.numel() >= 3:
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

    def _create_training_buffer_with_scene_coords(self, buffer_size=None):
        """Local variant of create_training_buffer() that also stores GT scene coordinates."""
        torch.backends.cudnn.benchmark = False
        effective_size = self.options.training_buffer_size if buffer_size is None else buffer_size
        self._current_buffer_size = effective_size

        buffer_batch_size = getattr(self.options, 'buffer_batch_size', 10)
        if self._is_ace_fcn_backend() and buffer_batch_size != 1:
            _logger.info('[ACE-FCN-LMC] Forcing buffer_batch_size=1 because the ACE dataset keeps variable image widths.')
            buffer_batch_size = 1
        if buffer_batch_size > 1:
            buffer_image_width = getattr(self.options, 'buffer_image_width', None)
            if buffer_image_width is None:
                buffer_image_width = (self.options.image_resolution * 4 // 3 + 13) // 14 * 14
            buffer_dataset = self._build_train_dataset(
                image_width=buffer_image_width,
                augment=False,
                aug_rotation=0,
                aug_scale_max=1.0,
                aug_scale_min=1.0,
            )
        else:
            buffer_dataset = self.dataset

        batch_sampler = sampler.BatchSampler(
            sampler.RandomSampler(buffer_dataset, generator=self.batch_generator),
            batch_size=buffer_batch_size,
            drop_last=False,
        )

        def seed_worker(worker_id):
            worker_seed = torch.initial_seed() % 2**32
            np.random.seed(worker_seed)
            random.seed(worker_seed)

        training_dataloader = DataLoader(
            dataset=buffer_dataset,
            sampler=batch_sampler,
            batch_size=None,
            worker_init_fn=seed_worker,
            generator=self.loader_generator,
            pin_memory=True,
            num_workers=self.num_data_loader_workers,
            persistent_workers=self.num_data_loader_workers > 0,
            timeout=60 if self.num_data_loader_workers > 0 else 0,
        )

        _logger.info("Starting creation of the training buffer.")
        buffer_on_cpu = self._current_buffer_should_be_on_cpu()
        self._current_training_buffer_on_cpu = buffer_on_cpu
        buffer_device = torch.device("cpu") if buffer_on_cpu else self.device
        if buffer_on_cpu:
            reason = (
                "buffer_on_cpu=True"
                if bool(getattr(self.options, "buffer_on_cpu", True))
                else "buffer_on_cpu_final=True"
            )
            _logger.info("Buffer will be allocated on CPU (%s) to avoid GPU OOM.", reason)

        self.training_buffer = {
            'features': torch.empty(
                (effective_size, self._buffer_feature_dim()),
                dtype=(torch.float32, torch.float16)[self.options.use_half],
                device=buffer_device,
            ),
            'target_px': torch.empty((effective_size, 2), dtype=torch.float32, device=buffer_device),
            'gt_poses_inv': torch.empty((effective_size, 3, 4), dtype=torch.float32, device=buffer_device),
            'intrinsics': torch.empty((effective_size, 3, 3), dtype=torch.float32, device=buffer_device),
            'intrinsics_inv': torch.empty((effective_size, 3, 3), dtype=torch.float32, device=buffer_device),
            'gt_scene_coords_world': torch.empty((effective_size, 3), dtype=torch.float32, device=buffer_device),
            'gt_scene_coords_valid': torch.empty((effective_size, 1), dtype=torch.bool, device=buffer_device),
        }
        if self._needs_image_indices_in_buffer():
            self.training_buffer['img_idx'] = torch.empty((effective_size,), dtype=torch.int64, device=buffer_device)

        regressor_mode_snapshot = self._capture_module_training_modes(
            self.regressor,
            getattr(self.regressor, "encoder", None),
            getattr(self.regressor, "heads", None),
        )
        self.regressor.eval()
        with torch.no_grad():
            buffer_idx = 0
            dataset_passes = 0
            sampled_total = 0
            sampled_duplicates = 0
            self._buffer_sample_valid_coord_selected = 0
            self._buffer_sample_valid_coord_available = 0
            self._buffer_sample_valid_coord_seed_available = 0
            self._buffer_sample_valid_coord_roi_available = 0
            self._buffer_sample_valid_coord_neighbor_available = 0
            self._buffer_sample_random_selected = 0
            pbar = tqdm(
                total=effective_size,
                unit="samples",
                unit_scale=True,
                desc="Buffer",
                dynamic_ncols=True,
            )

            while buffer_idx < effective_size:
                dataset_passes += 1
                for batch in training_dataloader:
                    image_BCHW, image_mask_B1HW, _, gt_pose_inv_B44, intrinsics_B33, intrinsics_inv_B33, *_ = batch
                    img_idx_B = None
                    if self._needs_image_indices_in_buffer():
                        img_idx_B = batch[-1]
                    coords_B3HW = self._extract_gt_scene_coords_from_batch(batch, image_BCHW=image_BCHW)

                    image_BCHW = image_BCHW.to(self.device, non_blocking=True)
                    image_mask_B1HW = image_mask_B1HW.to(self.device, non_blocking=True)
                    gt_pose_inv_B44 = gt_pose_inv_B44.to(self.device, non_blocking=True)
                    intrinsics_B33 = intrinsics_B33.to(self.device, non_blocking=True)
                    intrinsics_inv_B33 = intrinsics_inv_B33.to(self.device, non_blocking=True)
                    if coords_B3HW is not None:
                        coords_B3HW = coords_B3HW.to(self.device, non_blocking=True).float()
                    else:
                        self._warn_missing_aux_ref_coords_once()

                    if gt_pose_inv_B44.shape[1] == 4:
                        gt_pose_inv_B34 = gt_pose_inv_B44[:, :3, :]
                    else:
                        gt_pose_inv_B34 = gt_pose_inv_B44

                    if image_BCHW.dtype == torch.float16:
                        image_BCHW = image_BCHW.float()

                    with autocast("cuda", enabled=self.options.use_half):
                        local_features_BCHW = self.regressor.get_features(image_BCHW)
                    if self._is_glace_backend():
                        if img_idx_B is None:
                            raise ValueError('[Buffer] GLACE backend batch is missing img_idx.')
                        img_idx_B = self._resolve_batch_image_indices(img_idx_B, device=self.device)
                        features_BCHW = self._build_glace_decoder_feature_maps(
                            local_features_BCHW,
                            img_idx_B,
                            stage_tag='Buffer',
                        )
                    else:
                        if self._uses_image_global_features() and img_idx_B is None:
                            raise ValueError('[Buffer] image-global backend batch is missing img_idx.')
                        if img_idx_B is not None:
                            img_idx_B = self._resolve_batch_image_indices(img_idx_B, device=self.device)
                        features_BCHW = local_features_BCHW

                    B, C, H, W = features_BCHW.shape
                    image_mask_B1HW = TF.resize(image_mask_B1HW, [H, W], interpolation=TF.InterpolationMode.NEAREST)
                    image_mask_B1HW = image_mask_B1HW.bool()
                    if coords_B3HW is None:
                        coords_B3HW = torch.zeros((B, 3, H, W), dtype=torch.float32, device=self.device)
                        coords_valid_B1HW = torch.zeros((B, 1, H, W), dtype=torch.bool, device=self.device)
                    elif tuple(coords_B3HW.shape[-2:]) != (H, W):
                        coords_valid_B1HW = self._scene_coords_valid_mask(coords_B3HW, size_hw=(H, W), device=self.device)
                        coords_B3HW = Fnn.interpolate(coords_B3HW, size=(H, W), mode="nearest")
                    else:
                        coords_valid_B1HW = self._scene_coords_valid_mask(coords_B3HW, size_hw=(H, W), device=self.device)
                    coord_sampling_B1HW = self._expand_valid_coord_sampling_mask(
                        coords_valid_B1HW,
                        image_mask_B1HW,
                    )

                    if image_mask_B1HW.sum() == 0:
                        continue

                    pixel_positions_B2HW = self.pixel_grid_2HW[:, :H, :W].clone().unsqueeze(0).expand(B, 2, H, W)
                    gt_pose_inv = gt_pose_inv_B34.unsqueeze(1).expand(B, H * W, 3, 4).reshape(-1, 3, 4)
                    intrinsics = intrinsics_B33.unsqueeze(1).expand(B, H * W, 3, 3).reshape(-1, 3, 3)
                    intrinsics_inv = intrinsics_inv_B33.unsqueeze(1).expand(B, H * W, 3, 3).reshape(-1, 3, 3)

                    def normalize_shape(tensor_in):
                        return tensor_in.transpose(0, 1).flatten(1).transpose(0, 1)

                    batch_data = {
                        'features': normalize_shape(features_BCHW),
                        'target_px': normalize_shape(pixel_positions_B2HW),
                        'gt_poses_inv': gt_pose_inv,
                        'intrinsics': intrinsics,
                        'intrinsics_inv': intrinsics_inv,
                        'gt_scene_coords_world': normalize_shape(coords_B3HW),
                        'gt_scene_coords_valid': normalize_shape(coords_valid_B1HW),
                    }
                    if self._needs_image_indices_in_buffer():
                        if img_idx_B is None:
                            raise ValueError('[Buffer] image-global run requires img_idx but batch is missing it.')
                        batch_data['img_idx'] = img_idx_B.unsqueeze(1).expand(B, H * W).reshape(-1)

                    image_mask_B1HW = image_mask_B1HW.float()
                    image_mask_N1 = normalize_shape(image_mask_B1HW)
                    replacement_cfg = getattr(self.options, "buffer_sampling_replacement", None)
                    use_replacement = True if replacement_cfg is None else bool(replacement_cfg)
                    features_to_select = min(
                        self.options.samples_per_image * B,
                        effective_size - buffer_idx,
                    )
                    if not use_replacement:
                        valid_count = int((image_mask_N1.view(-1) > 0).sum().item())
                        features_to_select = min(features_to_select, valid_count)
                    if features_to_select <= 0:
                        continue
                    coord_valid_N1 = normalize_shape(coord_sampling_B1HW)
                    sample_idxs = self._sample_buffer_indices(
                        image_mask_N1,
                        coord_valid_N1,
                        features_to_select,
                        use_replacement,
                    )
                    if sample_idxs is None or sample_idxs.numel() <= 0:
                        continue
                    sampled_total += int(sample_idxs.numel())
                    sampled_duplicates += int(sample_idxs.numel() - torch.unique(sample_idxs).numel())
                    features_to_select = int(sample_idxs.numel())

                    for k in batch_data:
                        batch_data[k] = batch_data[k][sample_idxs].to(buffer_device, non_blocking=True)
                    buffer_offset = buffer_idx + features_to_select
                    for k in batch_data:
                        self.training_buffer[k][buffer_idx:buffer_offset] = batch_data[k]

                    buffer_idx = buffer_offset
                    pbar.update(features_to_select)
                    pbar.set_postfix(n_pass=dataset_passes)
                    if buffer_idx >= effective_size:
                        break

            pbar.close()

        buffer_memory = sum(v.element_size() * v.nelement() for v in self.training_buffer.values()) / (1024**3)
        dup_ratio = (sampled_duplicates / sampled_total) if sampled_total > 0 else 0.0
        replacement_cfg = getattr(self.options, "buffer_sampling_replacement", None)
        use_replacement = True if replacement_cfg is None else bool(replacement_cfg)
        _logger.info("Created buffer of {:.2f}GB with {} passes (buffer_batch_size={}).".format(
            buffer_memory, dataset_passes, buffer_batch_size))
        _logger.info(
            "Buffer sampling stats: replacement=%s, duplicate_ratio=%.4f (%d/%d).",
            use_replacement,
            dup_ratio,
            sampled_duplicates,
            sampled_total,
        )
        valid_selected = int(getattr(self, "_buffer_sample_valid_coord_selected", 0))
        random_selected = int(getattr(self, "_buffer_sample_random_selected", 0))
        if valid_selected > 0 or bool(getattr(self.options, "buffer_sample_valid_coords", True)):
            seed_available = int(getattr(self, "_buffer_sample_valid_coord_seed_available", 0))
            roi_available = int(getattr(self, "_buffer_sample_valid_coord_roi_available", 0))
            neighbor_available = int(getattr(self, "_buffer_sample_valid_coord_neighbor_available", 0))
            _logger.info(
                "Buffer valid-depth sampling: enabled=%s, selected_valid=%d, selected_random=%d, "
                "valid_ratio=%.4f, neighbor_radius=%d, neighbor_mode=%s, "
                "seed_available=%d, roi_available=%d, neighbor_available=%d.",
                bool(getattr(self.options, "buffer_sample_valid_coords", True)),
                valid_selected,
                random_selected,
                valid_selected / max(valid_selected + random_selected, 1),
                int(getattr(self.options, "buffer_valid_coord_neighbor_radius", 1)),
                str(getattr(self.options, "buffer_valid_coord_neighbor_mode", "cross")),
                seed_available,
                roi_available,
                neighbor_available,
            )
        self._restore_module_training_modes(regressor_mode_snapshot)

    # ------------------------------------------------------------------
    # Optimizer rebuild (includes compressor + fusion + head)
    # ------------------------------------------------------------------

    def _rebuild_optimizer(self):
        """Rebuild optimizer & scheduler to cover compressor + fusion + head."""
        params = (
            list(self.compressor.parameters()) +
            list(self.fusion.parameters()) +
            self._glace_residual_adapter_params() +
            self._local_residual_gate_params() +
            self._ace_lmc_global_gate_params() +
            list(self.regressor.heads.parameters())
        )
        self.optimizer = optim.AdamW([{'name': 'lmc_all', 'params': params}], lr=self.options.learning_rate_min)

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

    def _monotonic_repro_step(self):
        """Monotonic repro-loss step used by mapany_flow_v1 profile."""
        return min(int(self.iteration), self.repro_loss.total_iterations - 1)

    def _resolve_s1_sampled_loss_step(self, s1_step=0):
        """Resolve ReproLoss step for sampled S1 loss without changing legacy defaults."""
        mode = getattr(self, 's1_loss_step_mode', 'fixed_zero')
        if mode == 'fixed_zero':
            return 0
        if mode == 'per_iter':
            return min(max(int(s1_step), 0), self.repro_loss.total_iterations - 1)
        if mode == 'global_monotonic':
            return self._monotonic_repro_step()
        raise ValueError(f"Unsupported s1_loss_step_mode={mode!r}")

    def _compute_reprojection_invalid_loss_contract(
        self,
        pred_scene_coords_N31,
        target_px_N2,
        gt_inv_poses_N34,
        Ks_N33,
        invKs_N33,
        *,
        step_eff,
        normalizer,
        include_mask_N=None,
        apply_nuclear_mask=False,
        valid_subsample_max=0,
        invalid_max_delta=None,
        invalid_posinf=1e4,
        invalid_neginf=1e4,
    ):
        """Shared ACE reprojection + invalid proxy loss.

        This helper intentionally preserves existing stage-specific policy at
        the call site: S1/S2 step selection, S1 full-map masks, and S2 invalid
        clamping are still explicit arguments rather than hidden defaults.
        """
        n_points = int(pred_scene_coords_N31.shape[0])
        pred_scene_coords_N41 = to_homogeneous(pred_scene_coords_N31)
        pred_cam_coords_N31 = torch.bmm(gt_inv_poses_N34, pred_scene_coords_N41)
        pred_px_N31 = torch.bmm(Ks_N33, pred_cam_coords_N31)
        pred_px_N31[:, 2].clamp_(min=self.options.depth_min)
        pred_px_N21 = pred_px_N31[:, :2] / pred_px_N31[:, 2, None]

        reprojection_error_N2 = pred_px_N21.squeeze() - target_px_N2
        reprojection_error_l1_N1 = torch.norm(reprojection_error_N2, dim=1, keepdim=True, p=1)
        reprojection_error_l2_N1 = torch.norm(reprojection_error_N2, dim=1, keepdim=True, p=2)

        finite_repro_N = torch.isfinite(reprojection_error_l1_N1).flatten()
        finite_cam_N = torch.isfinite(pred_cam_coords_N31).all(dim=1).flatten()
        invalid_min_depth_N = (pred_cam_coords_N31[:, 2] < self.options.depth_min).flatten()
        invalid_repro_N = (reprojection_error_l1_N1 > self.options.repro_loss_hard_clamp).flatten()
        invalid_max_depth_N = (pred_cam_coords_N31[:, 2] > self.options.depth_max).flatten()
        invalid_nonfinite_N = ~(finite_repro_N & finite_cam_N)
        base_invalid_mask_N = (
            invalid_min_depth_N
            | invalid_repro_N
            | invalid_max_depth_N
            | invalid_nonfinite_N
        ).flatten()

        if include_mask_N is None:
            include_mask_N = torch.ones(n_points, device=pred_scene_coords_N31.device, dtype=torch.bool)
        else:
            include_mask_N = include_mask_N.reshape(-1).to(device=pred_scene_coords_N31.device, dtype=torch.bool)

        if apply_nuclear_mask:
            repro_flat_N = reprojection_error_l1_N1.flatten()
            coords_max_N = torch.abs(pred_scene_coords_N31).view(n_points, -1).max(dim=1)[0]
            nuclear_mask_N = (
                (repro_flat_N > self.SANITY_PIXEL_ERR)
                | (coords_max_N > self.SANITY_COORD_VAL)
                | (~torch.isfinite(repro_flat_N))
                | (~torch.isfinite(pred_cam_coords_N31).all(dim=1).flatten())
            ).flatten()
        else:
            nuclear_mask_N = torch.zeros(n_points, device=pred_scene_coords_N31.device, dtype=torch.bool)

        valid_mask_N = include_mask_N & (~base_invalid_mask_N) & (~nuclear_mask_N)
        invalid_mask_N = include_mask_N & base_invalid_mask_N & (~nuclear_mask_N)

        max_points = int(valid_subsample_max)
        if max_points > 0 and valid_mask_N.sum() > max_points:
            valid_idx = torch.where(valid_mask_N)[0]
            perm = torch.randperm(
                valid_idx.numel(),
                device=valid_idx.device,
                generator=getattr(self, "sampling_generator", None),
            )
            valid_reprojection_error = reprojection_error_l1_N1[valid_idx[perm[:max_points]]]
        else:
            valid_reprojection_error = reprojection_error_l1_N1[valid_mask_N]

        if valid_reprojection_error.numel() > 0:
            loss_valid = self.repro_loss.compute(valid_reprojection_error, int(step_eff))
            if not isinstance(loss_valid, torch.Tensor):
                loss_valid = torch.tensor(
                    loss_valid,
                    device=pred_scene_coords_N31.device,
                    dtype=pred_scene_coords_N31.dtype,
                )
        else:
            loss_valid = torch.zeros(
                (),
                device=pred_scene_coords_N31.device,
                dtype=pred_scene_coords_N31.dtype,
            )

        pixel_grid_crop_N31 = to_homogeneous(target_px_N2.unsqueeze(2))
        target_camera_coords_N31 = self.options.depth_target * torch.bmm(invKs_N33, pixel_grid_crop_N31)
        invalid_mask_N11 = invalid_mask_N.reshape(n_points, 1, 1)
        delta_cam_N31 = torch.abs(target_camera_coords_N31 - pred_cam_coords_N31)
        delta_cam_N31 = torch.nan_to_num(
            delta_cam_N31,
            nan=0.0,
            posinf=invalid_posinf,
            neginf=invalid_neginf,
        )
        if invalid_max_delta is not None and float(invalid_max_delta) > 0:
            delta_cam_N31 = delta_cam_N31.clamp(max=float(invalid_max_delta))
        loss_invalid = delta_cam_N31.masked_select(invalid_mask_N11).sum()
        loss = (loss_valid + loss_invalid) / normalizer

        finite_l1 = reprojection_error_l1_N1[torch.isfinite(reprojection_error_l1_N1)]
        finite_l2 = reprojection_error_l2_N1[torch.isfinite(reprojection_error_l2_N1)]
        stats = {
            "fraction_valid": float(valid_mask_N.sum() / normalizer),
            "pxerr_l1": float(finite_l1.mean().item()) if finite_l1.numel() > 0 else float("nan"),
            "pxerr_l2": float(finite_l2.mean().item()) if finite_l2.numel() > 0 else float("nan"),
            "valid_pxerr_l1": (
                float(valid_reprojection_error.mean().item())
                if valid_reprojection_error.numel() > 0
                else float("nan")
            ),
            "nonfinite_ratio": float(invalid_nonfinite_N.float().mean().item()),
            "nuclear_cnt": int(nuclear_mask_N.sum().item()),
        }
        return {
            "loss": loss,
            "stats": stats,
            "reprojection_error_l1": reprojection_error_l1_N1,
            "valid_mask": valid_mask_N,
            "invalid_mask": invalid_mask_N,
        }

    # ------------------------------------------------------------------
    # Head reset
    # ------------------------------------------------------------------

    def _align_head_mean_to_scene_center(self):
        """After head reset in LMC path: set head.mean buffer to memory scene_center so coordinate anchor is consistent."""
        if not self.use_lmc or not hasattr(self, 'memory_dict'):
            return
        contract = getattr(self, "reference_contract_state", None)
        if contract and contract.get("enabled", False):
            target = contract.get("head_mean")
            if target is None:
                return
            target = target.detach().to(self.device, dtype=self.regressor.heads.mean.dtype).view(1, 3, 1, 1)
            self.regressor.heads.mean.copy_(target)
            _logger.info(
                "[LMC] Head mean aligned to %s anchor for C1. target=(%.3f,%.3f,%.3f)",
                contract["output_space"],
                float(target.view(-1)[0]),
                float(target.view(-1)[1]),
                float(target.view(-1)[2]),
            )
            return
        sc = self.memory_dict.get('scene_center')
        if sc is None:
            return
        sc = sc.detach().to(self.device, dtype=self.regressor.heads.mean.dtype)
        if sc.dim() == 1:
            sc = sc.unsqueeze(0)

        # Full-pipeline normalization: scene_center is zeros, head.mean should be zeros
        # Skip the shift check since ||zeros - dataset.mean_cam_center|| will be large
        if hasattr(self, 'coord_sigma') and self.coord_sigma is not None:
            self.regressor.heads.mean.copy_(sc.view(1, 3, 1, 1))
            _logger.info(
                "[LMC] Head mean set to scene_center (normalized coords, shift check skipped). target=(%.3f,%.3f,%.3f)",
                float(sc.view(-1)[0]), float(sc.view(-1)[1]), float(sc.view(-1)[2]),
            )
            return

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
        if self._is_glace_backend() and getattr(self.options, "glace_init_head_path", None):
            if iteration_idx == 0:
                _logger.info(
                    "[GLACE-LMC] Keeping GLACE init scene head; skipping LMC head reset "
                    "(glace_init_head_path=%s).",
                    getattr(self.options, "glace_init_head_path", None),
                )
            return
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
        mode_snapshot = self._capture_module_training_modes(self.compressor)
        self.compressor.eval()
        try:
            with torch.no_grad():
                out = self.compressor(self.memory_dict)
            self._log_compressor_runtime_stats(out, "Compress")
            return out
        finally:
            self._restore_module_training_modes(mode_snapshot)

    def _set_compressor_trainable(self, trainable: bool):
        """Explicit S1/S2 compressor trainability boundary."""
        for param in self.compressor.parameters():
            param.requires_grad_(trainable)
        _logger.info("[LMC] Compressor trainable=%s", trainable)

    def _log_stage_trainability(self, stage_tag: str):
        """Log trainable parameter counts for stage-contract debugging."""
        if not self.use_lmc:
            return
        adapter = getattr(self, "glace_residual_adapter", None)
        gate_params = self._local_residual_gate_params()
        gate_trainable = sum(p.numel() for p in gate_params if p.requires_grad)
        gate_total = sum(p.numel() for p in gate_params)
        _logger.info(
            "[%s] trainable params: encoder=%d/%d head=%d/%d compressor=%d/%d fusion=%d/%d residual_adapter=%d/%d local_residual_gate=%d/%d alpha_eff=%.6f",
            stage_tag,
            self._count_trainable_params(getattr(self.regressor, "encoder", None)),
            self._count_total_params(getattr(self.regressor, "encoder", None)),
            self._count_trainable_params(getattr(self.regressor, "heads", None)),
            self._count_total_params(getattr(self.regressor, "heads", None)),
            self._count_trainable_params(getattr(self, "compressor", None)),
            self._count_total_params(getattr(self, "compressor", None)),
            self._count_trainable_params(getattr(self, "fusion", None)),
            self._count_total_params(getattr(self, "fusion", None)),
            self._count_trainable_params(adapter),
            self._count_total_params(adapter),
            gate_trainable,
            gate_total,
            self._current_local_residual_alpha_value() if gate_total > 0 or getattr(self, 'local_residual_mode', 'none') != 'none' else -1.0,
        )

    def _validate_s2_compressor_contract(self):
        """S2-G must not train compressor parameters."""
        if not self.use_lmc or self.lmc_flow != 'ace_g':
            return
        leaked = [name for name, param in self.compressor.named_parameters() if param.requires_grad]
        if leaked:
            preview = ", ".join(leaked[:5])
            raise RuntimeError(
                f"[S2-G] Compressor must be frozen during S2, but {len(leaked)} params require grad: {preview}"
            )
        if self._optimizer_contains_module_params(getattr(self, "optimizer_head", None), self.compressor):
            raise RuntimeError("[S2-G] optimizer_head unexpectedly contains compressor parameters.")

    def _validate_ace_lmc_stage2_fused_backbone_contract(self, stage_tag: str, *, check_optimizer: bool = True):
        if not self._ace_lmc_stage2_uses_fused_buffer():
            return
        frozen_modules = {
            'encoder': getattr(self.regressor, 'encoder', None),
            'compressor': self.compressor,
            'fusion': self.fusion,
        }
        for name, module in frozen_modules.items():
            if module is None:
                continue
            trainable = self._count_trainable_params(module)
            if trainable > 0:
                raise RuntimeError(f"[{stage_tag}] stage1_fused requires frozen {name}, but trainable_params={trainable}.")
            if check_optimizer and self._optimizer_contains_module_params(getattr(self, 'optimizer_head', None), module):
                raise RuntimeError(f"[{stage_tag}] stage1_fused optimizer unexpectedly contains {name} parameters.")
        if self.fusion.training:
            raise RuntimeError(f"[{stage_tag}] stage1_fused requires fusion.eval(); fusion is in train mode.")
        if self._uses_ace_lmc_global_residual_head():
            base_head = self._ace_lmc_global_residual_base_head()
            trainable = self._count_trainable_params(base_head)
            if trainable > 0:
                raise RuntimeError(f"[{stage_tag}] stage1_fused/glace_residual requires frozen Stage1 base head, but trainable_params={trainable}.")
            if check_optimizer and self._optimizer_contains_module_params(getattr(self, 'optimizer_head', None), base_head):
                raise RuntimeError(f"[{stage_tag}] stage1_fused/glace_residual optimizer unexpectedly contains Stage1 base head parameters.")

    def _lmc_semantics_for_logs(self):
        if not self.use_lmc:
            return {}
        keys = [
            "requested_lmc_mode",
            "effective_lmc_mode",
            "lmc_mode",
            "layers_idx",
            "lmc_key_slice_idx",
            "lmc_key_layer_label",
            "lmc_key_feature_mode",
            "lmc_key_mix_weights",
            "lmc_feature_hierarchy_mode",
            "lmc_level_merge_mode",
            "lmc_level_merge_init",
            "lmc_level_proj_shared",
            "lmc_level_cross_attn_shared",
            "lmc_level_gate_entropy_weight",
            "lmc_level_token_gate",
            "lmc_level_anchor_residual_gamma_init",
            "final_lmc_level_anchor_residual_gamma",
            "lmc_level_merge_weights",
            "lmc_level_gate_entropy",
            "geo_bias_mode",
            "geo_bias_rbf_scales",
            "geo_bias_rbf_alpha_init",
            "geo_bias_rbf_learn_weights",
            "geo_bias_rbf_per_head",
            "final_geo_bias_rbf_alpha",
            "final_geo_bias_rbf_weights",
            "attention_bias_stats",
            "pos_encoding_mode",
            "pos_fourier_v2_scales",
            "pos_fourier_coord_norm",
            "pos_fourier_radius",
            "pos_fourier_learnable_scale",
            "pos_fourier_residual_gate_init",
            "final_pos_fourier_residual_gate",
            "point_rope_coord_norm",
            "point_rope_radius",
            "point_rope_radius_policy",
            "point_rope_mixed_memory_ratio",
            "point_rope_seed_pe",
            "point_rope_base",
            "point_rope_axes",
            "point_rope_apply_to",
            "geo_bias_crpb_dim",
            "geo_bias_crpb_input",
            "geo_bias_crpb_radius",
            "geo_bias_crpb_per_head",
            "geo_bias_crpb_zero_init",
            "lmc_fps_start_policy",
            "pe_normalize_input",
            "lmc_compressor_pe_scale_mode",
            "lmc_compressor_pe_scene_scale",
            "s1_loss_step_mode",
            "requested_s1_loss_step_mode",
            "lmc_log_runtime_stats",
            "lmc_runtime_stats_interval",
            "lmc_runtime_stats_max_pixels",
            "lmc_fusion_geometry_mode",
            "lmc_fusion_key_geo_init",
            "lmc_fusion_scene_scale",
            "lmc_fusion_scene_scale_source",
            "lmc_fusion_refinement_mode",
            "lmc_fusion_coord_prior_scale_init",
            "lmc_fusion_cascade_layers",
            "lmc_fusion_assembly_mode",
            "lmc_fusion_assembly_gamma_init",
            "lmc_fusion_reread_delta_alpha",
            "lmc_fusion_reread_scalar_gate",
            "lmc_fusion_reread_gate_init",
            "lmc_fusion_reread_post_norm",
            "lmc_fusion_reread_trust_region_ratio",
            "lmc_fusion_reread_temperature",
            "lmc_fusion_reread_common_scale", "lmc_fusion_reread_effective_ratio_cap",
            "lmc_fusion_reread_qknorm_eps", "lmc_fusion_reread_qknorm_tau_init",
            "lmc_fusion_reread_layerscale_patch_init", "lmc_fusion_reread_layerscale_common_init",
            "lmc_fusion_dual_memory_layerscale_patch_init", "lmc_fusion_dual_memory_layerscale_common_init",
            "lmc_fusion_reread_warmup_mode", "lmc_fusion_reread_warmup_iters",
            "lmc_fusion_reread_warmup_start",
            "lmc_fusion_reread_geo_lambda",
            "lmc_fusion_reread_geo_sigma",
            "lmc_fusion_reread_geo_sigma_mode",
            "lmc_fusion_reread_geo_sigma_beta",
            "lmc_fusion_reread_geo_sigma_min",
            "local_residual_mode",
            "local_residual_alpha",
            "local_residual_alpha_init",
            "local_residual_alpha_max",
            "local_residual_alpha_warmup_steps",
            "final_local_residual_alpha",
            "ace_encoder_path",
            "ace_lmc_global_head_mode",
            "ace_lmc_local_checkpoint_path",
            "ace_lmc_freeze_local_stack",
            "ace_lmc_stage2_feature_source",
            "ace_lmc_global_normalize",
            "ace_lmc_global_noise_std",
            "ace_lmc_final_head_dim",
            "ace_lmc_head_impl",
        ]
        key_mix_logits = getattr(getattr(self, "compressor", None), "key_mix_logits", None)
        if key_mix_logits is not None:
            weights = torch.softmax(key_mix_logits.detach().float().cpu(), dim=0)
            self.lmc_config["lmc_key_mix_weights"] = weights.tolist()
        level_stats = getattr(getattr(self, "compressor", None), "last_levelwise_runtime_stats", None)
        if isinstance(level_stats, dict):
            self.lmc_config["lmc_level_merge_weights"] = level_stats.get("lmc_level_merge_weights")
            self.lmc_config["lmc_level_gate_entropy"] = level_stats.get("lmc_level_gate_entropy")
            self.lmc_config["final_lmc_level_anchor_residual_gamma"] = level_stats.get(
                "lmc_level_anchor_residual_gamma"
            )
        geo_bias_stats = getattr(getattr(self, "compressor", None), "last_geo_bias_runtime_stats", None)
        if isinstance(geo_bias_stats, dict):
            self.lmc_config["final_geo_bias_rbf_alpha"] = geo_bias_stats.get("final_geo_bias_rbf_alpha")
            self.lmc_config["final_geo_bias_rbf_weights"] = geo_bias_stats.get("final_geo_bias_rbf_weights")
            self.lmc_config["attention_bias_stats"] = geo_bias_stats.get("attention_bias_stats")
        level_anchor_gamma = getattr(getattr(self, "compressor", None), "level_anchor_residual_gamma", None)
        if level_anchor_gamma is not None:
            self.lmc_config["final_lmc_level_anchor_residual_gamma"] = float(
                level_anchor_gamma.detach().float().cpu().item()
            )
        pos_gate = getattr(getattr(getattr(self, "compressor", None), "pe_encoder", None), "residual_gate", None)
        if pos_gate is not None:
            self.lmc_config["final_pos_fourier_residual_gate"] = float(pos_gate.detach().float().cpu().item())
        if self._is_glace_backend():
            self.lmc_config["final_local_residual_alpha"] = self._current_local_residual_alpha_value()
        meta = {key: self._tensor_to_config_value(self.lmc_config.get(key)) for key in keys}
        meta["lmc_flow"] = str(getattr(self.options, "lmc_flow", "iterative"))
        meta["lmc_auto_mode_by_visibility"] = bool(getattr(self.options, "lmc_auto_mode_by_visibility", False))
        meta["ace_g_fusion_in_s2"] = bool(getattr(self.options, "ace_g_fusion_in_s2", False))
        return meta

    @staticmethod
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

    def _should_log_runtime_stats(self):
        if not bool(getattr(self, "lmc_log_runtime_stats", False)):
            return False
        self._lmc_runtime_stats_calls += 1
        return self._lmc_runtime_stats_calls == 1 or (
            self._lmc_runtime_stats_calls % self.lmc_runtime_stats_interval == 0
        )

    def _log_fusion_runtime_stats(self, stage_tag: str, stats: Dict[str, Any]):
        if not stats:
            return
        top5 = stats.get("token_usage_top5", [])
        top5_str = ",".join(f"{float(v):.4f}" for v in top5)
        extra_str = self._format_lmc_runtime_extra_stats(stats)
        _logger.info(
            "[LMC-Runtime][%s] call=%d entropy_mean=%.4f p10=%.4f p50=%.4f p90=%.4f "
            "effective_tokens=%.2f avg_max=%.4f usage_min=%.5f usage_max=%.5f top5=[%s] "
            "raw_norm=%.4f attn_out_norm=%.4f fused_norm=%.4f queries=%d tokens=%d "
            "fusion_mode=%s scene_scale=%.6f key_geo_scale=%.6f p_norm_std=%.4f "
            "p_norm_absmax=%.4f p_norm_finite=%s extra={%s}",
            stage_tag,
            int(getattr(self, "_lmc_runtime_stats_calls", 0)),
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

    def _log_compressor_runtime_stats(self, compressor_out, stage_tag: str):
        if not bool(getattr(self, "lmc_log_runtime_stats", False)):
            return
        with torch.no_grad():
            if isinstance(compressor_out, dict):
                latent_p = compressor_out.get("p_coarse", compressor_out.get("p_fine"))
            else:
                _, latent_p = compressor_out
            if latent_p is None:
                return
            latent_p = latent_p.detach().float()
            scene_center = self.memory_dict["scene_center"].to(device=latent_p.device, dtype=latent_p.dtype)
            if scene_center.ndim == 1:
                scene_center = scene_center.unsqueeze(0)
            if scene_center.shape[0] == 1 and latent_p.shape[0] > 1:
                scene_center = scene_center.expand(latent_p.shape[0], -1)
            centered = latent_p - scene_center.unsqueeze(1)
            radius = torch.linalg.norm(centered, dim=-1)
            scene_scale = float(self.lmc_config.get("lmc_compressor_pe_scene_scale", 1.0) or 1.0)
            centered_scaled = centered / max(scene_scale, 1e-6)
            pooled_points = self.memory_dict["pooled_points"].to(device=latent_p.device, dtype=latent_p.dtype)
            dist_log = torch.log(torch.cdist(latent_p, pooled_points, p=2).pow(2) + 1e-6)
            key_mix_logits = getattr(self.compressor, "key_mix_logits", None)
            key_mix = "n/a"
            if key_mix_logits is not None:
                key_mix_weights = torch.softmax(key_mix_logits.detach().float().cpu(), dim=0)
                key_mix = "[" + ",".join(f"{float(w):.3f}" for w in key_mix_weights) + "]"
                self.lmc_config["lmc_key_mix_weights"] = key_mix_weights.tolist()
            level_stats = getattr(self.compressor, "last_levelwise_runtime_stats", None)
            if isinstance(level_stats, dict):
                weights = level_stats.get("final_level_merge_weights") or []
                weights_str = "[" + ",".join(f"{float(w):.3f}" for w in weights) + "]"
                self.lmc_config["lmc_level_merge_weights"] = level_stats.get("lmc_level_merge_weights")
                self.lmc_config["lmc_level_gate_entropy"] = level_stats.get("lmc_level_gate_entropy")
                self.lmc_config["final_lmc_level_anchor_residual_gamma"] = level_stats.get(
                    "lmc_level_anchor_residual_gamma"
                )
                _logger.info(
                    "[LMC-Runtime][%s] level_merge_weights=%s level_gate_entropy=%.4f "
                    "anchor_gamma=%s anchor_norms(base/res/out)=%s/%s/%s "
                    "per_level_latent_norm_mean=%s per_level_latent_norm_std=%s "
                    "per_level_attention_entropy_mean=%s per_level_effective_memory_token_count=%s",
                    stage_tag,
                    weights_str,
                    float(level_stats.get("final_level_gate_entropy", 0.0)),
                    str(level_stats.get("lmc_level_anchor_residual_gamma")),
                    str(level_stats.get("level_anchor_base_norm")),
                    str(level_stats.get("level_anchor_residual_norm")),
                    str(level_stats.get("level_anchor_output_norm")),
                    level_stats.get("per_level_latent_norm_mean", []),
                    level_stats.get("per_level_latent_norm_std", []),
                    level_stats.get("per_level_attention_entropy_mean", []),
                    level_stats.get("per_level_effective_memory_token_count", []),
                )
            geo_bias_stats = getattr(self.compressor, "last_geo_bias_runtime_stats", None)
            if isinstance(geo_bias_stats, dict):
                self.lmc_config["final_geo_bias_rbf_alpha"] = geo_bias_stats.get("final_geo_bias_rbf_alpha")
                self.lmc_config["final_geo_bias_rbf_weights"] = geo_bias_stats.get("final_geo_bias_rbf_weights")
                _logger.info(
                    "[LMC-Runtime][%s] geo_bias_mode=%s rbf_alpha=%s rbf_weights=%s attention_bias_stats=%s",
                    stage_tag,
                    str(self.lmc_config.get("geo_bias_mode", "legacy")),
                    str(geo_bias_stats.get("final_geo_bias_rbf_alpha")),
                    str(geo_bias_stats.get("final_geo_bias_rbf_weights")),
                    str(geo_bias_stats.get("attention_bias_stats")),
                )
            pos_gate = getattr(getattr(self.compressor, "pe_encoder", None), "residual_gate", None)
            if pos_gate is not None:
                self.lmc_config["final_pos_fourier_residual_gate"] = float(pos_gate.detach().float().cpu().item())
            _logger.info(
                "[LMC-Runtime][%s] compressor K_eff=%d key_mode=%s key_slice=%s key_layer=%s "
                "key_mix=%s pe_scale_mode=%s scene_scale=%.6f latent_radius_mean=%.4f latent_radius_std=%.4f "
                "coord_std=%.4f coord_scaled_std=%.4f coord_scaled_absmax=%.4f "
                "dist_log_min=%.4f dist_log_max=%.4f dist_log_std=%.4f",
                stage_tag,
                int(latent_p.shape[1]),
                str(self.lmc_config.get("lmc_key_feature_mode", "slice")),
                str(self.lmc_config.get("lmc_key_slice_idx")),
                str(self.lmc_config.get("lmc_key_layer_label")),
                key_mix,
                str(self.lmc_config.get("lmc_compressor_pe_scale_mode", "raw")),
                scene_scale,
                float(radius.mean().item()),
                float(radius.std(unbiased=False).item()),
                float(centered.std(unbiased=False).item()),
                float(centered_scaled.std(unbiased=False).item()),
                float(centered_scaled.abs().max().item()),
                float(dist_log.min().item()),
                float(dist_log.max().item()),
                float(dist_log.std(unbiased=False).item()),
            )

    # ------------------------------------------------------------------
    # Fuse features with memory
    # ------------------------------------------------------------------

    def _fuse_features(self, features_BCHW, compressor_out, stage_tag="Fusion"):
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
        collect_stats = self._should_log_runtime_stats()
        fused = self.fusion(
            query,
            compressor_out,
            scene_center,
            return_stats=collect_stats,
            stats_max_pixels=self.lmc_runtime_stats_max_pixels,
        )
        if collect_stats:
            fused, stats = fused
            self._log_fusion_runtime_stats(stage_tag, stats)
        return fused.reshape(B, H, W, C).permute(0, 3, 1, 2)

    def _use_glace_local_fusion(self):
        return bool(self._is_glace_backend() and getattr(self, 'effective_lmc_fusion_target', 'decoder') == 'local')

    def _fuse_lmc_features_for_head(self, decoder_features_BCHW, compressor_out, *, stage_tag):
        """Return head input plus optional base decoder features for the legacy residual adapter."""
        if not self._is_glace_backend():
            return self._fuse_features(decoder_features_BCHW, compressor_out, stage_tag=stage_tag), None

        target = getattr(self, 'effective_lmc_fusion_target', 'decoder')
        if target == 'decoder':
            fused_decoder_BCHW = self._fuse_features(decoder_features_BCHW, compressor_out, stage_tag=stage_tag)
            return fused_decoder_BCHW, decoder_features_BCHW
        if target != 'local':
            raise ValueError(f"Unsupported effective_lmc_fusion_target={target!r}")

        global_BC, local_BCHW = self._split_glace_decoder_feature_maps(
            decoder_features_BCHW,
            stage_tag=stage_tag,
        )
        fused_local_BCHW = self._fuse_features(local_BCHW, compressor_out, stage_tag=f"{stage_tag}-Local")
        local_residual_mode = getattr(self, 'local_residual_mode', 'none')
        local_residual_alpha_eff = self._current_local_residual_alpha_tensor(
            device=local_BCHW.device,
            dtype=local_BCHW.dtype,
        )
        if local_residual_mode == 'none':
            local_out_BCHW = fused_local_BCHW
        elif local_residual_mode in ('fixed_alpha', 'learned_alpha'):
            local_out_BCHW = local_BCHW + local_residual_alpha_eff * (fused_local_BCHW - local_BCHW)
        else:
            raise ValueError(f"Unsupported local_residual_mode={local_residual_mode!r}")

        head_input_BCHW = self._build_glace_head_input(global_BC, local_out_BCHW)
        if tuple(head_input_BCHW.shape) != tuple(decoder_features_BCHW.shape):
            raise ValueError(
                f"[{stage_tag}] GLACE local fusion changed head input shape: "
                f"base={tuple(decoder_features_BCHW.shape)} head={tuple(head_input_BCHW.shape)}"
            )
        if not getattr(self, '_logged_glace_local_fusion_dims', False):
            _logger.info(
                "[GLACE-LMC] local-only fusion enabled: global_dim=%d local_dim=%d head_dim=%d local_residual=%s alpha_eff=%.6f alpha_target=%.6f alpha_max=%.6f warmup_steps=%d",
                int(global_BC.shape[1]),
                int(local_BCHW.shape[1]),
                int(head_input_BCHW.shape[1]),
                local_residual_mode,
                float(local_residual_alpha_eff.detach().float().cpu().item()),
                float(getattr(self, 'local_residual_alpha', 1.0)),
                float(getattr(self, 'local_residual_alpha_max', getattr(self, 'local_residual_alpha', 1.0))),
                int(getattr(self, 'local_residual_alpha_warmup_steps', 0)),
            )
            self._logged_glace_local_fusion_dims = True
        return head_input_BCHW, None

    def _apply_lmc_fusion_and_residual_for_head(self, decoder_features_BCHW, compressor_out, *, stage_tag):
        fused_BCHW, base_for_residual_BCHW = self._fuse_lmc_features_for_head(
            decoder_features_BCHW,
            compressor_out,
            stage_tag=stage_tag,
        )
        if base_for_residual_BCHW is None:
            return fused_BCHW
        return self._mix_glace_decoder_feature_maps(
            base_for_residual_BCHW,
            fused_BCHW,
            stage_tag=stage_tag,
        )

    # ------------------------------------------------------------------
    # Override: create_training_buffer (adds fusion step)
    # ------------------------------------------------------------------

    def create_training_buffer(self, buffer_size_override=None):
        """Fill buffer. When LMC is active, fuse features with memory first.
        If buffer_on_cpu is True (default for LMC), buffer is moved to CPU after fill to avoid GPU OOM.
        """
        if not self.use_lmc:
            return super().create_training_buffer()

        # Compress memory once for this buffer fill
        compressor_out = self._compress_memory()

        # S2 buffer collection should be deterministic: disable dropout etc.
        mode_snapshot = self._capture_module_training_modes(
            self.compressor,
            self.fusion,
            self.regressor,
            getattr(self.regressor, "encoder", None),
            getattr(self.regressor, "heads", None),
        )
        self.compressor.eval()
        self.fusion.eval()

        # Temporarily override buffer size if requested (for final iteration)
        orig_buf_size = self.options.training_buffer_size
        target_buf_size = int(buffer_size_override if buffer_size_override is not None else orig_buf_size)
        force_final_cpu = (
            buffer_size_override is not None
            and int(buffer_size_override) == int(self.buffer_size_final)
            and bool(getattr(self.options, "buffer_on_cpu_final", True))
        )
        if buffer_size_override is not None:
            self.options.training_buffer_size = buffer_size_override

        # We need to intercept the feature extraction to add fusion.
        # Strategy: monkey-patch regressor.get_features temporarily.
        original_get_features = self.regressor.get_features

        def fused_get_features(images):
            raw_feats = original_get_features(images)
            if self._use_glace_local_fusion():
                # _create_training_buffer_with_scene_coords appends the bypassed global features.
                return self._fuse_features(raw_feats, compressor_out, stage_tag="S2-Buffer-Local")
            return self._fuse_features(raw_feats, compressor_out, stage_tag="S2-Buffer")

        self.regressor.get_features = fused_get_features
        orig_force_current_buffer_on_cpu = getattr(self, "_force_current_buffer_on_cpu", False)
        self._force_current_buffer_on_cpu = force_final_cpu
        try:
            self._create_training_buffer_with_scene_coords()
        finally:
            self._force_current_buffer_on_cpu = orig_force_current_buffer_on_cpu
            self.regressor.get_features = original_get_features
            self.options.training_buffer_size = orig_buf_size
            self._restore_module_training_modes(mode_snapshot)

        # LMC: keep buffer on CPU to avoid GPU OOM (map-anything style: only batch on GPU)
        buffer_on_cpu = bool(getattr(self, "_current_training_buffer_on_cpu", self._current_buffer_should_be_on_cpu()))
        if buffer_on_cpu and self.training_buffer is not None:
            for k in list(self.training_buffer.keys()):
                self.training_buffer[k] = self.training_buffer[k].cpu()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            _logger.info("[LMC] Buffer kept on CPU to save GPU memory.")
        if self.training_buffer is not None:
            self._validate_training_buffer_schema(
                buffer_dict=self.training_buffer,
                schema_name=self._buffer_schema_name("fused_buffer"),
                expected_size=target_buf_size,
            )
        # Normalize poses for full-pipeline normalization
        self._normalize_buffer_poses()

    # ------------------------------------------------------------------
    # Override: create_training_buffer_ace_g (raw backbone features, no fusion)
    # ------------------------------------------------------------------

    def create_training_buffer_ace_g(self, buffer_size_override=None):
        """Fill the ACE-G S2 buffer.

        Default ACE-G stores raw backbone features and applies fusion per batch.
        stage1_fused stores frozen Stage1 fused features, treating Stage1 as an
        enhanced backbone for GLACE-style head training.
        """
        orig_buf_size = self.options.training_buffer_size
        target_buf_size = int(buffer_size_override if buffer_size_override is not None else orig_buf_size)
        force_final_cpu = (
            buffer_size_override is not None
            and int(buffer_size_override) == int(self.buffer_size_final)
            and bool(getattr(self.options, "buffer_on_cpu_final", True))
        )
        if buffer_size_override is not None:
            self.options.training_buffer_size = buffer_size_override

        orig_force_current_buffer_on_cpu = getattr(self, "_force_current_buffer_on_cpu", False)
        self._force_current_buffer_on_cpu = force_final_cpu
        mode_snapshot = self._capture_module_training_modes(
            self.regressor,
            getattr(self.regressor, "encoder", None),
            getattr(self.regressor, "heads", None),
            self.compressor,
            self.fusion,
        )
        original_get_features = self.regressor.get_features
        use_fused_buffer = self._ace_lmc_stage2_uses_fused_buffer()
        try:
            # Backbone/fusion should be in eval mode for deterministic feature extraction.
            self.regressor.eval()
            self.compressor.eval()
            self.fusion.eval()
            if use_fused_buffer:
                compressor_out = getattr(self, '_s2_compressor_out', None)
                if compressor_out is None:
                    compressor_out = self._compress_memory()

                def fused_get_features(images):
                    raw_feats = original_get_features(images)
                    fused_feats, _ = self._fuse_lmc_features_for_head(
                        raw_feats,
                        compressor_out,
                        stage_tag='S2-G-Buffer-FusedBackbone',
                    )
                    return fused_feats

                self.regressor.get_features = fused_get_features
                _logger.info(
                    "[ACE-G] Buffer feature_source=stage1_fused: storing frozen Stage1 fused features; fusion eval=%s trainable=%d.",
                    not self.fusion.training,
                    self._count_trainable_params(self.fusion),
                )
            else:
                _logger.info("[ACE-G] Buffer feature_source=raw_backbone: storing raw backbone features.")
            self._validate_ace_lmc_stage2_fused_backbone_contract("S2-G-Buffer", check_optimizer=False)
            self._create_training_buffer_with_scene_coords()
        finally:
            self.regressor.get_features = original_get_features
            self._force_current_buffer_on_cpu = orig_force_current_buffer_on_cpu
            self.options.training_buffer_size = orig_buf_size
            self._restore_module_training_modes(mode_snapshot)

        # Move buffer to CPU to save GPU memory (same as LMC iterative path).
        buffer_on_cpu = bool(getattr(self, "_current_training_buffer_on_cpu", self._current_buffer_should_be_on_cpu()))
        if buffer_on_cpu and self.training_buffer is not None:
            for k in list(self.training_buffer.keys()):
                self.training_buffer[k] = self.training_buffer[k].cpu()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            _logger.info("[ACE-G] Buffer kept on CPU (feature_source=%s).", self._ace_lmc_stage2_feature_source())
        if self.training_buffer is not None:
            self._validate_training_buffer_schema(
                buffer_dict=self.training_buffer,
                schema_name=self._buffer_schema_name("raw_buffer"),
                expected_size=target_buf_size,
            )
        # Normalize poses for full-pipeline normalization
        self._normalize_buffer_poses()

    # ------------------------------------------------------------------
    # Full-pipeline normalization: normalize buffer poses
    # ------------------------------------------------------------------

    def _normalize_buffer_poses(self):
        """Normalize gt_poses_inv in training_buffer for normalized scene coordinates.

        When memory uses normalized coords (raw-mu)/sigma, the inverse pose
        translation must also be normalized so that reprojection math stays
        consistent:
            inv_pose_norm @ x_norm = R^T @ (x_norm - C_norm) = R^T @ (x_world - C) / sigma
        Since pixel projection is px = cam_x/cam_z, sigma cancels, so pixel errors are identical.

        Formula: t_norm = (t + R @ mu) / sigma  where t = original translation = -R^T @ C
        """
        if not self.use_lmc or self.coord_sigma is None:
            return  # No normalization active, or not LMC mode

        if self.training_buffer is None or 'gt_poses_inv' not in self.training_buffer:
            return

        buf = self.training_buffer
        gt_poses = buf['gt_poses_inv']  # (B, 3, 4)
        mu = self.coord_mu.to(gt_poses.device)  # [3]
        sigma = self.coord_sigma

        R = gt_poses[:, :, :3]   # (B, 3, 3) — unchanged
        t = gt_poses[:, :, 3]    # (B, 3) — needs normalization

        # t_norm = (t + R @ mu) / sigma
        # R: (B, 3, 3), mu: (3,) → mu_broadcast: (B, 3, 1) → R @ mu: (B, 3, 1) → squeeze: (B, 3)
        R_mu = torch.bmm(R, mu.view(1, 3, 1).expand(R.shape[0], -1, -1)).squeeze(-1)  # (B, 3)
        t_norm = (t + R_mu) / sigma

        gt_poses_norm = torch.cat([R, t_norm.unsqueeze(-1)], dim=-1)
        buf['gt_poses_inv'] = gt_poses_norm
        _logger.info(
            "[LMC] Buffer poses normalized: sigma=%.4f, translation_before=(%.3f,%.3f,%.3f), "
            "translation_after=(%.3f,%.3f,%.3f) [sample 0]",
            sigma,
            float(t[0, 0]), float(t[0, 1]), float(t[0, 2]),
            float(t_norm[0, 0]), float(t_norm[0, 1]), float(t_norm[0, 2]),
        )

    # ------------------------------------------------------------------
    # S1: Train compressor for a few steps
    # ------------------------------------------------------------------

    # Sanity thresholds for full_map (align with map-anything tasks/ace/loss_utils.py)
    SANITY_PIXEL_ERR = 50000.0
    SANITY_COORD_VAL = 10000.0

    def _s1_compute_loss_full_map(
        self,
        fused_feats_BCHW,
        gt_pose_inv_B34,
        K_B33,
        invK_B33,
        image_mask_B1HW,
        gt_scene_coords_B3HW=None,
        gt_scene_coords_valid_B1HW=None,
        img_idx_B=None,
        s1_step=0,
        base_decoder_feats_BCHW=None,
    ):
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

        head_input_BCHW = self._mix_glace_decoder_feature_maps(
            base_decoder_feats_BCHW,
            fused_feats_BCHW,
            stage_tag='S1-FullMap',
        )
        head_input_BCHW = self._append_ace_lmc_global_to_feature_maps(
            head_input_BCHW,
            img_idx_B,
            stage_tag='S1-FullMap',
        )
        pred_scene_B3HW = self.regressor.get_scene_coordinates(head_input_BCHW)
        pred_scene_B3HW = self._recover_pred_scene_to_training_world(pred_scene_B3HW)
        pred_scene_N31 = pred_scene_B3HW.permute(0, 2, 3, 1).reshape(N, 3).unsqueeze(-1).float()

        gt_pose_inv_N34 = gt_pose_inv_B34.unsqueeze(1).unsqueeze(1).expand(B, H, W, 3, 4).reshape(N, 3, 4)
        K_N33 = K_B33.unsqueeze(1).unsqueeze(1).expand(B, H, W, 3, 3).reshape(N, 3, 3)
        invK_N33 = invK_B33.unsqueeze(1).unsqueeze(1).expand(B, H, W, 3, 3).reshape(N, 3, 3)

        pixel_grid_B2HW = self.pixel_grid_2HW[:, :H, :W].clone().unsqueeze(0).expand(B, 2, H, W)
        target_px_N2 = pixel_grid_B2HW.permute(0, 2, 3, 1).reshape(N, 2)

        loss_step = self._monotonic_repro_step() if self.mapany_flow_profile else int(s1_step)
        contract = self._compute_reprojection_invalid_loss_contract(
            pred_scene_N31,
            target_px_N2,
            gt_pose_inv_N34,
            K_N33,
            invK_N33,
            step_eff=loss_step,
            normalizer=N,
            include_mask_N=mask_flat,
            apply_nuclear_mask=True,
            valid_subsample_max=int(getattr(self.options, "s1_full_map_max_points", 0)),
            invalid_max_delta=None,
            invalid_posinf=1e4,
            invalid_neginf=1e4,
        )
        loss = contract["loss"]
        if gt_scene_coords_B3HW is not None:
            if tuple(gt_scene_coords_B3HW.shape[-2:]) != (H, W):
                gt_scene_coords_valid_B1HW = self._scene_coords_valid_mask(
                    gt_scene_coords_B3HW,
                    size_hw=(H, W),
                    device=fused_feats_BCHW.device,
                )
                gt_scene_coords_B3HW = Fnn.interpolate(gt_scene_coords_B3HW, size=(H, W), mode="nearest")
            elif gt_scene_coords_valid_B1HW is None:
                gt_scene_coords_valid_B1HW = self._scene_coords_valid_mask(
                    gt_scene_coords_B3HW,
                    size_hw=(H, W),
                    device=fused_feats_BCHW.device,
                )
            gt_scene_coords_N3 = gt_scene_coords_B3HW.permute(0, 2, 3, 1).reshape(N, 3)
            gt_scene_coords_valid_N1 = None
            if gt_scene_coords_valid_B1HW is not None:
                gt_scene_coords_valid_N1 = gt_scene_coords_valid_B1HW.permute(0, 2, 3, 1).reshape(N, 1)
            loss = loss + self._compute_c1_aux_ref_loss(
                pred_scene_B3HW,
                gt_scene_coords_N3,
                gt_scene_coords_valid_N1,
            )
        stats = contract["stats"]
        return loss, stats

    def _s1_compute_loss_from_features(
        self,
        features_bC,
        target_px_b2,
        gt_inv_poses_b34,
        Ks_b33,
        invKs_b33,
        gt_scene_coords_world_b3=None,
        gt_scene_coords_valid_b1=None,
        img_idx_b1=None,
        s1_step=0,
        base_decoder_features_bC=None,
    ):
        """Compute ace_depth reprojection loss from sampled fused features (same formula as TrainerACEDINOv2.training_step)."""
        channels = features_bC.shape[1]

        # Keep same head input reshaping logic as ace_depth training_step.
        features_bC, batch_size, h, w, trimmed = self._pack_feature_rows_for_head(
            "S1",
            features_bC,
            target_px_b2,
            gt_inv_poses_b34,
            Ks_b33,
            invKs_b33,
            gt_scene_coords_world_b3,
            gt_scene_coords_valid_b1,
        )
        if batch_size is None:
            return None, None
        target_px_b2, gt_inv_poses_b34, Ks_b33, invKs_b33, gt_scene_coords_world_b3, gt_scene_coords_valid_b1 = trimmed
        img_idx_b1 = img_idx_b1[:batch_size] if img_idx_b1 is not None else None
        if base_decoder_features_bC is not None:
            base_decoder_features_bC = base_decoder_features_bC[: batch_size * h * w]
        head_features_bC = self._mix_glace_decoder_features(base_decoder_features_bC, features_bC, stage_tag="S1")
        head_features_bC = self._append_ace_lmc_global_to_features(head_features_bC, img_idx_b1, stage_tag="S1")

        features_bCHW = head_features_bC.view(1, h, w, head_features_bC.shape[1]).permute(0, 3, 1, 2)
        pred_scene_coords_b3HW = self.regressor.get_scene_coordinates(features_bCHW)
        pred_scene_coords_b3HW = self._recover_pred_scene_to_training_world(pred_scene_coords_b3HW)

        pred_scene_coords_b31 = pred_scene_coords_b3HW.permute(0, 2, 3, 1).flatten(0, 2).unsqueeze(-1).float()
        iter_for_loss = self._resolve_s1_sampled_loss_step(s1_step)
        contract = self._compute_reprojection_invalid_loss_contract(
            pred_scene_coords_b31,
            target_px_b2,
            gt_inv_poses_b34,
            Ks_b33,
            invKs_b33,
            step_eff=iter_for_loss,
            normalizer=batch_size,
            invalid_max_delta=None,
            invalid_posinf=1e4,
            invalid_neginf=1e4,
        )
        loss = contract["loss"]
        loss = loss + self._compute_c1_aux_ref_loss(
            pred_scene_coords_b3HW,
            gt_scene_coords_world_b3,
            gt_scene_coords_valid_b1,
        )
        stats = contract["stats"]
        return loss, stats

    def _build_s1_dataloader(self):
        """Build S1 dataloader with fixed-size images for batched forward passes."""
        # s1_batch_size: default 8 to avoid OOM (full_map and sample_per_image both use encoder on full batch;
        # DINOv2 attention ~(B*N)^2 causes fragmentation OOM after many steps on 24GB GPUs).
        effective_batch = max(1, getattr(self.options, 's1_batch_size', 8))
        if self._is_ace_fcn_backend() and effective_batch != 1:
            _logger.info('[ACE-FCN-LMC] Forcing s1_batch_size=1 because the ACE dataset keeps variable image widths.')
            effective_batch = 1
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
            group_max_lrs = [float(pg.get('lr', base_lr)) for pg in optimizer_s1.param_groups]
            return optim.lr_scheduler.OneCycleLR(
                optimizer_s1, max_lr=group_max_lrs, total_steps=target_steps,
                pct_start=pct, div_factor=div, final_div_factor=final_div, anneal_strategy='cos',
            )
        # onecycle_legacy
        pct = getattr(self.options, 'lmc_lr_pct_start', 0.4)
        group_max_lrs = [float(pg.get('lr', base_lr)) for pg in optimizer_s1.param_groups]
        return optim.lr_scheduler.OneCycleLR(
            optimizer_s1, max_lr=group_max_lrs, total_steps=target_steps, pct_start=pct,
        )

    def _prepare_s1_buffer_for_iteration(self, iteration_idx, buffer_size_override=None):
        """Build S1 raw-feature buffer once per iteration."""
        total_size = int(buffer_size_override if buffer_size_override is not None else self.options.training_buffer_size)
        mode = self.s1_buffer_refill_mode

        if mode == 'full':
            _logger.info(
                "[S1-Buffer] Iteration %d: rebuilding raw feature buffer (mode=%s, size=%d).",
                iteration_idx + 1,
                mode,
                total_size,
            )
            t0 = time.time()
            self.create_training_buffer_ace_g(buffer_size_override=buffer_size_override)
            elapsed = time.time() - t0
            buf_len = int(self.training_buffer['features'].shape[0]) if self.training_buffer is not None else 0
            _logger.info(
                "[S1-Buffer] Ready: %d samples built in %.1fs.",
                buf_len,
                elapsed,
            )
            return

        if mode != 'partial':
            raise NotImplementedError(
                f"S1 buffer refill mode {mode!r} is not implemented yet."
            )

        if self.training_buffer is None or iteration_idx == 0:
            _logger.info(
                "[S1-Buffer] Iteration %d: partial refill requested but no previous buffer is available; fallback to full rebuild.",
                iteration_idx + 1,
            )
            t0 = time.time()
            self.create_training_buffer_ace_g(buffer_size_override=buffer_size_override)
            elapsed = time.time() - t0
            buf_len = int(self.training_buffer['features'].shape[0]) if self.training_buffer is not None else 0
            _logger.info(
                "[S1-Buffer] Ready: %d samples built in %.1fs (fallback_full).",
                buf_len,
                elapsed,
            )
            return

        previous_buffer = self.training_buffer
        previous_size = int(previous_buffer["features"].shape[0])
        self._validate_training_buffer_schema(
            buffer_dict=previous_buffer,
            schema_name=self._buffer_schema_name("raw_buffer"),
            expected_size=previous_size,
        )
        keep_count, refill_count = self._resolve_s1_partial_refill_counts(total_size=total_size)
        if previous_size < keep_count:
            raise ValueError(
                f"S1 partial refill needs keep_count={keep_count}, but previous buffer only has {previous_size} rows."
            )

        _logger.info(
            "[S1-Buffer] Iteration %d: partial refill keep=%d refill=%d (target=%d, prev=%d).",
            iteration_idx + 1,
            keep_count,
            refill_count,
            total_size,
            previous_size,
        )
        if not bool(getattr(self.options, "buffer_on_cpu", True)):
            _logger.warning(
                "[S1-Buffer] partial refill with buffer_on_cpu=False may transiently increase GPU memory usage."
            )
        t0 = time.time()

        prev_device = previous_buffer["features"].device
        keep_indices = torch.randperm(
            previous_size,
            generator=self._get_training_generator(prev_device),
            device=prev_device,
        )[:keep_count]
        kept_buffer = self._slice_training_buffer_rows(previous_buffer, self._buffer_schema_name("raw_buffer"), keep_indices)
        self.training_buffer = None
        del previous_buffer

        self.create_training_buffer_ace_g(buffer_size_override=refill_count)
        refill_buffer = self.training_buffer
        merged_buffer = self._merge_training_buffers(
            schema_name=self._buffer_schema_name("raw_buffer"),
            buffers=[kept_buffer, refill_buffer],
            expected_size=total_size,
        )
        self.training_buffer = merged_buffer

        elapsed = time.time() - t0
        _logger.info(
            "[S1-Buffer] Ready: %d samples built in %.1fs (partial keep=%d refill=%d).",
            int(self.training_buffer["features"].shape[0]),
            elapsed,
            keep_count,
            refill_count,
        )

    def _train_compressor_steps_from_buffer(self, iteration_idx, n_steps):
        """Stage 1 buffer mode: train compressor/fusion/head from raw feature buffer."""
        self._set_compressor_trainable(True)
        self.compressor.train()
        self.fusion.train()
        # When the base network is frozen, the head is excluded here and only
        # the plugin path is allowed to move.
        # Same freeze rule for standard S1 updates.
        # Same freeze rule carries into S2 / ACE-G head training.
        # Same freeze rule applies to the S2 polish phase.
        freeze_glace_head = self._glace_freeze_head() or self._glace_head_freeze_active(iteration_idx)
        if self._is_glace_backend():
            self._set_glace_head_trainable(not freeze_glace_head, stage_tag="S1-Buffer")
        else:
            self.regressor.heads.train()
        self.regressor.encoder.eval()
        self._log_stage_trainability("S1-Buffer")

        if self.training_buffer is None or 'features' not in self.training_buffer:
            raise RuntimeError("[S1-Buffer] training_buffer is empty; call _prepare_s1_buffer_for_iteration first.")
        buffer_len = int(self.training_buffer['features'].shape[0])
        self._validate_training_buffer_schema(
            buffer_dict=self.training_buffer,
            schema_name=self._buffer_schema_name("raw_buffer"),
            expected_size=buffer_len,
        )
        if buffer_len < 16:
            raise RuntimeError(f"[S1-Buffer] training_buffer too small: {buffer_len}.")

        s1_lr_scale_later = float(getattr(self.options, 's1_lr_scale_later', 0.4))
        base_lr_s1 = self.s1_learning_rate_max * (s1_lr_scale_later if iteration_idx > 0 else 1.0)
        if iteration_idx > 0:
            _logger.info("  [S1 LR] iter>0: base_lr scaled by %.2f -> %.2e", s1_lr_scale_later, base_lr_s1)

        s1_param_groups = [
            {'name': 'compressor', 'params': self.compressor.parameters()},
            {'name': 'fusion_residual', 'params': list(self.fusion.parameters()) + self._glace_residual_adapter_params()},
        ]
        gate_params = self._local_residual_gate_params()
        if gate_params:
            s1_param_groups.append({'name': 'local_residual_gate', 'params': gate_params, 'lr': base_lr_s1 * 0.1})
        if freeze_glace_head:
            _logger.info("  [S1-Buffer] GLACE head freeze warmup active at iter %d; head excluded from optimizer.", iteration_idx + 1)
        else:
            s1_param_groups.append({'name': 'head', 'params': self.regressor.heads.parameters(), 'lr': base_lr_s1 * 0.1})
        comp_optimizer = optim.AdamW(s1_param_groups, lr=base_lr_s1)
        self._validate_glace_head_optimizer_membership("S1-Buffer", comp_optimizer, expected_trainable=not freeze_glace_head)
        self._log_optimizer_groups("S1-Buffer", comp_optimizer)
        sched_lmc = self._build_s1_scheduler(comp_optimizer, n_steps, iteration_idx, base_lr=base_lr_s1)

        log_interval = 10
        skipped_nonfinite = 0
        skipped_sample = 0
        consecutive_unstable = 0
        max_consecutive_unstable = int(getattr(self.options, 's1_max_consecutive_unstable', 300))
        s1_lr_cap = base_lr_s1
        s1_lr_floor_ratio = float(getattr(self.options, 's1_lr_floor_ratio', 0.05))
        s1_lr_floor = base_lr_s1 * s1_lr_floor_ratio
        update_step = 0
        raw_step = 0
        max_attempts = max(n_steps * 20, n_steps + 1)
        s1_early_stop_cfg = self._get_s1_early_stop_cfg()
        ema_px_err = None
        best_ema_px_err = float('inf')
        no_improve_updates = 0
        _s1_loss_step1 = None
        _s1_loss_last = None

        _logger.info(
            "  [S1-Buffer] source=raw_buffer, target_updates=%d, s1_loss_mode=%s, s1_loss_step_mode=%s",
            n_steps,
            getattr(self.options, 's1_loss_mode', 'sample_per_image'),
            self.s1_loss_step_mode,
        )
        if s1_early_stop_cfg["enabled"]:
            _logger.info(
                "  [S1] early-stop ON (min_updates=%d, patience=%d, rel_improve=%.4f, ema_beta=%.2f)",
                s1_early_stop_cfg["min_updates"],
                s1_early_stop_cfg["patience"],
                s1_early_stop_cfg["rel_improve"],
                s1_early_stop_cfg["ema_beta"],
            )

        buf = self.training_buffer
        buf_device = buf['features'].device
        while update_step < n_steps:
            raw_step += 1
            if raw_step > max_attempts:
                _logger.warning(
                    "  [S1-Buffer] reached max attempts (%d) before target updates (%d). updates=%d",
                    max_attempts, n_steps, update_step
                )
                break

            draw_bs = min(self.options.batch_size, buffer_len)
            if draw_bs < 16:
                skipped_sample += 1
                continue
            sample_idxs = torch.randint(
                0,
                buffer_len,
                (draw_bs,),
                generator=self._get_training_generator(buf_device),
                device=buf_device,
            )

            def _to_dev(t):
                out = t[sample_idxs].contiguous()
                if out.device != self.device:
                    out = out.to(self.device, non_blocking=True)
                return out

            raw_features_bC = _to_dev(buf['features'])
            target_px_b2 = _to_dev(buf['target_px'])
            gt_inv_poses_b34 = _to_dev(buf['gt_poses_inv'])
            Ks_b33 = _to_dev(buf['intrinsics'])
            invKs_b33 = _to_dev(buf['intrinsics_inv'])
            img_idx_b1 = _to_dev(buf['img_idx']) if 'img_idx' in buf else None

            channels = raw_features_bC.shape[1]
            raw_features_bC, batch_size, h, w, trimmed = self._pack_feature_rows_for_head(
                "S1-Buffer",
                raw_features_bC,
                target_px_b2,
                gt_inv_poses_b34,
                Ks_b33,
                invKs_b33,
            )
            if batch_size is None:
                skipped_sample += 1
                continue
            target_px_b2, gt_inv_poses_b34, Ks_b33, invKs_b33 = trimmed
            img_idx_b1 = img_idx_b1[:batch_size] if img_idx_b1 is not None else None

            with autocast("cuda", enabled=self.options.use_half):
                comp_out = self.compressor(self.memory_dict)
                raw_features_bCHW = raw_features_bC.view(1, h, w, channels).permute(0, 3, 1, 2)
                fused_bCHW, base_for_residual_bCHW = self._fuse_lmc_features_for_head(
                    raw_features_bCHW,
                    comp_out,
                    stage_tag="S1-Buffer",
                )
            fused_features_bC = fused_bCHW.permute(0, 2, 3, 1).reshape(-1, fused_bCHW.shape[1])
            base_decoder_features_bC = None
            if base_for_residual_bCHW is not None:
                base_decoder_features_bC = base_for_residual_bCHW.permute(0, 2, 3, 1).reshape(-1, base_for_residual_bCHW.shape[1])

            loss, s1_stats = self._s1_compute_loss_from_features(
                fused_features_bC.contiguous(),
                target_px_b2.contiguous(),
                gt_inv_poses_b34.contiguous(),
                Ks_b33.contiguous(),
                invKs_b33.contiguous(),
                img_idx_b1=img_idx_b1.contiguous() if img_idx_b1 is not None else None,
                s1_step=update_step,
                base_decoder_features_bC=base_decoder_features_bC.contiguous() if base_decoder_features_bC is not None else None,
            )

            if loss is None:
                continue

            if s1_stats["nonfinite_ratio"] > 0.10:
                skipped_nonfinite += 1
                consecutive_unstable += 1
                s1_lr_cap = max(s1_lr_floor, s1_lr_cap * 0.7)
                for pg in comp_optimizer.param_groups:
                    pg["lr"] = min(pg["lr"], s1_lr_cap)
                # Skip this noisy update entirely: no optimizer/scheduler step, no update counter increment.
                comp_optimizer.zero_grad(set_to_none=True)
                if consecutive_unstable >= max_consecutive_unstable:
                    _logger.warning(
                        "  [S1-Buffer] stopping early: %d consecutive unstable batches (nonFinite=%.2f%%). updates=%d/%d.",
                        consecutive_unstable, s1_stats["nonfinite_ratio"] * 100.0, update_step, n_steps
                    )
                    break
                continue

            if not torch.isfinite(loss):
                skipped_nonfinite += 1
                consecutive_unstable += 1
                s1_lr_cap = max(s1_lr_floor, s1_lr_cap * 0.7)
                for pg in comp_optimizer.param_groups:
                    pg["lr"] = min(pg["lr"], s1_lr_cap)
                # Keep update-count semantics strict: non-finite loss must not advance optimizer/scheduler/step.
                comp_optimizer.zero_grad(set_to_none=True)
                if consecutive_unstable >= max_consecutive_unstable:
                    _logger.warning(
                        "  [S1-Buffer] stopping early: %d consecutive unstable/non-finite. updates=%d/%d.",
                        consecutive_unstable, update_step, n_steps
                    )
                    break
                continue

            consecutive_unstable = 0
            comp_optimizer.zero_grad(set_to_none=True)
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(comp_optimizer)
            torch.nn.utils.clip_grad_norm_(
                list(self.compressor.parameters()) +
                list(self.fusion.parameters()) +
                self._glace_residual_adapter_params() +
                list(self.regressor.heads.parameters()),
                max_norm=1.0
            )
            self.scaler.step(comp_optimizer)
            self.scaler.update()
            sched_lmc.step()
            for pg in comp_optimizer.param_groups:
                pg["lr"] = min(pg["lr"], s1_lr_cap)

            update_step += 1
            self.iteration += 1
            px_err = s1_stats["pxerr_l2"] if math.isfinite(s1_stats["pxerr_l2"]) else s1_stats["pxerr_l1"]
            if not math.isfinite(px_err):
                px_err = 0.0

            if s1_early_stop_cfg["enabled"]:
                if ema_px_err is None:
                    ema_px_err = px_err
                else:
                    ema_px_err = (
                        s1_early_stop_cfg["ema_beta"] * ema_px_err
                        + (1.0 - s1_early_stop_cfg["ema_beta"]) * px_err
                    )
                if update_step >= s1_early_stop_cfg["min_updates"]:
                    if ema_px_err < best_ema_px_err * (1.0 - s1_early_stop_cfg["rel_improve"]):
                        best_ema_px_err = ema_px_err
                        no_improve_updates = 0
                    else:
                        no_improve_updates += 1
                    if no_improve_updates >= s1_early_stop_cfg["patience"]:
                        _logger.info(
                            "  [S1-Buffer] early-stop at update %d/%d (ema_pxErr=%.2f, best=%.2f, patience=%d)",
                            update_step, n_steps, ema_px_err, best_ema_px_err, s1_early_stop_cfg["patience"]
                        )
                        break

            if update_step % log_interval == 0 or update_step == 1 or update_step == n_steps:
                _cur_loss = float(loss.item())
                if update_step == 1 and math.isfinite(_cur_loss):
                    _s1_loss_step1 = _cur_loss
                if math.isfinite(_cur_loss):
                    _s1_loss_last = _cur_loss
                self._append_step_log(
                    iter_idx=iteration_idx,
                    step=self.iteration,
                    stage="S1-BUF",
                    loss=_cur_loss,
                    px_err=float(px_err),
                    lr=float(comp_optimizer.param_groups[0]["lr"]),
                    mode="reproj",
                    med3d=-1.0,
                )
                _logger.info(
                    "  [S1-Buffer] update %d/%d (attempt=%d), samples=%d, loss=%.4f, valid=%.1f%%, pxErrL1=%.2f, pxErrL2=%.2f, nonFinite=%.2f%%, lr=%.2e",
                    update_step, n_steps, raw_step, batch_size, _cur_loss,
                    s1_stats["fraction_valid"] * 100.0, s1_stats["pxerr_l1"], s1_stats["pxerr_l2"],
                    s1_stats["nonfinite_ratio"] * 100.0, comp_optimizer.param_groups[0]["lr"],
                )

        if skipped_sample > 0 or skipped_nonfinite > 0:
            _logger.warning(
                "  [S1-Buffer] skipped steps: bad_sample=%d, nonfinite=%d (target=%d)",
                skipped_sample, skipped_nonfinite, n_steps
            )

        if (
            _s1_loss_step1 is not None
            and _s1_loss_last is not None
            and _s1_loss_step1 > 0
            and _s1_loss_last > _s1_loss_step1 * 0.9
        ):
            _logger.warning(
                "  [S1-Buffer] CONVERGENCE WARNING: loss did not decrease significantly "
                "(step1=%.4f → final=%.4f, ratio=%.3f). "
                "Possible compressor-fusion desync building up. "
                "If this persists across iterations, late-stage S2 may diverge.",
                _s1_loss_step1, _s1_loss_last, _s1_loss_last / _s1_loss_step1,
            )

    def _train_compressor_steps(self, iteration_idx, n_steps):
        """Stage 1: full-supervised reprojection training (compressor + fusion + head), aligned with map-anything logic."""
        if self.s1_use_buffer:
            s1_loss_mode = getattr(self.options, 's1_loss_mode', 'full_map')
            if s1_loss_mode == 'full_map':
                raise ValueError("S1 buffer mode does not support s1_loss_mode='full_map'.")
            return self._train_compressor_steps_from_buffer(iteration_idx, n_steps)

        self._set_compressor_trainable(True)
        self.compressor.train()
        self.fusion.train()
        freeze_glace_head = self._glace_freeze_head() or self._glace_head_freeze_active(iteration_idx)
        if self._is_glace_backend():
            self._set_glace_head_trainable(not freeze_glace_head, stage_tag="S1")
        else:
            self.regressor.heads.train()

        # Backbone remains frozen / inference-only in S1.
        self.regressor.encoder.eval()
        self._log_stage_trainability("S1")

        # From iter 1 onward use a lower S1 LR to avoid destabilizing the already-trained compressor/fusion.
        s1_lr_scale_later = float(getattr(self.options, 's1_lr_scale_later', 0.4))
        base_lr_s1 = self.s1_learning_rate_max * (s1_lr_scale_later if iteration_idx > 0 else 1.0)
        if iteration_idx > 0:
            _logger.info("  [S1 LR] iter>0: base_lr scaled by %.2f -> %.2e", s1_lr_scale_later, base_lr_s1)

        s1_param_groups = [
            {'name': 'compressor', 'params': self.compressor.parameters()},
            {'name': 'fusion_residual', 'params': list(self.fusion.parameters()) + self._glace_residual_adapter_params()},
        ]
        gate_params = self._local_residual_gate_params()
        if gate_params:
            s1_param_groups.append({'name': 'local_residual_gate', 'params': gate_params, 'lr': base_lr_s1 * 0.1})
        if freeze_glace_head:
            _logger.info("  [S1] GLACE head freeze warmup active at iter %d; head excluded from optimizer.", iteration_idx + 1)
        else:
            s1_param_groups.append({'name': 'head', 'params': self.regressor.heads.parameters(), 'lr': base_lr_s1 * 0.1})
        comp_optimizer = optim.AdamW(s1_param_groups, lr=base_lr_s1)
        self._validate_glace_head_optimizer_membership("S1", comp_optimizer, expected_trainable=not freeze_glace_head)
        self._log_optimizer_groups("S1", comp_optimizer)

        sched_lmc = self._build_s1_scheduler(comp_optimizer, n_steps, iteration_idx, base_lr=base_lr_s1)
        s1_loader = self._build_s1_dataloader()
        s1_loss_mode = getattr(self.options, 's1_loss_mode', 'full_map')
        s1_bs = max(1, getattr(self.options, 's1_batch_size', 8))
        _logger.info("  [S1] source=online_encoder")
        _logger.info(
            "  [S1] loss_mode=%s, loss_step_mode=%s, dataloader batch=%d, target_updates=%d (strict update-count mode)",
            s1_loss_mode,
            self.s1_loss_step_mode,
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
        s1_early_stop_cfg = self._get_s1_early_stop_cfg()
        ema_px_err = None
        best_ema_px_err = float('inf')
        no_improve_updates = 0
        _s1_loss_step1 = None
        _s1_loss_last = None
        if s1_early_stop_cfg["enabled"]:
            _logger.info(
                "  [S1] early-stop ON (min_updates=%d, patience=%d, rel_improve=%.4f, ema_beta=%.2f)",
                s1_early_stop_cfg["min_updates"],
                s1_early_stop_cfg["patience"],
                s1_early_stop_cfg["rel_improve"],
                s1_early_stop_cfg["ema_beta"],
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
            image_BCHW, image_mask_B1HW, _, gt_pose_inv_B44, intrinsics_B33, intrinsics_inv_B33, *_ = batch
            gt_scene_coords_B3HW = self._extract_gt_scene_coords_from_batch(batch, image_BCHW=image_BCHW)

            image_BCHW = image_BCHW.to(self.device, non_blocking=True)
            image_mask_B1HW = image_mask_B1HW.to(self.device, non_blocking=True)
            gt_pose_inv_B44 = gt_pose_inv_B44.to(self.device, non_blocking=True)
            intrinsics_B33 = intrinsics_B33.to(self.device, non_blocking=True)
            intrinsics_inv_B33 = intrinsics_inv_B33.to(self.device, non_blocking=True)
            if gt_scene_coords_B3HW is not None:
                gt_scene_coords_B3HW = gt_scene_coords_B3HW.to(self.device, non_blocking=True).float()
            else:
                self._warn_missing_aux_ref_coords_once()

            if gt_pose_inv_B44.shape[1] == 4:
                gt_pose_inv_B34 = gt_pose_inv_B44[:, :3, :]
            else:
                gt_pose_inv_B34 = gt_pose_inv_B44

            if image_BCHW.dtype == torch.float16:
                image_BCHW = image_BCHW.float()

            img_idx_B_for_head = None
            if self._uses_image_global_features():
                img_idx_B_for_head = batch[-1].to(self.device, non_blocking=True).long()

            with autocast("cuda", enabled=self.options.use_half):
                with torch.no_grad():
                    local_feats = self.regressor.get_features(image_BCHW)
                if self._is_glace_backend():
                    raw_feats = self._build_glace_decoder_feature_maps(
                        local_feats,
                        img_idx_B_for_head,
                        stage_tag='S1-Online',
                    )
                else:
                    raw_feats = local_feats

                comp_out = self.compressor(self.memory_dict)
                fused_feats, base_for_residual_feats = self._fuse_lmc_features_for_head(
                    raw_feats,
                    comp_out,
                    stage_tag="S1-Online",
                )

                B, C, H, W = fused_feats.shape
                image_mask_B1HW = TF.resize(image_mask_B1HW, [H, W], interpolation=TF.InterpolationMode.NEAREST)
                image_mask_B1HW = image_mask_B1HW.bool()
                if gt_scene_coords_B3HW is None:
                    gt_scene_coords_B3HW = torch.zeros((B, 3, H, W), dtype=torch.float32, device=self.device)
                    gt_scene_coords_valid_B1HW = torch.zeros((B, 1, H, W), dtype=torch.bool, device=self.device)
                elif tuple(gt_scene_coords_B3HW.shape[-2:]) != (H, W):
                    gt_scene_coords_valid_B1HW = self._scene_coords_valid_mask(
                        gt_scene_coords_B3HW,
                        size_hw=(H, W),
                        device=self.device,
                    )
                    gt_scene_coords_B3HW = Fnn.interpolate(gt_scene_coords_B3HW, size=(H, W), mode="nearest")
                else:
                    gt_scene_coords_valid_B1HW = self._scene_coords_valid_mask(
                        gt_scene_coords_B3HW,
                        size_hw=(H, W),
                        device=self.device,
                    )
                coord_sampling_B1HW = self._expand_valid_coord_sampling_mask(
                    gt_scene_coords_valid_B1HW,
                    image_mask_B1HW,
                )

                if image_mask_B1HW.sum() == 0:
                    skipped_mask += 1
                    continue

                s1_loss_mode = getattr(self.options, 's1_loss_mode', 'full_map')

                if s1_loss_mode == 'full_map':
                    s1_step = iteration_idx * n_steps + update_step
                    loss, s1_stats = self._s1_compute_loss_full_map(
                        fused_feats, gt_pose_inv_B34, intrinsics_B33, intrinsics_inv_B33, image_mask_B1HW,
                        gt_scene_coords_B3HW=gt_scene_coords_B3HW,
                        gt_scene_coords_valid_B1HW=gt_scene_coords_valid_B1HW,
                        img_idx_B=img_idx_B_for_head,
                        s1_step=s1_step,
                        base_decoder_feats_BCHW=base_for_residual_feats,
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
                        'gt_scene_coords_world': normalize_shape(gt_scene_coords_B3HW),
                        'gt_scene_coords_valid': normalize_shape(gt_scene_coords_valid_B1HW),
                    }
                    if base_for_residual_feats is not None:
                        batch_data['base_decoder_features'] = normalize_shape(base_for_residual_feats)
                    if self._uses_image_global_features():
                        batch_data['img_idx'] = img_idx_B_for_head.unsqueeze(1).expand(B, H * W).reshape(-1)

                    image_mask_N1 = normalize_shape(image_mask_B1HW.float())
                    coord_valid_N1 = normalize_shape(coord_sampling_B1HW)
                    n_per_image = self.options.samples_per_image

                    if s1_loss_mode == 'sample_per_image':
                        per_image_indices = []
                        for b in range(B):
                            valid_flat = torch.where(image_mask_B1HW[b].flatten())[0]
                            n_sample = min(n_per_image, valid_flat.numel())
                            if n_sample < 16:
                                continue
                            idx_b = self._sample_buffer_indices(
                                image_mask_B1HW[b].flatten().float().view(-1, 1),
                                coord_sampling_B1HW[b].flatten().view(-1, 1),
                                n_sample,
                                True,
                            )
                            if idx_b is None or idx_b.numel() <= 0:
                                continue
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
                        sample_idxs = self._sample_buffer_indices(
                            image_mask_N1,
                            coord_valid_N1,
                            features_to_select,
                            True,
                        )
                        if sample_idxs is None or sample_idxs.numel() <= 0:
                            skipped_sample += 1
                            continue

                    for k in batch_data:
                        batch_data[k] = batch_data[k][sample_idxs]

                    loss, s1_stats = self._s1_compute_loss_from_features(
                        batch_data['features'].contiguous(),
                        batch_data['target_px'].contiguous(),
                        batch_data['gt_poses_inv'].contiguous(),
                        batch_data['intrinsics'].contiguous(),
                        batch_data['intrinsics_inv'].contiguous(),
                        batch_data['gt_scene_coords_world'].contiguous(),
                        batch_data['gt_scene_coords_valid'].contiguous(),
                        batch_data['img_idx'].contiguous() if 'img_idx' in batch_data else None,
                        s1_step=update_step,
                        base_decoder_features_bC=batch_data['base_decoder_features'].contiguous() if 'base_decoder_features' in batch_data else None,
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
                # Skip this noisy update entirely: no optimizer/scheduler step, no update counter increment.
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
                # Keep update-count semantics strict: non-finite loss must not advance optimizer/scheduler/step.
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
                self._glace_residual_adapter_params() +
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
            if s1_early_stop_cfg["enabled"]:
                if ema_px_err is None:
                    ema_px_err = px_err
                else:
                    ema_px_err = (
                        s1_early_stop_cfg["ema_beta"] * ema_px_err
                        + (1.0 - s1_early_stop_cfg["ema_beta"]) * px_err
                    )

                if update_step >= s1_early_stop_cfg["min_updates"]:
                    if ema_px_err < best_ema_px_err * (1.0 - s1_early_stop_cfg["rel_improve"]):
                        best_ema_px_err = ema_px_err
                        no_improve_updates = 0
                    else:
                        no_improve_updates += 1

                    if no_improve_updates >= s1_early_stop_cfg["patience"]:
                        _logger.info(
                            "  [S1] early-stop at update %d/%d (ema_pxErr=%.2f, best=%.2f, patience=%d)",
                            update_step, n_steps, ema_px_err, best_ema_px_err, s1_early_stop_cfg["patience"]
                        )
                        break

            if update_step % log_interval == 0 or update_step == 1 or update_step == n_steps:
                _cur_loss = float(loss.item())
                if update_step == 1 and math.isfinite(_cur_loss):
                    _s1_loss_step1 = _cur_loss
                if math.isfinite(_cur_loss):
                    _s1_loss_last = _cur_loss
                self._append_step_log(
                    iter_idx=iteration_idx,
                    step=self.iteration,
                    stage="S1",
                    loss=_cur_loss,
                    px_err=float(px_err),
                    lr=float(comp_optimizer.param_groups[0]["lr"]),
                    mode="reproj",
                    med3d=-1.0,
                )
                _logger.info(
                    "  [S1] update %d/%d (attempt=%d), bs=%d, loss=%.4f, valid=%.1f%%, pxErrL1=%.2f, pxErrL2=%.2f, validPxErrL1=%.2f, nonFinite=%.2f%%, lr=%.2e",
                    update_step, n_steps, raw_step, image_BCHW.shape[0], _cur_loss,
                    s1_stats["fraction_valid"] * 100.0, s1_stats["pxerr_l1"], s1_stats["pxerr_l2"],
                    s1_stats["valid_pxerr_l1"], s1_stats["nonfinite_ratio"] * 100.0, comp_optimizer.param_groups[0]["lr"],
                )

        if skipped_mask > 0 or skipped_sample > 0 or skipped_nonfinite > 0:
            _logger.warning(
                "  [S1] skipped batches: empty_mask=%d, few_samples=%d, nonfinite_loss=%d (total steps=%d)",
                skipped_mask, skipped_sample, skipped_nonfinite, n_steps
            )

        if (
            _s1_loss_step1 is not None
            and _s1_loss_last is not None
            and _s1_loss_step1 > 0
            and _s1_loss_last > _s1_loss_step1 * 0.9
        ):
            _logger.warning(
                "  [S1] CONVERGENCE WARNING: loss did not decrease significantly "
                "(step1=%.4f → final=%.4f, ratio=%.3f). "
                "Possible compressor-fusion desync building up. "
                "If this persists across iterations, late-stage S2 may diverge.",
                _s1_loss_step1, _s1_loss_last, _s1_loss_last / _s1_loss_step1,
            )

    # ------------------------------------------------------------------
    # S2: Head-only optimizer and schedule (per phase)
    # ------------------------------------------------------------------

    def _setup_s2_optimizer_and_schedule(self, iteration_idx, is_last):
        """Build head-only optimizer and scheduler for this S2 phase; set rewind and local step."""
        current_buffer_size = self.buffer_size_final if is_last else self.options.training_buffer_size
        steps_per_epoch = max(1, current_buffer_size // self.options.batch_size)
        self.steps_per_s2_phase = max(1, self.options.epochs * steps_per_epoch)
        _logger.info(
            "[S2] steps_per_s2_phase=%d (epochs=%d, buffer=%d, batch=%d, steps_per_epoch=%d)",
            self.steps_per_s2_phase, self.options.epochs, current_buffer_size,
            self.options.batch_size, steps_per_epoch,
        )

        if self.mapany_flow_profile:
            rewind_ratio = 0.0
        else:
            rewind_ratio = self.s2_repro_rewind_first_ratio if iteration_idx == 0 else self.s2_repro_rewind_later_ratio
        self.s2_rewind_amount = rewind_ratio * self.steps_per_s2_phase
        self.current_lmc_iter = iteration_idx
        self.local_s2_step = 0

        base_lr = self.s2_learning_rate_max * self.head_lr_multiplier_s2
        if self.mapany_flow_profile:
            boost = 1.0
        else:
            boost = self.s2_lr_boost_first if iteration_idx == 0 else self.s2_lr_boost_later
        head_lr = base_lr * boost

        # ACE-G R2 path: optionally include fusion params with slow LR
        ace_g_fusion_in_s2 = (
            self.lmc_flow == 'ace_g'
            and getattr(self.options, 'ace_g_fusion_in_s2', False)
        )
        if self._ace_lmc_stage2_uses_fused_buffer() and ace_g_fusion_in_s2:
            raise RuntimeError('[S2-G] stage1_fused feature source requires frozen fusion; set --ace_g_fusion_in_s2 False.')
        freeze_glace_head = self._glace_freeze_head() or self._glace_head_freeze_active(iteration_idx)
        if self._is_glace_backend():
            self._set_glace_head_trainable(not freeze_glace_head, stage_tag="S2-G")
            if freeze_glace_head and not ace_g_fusion_in_s2:
                raise ValueError(
                    "GLACE head-freeze warmup requires --ace_g_fusion_in_s2 True so plugin parameters can train in S2."
                )

        max_lrs = []
        if ace_g_fusion_in_s2:
            fusion_lr_ratio = float(getattr(self.options, 'ace_g_fusion_lr_ratio', 0.01))
            residual_lr_ratio = self._glace_residual_lr_ratio(fusion_lr_ratio)
            fusion_lr = head_lr * fusion_lr_ratio
            residual_lr = head_lr * residual_lr_ratio
            # S2 optimizer: compressor stays frozen. During GLACE warmup, the
            # original GLACE head is also frozen and only the memory plugin moves.
            s2_param_groups = []
            if not freeze_glace_head and not self._uses_ace_lmc_global_residual_head():
                s2_param_groups.append({'name': 'head', 'params': self.regressor.heads.parameters(), 'lr': head_lr})
                max_lrs.append(head_lr)
            residual_head_params = self._ace_lmc_global_residual_params()
            if residual_head_params:
                s2_param_groups.append({'name': 'ace_lmc_global_residual', 'params': residual_head_params, 'lr': head_lr})
                max_lrs.append(head_lr)
            s2_param_groups.append({'name': 'fusion', 'params': list(self.fusion.parameters()), 'lr': fusion_lr})
            max_lrs.append(fusion_lr)
            adapter_params = self._glace_residual_adapter_params()
            if adapter_params:
                s2_param_groups.append({'name': 'residual_adapter', 'params': adapter_params, 'lr': residual_lr})
                max_lrs.append(residual_lr)
            gate_params = self._local_residual_gate_params()
            if gate_params:
                s2_param_groups.append({'name': 'local_residual_gate', 'params': gate_params, 'lr': residual_lr})
                max_lrs.append(residual_lr)
            ace_lmc_global_gate_params = self._ace_lmc_global_gate_params()
            if ace_lmc_global_gate_params:
                s2_param_groups.append({'name': 'ace_lmc_global_gate', 'params': ace_lmc_global_gate_params, 'lr': head_lr})
                max_lrs.append(head_lr)
            self.optimizer_head = optim.AdamW(s2_param_groups)
            _logger.info(
                "[S2-G] R2 active: head_frozen=%s head_lr=%.2e fusion_lr=%.2e (ratio=%.4f) residual_lr=%.2e (ratio=%.4f)",
                freeze_glace_head, head_lr, fusion_lr, fusion_lr_ratio, residual_lr, residual_lr_ratio,
            )
        else:
            if self._uses_ace_lmc_global_residual_head():
                s2_param_groups = [{'name': 'ace_lmc_global_residual', 'params': self._ace_lmc_global_residual_params(), 'lr': head_lr}]
            else:
                s2_param_groups = [{'name': 'head', 'params': self.regressor.heads.parameters(), 'lr': head_lr}]
            max_lrs = [head_lr]
            ace_lmc_global_gate_params = self._ace_lmc_global_gate_params()
            if ace_lmc_global_gate_params:
                s2_param_groups.append({'name': 'ace_lmc_global_gate', 'params': ace_lmc_global_gate_params, 'lr': head_lr})
                max_lrs.append(head_lr)
            self.optimizer_head = optim.AdamW(s2_param_groups)
        self._validate_glace_head_optimizer_membership("S2-G", self.optimizer_head, expected_trainable=not freeze_glace_head)
        self._log_optimizer_groups("S2-G", self.optimizer_head)
        self._validate_s2_compressor_contract()
        self._validate_ace_lmc_stage2_fused_backbone_contract("S2-G", check_optimizer=True)

        warmup_ratio = (self.s2_lr_warmup_steps / self.steps_per_s2_phase) if self.s2_lr_warmup_steps else 0.1
        warmup_ratio = min(0.5, max(0.0, warmup_ratio))
        # PyTorch OneCycleLR: first phase ends at pct_start * total_steps - 1. If
        # pct_start * total_steps == 1, phase 1 has zero width → (end_step - start_step)==0
        # and get_lr() divides by zero. Clamp pct_start into (1/ts, 1 - 1/ts).
        ts = max(int(self.steps_per_s2_phase), 1)
        _eps = 1e-5
        _lo = 1.0 / ts + _eps
        _hi = 1.0 - 1.0 / ts - _eps
        if ts <= 2 or not (_lo < _hi):
            warmup_ratio_clamped = 0.3
            _logger.warning(
                "[S2] OneCycleLR: total_steps=%d too small for a non-degenerate two-phase "
                "schedule; using pct_start=%.3f",
                ts,
                warmup_ratio_clamped,
            )
        else:
            warmup_ratio_clamped = min(_hi, max(_lo, warmup_ratio))
            if abs(warmup_ratio_clamped - warmup_ratio) > 1e-6:
                _logger.info(
                    "[S2] OneCycleLR: pct_start clamped %.4f → %.4f (total_steps=%d, "
                    "avoid PyTorch div-by-zero when pct_start*steps≈1)",
                    warmup_ratio,
                    warmup_ratio_clamped,
                    ts,
                )
        self.scheduler_head = optim.lr_scheduler.OneCycleLR(
            self.optimizer_head,
            max_lr=max_lrs,
            total_steps=self.steps_per_s2_phase,
            pct_start=warmup_ratio_clamped,
            anneal_strategy='cos',
        )
        _logger.info(
            "[S2] head lr=%.2e (boost=%.2f), steps=%d, rewind=%.1f (tau=%.0f), repro_step_mode=%s%s",
            head_lr, boost, self.steps_per_s2_phase, self.s2_rewind_amount, self.s2_repro_rewind_tau,
            self.repro_step_mode,
            " [ACE-G R2: fusion trainable, compressor frozen]" if ace_g_fusion_in_s2 else "",
        )

    def _run_s2_polish_phase(self, iteration_idx):
        """Run an optional low-LR S2 refinement pass on the current buffer."""
        if self.s2_polish_epochs <= 0:
            return
        if self.s2_polish_head_lr <= 0:
            _logger.warning(
                "[S2-Polish] Skipping because s2_polish_head_lr=%.2e <= 0.",
                self.s2_polish_head_lr,
            )
            return

        ace_g_fusion_in_s2 = (
            self.lmc_flow == 'ace_g'
            and getattr(self.options, 'ace_g_fusion_in_s2', False)
        )
        head_lr = self.s2_polish_head_lr
        freeze_glace_head = self._glace_freeze_head() or self._glace_head_freeze_active(iteration_idx)
        if self._is_glace_backend():
            self._set_glace_head_trainable(not freeze_glace_head, stage_tag="S2-Polish")
        if ace_g_fusion_in_s2:
            fusion_lr = head_lr * self.s2_polish_fusion_lr_ratio
            residual_lr_ratio = self._glace_residual_lr_ratio(self.s2_polish_fusion_lr_ratio)
            residual_lr = head_lr * residual_lr_ratio
            polish_param_groups = []
            max_lrs = []
            if not freeze_glace_head:
                polish_param_groups.append({'name': 'head', 'params': self.regressor.heads.parameters(), 'lr': head_lr})
                max_lrs.append(head_lr)
            polish_param_groups.append({'name': 'fusion', 'params': list(self.fusion.parameters()), 'lr': fusion_lr})
            max_lrs.append(fusion_lr)
            adapter_params = self._glace_residual_adapter_params()
            if adapter_params:
                polish_param_groups.append({'name': 'residual_adapter', 'params': adapter_params, 'lr': residual_lr})
                max_lrs.append(residual_lr)
            gate_params = self._local_residual_gate_params()
            if gate_params:
                polish_param_groups.append({'name': 'local_residual_gate', 'params': gate_params, 'lr': residual_lr})
                max_lrs.append(residual_lr)
            self.optimizer_head = optim.AdamW(polish_param_groups)
            _logger.info(
                "[S2-Polish] iter=%d epochs=%d head_frozen=%s head_lr=%.2e fusion_lr=%.2e "
                "(ratio=%.4f) residual_lr=%.2e (ratio=%.4f, constant LR)",
                iteration_idx + 1,
                self.s2_polish_epochs,
                freeze_glace_head,
                head_lr,
                fusion_lr,
                self.s2_polish_fusion_lr_ratio,
                residual_lr,
                residual_lr_ratio,
            )
        else:
            self.optimizer_head = optim.AdamW([{'name': 'head', 'params': self.regressor.heads.parameters(), 'lr': head_lr}])
            max_lrs = [head_lr]
            _logger.info(
                "[S2-Polish] iter=%d epochs=%d head_lr=%.2e (constant LR)",
                iteration_idx + 1,
                self.s2_polish_epochs,
                head_lr,
            )

        self._validate_glace_head_optimizer_membership("S2-Polish", self.optimizer_head, expected_trainable=not freeze_glace_head)
        self._log_optimizer_groups("S2-Polish", self.optimizer_head)

        # Keep local_s2_step untouched: in per_iter mode this pins ReproLoss at the
        # low-clamp tail, matching the late-S2 refinement regime we want to test.
        self.scheduler_head = optim.lr_scheduler.LambdaLR(
            self.optimizer_head,
            lr_lambda=[lambda _step: 1.0 for _ in max_lrs],
        )
        self.current_lmc_iter = iteration_idx
        for polish_epoch in range(self.s2_polish_epochs):
            self.epoch = self.options.epochs + polish_epoch
            self.run_epoch()

    # ------------------------------------------------------------------
    # Iteration eval / best-checkpoint helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _torch_load_trusted_checkpoint(path, *, map_location='cpu'):
        try:
            return torch.load(path, map_location=map_location, weights_only=False)
        except TypeError:
            return torch.load(path, map_location=map_location)

    def _resume_enabled(self):
        return bool(getattr(self, "_resume_active", False))

    def _resume_config_values_equal(self, lhs, rhs):
        lhs = self._tensor_to_config_value(lhs)
        rhs = self._tensor_to_config_value(rhs)
        if isinstance(lhs, bool) or isinstance(rhs, bool):
            return bool(lhs) == bool(rhs)
        if isinstance(lhs, (int, float)) and isinstance(rhs, (int, float)):
            return math.isclose(float(lhs), float(rhs), rel_tol=1e-6, abs_tol=1e-6)
        if isinstance(lhs, list) and isinstance(rhs, list):
            return len(lhs) == len(rhs) and all(
                self._resume_config_values_equal(a, b) for a, b in zip(lhs, rhs)
            )
        if lhs is None or rhs is None:
            return lhs is rhs
        return lhs == rhs

    def _load_ace_lmc_local_checkpoint_if_requested(self):
        if not self._uses_ace_lmc_global_head():
            return
        checkpoint_path = getattr(self.options, 'ace_lmc_local_checkpoint_path', None)
        if checkpoint_path is None:
            raise ValueError('[ACE-FCN-LMC] Stage2 global modes require --ace_lmc_local_checkpoint_path.')
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f'ACE-FCN local checkpoint not found: {checkpoint_path}')
        checkpoint = self._torch_load_trusted_checkpoint(checkpoint_path, map_location='cpu')
        if not isinstance(checkpoint, dict) or 'lmc_config' not in checkpoint:
            raise ValueError(f'ACE-FCN local checkpoint must be an LMC checkpoint: {checkpoint_path}')
        checkpoint_config = checkpoint.get('lmc_config') or {}
        if str(checkpoint_config.get('model_backend', '')) != 'ace_fcn_lmc':
            raise ValueError(
                f"ACE-FCN local checkpoint model_backend must be ace_fcn_lmc, got {checkpoint_config.get('model_backend')!r}"
            )
        if str(checkpoint_config.get('ace_lmc_global_head_mode', 'none')) != 'none':
            raise ValueError('Stage-2 --ace_lmc_local_checkpoint_path must point to a pure local stage-1 checkpoint.')
        self.compressor.load_state_dict(checkpoint['compressor_state_dict'], strict=True)
        self.fusion.load_state_dict(checkpoint['fusion_state_dict'], strict=True)
        head_state = checkpoint.get('head_state_dict')
        if self._uses_ace_lmc_global_residual_head():
            if head_state is None:
                raise ValueError('glace_residual Stage2 requires head_state_dict in the local Stage1 checkpoint.')
            self.regressor.heads.load_state_dict(head_state, strict=True)
            self.regressor.heads.eval()
            for param in self.regressor.heads.parameters():
                param.requires_grad_(False)
            _logger.info('[ACE-FCN-LMC] Loaded and froze Stage1 local head as global-residual base.')
        elif self._uses_ace_lmc_global_film_head():
            if head_state is None:
                raise ValueError('glace_film Stage2 requires head_state_dict in the local Stage1 checkpoint.')
            missing, unexpected = self.regressor.heads.load_state_dict(head_state, strict=False)
            unexpected = list(unexpected)
            missing = list(missing)
            bad_missing = [k for k in missing if not k.startswith('film.') and k != 'global_gate']
            if unexpected or bad_missing:
                raise ValueError(
                    f'glace_film Stage2 could not initialize from Stage1 head: missing={missing} unexpected={unexpected}'
                )
            _logger.info('[ACE-FCN-LMC] Loaded Stage1 local head into GLACE-FiLM head (new FiLM params zero-init).')
        if self._uses_ace_lmc_stage2_teacher():
            if head_state is None:
                raise ValueError('Stage2 local-teacher losses require head_state_dict in the local Stage1 checkpoint.')
            teacher_in_dim = int(
                checkpoint_config.get('ace_lmc_final_head_dim')
                or checkpoint_config.get('encoder_feature_dim')
                or getattr(self.options, 'ace_encoder_features', 512)
            )
            self.ace_lmc_stage2_teacher_head = Head(
                self.dataset.mean_cam_center,
                self.options.num_head_blocks,
                self.options.use_homogeneous,
                in_channels=teacher_in_dim,
            ).to(self.device)
            self.ace_lmc_stage2_teacher_head.load_state_dict(head_state, strict=True)
            self.ace_lmc_stage2_teacher_head.eval()
            for param in self.ace_lmc_stage2_teacher_head.parameters():
                param.requires_grad_(False)
            _logger.info(
                '[ACE-FCN-LMC] Stage2 local teacher enabled: cons_weight=%.4f guard_weight=%.4f guard_margin=%.3fpx loss=%s cons_warmup=%d guard_warmup=%d sample_limit=%d teacher_dim=%d.',
                self._ace_lmc_stage2_consistency_base_weight(),
                self._ace_lmc_stage2_guard_base_weight(),
                float(getattr(self.options, 'ace_lmc_stage2_guard_margin_px', 0.25)),
                str(getattr(self.options, 'ace_lmc_stage2_consistency_loss', 'smooth_l1')),
                int(getattr(self.options, 'ace_lmc_stage2_consistency_warmup_steps', 0)),
                int(getattr(self.options, 'ace_lmc_stage2_guard_warmup_steps', 0)),
                int(getattr(self.options, 'ace_lmc_stage2_consistency_sample_limit', 0)),
                teacher_in_dim,
            )
        self.lmc_config['ace_lmc_local_checkpoint_path'] = str(checkpoint_path)
        _logger.info('[ACE-FCN-LMC] Loaded stage-1 compressor/fusion from %s', checkpoint_path)

        if bool(getattr(self.options, 'ace_lmc_freeze_local_stack', True)):
            for module in (getattr(self.regressor, 'encoder', None), self.compressor, self.fusion):
                if module is None:
                    continue
                module.eval()
                for param in module.parameters():
                    param.requires_grad_(False)
            _logger.info('[ACE-FCN-LMC] Frozen ACE encoder + compressor + fusion for global-head stage.')

    def _validate_resume_lmc_config(self, checkpoint_config):
        if not isinstance(checkpoint_config, dict):
            raise ValueError("Resume checkpoint is missing lmc_config.")
        strict = bool(getattr(self.options, "resume_strict_config", True))
        strict = strict and not bool(getattr(self.options, "resume_allow_config_mismatch", False))
        critical_keys = [
            "lmc_flow", "lmc_mode", "effective_lmc_mode", "num_latent_tokens", "num_fine",
            "num_coarse", "num_attn_layers", "use_scale_token", "compress_dim", "num_layers",
            "layers_idx", "lmc_key_slice_idx", "lmc_key_feature_mode", "lmc_feature_hierarchy_mode",
            "lmc_level_merge_mode", "lmc_level_merge_init", "lmc_level_token_gate",
            "lmc_level_anchor_residual_gamma_init", "geo_bias_mode", "geo_bias_rbf_scales",
            "geo_bias_rbf_per_head", "pos_encoding_mode", "point_rope_coord_norm", "point_rope_radius",
            "point_rope_radius_policy", "point_rope_mixed_memory_ratio", "point_rope_seed_pe", "point_rope_base", "point_rope_axes",
            "point_rope_apply_to", "geo_bias_crpb_dim", "geo_bias_crpb_input", "geo_bias_crpb_radius",
            "geo_bias_crpb_per_head", "pe_normalize_input", "lmc_compressor_pe_scale_mode",
            "lmc_fusion_geometry_mode", "lmc_fusion_key_geo_init", "lmc_fusion_refinement_mode",
            "lmc_fusion_cascade_layers", "lmc_fusion_assembly_mode", "lmc_fusion_assembly_gamma_init",
            "lmc_fusion_reread_delta_alpha", "lmc_fusion_reread_scalar_gate", "lmc_fusion_reread_gate_init",
            "lmc_fusion_reread_post_norm", "lmc_fusion_reread_trust_region_ratio", "lmc_fusion_reread_temperature",
            "lmc_fusion_reread_common_scale", "lmc_fusion_reread_effective_ratio_cap",
            "lmc_fusion_reread_qknorm_eps", "lmc_fusion_reread_qknorm_tau_init",
            "lmc_fusion_reread_layerscale_patch_init", "lmc_fusion_reread_layerscale_common_init",
            "lmc_fusion_dual_memory_layerscale_patch_init", "lmc_fusion_dual_memory_layerscale_common_init",
            "lmc_fusion_reread_warmup_mode", "lmc_fusion_reread_warmup_iters",
            "lmc_fusion_reread_warmup_start",
            "lmc_fusion_reread_geo_lambda", "lmc_fusion_reread_geo_sigma",
            "lmc_fusion_reread_geo_sigma_mode", "lmc_fusion_reread_geo_sigma_beta", "lmc_fusion_reread_geo_sigma_min",
            "ace_g_fusion_in_s2", "lmc_fusion_target", "requested_lmc_fusion_target",
            "effective_lmc_fusion_target", "fusion_query_dim", "local_residual_mode",
            "local_residual_alpha", "local_residual_alpha_init", "local_residual_alpha_max",
            "local_residual_alpha_warmup_steps", "backbone_feature_dim", "encoder_feature_dim", "memory_feature_dim",
            "scale_token_dim", "memory_path", "model_backend", "ace_encoder_path", "ace_lmc_global_head_mode",
            "ace_lmc_local_checkpoint_path", "ace_lmc_freeze_local_stack", "ace_lmc_global_feature_mode",
            "ace_lmc_stage2_feature_source", "ace_lmc_global_normalize", "ace_lmc_global_noise_std",
            "ace_lmc_global_gate_init", "ace_lmc_global_gate_learnable", "ace_lmc_global_gate_max", "ace_lmc_global_gate_l1_weight",
            "ace_lmc_global_residual_gate_l1_weight", "ace_lmc_global_residual_delta_max_m",
            "ace_lmc_global_residual_use_xyz_condition", "ace_lmc_global_residual_xyz_condition_mode",
            "ace_lmc_global_residual_xyz_condition_scale", "ace_lmc_global_residual_bad_gate_weight",
            "ace_lmc_random_global_seed",
            "ace_lmc_stage2_consistency_weight", "ace_lmc_stage2_consistency_loss",
            "ace_lmc_stage2_consistency_warmup_steps", "ace_lmc_stage2_consistency_sample_limit",
            "ace_lmc_stage2_guard_weight", "ace_lmc_stage2_guard_margin_px",
            "ace_lmc_stage2_guard_max_px", "ace_lmc_stage2_guard_warmup_steps",
            "ace_lmc_allow_mismatched_memory",
            "ace_lmc_memory_feature_source", "ace_lmc_memory_output_subsample", "ace_lmc_memory_coord_source", "ace_lmc_final_head_dim",
            "glace_encoder_path", "glace_init_head_path", "glace_feat_name",
            "glace_global_feat_dim", "glace_head_channels", "glace_mlp_ratio", "glace_fusion_query",
            "glace_residual_mode", "glace_residual_gate_init", "glace_residual_global_dim", "glace_head_freeze_iters", "glace_residual_lr_ratio",
            "glace_antiregression_weight", "glace_antiregression_margin_px", "glace_antiregression_max_px", "glace_pixel_diag_interval",
            "glace_freeze_base_network", "glace_freeze_encoder", "glace_freeze_head",
        ]
        mismatches = []
        missing = []
        for key in critical_keys:
            if key not in checkpoint_config:
                missing.append(key)
                continue
            current_value = self.lmc_config.get(key)
            checkpoint_value = checkpoint_config.get(key)
            if not self._resume_config_values_equal(current_value, checkpoint_value):
                mismatches.append((key, checkpoint_value, current_value))
        if missing:
            _logger.warning("[Resume] checkpoint lmc_config missing keys: %s", ", ".join(missing))
        if mismatches:
            preview = "; ".join(
                f"{key}: ckpt={old!r} current={new!r}"
                for key, old, new in mismatches[:12]
            )
            message = f"Resume config mismatch for {len(mismatches)} critical LMC fields: {preview}"
            if strict:
                raise ValueError(message + " (use --resume_allow_config_mismatch to override)")
            _logger.warning("[Resume] %s", message)

    def _estimate_resume_iteration_counter(self, start_iter):
        if start_iter <= 0:
            return int(getattr(self, "iteration", 0))
        completed_s1 = self.lmc_warmup_steps + max(0, start_iter - 1) * self.lmc_train_steps
        default_s2_batches = max(1, int(self.options.training_buffer_size) // int(self.options.batch_size))
        completed_s2 = start_iter * int(self.options.epochs) * default_s2_batches
        if getattr(self, "mapany_flow_profile", False):
            estimated = completed_s1 + completed_s2
        else:
            estimated = completed_s2
        return max(int(getattr(self, "iteration", 0)), int(estimated))

    def _load_resume_checkpoint_if_requested(self):
        self._resume_active = False
        self.resume_start_iter = 0
        self.resume_checkpoint_path = None
        checkpoint_path = getattr(self.options, "resume_checkpoint_path", None)
        if checkpoint_path is None:
            return

        checkpoint_path = Path(checkpoint_path).resolve()
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {checkpoint_path}")
        checkpoint = self._torch_load_trusted_checkpoint(checkpoint_path, map_location='cpu')
        if not isinstance(checkpoint, dict):
            raise ValueError(f"Resume checkpoint must be a dict: {checkpoint_path}")

        checkpoint_config = checkpoint.get("lmc_config")
        self._validate_resume_lmc_config(checkpoint_config)

        required_state_keys = ["head_state_dict", "compressor_state_dict", "fusion_state_dict"]
        missing_state = [key for key in required_state_keys if key not in checkpoint]
        if missing_state:
            raise ValueError(f"Resume checkpoint missing state dicts: {missing_state}")
        self.regressor.heads.load_state_dict(checkpoint["head_state_dict"], strict=True)
        self.compressor.load_state_dict(checkpoint["compressor_state_dict"], strict=True)
        self.fusion.load_state_dict(checkpoint["fusion_state_dict"], strict=True)
        if self._is_glace_backend() and getattr(self, 'glace_residual_adapter', None) is not None:
            adapter_state = checkpoint.get('glace_residual_adapter_state_dict')
            if adapter_state is not None:
                self.glace_residual_adapter.load_state_dict(adapter_state, strict=True)
        if self.local_residual_mode == 'learned_alpha':
            gate_state = checkpoint.get('local_residual_gate_state_dict')
            if gate_state is None or 'local_residual_alpha_logit' not in gate_state:
                raise ValueError("Resume checkpoint is missing local_residual_gate_state_dict for learned_alpha.")
            with torch.no_grad():
                self.local_residual_alpha_logit.copy_(gate_state['local_residual_alpha_logit'].to(self.device, dtype=torch.float32))
        if self._uses_ace_lmc_global_head() and isinstance(getattr(self, 'ace_lmc_global_gate', None), torch.nn.Parameter):
            gate_state = checkpoint.get('ace_lmc_global_gate_state_dict')
            if gate_state is None or 'ace_lmc_global_gate' not in gate_state:
                raise ValueError("Resume checkpoint is missing ace_lmc_global_gate_state_dict for learnable ACE-LMC global gate.")
            with torch.no_grad():
                self.ace_lmc_global_gate.copy_(gate_state['ace_lmc_global_gate'].to(self.device, dtype=torch.float32))
        if self._uses_ace_lmc_global_residual_head():
            residual_state = checkpoint.get('ace_lmc_global_residual_state_dict')
            if residual_state is None:
                raise ValueError("Resume checkpoint is missing ace_lmc_global_residual_state_dict for glace_residual.")
            self.ace_lmc_global_residual_head.load_state_dict(residual_state, strict=True)

        start_iter = int(getattr(self.options, "resume_best_iter", 0))
        if start_iter <= 0:
            raise ValueError(f"Invalid resume_best_iter={start_iter}; expected a 1-based completed best iteration.")
        relative_depth_only = int(getattr(self.options, "relative_depth_only_steps", 0) or 0) > 0
        if start_iter >= int(self.lmc_iterations) and not relative_depth_only:
            raise ValueError(
                f"Resume best_iter={start_iter} leaves no remaining LMC iterations "
                f"for lmc_iterations={self.lmc_iterations}. Increase --lmc_iterations to continue."
            )
        if start_iter >= int(self.lmc_iterations) and relative_depth_only:
            _logger.info(
                "[Resume] best_iter=%d reaches lmc_iterations=%d; allowed because relative_depth_only_steps=%d.",
                start_iter,
                int(self.lmc_iterations),
                int(getattr(self.options, "relative_depth_only_steps", 0) or 0),
            )

        self.resume_start_iter = start_iter
        self.resume_checkpoint_path = checkpoint_path
        self.best_iter = start_iter
        self.best_score = float(getattr(self.options, "resume_best_score", -float('inf')))
        self.best_eval = getattr(self.options, "resume_best_meta", None)
        self.iteration = self._estimate_resume_iteration_counter(start_iter)
        self._resume_active = True

        output_path = Path(self.options.output_map).resolve()
        if output_path != checkpoint_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            self.save_model(output_path)
            if self.best_score > -float('inf'):
                self._write_best_checkpoint_meta(self.best_iter, self.best_score, self.best_eval)
            _logger.info(
                "[Resume] Materialized initial best checkpoint for new run: %s",
                output_path,
            )

        _logger.info(
            "[Resume] Loaded best checkpoint %s | best_iter=%d | best_score=%.4f | continuing at iter %d/%d | iteration_counter=%d",
            checkpoint_path,
            self.best_iter,
            self.best_score,
            self.resume_start_iter + 1,
            self.lmc_iterations,
            self.iteration,
        )

    def _iteration_eval_hypotheses(self):
        return int(getattr(self.options, 'iteration_eval_hypotheses', 64))

    def _iteration_eval_seed(self):
        seed = getattr(self.options, 'iteration_eval_seed', None)
        if seed is None:
            seed = getattr(self.options, 'eval_dsacstar_seed', 1305)
        return int(seed)

    def _write_fresh_train_header(self):
        with open(self.step_log_path, 'w', encoding='utf-8') as f:
            f.write("Timestamp   Iter      Step  Stage               Loss       PxErr          LR    3D_Med  Mode        \n")
        with open(self.training_log_path, 'w', encoding='utf-8') as f:
            f.write(
                "iter,s1_steps,s2_epochs,buffer_size,is_best,score,"
                "pct50_5,pct25_5,pct10_5,pct5,pct2,pct1,median_t_cm,median_r_deg,avg_time_ms,elapsed_s\n"
            )
        with open(self.eval_log_path, 'w', encoding='utf-8') as f:
            f.write("# Iteration evaluation log (aligns with post_train_eval.txt fields)\n")
            f.write(f"best_metric={self.best_metric}, keep_best_only={self.keep_best_only}\n")
            f.write(
                f"eval_type=iteration,hypotheses={self._iteration_eval_hypotheses()},"
                f"seed={self._iteration_eval_seed()},"
                f"deterministic={bool(getattr(self.options, 'eval_deterministic', False))},"
                f"dsacstar_seed_per_frame={bool(getattr(self.options, 'eval_dsacstar_seed_per_frame', True))}\n"
            )
            f.write(
                "# Each iter: median_rotation_deg, median_translation_cm, acc50/25/10/5/2/1cm%% "
                "same as post_train_eval; avg_time_ms=per-frame infer\n"
            )

    def _write_train_header(self):
        if not self._resume_enabled():
            self._write_fresh_train_header()
            return

        if not (self.step_log_path.exists() and self.training_log_path.exists() and self.eval_log_path.exists()):
            self._write_fresh_train_header()

        stamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        with open(self.step_log_path, 'a', encoding='utf-8') as f:
            f.write(
                f"# Resume {stamp}: checkpoint={self.resume_checkpoint_path} "
                f"best_iter={self.best_iter} best_score={self.best_score:.4f} "
                f"next_iter={self.resume_start_iter + 1}\n"
            )
        with open(self.training_log_path, 'a', encoding='utf-8') as f:
            f.write(
                f"# resume,{stamp},checkpoint={self.resume_checkpoint_path},"
                f"best_iter={self.best_iter},best_score={self.best_score:.4f},"
                f"next_iter={self.resume_start_iter + 1}\n"
            )
        with open(self.eval_log_path, 'a', encoding='utf-8') as f:
            f.write(
                f"# Resume {stamp}: checkpoint={self.resume_checkpoint_path} "
                f"best_iter={self.best_iter} best_score={self.best_score:.4f} "
                f"next_iter={self.resume_start_iter + 1}\n"
            )

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

    def _evaluate_checkpoint(self, ckpt_path, iter_idx, eval_stage="posts2"):
        from test_ace_dinov2_lmc import run_evaluation_lmc
        # Always use 'cuda:0': after setup_cuda_environment() sets CUDA_VISIBLE_DEVICES to the
        # physical GPU index, that GPU is always visible as logical cuda:0 within this process.
        # str(self.device) returns 'cuda' (no index) which can behave unexpectedly in eval.
        _eval_d = self.device
        if _eval_d.type == 'cuda':
            eval_device = 'cuda:0'
        else:
            eval_device = str(_eval_d)
        session = f"iter_{iter_idx+1:02d}" if not eval_stage else f"iter_{iter_idx+1:02d}_{eval_stage}"
        _logger.info(
            "[Eval] iter %d stage=%s: ckpt=%s  eval_device=%s",
            iter_idx + 1,
            eval_stage,
            ckpt_path,
            eval_device,
        )
        eval_opt = SimpleNamespace(
            scene=getattr(self.options, 'post_train_eval_scene', None) or self.options.scene,
            network=ckpt_path,
            data_backend=getattr(self.options, 'data_backend', 'ace'),
            wai_repo_root=getattr(self.options, 'wai_repo_root', None),
            wai_image_modality=getattr(self.options, 'wai_image_modality', 'image'),
            dinov2_path=self.options.dinov2_path,
            ace_encoder_path=getattr(self.options, 'ace_encoder_path', None),
            glace_root=getattr(self.options, 'glace_root', None),
            glace_feat_name=getattr(self.options, 'glace_feat_name', 'features.npy'),
            device=eval_device,
            image_resolution=self.options.image_resolution,
            session=session,
            hypotheses=self._iteration_eval_hypotheses(),
            eval_deterministic=getattr(self.options, 'eval_deterministic', False),
            dsacstar_seed=self._iteration_eval_seed(),
            dsacstar_seed_per_frame=getattr(self.options, 'eval_dsacstar_seed_per_frame', True),
            eval_num_workers=getattr(self.options, 'eval_num_workers', 6),
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
        if self.best_metric == 'pct25_5':
            return float(eval_result.get('pct25_5', 0.0))
        if self.best_metric == 'pct50_5':
            return float(eval_result.get('pct50_5', 0.0))
        if self.best_metric == 'composite':
            pct5 = float(eval_result.get('pct5', 0.0))
            med_t = float(eval_result.get('median_tErr', 1e9))  # cm
            med_r = float(eval_result.get('median_rErr', 1e9))  # deg
            return pct5 - 2.0 * med_t - 2.0 * med_r
        if self.best_metric in ('rt_error', 'median_error'):
            # Lower median error is better; convert to a score where larger is better.
            # Translation is in cm and rotation is in deg; this mirrors the legacy
            # rt_error behavior while exposing a clearer Cambridge-facing name.
            med_t = float(eval_result.get('median_tErr', 1e9))
            med_r = float(eval_result.get('median_rErr', 1e9))
            return -(med_t + med_r)
        if self.best_metric == 'median_t':
            return -float(eval_result.get('median_tErr', 1e9))
        if self.best_metric == 'median_r':
            return -float(eval_result.get('median_rErr', 1e9))
        return float(eval_result.get('pct5', 0.0)) - 1e-3 * float(eval_result.get('median_tErr', 0.0)) - 1e-4 * float(eval_result.get('median_rErr', 0.0))

    def _log_iteration_summary(self, it, s1_steps, is_last, is_best, score, eval_result, elapsed_s):
        buffer_size = self.buffer_size_final if is_last else self.options.training_buffer_size
        if eval_result:
            pct50_5 = float(eval_result.get('pct50_5', 0.0))
            pct25_5 = float(eval_result.get('pct25_5', 0.0))
            pct10_5 = float(eval_result.get('pct10_5', 0.0))
            pct5 = float(eval_result.get('pct5', 0.0))
            pct2 = float(eval_result.get('pct2', 0.0))
            pct1 = float(eval_result.get('pct1', 0.0))
            med_t = float(eval_result.get('median_tErr', 0.0))
            med_r = float(eval_result.get('median_rErr', 0.0))
            avg_ms = float(eval_result.get('avg_time', 0.0)) * 1000.0
        else:
            pct50_5 = pct25_5 = pct10_5 = pct5 = pct2 = pct1 = med_t = med_r = avg_ms = 0.0
        with open(self.training_log_path, 'a', encoding='utf-8') as f:
            f.write(
                f"{it + 1},{s1_steps},{self.options.epochs},{buffer_size},{int(is_best)},"
                f"{score:.6f},{pct50_5:.4f},{pct25_5:.4f},{pct10_5:.4f},{pct5:.4f},{pct2:.4f},{pct1:.4f},"
                f"{med_t:.4f},{med_r:.4f},{avg_ms:.2f},{elapsed_s:.2f}\n"
            )
        with open(self.eval_log_path, 'a', encoding='utf-8') as f:
            tag = "BEST" if is_best else "-"
            f.write(
                f"iter={it + 1:02d} tag={tag} score={score:.4f} "
                f"median_rotation_deg={med_r:.4f} median_translation_cm={med_t:.4f} "
                f"acc50_5={pct50_5:.2f} acc25_5={pct25_5:.2f} acc10_5={pct10_5:.2f} acc5={pct5:.2f} "
                f"acc2={pct2:.2f} acc1={pct1:.2f} avg_time_ms={avg_ms:.2f} elapsed={elapsed_s:.1f}s\n"
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
        meta.update(self._lmc_semantics_for_logs())
        if eval_result:
            meta["pct50_5"] = float(eval_result.get("pct50_5", 0.0))
            meta["pct25_5"] = float(eval_result.get("pct25_5", 0.0))
            meta["pct10_5"] = float(eval_result.get("pct10_5", 0.0))
            meta["pct5"] = float(eval_result.get("pct5", 0.0))
            meta["pct2"] = float(eval_result.get("pct2", 0.0))
            meta["pct1"] = float(eval_result.get("pct1", 0.0))
            meta["median_tErr"] = float(eval_result.get("median_tErr", 0.0))
            meta["median_rErr"] = float(eval_result.get("median_rErr", 0.0))
            meta["avg_time_ms"] = float(eval_result.get("avg_time", 0.0)) * 1000.0
        import json
        path = self.options.output_map.parent / "best_checkpoint_meta.json"
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(meta, f, indent=2)

    def _reset_vanilla_optimizer_scheduler_DEPRECATED(self):
        """DEPRECATED: 此方法已废弃，vanilla路径现在使用与train_ace_dinov2_iterative.py完全相同的逻辑"""
        raise NotImplementedError("This method is deprecated. Use reset_optimizer_scheduler() from parent class instead.")

    def _train_vanilla_iterations(self):
        """
        Iterative vanilla baseline using the EXACT same logic as train_ace_dinov2_iterative.py.
        This ensures consistency and correctness with the verified implementation.
        """
        self.training_start = time.time()
        self._write_train_header()
        best_ckpt_exists = False

        _logger.info("=" * 80)
        _logger.info("[Vanilla-Iter] Using train_ace_dinov2_iterative.py logic")
        _logger.info("[Vanilla-Iter] Iterations: %d | Buffer per iter: %d | Epochs: %d",
                     self.vanilla_iterations, self.options.training_buffer_size, self.options.epochs)
        _logger.info("=" * 80)

        # 关键修复：使用父类的 reset_optimizer_scheduler，与 train_ace_dinov2_iterative.py 完全一致
        # 不重置 self.iteration，让 ReproLoss 正确调度
        for it in range(self.vanilla_iterations):
            iter_start = time.time()
            _logger.info("=== Iteration %d/%d ===", it + 1, self.vanilla_iterations)

            # 第一次迭代后，重置 optimizer/scheduler（与 train_ace_dinov2_iterative.py:252-256 一致）
            if it > 0:
                self.reset_optimizer_scheduler(
                    keep_optimizer_state=False,  # 默认不保持 optimizer 状态，与原逻辑一致
                    buffer_size=self.options.training_buffer_size,
                )

            # 填充 buffer（与 train_ace_dinov2_iterative.py:258-260 一致）
            # 注：LMC 重写的方法参数为 buffer_size_override，vanilla 路径下调用 super() 用 options.training_buffer_size
            buffer_start = time.time()
            self.create_training_buffer(buffer_size_override=self.options.training_buffer_size)
            _logger.info("Filled training buffer in %.1fs.", time.time() - buffer_start)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # 训练 epochs（与 train_ace_dinov2_iterative.py:262-263 一致）
            for self.epoch in range(self.options.epochs):
                self.run_epoch()

            # 保存和评估（与 train_ace_dinov2_iterative.py:265-268 一致）
            iter_ckpt = self.options.output_map.parent / f"{self.options.output_map.stem}.iter_{it+1:02d}.tmp.pt"
            self.save_model(iter_ckpt)

            eval_result = None
            score = -float('inf')
            if self.eval_each_iteration:
                try:
                    eval_result = self._evaluate_checkpoint(iter_ckpt, it, eval_stage="posts2")
                    score = self._score_eval(eval_result)
                except Exception as e:
                    _logger.warning("[Eval] vanilla iteration %d failed: %s", it + 1, e, exc_info=True)
            else:
                score = float(it + 1)

            # 管理最优模型
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

        # 保存最终模型（与 train_ace_dinov2_iterative.py:270 一致）
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
    # Reread residual warmup
    # ------------------------------------------------------------------

    def _set_fusion_reread_warmup_for_iteration(self, iteration_idx):
        if not hasattr(self, 'fusion') or self.fusion is None:
            return 1.0
        setter = getattr(self.fusion, 'set_reread_warmup_progress', None)
        if setter is None:
            return 1.0
        scale = float(setter(iteration_idx))
        mode = getattr(self.fusion, 'fusion_reread_warmup_mode', 'none')
        if mode != 'none':
            _logger.info(
                "[LMC-Fusion] reread warmup iteration=%d/%d mode=%s scale=%.6f",
                iteration_idx + 1,
                int(getattr(self, 'lmc_iterations', 0)),
                mode,
                scale,
            )
        return scale

    # ------------------------------------------------------------------
    # Override: train (two-stage multi-iteration)
    # ------------------------------------------------------------------

    def _train_iterative(self):
        """Two-stage iterative training (S1+S2) with iteration-level eval and best-checkpoint policy."""
        self.training_start = time.time()
        self._write_train_header()
        best_ckpt_exists = bool(self._resume_enabled() and Path(self.options.output_map).exists())
        start_iter = int(getattr(self, 'resume_start_iter', 0))
        if start_iter > 0:
            _logger.info(
                "[Resume] Iterative flow continuing from completed best_iter=%d; next iteration=%d/%d",
                self.best_iter, start_iter + 1, self.lmc_iterations,
            )

        for it in range(start_iter, self.lmc_iterations):
            iter_start = time.time()
            is_last = (it == self.lmc_iterations - 1)
            self.current_lmc_iteration = int(it)
            self._set_fusion_reread_warmup_for_iteration(it)
            _logger.info(f"\n{'='*60}")
            _logger.info(f"[LMC] Iteration {it+1}/{self.lmc_iterations}"
                         f"{' (FINAL)' if is_last else ''}")
            _logger.info(f"{'='*60}")

            # --- Stage 1 ---
            skip_s1_for_frozen_ace_global = (
                self._uses_ace_lmc_global_head()
                and bool(getattr(self.options, 'ace_lmc_freeze_local_stack', True))
            )
            if skip_s1_for_frozen_ace_global:
                s1_steps = 0
                _logger.info(
                    "[S1] Skipping compressor/fusion updates for ACE-FCN global-head stage; "
                    "using frozen stage-1 local LMC stack."
                )
            else:
                s1_steps = self.lmc_warmup_steps if it == 0 else self.lmc_train_steps
                _logger.info(f"[S1] Training compressor for {s1_steps} steps")
                if self.s1_use_buffer:
                    self._prepare_s1_buffer_for_iteration(it)
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
                elapsed_s = time.time() - iter_start
                with open(self.eval_log_path, "a", encoding="utf-8") as f:
                    f.write(
                        f"iter={it + 1:02d} tag=SKIP_S1_NAN score=-inf "
                        f"median_rotation_deg=n/a median_translation_cm=n/a "
                        f"acc50_5=0 acc25_5=0 acc10_5=0 acc5=0 acc2=0 acc1=0 avg_time_ms=0 elapsed={elapsed_s:.1f}s\n"
                    )
                with open(self.training_log_path, "a", encoding="utf-8") as f:
                    f.write(
                        f"{it + 1},{s1_steps},{self.options.epochs},"
                        f"{self.training_buffer_size},0,-999999.000000,"
                        f"0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.00,{elapsed_s:.2f}\n"
                    )
                continue

            # --- Head reset before Stage 2 ---
            self._maybe_reset_head(it)

            # --- Stage 2 ---
            self._set_compressor_trainable(False)
            buf_size = self.buffer_size_final if is_last else None
            _logger.info(f"[S2] Filling buffer"
                         f" (size={'FINAL ' + str(self.buffer_size_final) if is_last else 'default'})")
            self.create_training_buffer(buffer_size_override=buf_size)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            self._setup_s2_optimizer_and_schedule(it, is_last)
            self._reset_s2_nan_guard()
            _logger.info(f"[S2] Training head for {self.options.epochs} epochs")
            for self.epoch in range(self.options.epochs):
                self.run_epoch()
                if bool(getattr(self, '_s2_abort_current_iteration', False)):
                    break
            if bool(getattr(self, '_s2_abort_current_iteration', False)):
                _logger.error(
                    "[S2] Iteration %d aborted by NaN guard before checkpoint/eval; keeping previous best.",
                    it + 1,
                )
                break

            # --- Iteration checkpoint + eval ---
            iter_ckpt = self.options.output_map.parent / f"{self.options.output_map.stem}.iter_{it+1:02d}.tmp.pt"
            self.save_model(iter_ckpt)
            eval_result = None
            score = -float('inf')
            if self.eval_each_iteration:
                try:
                    eval_result = self._evaluate_checkpoint(iter_ckpt, it, eval_stage="posts2")
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

    def _train_ace_g(self):
        """ACE-G hybrid training path.

        Key difference from _train_iterative():
        - Buffer stores raw backbone features (no fusion) via create_training_buffer_ace_g()
        - Fusion is applied per-batch in S2 via _training_step_ace_g()
        - Compressed memory is cached once at S2 start (_s2_compressor_out)
        - R1 (default): fusion frozen in S2; R2: fusion trainable with slow LR

        Shared with _train_iterative():
        - S1 compressor training (_train_compressor_steps)
        - Head reset (_maybe_reset_head)
        - S2 optimizer/schedule (_setup_s2_optimizer_and_schedule)
        - Eval/checkpoint logic
        """
        self.training_start = time.time()
        self._write_train_header()
        best_ckpt_exists = bool(self._resume_enabled() and Path(self.options.output_map).exists())
        start_iter = int(getattr(self, 'resume_start_iter', 0))
        if start_iter > 0:
            _logger.info(
                "[Resume] ACE-G flow continuing from completed best_iter=%d; next iteration=%d/%d",
                self.best_iter, start_iter + 1, self.lmc_iterations,
            )

        # ACE-G S2 may optionally run fusion on the fly during training.
        ace_g_fusion_in_s2 = getattr(self.options, 'ace_g_fusion_in_s2', False)
        ace_g_cross_iter_eval = getattr(self.options, 'ace_g_cross_iter_eval', False)
        _logger.info("[ACE-G] R2 (fusion trainable in S2): %s", ace_g_fusion_in_s2)
        _logger.info("[ACE-G] Cross-iter eval: %s", ace_g_cross_iter_eval)
        _logger.info("[ACE-G] S2 feature_source: %s", self._ace_lmc_stage2_feature_source())

        prev_post_s2_score = None  # Tracks previous iteration post-S2 score
        prev_post_s2_head_state = None  # Snapshot of previous iteration head after S2

        for it in range(start_iter, self.lmc_iterations):
            iter_start = time.time()
            is_last = (it == self.lmc_iterations - 1)
            self.current_lmc_iteration = int(it)
            self._set_fusion_reread_warmup_for_iteration(it)
            _logger.info(f"\n{'='*60}")
            _logger.info(f"[ACE-G] Iteration {it+1}/{self.lmc_iterations}"
                         f"{' (FINAL)' if is_last else ''}")
            _logger.info(f"{'='*60}")

            # --- Stage 1: train compressor + fusion + head (same as iterative) ---
            skip_s1_for_frozen_ace_global = (
                self._uses_ace_lmc_global_head()
                and bool(getattr(self.options, 'ace_lmc_freeze_local_stack', True))
            )
            if skip_s1_for_frozen_ace_global:
                s1_steps = 0
                _logger.info(
                    "[S1] Skipping compressor/fusion updates for ACE-FCN global-head stage; "
                    "using frozen stage-1 local LMC stack."
                )
            else:
                s1_steps = self.lmc_warmup_steps if it == 0 else self.lmc_train_steps
                _logger.info(f"[S1] Training compressor for {s1_steps} steps")
                if self.s1_use_buffer:
                    s1_last_use_final = getattr(self.options, 's1_last_iter_use_final_buffer', False)
                    s1_buf_size = self.buffer_size_final if (is_last and s1_last_use_final) else None
                    self._prepare_s1_buffer_for_iteration(it, buffer_size_override=s1_buf_size)
                self._train_compressor_steps(it, s1_steps)

            # S1 NaN guard
            _s1_nan = any(
                p.data.isnan().any().item()
                for p in list(self.compressor.parameters()) + list(self.fusion.parameters())
            )
            if _s1_nan:
                _logger.error(
                    "[ACE-G] iter %d: compressor/fusion has NaN weights after S1 — "
                    "skipping S2. Retry with --use_half False.",
                    it + 1,
                )
                elapsed_s = time.time() - iter_start
                with open(self.eval_log_path, "a", encoding="utf-8") as f:
                    f.write(
                        f"iter={it + 1:02d} tag=SKIP_S1_NAN score=-inf "
                        f"median_rotation_deg=n/a median_translation_cm=n/a "
                        f"acc50_5=0 acc25_5=0 acc10_5=0 acc5=0 acc2=0 acc1=0 avg_time_ms=0 elapsed={elapsed_s:.1f}s\n"
                    )
                with open(self.training_log_path, "a", encoding="utf-8") as f:
                    f.write(
                        f"{it + 1},{s1_steps},{self.options.epochs},"
                        f"{self.training_buffer_size},0,-999999.000000,"
                        f"0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.00,{elapsed_s:.2f}\n"
                    )
                continue

            # --- Cross-iteration eval (C3): old-head + new-compressor before current S2 ---
            if (
                ace_g_cross_iter_eval
                and it > 0
                and prev_post_s2_score is not None
                and prev_post_s2_head_state is not None
            ):
                _logger.info(
                    "[ACE-G] Cross-iter eval: old-head (iter %d post-S2) + new-compressor (iter %d post-S1)",
                    it,
                    it + 1,
                )
                cross_ckpt = self.options.output_map.parent / f"{self.options.output_map.stem}.cross_iter_{it+1:02d}.tmp.pt"
                current_head_state = {
                    k: v.detach().cpu().clone()
                    for k, v in self.regressor.heads.state_dict().items()
                }
                self.save_model(cross_ckpt)
                try:
                    # Replace current head with previous-iteration post-S2 head to isolate compressor shift.
                    self.regressor.heads.load_state_dict(prev_post_s2_head_state, strict=True)
                    self.save_model(cross_ckpt)
                    cross_eval = self._evaluate_checkpoint(cross_ckpt, it, eval_stage="cross")
                    cross_score = self._score_eval(cross_eval)
                    drop = prev_post_s2_score - cross_score
                    drop_ratio = drop / max(abs(prev_post_s2_score), 1e-6)
                    _logger.info(
                        "[ACE-G] Cross-iter eval: prev_post_s2=%.4f -> old_head+new_comp=%.4f "
                        "(drop=%.4f, drop_ratio=%.4f)",
                        prev_post_s2_score, cross_score, drop, drop_ratio,
                    )
                except Exception as e:
                    _logger.warning("[ACE-G] Cross-iter eval failed: %s", e)
                finally:
                    # Restore current (post-S1) head for subsequent S2.
                    self.regressor.heads.load_state_dict(current_head_state, strict=True)
                    if cross_ckpt.exists():
                        cross_ckpt.unlink()

            # --- Head reset before Stage 2 (same as iterative) ---
            self._maybe_reset_head(it)

            # --- Stage 2: ACE-G specific ---
            # 1. Compress memory once, cache for training step
            self._set_compressor_trainable(False)
            self._log_stage_trainability("S2-G")
            self._validate_s2_compressor_contract()
            self._s2_compressor_out = self._compress_memory()

            # 2. Set fusion mode for S2
            if ace_g_fusion_in_s2:
                self.fusion.train()
                _logger.info("[S2-G] Fusion set to TRAIN mode (R2)")
            else:
                self.fusion.eval()
                _logger.info("[S2-G] Fusion set to EVAL mode (R1, frozen)")

            # 3. Fill S2 buffer with raw or frozen Stage1-fused features.
            buf_size = self.buffer_size_final if is_last else None
            _logger.info(
                "[S2-G] Filling buffer with feature_source=%s (size=%s)",
                self._ace_lmc_stage2_feature_source(),
                'FINAL ' + str(self.buffer_size_final) if is_last else 'default',
            )
            self.create_training_buffer_ace_g(buffer_size_override=buf_size)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # 4. Build S2 optimizer/schedule (includes fusion params if R2)
            self._setup_s2_optimizer_and_schedule(it, is_last)
            self._reset_s2_nan_guard()

            # 5. Train head (with on-the-fly fusion via run_epoch routing)
            _logger.info(f"[S2-G] Training head for {self.options.epochs} epochs")
            for self.epoch in range(self.options.epochs):
                self.run_epoch()
                if bool(getattr(self, '_s2_abort_current_iteration', False)):
                    break
            if bool(getattr(self, '_s2_abort_current_iteration', False)):
                self._s2_compressor_out = None
                _logger.error(
                    "[S2-G] Iteration %d aborted by NaN guard before checkpoint/eval; keeping previous best.",
                    it + 1,
                )
                break
            self._run_s2_polish_phase(it)

            # Clean up cached compressor output
            self._s2_compressor_out = None

            # --- Iteration checkpoint + eval (same as iterative) ---
            iter_ckpt = self.options.output_map.parent / f"{self.options.output_map.stem}.iter_{it+1:02d}.tmp.pt"
            self.save_model(iter_ckpt)
            eval_result = None
            score = -float('inf')
            if self.eval_each_iteration:
                try:
                    eval_result = self._evaluate_checkpoint(iter_ckpt, it, eval_stage="posts2")
                    score = self._score_eval(eval_result)
                except Exception as e:
                    _logger.warning("[Eval] iteration %d failed: %s", it + 1, e, exc_info=True)
            else:
                score = float(it + 1)

            prev_post_s2_score = score  # Store post-S2 score for next iteration cross-iter comparison
            prev_post_s2_head_state = {
                k: v.detach().cpu().clone()
                for k, v in self.regressor.heads.state_dict().items()
            }

            is_best = (score > self.best_score)
            if is_best:
                self.best_score = score
                self.best_iter = it + 1
                self.best_eval = eval_result
                os.replace(iter_ckpt, self.options.output_map)
                best_ckpt_exists = True
                _logger.info("[Best] Updated best ACE-G checkpoint at iter %d (score=%.4f)", it + 1, score)
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

        self._free_training_gpu_memory()

        _logger.info(
            "Done ACE-G. Total time: %.1fs | best_iter=%s | best_score=%.4f",
            time.time() - self.training_start, self.best_iter, self.best_score,
        )
        _logger.info("Logs: %s | %s | %s", self.step_log_path, self.training_log_path, self.eval_log_path)

    def _train_relative_depth_only(self):
        """Fine-tune a resumed checkpoint using only image-level relative depth."""
        steps = int(getattr(self.options, "relative_depth_only_steps", 0) or 0)
        if steps <= 0:
            raise ValueError("relative_depth_only_steps must be > 0.")
        if not self._resume_enabled():
            raise ValueError("relative_depth_only training requires --resume_checkpoint_path or --resume_from_run_dir.")
        if self.relative_depth_distiller is None:
            raise ValueError("relative_depth_only training requires --use_relative_depth_loss True.")
        lr = float(getattr(self.options, "relative_depth_only_lr", 1e-5) or 0.0)
        if lr <= 0.0:
            raise ValueError("relative_depth_only_lr must be > 0.")

        self.training_start = time.time()
        self._write_train_header()
        self._set_compressor_trainable(False)
        self.compressor.eval()
        self.regressor.encoder.eval()
        for param in self.regressor.encoder.parameters():
            param.requires_grad_(False)

        train_fusion = bool(getattr(self.options, "relative_depth_only_train_fusion", True))
        train_fusion = train_fusion and self.lmc_flow == "ace_g"
        for param in self.fusion.parameters():
            param.requires_grad_(train_fusion)
        self.fusion.train(train_fusion)

        head_params = []
        if self._uses_ace_lmc_global_residual_head():
            head_params.extend(self._ace_lmc_global_residual_params())
            for param in self.regressor.heads.parameters():
                param.requires_grad_(False)
        else:
            for param in self.regressor.heads.parameters():
                param.requires_grad_(True)
            self.regressor.heads.train()
            head_params.extend(self.regressor.heads.parameters())

        aux_params = []
        if train_fusion:
            aux_params.extend(self.fusion.parameters())
            aux_params.extend(self._glace_residual_adapter_params())
            aux_params.extend(self._local_residual_gate_params())
        aux_params.extend(self._ace_lmc_global_gate_params())

        seen = set()
        param_groups = []
        for name, params in (("head", head_params), ("fusion", aux_params)):
            unique = []
            for param in params:
                if param is None or not param.requires_grad or id(param) in seen:
                    continue
                seen.add(id(param))
                unique.append(param)
            if unique:
                param_groups.append({"name": name, "params": unique, "lr": lr})
        if not param_groups:
            raise RuntimeError("relative_depth_only optimizer has no trainable parameters.")

        optimizer = optim.AdamW(param_groups, lr=lr)
        self.optimizer_head = optimizer
        self.scheduler_head = None
        self.steps_per_s2_phase = steps
        self.local_s2_step = 0
        self.current_lmc_iter = int(getattr(self, "resume_start_iter", 0))
        self._s2_compressor_out = self._compress_memory()
        self._relative_depth_image_loader = self._build_relative_depth_image_dataloader()
        self._relative_depth_image_iterator = iter(self._relative_depth_image_loader)
        stage_tag = "S2-G" if self.lmc_flow == "ace_g" else "S2"
        log_interval = max(1, min(50, int(getattr(self, "iterations_output", 10))))
        updates = 0
        skipped = 0
        last_stats = None

        _logger.info(
            "[RelDepth-Only] steps=%d lr=%.2e train_fusion=%s start_ratio=%.3f weight=%.4f stage=%s",
            steps,
            lr,
            train_fusion,
            float(getattr(self.options, "relative_depth_start_ratio", 0.0) or 0.0),
            float(getattr(self.options, "relative_depth_loss_weight", 0.0) or 0.0),
            stage_tag,
        )
        self._log_optimizer_groups("RelDepth-Only", optimizer)

        try:
            for step in range(steps):
                self.local_s2_step = step
                try:
                    image_batch = next(self._relative_depth_image_iterator)
                except StopIteration:
                    self._relative_depth_image_iterator = iter(self._relative_depth_image_loader)
                    image_batch = next(self._relative_depth_image_iterator)

                loss, stats = self._compute_image_relative_depth_loss(
                    stage_tag=stage_tag,
                    image_batch=image_batch,
                )
                last_stats = stats
                if not bool(torch.isfinite(loss).all().item()) or not bool(getattr(loss, "requires_grad", False)):
                    optimizer.zero_grad(set_to_none=True)
                    skipped += 1
                    continue

                optimizer.zero_grad(set_to_none=True)
                self.scaler.scale(loss).backward()
                if self._s2_grad_clip_max_norm > 0:
                    self.scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        [param for group in optimizer.param_groups for param in group["params"]],
                        max_norm=self._s2_grad_clip_max_norm,
                    )
                self.scaler.step(optimizer)
                self.scaler.update()
                updates += 1
                self.iteration += 1

                if updates == 1 or updates % log_interval == 0 or step + 1 == steps:
                    _logger.info(
                        "[RelDepth-Only] step=%d/%d updates=%d loss=%.6f%s",
                        step + 1,
                        steps,
                        updates,
                        float(loss.detach().cpu().item()),
                        self._format_relative_depth_stats(stats),
                    )
        finally:
            self._s2_compressor_out = None

        candidate_path = self.options.output_map.parent / f"{self.options.output_map.stem}.relative_depth_only.pt"
        self.save_model(candidate_path)
        result_meta = {
            "source_checkpoint": str(self.resume_checkpoint_path),
            "candidate_checkpoint": str(candidate_path),
            "steps": steps,
            "updates": updates,
            "skipped": skipped,
            "learning_rate": lr,
            "train_fusion": train_fusion,
            "last_stats": last_stats,
        }
        try:
            eval_result = self._evaluate_checkpoint(
                candidate_path,
                int(getattr(self, "resume_start_iter", 0)),
                eval_stage="reldepth_only",
            )
            score = self._score_eval(eval_result)
            result_meta["score"] = score
            result_meta["eval"] = eval_result
            if score > self.best_score:
                import shutil

                shutil.copy2(candidate_path, self.options.output_map)
                self.best_score = score
                self.best_iter = int(getattr(self, "resume_start_iter", 0)) + 1
                self.best_eval = eval_result
                self._write_best_checkpoint_meta(self.best_iter, score, eval_result)
                result_meta["promoted_to_best"] = True
            else:
                result_meta["promoted_to_best"] = False
        except Exception as exc:
            _logger.warning("[RelDepth-Only] candidate evaluation failed: %s", exc, exc_info=True)
            result_meta["evaluation_error"] = str(exc)

        result_path = self.options.output_map.parent / "relative_depth_only_meta.json"
        with open(result_path, "w", encoding="utf-8") as handle:
            json.dump(result_meta, handle, indent=2, ensure_ascii=False)
        self._free_training_gpu_memory()
        _logger.info(
            "[RelDepth-Only] done updates=%d skipped=%d candidate=%s meta=%s",
            updates,
            skipped,
            candidate_path,
            result_path,
        )

    def train(self):
        """Route to the appropriate training flow."""
        if not self.use_lmc:
            if self.vanilla_iterations <= 1:
                return super().train()
            return self._train_vanilla_iterations()

        if int(getattr(self.options, "relative_depth_only_steps", 0) or 0) > 0:
            return self._train_relative_depth_only()

        _logger.info("[LMC] Training flow: %s", self.lmc_flow)
        if self.lmc_flow == 'iterative':
            return self._train_iterative()
        elif self.lmc_flow == 'ace_g':
            return self._train_ace_g()
        else:
            raise ValueError(f"Unknown lmc_flow={self.lmc_flow!r}")

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

    def _reset_s2_nan_guard(self):
        self._s2_nan_guard_bad_steps = 0
        self._s2_abort_current_iteration = False

    def _update_s2_nan_guard(self, stage_tag, fraction_valid, pxerr_naninf_count, batch_size):
        if not bool(getattr(self, '_s2_nan_guard_enabled', True)):
            return
        batch_size = max(1, int(batch_size))
        naninf_ratio = float(pxerr_naninf_count) / float(batch_size)
        bad_batch = (float(fraction_valid) <= 0.0) or (naninf_ratio >= self._s2_nan_guard_naninf_ratio)
        if bad_batch:
            self._s2_nan_guard_bad_steps += 1
        else:
            self._s2_nan_guard_bad_steps = 0
        if self._s2_nan_guard_bad_steps >= self._s2_nan_guard_patience:
            self._s2_abort_current_iteration = True
            self._abort_lmc_training = True
            _logger.error(
                "[%s] NaN guard triggered at step=%d: consecutive_bad=%d, valid=%.2f%%, naninf=%d/%d. "
                "Stopping before checkpoint/eval and keeping previous best checkpoint.",
                stage_tag,
                self.iteration,
                self._s2_nan_guard_bad_steps,
                float(fraction_valid) * 100.0,
                int(pxerr_naninf_count),
                batch_size,
            )

    def run_epoch(self):
        """Use actual buffer size (e.g. buffer_size_final on last iter); step iteration and S2 counters.
        When buffer is on CPU (buffer_on_cpu=True), each batch is moved to GPU here to avoid OOM.
        """
        if not self.use_lmc or self.optimizer_head is None:
            return super().run_epoch()
        if bool(getattr(self, '_abort_lmc_training', False)):
            return
        torch.backends.cudnn.benchmark = True
        buf = self.training_buffer
        schema_name = self._buffer_schema_name("raw_buffer") if self.lmc_flow == 'ace_g' else self._buffer_schema_name("fused_buffer")
        self._validate_training_buffer_schema(
            buffer_dict=buf,
            schema_name=schema_name,
            expected_size=buf['features'].shape[0],
        )
        buffer_len = buf['features'].shape[0]
        buf_device = buf['features'].device
        # Randperm on same device as buffer so indexing is cheap (no cross-device)
        random_indices = torch.randperm(
            buffer_len,
            generator=self._get_training_generator(buf_device),
            device=buf_device,
        )
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
            step_fn = self._training_step_ace_g if self.lmc_flow == 'ace_g' else self.training_step
            img_idx_batch = _to_dev(buf['img_idx'][random_batch_indices]) if 'img_idx' in buf else None
            stage_tag = "S2-G" if self.lmc_flow == 'ace_g' else "S2"
            relative_depth_image_batch = self._next_relative_depth_image_batch(stage_tag)
            step_fn(
                _to_dev(buf['features'][random_batch_indices]),
                _to_dev(buf['target_px'][random_batch_indices]),
                _to_dev(buf['gt_poses_inv'][random_batch_indices]),
                _to_dev(buf['intrinsics'][random_batch_indices]),
                _to_dev(buf['intrinsics_inv'][random_batch_indices]),
                _to_dev(buf['gt_scene_coords_world'][random_batch_indices]),
                _to_dev(buf['gt_scene_coords_valid'][random_batch_indices]),
                img_idx_batch,
                relative_depth_image_batch,
            )
            if bool(getattr(self, '_s2_abort_current_iteration', False)):
                break
            if not bool(getattr(self, "_s2_update_applied_last", True)):
                continue
            self.iteration += 1
            self.global_s2_step += 1
            self.local_s2_step += 1

    def training_step(self, features_bC, target_px_b2, gt_inv_poses_b34, Ks_b33, invKs_b33, gt_scene_coords_world_b3=None, gt_scene_coords_valid_b1=None, img_idx_b1=None, relative_depth_image_batch=None):
        """When LMC S2: use step_eff for ReproLoss and head-only optimizer/scheduler."""
        if not self.use_lmc or self.optimizer_head is None:
            return super().training_step(features_bC, target_px_b2, gt_inv_poses_b34, Ks_b33, invKs_b33)

        channels = features_bC.shape[1]
        features_bC, batch_size, h, w, trimmed = self._pack_feature_rows_for_head(
            "S2",
            features_bC,
            target_px_b2,
            gt_inv_poses_b34,
            Ks_b33,
            invKs_b33,
            gt_scene_coords_world_b3,
            gt_scene_coords_valid_b1,
        )
        if batch_size is None:
            self._s2_update_applied_last = False
            return None
        target_px_b2, gt_inv_poses_b34, Ks_b33, invKs_b33, gt_scene_coords_world_b3, gt_scene_coords_valid_b1 = trimmed
        img_idx_b1 = img_idx_b1[:batch_size] if img_idx_b1 is not None else None
        head_features_bC = self._mix_glace_decoder_features(None, features_bC, stage_tag="S2")
        local_head_features_bC = head_features_bC
        global_residual_loss = head_features_bC.new_zeros(())
        global_residual_stats = {"enabled": False}
        global_film_stats = {"enabled": False}
        global_gate_l1_stats = {"enabled": False}
        global_residual_gate_B1HW = None
        if self._uses_ace_lmc_concat_head():
            head_features_bC = self._append_ace_lmc_global_to_features(head_features_bC, img_idx_b1, stage_tag="S2")
        features_bCHW = head_features_bC.view(1, h, w, head_features_bC.shape[1]).permute(0, 3, 1, 2)

        if self.repro_step_mode == "global_monotonic":
            step_eff = self._monotonic_repro_step()
        elif self.repro_step_mode == "per_iter":
            # 每轮 S2 独立调度：local_s2_step 按比例映射到完整调度区间，保证每轮都经历完整 soft_clamp 范围
            step_eff = int(self.local_s2_step * self.repro_loss.total_iterations / max(1, self.steps_per_s2_phase))
            step_eff = min(step_eff, self.repro_loss.total_iterations - 1)
        else:
            # legacy_rewind: 每轮 S2 开始时小幅回拨，再随 local_s2_step 指数恢复至全局轨道
            rewind = self.s2_rewind_amount * math.exp(-self.local_s2_step / max(1e-6, self.s2_repro_rewind_tau))
            step_eff = max(0, self.global_s2_step - rewind)
            step_eff = min(step_eff, self.repro_loss.total_iterations - 1)

        with autocast("cuda", enabled=self.options.use_half):
            if self._uses_ace_lmc_global_residual_head():
                pred_scene_coords_b3HW, global_residual_loss, global_residual_stats, global_residual_gate_B1HW = self._predict_ace_lmc_global_residual_coords(
                    local_head_features_bC,
                    img_idx_b1,
                    h,
                    w,
                    stage_tag="S2",
                )
            elif self._uses_ace_lmc_global_film_head():
                pred_scene_coords_b3HW, global_film_stats = self._predict_ace_lmc_global_film_coords(
                    local_head_features_bC,
                    img_idx_b1,
                    h,
                    w,
                    stage_tag="S2",
                )
            else:
                pred_scene_coords_b3HW = self.regressor.get_scene_coordinates(features_bCHW)
        pred_scene_coords_b3HW = self._recover_pred_scene_to_training_world(pred_scene_coords_b3HW)

        pred_scene_coords_b31 = pred_scene_coords_b3HW.permute(0, 2, 3, 1).flatten(0, 2).unsqueeze(-1).float()
        contract = self._compute_reprojection_invalid_loss_contract(
            pred_scene_coords_b31,
            target_px_b2,
            gt_inv_poses_b34,
            Ks_b33,
            invKs_b33,
            step_eff=step_eff,
            normalizer=batch_size,
            invalid_max_delta=self._loss_invalid_max_delta,
            invalid_posinf=0.0,
            invalid_neginf=0.0,
        )
        loss = contract["loss"] + global_residual_loss
        global_gate_l1_loss, global_gate_l1_stats = self._compute_ace_lmc_global_gate_l1_loss()
        loss = loss + global_gate_l1_loss
        loss = loss + self._compute_c1_aux_ref_loss(
            pred_scene_coords_b3HW,
            gt_scene_coords_world_b3,
            gt_scene_coords_valid_b1,
        )
        relative_depth_loss, relative_depth_stats = self._compute_image_relative_depth_loss(
            stage_tag="S2",
            image_batch=relative_depth_image_batch,
        )
        loss = loss + relative_depth_loss
        consistency_loss, consistency_stats = self._compute_ace_lmc_stage2_consistency_loss(
            local_head_features_bC=local_head_features_bC,
            student_pred_scene_coords_b3HW=pred_scene_coords_b3HW,
        )
        loss = loss + consistency_loss
        reprojection_error_b1 = contract["reprojection_error_l1"]
        guard_loss, guard_stats = self._compute_ace_lmc_stage2_reprojection_guard_loss(
            local_head_features_bC=local_head_features_bC,
            student_pred_scene_coords_b3HW=pred_scene_coords_b3HW,
            student_reprojection_error_N1=reprojection_error_b1,
            target_px_N2=target_px_b2,
            gt_inv_poses_N34=gt_inv_poses_b34,
            Ks_N33=Ks_b33,
            invKs_N33=invKs_b33,
            step_eff=step_eff,
            normalizer=batch_size,
            student_gate_B1HW=global_residual_gate_B1HW,
        )
        loss = loss + guard_loss
        valid_mask_b1 = contract["valid_mask"]

        # Guard NaN/Inf: truly skip optimizer/scheduler updates to avoid consuming LR schedule.
        loss_is_finite = bool(torch.isfinite(loss).all().item())
        loss_for_log = float(loss.item()) if loss_is_finite else -1.0  # -1 means skipped update
        if not loss_is_finite:
            self.optimizer_head.zero_grad(set_to_none=True)
            self._s2_update_applied_last = False
            _logger.debug(
                "S2 step %d: non-finite loss (valid_frac=%.2f), skipping optimizer/scheduler step.",
                self.iteration, float(valid_mask_b1.sum() / batch_size),
            )
        elif not bool(getattr(loss, "requires_grad", False)):
            self.optimizer_head.zero_grad(set_to_none=True)
            if not getattr(self, "_logged_s2_no_grad_identity", False):
                _logger.info(
                    "S2 step %d: finite no-grad loss; treating as frozen identity sanity step without optimizer update.",
                    self.iteration,
                )
                self._logged_s2_no_grad_identity = True
            self._s2_update_applied_last = True
        else:
            self.optimizer_head.zero_grad(set_to_none=True)
            self.scaler.scale(loss).backward()
            # FIX: gradient clipping (same as S1) to prevent NaN divergence from large gradients.
            if self._s2_grad_clip_max_norm > 0:
                self.scaler.unscale_(self.optimizer_head)
                params_to_clip = (
                    list(self.regressor.heads.parameters())
                    + self._ace_lmc_global_residual_params()
                    + self._ace_lmc_global_gate_params()
                )
                torch.nn.utils.clip_grad_norm_(
                    params_to_clip,
                    max_norm=self._s2_grad_clip_max_norm,
                )
            self.scaler.step(self.optimizer_head)
            self.scaler.update()
            self.scheduler_head.step()
            self._s2_update_applied_last = True

        fraction_valid = float(valid_mask_b1.sum().item() / max(1, batch_size))
        finite_pxerr = torch.isfinite(reprojection_error_b1)
        pxerr_naninf_count = int((~finite_pxerr).sum().item())
        self._update_s2_nan_guard("S2", fraction_valid, pxerr_naninf_count, batch_size)

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
                'Iteration: {:6d} | S2 global={:6d} local={:6d} step_eff={:.0f} | Epoch {:03d}|{:03d}, Loss: {:.4f}, Valid: {:.1f}%, pxErr_finite: {:.2f}, pxerr_naninf: {:d}{}, Time: {:.2f}s'.format(
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
                    self._format_relative_depth_stats(relative_depth_stats) + self._format_ace_lmc_global_residual_stats(global_residual_stats) + self._format_ace_lmc_global_film_stats(global_film_stats) + self._format_ace_lmc_global_gate_l1_stats(global_gate_l1_stats) + self._format_ace_lmc_stage2_consistency_stats(consistency_stats) + self._format_ace_lmc_stage2_guard_stats(guard_stats),
                    time_since_start,
                )
            )
        return loss

    def _training_step_ace_g(self, features_bC, target_px_b2, gt_inv_poses_b34, Ks_b33, invKs_b33, gt_scene_coords_world_b3=None, gt_scene_coords_valid_b1=None, img_idx_b1=None, relative_depth_image_batch=None):
        """ACE-G S2 training step: apply fusion on-the-fly then head.

        Key difference from training_step(): raw backbone features are fused
        with compressed memory per-batch, rather than being pre-fused in buffer.
        """
        channels = features_bC.shape[1]
        features_bC, batch_size, h, w, trimmed = self._pack_feature_rows_for_head(
            "S2-G",
            features_bC,
            target_px_b2,
            gt_inv_poses_b34,
            Ks_b33,
            invKs_b33,
            gt_scene_coords_world_b3,
            gt_scene_coords_valid_b1,
        )
        if batch_size is None:
            self._s2_update_applied_last = False
            return None
        target_px_b2, gt_inv_poses_b34, Ks_b33, invKs_b33, gt_scene_coords_world_b3, gt_scene_coords_valid_b1 = trimmed
        img_idx_b1 = img_idx_b1[:batch_size] if img_idx_b1 is not None else None
        features_bCHW = features_bC.view(1, h, w, channels).permute(0, 3, 1, 2)

        # --- Same repro loss as training_step from here on ---
        if self.repro_step_mode == "global_monotonic":
            step_eff = self._monotonic_repro_step()
        elif self.repro_step_mode == "per_iter":
            step_eff = int(self.local_s2_step * self.repro_loss.total_iterations / max(1, self.steps_per_s2_phase))
            step_eff = min(step_eff, self.repro_loss.total_iterations - 1)
        else:
            rewind = self.s2_rewind_amount * math.exp(-self.local_s2_step / max(1e-6, self.s2_repro_rewind_tau))
            step_eff = max(0, self.global_s2_step - rewind)
            step_eff = min(step_eff, self.repro_loss.total_iterations - 1)

        ace_g_fusion_in_s2 = getattr(self.options, 'ace_g_fusion_in_s2', False)
        use_fused_buffer = self._ace_lmc_stage2_uses_fused_buffer()
        with autocast("cuda", enabled=self.options.use_half):
            # --- ACE-G core: raw buffer applies fusion on-the-fly; stage1_fused buffer is already fused. ---
            if use_fused_buffer:
                fused_bCHW = features_bCHW
                base_for_residual_bCHW = None
            elif ace_g_fusion_in_s2:
                # R2 path: fusion is trainable with slow LR
                fused_bCHW, base_for_residual_bCHW = self._fuse_lmc_features_for_head(
                    features_bCHW,
                    self._s2_compressor_out,
                    stage_tag="S2-G",
                )
            else:
                # R1 path: fusion frozen (default)
                with torch.no_grad():
                    fused_bCHW, base_for_residual_bCHW = self._fuse_lmc_features_for_head(
                        features_bCHW,
                        self._s2_compressor_out,
                        stage_tag="S2-G",
                    )
            fused_features_bC = fused_bCHW.permute(0, 2, 3, 1).reshape(-1, fused_bCHW.shape[1])
            base_features_bC = features_bCHW.permute(0, 2, 3, 1).reshape(-1, channels)
            if base_for_residual_bCHW is not None:
                base_for_residual_bC = base_for_residual_bCHW.permute(0, 2, 3, 1).reshape(-1, base_for_residual_bCHW.shape[1])
                head_features_bC = self._mix_glace_decoder_features(base_for_residual_bC, fused_features_bC, stage_tag="S2-G")
            else:
                head_features_bC = fused_features_bC
            local_head_features_bC = head_features_bC
            global_residual_loss = head_features_bC.new_zeros(())
            global_residual_stats = {"enabled": False}
            global_film_stats = {"enabled": False}
            global_gate_l1_stats = {"enabled": False}
            global_residual_gate_B1HW = None
            if self._uses_ace_lmc_global_residual_head():
                pred_scene_coords_b3HW, global_residual_loss, global_residual_stats, global_residual_gate_B1HW = self._predict_ace_lmc_global_residual_coords(
                    local_head_features_bC,
                    img_idx_b1,
                    h,
                    w,
                    stage_tag="S2-G",
                )
            elif self._uses_ace_lmc_global_film_head():
                pred_scene_coords_b3HW, global_film_stats = self._predict_ace_lmc_global_film_coords(
                    local_head_features_bC,
                    img_idx_b1,
                    h,
                    w,
                    stage_tag="S2-G",
                )
            else:
                if self._uses_ace_lmc_concat_head():
                    head_features_bC = self._append_ace_lmc_global_to_features(head_features_bC, img_idx_b1, stage_tag="S2-G")
                head_features_bCHW = head_features_bC.view(1, h, w, head_features_bC.shape[1]).permute(0, 3, 1, 2)
                pred_scene_coords_b3HW = self.regressor.get_scene_coordinates(head_features_bCHW)
        pred_scene_coords_b3HW = self._recover_pred_scene_to_training_world(pred_scene_coords_b3HW)

        pred_scene_coords_b31 = pred_scene_coords_b3HW.permute(0, 2, 3, 1).flatten(0, 2).unsqueeze(-1).float()
        contract = self._compute_reprojection_invalid_loss_contract(
            pred_scene_coords_b31,
            target_px_b2,
            gt_inv_poses_b34,
            Ks_b33,
            invKs_b33,
            step_eff=step_eff,
            normalizer=batch_size,
            invalid_max_delta=self._loss_invalid_max_delta,
            invalid_posinf=0.0,
            invalid_neginf=0.0,
        )
        loss = contract["loss"]
        global_gate_l1_loss, global_gate_l1_stats = self._compute_ace_lmc_global_gate_l1_loss()
        loss = loss + global_gate_l1_loss
        loss = loss + self._compute_c1_aux_ref_loss(
            pred_scene_coords_b3HW,
            gt_scene_coords_world_b3,
            gt_scene_coords_valid_b1,
        )
        relative_depth_loss, relative_depth_stats = self._compute_image_relative_depth_loss(
            stage_tag="S2-G",
            image_batch=relative_depth_image_batch,
        )
        loss = loss + relative_depth_loss
        consistency_loss, consistency_stats = self._compute_ace_lmc_stage2_consistency_loss(
            local_head_features_bC=local_head_features_bC,
            student_pred_scene_coords_b3HW=pred_scene_coords_b3HW,
        )
        loss = loss + consistency_loss
        reprojection_error_b1 = contract["reprojection_error_l1"]
        guard_loss, guard_stats = self._compute_ace_lmc_stage2_reprojection_guard_loss(
            local_head_features_bC=local_head_features_bC,
            student_pred_scene_coords_b3HW=pred_scene_coords_b3HW,
            student_reprojection_error_N1=reprojection_error_b1,
            target_px_N2=target_px_b2,
            gt_inv_poses_N34=gt_inv_poses_b34,
            Ks_N33=Ks_b33,
            invKs_N33=invKs_b33,
            step_eff=step_eff,
            normalizer=batch_size,
            student_gate_B1HW=global_residual_gate_B1HW,
        )
        loss = loss + guard_loss
        valid_mask_b1 = contract["valid_mask"]

        glace_pixel_diag = self._compute_glace_base_vs_fused_pixel_diag(
            features_bCHW=features_bCHW,
            fused_reprojection_error_b1=reprojection_error_b1,
            target_px_b2=target_px_b2,
            gt_inv_poses_b34=gt_inv_poses_b34,
            Ks_b33=Ks_b33,
            invKs_b33=invKs_b33,
            step_eff=step_eff,
            normalizer=batch_size,
        )
        self._log_glace_pixel_diag(glace_pixel_diag)

        glace_delta_ratio = None
        glace_delta_stats_scope = None
        glace_residual_gain = None
        glace_guard_loss = pred_scene_coords_b31.new_tensor(0.0)
        glace_guard_px = None
        glace_base_px = None
        if self._is_glace_backend():
            base_local, head_local, glace_delta_stats_scope = self._select_glace_delta_local_features(
                base_features_bC,
                head_features_bC,
            )
            base_local = base_local.float()
            head_local = head_local.float()
            glace_delta_ratio = float(
                (head_local - base_local).norm().detach().cpu().item()
                / max(base_local.norm().detach().cpu().item(), 1e-6)
            )
            if not getattr(self, '_logged_glace_delta_stats_scope', False):
                _logger.info(
                    "[GLACE-LMC] delta stats scope=%s base_dim=%d head_dim=%d ratio=%.6f",
                    glace_delta_stats_scope,
                    int(base_local.shape[1]),
                    int(head_local.shape[1]),
                    glace_delta_ratio,
                )
                self._logged_glace_delta_stats_scope = True
            if getattr(self, 'glace_residual_adapter', None) is not None:
                glace_residual_gain = float(
                    self.glace_residual_adapter.residual_gain().detach().float().cpu().item()
                )

        guard_weight = float(getattr(self, 'glace_antiregression_weight', 0.0) or 0.0) if self._is_glace_backend() else 0.0
        # Guard loss is computed against a frozen GLACE reference, not the
        # current trainable path, so regressions are visible immediately.
        if guard_weight > 0.0:
            with torch.no_grad():
                with autocast("cuda", enabled=self.options.use_half):
                    reference_regressor = getattr(self, "glace_reference_regressor", None)
                    if reference_regressor is None:
                        reference_regressor = self.regressor
                    base_pred_b3HW = reference_regressor.get_scene_coordinates(features_bCHW)
                base_pred_b3HW = self._recover_pred_scene_to_training_world(base_pred_b3HW)
                base_pred_b31 = base_pred_b3HW.permute(0, 2, 3, 1).flatten(0, 2).unsqueeze(-1).float()
                base_contract = self._compute_reprojection_invalid_loss_contract(
                    base_pred_b31,
                    target_px_b2,
                    gt_inv_poses_b34,
                    Ks_b33,
                    invKs_b33,
                    step_eff=step_eff,
                    normalizer=batch_size,
                    invalid_max_delta=self._loss_invalid_max_delta,
                    invalid_posinf=0.0,
                    invalid_neginf=0.0,
                )
            fused_err = reprojection_error_b1.float().flatten()
            base_err = base_contract["reprojection_error_l1"].detach().float().flatten()
            base_valid = base_contract["valid_mask"].detach().flatten().bool()
            guard_mask = torch.isfinite(fused_err) & torch.isfinite(base_err) & base_valid
            if guard_mask.any():
                margin = float(getattr(self, 'glace_antiregression_margin_px', 0.25))
                max_px = float(getattr(self, 'glace_antiregression_max_px', 100.0))
                guard_terms = Fnn.relu(fused_err[guard_mask] - base_err[guard_mask] - margin)
                if max_px > 0.0:
                    guard_terms = guard_terms.clamp(max=max_px)
                raw_guard = guard_terms.mean()
                glace_guard_loss = raw_guard * guard_weight
                loss = loss + glace_guard_loss
                glace_guard_px = float(raw_guard.detach().cpu().item())
                glace_base_px = float(base_err[guard_mask].mean().detach().cpu().item())
            else:
                glace_guard_px = 0.0
                glace_base_px = -1.0

        loss_is_finite = bool(torch.isfinite(loss).all().item())
        loss_for_log = float(loss.item()) if loss_is_finite else -1.0
        if not loss_is_finite:
            self.optimizer_head.zero_grad(set_to_none=True)
            self._s2_update_applied_last = False
            _logger.debug(
                "S2-G step %d: non-finite loss (valid_frac=%.2f), skipping optimizer/scheduler step.",
                self.iteration, float(valid_mask_b1.sum() / batch_size),
            )
        elif not bool(getattr(loss, "requires_grad", False)):
            self.optimizer_head.zero_grad(set_to_none=True)
            if not getattr(self, "_logged_s2g_no_grad_identity", False):
                _logger.info(
                    "S2-G step %d: finite no-grad loss; treating as frozen identity sanity step without optimizer update.",
                    self.iteration,
                )
                self._logged_s2g_no_grad_identity = True
            self._s2_update_applied_last = True
        else:
            self.optimizer_head.zero_grad(set_to_none=True)
            self.scaler.scale(loss).backward()
            # FIX: gradient clipping (same as S1) to prevent NaN divergence from large gradients.
            if self._s2_grad_clip_max_norm > 0:
                self.scaler.unscale_(self.optimizer_head)
                params_to_clip = list(self.regressor.heads.parameters()) + self._ace_lmc_global_residual_params()
                if ace_g_fusion_in_s2:
                    params_to_clip += list(self.fusion.parameters())
                    params_to_clip += self._glace_residual_adapter_params()
                    params_to_clip += self._local_residual_gate_params()
                params_to_clip += self._ace_lmc_global_gate_params()
                torch.nn.utils.clip_grad_norm_(params_to_clip, max_norm=self._s2_grad_clip_max_norm)
            self.scaler.step(self.optimizer_head)
            self.scaler.update()
            self.scheduler_head.step()
            self._s2_update_applied_last = True

        fraction_valid = float(valid_mask_b1.sum().item() / max(1, batch_size))
        finite_pxerr = torch.isfinite(reprojection_error_b1)
        pxerr_naninf_count = int((~finite_pxerr).sum().item())
        self._update_s2_nan_guard("S2-G", fraction_valid, pxerr_naninf_count, batch_size)

        if self.iteration % self.iterations_output == 0:
            time_since_start = time.time() - self.training_start
            fraction_valid = float(valid_mask_b1.sum() / batch_size)
            finite_pxerr = torch.isfinite(reprojection_error_b1)
            pxerr_naninf_count = int((~finite_pxerr).sum().item())
            if finite_pxerr.any():
                px_err_finite = float(reprojection_error_b1[finite_pxerr].mean().item())
                px_err_for_log = px_err_finite
            else:
                px_err_finite = float('nan')
                px_err_for_log = -1.0
            lr = float(self.optimizer_head.param_groups[0]["lr"])
            # Emit compact GLACE diagnostics so you can spot drift without
            # opening the tensor path.
            glace_diag_suffix = ""
            if self._is_glace_backend():
                glace_diag_suffix = (
                    f", gGain={glace_residual_gain if glace_residual_gain is not None else -1.0:.4f}"
                    f", dRatio={glace_delta_ratio if glace_delta_ratio is not None else -1.0:.5f}"
                    f", dScope={glace_delta_stats_scope or 'n/a'}"
                )
                if float(getattr(self, 'glace_antiregression_weight', 0.0) or 0.0) > 0.0:
                    glace_diag_suffix += (
                        f", guardPx={glace_guard_px if glace_guard_px is not None else -1.0:.3f}"
                        f", basePx={glace_base_px if glace_base_px is not None else -1.0:.2f}"
                        f", guardLoss={float(glace_guard_loss.detach().cpu().item()):.4f}"
                    )
            glace_diag_suffix += self._format_relative_depth_stats(relative_depth_stats)
            glace_diag_suffix += self._format_ace_lmc_global_residual_stats(global_residual_stats)
            glace_diag_suffix += self._format_ace_lmc_global_film_stats(global_film_stats)
            glace_diag_suffix += self._format_ace_lmc_global_gate_l1_stats(global_gate_l1_stats)
            glace_diag_suffix += self._format_ace_lmc_stage2_consistency_stats(consistency_stats)
            glace_diag_suffix += self._format_ace_lmc_stage2_guard_stats(guard_stats)
            self._append_step_log(
                iter_idx=self.current_lmc_iter,
                step=self.iteration,
                stage="S2-G",
                loss=loss_for_log,
                px_err=px_err_for_log,
                lr=lr,
                mode="reproj",
                med3d=-1.0,
            )
            _logger.info(
                'Iteration: {:6d} | S2-G global={:6d} local={:6d} step_eff={:.0f} | Epoch {:03d}|{:03d}, Loss: {:.4f}, Valid: {:.1f}%, pxErr: {:.2f}, naninf: {:d}{}, Time: {:.2f}s'.format(
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
                    glace_diag_suffix,
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
            'lmc_config': self._lmc_config_for_checkpoint(),
        }
        if self._is_glace_backend() and getattr(self, 'glace_residual_adapter', None) is not None:
            checkpoint['glace_residual_adapter_state_dict'] = self.glace_residual_adapter.state_dict()
        if self.local_residual_mode == 'learned_alpha':
            checkpoint['local_residual_gate_state_dict'] = {
                'local_residual_alpha_logit': self.local_residual_alpha_logit.detach().float().cpu(),
            }
        if self._uses_ace_lmc_global_head() and isinstance(getattr(self, 'ace_lmc_global_gate', None), torch.nn.Parameter):
            checkpoint['ace_lmc_global_gate_state_dict'] = {
                'ace_lmc_global_gate': self.ace_lmc_global_gate.detach().float().cpu(),
            }
        if self._uses_ace_lmc_global_residual_head() and getattr(self, 'ace_lmc_global_residual_head', None) is not None:
            checkpoint['ace_lmc_global_residual_state_dict'] = self.ace_lmc_global_residual_head.state_dict()
            base_head = self._ace_lmc_global_residual_base_head()
            if base_head is not None:
                checkpoint['ace_lmc_global_residual_base_head_state_dict'] = {
                    k: v.detach().half().cpu() for k, v in base_head.state_dict().items()
                }
        torch.save(checkpoint, output_path)
        _logger.info(f"Saved LMC checkpoint to: {output_path}")

    def _lmc_config_for_checkpoint(self):
        config = dict(self.lmc_config)
        key_mix_logits = getattr(self.compressor, "key_mix_logits", None)
        if key_mix_logits is not None:
            weights = torch.softmax(key_mix_logits.detach().float().cpu(), dim=0)
            config["lmc_key_mix_weights"] = weights.tolist()
        level_stats = getattr(self.compressor, "last_levelwise_runtime_stats", None)
        if isinstance(level_stats, dict):
            config["lmc_level_merge_weights"] = level_stats.get("lmc_level_merge_weights")
            config["lmc_level_gate_entropy"] = level_stats.get("lmc_level_gate_entropy")
            config["final_lmc_level_anchor_residual_gamma"] = level_stats.get(
                "lmc_level_anchor_residual_gamma"
            )
        level_anchor_gamma = getattr(self.compressor, "level_anchor_residual_gamma", None)
        if level_anchor_gamma is not None:
            config["final_lmc_level_anchor_residual_gamma"] = float(
                level_anchor_gamma.detach().float().cpu().item()
            )
        geo_bias_stats = getattr(self.compressor, "last_geo_bias_runtime_stats", None)
        if isinstance(geo_bias_stats, dict):
            config["final_geo_bias_rbf_alpha"] = geo_bias_stats.get("final_geo_bias_rbf_alpha")
            config["final_geo_bias_rbf_weights"] = geo_bias_stats.get("final_geo_bias_rbf_weights")
        pos_gate = getattr(getattr(self.compressor, "pe_encoder", None), "residual_gate", None)
        if pos_gate is not None:
            config["final_pos_fourier_residual_gate"] = float(pos_gate.detach().float().cpu().item())
        fusion_gamma = getattr(self.fusion, "fusion_assembly_gamma", None)
        if fusion_gamma is not None:
            config["final_lmc_fusion_assembly_gamma"] = float(fusion_gamma.detach().float().cpu().item())
        reread_gate_logit = getattr(self.fusion, "fusion_reread_gate_logit", None)
        if reread_gate_logit is not None:
            reread_gate_logit_value = float(reread_gate_logit.detach().float().cpu().item())
            config["final_lmc_fusion_reread_gate_logit"] = reread_gate_logit_value
            config["final_lmc_fusion_reread_gate"] = float(torch.sigmoid(
                reread_gate_logit.detach().float().cpu()
            ).item())
        reread_gamma_patch = getattr(self.fusion, "fusion_reread_gamma_patch", None)
        if reread_gamma_patch is not None:
            reread_gamma_patch = reread_gamma_patch.detach().float().cpu()
            config["final_lmc_fusion_reread_gamma_patch_mean"] = float(reread_gamma_patch.mean().item())
            config["final_lmc_fusion_reread_gamma_patch_absmax"] = float(
                reread_gamma_patch.abs().max().item()
            )
        reread_gamma_common = getattr(self.fusion, "fusion_reread_gamma_common", None)
        if reread_gamma_common is not None:
            reread_gamma_common = reread_gamma_common.detach().float().cpu()
            config["final_lmc_fusion_reread_gamma_common_mean"] = float(reread_gamma_common.mean().item())
            config["final_lmc_fusion_reread_gamma_common_absmax"] = float(
                reread_gamma_common.abs().max().item()
            )
        dual_memory_gamma_patch = getattr(self.fusion, "fusion_dual_memory_gamma_patch", None)
        if dual_memory_gamma_patch is not None:
            dual_memory_gamma_patch = dual_memory_gamma_patch.detach().float().cpu()
            config["final_lmc_fusion_dual_memory_gamma_patch_mean"] = float(
                dual_memory_gamma_patch.mean().item()
            )
            config["final_lmc_fusion_dual_memory_gamma_patch_absmax"] = float(
                dual_memory_gamma_patch.abs().max().item()
            )
        dual_memory_gamma_common = getattr(self.fusion, "fusion_dual_memory_gamma_common", None)
        if dual_memory_gamma_common is not None:
            dual_memory_gamma_common = dual_memory_gamma_common.detach().float().cpu()
            config["final_lmc_fusion_dual_memory_gamma_common_mean"] = float(
                dual_memory_gamma_common.mean().item()
            )
            config["final_lmc_fusion_dual_memory_gamma_common_absmax"] = float(
                dual_memory_gamma_common.abs().max().item()
            )
        coord_prior_gamma = getattr(self.fusion, "coord_prior_gamma", None)
        if coord_prior_gamma is not None:
            config["final_lmc_fusion_coord_prior_gamma"] = float(
                coord_prior_gamma.detach().float().cpu().item()
            )
        if self._is_glace_backend() and getattr(self, 'glace_residual_adapter', None) is not None:
            config['final_glace_residual_gain'] = float(self.glace_residual_adapter.residual_gain().detach().float().cpu().item())
        if self._is_glace_backend():
            config['final_local_residual_alpha'] = self._current_local_residual_alpha_value()
            if self.local_residual_mode == 'learned_alpha':
                config['final_local_residual_alpha_logit'] = float(self.local_residual_alpha_logit.detach().float().cpu().item())
        if self._uses_ace_lmc_global_film_head() and hasattr(self.regressor.heads, 'gate_value'):
            config['final_ace_lmc_global_gate'] = float(self.regressor.heads.gate_value())
            stats = getattr(self.regressor.heads, 'last_stats', {}) or {}
            config['final_ace_lmc_global_film_gamma_abs'] = stats.get('gamma_abs')
            config['final_ace_lmc_global_film_beta_abs'] = stats.get('beta_abs')
            config['final_ace_lmc_global_film_step_abs'] = stats.get('step_abs')
        elif self._uses_ace_lmc_global_head():
            config['final_ace_lmc_global_gate'] = self._current_ace_lmc_global_gate_value()
        if self._uses_ace_lmc_global_residual_head() and getattr(self, 'ace_lmc_global_residual_head', None) is not None:
            stats = getattr(self.ace_lmc_global_residual_head, 'last_stats', {}) or {}
            config['final_ace_lmc_global_residual_gate_mean'] = stats.get('gate_mean')
            config['final_ace_lmc_global_residual_gate_max'] = stats.get('gate_max')
            config['final_ace_lmc_global_residual_delta_l2'] = stats.get('delta_l2')
            config['final_ace_lmc_global_residual_gated_delta_l2'] = stats.get('gated_delta_l2')
        if self._is_glace_backend() and isinstance(getattr(self, '_last_glace_pixel_diag', None), dict):
            config['last_glace_pixel_diag'] = dict(self._last_glace_pixel_diag)
        return config
