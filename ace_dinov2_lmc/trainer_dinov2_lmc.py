# trainer_dinov2_lmc.py
# Two-stage iterative trainer with Latent Memory Compression.
# Extends TrainerACEDINOv2 — when use_lmc=False, degrades to vanilla DINO ACE.
# Memory loading mirrors map-anything/tasks/ace/utils.py load_memory_features.

import gc
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
from skimage.transform import rotate, resize
from torch.amp import autocast
from torch.utils.data import DataLoader, sampler
from tqdm import tqdm

from ace_network_dinov2 import Regressor
from ace_util import to_homogeneous
from trainer_dinov2 import TrainerACEDINOv2, set_seed
from ace_compressor import GeoLMC
from ace_fusion import LMCFeatureFusion
from ace_loss import ReproLoss
from utils_lmc import (
    _normalize_scene_tag,
    estimate_memory_front_visibility,
    load_memory_features,
    preflight_memory_features,
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
    def _optimizer_contains_module_params(optimizer, module):
        if optimizer is None or module is None:
            return False
        module_param_ids = {id(p) for p in module.parameters()}
        for group in optimizer.param_groups:
            for param in group.get("params", []):
                if id(param) in module_param_ids:
                    return True
        return False

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
        if explicit_root is not None and str(explicit_root):
            explicit_root = Path(explicit_root)
            if explicit_root.name in {"gt_depth", "colmap_depth"}:
                return explicit_root
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
        depth = np.load(path).astype(np.float64, copy=False)
        if depth.ndim == 3:
            depth = np.squeeze(depth)
        if depth.ndim != 2:
            raise ValueError(f"Unsupported aux depth shape {depth.shape} in {path}")
        depth = np.where(np.isfinite(depth), depth, 0.0)
        return depth

    @staticmethod
    def _sample_patch_depth_nearest_valid(depth: np.ndarray, stride: int, coords_h: int, coords_w: int):
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
                if np.isfinite(center_depth) and center_depth > 0.0 and center_depth <= 1000.0:
                    patch_depth[gy, gx] = center_depth
                    patch_px[gy, gx] = cx
                    patch_py[gy, gx] = cy
                    patch_valid[gy, gx] = True
                    continue

                window = depth[y0:y1, x0:x1]
                valid = np.isfinite(window) & (window > 0.0) & (window <= 1000.0)
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

        stride = int(Regressor.OUTPUT_SUBSAMPLE)
        coords_h = math.ceil(image_h / stride)
        coords_w = math.ceil(image_w / stride)
        depth_patch, px, py, valid = TrainerACEDINOv2LMC._sample_patch_depth_nearest_valid(
            depth,
            stride,
            coords_h,
            coords_w,
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

        for real_idx, pose_file in enumerate(dataset.pose_files):
            ace_pose = np.loadtxt(pose_file).astype(np.float64)
            ace_k = np.loadtxt(dataset.calibration_files[real_idx]).astype(np.float64)
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

    def _attach_ace_aux_depth_dataset(self, dataset, train_root: Path, depth_dir: Path):
        if not depth_dir.exists():
            raise FileNotFoundError(
                f"[LMC] C1 aux_ref requested but aux depth dir is missing: {depth_dir}. "
                "Set --c1_aux_depth_root or provide ACE train/depth."
            )

        rgb_files = getattr(dataset, "rgb_files", [])
        depth_paths_by_index = self._build_aux_depth_map_from_scene_meta(dataset, depth_dir)
        if depth_paths_by_index is None:
            depth_files = {
                p.stem.replace("image-", ""): p
                for p in depth_dir.iterdir()
                if p.is_file() and p.suffix.lower() == ".npy"
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
            # Keep the original path for non-ACE edge cases.
            idx = int(idx)
            real_idx = int(ds.valid_file_indices[idx])
            rgb_path = Path(ds.rgb_files[real_idx])
            depth_path = depth_paths_by_index[real_idx] if real_idx < len(depth_paths_by_index) else None

            image = ds._load_image(real_idx)
            k = np.loadtxt(ds.calibration_files[real_idx])
            if k.size == 1:
                focal_length = float(k)
                centre_point = None
            elif k.shape == (3, 3):
                k = k.tolist()
                focal_length = [float(k[0][0]), float(k[1][1])]
                centre_point = [float(k[0][2]), float(k[1][2])]
            else:
                raise ValueError("Calibration file must contain either a 3x3 matrix or a single float.")

            image_height_rounded = ds._round_to_patch_size(image_height)
            h_scale = image_height_rounded / image.shape[0]
            if centre_point:
                centre_point = [centre_point[0] * h_scale, centre_point[1] * h_scale]
                focal_length = [focal_length[0] * h_scale, focal_length[1] * h_scale]
            else:
                focal_length *= h_scale

            image = ds._resize_image(image, image_height_rounded)
            current_width = image.size[0]
            target_width = ds.image_width if ds.image_width is not None else ds._round_to_patch_size(current_width)
            if target_width != current_width:
                image = TF.resize(image, (image_height_rounded, target_width))
                w_scale = target_width / current_width
                if centre_point:
                    centre_point[0] *= w_scale
                    focal_length[0] *= w_scale
                else:
                    focal_length *= w_scale

            image_hw = (image.size[1], image.size[0])
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

            image_mask = torch.ones((1, image.size[1], image.size[0]))
            image = ds.image_transform(image)
            pose = ds._load_pose(real_idx)

            if ds.augment:
                angle = random.uniform(-ds.aug_rotation, ds.aug_rotation)
                image = ds._rotate_image(image, angle, 1, "reflect")
                image_mask = ds._rotate_image(image_mask, angle, order=1, mode="constant")
                depth = rotate(depth, angle, order=0, mode="constant", preserve_range=True)

                angle_rad = angle * math.pi / 180.0
                pose_rot = torch.eye(4)
                pose_rot[0, 0] = math.cos(angle_rad)
                pose_rot[0, 1] = -math.sin(angle_rad)
                pose_rot[1, 0] = math.sin(angle_rad)
                pose_rot[1, 1] = math.cos(angle_rad)
                pose = torch.matmul(pose, pose_rot)

            coords = TrainerACEDINOv2LMC._depth_to_patch_scene_coords(
                depth,
                pose,
                image_hw=(image.shape[1], image.shape[2]),
                focal_length=focal_length,
                centre_point=centre_point,
            )

            if ds.use_half and torch.cuda.is_available():
                image = image.half()
            image_mask = image_mask > 0
            pose_inv = pose.inverse()

            intrinsics = torch.eye(3)
            if centre_point:
                intrinsics[0, 2] = centre_point[0]
                intrinsics[1, 2] = centre_point[1]
                intrinsics[0, 0] = focal_length[0]
                intrinsics[1, 1] = focal_length[1]
            else:
                intrinsics[0, 2] = image.shape[2] / 2
                intrinsics[1, 2] = image.shape[1] / 2
                intrinsics[0, 0] = focal_length
                intrinsics[1, 1] = focal_length

            return image, image_mask, pose, pose_inv, intrinsics, intrinsics.inverse(), coords, str(rgb_path)

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

    def _build_train_dataset(self, image_width=None, augment=False, aug_rotation=0, aug_scale_max=1.0, aug_scale_min=1.0):
        dataset = super()._build_train_dataset(
            image_width=image_width,
            augment=augment,
            aug_rotation=aug_rotation,
            aug_scale_max=aug_scale_max,
            aug_scale_min=aug_scale_min,
        )
        aux_ref_required = float(getattr(self.options, "c1_aux_ref_loss_weight", 0.0)) > 0.0
        valid_coord_sampling = bool(getattr(self.options, "buffer_sample_valid_coords", True))
        if (
            (aux_ref_required or valid_coord_sampling)
            and getattr(self.options, "data_backend", "ace") == "ace"
            and hasattr(dataset, "init")
        ):
            train_root = self._get_train_root()
            ace_depth_dir = train_root / "depth"
            if ace_depth_dir.exists():
                dataset.init = True
                dataset.sparse = False
                dataset.eye = False
                dataset.coord_files = sorted(ace_depth_dir.iterdir())
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
        return dataset

    def __init__(self, options):
        # Determine if LMC is active *before* parent __init__
        self.use_lmc = getattr(options, 'use_lmc', False)
        memory_path = getattr(options, 'memory_path', None)
        if memory_path is None:
            self.use_lmc = False

        # Parent builds: dataset, regressor, optimizer, scheduler, loss, buffer
        super().__init__(options)
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
        auto_mode = bool(getattr(options, "lmc_auto_mode_by_visibility", False))
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
        if lmc_feature_hierarchy_mode not in ('selected_key_concat_value', 'levelwise_latent_merge'):
            raise ValueError(f"Unsupported lmc_feature_hierarchy_mode={lmc_feature_hierarchy_mode!r}")
        lmc_level_merge_mode = str(getattr(options, 'lmc_level_merge_mode', 'softmax_gate'))
        lmc_level_merge_init = str(getattr(options, 'lmc_level_merge_init', 'uniform'))
        lmc_level_proj_shared = bool(getattr(options, 'lmc_level_proj_shared', False))
        lmc_level_cross_attn_shared = bool(getattr(options, 'lmc_level_cross_attn_shared', True))
        lmc_level_gate_entropy_weight = float(getattr(options, 'lmc_level_gate_entropy_weight', 0.0))
        lmc_level_token_gate = bool(getattr(options, 'lmc_level_token_gate', False))
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
        lmc_point_rope_base = float(getattr(options, 'lmc_point_rope_base', 10000.0))
        lmc_point_rope_axes = str(getattr(options, 'lmc_point_rope_axes', 'xyz_split'))
        lmc_point_rope_apply_to = str(getattr(options, 'lmc_point_rope_apply_to', 'qk'))
        lmc_geo_bias_crpb_dim = int(getattr(options, 'lmc_geo_bias_crpb_dim', 32))
        lmc_geo_bias_crpb_input = str(getattr(options, 'lmc_geo_bias_crpb_input', 'delta_dist_log'))
        lmc_geo_bias_crpb_radius = float(getattr(options, 'lmc_geo_bias_crpb_radius', 4.0))
        lmc_geo_bias_crpb_per_head = bool(getattr(options, 'lmc_geo_bias_crpb_per_head', False))
        lmc_geo_bias_crpb_zero_init = bool(getattr(options, 'lmc_geo_bias_crpb_zero_init', True))
        if lmc_feature_hierarchy_mode == 'levelwise_latent_merge':
            if lmc_mode not in ('global', 'local'):
                raise ValueError(
                    "levelwise_latent_merge currently supports only global/local LMC modes, "
                    f"got {lmc_mode!r}."
                )
            if lmc_level_merge_mode != 'softmax_gate':
                raise ValueError(f"Unsupported lmc_level_merge_mode={lmc_level_merge_mode!r}")
            if lmc_level_merge_init != 'uniform':
                raise ValueError(f"Unsupported lmc_level_merge_init={lmc_level_merge_init!r}")
            if lmc_level_gate_entropy_weight != 0.0:
                raise ValueError("lmc_level_gate_entropy_weight must be 0.0 for B3-lite.")
            if lmc_level_token_gate:
                raise ValueError("lmc_level_token_gate must be False for B3-lite.")
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

        backbone_feature_dim = getattr(self.regressor.encoder, 'feature_dim', 1024)
        _logger.info("[LMC] backbone_feature_dim=%d (from regressor.encoder)", backbone_feature_dim)

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

        self.lmc_config = {
            'use_lmc': True,
            'lmc_flow': str(getattr(options, 'lmc_flow', 'iterative')),
            'lmc_mode': lmc_mode,
            'requested_lmc_mode': requested_lmc_mode,
            'effective_lmc_mode': lmc_mode,
            'lmc_auto_mode_by_visibility': bool(getattr(options, 'lmc_auto_mode_by_visibility', False)),
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
            'lmc_level_merge_weights': None,
            'lmc_level_gate_entropy': None,
            'geo_bias_mode': lmc_geo_bias_mode,
            'geo_bias_rbf_scales': lmc_geo_bias_rbf_scales,
            'geo_bias_rbf_alpha_init': lmc_geo_bias_rbf_alpha_init,
            'geo_bias_rbf_learn_weights': lmc_geo_bias_rbf_learn_weights,
            'geo_bias_rbf_per_head': lmc_geo_bias_rbf_per_head,
            'final_geo_bias_rbf_alpha': None,
            'final_geo_bias_rbf_weights': None,
            'pos_encoding_mode': lmc_pos_encoding_mode,
            'pos_fourier_v2_scales': lmc_pos_fourier_v2_scales,
            'pos_fourier_coord_norm': lmc_pos_fourier_coord_norm,
            'pos_fourier_radius': lmc_pos_fourier_radius,
            'pos_fourier_learnable_scale': lmc_pos_fourier_learnable_scale,
            'pos_fourier_residual_gate_init': lmc_pos_fourier_residual_gate_init,
            'final_pos_fourier_residual_gate': None,
            'point_rope_coord_norm': lmc_point_rope_coord_norm,
            'point_rope_radius': lmc_point_rope_radius,
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
            'ace_g_fusion_in_s2': bool(getattr(options, 'ace_g_fusion_in_s2', False)),
            'backbone_feature_dim': backbone_feature_dim,
            'scale_token_dim': scale_token_dim,
            'memory_path': str(memory_path),
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

        # --- Build fusion (query=backbone 1024, memory=compressor output feature_dim) ---
        self.fusion = LMCFeatureFusion(
            feature_dim=backbone_feature_dim,
            mode=lmc_mode,
            query_feature_dim=backbone_feature_dim,
            memory_feature_dim=feature_dim,
            fusion_geometry_mode=lmc_fusion_geometry_mode,
            fusion_scene_scale=lmc_geometry_scene_scale,
            fusion_key_geo_init=lmc_fusion_key_geo_init,
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

        # Rebuild optimizer to include compressor + fusion params (used only when not in LMC; S1/S2 use their own)
        self._rebuild_optimizer()

        _logger.info(
            f"[LMC] mode={lmc_mode}, tokens={num_latent_tokens}, "
            f"attn_layers={num_attn_layers}, iterations={self.lmc_iterations}, "
            f"profile={self.lmc_profile}, repro_step_mode={self.repro_step_mode}, "
            f"repro_total_iterations={self.repro_loss.total_iterations}, "
            f"loss_invalid_max_delta={self._loss_invalid_max_delta}, "
            f"s2_grad_clip={self._s2_grad_clip_max_norm}"
        )

    def _resolve_buffer_schema_dim(self, dim_spec):
        if dim_spec == "feature_dim":
            return int(self.regressor.feature_dim)
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
                (effective_size, self.regressor.feature_dim),
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
                        features_BCHW = self.regressor.get_features(image_BCHW)

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
        _logger.info(
            "[%s] trainable params: compressor=%d fusion=%d head=%d encoder=%d",
            stage_tag,
            self._count_trainable_params(getattr(self, "compressor", None)),
            self._count_trainable_params(getattr(self, "fusion", None)),
            self._count_trainable_params(getattr(self.regressor, "heads", None)),
            self._count_trainable_params(getattr(self.regressor, "encoder", None)),
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
            "lmc_level_merge_weights",
            "lmc_level_gate_entropy",
            "geo_bias_mode",
            "geo_bias_rbf_scales",
            "geo_bias_rbf_alpha_init",
            "geo_bias_rbf_learn_weights",
            "geo_bias_rbf_per_head",
            "final_geo_bias_rbf_alpha",
            "final_geo_bias_rbf_weights",
            "pos_encoding_mode",
            "pos_fourier_v2_scales",
            "pos_fourier_coord_norm",
            "pos_fourier_radius",
            "pos_fourier_learnable_scale",
            "pos_fourier_residual_gate_init",
            "final_pos_fourier_residual_gate",
            "point_rope_coord_norm",
            "point_rope_radius",
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
        ]
        key_mix_logits = getattr(getattr(self, "compressor", None), "key_mix_logits", None)
        if key_mix_logits is not None:
            weights = torch.softmax(key_mix_logits.detach().float().cpu(), dim=0)
            self.lmc_config["lmc_key_mix_weights"] = weights.tolist()
        level_stats = getattr(getattr(self, "compressor", None), "last_levelwise_runtime_stats", None)
        if isinstance(level_stats, dict):
            self.lmc_config["lmc_level_merge_weights"] = level_stats.get("lmc_level_merge_weights")
            self.lmc_config["lmc_level_gate_entropy"] = level_stats.get("lmc_level_gate_entropy")
        geo_bias_stats = getattr(getattr(self, "compressor", None), "last_geo_bias_runtime_stats", None)
        if isinstance(geo_bias_stats, dict):
            self.lmc_config["final_geo_bias_rbf_alpha"] = geo_bias_stats.get("final_geo_bias_rbf_alpha")
            self.lmc_config["final_geo_bias_rbf_weights"] = geo_bias_stats.get("final_geo_bias_rbf_weights")
        pos_gate = getattr(getattr(getattr(self, "compressor", None), "pe_encoder", None), "residual_gate", None)
        if pos_gate is not None:
            self.lmc_config["final_pos_fourier_residual_gate"] = float(pos_gate.detach().float().cpu().item())
        meta = {key: self._tensor_to_config_value(self.lmc_config.get(key)) for key in keys}
        meta["lmc_flow"] = str(getattr(self.options, "lmc_flow", "iterative"))
        meta["lmc_auto_mode_by_visibility"] = bool(getattr(self.options, "lmc_auto_mode_by_visibility", False))
        meta["ace_g_fusion_in_s2"] = bool(getattr(self.options, "ace_g_fusion_in_s2", False))
        return meta

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
        _logger.info(
            "[LMC-Runtime][%s] call=%d entropy_mean=%.4f p10=%.4f p50=%.4f p90=%.4f "
            "effective_tokens=%.2f avg_max=%.4f usage_min=%.5f usage_max=%.5f top5=[%s] "
            "raw_norm=%.4f attn_out_norm=%.4f fused_norm=%.4f queries=%d tokens=%d "
            "fusion_mode=%s scene_scale=%.6f key_geo_scale=%.6f p_norm_std=%.4f "
            "p_norm_absmax=%.4f p_norm_finite=%s",
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
                _logger.info(
                    "[LMC-Runtime][%s] level_merge_weights=%s level_gate_entropy=%.4f "
                    "per_level_latent_norm_mean=%s per_level_latent_norm_std=%s "
                    "per_level_attention_entropy_mean=%s per_level_effective_memory_token_count=%s",
                    stage_tag,
                    weights_str,
                    float(level_stats.get("final_level_gate_entropy", 0.0)),
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
                    "[LMC-Runtime][%s] geo_bias_mode=%s rbf_alpha=%s rbf_weights=%s",
                    stage_tag,
                    str(self.lmc_config.get("geo_bias_mode", "legacy")),
                    str(geo_bias_stats.get("final_geo_bias_rbf_alpha")),
                    str(geo_bias_stats.get("final_geo_bias_rbf_weights")),
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
                schema_name="fused_buffer",
                expected_size=target_buf_size,
            )
        # Normalize poses for full-pipeline normalization
        self._normalize_buffer_poses()

    # ------------------------------------------------------------------
    # Override: create_training_buffer_ace_g (raw backbone features, no fusion)
    # ------------------------------------------------------------------

    def create_training_buffer_ace_g(self, buffer_size_override=None):
        """Fill buffer with raw backbone features (no fusion).

        ACE-G path: fusion is applied per-batch in S2 training_step.
        This directly calls the base class buffer fill without injecting fusion,
        then moves the buffer to CPU if buffer_on_cpu is True.
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
        )
        try:
            # Backbone should be in eval mode for deterministic feature extraction.
            self.regressor.eval()
            self._create_training_buffer_with_scene_coords()
        finally:
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
            _logger.info("[ACE-G] Buffer kept on CPU (raw backbone features, no fusion).")
        if self.training_buffer is not None:
            self._validate_training_buffer_schema(
                buffer_dict=self.training_buffer,
                schema_name="raw_buffer",
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
        s1_step=0,
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

        pred_scene_B3HW = self.regressor.get_scene_coordinates(fused_feats_BCHW)
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
        s1_step=0,
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

        features_bCHW = features_bC.view(1, h, w, channels).permute(0, 3, 1, 2)
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
            schema_name="raw_buffer",
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
        kept_buffer = self._slice_training_buffer_rows(previous_buffer, "raw_buffer", keep_indices)
        self.training_buffer = None
        del previous_buffer

        self.create_training_buffer_ace_g(buffer_size_override=refill_count)
        refill_buffer = self.training_buffer
        merged_buffer = self._merge_training_buffers(
            schema_name="raw_buffer",
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
        self.regressor.heads.train()
        self.regressor.encoder.eval()
        self._log_stage_trainability("S1-Buffer")

        if self.training_buffer is None or 'features' not in self.training_buffer:
            raise RuntimeError("[S1-Buffer] training_buffer is empty; call _prepare_s1_buffer_for_iteration first.")
        buffer_len = int(self.training_buffer['features'].shape[0])
        self._validate_training_buffer_schema(
            buffer_dict=self.training_buffer,
            schema_name="raw_buffer",
            expected_size=buffer_len,
        )
        if buffer_len < 16:
            raise RuntimeError(f"[S1-Buffer] training_buffer too small: {buffer_len}.")

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

            with autocast("cuda", enabled=self.options.use_half):
                comp_out = self.compressor(self.memory_dict)
                raw_features_bCHW = raw_features_bC.view(1, h, w, channels).permute(0, 3, 1, 2)
                fused_bCHW = self._fuse_features(raw_features_bCHW, comp_out, stage_tag="S1-Buffer")
            fused_features_bC = fused_bCHW.permute(0, 2, 3, 1).reshape(-1, channels)

            loss, s1_stats = self._s1_compute_loss_from_features(
                fused_features_bC.contiguous(),
                target_px_b2.contiguous(),
                gt_inv_poses_b34.contiguous(),
                Ks_b33.contiguous(),
                invKs_b33.contiguous(),
                s1_step=update_step,
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
        self.regressor.heads.train()

        # Backbone remains frozen / inference-only in S1.
        self.regressor.encoder.eval()
        self._log_stage_trainability("S1")

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

            with autocast("cuda", enabled=self.options.use_half):
                with torch.no_grad():
                    raw_feats = self.regressor.get_features(image_BCHW)

                comp_out = self.compressor(self.memory_dict)
                fused_feats = self._fuse_features(raw_feats, comp_out, stage_tag="S1-Online")

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
                        'gt_scene_coords_world': normalize_shape(gt_scene_coords_B3HW),
                        'gt_scene_coords_valid': normalize_shape(gt_scene_coords_valid_B1HW),
                    }

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
                        s1_step=update_step,
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
        if ace_g_fusion_in_s2:
            fusion_lr_ratio = float(getattr(self.options, 'ace_g_fusion_lr_ratio', 0.01))
            fusion_lr = head_lr * fusion_lr_ratio
            # S2 optimizer: only head + fusion. Compressor stays frozen during S2.
            # Reason: S1 trains compressor via reprojection loss; S2 trains head via
            # scene-coord loss. Allowing S2 to drift compressor (even at low LR)
            # introduces cross-objective conflict that accumulates across iterations,
            # causing S1 convergence to worsen in late iterations (iter 26+).
            self.optimizer_head = optim.AdamW([
                {'params': self.regressor.heads.parameters(), 'lr': head_lr},
                {'params': self.fusion.parameters(), 'lr': fusion_lr},
            ])
            _logger.info(
                "[S2-G] R2 active: fusion_lr=%.2e (ratio=%.4f of head_lr=%.2e)",
                fusion_lr, fusion_lr_ratio, head_lr,
            )
        else:
            self.optimizer_head = optim.AdamW(self.regressor.heads.parameters(), lr=head_lr)
        self._validate_s2_compressor_contract()

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
            max_lr=[head_lr, fusion_lr] if ace_g_fusion_in_s2 else head_lr,
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
        if ace_g_fusion_in_s2:
            fusion_lr = head_lr * self.s2_polish_fusion_lr_ratio
            self.optimizer_head = optim.AdamW([
                {'params': self.regressor.heads.parameters(), 'lr': head_lr},
                {'params': self.fusion.parameters(), 'lr': fusion_lr},
            ])
            max_lrs = [head_lr, fusion_lr]
            _logger.info(
                "[S2-Polish] iter=%d epochs=%d head_lr=%.2e fusion_lr=%.2e "
                "(ratio=%.4f, constant LR)",
                iteration_idx + 1,
                self.s2_polish_epochs,
                head_lr,
                fusion_lr,
                self.s2_polish_fusion_lr_ratio,
            )
        else:
            self.optimizer_head = optim.AdamW(self.regressor.heads.parameters(), lr=head_lr)
            max_lrs = [head_lr]
            _logger.info(
                "[S2-Polish] iter=%d epochs=%d head_lr=%.2e (constant LR)",
                iteration_idx + 1,
                self.s2_polish_epochs,
                head_lr,
            )

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

    def _write_train_header(self):
        with open(self.step_log_path, 'w', encoding='utf-8') as f:
            f.write("Timestamp   Iter      Step  Stage               Loss       PxErr          LR    3D_Med  Mode        \n")
        with open(self.training_log_path, 'w', encoding='utf-8') as f:
            f.write(
                "iter,s1_steps,s2_epochs,buffer_size,is_best,score,"
                "pct25_5,pct10_5,pct5,pct2,pct1,median_t_cm,median_r_deg,avg_time_ms,elapsed_s\n"
            )
        with open(self.eval_log_path, 'w', encoding='utf-8') as f:
            f.write("# Iteration evaluation log (aligns with post_train_eval.txt fields)\n")
            f.write(f"best_metric={self.best_metric}, keep_best_only={self.keep_best_only}\n")
            f.write(
                "# Each iter: median_rotation_deg, median_translation_cm, acc25/10/5/2/1cm%% "
                "same as post_train_eval; avg_time_ms=per-frame infer\n"
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
            eval_deterministic=getattr(self.options, 'eval_deterministic', False),
            dsacstar_seed=getattr(self.options, 'eval_dsacstar_seed', 1305),
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
        if self.best_metric == 'composite':
            pct5 = float(eval_result.get('pct5', 0.0))
            med_t = float(eval_result.get('median_tErr', 1e9))  # cm
            med_r = float(eval_result.get('median_rErr', 1e9))  # deg
            return pct5 - 2.0 * med_t - 2.0 * med_r
        if self.best_metric == 'rt_error':
            # Lower error is better; convert to a score where larger is better.
            med_t = float(eval_result.get('median_tErr', 1e9))  # cm
            med_r = float(eval_result.get('median_rErr', 1e9))  # deg
            return -(med_t + med_r)
        return float(eval_result.get('pct5', 0.0)) - 1e-3 * float(eval_result.get('median_tErr', 0.0)) - 1e-4 * float(eval_result.get('median_rErr', 0.0))

    def _log_iteration_summary(self, it, s1_steps, is_last, is_best, score, eval_result, elapsed_s):
        buffer_size = self.buffer_size_final if is_last else self.options.training_buffer_size
        if eval_result:
            pct25_5 = float(eval_result.get('pct25_5', 0.0))
            pct10_5 = float(eval_result.get('pct10_5', 0.0))
            pct5 = float(eval_result.get('pct5', 0.0))
            pct2 = float(eval_result.get('pct2', 0.0))
            pct1 = float(eval_result.get('pct1', 0.0))
            med_t = float(eval_result.get('median_tErr', 0.0))
            med_r = float(eval_result.get('median_rErr', 0.0))
            avg_ms = float(eval_result.get('avg_time', 0.0)) * 1000.0
        else:
            pct25_5 = pct10_5 = pct5 = pct2 = pct1 = med_t = med_r = avg_ms = 0.0
        with open(self.training_log_path, 'a', encoding='utf-8') as f:
            f.write(
                f"{it + 1},{s1_steps},{self.options.epochs},{buffer_size},{int(is_best)},"
                f"{score:.6f},{pct25_5:.4f},{pct10_5:.4f},{pct5:.4f},{pct2:.4f},{pct1:.4f},"
                f"{med_t:.4f},{med_r:.4f},{avg_ms:.2f},{elapsed_s:.2f}\n"
            )
        with open(self.eval_log_path, 'a', encoding='utf-8') as f:
            tag = "BEST" if is_best else "-"
            f.write(
                f"iter={it + 1:02d} tag={tag} score={score:.4f} "
                f"median_rotation_deg={med_r:.4f} median_translation_cm={med_t:.4f} "
                f"acc25_5={pct25_5:.2f} acc10_5={pct10_5:.2f} acc5={pct5:.2f} "
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
                    eval_result = self._evaluate_checkpoint(iter_ckpt, it)
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
    # Override: train (two-stage multi-iteration)
    # ------------------------------------------------------------------

    def _train_iterative(self):
        """Two-stage iterative training (S1+S2) with iteration-level eval and best-checkpoint policy."""
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
                        f"acc25_5=0 acc10_5=0 acc5=0 acc2=0 acc1=0 avg_time_ms=0 elapsed={elapsed_s:.1f}s\n"
                    )
                with open(self.training_log_path, "a", encoding="utf-8") as f:
                    f.write(
                        f"{it + 1},{s1_steps},{self.options.epochs},"
                        f"{self.training_buffer_size},0,-999999.000000,"
                        f"0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.00,{elapsed_s:.2f}\n"
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
        best_ckpt_exists = False

        ace_g_fusion_in_s2 = getattr(self.options, 'ace_g_fusion_in_s2', False)
        ace_g_cross_iter_eval = getattr(self.options, 'ace_g_cross_iter_eval', False)
        _logger.info("[ACE-G] R2 (fusion trainable in S2): %s", ace_g_fusion_in_s2)
        _logger.info("[ACE-G] Cross-iter eval: %s", ace_g_cross_iter_eval)

        prev_post_s2_score = None  # Tracks previous iteration post-S2 score
        prev_post_s2_head_state = None  # Snapshot of previous iteration head after S2

        for it in range(self.lmc_iterations):
            iter_start = time.time()
            is_last = (it == self.lmc_iterations - 1)
            _logger.info(f"\n{'='*60}")
            _logger.info(f"[ACE-G] Iteration {it+1}/{self.lmc_iterations}"
                         f"{' (FINAL)' if is_last else ''}")
            _logger.info(f"{'='*60}")

            # --- Stage 1: train compressor + fusion + head (same as iterative) ---
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
                        f"acc25_5=0 acc10_5=0 acc5=0 acc2=0 acc1=0 avg_time_ms=0 elapsed={elapsed_s:.1f}s\n"
                    )
                with open(self.training_log_path, "a", encoding="utf-8") as f:
                    f.write(
                        f"{it + 1},{s1_steps},{self.options.epochs},"
                        f"{self.training_buffer_size},0,-999999.000000,"
                        f"0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.00,{elapsed_s:.2f}\n"
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
                    cross_eval = self._evaluate_checkpoint(cross_ckpt, it)
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

            # 3. Fill buffer with raw backbone features (no fusion)
            buf_size = self.buffer_size_final if is_last else None
            _logger.info(f"[S2-G] Filling buffer with raw backbone features"
                         f" (size={'FINAL ' + str(self.buffer_size_final) if is_last else 'default'})")
            self.create_training_buffer_ace_g(buffer_size_override=buf_size)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # 4. Build S2 optimizer/schedule (includes fusion params if R2)
            self._setup_s2_optimizer_and_schedule(it, is_last)

            # 5. Train head (with on-the-fly fusion via run_epoch routing)
            _logger.info(f"[S2-G] Training head for {self.options.epochs} epochs")
            for self.epoch in range(self.options.epochs):
                self.run_epoch()
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
                    eval_result = self._evaluate_checkpoint(iter_ckpt, it)
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

    def train(self):
        """Route to the appropriate training flow."""
        if not self.use_lmc:
            if self.vanilla_iterations <= 1:
                return super().train()
            return self._train_vanilla_iterations()

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

    def run_epoch(self):
        """Use actual buffer size (e.g. buffer_size_final on last iter); step iteration and S2 counters.
        When buffer is on CPU (buffer_on_cpu=True), each batch is moved to GPU here to avoid OOM.
        """
        if not self.use_lmc or self.optimizer_head is None:
            return super().run_epoch()
        torch.backends.cudnn.benchmark = True
        buf = self.training_buffer
        schema_name = "raw_buffer" if self.lmc_flow == 'ace_g' else "fused_buffer"
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
            step_fn(
                _to_dev(buf['features'][random_batch_indices]),
                _to_dev(buf['target_px'][random_batch_indices]),
                _to_dev(buf['gt_poses_inv'][random_batch_indices]),
                _to_dev(buf['intrinsics'][random_batch_indices]),
                _to_dev(buf['intrinsics_inv'][random_batch_indices]),
                _to_dev(buf['gt_scene_coords_world'][random_batch_indices]),
                _to_dev(buf['gt_scene_coords_valid'][random_batch_indices]),
            )
            if not bool(getattr(self, "_s2_update_applied_last", True)):
                continue
            self.iteration += 1
            self.global_s2_step += 1
            self.local_s2_step += 1

    def training_step(self, features_bC, target_px_b2, gt_inv_poses_b34, Ks_b33, invKs_b33, gt_scene_coords_world_b3=None, gt_scene_coords_valid_b1=None):
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
        features_bCHW = features_bC.view(1, h, w, channels).permute(0, 3, 1, 2)

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
        loss = contract["loss"]
        loss = loss + self._compute_c1_aux_ref_loss(
            pred_scene_coords_b3HW,
            gt_scene_coords_world_b3,
            gt_scene_coords_valid_b1,
        )
        reprojection_error_b1 = contract["reprojection_error_l1"]
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
        else:
            self.optimizer_head.zero_grad(set_to_none=True)
            self.scaler.scale(loss).backward()
            # FIX: gradient clipping (same as S1) to prevent NaN divergence from large gradients.
            if self._s2_grad_clip_max_norm > 0:
                self.scaler.unscale_(self.optimizer_head)
                torch.nn.utils.clip_grad_norm_(
                    list(self.regressor.heads.parameters()),
                    max_norm=self._s2_grad_clip_max_norm,
                )
            self.scaler.step(self.optimizer_head)
            self.scaler.update()
            self.scheduler_head.step()
            self._s2_update_applied_last = True

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

    def _training_step_ace_g(self, features_bC, target_px_b2, gt_inv_poses_b34, Ks_b33, invKs_b33, gt_scene_coords_world_b3=None, gt_scene_coords_valid_b1=None):
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

        with autocast("cuda", enabled=self.options.use_half):
            # --- ACE-G core: apply fusion on-the-fly in S2 ---
            ace_g_fusion_in_s2 = getattr(self.options, 'ace_g_fusion_in_s2', False)
            if ace_g_fusion_in_s2:
                # R2 path: fusion is trainable with slow LR
                fused_bCHW = self._fuse_features(features_bCHW, self._s2_compressor_out, stage_tag="S2-G")
            else:
                # R1 path: fusion frozen (default)
                with torch.no_grad():
                    fused_bCHW = self._fuse_features(features_bCHW, self._s2_compressor_out, stage_tag="S2-G")
            pred_scene_coords_b3HW = self.regressor.get_scene_coordinates(fused_bCHW)
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
        loss = loss + self._compute_c1_aux_ref_loss(
            pred_scene_coords_b3HW,
            gt_scene_coords_world_b3,
            gt_scene_coords_valid_b1,
        )
        reprojection_error_b1 = contract["reprojection_error_l1"]
        valid_mask_b1 = contract["valid_mask"]

        loss_is_finite = bool(torch.isfinite(loss).all().item())
        loss_for_log = float(loss.item()) if loss_is_finite else -1.0
        if not loss_is_finite:
            self.optimizer_head.zero_grad(set_to_none=True)
            self._s2_update_applied_last = False
            _logger.debug(
                "S2-G step %d: non-finite loss (valid_frac=%.2f), skipping optimizer/scheduler step.",
                self.iteration, float(valid_mask_b1.sum() / batch_size),
            )
        else:
            self.optimizer_head.zero_grad(set_to_none=True)
            self.scaler.scale(loss).backward()
            # FIX: gradient clipping (same as S1) to prevent NaN divergence from large gradients.
            if self._s2_grad_clip_max_norm > 0:
                self.scaler.unscale_(self.optimizer_head)
                params_to_clip = list(self.regressor.heads.parameters())
                if ace_g_fusion_in_s2:
                    params_to_clip += list(self.fusion.parameters())
                torch.nn.utils.clip_grad_norm_(params_to_clip, max_norm=self._s2_grad_clip_max_norm)
            self.scaler.step(self.optimizer_head)
            self.scaler.update()
            self.scheduler_head.step()
            self._s2_update_applied_last = True

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
                'Iteration: {:6d} | S2-G global={:6d} local={:6d} step_eff={:.0f} | Epoch {:03d}|{:03d}, Loss: {:.4f}, Valid: {:.1f}%, pxErr: {:.2f}, naninf: {:d}, Time: {:.2f}s'.format(
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
            'lmc_config': self._lmc_config_for_checkpoint(),
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
        geo_bias_stats = getattr(self.compressor, "last_geo_bias_runtime_stats", None)
        if isinstance(geo_bias_stats, dict):
            config["final_geo_bias_rbf_alpha"] = geo_bias_stats.get("final_geo_bias_rbf_alpha")
            config["final_geo_bias_rbf_weights"] = geo_bias_stats.get("final_geo_bias_rbf_weights")
        pos_gate = getattr(getattr(self.compressor, "pe_encoder", None), "residual_gate", None)
        if pos_gate is not None:
            config["final_pos_fourier_residual_gate"] = float(pos_gate.detach().float().cpu().item())
        return config
