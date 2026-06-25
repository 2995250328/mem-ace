"""Relative-depth distillation for ACE scene-coordinate training.

The primary path supervises a full image-grid prediction independently from
the sampled ACE training buffer. Student depth is camera-space z recovered from
the predicted scene-coordinate map; the frozen teacher supplies relative depth
for the same image.
"""

from __future__ import annotations

import logging
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Callable

sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ace_util import to_homogeneous


_logger = logging.getLogger(__name__)


_DEPTH_ANYTHING_CONFIGS = {
    "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
    "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
    "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
    "vitg": {"encoder": "vitg", "features": 384, "out_channels": [1536, 1536, 1536, 1536]},
}


class SampledRelativeDepthLoss(nn.Module):
    """Per-image relative-depth loss for unordered sampled pixels."""

    def __init__(
        self,
        *,
        pair_weight: float = 0.5,
        max_samples: int = 1024,
        max_pairs: int = 4096,
        min_points: int = 16,
        depth_min: float = 1e-3,
        depth_max: float = 1000.0,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.pair_weight = float(pair_weight)
        self.max_samples = int(max_samples)
        self.max_pairs = int(max_pairs)
        self.min_points = int(min_points)
        self.depth_min = float(depth_min)
        self.depth_max = float(depth_max)
        self.eps = float(eps)

    def _robust_normalize(self, values: torch.Tensor) -> torch.Tensor:
        center = values.median()
        scale = (values - center).abs().mean().clamp_min(self.eps)
        return (values - center) / scale

    def forward(
        self,
        student_depth: torch.Tensor,
        teacher_depth: torch.Tensor,
        pixel_xy: torch.Tensor,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        student_depth = student_depth.float().view(-1)
        teacher_depth = teacher_depth.detach().float().view(-1)
        pixel_xy = pixel_xy.float().view(-1, 2)
        valid = (
            torch.isfinite(student_depth)
            & torch.isfinite(teacher_depth)
            & (student_depth > self.depth_min)
            & (student_depth < self.depth_max)
            & (teacher_depth > self.depth_min)
        )
        if int(valid.sum().item()) < self.min_points:
            zero = student_depth.new_zeros(())
            return zero, {"valid_points": float(valid.sum().item()), "ssi": 0.0, "pair": 0.0}

        student = student_depth[valid]
        teacher = teacher_depth[valid]
        coords = pixel_xy[valid]
        if student.numel() > self.max_samples:
            perm = torch.randperm(student.numel(), device=student.device, generator=generator)[: self.max_samples]
            student = student[perm]
            teacher = teacher[perm]
            coords = coords[perm]

        student_n = self._robust_normalize(student)
        teacher_n = self._robust_normalize(teacher)
        ssi_loss = F.smooth_l1_loss(student_n, teacher_n, beta=0.5)

        pair_loss = student.new_zeros(())
        if self.pair_weight > 0 and student_n.numel() >= 2 and self.max_pairs > 0:
            n = int(student_n.numel())
            pair_count = min(self.max_pairs, n * max(1, n - 1))
            idx_a = torch.randint(0, n, (pair_count,), device=student.device, generator=generator)
            idx_b = torch.randint(0, n, (pair_count,), device=student.device, generator=generator)
            keep = idx_a != idx_b
            if bool(keep.any().item()):
                idx_a = idx_a[keep]
                idx_b = idx_b[keep]
                diff_student = student_n[idx_a] - student_n[idx_b]
                diff_teacher = teacher_n[idx_a] - teacher_n[idx_b]
                # Mildly emphasize local image-space relations without requiring a dense grid.
                spatial_dist = (coords[idx_a] - coords[idx_b]).norm(dim=-1).clamp_min(1.0)
                spatial_weight = (1.0 / spatial_dist.sqrt()).detach()
                pair_terms = F.smooth_l1_loss(diff_student, diff_teacher, beta=0.5, reduction="none")
                pair_loss = (pair_terms * spatial_weight).sum() / spatial_weight.sum().clamp_min(self.eps)

        total = (1.0 - self.pair_weight) * ssi_loss + self.pair_weight * pair_loss
        return total, {
            "valid_points": float(student.numel()),
            "ssi": float(ssi_loss.detach().cpu().item()),
            "pair": float(pair_loss.detach().cpu().item()),
        }


class ImageRelativeDepthLoss(nn.Module):
    """Scale/shift-invariant relative-depth loss on complete image grids."""

    def __init__(
        self,
        *,
        pair_weight: float = 0.5,
        max_samples: int = 1024,
        max_pairs: int = 4096,
        min_points: int = 16,
        depth_min: float = 1e-3,
        depth_max: float = 1000.0,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.pair_weight = float(pair_weight)
        self.max_samples = int(max_samples)
        self.max_pairs = int(max_pairs)
        self.min_points = int(min_points)
        self.depth_min = float(depth_min)
        self.depth_max = float(depth_max)
        self.eps = float(eps)

    def _normalize(self, values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        center = values.median()
        scale = (values - center).abs().mean().clamp_min(self.eps)
        return (values - center) / scale, center, scale

    def forward(
        self,
        student_depth_BHW: torch.Tensor,
        teacher_depth_BHW: torch.Tensor,
        valid_mask_BHW: torch.Tensor,
        generator: torch.Generator | None = None,
        ray_weight_BHW: torch.Tensor | None = None,
        student_space: str = "neg_z_legacy",
    ) -> tuple[torch.Tensor, dict[str, float]]:
        if student_space not in {"neg_z_legacy", "inverse_depth"}:
            raise ValueError(f"Unsupported relative-depth student_space={student_space!r}")

        losses = []
        ssi_values = []
        grad_values = []
        pair_values = []
        valid_points = 0

        # First-step plumbing only: soft ray weights are accepted and detached so
        # callers can log/ablate them without changing the legacy default loss.
        if ray_weight_BHW is not None:
            ray_weight_BHW = ray_weight_BHW.detach().float().view_as(student_depth_BHW)

        for student_hw, teacher_hw, mask_hw in zip(
            student_depth_BHW.float(),
            teacher_depth_BHW.detach().float(),
            valid_mask_BHW.bool(),
        ):
            valid = (
                mask_hw
                & torch.isfinite(student_hw)
                & torch.isfinite(teacher_hw)
                & (student_hw > self.depth_min)
                & (student_hw < self.depth_max)
                & (teacher_hw > self.eps)
            )
            count = int(valid.sum().item())
            if count < self.min_points:
                continue

            if student_space == "inverse_depth":
                student_source_hw = 1.0 / student_hw.clamp(min=self.depth_min, max=self.depth_max)
            else:
                student_source_hw = student_hw

            student_vec = student_source_hw[valid]
            teacher_vec = teacher_hw[valid]
            student_n, student_center, student_scale = self._normalize(student_vec)
            teacher_n, teacher_center, teacher_scale = self._normalize(teacher_vec)

            if student_space == "neg_z_legacy":
                # Depth Anything predicts relative proximity (larger means nearer),
                # while camera-space z grows with distance. This is the legacy
                # behavior and remains the default for exact rollback.
                student_rel = -student_n
                student_map = -(student_source_hw - student_center) / student_scale
            else:
                student_rel = student_n
                student_map = (student_source_hw - student_center) / student_scale
            teacher_map = (teacher_hw - teacher_center) / teacher_scale
            ssi_loss = F.smooth_l1_loss(student_rel, teacher_n, beta=0.5)

            valid_x = valid[:, 1:] & valid[:, :-1]
            valid_y = valid[1:, :] & valid[:-1, :]
            grad_terms = []
            if bool(valid_x.any().item()):
                grad_terms.append(
                    (student_map[:, 1:] - student_map[:, :-1]
                     - teacher_map[:, 1:] + teacher_map[:, :-1]).abs()[valid_x].mean()
                )
            if bool(valid_y.any().item()):
                grad_terms.append(
                    (student_map[1:, :] - student_map[:-1, :]
                     - teacher_map[1:, :] + teacher_map[:-1, :]).abs()[valid_y].mean()
                )
            grad_loss = torch.stack(grad_terms).mean() if grad_terms else student_vec.new_zeros(())

            pair_loss = student_vec.new_zeros(())
            if self.pair_weight > 0.0 and self.max_pairs > 0:
                coords = torch.nonzero(valid, as_tuple=False)
                if coords.shape[0] > self.max_samples:
                    perm = torch.randperm(coords.shape[0], device=coords.device, generator=generator)[:self.max_samples]
                    coords = coords[perm]
                n = int(coords.shape[0])
                if n >= 2:
                    pair_count = min(self.max_pairs, n * (n - 1))
                    idx_a = torch.randint(0, n, (pair_count,), device=coords.device, generator=generator)
                    idx_b = torch.randint(0, n, (pair_count,), device=coords.device, generator=generator)
                    keep = idx_a != idx_b
                    if bool(keep.any().item()):
                        idx_a = idx_a[keep]
                        idx_b = idx_b[keep]
                        ys, xs = coords[:, 0], coords[:, 1]
                        student_samples = student_map[ys, xs]
                        teacher_samples = teacher_map[ys, xs]
                        pair_loss = F.smooth_l1_loss(
                            student_samples[idx_a] - student_samples[idx_b],
                            teacher_samples[idx_a] - teacher_samples[idx_b],
                            beta=0.5,
                        )

            image_loss = (1.0 - self.pair_weight) * (ssi_loss + grad_loss) + self.pair_weight * pair_loss
            losses.append(image_loss)
            valid_points += count
            ssi_values.append(float(ssi_loss.detach().cpu().item()))
            grad_values.append(float(grad_loss.detach().cpu().item()))
            pair_values.append(float(pair_loss.detach().cpu().item()))

        if not losses:
            zero = student_depth_BHW.new_zeros(())
            return zero, {"valid_points": 0.0, "groups": 0.0, "ssi": 0.0, "grad": 0.0, "pair": 0.0}
        loss = torch.stack(losses).mean()
        return loss, {
            "valid_points": float(valid_points),
            "groups": float(len(losses)),
            "ssi": float(np.mean(ssi_values)),
            "grad": float(np.mean(grad_values)),
            "pair": float(np.mean(pair_values)),
        }


class RelativeDepthDistiller(nn.Module):
    """Teacher-depth provider plus image-level relative-depth loss."""

    def __init__(
        self,
        *,
        teacher: str,
        teacher_checkpoint: Path | None,
        teacher_encoder: str,
        teacher_input_size: int,
        image_height: int,
        image_path_getter: Callable[[int], Path | None],
        device: torch.device,
        pair_weight: float,
        max_samples: int,
        max_pairs: int,
        min_points: int,
        depth_min: float,
        depth_max: float,
        cache_size: int,
    ) -> None:
        super().__init__()
        self.teacher = str(teacher)
        self.teacher_checkpoint = Path(teacher_checkpoint) if teacher_checkpoint else None
        self.teacher_encoder = str(teacher_encoder)
        self.teacher_input_size = int(teacher_input_size)
        self.image_height = int(image_height)
        self.image_path_getter = image_path_getter
        self.device = device
        self.cache_size = max(0, int(cache_size))
        self._depth_cache: OrderedDict[int, torch.Tensor] = OrderedDict()
        self.loss_fn = SampledRelativeDepthLoss(
            pair_weight=pair_weight,
            max_samples=max_samples,
            max_pairs=max_pairs,
            min_points=min_points,
            depth_min=depth_min,
            depth_max=depth_max,
        )
        self.image_loss_fn = ImageRelativeDepthLoss(
            pair_weight=pair_weight,
            max_samples=max_samples,
            max_pairs=max_pairs,
            min_points=min_points,
            depth_min=depth_min,
            depth_max=depth_max,
        )
        self.teacher_model = None
        if self.teacher == "depth_anything_v2_online":
            self._init_depth_anything_v2()
        elif self.teacher == "none":
            pass
        else:
            raise ValueError(f"Unsupported relative depth teacher: {teacher!r}")

    def _init_depth_anything_v2(self) -> None:
        if self.teacher_encoder not in _DEPTH_ANYTHING_CONFIGS:
            raise ValueError(
                f"Unsupported Depth Anything V2 encoder {self.teacher_encoder!r}; "
                f"expected one of {sorted(_DEPTH_ANYTHING_CONFIGS)}"
            )
        if self.teacher_checkpoint is None or not self.teacher_checkpoint.exists():
            raise FileNotFoundError(
                "--relative_depth_teacher_checkpoint is required for depth_anything_v2_online "
                f"and was not found: {self.teacher_checkpoint}"
            )
        from depth_anything_v2.dpt import DepthAnythingV2

        model = DepthAnythingV2(**_DEPTH_ANYTHING_CONFIGS[self.teacher_encoder])
        state = torch.load(self.teacher_checkpoint, map_location="cpu")
        model.load_state_dict(state)
        model = model.to(self.device).eval()
        for param in model.parameters():
            param.requires_grad_(False)
        self.teacher_model = model
        _logger.info(
            "[RelDepth] Loaded Depth Anything V2 teacher: encoder=%s checkpoint=%s input_size=%d cache=%d",
            self.teacher_encoder,
            self.teacher_checkpoint,
            self.teacher_input_size,
            self.cache_size,
        )

    @staticmethod
    def _resize_bgr_to_height(image_bgr: np.ndarray, image_height: int) -> np.ndarray:
        h, w = image_bgr.shape[:2]
        if h == image_height:
            return image_bgr
        new_w = max(1, int(round(float(w) * float(image_height) / max(1.0, float(h)))))
        return cv2.resize(image_bgr, (new_w, image_height), interpolation=cv2.INTER_AREA)

    def _load_teacher_depth(self, image_idx: int) -> torch.Tensor | None:
        if image_idx in self._depth_cache:
            depth = self._depth_cache.pop(image_idx)
            self._depth_cache[image_idx] = depth
            return depth.to(self.device, non_blocking=True).float()

        image_path = self.image_path_getter(image_idx)
        if image_path is None or not image_path.exists():
            return None
        import cv2

        image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            return None
        image_bgr = self._resize_bgr_to_height(image_bgr, self.image_height)

        if self.teacher == "depth_anything_v2_online":
            with torch.no_grad():
                depth_np = self.teacher_model.infer_image(image_bgr, self.teacher_input_size)
            depth = torch.from_numpy(np.asarray(depth_np, dtype=np.float32)).cpu()
        else:
            return None

        if self.cache_size > 0:
            self._depth_cache[image_idx] = depth.half()
            while len(self._depth_cache) > self.cache_size:
                self._depth_cache.popitem(last=False)
        return depth.to(self.device, non_blocking=True).float()

    @staticmethod
    def _sample_depth_at_pixels(depth_hw: torch.Tensor, pixel_xy_n2: torch.Tensor) -> torch.Tensor:
        h, w = depth_hw.shape[-2:]
        xy = pixel_xy_n2.float()
        x = xy[:, 0].clamp(0, max(w - 1, 0))
        y = xy[:, 1].clamp(0, max(h - 1, 0))
        if w > 1:
            gx = x / float(w - 1) * 2.0 - 1.0
        else:
            gx = torch.zeros_like(x)
        if h > 1:
            gy = y / float(h - 1) * 2.0 - 1.0
        else:
            gy = torch.zeros_like(y)
        grid = torch.stack((gx, gy), dim=-1).view(1, -1, 1, 2)
        sampled = F.grid_sample(
            depth_hw.view(1, 1, h, w),
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )
        return sampled.view(-1)

    def forward_image(
        self,
        *,
        student_depth_BHW: torch.Tensor,
        valid_mask_B1HW: torch.Tensor,
        img_idx_B: torch.Tensor,
        generator: torch.Generator | None = None,
        ray_valid_mask_BHW: torch.Tensor | None = None,
        ray_weight_BHW: torch.Tensor | None = None,
        student_space: str = "neg_z_legacy",
        min_ray_coverage: float = 0.0,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        B, H, W = student_depth_BHW.shape
        teacher_depths = []
        teacher_valid = []
        missing = 0
        for image_idx_tensor in img_idx_B.detach().view(-1).cpu():
            depth_hw = self._load_teacher_depth(int(image_idx_tensor.item()))
            if depth_hw is None:
                teacher_depths.append(student_depth_BHW.new_zeros((H, W)))
                teacher_valid.append(student_depth_BHW.new_zeros((H, W), dtype=torch.bool))
                missing += 1
                continue
            resized = F.interpolate(
                depth_hw.view(1, 1, *depth_hw.shape[-2:]),
                size=(H, W),
                mode="bilinear",
                align_corners=True,
            )[0, 0]
            teacher_depths.append(resized)
            teacher_valid.append(torch.isfinite(resized) & (resized > 0))

        teacher_BHW = torch.stack(teacher_depths, dim=0)
        image_valid_BHW = valid_mask_B1HW.bool().view(B, H, W) & torch.stack(teacher_valid, dim=0)
        valid_BHW = image_valid_BHW
        image_valid_points = int(image_valid_BHW.sum().item())
        ray_valid_points = image_valid_points
        gate_skipped = 0

        if ray_valid_mask_BHW is not None:
            ray_valid_BHW = ray_valid_mask_BHW.detach().to(device=student_depth_BHW.device, dtype=torch.bool).view(B, H, W)
            gated_BHW = image_valid_BHW & ray_valid_BHW
            min_cov = max(0.0, float(min_ray_coverage))
            if min_cov > 0.0:
                gated_items = []
                for image_valid_hw, gated_hw in zip(image_valid_BHW, gated_BHW):
                    denom = int(image_valid_hw.sum().item())
                    numer = int(gated_hw.sum().item())
                    if denom > 0 and float(numer) / float(denom) < min_cov:
                        gated_items.append(torch.zeros_like(gated_hw))
                        gate_skipped += 1
                    else:
                        gated_items.append(gated_hw)
                gated_BHW = torch.stack(gated_items, dim=0)
            valid_BHW = gated_BHW
            ray_valid_points = int(valid_BHW.sum().item())

        ray_weight_mean = 0.0
        if ray_weight_BHW is not None:
            ray_weight_BHW = ray_weight_BHW.detach().to(device=student_depth_BHW.device, dtype=torch.float32).view(B, H, W)
            if image_valid_points > 0:
                ray_weight_mean = float(ray_weight_BHW[image_valid_BHW].mean().detach().cpu().item())

        loss, stats = self.image_loss_fn(
            student_depth_BHW,
            teacher_BHW,
            valid_BHW,
            generator=generator,
            ray_weight_BHW=ray_weight_BHW,
            student_space=student_space,
        )
        stats = dict(stats)
        pixel_count = max(1, B * H * W)
        ray_coverage = float(ray_valid_points) / float(max(1, image_valid_points)) if image_valid_points > 0 else 0.0
        stats.update({
            "enabled": 1.0,
            "missing_images": float(missing),
            "loss_raw": float(loss.detach().cpu().item()),
            "image_valid_points": float(image_valid_points),
            "image_coverage": float(image_valid_points) / float(pixel_count),
            "ray_valid_points": float(ray_valid_points),
            "ray_coverage": ray_coverage,
            "gate_skipped": float(gate_skipped),
            "ray_weight_mean": ray_weight_mean,
        })
        return loss, stats


    def forward(
        self,
        *,
        pred_scene_coords_N31: torch.Tensor,
        target_px_N2: torch.Tensor,
        gt_inv_poses_N34: torch.Tensor,
        img_idx_N: torch.Tensor | None,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        if img_idx_N is None:
            zero = pred_scene_coords_N31.new_zeros(())
            return zero, {"enabled": 0.0, "valid_points": 0.0, "groups": 0.0, "loss_raw": 0.0}

        pred_scene_coords_N31 = pred_scene_coords_N31.float()
        gt_inv_poses_N34 = gt_inv_poses_N34.float()
        target_px_N2 = target_px_N2.float()
        img_idx_N = img_idx_N.long().view(-1)
        pred_cam_N31 = torch.bmm(gt_inv_poses_N34, to_homogeneous(pred_scene_coords_N31))
        student_depth_N = pred_cam_N31[:, 2, 0]

        losses: list[torch.Tensor] = []
        valid_points = 0.0
        ssi_vals = []
        pair_vals = []
        groups = 0
        missing = 0
        for image_idx_tensor in torch.unique(img_idx_N.detach()).cpu():
            image_idx = int(image_idx_tensor.item())
            mask = img_idx_N == image_idx
            if int(mask.sum().item()) < self.loss_fn.min_points:
                continue
            depth_hw = self._load_teacher_depth(image_idx)
            if depth_hw is None:
                missing += 1
                continue
            teacher_depth = self._sample_depth_at_pixels(depth_hw, target_px_N2[mask])
            loss_i, stats_i = self.loss_fn(
                student_depth_N[mask],
                teacher_depth,
                target_px_N2[mask],
                generator=generator,
            )
            if torch.isfinite(loss_i):
                losses.append(loss_i)
                valid_points += stats_i["valid_points"]
                ssi_vals.append(stats_i["ssi"])
                pair_vals.append(stats_i["pair"])
                groups += 1

        if not losses:
            zero = pred_scene_coords_N31.new_zeros(())
            return zero, {
                "enabled": 1.0,
                "valid_points": valid_points,
                "groups": float(groups),
                "missing_images": float(missing),
                "loss_raw": 0.0,
                "ssi": 0.0,
                "pair": 0.0,
            }
        loss = torch.stack(losses).mean()
        return loss, {
            "enabled": 1.0,
            "valid_points": valid_points,
            "groups": float(groups),
            "missing_images": float(missing),
            "loss_raw": float(loss.detach().cpu().item()),
            "ssi": float(np.mean(ssi_vals)) if ssi_vals else 0.0,
            "pair": float(np.mean(pair_vals)) if pair_vals else 0.0,
        }
