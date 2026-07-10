"""COLMAP track sidecar contract used by STGS and inter-frame supervision."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .contracts import TrackBatch


@dataclass(frozen=True)
class STGSSidecar:
    anchor_image_indices: torch.Tensor
    anchor_feature_yx: torch.Tensor
    anchor_pixels: torch.Tensor
    target_image_indices: torch.Tensor
    target_pixels: torch.Tensor
    track_point_ids: torch.Tensor
    track_points_world: torch.Tensor
    alignment_weights: torch.Tensor
    track_flags: torch.Tensor
    parallax_degrees: torch.Tensor
    colmap_reprojection_errors: torch.Tensor
    track_lengths: torch.Tensor

    @property
    def count(self) -> int:
        return int(self.anchor_image_indices.numel())

    def filtered(
        self,
        *,
        min_track_length: int = 0,
        max_colmap_reprojection_error: float = 0.0,
    ) -> "STGSSidecar":
        keep = self.track_flags.bool()
        if min_track_length > 0:
            keep &= self.track_lengths >= min_track_length
        if max_colmap_reprojection_error > 0:
            keep &= self.colmap_reprojection_errors <= max_colmap_reprojection_error
        values = {
            name: getattr(self, name)[keep]
            for name in self.__dataclass_fields__
        }
        return STGSSidecar(**values)


def load_stgs_sidecar(
    path: str | Path,
    *,
    target_mode: str = "patch_center",
) -> STGSSidecar:
    if target_mode not in {"patch_center", "exact_target"}:
        raise ValueError("target_mode must be 'patch_center' or 'exact_target'")
    with np.load(path, allow_pickle=False) as payload:
        target_key = "target_patch_center_xy" if target_mode == "patch_center" else "target_xy_model"
        return STGSSidecar(
            anchor_image_indices=torch.from_numpy(payload["anchor_image_idx"].astype(np.int64)),
            anchor_feature_yx=torch.from_numpy(payload["anchor_feature_yx"].astype(np.int64)),
            anchor_pixels=torch.from_numpy(payload["anchor_patch_center_xy"].astype(np.float32)),
            target_image_indices=torch.from_numpy(payload["target_image_idx"].astype(np.int64)),
            target_pixels=torch.from_numpy(payload[target_key].astype(np.float32)),
            track_point_ids=torch.from_numpy(payload["point3D_id"].astype(np.int64)),
            track_points_world=torch.from_numpy(payload["track_xyz_world"].astype(np.float32)),
            alignment_weights=torch.from_numpy(payload["alignment_weight"].astype(np.float32)),
            track_flags=torch.from_numpy(payload["track_flag"].astype(bool)),
            parallax_degrees=torch.from_numpy(payload["parallax_deg"].astype(np.float32)),
            colmap_reprojection_errors=torch.from_numpy(payload["colmap_reproj_error"].astype(np.float32)),
            track_lengths=torch.from_numpy(payload["track_length"].astype(np.int32)),
        )


def make_track_batch(
    sidecar: STGSSidecar,
    row_indices: torch.Tensor,
    camera_world_to_camera: torch.Tensor,
    camera_intrinsics: torch.Tensor,
    device: str | torch.device,
) -> TrackBatch:
    """Resolve target camera indices after track-guided row sampling."""
    row_indices = row_indices.long().cpu()
    target_indices = sidecar.target_image_indices[row_indices].long()
    pose_indices = target_indices.to(camera_world_to_camera.device)
    intrinsics_indices = target_indices.to(camera_intrinsics.device)
    return TrackBatch(
        target_pixels=sidecar.target_pixels[row_indices].to(device),
        target_world_to_camera=camera_world_to_camera[pose_indices].to(device),
        target_intrinsics=camera_intrinsics[intrinsics_indices].to(device),
        valid=sidecar.track_flags[row_indices].to(device),
        weights=sidecar.alignment_weights[row_indices].to(device),
    )
