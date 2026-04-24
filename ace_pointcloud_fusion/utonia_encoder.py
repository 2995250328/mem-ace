"""Utonia point feature extraction wrapper.

This module imports Utonia lazily so the main ACE training code can be imported
without requiring the Utonia environment. Use ``extract_pointcloud_features.py``
to build a reusable feature bank before ACE training.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np
import torch


def _ensure_utonia_importable(utonia_root: Optional[Path] = None):
    if utonia_root is not None:
        sys.path.insert(0, str(Path(utonia_root).resolve()))
    import utonia  # noqa: WPS433

    return utonia


def _move_tensors_to_device(point: Dict, device: torch.device) -> Dict:
    for key, value in list(point.items()):
        if isinstance(value, torch.Tensor):
            point[key] = value.to(device, non_blocking=True)
    return point


def _upcast_to_original_points(point, levels_to_concat: int = 2):
    """Mirror Utonia demo logic and map features back through GridSample inverse."""
    for _ in range(levels_to_concat):
        if "pooling_parent" not in point.keys():
            break
        parent = point.pop("pooling_parent")
        inverse = point.pop("pooling_inverse")
        parent.feat = torch.cat([parent.feat, point.feat[inverse]], dim=-1)
        point = parent

    while "pooling_parent" in point.keys():
        parent = point.pop("pooling_parent")
        inverse = point.pop("pooling_inverse")
        parent.feat = point.feat[inverse]
        point = parent

    feat = point.feat[point.inverse] if "inverse" in point.keys() else point.feat
    return point.coord, feat


@torch.inference_mode()
def encode_with_utonia(
    coord: np.ndarray,
    color: Optional[np.ndarray] = None,
    normal: Optional[np.ndarray] = None,
    *,
    utonia_root: Optional[Path] = Path("/home/xwh/project/Utonia"),
    model_name_or_path: str = "utonia",
    repo_id: str = "Pointcept/Utonia",
    device: str = "cuda",
    scale: float = 1.0,
    apply_z_positive: bool = True,
    normalize_coord: bool = False,
    levels_to_concat: int = 2,
    enable_flash: Optional[bool] = None,
) -> Dict[str, torch.Tensor]:
    """Encode a single scene point cloud with Utonia."""
    utonia = _ensure_utonia_importable(utonia_root)
    device_t = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")

    coord = np.asarray(coord, dtype=np.float32)
    if color is None:
        color = np.zeros_like(coord, dtype=np.float32)
    else:
        color = np.asarray(color, dtype=np.float32)
    if normal is None:
        normal = np.zeros_like(coord, dtype=np.float32)
    else:
        normal = np.asarray(normal, dtype=np.float32)

    point = {
        "coord": coord,
        "color": color,
        "normal": normal,
    }

    transform = utonia.transform.default(
        scale=scale,
        apply_z_positive=apply_z_positive,
        normalize_coord=normalize_coord,
    )
    point = transform(point)

    load_kwargs = {"repo_id": repo_id} if model_name_or_path == "utonia" else {}
    if enable_flash is False:
        load_kwargs["custom_config"] = {
            "enc_patch_size": [1024 for _ in range(5)],
            "enable_flash": False,
        }
    model = utonia.load(model_name_or_path, **load_kwargs).to(device_t)
    model.eval()

    point = _move_tensors_to_device(point, device_t)
    point = model(point)
    encoded_coord, encoded_feat = _upcast_to_original_points(point, levels_to_concat=levels_to_concat)

    return {
        "points": torch.as_tensor(coord, dtype=torch.float32),
        "features": encoded_feat.detach().float().cpu(),
        "encoded_points": encoded_coord.detach().float().cpu(),
        "colors": torch.as_tensor(color, dtype=torch.float32),
        "scene_center": torch.as_tensor(coord, dtype=torch.float32).mean(dim=0),
        "feature_dim": torch.tensor(int(encoded_feat.shape[-1]), dtype=torch.int64),
    }


def save_feature_bank(
    output_path: Path,
    feature_bank: Dict[str, torch.Tensor],
    *,
    metadata: Optional[Dict] = None,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(feature_bank)
    payload["metadata"] = metadata or {}
    torch.save(payload, output_path)


def load_feature_bank(path: Path, device: str | torch.device = "cpu") -> Dict[str, torch.Tensor]:
    try:
        bank = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        bank = torch.load(path, map_location=device)
    required = ("points", "features")
    missing = [key for key in required if key not in bank]
    if missing:
        raise ValueError(f"Point feature bank missing keys: {missing}")
    bank["points"] = torch.as_tensor(bank["points"], dtype=torch.float32, device=device)
    bank["features"] = torch.as_tensor(bank["features"], dtype=torch.float32, device=device)
    if "scene_center" not in bank or bank["scene_center"] is None:
        bank["scene_center"] = bank["points"].mean(dim=0)
    bank["scene_center"] = torch.as_tensor(bank["scene_center"], dtype=torch.float32, device=device)
    return bank
