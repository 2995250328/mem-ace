"""Point-cloud loading helpers for ACE point-cloud fusion.

The Indoor6 COLMAP folders currently expose sparse reconstructions as
``sparse/0/points3D.bin``. This module also accepts common point-cloud exports
so the encoder path is not tied to COLMAP.
"""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Dict, Optional

import numpy as np


def _read_bytes(fid, num_bytes: int, fmt: str):
    data = fid.read(num_bytes)
    if len(data) != num_bytes:
        raise EOFError("Unexpected end of COLMAP binary file.")
    return struct.unpack("<" + fmt, data)


def load_colmap_points3d_bin(path: Path) -> Dict[str, np.ndarray]:
    """Load COLMAP ``points3D.bin`` into numpy arrays."""
    path = Path(path)
    xyz = []
    rgb = []
    error = []
    track_length = []

    with path.open("rb") as fid:
        (num_points,) = _read_bytes(fid, 8, "Q")
        for _ in range(num_points):
            _point_id = _read_bytes(fid, 8, "Q")[0]
            x, y, z = _read_bytes(fid, 24, "ddd")
            r, g, b = _read_bytes(fid, 3, "BBB")
            (err,) = _read_bytes(fid, 8, "d")
            (trk_len,) = _read_bytes(fid, 8, "Q")
            # Track elements are (image_id:int32, point2D_idx:int32).
            fid.seek(int(trk_len) * 8, 1)
            xyz.append((x, y, z))
            rgb.append((r, g, b))
            error.append(err)
            track_length.append(trk_len)

    return {
        "coord": np.asarray(xyz, dtype=np.float32),
        "color": np.asarray(rgb, dtype=np.float32) / 255.0,
        "error": np.asarray(error, dtype=np.float32),
        "track_length": np.asarray(track_length, dtype=np.int32),
    }


def _load_np_point_cloud(path: Path) -> Dict[str, np.ndarray]:
    loaded = np.load(path, allow_pickle=True)
    if isinstance(loaded, np.lib.npyio.NpzFile):
        data = {k: loaded[k] for k in loaded.files}
    elif loaded.shape == () and isinstance(loaded.item(), dict):
        data = loaded.item()
    else:
        data = {"coord": loaded}

    coord = data.get("coord", data.get("points", data.get("xyz")))
    if coord is None:
        raise ValueError(f"{path} does not contain coord/points/xyz.")
    color = data.get("color", data.get("rgb"))
    if color is None:
        color = np.zeros_like(coord, dtype=np.float32)
    color = np.asarray(color, dtype=np.float32)
    if color.max(initial=0.0) > 1.5:
        color = color / 255.0
    return {"coord": np.asarray(coord, dtype=np.float32), "color": color}


def _load_text_point_cloud(path: Path) -> Dict[str, np.ndarray]:
    arr = np.loadtxt(path).astype(np.float32)
    if arr.ndim != 2 or arr.shape[1] < 3:
        raise ValueError(f"Expected text cloud with at least 3 columns, got {arr.shape}.")
    coord = arr[:, :3]
    color = arr[:, 3:6] if arr.shape[1] >= 6 else np.zeros_like(coord)
    if color.max(initial=0.0) > 1.5:
        color = color / 255.0
    return {"coord": coord, "color": color.astype(np.float32)}


def apply_transform(coord: np.ndarray, transform_4x4: np.ndarray) -> np.ndarray:
    """Apply a homogeneous 4x4 transform to Nx3 coordinates."""
    transform_4x4 = np.asarray(transform_4x4, dtype=np.float32)
    if transform_4x4.shape != (4, 4):
        raise ValueError(f"Expected transform shape (4, 4), got {transform_4x4.shape}.")
    homo = np.concatenate([coord, np.ones((coord.shape[0], 1), dtype=coord.dtype)], axis=1)
    return (homo @ transform_4x4.T)[:, :3].astype(np.float32)


def voxel_downsample(
    coord: np.ndarray,
    color: Optional[np.ndarray] = None,
    voxel_size: Optional[float] = None,
    max_points: Optional[int] = None,
    seed: int = 2089,
) -> Dict[str, np.ndarray]:
    """Voxel downsample and optionally cap point count deterministically."""
    coord = np.asarray(coord, dtype=np.float32)
    if color is None:
        color = np.zeros_like(coord, dtype=np.float32)
    color = np.asarray(color, dtype=np.float32)

    if voxel_size is not None and voxel_size > 0:
        keys = np.floor(coord / float(voxel_size)).astype(np.int64)
        _, unique_idx = np.unique(keys, axis=0, return_index=True)
        unique_idx = np.sort(unique_idx)
        coord = coord[unique_idx]
        color = color[unique_idx]

    if max_points is not None and max_points > 0 and coord.shape[0] > max_points:
        rng = np.random.default_rng(seed)
        idx = rng.choice(coord.shape[0], size=max_points, replace=False)
        idx.sort()
        coord = coord[idx]
        color = color[idx]

    return {"coord": coord.astype(np.float32), "color": color.astype(np.float32)}


def load_scene_point_cloud(
    path: Path,
    transform_path: Optional[Path] = None,
    voxel_size: Optional[float] = None,
    max_points: Optional[int] = None,
    seed: int = 2089,
) -> Dict[str, np.ndarray]:
    """Load a scene cloud from COLMAP, numpy, or text formats."""
    path = Path(path)
    if path.is_dir():
        colmap_bin = path / "sparse" / "0" / "points3D.bin"
        if not colmap_bin.exists():
            raise FileNotFoundError(f"Directory has no sparse/0/points3D.bin: {path}")
        cloud = load_colmap_points3d_bin(colmap_bin)
    elif path.name == "points3D.bin":
        cloud = load_colmap_points3d_bin(path)
    elif path.suffix.lower() in {".npy", ".npz"}:
        cloud = _load_np_point_cloud(path)
    elif path.suffix.lower() in {".txt", ".xyz", ".pts"}:
        cloud = _load_text_point_cloud(path)
    else:
        raise ValueError(f"Unsupported point-cloud format: {path}")

    coord = cloud["coord"]
    if transform_path is not None:
        transform_path = Path(transform_path)
        transform = np.load(transform_path) if transform_path.suffix == ".npy" else np.loadtxt(transform_path)
        coord = apply_transform(coord, transform)

    sampled = voxel_downsample(coord, cloud.get("color"), voxel_size, max_points, seed)
    sampled["source_path"] = str(path)
    if transform_path is not None:
        sampled["transform_path"] = str(transform_path)
    return sampled
