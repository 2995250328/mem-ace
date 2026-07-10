"""Canonical pooled-memory I/O shared by DINO/MapAnything and ACE-FCN."""

from pathlib import Path

import torch

from .contracts import SceneMemory


def make_scene_memory(
    points_world: torch.Tensor,
    aligned_features: torch.Tensor,
    scene_center: torch.Tensor | None = None,
) -> SceneMemory:
    """Create memory after feature-to-COLMAP-point correspondence and pooling."""
    points_world = torch.as_tensor(points_world, dtype=torch.float32)
    aligned_features = torch.as_tensor(aligned_features, dtype=torch.float32)
    if scene_center is None:
        scene_center = points_world.mean(dim=-2)
    else:
        scene_center = torch.as_tensor(scene_center, dtype=torch.float32)
    return SceneMemory(points_world, aligned_features, scene_center).validate()


def load_scene_memory(path: str | Path, device: str | torch.device = "cpu") -> SceneMemory:
    """Load the canonical paper attachment format.

    Production memories should first be exported to the three canonical fields
    below so that reference-frame conversion remains explicit.
    """
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    missing = {"pooled_points", "pooled_features", "scene_center"} - payload.keys()
    if missing:
        raise KeyError(f"canonical scene memory is missing fields: {sorted(missing)}")
    return SceneMemory(
        points=torch.as_tensor(payload["pooled_points"], dtype=torch.float32, device=device),
        features=torch.as_tensor(payload["pooled_features"], dtype=torch.float32, device=device),
        scene_center=torch.as_tensor(payload["scene_center"], dtype=torch.float32, device=device),
    ).validate()


def save_scene_memory(memory: SceneMemory, path: str | Path) -> None:
    memory.validate()
    torch.save(
        {
            "schema_version": "aceg_pooled_scene_memory_v1",
            "coordinate_frame": "world",
            "pooled_points": memory.points.detach().cpu(),
            "pooled_features": memory.features.detach().cpu(),
            "scene_center": memory.scene_center.detach().cpu(),
        },
        path,
    )
