"""Paper-facing ACE-G scene-compressed memory reference implementation."""

from .contracts import AnchoredSceneTokens, SceneMemory, TrackBatch
from .fusion import SingleMemoryFusion
from .geometry import (
    ReprojectionConfig,
    inter_frame_track_reprojection_loss,
    single_frame_reprojection_loss,
)
from .geolmc import GlobalGeoLMC
from .memory import load_scene_memory, make_scene_memory, save_scene_memory
from .model import MemoryEnhancedSCR, MinimalCoordinateHead
from .schedule import ACEGTwoStageController
from .stgs import STGSSidecar, load_stgs_sidecar, make_track_batch

__all__ = [
    "ACEGTwoStageController",
    "AnchoredSceneTokens",
    "GlobalGeoLMC",
    "MemoryEnhancedSCR",
    "MinimalCoordinateHead",
    "ReprojectionConfig",
    "STGSSidecar",
    "SceneMemory",
    "SingleMemoryFusion",
    "TrackBatch",
    "inter_frame_track_reprojection_loss",
    "load_scene_memory",
    "load_stgs_sidecar",
    "make_scene_memory",
    "make_track_batch",
    "save_scene_memory",
    "single_frame_reprojection_loss",
]
