"""Memory-enhanced scene coordinate regression composition."""

import torch
from torch import nn

from .contracts import AnchoredSceneTokens, SceneMemory
from .fusion import SingleMemoryFusion
from .geolmc import GlobalGeoLMC


class MemoryEnhancedSCR(nn.Module):
    """Compose global compression, single fusion, and an SCR coordinate head."""

    def __init__(
        self,
        compressor: GlobalGeoLMC,
        fusion: SingleMemoryFusion,
        coordinate_head: nn.Module,
    ):
        super().__init__()
        self.compressor = compressor
        self.fusion = fusion
        self.coordinate_head = coordinate_head

    def encode_scene(self, memory: SceneMemory) -> AnchoredSceneTokens:
        return self.compressor(memory)

    def regress_coordinates(
        self,
        query_feature_map: torch.Tensor,
        scene_tokens: AnchoredSceneTokens,
    ) -> torch.Tensor:
        batch_size, channels, height, width = query_feature_map.shape
        query = query_feature_map.flatten(2).transpose(1, 2)
        fused = self.fusion(query, scene_tokens)
        fused_map = fused.transpose(1, 2).reshape(batch_size, -1, height, width)
        return self.coordinate_head(fused_map)

    def forward(
        self,
        query_feature_map: torch.Tensor,
        memory: SceneMemory,
    ) -> torch.Tensor:
        return self.regress_coordinates(query_feature_map, self.encode_scene(memory))


class MinimalCoordinateHead(nn.Module):
    """Small smoke-test head; replace with the ACE/DINO-ACE/GLACE SCR head."""

    def __init__(self, feature_dim: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_dim, 3, kernel_size=1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.layers(features)
