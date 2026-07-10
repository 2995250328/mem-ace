"""ACE-G S1/S2-G module-state and scene-token caching contract."""

from dataclasses import dataclass

import torch

from .contracts import AnchoredSceneTokens, SceneMemory
from .model import MemoryEnhancedSCR


def _set_trainable(module: torch.nn.Module, trainable: bool) -> None:
    module.train(trainable)
    for parameter in module.parameters():
        parameter.requires_grad_(trainable)


@dataclass
class ACEGTwoStageController:
    model: MemoryEnhancedSCR
    cached_tokens: AnchoredSceneTokens | None = None

    def enter_stage1(self) -> None:
        """S1 jointly trains compressor, single fusion, and coordinate head."""
        _set_trainable(self.model.compressor, True)
        _set_trainable(self.model.fusion, True)
        _set_trainable(self.model.coordinate_head, True)
        self.cached_tokens = None

    @torch.no_grad()
    def cache_scene_tokens(self, memory: SceneMemory) -> AnchoredSceneTokens:
        self.model.compressor.eval()
        self.cached_tokens = self.model.encode_scene(memory).detach()
        return self.cached_tokens

    def enter_stage2_global(self, memory: SceneMemory) -> None:
        """S2-G freezes GeoLMC and trains fusion/head with cached tokens."""
        _set_trainable(self.model.compressor, False)
        _set_trainable(self.model.fusion, True)
        _set_trainable(self.model.coordinate_head, True)
        self.cache_scene_tokens(memory)

    def stage1_forward(
        self,
        query_feature_map: torch.Tensor,
        memory: SceneMemory,
    ) -> torch.Tensor:
        if self.cached_tokens is not None:
            raise RuntimeError("stage1_forward called while S2-G tokens are cached")
        return self.model(query_feature_map, memory)

    def stage2_global_forward(self, query_feature_map: torch.Tensor) -> torch.Tensor:
        if self.cached_tokens is None:
            raise RuntimeError("call enter_stage2_global before S2-G forward")
        return self.model.regress_coordinates(query_feature_map, self.cached_tokens)
