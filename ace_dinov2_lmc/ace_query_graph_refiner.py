"""Query-side feature refinement after LMC memory fusion.

This module intentionally starts with only a per-node MLP control. It provides
the feature-residual insertion point and diagnostics needed before adding real
same-image graph message passing.
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn


class QueryGraphRefiner(nn.Module):
    """Feature residual refiner for scene-conditioned query patches."""

    VALID_MODES = {"none", "mlp_control"}

    def __init__(
        self,
        feature_dim: int,
        *,
        mode: str = "none",
        hidden_dim: int = 0,
        layerscale_init: float = 0.01,
        gate_init: float = -4.0,
    ) -> None:
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.mode = str(mode)
        self.hidden_dim = int(hidden_dim) if int(hidden_dim) > 0 else max(64, self.feature_dim // 2)
        self.layerscale_init = float(layerscale_init)
        self.gate_init = float(gate_init)

        if self.feature_dim <= 0:
            raise ValueError(f"feature_dim must be > 0, got {self.feature_dim}.")
        if self.mode not in self.VALID_MODES:
            raise ValueError(f"Unsupported query graph refine mode={self.mode!r}.")
        if self.layerscale_init < 0.0:
            raise ValueError(f"layerscale_init must be >= 0, got {self.layerscale_init}.")

        if self.mode == "none":
            self.norm = None
            self.mlp = None
            self.gate = None
            self.layerscale = None
            return

        self.norm = nn.LayerNorm(self.feature_dim)
        self.mlp = nn.Sequential(
            nn.Linear(self.feature_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.feature_dim),
        )
        self.gate = nn.Sequential(
            nn.Linear(self.feature_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, 1),
        )
        self.layerscale = nn.Parameter(
            torch.full((self.feature_dim,), self.layerscale_init, dtype=torch.float32)
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.mode == "none":
            return
        # Keep the initial module exactly near identity; the output projection
        # learns first before upstream hidden layers matter.
        final_delta = self.mlp[-1]
        nn.init.zeros_(final_delta.weight)
        nn.init.zeros_(final_delta.bias)
        final_gate = self.gate[-1]
        nn.init.zeros_(final_gate.weight)
        nn.init.constant_(final_gate.bias, self.gate_init)

    def forward(
        self,
        features_gpc: torch.Tensor,
        target_px_gp2: torch.Tensor | None = None,
        *,
        return_stats: bool = False,
    ):
        """Refine grouped query features.

        Args:
            features_gpc: [G, P, C] scene-conditioned feature rows.
            target_px_gp2: [G, P, 2] real image coordinates. The MLP control
                does not consume it, but the argument keeps the public contract
                aligned with future graph modes.
        """
        if features_gpc.dim() != 3:
            raise ValueError(f"features_gpc must be [G,P,C], got {tuple(features_gpc.shape)}.")
        if features_gpc.shape[-1] != self.feature_dim:
            raise ValueError(
                f"features_gpc has C={features_gpc.shape[-1]}, expected {self.feature_dim}."
            )
        if target_px_gp2 is not None and tuple(target_px_gp2.shape[:2]) != tuple(features_gpc.shape[:2]):
            raise ValueError(
                "target_px_gp2 must match [G,P] of features_gpc: "
                f"target={tuple(target_px_gp2.shape)} features={tuple(features_gpc.shape)}."
            )

        if self.mode == "none":
            if return_stats:
                return features_gpc, self.empty_stats(features_gpc), features_gpc.new_zeros(())
            return features_gpc

        x = self.norm(features_gpc)
        delta = self.mlp(x)
        gate = torch.sigmoid(self.gate(x))
        layerscale = self.layerscale.to(device=features_gpc.device, dtype=features_gpc.dtype)
        effective_update = gate * layerscale.view(1, 1, -1) * delta
        refined = features_gpc + effective_update
        reg_loss = effective_update.float().abs().mean()

        if not return_stats:
            return refined
        stats = self._summarize(delta, gate, effective_update)
        return refined, stats, reg_loss

    def refine_feature_map(self, features_bchw: torch.Tensor, *, return_stats: bool = False):
        """Apply the per-node refiner to a full feature map for evaluation."""
        if features_bchw.dim() != 4:
            raise ValueError(f"features_bchw must be [B,C,H,W], got {tuple(features_bchw.shape)}.")
        b, c, h, w = features_bchw.shape
        grouped = features_bchw.permute(0, 2, 3, 1).reshape(b, h * w, c)
        if return_stats:
            refined, stats, reg_loss = self(grouped, None, return_stats=True)
            out = refined.reshape(b, h, w, c).permute(0, 3, 1, 2)
            return out, stats, reg_loss
        refined = self(grouped, None, return_stats=False)
        return refined.reshape(b, h, w, c).permute(0, 3, 1, 2)

    def final_state_summary(self) -> Dict[str, float]:
        if self.mode == "none" or self.layerscale is None:
            return {
                "final_query_graph_layerscale_absmean": 0.0,
                "final_query_graph_gate_bias": 0.0,
            }
        gate_bias = self.gate[-1].bias.detach().float()
        return {
            "final_query_graph_layerscale_absmean": float(
                self.layerscale.detach().float().abs().mean().item()
            ),
            "final_query_graph_gate_bias": float(gate_bias.mean().item()),
        }

    @staticmethod
    def empty_stats(features: torch.Tensor) -> Dict[str, float]:
        return {
            "query_graph_delta_norm": 0.0,
            "query_graph_effective_update_norm": 0.0,
            "query_graph_gate_mean": 0.0,
            "query_graph_gate_min": 0.0,
            "query_graph_gate_max": 0.0,
            "query_graph_layerscale_absmean": 0.0,
        }

    def _summarize(
        self,
        delta: torch.Tensor,
        gate: torch.Tensor,
        effective_update: torch.Tensor,
    ) -> Dict[str, float]:
        with torch.no_grad():
            return {
                "query_graph_delta_norm": float(delta.detach().float().norm(dim=-1).mean().item()),
                "query_graph_effective_update_norm": float(
                    effective_update.detach().float().norm(dim=-1).mean().item()
                ),
                "query_graph_gate_mean": float(gate.detach().float().mean().item()),
                "query_graph_gate_min": float(gate.detach().float().min().item()),
                "query_graph_gate_max": float(gate.detach().float().max().item()),
                "query_graph_layerscale_absmean": float(
                    self.layerscale.detach().float().abs().mean().item()
                    if self.layerscale is not None else 0.0
                ),
            }
