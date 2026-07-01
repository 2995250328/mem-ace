"""Query-side feature refinement after LMC memory fusion.

This module intentionally keeps the first query-side refinement controls simple.
The modes here validate the feature-residual insertion point and diagnostics
before adding real same-image graph message passing.
"""

from __future__ import annotations

import math
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class QueryGraphRefiner(nn.Module):
    """Feature residual refiner for scene-conditioned query patches."""

    VALID_MODES = {"none", "identity_control", "mlp_control", "safe_mlp_control"}

    def __init__(
        self,
        feature_dim: int,
        *,
        mode: str = "none",
        hidden_dim: int = 0,
        layerscale_init: float = 0.01,
        gate_init: float = -4.0,
        gate_max: float = 0.05,
        layerscale_max: float = 0.05,
        update_norm_cap: float = 0.0,
        update_norm_cap_ratio: float = 0.005,
        anchor_weight: float = 0.0,
    ) -> None:
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.mode = str(mode)
        self.hidden_dim = int(hidden_dim) if int(hidden_dim) > 0 else max(64, self.feature_dim // 2)
        self.layerscale_init = float(layerscale_init)
        self.gate_init = float(gate_init)
        self.gate_max = float(gate_max)
        self.layerscale_max = float(layerscale_max)
        self.update_norm_cap = float(update_norm_cap)
        self.update_norm_cap_ratio = float(update_norm_cap_ratio)
        self.anchor_weight = float(anchor_weight)

        if self.feature_dim <= 0:
            raise ValueError(f"feature_dim must be > 0, got {self.feature_dim}.")
        if self.mode not in self.VALID_MODES:
            raise ValueError(f"Unsupported query graph refine mode={self.mode!r}.")
        if self.layerscale_init < 0.0:
            raise ValueError(f"layerscale_init must be >= 0, got {self.layerscale_init}.")
        for name, value in (
            ("gate_max", self.gate_max),
            ("layerscale_max", self.layerscale_max),
            ("update_norm_cap", self.update_norm_cap),
            ("update_norm_cap_ratio", self.update_norm_cap_ratio),
            ("anchor_weight", self.anchor_weight),
        ):
            if value < 0.0:
                raise ValueError(f"{name} must be >= 0, got {value}.")

        self.norm = None
        self.mlp = None
        self.gate = None
        self.layerscale = None
        self.layerscale_raw = None
        self.identity_dummy = None

        if self.mode == "none":
            return

        if self.mode == "identity_control":
            self.identity_dummy = nn.Parameter(torch.zeros(()))
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
        if self.mode == "mlp_control":
            # Keep the legacy parameter name and unbounded behavior so existing
            # mlp_control checkpoints load strictly and defaults stay unchanged.
            self.layerscale = nn.Parameter(
                torch.full((self.feature_dim,), self.layerscale_init, dtype=torch.float32)
            )
        elif self.mode == "safe_mlp_control":
            self.layerscale_raw = nn.Parameter(
                torch.full((self.feature_dim,), self._bounded_layerscale_init_logit(), dtype=torch.float32)
            )
        self.reset_parameters()

    def _bounded_layerscale_init_logit(self) -> float:
        if self.layerscale_max <= 0.0:
            return -20.0
        if self.layerscale_init <= 0.0:
            return -20.0
        target = min(self.layerscale_init, self.layerscale_max * 0.999)
        target = max(target, self.layerscale_max * 1e-6)
        p = min(max(target / self.layerscale_max, 1e-6), 1.0 - 1e-6)
        return float(math.log(p / (1.0 - p)))

    def reset_parameters(self) -> None:
        if self.mode in ("none", "identity_control"):
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
            target_px_gp2: [G, P, 2] real image coordinates. Current controls do
                not consume it, but the argument preserves the future graph API.
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

        if self.mode == "identity_control":
            dummy = self.identity_dummy.to(device=features_gpc.device, dtype=features_gpc.dtype)
            refined = features_gpc + dummy * features_gpc.new_zeros(())
            if return_stats:
                return refined, self.empty_stats(features_gpc), refined.float().sum() * 0.0
            return refined

        x = self.norm(features_gpc)
        delta = self.mlp(x)

        if self.mode == "mlp_control":
            gate = torch.sigmoid(self.gate(x))
            layerscale = self.layerscale.to(device=features_gpc.device, dtype=features_gpc.dtype)
            raw_update = gate * layerscale.view(1, 1, -1) * delta
            effective_update = raw_update
            cap_scale = torch.ones_like(raw_update[..., :1])
            anchor_loss = features_gpc.new_zeros(())
        elif self.mode == "safe_mlp_control":
            gate = self._bounded_gate(x)
            layerscale = self._bounded_layerscale(features_gpc)
            raw_update = gate * layerscale.view(1, 1, -1) * delta
            effective_update, cap_scale = self._cap_update(features_gpc, raw_update)
            anchor_loss = self._anchor_loss(features_gpc, effective_update)
        else:
            raise RuntimeError(f"Unhandled query graph refine mode={self.mode!r}.")

        refined = features_gpc + effective_update
        reg_loss = effective_update.float().abs().mean()
        if self.anchor_weight > 0.0:
            reg_loss = reg_loss + self.anchor_weight * anchor_loss

        if not return_stats:
            return refined
        stats = self._summarize(
            features_gpc=features_gpc,
            delta=delta,
            gate=gate,
            layerscale=layerscale,
            raw_update=raw_update,
            effective_update=effective_update,
            cap_scale=cap_scale,
            anchor_loss=anchor_loss,
        )
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

    def _bounded_gate(self, x: torch.Tensor) -> torch.Tensor:
        if self.gate_max <= 0.0:
            return x.new_zeros((*x.shape[:2], 1))
        return self.gate_max * torch.sigmoid(self.gate(x))

    def _bounded_layerscale(self, features: torch.Tensor) -> torch.Tensor:
        if self.mode == "mlp_control":
            return self.layerscale.to(device=features.device, dtype=features.dtype)
        if self.layerscale_max <= 0.0:
            return features.new_zeros((self.feature_dim,))
        layerscale = self.layerscale_max * torch.sigmoid(self.layerscale_raw)
        return layerscale.to(device=features.device, dtype=features.dtype)

    def _cap_update(self, features: torch.Tensor, raw_update: torch.Tensor):
        update_norm = raw_update.detach().float().norm(dim=-1, keepdim=True)
        caps = []
        if self.update_norm_cap > 0.0:
            caps.append(torch.full_like(update_norm, self.update_norm_cap))
        if self.update_norm_cap_ratio > 0.0:
            feature_norm = features.detach().float().norm(dim=-1, keepdim=True).clamp_min(1e-6)
            caps.append(feature_norm * self.update_norm_cap_ratio)
        if not caps:
            scale = torch.ones_like(update_norm)
            return raw_update, scale.to(device=raw_update.device, dtype=raw_update.dtype)
        cap = caps[0]
        for next_cap in caps[1:]:
            cap = torch.minimum(cap, next_cap)
        scale = torch.clamp(cap / update_norm.clamp_min(1e-12), max=1.0)
        return raw_update * scale.to(device=raw_update.device, dtype=raw_update.dtype), scale

    @staticmethod
    def _anchor_loss(features: torch.Tensor, effective_update: torch.Tensor) -> torch.Tensor:
        refined = features + effective_update
        cosine = F.cosine_similarity(refined.float(), features.float(), dim=-1, eps=1e-6)
        return (1.0 - cosine).mean()

    def final_state_summary(self) -> Dict[str, float]:
        summary = {
            "final_query_graph_layerscale_absmean": 0.0,
            "final_query_graph_gate_bias": 0.0,
            "final_query_graph_gate_max": self.gate_max,
            "final_query_graph_layerscale_max": self.layerscale_max,
            "final_query_graph_update_norm_cap": self.update_norm_cap,
            "final_query_graph_update_norm_cap_ratio": self.update_norm_cap_ratio,
            "final_query_graph_layerscale_eff_absmean": 0.0,
            "final_query_graph_anchor_weight": self.anchor_weight,
        }
        if self.mode in ("none", "identity_control"):
            return summary
        gate_bias = self.gate[-1].bias.detach().float()
        layerscale = self._bounded_layerscale(torch.empty(0, device=gate_bias.device)).detach().float()
        summary.update(
            {
                "final_query_graph_layerscale_absmean": float(layerscale.abs().mean().item()),
                "final_query_graph_gate_bias": float(gate_bias.mean().item()),
                "final_query_graph_layerscale_eff_absmean": float(layerscale.abs().mean().item()),
            }
        )
        return summary

    @staticmethod
    def empty_stats(features: torch.Tensor) -> Dict[str, float]:
        return {
            "query_graph_delta_norm": 0.0,
            "query_graph_effective_update_norm": 0.0,
            "query_graph_raw_update_norm": 0.0,
            "query_graph_effective_update_ratio": 0.0,
            "query_graph_raw_update_ratio": 0.0,
            "query_graph_update_cap_scale_mean": 1.0,
            "query_graph_update_cap_scale_min": 1.0,
            "query_graph_gate_mean": 0.0,
            "query_graph_gate_min": 0.0,
            "query_graph_gate_max": 0.0,
            "query_graph_gate_eff_mean": 0.0,
            "query_graph_gate_eff_max": 0.0,
            "query_graph_layerscale_absmean": 0.0,
            "query_graph_layerscale_eff_absmean": 0.0,
            "query_graph_layerscale_eff_absmax": 0.0,
            "query_graph_anchor_loss": 0.0,
        }

    def _summarize(
        self,
        *,
        features_gpc: torch.Tensor,
        delta: torch.Tensor,
        gate: torch.Tensor,
        layerscale: torch.Tensor,
        raw_update: torch.Tensor,
        effective_update: torch.Tensor,
        cap_scale: torch.Tensor,
        anchor_loss: torch.Tensor,
    ) -> Dict[str, float]:
        with torch.no_grad():
            feature_norm = features_gpc.detach().float().norm(dim=-1).clamp_min(1e-6)
            raw_norm = raw_update.detach().float().norm(dim=-1)
            eff_norm = effective_update.detach().float().norm(dim=-1)
            return {
                "query_graph_delta_norm": float(delta.detach().float().norm(dim=-1).mean().item()),
                "query_graph_effective_update_norm": float(eff_norm.mean().item()),
                "query_graph_raw_update_norm": float(raw_norm.mean().item()),
                "query_graph_effective_update_ratio": float((eff_norm / feature_norm).mean().item()),
                "query_graph_raw_update_ratio": float((raw_norm / feature_norm).mean().item()),
                "query_graph_update_cap_scale_mean": float(cap_scale.detach().float().mean().item()),
                "query_graph_update_cap_scale_min": float(cap_scale.detach().float().min().item()),
                "query_graph_gate_mean": float(gate.detach().float().mean().item()),
                "query_graph_gate_min": float(gate.detach().float().min().item()),
                "query_graph_gate_max": float(gate.detach().float().max().item()),
                "query_graph_gate_eff_mean": float(gate.detach().float().mean().item()),
                "query_graph_gate_eff_max": float(gate.detach().float().max().item()),
                "query_graph_layerscale_absmean": float(layerscale.detach().float().abs().mean().item()),
                "query_graph_layerscale_eff_absmean": float(layerscale.detach().float().abs().mean().item()),
                "query_graph_layerscale_eff_absmax": float(layerscale.detach().float().abs().max().item()),
                "query_graph_anchor_loss": float(anchor_loss.detach().float().item()),
            }
