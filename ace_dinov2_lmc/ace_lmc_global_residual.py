"""Adaptive GLACE global residual module for ACE-FCN-LMC Stage2."""

import math

import torch
import torch.nn as nn

from ace_network_dinov2 import Head


def _logit_from_unit_interval(value):
    value = min(max(float(value), 1e-6), 1.0 - 1e-6)
    return math.log(value / (1.0 - value))


class ACEGlobalResidualHead(nn.Module):
    """Predict a gated coordinate residual on top of the frozen local head."""

    def __init__(
        self,
        *,
        mean,
        num_head_blocks,
        in_channels,
        gate_init=0.001,
        gate_max=0.1,
        delta_max_m=1.0,
    ):
        super().__init__()
        self.gate_max = max(0.0, float(gate_max))
        self.delta_max_m = max(0.0, float(delta_max_m))
        self.delta_head = Head(
            torch.zeros_like(mean),
            num_head_blocks,
            False,
            in_channels=in_channels,
        )
        nn.init.zeros_(self.delta_head.fc3.weight)
        nn.init.zeros_(self.delta_head.fc3.bias)

        self.gate = nn.Conv2d(in_channels, 1, 1, 1, 0)
        nn.init.zeros_(self.gate.weight)
        gate_init = max(0.0, float(gate_init))
        if self.gate_max > 0.0:
            bias = _logit_from_unit_interval(min(gate_init, self.gate_max) / self.gate_max)
        else:
            bias = gate_init
        nn.init.constant_(self.gate.bias, bias)
        self.last_stats = {}

    def gate_map(self, features_BCHW):
        raw = self.gate(features_BCHW)
        if self.gate_max <= 0.0:
            return torch.zeros_like(raw)
        return torch.sigmoid(raw) * torch.tensor(
            self.gate_max,
            device=features_BCHW.device,
            dtype=features_BCHW.dtype,
        )

    def forward(self, local_pred_B3HW, features_BCHW):
        raw_delta_B3HW = self.delta_head(features_BCHW)
        if self.delta_max_m <= 0.0:
            delta_B3HW = torch.zeros_like(raw_delta_B3HW)
        else:
            delta_B3HW = torch.tanh(raw_delta_B3HW) * torch.tensor(
                self.delta_max_m,
                device=features_BCHW.device,
                dtype=features_BCHW.dtype,
            )
        gate_B1HW = self.gate_map(features_BCHW)
        gated_delta_B3HW = gate_B1HW * delta_B3HW
        out = local_pred_B3HW + gated_delta_B3HW
        with torch.no_grad():
            self.last_stats = {
                "gate_mean": float(gate_B1HW.detach().float().mean().cpu().item()),
                "gate_max": float(gate_B1HW.detach().float().max().cpu().item()),
                "delta_l2": float(delta_B3HW.detach().float().pow(2).mean().sqrt().cpu().item()),
                "gated_delta_l2": float(gated_delta_B3HW.detach().float().pow(2).mean().sqrt().cpu().item()),
            }
        return out, gate_B1HW, delta_B3HW
