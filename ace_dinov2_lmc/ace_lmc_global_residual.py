"""Adaptive GLACE global residual module for ACE-FCN-LMC Stage2."""

import math

import torch
import torch.nn as nn

from ace_network_dinov2 import Head


def _logit_from_unit_interval(value):
    value = min(max(float(value), 1e-6), 1.0 - 1e-6)
    return math.log(value / (1.0 - value))


class ACEGlobalResidualHead(nn.Module):
    """Predict a gated coordinate residual on top of the frozen local head.

    Optional xyz conditioning is not concatenated into the residual head input.
    The frozen Stage1 prediction is first Fourier encoded, then used to generate
    AdaLN/FiLM-style residual modulation for the feature/global tensor. The
    conditioning path is zero-initialized, so the module starts from the exact
    legacy feature path and the overall output starts as local_pred because the
    coordinate-delta head is also zero-initialized.
    """

    def __init__(
        self,
        *,
        mean,
        num_head_blocks,
        in_channels,
        gate_init=0.001,
        gate_max=0.1,
        delta_max_m=1.0,
        xyz_condition_mode="none",
        xyz_condition_scale=10.0,
        xyz_fourier_frequencies=(1.0, 2.0, 4.0, 8.0),
        xyz_adaln_hidden=128,
    ):
        super().__init__()
        self.gate_max = max(0.0, float(gate_max))
        self.delta_max_m = max(0.0, float(delta_max_m))
        self.xyz_condition_mode = str(xyz_condition_mode or "none").lower()
        if self.xyz_condition_mode not in ("none", "fourier_adaln"):
            raise ValueError(f"Unsupported xyz_condition_mode={xyz_condition_mode!r}")
        self.xyz_condition_scale = float(xyz_condition_scale or 1.0)
        if not math.isfinite(self.xyz_condition_scale) or self.xyz_condition_scale <= 0.0:
            self.xyz_condition_scale = 1.0

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

        if self.xyz_condition_mode == "fourier_adaln":
            freqs = torch.tensor(tuple(float(x) for x in xyz_fourier_frequencies), dtype=torch.float32)
            if freqs.numel() <= 0:
                raise ValueError("xyz_fourier_frequencies must be non-empty for fourier_adaln conditioning.")
            self.register_buffer("xyz_fourier_frequencies", freqs, persistent=False)
            xyz_pe_channels = 3 + 2 * 3 * int(freqs.numel())
            hidden = max(16, int(xyz_adaln_hidden))
            self.xyz_norm = nn.GroupNorm(1, in_channels, affine=False)
            self.xyz_adaln = nn.Sequential(
                nn.Conv2d(xyz_pe_channels, hidden, 1, 1, 0),
                nn.SiLU(inplace=True),
                nn.Conv2d(hidden, 2 * in_channels, 1, 1, 0),
            )
            nn.init.zeros_(self.xyz_adaln[-1].weight)
            nn.init.zeros_(self.xyz_adaln[-1].bias)
        else:
            self.register_buffer("xyz_fourier_frequencies", torch.empty(0, dtype=torch.float32), persistent=False)
            self.xyz_norm = None
            self.xyz_adaln = None
        self.last_stats = {}

    def _xyz_fourier_pe(self, local_pred_B3HW):
        xyz = local_pred_B3HW.detach().to(dtype=torch.float32) / float(self.xyz_condition_scale)
        freqs = self.xyz_fourier_frequencies.to(device=xyz.device, dtype=xyz.dtype).view(1, -1, 1, 1, 1)
        xyz_f = xyz[:, None, :, :, :] * freqs * math.pi
        sin = torch.sin(xyz_f).flatten(1, 2)
        cos = torch.cos(xyz_f).flatten(1, 2)
        return torch.cat((xyz, sin, cos), dim=1)

    def _condition_features(self, features_BCHW, local_pred_B3HW):
        if self.xyz_condition_mode != "fourier_adaln":
            return features_BCHW, 0.0, 0.0
        xyz_pe = self._xyz_fourier_pe(local_pred_B3HW).to(device=features_BCHW.device, dtype=features_BCHW.dtype)
        scale_shift = self.xyz_adaln(xyz_pe)
        scale, shift = scale_shift.chunk(2, dim=1)
        normed = self.xyz_norm(features_BCHW.float()).to(dtype=features_BCHW.dtype)
        conditioned = features_BCHW + normed * scale + shift
        with torch.no_grad():
            scale_abs = float(scale.detach().float().abs().mean().cpu().item())
            shift_abs = float(shift.detach().float().abs().mean().cpu().item())
        return conditioned, scale_abs, shift_abs

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
        conditioned_BCHW, xyz_scale_abs, xyz_shift_abs = self._condition_features(features_BCHW, local_pred_B3HW)
        raw_delta_B3HW = self.delta_head(conditioned_BCHW)
        if self.delta_max_m <= 0.0:
            delta_B3HW = torch.zeros_like(raw_delta_B3HW)
        else:
            delta_B3HW = torch.tanh(raw_delta_B3HW) * torch.tensor(
                self.delta_max_m,
                device=features_BCHW.device,
                dtype=features_BCHW.dtype,
            )
        gate_B1HW = self.gate_map(conditioned_BCHW)
        gated_delta_B3HW = gate_B1HW * delta_B3HW
        out = local_pred_B3HW + gated_delta_B3HW
        with torch.no_grad():
            self.last_stats = {
                "gate_mean": float(gate_B1HW.detach().float().mean().cpu().item()),
                "gate_max": float(gate_B1HW.detach().float().max().cpu().item()),
                "delta_l2": float(delta_B3HW.detach().float().pow(2).mean().sqrt().cpu().item()),
                "gated_delta_l2": float(gated_delta_B3HW.detach().float().pow(2).mean().sqrt().cpu().item()),
                "xyz_condition_mode": self.xyz_condition_mode,
                "xyz_adaln_scale_abs": xyz_scale_abs,
                "xyz_adaln_shift_abs": xyz_shift_abs,
            }
        return out, gate_B1HW, delta_B3HW
