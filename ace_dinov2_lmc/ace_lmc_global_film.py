"""Global feature FiLM head for ACE-FCN-LMC Stage2.

The module mirrors the ACE 1x1-conv coordinate head while adding a zero-
initialized global feature modulation path. With gate=0 and the default zero
FiLM initialization, it is exactly equivalent to the loaded local Stage1 head.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ACEGlobalFiLMHead(nn.Module):
    """ACE coordinate head with image-level FiLM conditioning."""

    def __init__(
        self,
        mean,
        num_head_blocks: int,
        use_homogeneous: bool,
        *,
        in_channels: int = 512,
        global_dim: int = 256,
        gate_init: float = 0.0,
        gate_max: float = 1.0,
        film_hidden_dim: int = 512,
        homogeneous_min_scale: float = 0.01,
        homogeneous_max_scale: float = 4.0,
    ):
        super().__init__()

        self.use_homogeneous = bool(use_homogeneous)
        self.in_channels = int(in_channels)
        self.global_dim = int(global_dim)
        self.head_channels = 512
        self.gate_max = float(gate_max)

        if self.global_dim <= 0:
            raise ValueError(f"global_dim must be positive, got {self.global_dim}")

        self.head_skip = (
            nn.Identity()
            if self.in_channels == self.head_channels
            else nn.Conv2d(self.in_channels, self.head_channels, 1, 1, 0)
        )

        self.res3_conv1 = nn.Conv2d(self.in_channels, self.head_channels, 1, 1, 0)
        self.res3_conv2 = nn.Conv2d(self.head_channels, self.head_channels, 1, 1, 0)
        self.res3_conv3 = nn.Conv2d(self.head_channels, self.head_channels, 1, 1, 0)

        self.res_blocks = []
        for block in range(int(num_head_blocks)):
            self.res_blocks.append((
                nn.Conv2d(self.head_channels, self.head_channels, 1, 1, 0),
                nn.Conv2d(self.head_channels, self.head_channels, 1, 1, 0),
                nn.Conv2d(self.head_channels, self.head_channels, 1, 1, 0),
            ))
            self.add_module(str(block) + "c0", self.res_blocks[block][0])
            self.add_module(str(block) + "c1", self.res_blocks[block][1])
            self.add_module(str(block) + "c2", self.res_blocks[block][2])

        self.fc1 = nn.Conv2d(self.head_channels, self.head_channels, 1, 1, 0)
        self.fc2 = nn.Conv2d(self.head_channels, self.head_channels, 1, 1, 0)

        if self.use_homogeneous:
            self.fc3 = nn.Conv2d(self.head_channels, 4, 1, 1, 0)
            self.register_buffer("max_scale", torch.tensor([homogeneous_max_scale]))
            self.register_buffer("min_scale", torch.tensor([homogeneous_min_scale]))
            self.register_buffer("max_inv_scale", 1.0 / self.max_scale)
            self.register_buffer("h_beta", torch.tensor([math.log(2) / (1.0 - float(self.max_inv_scale.item()))]))
            self.register_buffer("min_inv_scale", 1.0 / self.min_scale)
        else:
            self.fc3 = nn.Conv2d(self.head_channels, 3, 1, 1, 0)

        self.register_buffer("mean", mean.clone().detach().view(1, 3, 1, 1))

        self.film = nn.Sequential(
            nn.Linear(self.global_dim, int(film_hidden_dim)),
            nn.ReLU(inplace=True),
            nn.Linear(int(film_hidden_dim), self.head_channels * 2),
        )
        nn.init.zeros_(self.film[-1].weight)
        nn.init.zeros_(self.film[-1].bias)

        raw_gate_init = float(gate_init)
        if self.gate_max > 0.0:
            unit = min(max(raw_gate_init / self.gate_max, 1e-6), 1.0 - 1e-6)
            raw_gate_init = math.log(unit / (1.0 - unit))
        self.global_gate = nn.Parameter(torch.tensor(raw_gate_init, dtype=torch.float32))
        self.last_stats = {}

    def _gate(self, dtype, device):
        gate = self.global_gate.to(device=device, dtype=dtype)
        if self.gate_max > 0.0:
            return torch.sigmoid(gate) * torch.tensor(self.gate_max, device=device, dtype=dtype)
        return gate

    def gate_value(self) -> float:
        return float(self._gate(torch.float32, self.global_gate.device).detach().cpu().item())

    def _apply_film(self, hidden, global_features):
        if global_features is None:
            raise ValueError("ACEGlobalFiLMHead requires global_features.")
        if global_features.dim() != 2:
            raise ValueError(f"Expected global_features [B,C], got {tuple(global_features.shape)}.")
        if hidden.shape[0] != global_features.shape[0]:
            raise ValueError(
                f"Global/local batch mismatch: hidden={tuple(hidden.shape)} global={tuple(global_features.shape)}"
            )
        if global_features.shape[1] != self.global_dim:
            raise ValueError(f"Expected global_dim={self.global_dim}, got {global_features.shape[1]}.")

        film = self.film(global_features.to(dtype=hidden.dtype))
        gamma, beta = film.chunk(2, dim=1)
        gate = self._gate(hidden.dtype, hidden.device)
        gamma = torch.tanh(gamma).view(hidden.shape[0], self.head_channels, 1, 1)
        beta = beta.view(hidden.shape[0], self.head_channels, 1, 1)
        out = hidden * (1.0 + gate * gamma) + gate * beta

        with torch.no_grad():
            self.last_stats = {
                "enabled": True,
                "gate": float(gate.detach().float().cpu().item()),
                "gamma_abs": float(gamma.detach().float().abs().mean().cpu().item()),
                "beta_abs": float(beta.detach().float().abs().mean().cpu().item()),
                "step_abs": float((out - hidden).detach().float().abs().mean().cpu().item()),
            }
        return out

    def forward(self, res, global_features=None):
        x = F.relu(self.res3_conv1(res))
        x = F.relu(self.res3_conv2(x))
        x = F.relu(self.res3_conv3(x))
        res = self.head_skip(res) + x

        for res_block in self.res_blocks:
            x = F.relu(res_block[0](res))
            x = F.relu(res_block[1](x))
            x = F.relu(res_block[2](x))
            res = res + x

        res = self._apply_film(res, global_features)

        sc = F.relu(self.fc1(res))
        sc = F.relu(self.fc2(sc))
        sc = self.fc3(sc)

        if self.use_homogeneous:
            h_slice = F.softplus(sc[:, 3, :, :].unsqueeze(1), beta=float(self.h_beta.item())) + self.max_inv_scale
            h_slice.clamp_(max=self.min_inv_scale)
            sc = sc[:, :3] / h_slice

        sc += self.mean
        return sc

