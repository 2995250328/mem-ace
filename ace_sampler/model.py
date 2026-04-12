import torch
import torch.nn as nn
import torch.nn.functional as F


class _ECALite(nn.Module):
    """Efficient Channel Attention (1D conv variant). Works with any batch size."""
    def __init__(self, channels: int, k: int = 3):
        super().__init__()
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=k // 2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = x.mean(dim=(2, 3)).unsqueeze(1)          # (B, 1, C)
        w = torch.sigmoid(self.conv(w))
        w = w.squeeze(1).unsqueeze(-1).unsqueeze(-1)  # (B, C, 1, 1)
        return x * w


class SamplerNet(nn.Module):
    """Confidence predictor on top of ACE encoder features.

    Configurable capacity via mid_channels and num_branches:

      small  (mid_channels=128, num_branches=2): ~81K params
        Stage 1 : DW-Conv(512,k=3) → GN → PW-Conv(128) → GN → SiLU
        Stage 2 : 2-branch Lite-ASPP (dilation 1,3), each → 32ch, concat → 64ch
        Stage 3 : ECA-lite(64)
        Stage 4 : Conv1×1(64→1) → Sigmoid

      large  (mid_channels=256, num_branches=3): ~254K params  ← recommended for multi-scene
        Stage 1 : DW-Conv(512,k=3) → GN → PW-Conv(256) → GN → SiLU
        Stage 2 : 3-branch Lite-ASPP (dilation 1,3,5), each → 64ch, concat → 192ch
        Stage 3 : ECA-lite(192)
        Stage 4 : Conv1×1(192→64) → GN → SiLU → Conv1×1(64→1) → Sigmoid

    Uses GroupNorm (not BatchNorm) — safe with batch_size=1.

    Input:  (B, 512, H, W) — frozen ACE encoder features
    Output: (B, 1,   H, W) — confidence map in [0, 1]
    """
    # dilation schedule per branch index
    _DILATIONS = [1, 3, 5]

    def __init__(self, in_channels: int = 512,
                 mid_channels: int = 128,
                 num_branches: int = 2):
        super().__init__()
        assert num_branches in (2, 3), "num_branches must be 2 or 3"
        branch_out = mid_channels // 2          # 64 (small) or 128 (large)
        merged_ch  = branch_out * num_branches  # 64 (small) or 192 (large)

        # Stage 1
        self.stage1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, 1, 1, groups=in_channels, bias=False),
            nn.GroupNorm(32, in_channels),
            nn.Conv2d(in_channels, mid_channels, 1, bias=False),
            nn.GroupNorm(32, mid_channels),
            nn.SiLU(inplace=True),
        )

        # Stage 2: Lite-ASPP branches
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(mid_channels, mid_channels, 3, 1,
                          padding=d, dilation=d, groups=mid_channels, bias=False),
                nn.Conv2d(mid_channels, branch_out, 1, bias=False),
                nn.SiLU(inplace=True),
            )
            for d in self._DILATIONS[:num_branches]
        ])

        # Residual projection: mid_channels → merged_ch
        self.residual_proj = nn.Conv2d(mid_channels, merged_ch, 1, bias=False)

        # Stage 3: ECA-lite
        self.eca = _ECALite(channels=merged_ch, k=3)

        # Stage 4: prediction head
        if num_branches == 3:
            # deeper head for large capacity
            self.head = nn.Sequential(
                nn.Conv2d(merged_ch, 64, 1, bias=False),
                nn.GroupNorm(32, 64),
                nn.SiLU(inplace=True),
                nn.Conv2d(64, 1, 1),
            )
        else:
            self.head = nn.Conv2d(merged_ch, 1, 1)

        # store config for save/load
        self._in_channels  = in_channels
        self._mid_channels = mid_channels
        self._num_branches = num_branches

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        x1 = self.stage1(features)
        x_merged = torch.cat([b(x1) for b in self.branches], dim=1)
        x_merged = x_merged + self.residual_proj(x1)
        x_attn   = self.eca(x_merged)
        if isinstance(self.head, nn.Sequential):
            return torch.sigmoid(self.head(x_attn))
        return torch.sigmoid(self.head(x_attn))

    def save(self, path: str, meta: dict = None):
        payload = {
            'state_dict':   self.state_dict(),
            'in_channels':  self._in_channels,
            'mid_channels': self._mid_channels,
            'num_branches': self._num_branches,
        }
        if meta:
            payload.update(meta)
        torch.save(payload, path)

    @classmethod
    def load(cls, path: str, device) -> 'SamplerNet':
        ckpt = torch.load(path, map_location=device, weights_only=False)
        net = cls(
            in_channels  = ckpt.get('in_channels',  512),
            mid_channels = ckpt.get('mid_channels', 128),
            num_branches = ckpt.get('num_branches', 2),
        )
        net.load_state_dict(ckpt['state_dict'])
        return net.to(device)
