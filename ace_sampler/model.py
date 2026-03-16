import torch
import torch.nn as nn


class SamplerNet(nn.Module):
    """Lightweight confidence predictor on top of ACE encoder features.
    Input:  (B, 512, H, W) encoder feature map
    Output: (B, 1,   H, W) confidence map in [0, 1]
    """
    def __init__(self, in_channels: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            # Block 1: depthwise + pointwise
            nn.Conv2d(in_channels, in_channels, 3, 1, 1, groups=in_channels, bias=False),
            nn.Conv2d(in_channels, 128, 1, bias=False),
            nn.ReLU(inplace=True),
            # Block 2
            nn.Conv2d(128, 128, 3, 1, 1, groups=128, bias=False),
            nn.Conv2d(128, 64, 1, bias=False),
            nn.ReLU(inplace=True),
            # Head
            nn.Conv2d(64, 1, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(features))

    def save(self, path: str, meta: dict = None):
        payload = {'state_dict': self.state_dict(), 'in_channels': self.net[1].in_channels}
        if meta:
            payload.update(meta)
        torch.save(payload, path)

    @classmethod
    def load(cls, path: str, device) -> 'SamplerNet':
        ckpt = torch.load(path, map_location=device)
        net = cls(in_channels=ckpt.get('in_channels', 512))
        net.load_state_dict(ckpt['state_dict'])
        return net.to(device)
