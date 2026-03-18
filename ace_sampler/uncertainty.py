import torch
import torch.nn as nn


class UncertaintyHead(nn.Module):
    """
    Lightweight coordinate predictor with MC Dropout.
    Input:  (B, in_channels, Hf, Wf) frozen encoder features
    Output: (B, 3, Hf, Wf) scene coordinate prediction

    Used in Phase 1 SamplerNet training to estimate per-pixel prediction
    uncertainty via multiple stochastic forward passes (MC Dropout).
    """
    def __init__(self, in_channels: int = 512, dropout_p: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 256, 1, bias=False),
            nn.Dropout2d(p=dropout_p),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 3, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)

    def mc_coordinate_samples(self, features: torch.Tensor, T: int = 10) -> torch.Tensor:
        """
        Run T stochastic forward passes with dropout enabled.
        Returns (T, B, 3, Hf, Wf).
        """
        self.train()
        with torch.no_grad():
            return torch.stack([self.forward(features) for _ in range(T)], dim=0)

    def save(self, path: str, meta: dict = None):
        payload = {
            'state_dict': self.state_dict(),
            'in_channels': self.net[0].in_channels,
            'dropout_p': self.net[1].p,
        }
        if meta:
            payload.update(meta)
        torch.save(payload, path)

    @classmethod
    def load(cls, path: str, device) -> 'UncertaintyHead':
        ckpt = torch.load(path, map_location=device)
        head = cls(
            in_channels=ckpt.get('in_channels', 512),
            dropout_p=ckpt.get('dropout_p', 0.1),
        )
        head.load_state_dict(ckpt['state_dict'])
        return head.to(device)
