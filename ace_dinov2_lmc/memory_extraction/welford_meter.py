"""
Welford streaming normalization for O(1) memory global statistics.
"""

import torch
from torch import Tensor
from typing import Tuple


class WelfordNormalizer:
    """Numerically stable streaming mean and variance computation."""

    def __init__(self):
        self.count = 0
        self.mean = None
        self.M2 = None

    def update(self, points: Tensor) -> None:
        """Accumulate statistics from a chunk of points.

        Args:
            points: [N, 3] tensor of 3D coordinates
        """
        points = points.double()  # FP64 for numerical stability

        if self.mean is None:
            self.mean = torch.zeros(3, dtype=torch.float64, device=points.device)
            self.M2 = torch.zeros(3, dtype=torch.float64, device=points.device)

        for p in points:
            self.count += 1
            delta = p - self.mean
            self.mean += delta / self.count
            self.M2 += delta * (p - self.mean)

    def finalize(self) -> Tuple[Tensor, float]:
        """Compute final statistics.

        Returns:
            mu_scene: [3] mean (FP32)
            sigma_scene: scalar std (FP32)
        """
        if self.count == 0:
            raise ValueError("No points accumulated")

        mu = self.mean.float()
        variance = self.M2 / self.count
        sigma = torch.sqrt(variance.mean()).item()

        return mu, sigma

    def normalize(self, points: Tensor) -> Tensor:
        """Apply normalization: (points - mu) / sigma.

        Args:
            points: [N, 3] tensor

        Returns:
            Normalized points [N, 3]
        """
        if self.mean is None:
            raise ValueError("Must call finalize() first")

        mu, sigma = self.finalize()
        return (points - mu) / sigma
