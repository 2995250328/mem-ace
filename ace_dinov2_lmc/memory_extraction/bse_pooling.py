"""
Bilateral Supervoxel Extraction (BSE) with Otsu adaptive thresholding.
"""

import torch
from torch import Tensor
from typing import Dict


class BSEPooler:
    """Boundary-preserving bilateral clustering pooler."""

    def __init__(
        self,
        voxel_size: float = 0.05,
        use_otsu: bool = True,
        otsu_bins: int = 256,
        unimodal_threshold: float = 0.02
    ):
        """
        Args:
            voxel_size: Coarse voxel grid size
            use_otsu: Use Otsu adaptive thresholding (else fixed tau=0.90)
            otsu_bins: Number of bins for Otsu histogram
            unimodal_threshold: Skip split if std(sim) < threshold
        """
        self.voxel_size = voxel_size
        self.use_otsu = use_otsu
        self.otsu_bins = otsu_bins
        self.unimodal_threshold = unimodal_threshold

    def pool(
        self,
        points: Tensor,
        features: Tensor,
        colors: Tensor,
        ray_dirs: Tensor
    ) -> Dict[str, Tensor]:
        """Apply BSE pooling.

        Args:
            points: [N, 3] world coordinates
            features: [N, C] multi-scale features (fp16)
            colors: [N, 3] RGB colors
            ray_dirs: [N, 3] unit ray directions

        Returns:
            Dict with keys {points, features, colors, ray_dirs}
        """
        # Step 1: Coarse voxel hash
        cluster_ids = self._voxel_hash(points)

        # Step 2: Compute cluster mean features (fp32 for stability)
        F_mean = self._scatter_mean(features.float(), cluster_ids)

        # Step 3: Otsu adaptive thresholding
        sim = self._cosine_similarity(features.float(), F_mean[cluster_ids])
        tau = self._otsu_threshold(sim) if self.use_otsu else 0.90

        # Step 4: Binary split
        is_outlier = (sim < tau).long()
        sub_ids = cluster_ids * 2 + is_outlier

        # Step 5: Fine-grained pooling
        return self._scatter_mean_all(points, features, colors, ray_dirs, sub_ids)

    def _voxel_hash(self, points: Tensor) -> Tensor:
        """Quantize points to voxel grid and return cluster IDs."""
        quantized = torch.floor(points / self.voxel_size).long()
        # Encode 3D -> 1D via prime hashing
        hash_vals = quantized[:, 0] * 73856093 + quantized[:, 1] * 19349663 + quantized[:, 2] * 83492791
        _, cluster_ids = torch.unique(hash_vals, return_inverse=True)
        return cluster_ids

    def _scatter_mean(self, features: Tensor, cluster_ids: Tensor) -> Tensor:
        """Compute mean features per cluster using index_add_."""
        num_clusters = cluster_ids.max().item() + 1
        C = features.shape[1]

        # Accumulate features
        cluster_sum = torch.zeros(num_clusters, C, device=features.device, dtype=features.dtype)
        cluster_sum.index_add_(0, cluster_ids, features)

        # Count points per cluster
        cluster_count = torch.zeros(num_clusters, device=features.device, dtype=features.dtype)
        cluster_count.index_add_(0, cluster_ids, torch.ones(len(cluster_ids), device=features.device, dtype=features.dtype))

        # Compute mean
        return cluster_sum / cluster_count.unsqueeze(1).clamp(min=1)

    def _cosine_similarity(self, features: Tensor, cluster_means: Tensor) -> Tensor:
        """Compute cosine similarity between features and their cluster means."""
        features_norm = torch.nn.functional.normalize(features, dim=1, eps=1e-6)
        means_norm = torch.nn.functional.normalize(cluster_means, dim=1, eps=1e-6)
        return (features_norm * means_norm).sum(dim=1)

    def _otsu_threshold(self, sim: Tensor) -> float:
        """Otsu's method for adaptive threshold selection."""
        # Check for unimodal distribution
        if sim.std().item() < self.unimodal_threshold:
            return 0.90  # Fallback to fixed threshold

        # Build histogram
        hist = torch.histc(sim, bins=self.otsu_bins, min=0.0, max=1.0)
        bin_centers = torch.linspace(0.0, 1.0, self.otsu_bins, device=sim.device)

        # Compute weights and means
        weight_total = hist.sum()
        mean_total = (hist * bin_centers).sum() / weight_total

        # Find threshold that maximizes inter-class variance
        max_variance = 0.0
        best_threshold = 0.90

        weight_bg = 0.0
        mean_bg = 0.0

        for i in range(self.otsu_bins - 1):
            weight_bg += hist[i]
            if weight_bg == 0:
                continue

            weight_fg = weight_total - weight_bg
            if weight_fg == 0:
                break

            mean_bg += hist[i] * bin_centers[i]
            mean_bg_val = mean_bg / weight_bg
            mean_fg_val = (mean_total * weight_total - mean_bg) / weight_fg

            variance = weight_bg * weight_fg * (mean_bg_val - mean_fg_val) ** 2

            if variance > max_variance:
                max_variance = variance
                best_threshold = bin_centers[i].item()

        return best_threshold

    def _scatter_mean_all(
        self,
        points: Tensor,
        features: Tensor,
        colors: Tensor,
        ray_dirs: Tensor,
        cluster_ids: Tensor
    ) -> Dict[str, Tensor]:
        """Pool all attributes by cluster mean."""
        pooled_points = self._scatter_mean(points, cluster_ids)
        pooled_features = self._scatter_mean(features, cluster_ids)
        pooled_colors = self._scatter_mean(colors, cluster_ids)
        pooled_ray_dirs = self._scatter_mean(ray_dirs, cluster_ids)

        # L2-normalize ray directions
        pooled_ray_dirs = torch.nn.functional.normalize(pooled_ray_dirs, dim=1, eps=1e-6)

        return {
            'points': pooled_points,
            'features': pooled_features.half(),  # Back to fp16
            'colors': pooled_colors,
            'ray_dirs': pooled_ray_dirs
        }

