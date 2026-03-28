"""
Bilateral Supervoxel Extraction (BSE) with Otsu adaptive thresholding.

Ray direction handling strategies:
- 'mean': Average + L2 normalize (original, may lose multi-view info)
- 'dominant': Use ray with highest feature similarity to cluster mean
- 'first': Use first ray in cluster (simple, preserves original direction)
- 'all': Keep all rays per cluster (for downstream processing)
"""

import torch
from torch import Tensor
from typing import Dict, List, Optional, Literal

# Ray pooling strategies
RayPoolStrategy = Literal['mean', 'dominant', 'first', 'all']


def compute_plucker_rays(
    points: Tensor,
    ray_dirs: Tensor,
    camera_centers: Tensor
) -> Tensor:
    """
    Compute Plücker ray encoding for rays.

    Plücker coordinates: (direction, moment) where moment = point × direction
    This is a 6D representation that uniquely identifies a 3D line.

    Args:
        points: [N, 3] world coordinates of 3D points
        ray_dirs: [N, 3] unit ray directions
        camera_centers: [N, 3] camera centers (origin of each ray)

    Returns:
        plucker: [N, 6] Plücker coordinates (direction, moment)
    """
    # Moment = point × direction (or camera_center × direction)
    # Using camera_center as the reference point on the ray
    moment = torch.cross(camera_centers, ray_dirs, dim=1)

    # Concatenate direction and moment
    plucker = torch.cat([ray_dirs, moment], dim=1)  # [N, 6]

    return plucker


class BSEPooler:
    """Boundary-preserving bilateral clustering pooler."""

    def __init__(
        self,
        voxel_size: float = 0.05,
        use_otsu: bool = True,
        otsu_bins: int = 256,
        unimodal_threshold: float = 0.02,
        ray_pool_strategy: RayPoolStrategy = 'mean',
        save_all_ray_strategies: bool = True
    ):
        """
        Args:
            voxel_size: Coarse voxel grid size
            use_otsu: Use Otsu adaptive thresholding (else fixed tau=0.90)
            otsu_bins: Number of bins for Otsu histogram
            unimodal_threshold: Skip split if std(sim) < threshold
            ray_pool_strategy: Strategy for pooling ray directions
                - 'mean': Average + L2 normalize (may lose multi-view info)
                - 'dominant': Use ray with highest feature similarity
                - 'first': Use first ray in cluster
                - 'all': Keep all rays per cluster
            save_all_ray_strategies: If True, save all strategy results for comparison
        """
        self.voxel_size = voxel_size
        self.use_otsu = use_otsu
        self.otsu_bins = otsu_bins
        self.unimodal_threshold = unimodal_threshold
        self.ray_pool_strategy = ray_pool_strategy
        self.save_all_ray_strategies = save_all_ray_strategies

    def pool(
        self,
        points: Tensor,
        features: Tensor,
        colors: Tensor,
        ray_dirs: Tensor,
        camera_centers: Optional[Tensor] = None
    ) -> Dict[str, Tensor]:
        """Apply BSE pooling.

        Args:
            points: [N, 3] world coordinates
            features: [N, C] multi-scale features (fp16)
            colors: [N, 3] RGB colors
            ray_dirs: [N, 3] unit ray directions
            camera_centers: [N, 3] camera centers (optional, for Plücker encoding)

        Returns:
            Dict with keys:
                - points, features, colors: pooled attributes
                - ray_dirs: pooled ray directions (using ray_pool_strategy)
                - ray_dirs_mean: mean + normalize (always computed)
                - ray_dirs_dominant: dominant ray per cluster (if save_all_ray_strategies)
                - ray_dirs_first: first ray per cluster (if save_all_ray_strategies)
                - plucker_rays: [N_pooled, 6] Plücker coordinates (if camera_centers provided)
                - cluster_sizes: [N_pooled] number of points per cluster
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
        return self._scatter_mean_all(
            points, features, colors, ray_dirs, sub_ids,
            camera_centers=camera_centers,
            features_original=features.float(),
            similarity=sim
        )

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
        cluster_ids: Tensor,
        camera_centers: Optional[Tensor] = None,
        features_original: Optional[Tensor] = None,
        similarity: Optional[Tensor] = None
    ) -> Dict[str, Tensor]:
        """Pool all attributes by cluster mean with multiple ray strategies."""
        pooled_points = self._scatter_mean(points, cluster_ids)
        pooled_features = self._scatter_mean(features, cluster_ids)
        pooled_colors = self._scatter_mean(colors, cluster_ids)

        # Compute cluster sizes
        num_clusters = cluster_ids.max().item() + 1
        cluster_sizes = torch.zeros(num_clusters, device=cluster_ids.device, dtype=torch.long)
        cluster_sizes.index_add_(0, cluster_ids, torch.ones_like(cluster_ids))

        # ========================================
        # Ray direction pooling - Multiple strategies
        # ========================================

        # Strategy 1: Mean + Normalize (always compute as baseline)
        pooled_ray_dirs_mean = self._scatter_mean(ray_dirs, cluster_ids)
        pooled_ray_dirs_mean = torch.nn.functional.normalize(pooled_ray_dirs_mean, dim=1, eps=1e-6)

        # Strategy 2: Dominant ray (highest feature similarity to cluster mean)
        pooled_ray_dirs_dominant = self._pool_dominant_ray(
            ray_dirs, cluster_ids, similarity, num_clusters
        )

        # Strategy 3: First ray in cluster (simple, preserves original direction)
        pooled_ray_dirs_first = self._pool_first_ray(ray_dirs, cluster_ids, num_clusters)

        # Select primary ray_dirs based on strategy
        if self.ray_pool_strategy == 'mean':
            pooled_ray_dirs = pooled_ray_dirs_mean
        elif self.ray_pool_strategy == 'dominant':
            pooled_ray_dirs = pooled_ray_dirs_dominant
        elif self.ray_pool_strategy == 'first':
            pooled_ray_dirs = pooled_ray_dirs_first
        elif self.ray_pool_strategy == 'all':
            # For 'all' strategy, we keep the mean as default
            pooled_ray_dirs = pooled_ray_dirs_mean
        else:
            pooled_ray_dirs = pooled_ray_dirs_mean

        # Build result dict
        result = {
            'points': pooled_points,
            'features': pooled_features.half(),  # Back to fp16
            'colors': pooled_colors,
            'ray_dirs': pooled_ray_dirs,
            'ray_dirs_mean': pooled_ray_dirs_mean,
            'cluster_sizes': cluster_sizes
        }

        # Save all strategies for comparison
        if self.save_all_ray_strategies:
            result['ray_dirs_dominant'] = pooled_ray_dirs_dominant
            result['ray_dirs_first'] = pooled_ray_dirs_first

        # ========================================
        # Plücker ray encoding (if camera centers provided)
        # ========================================
        if camera_centers is not None:
            # Compute Plücker coordinates for original rays
            plucker_rays = compute_plucker_rays(points, ray_dirs, camera_centers)

            # Pool Plücker coordinates (mean)
            pooled_plucker = self._scatter_mean(plucker_rays, cluster_ids)

            # Normalize the direction part (first 3 components)
            pooled_plucker[:, :3] = torch.nn.functional.normalize(
                pooled_plucker[:, :3], dim=1, eps=1e-6
            )

            result['plucker_rays'] = pooled_plucker

            # Also save per-cluster camera centers for potential multi-view analysis
            pooled_camera_centers = self._scatter_mean(camera_centers, cluster_ids)
            result['camera_centers'] = pooled_camera_centers

        return result

    def _pool_dominant_ray(
        self,
        ray_dirs: Tensor,
        cluster_ids: Tensor,
        similarity: Optional[Tensor],
        num_clusters: int
    ) -> Tensor:
        """Select ray with highest feature similarity per cluster."""
        if similarity is None:
            # Fallback to mean if no similarity available
            pooled = self._scatter_mean(ray_dirs, cluster_ids)
            return torch.nn.functional.normalize(pooled, dim=1, eps=1e-6)

        # Find dominant ray index per cluster
        device = ray_dirs.device
        pooled = torch.zeros(num_clusters, 3, device=device, dtype=ray_dirs.dtype)

        for cluster_id in range(num_clusters):
            mask = cluster_ids == cluster_id
            if mask.sum() == 0:
                continue

            # Find index of highest similarity in this cluster
            cluster_sim = similarity[mask]
            max_idx = cluster_sim.argmax()

            # Get the corresponding ray
            global_indices = mask.nonzero(as_tuple=True)[0]
            dominant_ray_idx = global_indices[max_idx]
            pooled[cluster_id] = ray_dirs[dominant_ray_idx]

        return pooled

    def _pool_first_ray(
        self,
        ray_dirs: Tensor,
        cluster_ids: Tensor,
        num_clusters: int
    ) -> Tensor:
        """Select first ray per cluster (simple, preserves original direction)."""
        device = ray_dirs.device
        pooled = torch.zeros(num_clusters, 3, device=device, dtype=ray_dirs.dtype)

        # Find first occurrence of each cluster
        for cluster_id in range(num_clusters):
            mask = cluster_ids == cluster_id
            if mask.sum() == 0:
                continue

            first_idx = mask.nonzero(as_tuple=True)[0][0]
            pooled[cluster_id] = ray_dirs[first_idx]

        return pooled

