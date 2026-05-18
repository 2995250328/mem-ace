# ace_compressor.py
# GeoLMC: Geometric Latent Memory Compressor
# Ported from map-anything/mapanything/tasks/ace/compressor.py
# Self-contained — no dependency on mapanything package.

import logging
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

_logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. Base Components
# ---------------------------------------------------------------------------

class FourierPositionEncoding(nn.Module):
    """3D Positional Encoding using Random Fourier Features."""

    def __init__(self, input_dim=3, hidden_dim=256, output_dim=1024,
                 num_frequencies=10, sigma=1.0, normalize_input=False,
                 scale_mode=None, scene_scale=1.0):
        super().__init__()
        if scale_mode is None:
            scale_mode = "std" if normalize_input else "raw"
        if scale_mode not in ("raw", "std", "scene_scale"):
            raise ValueError(f"Unsupported PE scale_mode={scale_mode!r}.")
        self.normalize_input = normalize_input
        self.scale_mode = scale_mode
        scene_scale = float(scene_scale)
        if (not torch.isfinite(torch.tensor(scene_scale)).item()) or scene_scale <= 0.0:
            raise ValueError(f"scene_scale must be finite and > 0, got {scene_scale!r}")
        self.register_buffer(
            "scene_scale",
            torch.tensor(scene_scale, dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "B_gauss", torch.randn(input_dim, num_frequencies) * sigma)
        self.mlp = nn.Sequential(
            nn.Linear(num_frequencies * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, coords):
        if self.scale_mode == "std":
            coords = coords / (coords.std(dim=1, keepdim=True).clamp(min=1e-6))
        elif self.scale_mode == "scene_scale":
            scale = self.scene_scale.to(device=coords.device, dtype=coords.dtype)
            coords = coords / scale.clamp(min=1e-6)
        projected = torch.matmul(coords, self.B_gauss)
        fourier = torch.cat([torch.sin(2 * math.pi * projected),
                             torch.cos(2 * math.pi * projected)], dim=-1)
        return self.mlp(fourier)


def farthest_point_sampling(points, K, start_policy="farthest_from_center", scene_center=None, generator=None):
    """GPU-accelerated Farthest Point Sampling with explicit first-point policy."""
    B, N, C = points.shape
    device = points.device
    centroids = torch.zeros((B, K), dtype=torch.long, device=device)
    distance = torch.ones((B, N), dtype=points.dtype, device=device) * 1e10
    if start_policy == "legacy_random":
        farthest = torch.randint(0, N, (B,), dtype=torch.long, device=device, generator=generator)
    elif start_policy == "lowest_index":
        farthest = torch.zeros((B,), dtype=torch.long, device=device)
    elif start_policy == "highest_index":
        farthest = torch.full((B,), N - 1, dtype=torch.long, device=device)
    elif start_policy == "farthest_from_center":
        if scene_center is None:
            center = points.mean(dim=1)
        else:
            center = scene_center.to(device=device, dtype=points.dtype)
            if center.ndim == 1:
                center = center.unsqueeze(0).expand(B, -1)
        dist_to_center = torch.sum((points - center.unsqueeze(1)) ** 2, dim=-1)
        farthest = torch.argmax(dist_to_center, dim=1)
    else:
        raise ValueError(f"Unsupported FPS start policy: {start_policy}")
    batch_indices = torch.arange(B, dtype=torch.long, device=device)
    for i in range(K):
        centroids[:, i] = farthest
        centroid = points[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((points - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids


# ---------------------------------------------------------------------------
# 2. Geometric Bias Modules
# ---------------------------------------------------------------------------

class LocalGeometricBias(nn.Module):
    """Hard 'Spotlight' Bias for Local Mode."""

    def __init__(self, sigma=0.5, hard_cutoff_sigma=3.0):
        super().__init__()
        self.sigma = sigma
        self.hard_cutoff_sigma = hard_cutoff_sigma

    def forward(self, dist_sq):
        bias = -dist_sq / (2 * self.sigma ** 2)
        cutoff_dist_sq = (self.hard_cutoff_sigma * self.sigma) ** 2
        bias = bias.masked_fill(dist_sq > cutoff_dist_sq, -float('inf'))
        return bias.unsqueeze(1)


class AdaptiveGeometricBias(nn.Module):
    """For 'Fine' Level: Learns dynamic sigma per query."""

    def __init__(self, dim, base_sigma=0.5):
        super().__init__()
        self.base_sigma = base_sigma
        self.sigma_predictor = nn.Sequential(
            nn.Linear(dim, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, query, dist_sq):
        log_dev = torch.tanh(self.sigma_predictor(query)) * 2.0
        sigma = self.base_sigma * torch.exp(log_dev)
        bias = -dist_sq / (2 * sigma ** 2 + 1e-6)
        return bias.unsqueeze(1)


# ---------------------------------------------------------------------------
# 3. Attention Modules
# ---------------------------------------------------------------------------

class DecoupledCrossAttention(nn.Module):
    """Standard Cross Attention that accepts a pre-computed attn_bias."""

    def __init__(self, dim, num_heads=8, qkv_bias=False,
                 attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, query, key, value, attn_bias=None):
        B, N_q, C = query.shape
        N_k = key.shape[1]
        q = self.q_proj(query).reshape(B, N_q, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.k_proj(key).reshape(B, N_k, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.v_proj(value).reshape(B, N_k, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        if attn_bias is not None:
            attn = attn + attn_bias
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N_q, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class GlobalSoftAttention(nn.Module):
    """Attention that learns a Soft Bias via MLP from dist_sq."""

    def __init__(self, dim, num_heads=8, qkv_bias=False,
                 attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.geo_bias_mlp = nn.Sequential(
            nn.Linear(1, 32), nn.ReLU(), nn.Linear(32, num_heads))
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, query, key, value, dist_sq=None):
        B, N_q, C = query.shape
        N_k = key.shape[1]
        q = self.q_proj(query).reshape(B, N_q, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.k_proj(key).reshape(B, N_k, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.v_proj(value).reshape(B, N_k, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        if dist_sq is not None:
            d_log = torch.log(dist_sq.unsqueeze(-1) + 1e-6)
            geo_bias = self.geo_bias_mlp(d_log).permute(0, 3, 1, 2)
            attn = attn + geo_bias
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N_q, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


# ---------------------------------------------------------------------------
# 4. Attention Blocks
# ---------------------------------------------------------------------------

class GeoAttentionBlock(nn.Module):
    """Transformer Block for Geometric Modes (global / local / fine)."""

    def __init__(self, dim, mode, num_heads=8, geo_sigma=0.5):
        super().__init__()
        self.mode = mode
        if mode == 'global':
            self.attn = GlobalSoftAttention(dim, num_heads=num_heads)
        else:
            self.attn = DecoupledCrossAttention(dim, num_heads=num_heads)
        if mode == 'fine':
            self.fine_bias_gen = AdaptiveGeometricBias(dim, base_sigma=geo_sigma)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim))

    def forward(self, q, k, v, geometry_info=None):
        residual = q
        q_norm = self.norm1(q)
        if self.mode == 'global':
            q = residual + self.attn(q_norm, k, v, dist_sq=geometry_info)
        elif self.mode == 'local':
            q = residual + self.attn(q_norm, k, v, attn_bias=geometry_info)
        elif self.mode == 'fine':
            bias = self.fine_bias_gen(q_norm, geometry_info)
            q = residual + self.attn(q_norm, k, v, attn_bias=bias)
        q = q + self.mlp(self.norm2(q))
        return q


class StandardAttentionBlock(nn.Module):
    """For 'Learned' Mode: Pure Transformer Block."""

    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.self_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim))

    def forward(self, q, k, v):
        q = q + self.self_attn(self.norm1(q), self.norm1(q), self.norm1(q))[0]
        q = q + self.cross_attn(self.norm2(q), k, v)[0]
        q = q + self.mlp(self.norm3(q))
        return q


# ---------------------------------------------------------------------------
# 5. Main Compressor: GeoLMC
# ---------------------------------------------------------------------------

class GeoLMC(nn.Module):
    """Geometric Latent Memory Compressor.

    Compresses large-scale point cloud features into a small set of latent
    tokens via geometry-aware Transformer attention.

    Modes: 'global', 'local', 'hierarchical', 'learned'.
    """

    def __init__(self,
                 input_dim=1024,
                 compress_dim=1024,
                 num_latent_tokens=64,
                 num_fine=128,
                 num_coarse=16,
                 num_layers=4,
                 geo_sigma=0.5,
                 mode='global',
                 use_scale_token=True,
                 scale_token_dim=None,
                 num_attn_layers=2,
                 pe_normalize_input=False,
                 pe_scale_mode=None,
                 pe_scene_scale=1.0,
                 fps_start_policy="farthest_from_center",
                 key_slice_idx=None,
                 key_feature_mode="slice"):
        super().__init__()

        self.mode = mode
        self.K = num_latent_tokens
        self.K_fine = num_fine
        self.K_coarse = num_coarse
        self.num_layers = num_layers
        self.use_scale_token = use_scale_token
        self.fps_start_policy = fps_start_policy
        self.key_slice_idx = self._resolve_key_slice_idx(key_slice_idx)
        self.key_feature_mode = str(key_feature_mode)
        if self.key_feature_mode not in ("slice", "scalar_mix"):
            raise ValueError(f"Unsupported key_feature_mode={key_feature_mode!r}.")
        self.pe_scale_mode = pe_scale_mode if pe_scale_mode is not None else (
            "std" if pe_normalize_input else "raw"
        )
        self.pe_scene_scale = float(pe_scene_scale)

        # --- Input Projection ---
        self.total_input_dim = input_dim * num_layers
        self.pe_encoder = FourierPositionEncoding(
            input_dim=3,
            output_dim=compress_dim,
            normalize_input=pe_normalize_input,
            scale_mode=self.pe_scale_mode,
            scene_scale=self.pe_scene_scale,
        )
        self.k_proj = nn.Linear(input_dim, compress_dim)
        if self.key_feature_mode == "scalar_mix":
            key_mix_logits = torch.full((num_layers,), -4.0)
            key_mix_logits[self.key_slice_idx] = 4.0
            self.key_mix_logits = nn.Parameter(key_mix_logits)
        self.v_proj = nn.Sequential(
            nn.Linear(self.total_input_dim, compress_dim * 2),
            nn.GELU(),
            nn.Linear(compress_dim * 2, compress_dim),
        )

        if use_scale_token:
            s_dim = scale_token_dim if scale_token_dim is not None else input_dim
            self.scale_mlp = nn.Sequential(
                nn.Linear(s_dim, compress_dim), nn.GELU(),
                nn.Linear(compress_dim, compress_dim))

        # --- Mode-specific layers ---
        if mode == 'learned':
            self.learned_queries = nn.Parameter(
                torch.randn(1, num_latent_tokens, compress_dim))
            self.learned_layers = nn.ModuleList([
                StandardAttentionBlock(compress_dim, num_heads=8)
                for _ in range(num_attn_layers)])
            self.coord_head = nn.Sequential(
                nn.Linear(compress_dim, compress_dim // 2), nn.ReLU(),
                nn.Linear(compress_dim // 2, 3))

        elif mode == 'hierarchical':
            self.coarse_layers = nn.ModuleList([
                GeoAttentionBlock(compress_dim, mode='global', num_heads=8)
                for _ in range(num_attn_layers)])
            self.fine_layers = nn.ModuleList([
                GeoAttentionBlock(compress_dim, mode='fine', num_heads=8,
                                  geo_sigma=geo_sigma)
                for _ in range(num_attn_layers)])

        else:  # global / local
            if mode == 'local':
                self.local_bias_gen = LocalGeometricBias(
                    sigma=geo_sigma, hard_cutoff_sigma=3.0)
            self.layers = nn.ModuleList([
                GeoAttentionBlock(compress_dim, mode=mode, num_heads=8,
                                  geo_sigma=geo_sigma)
                for _ in range(num_attn_layers)])

    # -- helpers --

    def _resolve_key_slice_idx(self, key_slice_idx):
        if self.num_layers <= 1:
            if key_slice_idx not in (None, 0):
                raise ValueError(
                    f"key_slice_idx must be 0/None for num_layers={self.num_layers}, got {key_slice_idx}."
                )
            return 0
        if key_slice_idx is None:
            # Legacy-compatible default: previous _get_layer_slice(..., 1)
            # selected slice 2 for multi-layer MapAnything memory.
            return min(2, self.num_layers - 1)
        key_slice_idx = int(key_slice_idx)
        if key_slice_idx < 0 or key_slice_idx >= self.num_layers:
            raise ValueError(
                f"key_slice_idx={key_slice_idx} out of range for num_layers={self.num_layers}."
            )
        return key_slice_idx

    def _get_layer_slice(self, features, slice_idx):
        if self.num_layers <= 1:
            return features  # single-layer: return as-is
        C = features.shape[-1] // self.num_layers
        start = slice_idx * C
        end = start + C
        return features[:, :, start:end]

    def _get_key_features(self, features):
        if self.key_feature_mode == "slice":
            return self._get_layer_slice(features, self.key_slice_idx)
        B, N, total_c = features.shape
        C = total_c // self.num_layers
        per_layer = features.reshape(B, N, self.num_layers, C)
        weights = torch.softmax(self.key_mix_logits, dim=0).to(
            device=features.device,
            dtype=features.dtype,
        )
        return torch.sum(per_layer * weights.view(1, 1, self.num_layers, 1), dim=2)

    def _inject_scale(self, x, memory_dict):
        if self.use_scale_token and "all_scale_tokens" in memory_dict:
            scales = memory_dict["all_scale_tokens"]
            if scales is not None:
                global_scale = scales.mean(dim=1)
                scale_emb = self.scale_mlp(global_scale).unsqueeze(1)
                x = x + scale_emb
        return x

    # -- forward --

    def forward(self, memory_dict):
        """
        Args:
            memory_dict: dict with keys
                pooled_points   (B, N, 3)
                pooled_features (B, N, C*num_layers)
                scene_center    (B, 3)
                all_scale_tokens (B, V, D)  [optional]
        Returns:
            global/local  -> (latent_z, latent_coords)
            hierarchical  -> dict{z_coarse, p_coarse, z_fine, p_fine}
            learned       -> (latent_z, pred_coords)
        """
        pooled_points = memory_dict["pooled_points"]
        pooled_features = memory_dict["pooled_features"]
        scene_center = memory_dict["scene_center"]

        raw_key_feats = self._get_key_features(pooled_features)
        k_base = self.k_proj(raw_key_feats)
        v = self.v_proj(pooled_features)

        # --- Learned ---
        if self.mode == 'learned':
            B = pooled_points.shape[0]
            q = self.learned_queries.expand(B, -1, -1)
            k_pos = self.pe_encoder(pooled_points - scene_center.unsqueeze(1))
            k = k_base + k_pos
            for layer in self.learned_layers:
                q = layer(q, k, v)
            q = self._inject_scale(q, memory_dict)
            pred_coords = self.coord_head(q) + scene_center.unsqueeze(1)
            return q, pred_coords

        # --- Hierarchical ---
        if self.mode == 'hierarchical':
            idx_fine = farthest_point_sampling(
                pooled_points,
                self.K_fine,
                start_policy=self.fps_start_policy,
                scene_center=scene_center,
            )
            coords_fine = torch.gather(
                pooled_points, 1,
                idx_fine.unsqueeze(-1).expand(-1, -1, 3))
            idx_coarse_local = farthest_point_sampling(
                coords_fine,
                self.K_coarse,
                start_policy=self.fps_start_policy,
                scene_center=scene_center,
            )
            coords_coarse = torch.gather(
                coords_fine, 1,
                idx_coarse_local.unsqueeze(-1).expand(-1, -1, 3))

            dist_coarse = torch.cdist(coords_coarse, pooled_points, p=2) ** 2
            q_coarse = self.pe_encoder(
                coords_coarse - scene_center.unsqueeze(1))
            for layer in self.coarse_layers:
                q_coarse = layer(q_coarse, k_base, v,
                                 geometry_info=dist_coarse)
            z_coarse = self._inject_scale(q_coarse, memory_dict)

            dist_fine = torch.cdist(coords_fine, pooled_points, p=2) ** 2
            q_fine = self.pe_encoder(
                coords_fine - scene_center.unsqueeze(1))
            for layer in self.fine_layers:
                q_fine = layer(q_fine, k_base, v, geometry_info=dist_fine)
            z_fine = self._inject_scale(q_fine, memory_dict)

            return {"z_coarse": z_coarse, "p_coarse": coords_coarse,
                    "z_fine": z_fine, "p_fine": coords_fine}

        # --- Single Level (global / local) ---
        K_curr = self.K
        if pooled_points.shape[1] > K_curr:
            fps_idx = farthest_point_sampling(
                pooled_points,
                K_curr,
                start_policy=self.fps_start_policy,
                scene_center=scene_center,
            )
            latent_coords = torch.gather(
                pooled_points, 1,
                fps_idx.unsqueeze(-1).expand(-1, -1, 3))
        else:
            latent_coords = pooled_points[:, :K_curr, :]

        norm_coords = latent_coords - scene_center.unsqueeze(1)
        q = self.pe_encoder(norm_coords)

        # One-time PE diagnostic log
        if not hasattr(self, '_pe_logged'):
            _logger.info(
                "[PE] scale_mode=%s scene_scale=%.6f input range: %.3f~%.3f, "
                "input_std=%.4f, output range: %.3f~%.3f",
                self.pe_scale_mode,
                self.pe_scene_scale,
                float(norm_coords.min()), float(norm_coords.max()),
                float(norm_coords.std(unbiased=False)),
                float(q.min()), float(q.max()),
            )
            self._pe_logged = True

        dist_sq = torch.cdist(latent_coords, pooled_points, p=2) ** 2
        if self.mode == 'local':
            geometry_info = self.local_bias_gen(dist_sq)
        else:
            geometry_info = dist_sq

        for layer in self.layers:
            q = layer(q, k_base, v, geometry_info=geometry_info)

        x = self._inject_scale(q, memory_dict)
        return x, latent_coords
