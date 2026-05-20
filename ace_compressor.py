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


def _summarize_attention_scalars(attn):
    """Return scalar-only attention diagnostics."""
    with torch.no_grad():
        attn_eval = attn.detach().float()
        eps = 1e-8
        entropy = -(attn_eval * (attn_eval + eps).log()).sum(dim=-1)
        usage = attn_eval.mean(dim=(0, 1, 2))
        usage = usage / usage.sum().clamp(min=eps)
        usage_entropy = -(usage * (usage + eps).log()).sum()
        max_attn = attn_eval.max(dim=-1).values
        return {
            "attn_entropy_mean": float(entropy.mean().item()),
            "effective_token_count": float(torch.exp(usage_entropy).item()),
            "avg_max_attention": float(max_attn.mean().item()),
        }


# ---------------------------------------------------------------------------
# 1. Base Components
# ---------------------------------------------------------------------------

class FourierPositionEncoding(nn.Module):
    """3D positional encoding with legacy and residual-v2 modes."""

    def __init__(self, input_dim=3, hidden_dim=256, output_dim=1024,
                 num_frequencies=10, sigma=1.0, normalize_input=False,
                 scale_mode=None, scene_scale=1.0,
                 mode="fourier_legacy",
                 fourier_v2_scales=None,
                 coord_norm="scene_radius",
                 radius=4.0,
                 learnable_scale=False,
                 residual_gate_init=0.0):
        super().__init__()
        if scale_mode is None:
            scale_mode = "std" if normalize_input else "raw"
        if scale_mode not in ("raw", "std", "scene_scale"):
            raise ValueError(f"Unsupported PE scale_mode={scale_mode!r}.")
        mode = str(mode)
        if mode not in ("fourier_legacy", "fourier_v2"):
            raise ValueError(f"Unsupported PE mode={mode!r}.")
        coord_norm = str(coord_norm)
        if coord_norm not in ("scene_radius",):
            raise ValueError(f"Unsupported coord_norm={coord_norm!r}.")
        radius = float(radius)
        if (not torch.isfinite(torch.tensor(radius)).item()) or radius <= 0.0:
            raise ValueError(f"radius must be finite and > 0, got {radius!r}")
        if fourier_v2_scales is None:
            fourier_v2_scales = [1.0, 2.0, 4.0, 8.0, 16.0]
        fourier_v2_scales = [float(v) for v in fourier_v2_scales]
        if len(fourier_v2_scales) == 0 or any(v <= 0.0 for v in fourier_v2_scales):
            raise ValueError(f"Invalid fourier_v2_scales={fourier_v2_scales!r}")

        self.normalize_input = normalize_input
        self.scale_mode = scale_mode
        self.mode = mode
        self.coord_norm = coord_norm
        self.learnable_scale = bool(learnable_scale)
        scene_scale = float(scene_scale)
        if (not torch.isfinite(torch.tensor(scene_scale)).item()) or scene_scale <= 0.0:
            raise ValueError(f"scene_scale must be finite and > 0, got {scene_scale!r}")
        self.register_buffer(
            "scene_scale",
            torch.tensor(scene_scale, dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "v2_radius",
            torch.tensor(radius, dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "B_gauss", torch.randn(input_dim, num_frequencies) * sigma)
        self.mlp = nn.Sequential(
            nn.Linear(num_frequencies * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

        scales_tensor = torch.tensor(fourier_v2_scales, dtype=torch.float32)
        if self.learnable_scale:
            self.v2_scales = nn.Parameter(scales_tensor)
        else:
            self.register_buffer("v2_scales", scales_tensor, persistent=False)
        self.residual_gate = None
        self.v2_mlp = None
        if self.mode == "fourier_v2":
            self.residual_gate = nn.Parameter(
                torch.tensor(float(residual_gate_init), dtype=torch.float32)
            )
            fourier_dim = input_dim * scales_tensor.numel() * 2
            self.v2_mlp = nn.Sequential(
                nn.Linear(fourier_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, output_dim),
            )

    def _encode_legacy(self, coords):
        if self.scale_mode == "std":
            coords = coords / (coords.std(dim=1, keepdim=True).clamp(min=1e-6))
        elif self.scale_mode == "scene_scale":
            scale = self.scene_scale.to(device=coords.device, dtype=coords.dtype)
            coords = coords / scale.clamp(min=1e-6)
        projected = torch.matmul(coords, self.B_gauss)
        fourier = torch.cat([torch.sin(2 * math.pi * projected),
                             torch.cos(2 * math.pi * projected)], dim=-1)
        return self.mlp(fourier)

    def _encode_v2_residual(self, coords):
        if self.coord_norm == "scene_radius":
            radius = self.v2_radius.to(device=coords.device, dtype=coords.dtype)
            coords = coords / radius.clamp(min=1e-6)
        scales = self.v2_scales.to(device=coords.device, dtype=coords.dtype)
        projected = coords.unsqueeze(-1) * scales.view(1, 1, 1, -1)
        fourier = torch.cat([torch.sin(2 * math.pi * projected),
                             torch.cos(2 * math.pi * projected)], dim=-1)
        fourier = fourier.reshape(coords.shape[0], coords.shape[1], -1)
        return self.v2_mlp(fourier)

    def forward(self, coords):
        legacy = self._encode_legacy(coords)
        if self.mode == "fourier_legacy":
            return legacy
        gate = self.residual_gate.to(device=coords.device, dtype=coords.dtype)
        return legacy + gate * self._encode_v2_residual(coords)


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


class ResidualRBFDistanceBias(nn.Module):
    """Residual multi-scale RBF bias added to attention logits."""

    def __init__(self, num_heads, mode="legacy", scales=None,
                 alpha_init=0.0, learn_weights=True, per_head=False):
        super().__init__()
        self.num_heads = int(num_heads)
        self.mode = str(mode)
        if self.mode not in ("legacy", "rbf_residual", "crpb"):
            raise ValueError(f"Unsupported geo_bias_mode={mode!r}.")
        if scales is None:
            scales = [0.25, 0.5, 1.0, 2.0, 4.0]
        scales = [float(v) for v in scales]
        if len(scales) == 0 or any(v <= 0.0 for v in scales):
            raise ValueError(f"Invalid RBF scales={scales!r}.")
        self.learn_weights = bool(learn_weights)
        self.per_head = bool(per_head)
        self.register_buffer(
            "rbf_scales",
            torch.tensor(scales, dtype=torch.float32),
            persistent=False,
        )
        self.alpha = None
        self.weight_logits = None
        if self.mode == "rbf_residual":
            self.alpha = nn.Parameter(torch.tensor(float(alpha_init), dtype=torch.float32))
            shape = (self.num_heads if self.per_head else 1, len(scales))
            weight_logits = torch.zeros(shape, dtype=torch.float32)
            if self.learn_weights:
                self.weight_logits = nn.Parameter(weight_logits)
            else:
                self.register_buffer("weight_logits", weight_logits, persistent=False)

    def forward(self, dist_sq):
        if self.mode != "rbf_residual" or dist_sq is None:
            return None
        scales = self.rbf_scales.to(device=dist_sq.device, dtype=dist_sq.dtype)
        denom = 2.0 * scales.pow(2).view(1, 1, 1, 1, -1).clamp(min=1e-6)
        basis = torch.exp(-dist_sq.unsqueeze(1).unsqueeze(-1) / denom)
        weights = torch.softmax(
            self.weight_logits.to(device=dist_sq.device, dtype=dist_sq.dtype),
            dim=-1,
        )
        if self.per_head:
            basis = basis.expand(-1, self.num_heads, -1, -1, -1)
            weights = weights.view(1, self.num_heads, 1, 1, -1)
        else:
            weights = weights.view(1, 1, 1, 1, -1)
        alpha = self.alpha.to(device=dist_sq.device, dtype=dist_sq.dtype).view(1, 1, 1, 1)
        return alpha * (basis * weights).sum(dim=-1)

    def summary(self):
        if self.mode != "rbf_residual":
            return None
        weights = torch.softmax(self.weight_logits.detach().float().cpu(), dim=-1)
        weights_out = weights.tolist()
        if not self.per_head and len(weights_out) == 1:
            weights_out = weights_out[0]
        return {
            "final_geo_bias_rbf_alpha": float(self.alpha.detach().float().cpu().item()),
            "final_geo_bias_rbf_weights": weights_out,
        }


# ---------------------------------------------------------------------------
# 3. Attention Modules
# ---------------------------------------------------------------------------

class DecoupledCrossAttention(nn.Module):
    """Standard Cross Attention that accepts optional geometric biases."""

    def __init__(self, dim, num_heads=8, qkv_bias=False,
                 attn_drop=0., proj_drop=0.,
                 geo_bias_mode="legacy",
                 geo_bias_rbf_scales=None,
                 geo_bias_rbf_alpha_init=0.0,
                 geo_bias_rbf_learn_weights=True,
                 geo_bias_rbf_per_head=False,
                 geo_bias_crpb_dim=32,
                 geo_bias_crpb_input="delta_dist_log",
                 geo_bias_crpb_radius=4.0,
                 geo_bias_crpb_per_head=False,
                 geo_bias_crpb_zero_init=True,
                 pos_encoding_mode="fourier_legacy",
                 point_rope_coord_norm="scene_radius",
                 point_rope_radius=4.0,
                 point_rope_base=10000.0,
                 point_rope_axes="xyz_split",
                 point_rope_apply_to="qk"):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.rbf_bias = ResidualRBFDistanceBias(
            num_heads=num_heads,
            mode=geo_bias_mode,
            scales=geo_bias_rbf_scales,
            alpha_init=geo_bias_rbf_alpha_init,
            learn_weights=geo_bias_rbf_learn_weights,
            per_head=geo_bias_rbf_per_head,
        )
        self.geo_bias_mode = str(geo_bias_mode)
        self.geo_bias_crpb_radius = float(geo_bias_crpb_radius)
        self.pos_encoding_mode = str(pos_encoding_mode)
        self.point_rope_coord_norm = str(point_rope_coord_norm)
        self.point_rope_radius = float(point_rope_radius)
        self.point_rope_base = float(point_rope_base)
        self.point_rope_axes = str(point_rope_axes)
        self.point_rope_apply_to = str(point_rope_apply_to)
        self.crpb = None
        if self.geo_bias_mode == 'crpb':
            out_dim = num_heads if bool(geo_bias_crpb_per_head) else 1
            self.crpb = nn.Sequential(
                nn.Linear(5, int(geo_bias_crpb_dim)),
                nn.GELU(),
                nn.Linear(int(geo_bias_crpb_dim), out_dim),
            )
            if bool(geo_bias_crpb_zero_init):
                nn.init.zeros_(self.crpb[-1].weight)
                nn.init.zeros_(self.crpb[-1].bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.collect_runtime_stats = False
        self.last_runtime_stats = None

    def _apply_point_rope(self, x_bhnd, coords_bnd):
        if coords_bnd is None or self.pos_encoding_mode != 'point_rope':
            return x_bhnd
        if self.point_rope_coord_norm != 'scene_radius' or self.point_rope_axes != 'xyz_split':
            return x_bhnd
        B, H, N, D = x_bhnd.shape
        split = D // 3
        seg = (split // 2) * 2
        if seg <= 0:
            return x_bhnd
        coords = coords_bnd.to(device=x_bhnd.device, dtype=x_bhnd.dtype) / max(self.point_rope_radius, 1e-6)
        out = x_bhnd
        base = torch.tensor(self.point_rope_base, device=x_bhnd.device, dtype=x_bhnd.dtype).clamp(min=1.0001)
        freq_idx = torch.arange(0, seg, 2, device=x_bhnd.device, dtype=x_bhnd.dtype)
        inv_freq = torch.pow(base, -freq_idx / max(float(seg), 1.0))
        for axis in range(3):
            st = axis * split
            ed = st + seg
            if ed > D:
                continue
            part = out[..., st:ed]
            angle = coords[..., axis].unsqueeze(1).unsqueeze(-1) * inv_freq.view(1, 1, 1, -1)
            cos = torch.cos(angle)
            sin = torch.sin(angle)
            even = part[..., 0::2]
            odd = part[..., 1::2]
            rot = torch.stack([even * cos - odd * sin, even * sin + odd * cos], dim=-1).reshape_as(part)
            out = torch.cat([out[..., :st], rot, out[..., ed:]], dim=-1)
        return out

    def _crpb_bias(self, query_coords, key_coords, dtype, device):
        if self.crpb is None or query_coords is None or key_coords is None:
            return None
        delta = (key_coords.unsqueeze(1) - query_coords.unsqueeze(2)) / max(self.geo_bias_crpb_radius, 1e-6)
        dist = torch.linalg.norm(delta, dim=-1, keepdim=True)
        feat = torch.cat([delta, dist, torch.log(dist + 1e-6)], dim=-1).to(device=device, dtype=dtype)
        out = self.crpb(feat).permute(0, 3, 1, 2)
        return out

    def forward(self, query, key, value, attn_bias=None, dist_sq=None, query_coords=None, key_coords=None):
        B, N_q, C = query.shape
        N_k = key.shape[1]
        q = self.q_proj(query).reshape(B, N_q, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.k_proj(key).reshape(B, N_k, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.v_proj(value).reshape(B, N_k, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        if self.point_rope_apply_to in ('qk', 'q'):
            q = self._apply_point_rope(q, query_coords)
        if self.point_rope_apply_to in ('qk', 'k'):
            k = self._apply_point_rope(k, key_coords)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        if attn_bias is not None:
            attn = attn + attn_bias
        rbf_bias = self.rbf_bias(dist_sq)
        if rbf_bias is not None:
            attn = attn + rbf_bias
        crpb = self._crpb_bias(query_coords, key_coords, dtype=attn.dtype, device=attn.device)
        if crpb is not None:
            attn = attn + crpb
        attn = attn.softmax(dim=-1)
        self.last_runtime_stats = _summarize_attention_scalars(attn) if self.collect_runtime_stats else None
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N_q, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class GlobalSoftAttention(nn.Module):
    """Attention that learns a soft bias from distances."""

    def __init__(self, dim, num_heads=8, qkv_bias=False,
                 attn_drop=0., proj_drop=0.,
                 geo_bias_mode="legacy",
                 geo_bias_rbf_scales=None,
                 geo_bias_rbf_alpha_init=0.0,
                 geo_bias_rbf_learn_weights=True,
                 geo_bias_rbf_per_head=False,
                 geo_bias_crpb_dim=32,
                 geo_bias_crpb_input="delta_dist_log",
                 geo_bias_crpb_radius=4.0,
                 geo_bias_crpb_per_head=False,
                 geo_bias_crpb_zero_init=True,
                 pos_encoding_mode="fourier_legacy",
                 point_rope_coord_norm="scene_radius",
                 point_rope_radius=4.0,
                 point_rope_base=10000.0,
                 point_rope_axes="xyz_split",
                 point_rope_apply_to="qk"):
        super().__init__()
        self.geo_bias_mlp = nn.Sequential(nn.Linear(1, 32), nn.ReLU(), nn.Linear(32, num_heads))
        self.inner = DecoupledCrossAttention(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            geo_bias_mode=geo_bias_mode,
            geo_bias_rbf_scales=geo_bias_rbf_scales,
            geo_bias_rbf_alpha_init=geo_bias_rbf_alpha_init,
            geo_bias_rbf_learn_weights=geo_bias_rbf_learn_weights,
            geo_bias_rbf_per_head=geo_bias_rbf_per_head,
            geo_bias_crpb_dim=geo_bias_crpb_dim,
            geo_bias_crpb_input=geo_bias_crpb_input,
            geo_bias_crpb_radius=geo_bias_crpb_radius,
            geo_bias_crpb_per_head=geo_bias_crpb_per_head,
            geo_bias_crpb_zero_init=geo_bias_crpb_zero_init,
            pos_encoding_mode=pos_encoding_mode,
            point_rope_coord_norm=point_rope_coord_norm,
            point_rope_radius=point_rope_radius,
            point_rope_base=point_rope_base,
            point_rope_axes=point_rope_axes,
            point_rope_apply_to=point_rope_apply_to,
        )
        self.collect_runtime_stats = False
        self.last_runtime_stats = None

    def forward(self, query, key, value, dist_sq=None, query_coords=None, key_coords=None):
        attn_bias = None
        if dist_sq is not None:
            d_log = torch.log(dist_sq.unsqueeze(-1) + 1e-6)
            attn_bias = self.geo_bias_mlp(d_log).permute(0, 3, 1, 2)
        self.inner.collect_runtime_stats = self.collect_runtime_stats
        out = self.inner(
            query, key, value,
            attn_bias=attn_bias,
            dist_sq=dist_sq,
            query_coords=query_coords,
            key_coords=key_coords,
        )
        self.last_runtime_stats = self.inner.last_runtime_stats
        return out


class GeoAttentionBlock(nn.Module):
    """Transformer Block for Geometric Modes (global / local / fine)."""

    def __init__(self, dim, mode, num_heads=8, geo_sigma=0.5,
                 geo_bias_mode="legacy",
                 geo_bias_rbf_scales=None,
                 geo_bias_rbf_alpha_init=0.0,
                 geo_bias_rbf_learn_weights=True,
                 geo_bias_rbf_per_head=False,
                 geo_bias_crpb_dim=32,
                 geo_bias_crpb_input="delta_dist_log",
                 geo_bias_crpb_radius=4.0,
                 geo_bias_crpb_per_head=False,
                 geo_bias_crpb_zero_init=True,
                 pos_encoding_mode="fourier_legacy",
                 point_rope_coord_norm="scene_radius",
                 point_rope_radius=4.0,
                 point_rope_base=10000.0,
                 point_rope_axes="xyz_split",
                 point_rope_apply_to="qk"):
        super().__init__()
        self.mode = mode
        attn_kwargs = dict(
            geo_bias_mode=geo_bias_mode,
            geo_bias_rbf_scales=geo_bias_rbf_scales,
            geo_bias_rbf_alpha_init=geo_bias_rbf_alpha_init,
            geo_bias_rbf_learn_weights=geo_bias_rbf_learn_weights,
            geo_bias_rbf_per_head=geo_bias_rbf_per_head,
            geo_bias_crpb_dim=geo_bias_crpb_dim,
            geo_bias_crpb_input=geo_bias_crpb_input,
            geo_bias_crpb_radius=geo_bias_crpb_radius,
            geo_bias_crpb_per_head=geo_bias_crpb_per_head,
            geo_bias_crpb_zero_init=geo_bias_crpb_zero_init,
            pos_encoding_mode=pos_encoding_mode,
            point_rope_coord_norm=point_rope_coord_norm,
            point_rope_radius=point_rope_radius,
            point_rope_base=point_rope_base,
            point_rope_axes=point_rope_axes,
            point_rope_apply_to=point_rope_apply_to,
        )
        if mode == 'global':
            self.attn = GlobalSoftAttention(dim, num_heads=num_heads, **attn_kwargs)
        else:
            self.attn = DecoupledCrossAttention(dim, num_heads=num_heads, **attn_kwargs)
        if mode == 'fine':
            self.fine_bias_gen = AdaptiveGeometricBias(dim, base_sigma=geo_sigma)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim))

    def forward(self, q, k, v, geometry_info=None):
        residual = q
        q_norm = self.norm1(q)
        if isinstance(geometry_info, dict):
            dist_sq = geometry_info.get("dist_sq")
            attn_bias = geometry_info.get("attn_bias")
            query_coords = geometry_info.get("query_coords")
            key_coords = geometry_info.get("key_coords")
        else:
            dist_sq = geometry_info
            attn_bias = geometry_info
            query_coords = None
            key_coords = None
        if self.mode == 'global':
            q = residual + self.attn(q_norm, k, v, dist_sq=dist_sq, query_coords=query_coords, key_coords=key_coords)
        elif self.mode == 'local':
            q = residual + self.attn(q_norm, k, v, attn_bias=attn_bias, dist_sq=dist_sq, query_coords=query_coords, key_coords=key_coords)
        elif self.mode == 'fine':
            bias = self.fine_bias_gen(q_norm, dist_sq)
            q = residual + self.attn(q_norm, k, v, attn_bias=bias, dist_sq=dist_sq, query_coords=query_coords, key_coords=key_coords)
        q = q + self.mlp(self.norm2(q))
        return q


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
                 key_feature_mode="slice",
                 feature_hierarchy_mode="selected_key_concat_value",
                 level_merge_mode="softmax_gate",
                 level_merge_init="uniform",
                 level_proj_shared=False,
                 level_cross_attn_shared=True,
                 level_gate_entropy_weight=0.0,
                 level_token_gate=False,
                 geo_bias_mode="legacy",
                 geo_bias_rbf_scales=(0.25, 0.5, 1.0, 2.0, 4.0),
                 geo_bias_rbf_alpha_init=0.0,
                 geo_bias_rbf_learn_weights=True,
                 geo_bias_rbf_per_head=False,
                 pos_encoding_mode="fourier_legacy",
                 point_rope_coord_norm="scene_radius",
                 point_rope_radius=4.0,
                 point_rope_base=10000.0,
                 point_rope_axes="xyz_split",
                 point_rope_apply_to="qk",
                 geo_bias_crpb_dim=32,
                 geo_bias_crpb_input="delta_dist_log",
                 geo_bias_crpb_radius=4.0,
                 geo_bias_crpb_per_head=False,
                 geo_bias_crpb_zero_init=True,
                 pos_fourier_v2_scales=(1.0, 2.0, 4.0, 8.0, 16.0),
                 pos_fourier_coord_norm="scene_radius",
                 pos_fourier_radius=4.0,
                 pos_fourier_learnable_scale=False,
                 pos_fourier_residual_gate_init=0.0):
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
        self.feature_hierarchy_mode = str(feature_hierarchy_mode)
        if self.feature_hierarchy_mode not in ("selected_key_concat_value", "levelwise_latent_merge"):
            raise ValueError(f"Unsupported feature_hierarchy_mode={feature_hierarchy_mode!r}.")
        self.level_merge_mode = str(level_merge_mode)
        if self.level_merge_mode != "softmax_gate":
            raise ValueError(f"Unsupported level_merge_mode={level_merge_mode!r}.")
        self.level_merge_init = str(level_merge_init)
        if self.level_merge_init != "uniform":
            raise ValueError(f"Unsupported level_merge_init={level_merge_init!r}.")
        self.level_proj_shared = bool(level_proj_shared)
        self.level_cross_attn_shared = bool(level_cross_attn_shared)
        self.level_gate_entropy_weight = float(level_gate_entropy_weight)
        self.level_token_gate = bool(level_token_gate)
        if self.feature_hierarchy_mode == "levelwise_latent_merge":
            if self.mode not in ("global", "local"):
                raise ValueError(
                    "levelwise_latent_merge currently supports only global/local GeoLMC modes, "
                    f"got mode={self.mode!r}."
                )
            if self.level_token_gate:
                raise ValueError("level_token_gate is reserved for a future ablation; use False for B3-lite.")
            if self.level_gate_entropy_weight != 0.0:
                raise ValueError("level_gate_entropy_weight is reserved for a future ablation; use 0.0 for B3-lite.")
        self.geo_bias_mode = str(geo_bias_mode)
        if self.geo_bias_mode not in ("legacy", "rbf_residual", "crpb"):
            raise ValueError(f"Unsupported geo_bias_mode={geo_bias_mode!r}.")
        self.geo_bias_rbf_scales = [float(v) for v in geo_bias_rbf_scales]
        if len(self.geo_bias_rbf_scales) == 0 or any(v <= 0.0 for v in self.geo_bias_rbf_scales):
            raise ValueError(f"Invalid geo_bias_rbf_scales={self.geo_bias_rbf_scales!r}.")
        self.geo_bias_rbf_alpha_init = float(geo_bias_rbf_alpha_init)
        self.geo_bias_rbf_learn_weights = bool(geo_bias_rbf_learn_weights)
        self.geo_bias_rbf_per_head = bool(geo_bias_rbf_per_head)
        self.pos_encoding_mode = str(pos_encoding_mode)
        if self.pos_encoding_mode not in ("fourier_legacy", "fourier_v2", "point_rope"):
            raise ValueError(f"Unsupported pos_encoding_mode={pos_encoding_mode!r}.")
        self.pos_fourier_v2_scales = [float(v) for v in pos_fourier_v2_scales]
        if len(self.pos_fourier_v2_scales) == 0 or any(v <= 0.0 for v in self.pos_fourier_v2_scales):
            raise ValueError(f"Invalid pos_fourier_v2_scales={self.pos_fourier_v2_scales!r}.")
        self.pos_fourier_coord_norm = str(pos_fourier_coord_norm)
        if self.pos_fourier_coord_norm not in ("scene_radius",):
            raise ValueError(f"Unsupported pos_fourier_coord_norm={pos_fourier_coord_norm!r}.")
        self.pos_fourier_radius = float(pos_fourier_radius)
        if self.pos_fourier_radius <= 0.0:
            raise ValueError(f"pos_fourier_radius must be > 0, got {self.pos_fourier_radius!r}.")
        self.pos_fourier_learnable_scale = bool(pos_fourier_learnable_scale)
        self.pos_fourier_residual_gate_init = float(pos_fourier_residual_gate_init)
        self.point_rope_coord_norm = str(point_rope_coord_norm)
        self.point_rope_radius = float(point_rope_radius)
        self.point_rope_base = float(point_rope_base)
        self.point_rope_axes = str(point_rope_axes)
        self.point_rope_apply_to = str(point_rope_apply_to)
        self.geo_bias_crpb_dim = int(geo_bias_crpb_dim)
        self.geo_bias_crpb_input = str(geo_bias_crpb_input)
        self.geo_bias_crpb_radius = float(geo_bias_crpb_radius)
        self.geo_bias_crpb_per_head = bool(geo_bias_crpb_per_head)
        self.geo_bias_crpb_zero_init = bool(geo_bias_crpb_zero_init)
        self.pe_scale_mode = pe_scale_mode if pe_scale_mode is not None else (
            "std" if pe_normalize_input else "raw"
        )
        self.pe_scene_scale = float(pe_scene_scale)
        self.last_geo_bias_runtime_stats = None

        # --- Input Projection ---
        self.total_input_dim = input_dim * num_layers
        self.pe_encoder = FourierPositionEncoding(
            input_dim=3,
            output_dim=compress_dim,
            normalize_input=pe_normalize_input,
            scale_mode=self.pe_scale_mode,
            scene_scale=self.pe_scene_scale,
            mode="fourier_legacy" if self.pos_encoding_mode == "point_rope" else self.pos_encoding_mode,
            fourier_v2_scales=self.pos_fourier_v2_scales,
            coord_norm=self.pos_fourier_coord_norm,
            radius=self.pos_fourier_radius,
            learnable_scale=self.pos_fourier_learnable_scale,
            residual_gate_init=self.pos_fourier_residual_gate_init,
        )
        if self.feature_hierarchy_mode == "levelwise_latent_merge":
            self.level_latent_tokens = nn.Parameter(
                torch.zeros(1, num_latent_tokens, compress_dim)
            )
            if self.level_proj_shared:
                self.level_proj = nn.Sequential(
                    nn.Linear(input_dim, compress_dim),
                    nn.LayerNorm(compress_dim),
                )
            else:
                self.level_proj = nn.ModuleList([
                    nn.Sequential(
                        nn.Linear(input_dim, compress_dim),
                        nn.LayerNorm(compress_dim),
                    )
                    for _ in range(num_layers)
                ])
            self.level_logits = nn.Parameter(torch.zeros(num_layers))
        else:
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
                GeoAttentionBlock(compress_dim, mode='global', num_heads=8,
                                  geo_bias_mode=self.geo_bias_mode,
                                  geo_bias_rbf_scales=self.geo_bias_rbf_scales,
                                  geo_bias_rbf_alpha_init=self.geo_bias_rbf_alpha_init,
                                  geo_bias_rbf_learn_weights=self.geo_bias_rbf_learn_weights,
                                  geo_bias_rbf_per_head=self.geo_bias_rbf_per_head,
                                  geo_bias_crpb_dim=self.geo_bias_crpb_dim,
                                  geo_bias_crpb_input=self.geo_bias_crpb_input,
                                  geo_bias_crpb_radius=self.geo_bias_crpb_radius,
                                  geo_bias_crpb_per_head=self.geo_bias_crpb_per_head,
                                  geo_bias_crpb_zero_init=self.geo_bias_crpb_zero_init,
                                  pos_encoding_mode=self.pos_encoding_mode,
                                  point_rope_coord_norm=self.point_rope_coord_norm,
                                  point_rope_radius=self.point_rope_radius,
                                  point_rope_base=self.point_rope_base,
                                  point_rope_axes=self.point_rope_axes,
                                  point_rope_apply_to=self.point_rope_apply_to)
                for _ in range(num_attn_layers)])
            self.fine_layers = nn.ModuleList([
                GeoAttentionBlock(compress_dim, mode='fine', num_heads=8,
                                  geo_sigma=geo_sigma,
                                  geo_bias_mode=self.geo_bias_mode,
                                  geo_bias_rbf_scales=self.geo_bias_rbf_scales,
                                  geo_bias_rbf_alpha_init=self.geo_bias_rbf_alpha_init,
                                  geo_bias_rbf_learn_weights=self.geo_bias_rbf_learn_weights,
                                  geo_bias_rbf_per_head=self.geo_bias_rbf_per_head,
                                  geo_bias_crpb_dim=self.geo_bias_crpb_dim,
                                  geo_bias_crpb_input=self.geo_bias_crpb_input,
                                  geo_bias_crpb_radius=self.geo_bias_crpb_radius,
                                  geo_bias_crpb_per_head=self.geo_bias_crpb_per_head,
                                  geo_bias_crpb_zero_init=self.geo_bias_crpb_zero_init,
                                  pos_encoding_mode=self.pos_encoding_mode,
                                  point_rope_coord_norm=self.point_rope_coord_norm,
                                  point_rope_radius=self.point_rope_radius,
                                  point_rope_base=self.point_rope_base,
                                  point_rope_axes=self.point_rope_axes,
                                  point_rope_apply_to=self.point_rope_apply_to)
                for _ in range(num_attn_layers)])

        else:  # global / local
            if mode == 'local':
                self.local_bias_gen = LocalGeometricBias(
                    sigma=geo_sigma, hard_cutoff_sigma=3.0)
            if self.feature_hierarchy_mode == "levelwise_latent_merge":
                if self.level_cross_attn_shared:
                    self.level_layers = nn.ModuleList([
                        GeoAttentionBlock(compress_dim, mode=mode, num_heads=8,
                                          geo_sigma=geo_sigma,
                                          geo_bias_mode=self.geo_bias_mode,
                                          geo_bias_rbf_scales=self.geo_bias_rbf_scales,
                                          geo_bias_rbf_alpha_init=self.geo_bias_rbf_alpha_init,
                                          geo_bias_rbf_learn_weights=self.geo_bias_rbf_learn_weights,
                                          geo_bias_rbf_per_head=self.geo_bias_rbf_per_head,
                                  geo_bias_crpb_dim=self.geo_bias_crpb_dim,
                                  geo_bias_crpb_input=self.geo_bias_crpb_input,
                                  geo_bias_crpb_radius=self.geo_bias_crpb_radius,
                                  geo_bias_crpb_per_head=self.geo_bias_crpb_per_head,
                                  geo_bias_crpb_zero_init=self.geo_bias_crpb_zero_init,
                                  pos_encoding_mode=self.pos_encoding_mode,
                                  point_rope_coord_norm=self.point_rope_coord_norm,
                                  point_rope_radius=self.point_rope_radius,
                                  point_rope_base=self.point_rope_base,
                                  point_rope_axes=self.point_rope_axes,
                                  point_rope_apply_to=self.point_rope_apply_to)
                        for _ in range(num_attn_layers)])
                else:
                    self.level_layers = nn.ModuleList([
                        nn.ModuleList([
                            GeoAttentionBlock(compress_dim, mode=mode, num_heads=8,
                                              geo_sigma=geo_sigma,
                                              geo_bias_mode=self.geo_bias_mode,
                                              geo_bias_rbf_scales=self.geo_bias_rbf_scales,
                                              geo_bias_rbf_alpha_init=self.geo_bias_rbf_alpha_init,
                                              geo_bias_rbf_learn_weights=self.geo_bias_rbf_learn_weights,
                                              geo_bias_rbf_per_head=self.geo_bias_rbf_per_head,
                                  geo_bias_crpb_dim=self.geo_bias_crpb_dim,
                                  geo_bias_crpb_input=self.geo_bias_crpb_input,
                                  geo_bias_crpb_radius=self.geo_bias_crpb_radius,
                                  geo_bias_crpb_per_head=self.geo_bias_crpb_per_head,
                                  geo_bias_crpb_zero_init=self.geo_bias_crpb_zero_init,
                                  pos_encoding_mode=self.pos_encoding_mode,
                                  point_rope_coord_norm=self.point_rope_coord_norm,
                                  point_rope_radius=self.point_rope_radius,
                                  point_rope_base=self.point_rope_base,
                                  point_rope_axes=self.point_rope_axes,
                                  point_rope_apply_to=self.point_rope_apply_to)
                            for _ in range(num_attn_layers)
                        ])
                        for _ in range(num_layers)
                    ])
            else:
                self.layers = nn.ModuleList([
                    GeoAttentionBlock(compress_dim, mode=mode, num_heads=8,
                                      geo_sigma=geo_sigma,
                                      geo_bias_mode=self.geo_bias_mode,
                                      geo_bias_rbf_scales=self.geo_bias_rbf_scales,
                                      geo_bias_rbf_alpha_init=self.geo_bias_rbf_alpha_init,
                                      geo_bias_rbf_learn_weights=self.geo_bias_rbf_learn_weights,
                                      geo_bias_rbf_per_head=self.geo_bias_rbf_per_head,
                                  geo_bias_crpb_dim=self.geo_bias_crpb_dim,
                                  geo_bias_crpb_input=self.geo_bias_crpb_input,
                                  geo_bias_crpb_radius=self.geo_bias_crpb_radius,
                                  geo_bias_crpb_per_head=self.geo_bias_crpb_per_head,
                                  geo_bias_crpb_zero_init=self.geo_bias_crpb_zero_init,
                                  pos_encoding_mode=self.pos_encoding_mode,
                                  point_rope_coord_norm=self.point_rope_coord_norm,
                                  point_rope_radius=self.point_rope_radius,
                                  point_rope_base=self.point_rope_base,
                                  point_rope_axes=self.point_rope_axes,
                                  point_rope_apply_to=self.point_rope_apply_to)
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

    def _get_per_layer_features(self, features):
        if self.num_layers <= 1:
            return features.unsqueeze(2)
        B, N, total_c = features.shape
        if total_c % self.num_layers != 0:
            raise ValueError(
                f"pooled_features last dim {total_c} is not divisible by num_layers={self.num_layers}."
            )
        C = total_c // self.num_layers
        return features.reshape(B, N, self.num_layers, C)

    def _project_level_features(self, level_features, level_idx):
        if self.level_proj_shared:
            return self.level_proj(level_features)
        return self.level_proj[level_idx](level_features)

    def _set_cross_attention_stats(self, layers, enabled):
        for layer in layers:
            attn = getattr(layer, "attn", None)
            if attn is not None:
                attn.collect_runtime_stats = bool(enabled)
                attn.last_runtime_stats = None

    def _iter_attention_modules(self):
        if self.mode == 'hierarchical':
            for layer in self.coarse_layers:
                yield getattr(layer, 'attn', None)
            for layer in self.fine_layers:
                yield getattr(layer, 'attn', None)
        elif self.mode == 'learned':
            return
        elif self.feature_hierarchy_mode == "levelwise_latent_merge":
            if self.level_cross_attn_shared:
                for layer in self.level_layers:
                    yield getattr(layer, 'attn', None)
            else:
                for layer_group in self.level_layers:
                    for layer in layer_group:
                        yield getattr(layer, 'attn', None)
        else:
            for layer in self.layers:
                yield getattr(layer, 'attn', None)

    def _update_geo_bias_runtime_stats(self):
        summaries = []
        for idx, attn in enumerate(self._iter_attention_modules() or []):
            if attn is None:
                continue
            rbf_bias = getattr(attn, 'rbf_bias', None)
            if rbf_bias is None:
                continue
            summary = rbf_bias.summary()
            if summary is not None:
                summaries.append({"layer": idx, **summary})
        if not summaries:
            self.last_geo_bias_runtime_stats = None
            return
        if len(summaries) == 1:
            summary = summaries[0]
            self.last_geo_bias_runtime_stats = {
                "final_geo_bias_rbf_alpha": summary["final_geo_bias_rbf_alpha"],
                "final_geo_bias_rbf_weights": summary["final_geo_bias_rbf_weights"],
            }
            return
        self.last_geo_bias_runtime_stats = {
            "final_geo_bias_rbf_alpha": [item["final_geo_bias_rbf_alpha"] for item in summaries],
            "final_geo_bias_rbf_weights": [item["final_geo_bias_rbf_weights"] for item in summaries],
        }

    def _build_local_geometry_info(self, dist_sq):
        return {
            "dist_sq": dist_sq,
            "attn_bias": self.local_bias_gen(dist_sq),
        }

    def _build_geometry_info(self, *, dist_sq, query_coords, key_coords, attn_bias=None):
        return {
            "dist_sq": dist_sq,
            "attn_bias": attn_bias,
            "query_coords": query_coords,
            "key_coords": key_coords,
        }

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

    def _sample_latent_coords(self, pooled_points, scene_center, K_curr):
        if pooled_points.shape[1] > K_curr:
            fps_idx = farthest_point_sampling(
                pooled_points,
                K_curr,
                start_policy=self.fps_start_policy,
                scene_center=scene_center,
            )
            return torch.gather(
                pooled_points, 1,
                fps_idx.unsqueeze(-1).expand(-1, -1, 3))
        return pooled_points[:, :K_curr, :]

    def _forward_levelwise_latent_merge(self, memory_dict):
        pooled_points = memory_dict["pooled_points"]
        pooled_features = memory_dict["pooled_features"]
        scene_center = memory_dict["scene_center"]

        latent_coords = self._sample_latent_coords(pooled_points, scene_center, self.K)
        centered_latent = latent_coords - scene_center.unsqueeze(1)
        centered_points = pooled_points - scene_center.unsqueeze(1)
        q_seed = self.pe_encoder(centered_latent) + self.level_latent_tokens.to(
            device=pooled_points.device,
            dtype=pooled_features.dtype,
        )
        k_pos = self.pe_encoder(centered_points)

        if not hasattr(self, '_pe_logged'):
            _logger.info(
                "[PE] scale_mode=%s scene_scale=%.6f input range: %.3f~%.3f, "
                "input_std=%.4f, output range: %.3f~%.3f",
                self.pe_scale_mode,
                self.pe_scene_scale,
                float(centered_latent.min()), float(centered_latent.max()),
                float(centered_latent.std(unbiased=False)),
                float(q_seed.min()), float(q_seed.max()),
            )
            self._pe_logged = True

        dist_sq = torch.cdist(latent_coords, pooled_points, p=2) ** 2
        geometry_info = self._build_geometry_info(
            dist_sq=dist_sq,
            query_coords=latent_coords,
            key_coords=pooled_points,
            attn_bias=self.local_bias_gen(dist_sq) if self.mode == 'local' else None,
        )

        collect_stats = bool(getattr(self, "collect_runtime_stats", False))
        per_layer = self._get_per_layer_features(pooled_features)
        z_levels = []
        level_norm_mean = []
        level_norm_std = []
        attn_entropy_mean = []
        effective_token_count = []

        for level_idx in range(self.num_layers):
            level_features = per_layer[:, :, level_idx, :]
            projected = self._project_level_features(level_features, level_idx)
            k = projected + k_pos
            q = q_seed
            layers = self.level_layers if self.level_cross_attn_shared else self.level_layers[level_idx]
            self._set_cross_attention_stats(layers, collect_stats)
            last_attn_stats = None
            for layer in layers:
                q = layer(q, k, projected, geometry_info=geometry_info)
                attn = getattr(layer, "attn", None)
                if attn is not None and attn.last_runtime_stats is not None:
                    last_attn_stats = attn.last_runtime_stats
            z_levels.append(q)
            if collect_stats:
                norms = q.detach().float().norm(dim=-1)
                level_norm_mean.append(float(norms.mean().item()))
                level_norm_std.append(float(norms.std(unbiased=False).item()))
                if last_attn_stats is not None:
                    attn_entropy_mean.append(float(last_attn_stats["attn_entropy_mean"]))
                    effective_token_count.append(float(last_attn_stats["effective_token_count"]))

        z_all = torch.stack(z_levels, dim=1)
        gate = torch.softmax(self.level_logits, dim=0).to(device=z_all.device, dtype=z_all.dtype)
        merged = torch.sum(z_all * gate.view(1, self.num_layers, 1, 1), dim=1)
        x = self._inject_scale(merged, memory_dict)

        with torch.no_grad():
            gate_float = gate.detach().float().cpu()
            gate_entropy = -(gate_float * (gate_float + 1e-8).log()).sum()
            self.last_levelwise_runtime_stats = {
                "per_level_latent_norm_mean": level_norm_mean,
                "per_level_latent_norm_std": level_norm_std,
                "per_level_attention_entropy_mean": attn_entropy_mean,
                "per_level_effective_memory_token_count": effective_token_count,
                "final_level_merge_weights": gate_float.tolist(),
                "final_level_gate_entropy": float(gate_entropy.item()),
                "lmc_level_merge_weights": gate_float.tolist(),
                "lmc_level_gate_entropy": float(gate_entropy.item()),
            }
        self._update_geo_bias_runtime_stats()
        return x, latent_coords

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

        if self.feature_hierarchy_mode == "levelwise_latent_merge":
            return self._forward_levelwise_latent_merge(memory_dict)

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
            self._update_geo_bias_runtime_stats()
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
            coarse_geo = self._build_geometry_info(
                dist_sq=dist_coarse,
                query_coords=coords_coarse,
                key_coords=pooled_points,
                attn_bias=None,
            )
            for layer in self.coarse_layers:
                q_coarse = layer(q_coarse, k_base, v, geometry_info=coarse_geo)
            z_coarse = self._inject_scale(q_coarse, memory_dict)

            dist_fine = torch.cdist(coords_fine, pooled_points, p=2) ** 2
            q_fine = self.pe_encoder(
                coords_fine - scene_center.unsqueeze(1))
            fine_geo = self._build_geometry_info(
                dist_sq=dist_fine,
                query_coords=coords_fine,
                key_coords=pooled_points,
                attn_bias=None,
            )
            for layer in self.fine_layers:
                q_fine = layer(q_fine, k_base, v, geometry_info=fine_geo)
            z_fine = self._inject_scale(q_fine, memory_dict)

            self._update_geo_bias_runtime_stats()
            return {"z_coarse": z_coarse, "p_coarse": coords_coarse,
                    "z_fine": z_fine, "p_fine": coords_fine}

        # --- Single Level (global / local) ---
        latent_coords = self._sample_latent_coords(pooled_points, scene_center, self.K)

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
        geometry_info = self._build_geometry_info(
            dist_sq=dist_sq,
            query_coords=latent_coords,
            key_coords=pooled_points,
            attn_bias=self.local_bias_gen(dist_sq) if self.mode == 'local' else None,
        )

        for layer in self.layers:
            q = layer(q, k_base, v, geometry_info=geometry_info)

        x = self._inject_scale(q, memory_dict)
        self._update_geo_bias_runtime_stats()
        return x, latent_coords
