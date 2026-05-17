# ace_fusion.py
# LMCFeatureFusion: Fuse compressed memory tokens with query image features.
# Ported from map-anything/mapanything/tasks/ace/fusion.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from ace_compressor import FourierPositionEncoding


class LMCFusionBlock(nn.Module):
    """
    Single fusion block:
      1. Encode memory 3D coords via Fourier PE and inject into Value.
      2. Cross-Attention  query_features -> memory tokens.
      3. Residual + FFN.
    """

    def __init__(self, feature_dim, num_heads=8, dropout=0.1,
                 query_feature_dim=None, memory_feature_dim=None):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (feature_dim // num_heads) ** -0.5

        query_dim = query_feature_dim or feature_dim
        memory_dim = memory_feature_dim or feature_dim

        self.coord_encoder = FourierPositionEncoding(
            input_dim=3, output_dim=feature_dim)
        self.pe_proj = nn.Linear(feature_dim, memory_dim)

        self.q_proj = nn.Linear(query_dim, feature_dim)
        self.k_proj = nn.Linear(memory_dim, feature_dim)
        self.v_proj = nn.Linear(memory_dim, feature_dim)
        self.out_proj = nn.Linear(feature_dim, feature_dim)

        self.attn_drop = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(feature_dim)
        self.norm2 = nn.LayerNorm(feature_dim)
        self.ffn = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 4),
            nn.GELU(),
            nn.Linear(feature_dim * 4, feature_dim),
            nn.Dropout(dropout),
        )

    @staticmethod
    def _summarize_attention(attn, query_feats, attention_out, fused_feats, max_pixels=4096):
        """Return scalar diagnostics for attention without changing model outputs."""
        with torch.no_grad():
            if max_pixels is not None and max_pixels > 0 and attn.shape[2] > max_pixels:
                step = max(1, (attn.shape[2] + max_pixels - 1) // max_pixels)
                attn_eval = attn[:, :, ::step, :][:, :, :max_pixels, :]
            else:
                attn_eval = attn
            attn_eval = attn_eval.float()
            eps = 1e-8
            entropy = -(attn_eval * (attn_eval + eps).log()).sum(dim=-1)
            max_attn = attn_eval.max(dim=-1).values
            usage = attn_eval.mean(dim=(0, 1, 2))
            usage = usage / usage.sum().clamp(min=eps)
            usage_entropy = -(usage * (usage + eps).log()).sum()
            topk = min(5, usage.numel())
            usage_top = torch.topk(usage, k=topk).values
            return {
                "attn_entropy_mean": float(entropy.mean().item()),
                "attn_entropy_p10": float(torch.quantile(entropy.flatten(), 0.10).item()),
                "attn_entropy_p50": float(torch.quantile(entropy.flatten(), 0.50).item()),
                "attn_entropy_p90": float(torch.quantile(entropy.flatten(), 0.90).item()),
                "avg_max_attention": float(max_attn.mean().item()),
                "effective_token_count": float(torch.exp(usage_entropy).item()),
                "token_usage_min": float(usage.min().item()),
                "token_usage_max": float(usage.max().item()),
                "token_usage_top5": [float(v.item()) for v in usage_top],
                "raw_feature_norm": float(query_feats.float().norm(dim=-1).mean().item()),
                "attention_out_norm": float(attention_out.float().norm(dim=-1).mean().item()),
                "fused_feature_norm": float(fused_feats.float().norm(dim=-1).mean().item()),
                "num_tokens": int(attn.shape[-1]),
                "num_queries_used": int(attn_eval.shape[2]),
            }

    def forward(self, query_feats, memory_z, memory_p, scene_center,
                return_stats=False, stats_max_pixels=4096):
        """
        Args:
            query_feats:  (B, N_q, C)  image features
            memory_z:     (B, K, C)    latent features from compressor
            memory_p:     (B, K, 3)    latent 3D coordinates
            scene_center: (B, 3)
        Returns:
            fused features (B, N_q, C)
        """
        B, N_q, C = query_feats.shape
        K = memory_z.shape[1]

        residual = query_feats

        q = self.q_proj(query_feats)
        k = self.k_proj(memory_z)

        norm_p = memory_p - scene_center.unsqueeze(1)
        pe = self.coord_encoder(norm_p)
        v_input = memory_z + self.pe_proj(pe)
        v = self.v_proj(v_input)

        head_dim = C // self.num_heads
        q = q.reshape(B, N_q, self.num_heads, head_dim).transpose(1, 2)
        k = k.reshape(B, K, self.num_heads, head_dim).transpose(1, 2)
        v = v.reshape(B, K, self.num_heads, head_dim).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn_soft = attn.softmax(dim=-1)
        attn = self.attn_drop(attn_soft)

        out = (attn @ v).transpose(1, 2).reshape(B, N_q, C)
        out = self.out_proj(out)

        x = self.norm1(residual + out)
        x = self.norm2(x + self.ffn(x))
        if return_stats:
            stats = self._summarize_attention(
                attn_soft,
                query_feats,
                out,
                x,
                max_pixels=stats_max_pixels,
            )
            return x, stats
        return x


class LMCFeatureFusion(nn.Module):
    """
    Main Fusion Module.
    Adapts to 'local', 'global', 'hierarchical', or 'learned' modes.
    """

    def __init__(self, feature_dim=1024, mode='global', num_heads=8,
                 dropout=0.1, query_feature_dim=None,
                 memory_feature_dim=None):
        super().__init__()
        self.mode = mode
        self.feature_dim = feature_dim

        if mode == 'hierarchical':
            self.fusion_coarse = LMCFusionBlock(
                feature_dim, num_heads, dropout,
                query_feature_dim, memory_feature_dim)
            self.fusion_fine = LMCFusionBlock(
                feature_dim, num_heads, dropout,
                query_feature_dim, memory_feature_dim)
        else:
            self.fusion_single = LMCFusionBlock(
                feature_dim, num_heads, dropout,
                query_feature_dim, memory_feature_dim)

    def forward(self, query_feats, compressor_out, scene_center,
                return_stats=False, stats_max_pixels=4096):
        """
        Args:
            query_feats:    (B, N_q, C)
            compressor_out: tuple (z, p) or dict (hierarchical)
            scene_center:   (B, 3)
        """
        if self.mode == 'hierarchical':
            if not isinstance(compressor_out, dict):
                raise ValueError("hierarchical mode expects dict output.")
            coarse_out = self.fusion_coarse(
                query_feats,
                compressor_out['z_coarse'],
                compressor_out['p_coarse'],
                scene_center,
                return_stats=return_stats,
                stats_max_pixels=stats_max_pixels)
            if return_stats:
                feats_mid, coarse_stats = coarse_out
            else:
                feats_mid = coarse_out
            fine_out = self.fusion_fine(
                feats_mid,
                compressor_out['z_fine'],
                compressor_out['p_fine'],
                scene_center,
                return_stats=return_stats,
                stats_max_pixels=stats_max_pixels)
            if return_stats:
                feats_final, fine_stats = fine_out
                stats = {f"coarse_{k}": v for k, v in coarse_stats.items()}
                stats.update({f"fine_{k}": v for k, v in fine_stats.items()})
                return feats_final, stats
            return fine_out
        else:
            latent_z, latent_p = compressor_out
            return self.fusion_single(
                query_feats,
                latent_z,
                latent_p,
                scene_center,
                return_stats=return_stats,
                stats_max_pixels=stats_max_pixels)
