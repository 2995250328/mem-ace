# ace_fusion.py
# LMCFeatureFusion: Fuse compressed memory tokens with query image features.
# Ported from map-anything/mapanything/tasks/ace/fusion.py

import math

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

    VALID_GEOMETRY_MODES = {"value_only_raw", "value_only_norm", "geokey_norm"}

    def __init__(self, feature_dim, num_heads=8, dropout=0.1,
                 query_feature_dim=None, memory_feature_dim=None,
                 fusion_geometry_mode="value_only_raw",
                 fusion_scene_scale=1.0,
                 fusion_key_geo_init=0.0,
                 single_qk_norm=False,
                 single_qk_norm_eps=1e-6,
                 single_qk_norm_tau_init=0.0,
                 single_out_layerscale=False,
                 single_out_layerscale_init=1.0):
        super().__init__()
        if feature_dim % num_heads != 0:
            raise ValueError("feature_dim must be divisible by num_heads.")
        if float(single_qk_norm_eps) <= 0.0:
            raise ValueError("single_qk_norm_eps must be > 0.")
        if float(single_qk_norm_tau_init) < 0.0:
            raise ValueError("single_qk_norm_tau_init must be >= 0. Use 0 for sqrt(head_dim).")
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.scale = (feature_dim // num_heads) ** -0.5
        self.single_qk_norm = bool(single_qk_norm)
        self.single_qk_norm_eps = float(single_qk_norm_eps)
        self.single_out_layerscale = bool(single_out_layerscale)
        head_dim = feature_dim // num_heads
        tau_init = (
            float(single_qk_norm_tau_init)
            if float(single_qk_norm_tau_init) > 0.0
            else math.sqrt(float(head_dim))
        )
        if self.single_qk_norm:
            self.single_qk_norm_log_tau = nn.Parameter(
                torch.full((num_heads,), math.log(tau_init), dtype=torch.float32)
            )
        else:
            self.register_buffer(
                "single_qk_norm_log_tau", torch.empty(0, dtype=torch.float32), persistent=False
            )
        if self.single_out_layerscale:
            self.single_out_gamma = nn.Parameter(
                torch.full((feature_dim,), float(single_out_layerscale_init), dtype=torch.float32)
            )
        else:
            self.register_buffer(
                "single_out_gamma", torch.empty(0, dtype=torch.float32), persistent=False
            )
        if fusion_geometry_mode not in self.VALID_GEOMETRY_MODES:
            raise ValueError(
                f"Unsupported fusion_geometry_mode={fusion_geometry_mode!r}. "
                f"Expected one of {sorted(self.VALID_GEOMETRY_MODES)}."
            )
        self.fusion_geometry_mode = fusion_geometry_mode
        fusion_scene_scale = float(fusion_scene_scale)
        if (not torch.isfinite(torch.tensor(fusion_scene_scale)).item()) or fusion_scene_scale <= 0.0:
            raise ValueError(f"fusion_scene_scale must be finite and > 0, got {fusion_scene_scale!r}")
        self.register_buffer(
            "fusion_scene_scale",
            torch.tensor(fusion_scene_scale, dtype=torch.float32),
            persistent=False,
        )

        query_dim = query_feature_dim or feature_dim
        memory_dim = memory_feature_dim or feature_dim

        self.coord_encoder = FourierPositionEncoding(
            input_dim=3, output_dim=feature_dim)
        self.pe_proj = nn.Linear(feature_dim, memory_dim)
        if fusion_geometry_mode == "geokey_norm":
            self.key_geo_scale = nn.Parameter(
                torch.tensor(float(fusion_key_geo_init), dtype=torch.float32)
            )

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
    def _summarize_attention(attn, query_feats, attention_out, fused_feats,
                             max_pixels=4096, extra_stats=None):
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
            stats = {
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
            if extra_stats:
                stats.update(extra_stats)
            return stats

    def encode_memory(self, memory_z, memory_p, scene_center):
        """Encode memory features and geometry into shared attention keys/values."""
        B, K, _ = memory_z.shape

        centered_p = memory_p - scene_center.unsqueeze(1)
        if self.fusion_geometry_mode == "value_only_raw":
            pe_input = centered_p
        else:
            scale = self.fusion_scene_scale.to(device=centered_p.device, dtype=centered_p.dtype)
            pe_input = centered_p / scale.clamp(min=1e-6)
        pe = self.coord_encoder(pe_input)
        pe_mem = self.pe_proj(pe)

        k_input = memory_z
        if self.fusion_geometry_mode == "geokey_norm":
            k_input = memory_z + self.key_geo_scale.to(dtype=memory_z.dtype) * pe_mem
        k = self.k_proj(k_input)
        v = self.v_proj(memory_z + pe_mem)

        head_dim = self.feature_dim // self.num_heads
        k = k.reshape(B, K, self.num_heads, head_dim).transpose(1, 2)
        v = v.reshape(B, K, self.num_heads, head_dim).transpose(1, 2)
        return k, v, pe_input

    def read_memory(self, query_feats, k, v):
        """Read pre-encoded memory keys/values with the supplied query features."""
        B, N_q, C = query_feats.shape
        head_dim = C // self.num_heads

        q = self.q_proj(query_feats)
        q = q.reshape(B, N_q, self.num_heads, head_dim).transpose(1, 2)

        if self.single_qk_norm:
            q_attn = F.normalize(q, p=2, dim=-1, eps=self.single_qk_norm_eps)
            k_attn = F.normalize(k, p=2, dim=-1, eps=self.single_qk_norm_eps)
            tau = self.single_qk_norm_log_tau.exp().to(
                device=q.device, dtype=q.dtype
            ).view(1, self.num_heads, 1, 1)
            attn = (q_attn @ k_attn.transpose(-2, -1)) * tau
        else:
            attn = (q @ k.transpose(-2, -1)) * self.scale
        attn_soft = attn.softmax(dim=-1)
        attn = self.attn_drop(attn_soft)

        out = (attn @ v).transpose(1, 2).reshape(B, N_q, C)
        out = self.out_proj(out)
        if self.single_out_layerscale:
            gamma = self.single_out_gamma.to(device=out.device, dtype=out.dtype).view(1, 1, -1)
            out = out * gamma
        return out, attn_soft

    def fuse_encoded_memory(self, query_feats, k, v):
        """Fuse query features with pre-encoded memory keys/values."""
        out, attn_soft = self.read_memory(query_feats, k, v)
        x = self.norm1(query_feats + out)
        x = self.norm2(x + self.ffn(x))
        return x, out, attn_soft

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
        k, v, pe_input = self.encode_memory(memory_z, memory_p, scene_center)
        x, out, attn_soft = self.fuse_encoded_memory(query_feats, k, v)
        if return_stats:
            with torch.no_grad():
                pe_eval = pe_input.detach().float()
                extra_stats = {
                    "fusion_geometry_mode": self.fusion_geometry_mode,
                    "fusion_scene_scale": float(self.fusion_scene_scale.detach().cpu().item()),
                    "key_geo_scale": float(
                        getattr(self, "key_geo_scale", torch.tensor(0.0, device=pe_input.device))
                        .detach()
                        .float()
                        .cpu()
                        .item()
                    ),
                    "memory_p_norm_std": float(pe_eval.std(unbiased=False).item()),
                    "memory_p_norm_absmax": float(pe_eval.abs().max().item()),
                    "memory_p_norm_finite": bool(torch.isfinite(pe_eval).all().item()),
                    "single_qk_norm": bool(self.single_qk_norm),
                    "single_qknorm_tau_mean": float(
                        self.single_qk_norm_log_tau.detach().float().exp().mean().item()
                        if self.single_qk_norm and self.single_qk_norm_log_tau.numel() > 0 else 0.0
                    ),
                    "single_qknorm_tau_min": float(
                        self.single_qk_norm_log_tau.detach().float().exp().min().item()
                        if self.single_qk_norm and self.single_qk_norm_log_tau.numel() > 0 else 0.0
                    ),
                    "single_qknorm_tau_max": float(
                        self.single_qk_norm_log_tau.detach().float().exp().max().item()
                        if self.single_qk_norm and self.single_qk_norm_log_tau.numel() > 0 else 0.0
                    ),
                    "single_out_layerscale": bool(self.single_out_layerscale),
                    "single_out_gamma_mean": float(
                        self.single_out_gamma.detach().float().mean().item()
                        if self.single_out_layerscale and self.single_out_gamma.numel() > 0 else 1.0
                    ),
                    "single_out_gamma_absmax": float(
                        self.single_out_gamma.detach().float().abs().max().item()
                        if self.single_out_layerscale and self.single_out_gamma.numel() > 0 else 1.0
                    ),
                }
            stats = self._summarize_attention(
                attn_soft,
                query_feats,
                out,
                x,
                max_pixels=stats_max_pixels,
                extra_stats=extra_stats,
            )
            return x, stats
        return x


class LMCProgressiveRereadBlock(nn.Module):
    """A lightweight second query pass over already encoded memory tokens."""

    def __init__(self, feature_dim, num_heads=8, dropout=0.1, attention_temperature=1.0,
                 qk_norm=False, qk_norm_eps=1e-6, qk_norm_tau_init=0.0):
        super().__init__()
        if feature_dim % num_heads != 0:
            raise ValueError("feature_dim must be divisible by num_heads.")
        if float(attention_temperature) <= 0.0:
            raise ValueError("attention_temperature must be > 0.")
        if float(qk_norm_eps) <= 0.0:
            raise ValueError("qk_norm_eps must be > 0.")
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.attention_temperature = float(attention_temperature)
        self.qk_norm = bool(qk_norm)
        self.qk_norm_eps = float(qk_norm_eps)
        self.scale = (feature_dim // num_heads) ** -0.5
        head_dim = feature_dim // num_heads
        tau_init = float(qk_norm_tau_init) if float(qk_norm_tau_init) > 0.0 else math.sqrt(float(head_dim))
        if self.qk_norm:
            self.qk_norm_log_tau = nn.Parameter(
                torch.full((num_heads,), math.log(tau_init), dtype=torch.float32)
            )
        else:
            self.register_buffer(
                "qk_norm_log_tau", torch.empty(0, dtype=torch.float32), persistent=False
            )
        self.query_norm = nn.LayerNorm(feature_dim)
        self.q_proj = nn.Linear(feature_dim, feature_dim)
        self.out_proj = nn.Linear(feature_dim, feature_dim)
        self.attn_drop = nn.Dropout(dropout)
        self.output_norm = nn.LayerNorm(feature_dim)

    def forward(self, query_feats, k, v, attention_bias=None, reference_attn=None):
        B, N_q, C = query_feats.shape
        head_dim = C // self.num_heads
        q = self.q_proj(self.query_norm(query_feats))
        q = q.reshape(B, N_q, self.num_heads, head_dim).transpose(1, 2)

        if self.qk_norm:
            q_attn = F.normalize(q, p=2, dim=-1, eps=self.qk_norm_eps)
            k_attn = F.normalize(k, p=2, dim=-1, eps=self.qk_norm_eps)
            tau = self.qk_norm_log_tau.exp().to(device=q.device, dtype=q.dtype).view(1, self.num_heads, 1, 1)
            attn_logits = (q_attn @ k_attn.transpose(-2, -1)) * tau
        else:
            attn_logits = (q @ k.transpose(-2, -1)) * self.scale
        if attention_bias is not None:
            attn_logits = attn_logits + attention_bias.to(
                device=attn_logits.device,
                dtype=attn_logits.dtype,
            )
        if self.attention_temperature != 1.0:
            attn_logits = attn_logits / self.attention_temperature
        attn_soft = attn_logits.softmax(dim=-1)
        if reference_attn is None:
            attn = self.attn_drop(attn_soft)
            context = (attn @ v).transpose(1, 2).reshape(B, N_q, C)
            delta = self.out_proj(context)
        else:
            shared_drop_mask = self.attn_drop(torch.ones_like(attn_soft))
            reread_context = (
                (attn_soft * shared_drop_mask) @ v
            ).transpose(1, 2).reshape(B, N_q, C)
            reference_context = (
                (reference_attn.detach() * shared_drop_mask) @ v
            ).transpose(1, 2).reshape(B, N_q, C)
            delta = self.out_proj(reread_context) - self.out_proj(reference_context)
        fused = self.output_norm(query_feats + delta)
        return fused, delta, attn_soft


class LMCAdapterFFNBlock(nn.Module):
    """Parameter-matched FFN control for progressive reread experiments."""

    def __init__(self, feature_dim, dropout=0.1):
        super().__init__()
        self.input_norm = nn.LayerNorm(feature_dim)
        self.adapter = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim, feature_dim),
        )
        self.output_norm = nn.LayerNorm(feature_dim)

    def compute_delta(self, query_feats):
        return self.adapter(self.input_norm(query_feats))

    def forward(self, query_feats):
        delta = self.compute_delta(query_feats)
        fused = self.output_norm(query_feats + delta)
        return fused, delta


class LMCFeatureFusion(nn.Module):
    """
    Main Fusion Module.
    Adapts to 'local', 'global', 'hierarchical', or 'learned' modes.
    """

    VALID_REFINEMENT_MODES = {
        "single",
        "single_qknorm_layerscale",
        "cascade_internal",
        "progressive_reread",
        "centered_reread",
        "centered_reread_qknorm_layerscale",
        "ccf_centered_reread_qknorm_layerscale",
        "geometry_reread_lite",
        "adapter_ffn",
        "weak_residual_ffn",
        "v3_adapter_control",
        "v3_dual_refine",
        "coord_prior_v1",
    }
    VALID_ASSEMBLY_MODES = {"concat_mlp"}

    def __init__(self, feature_dim=1024, mode='global', num_heads=8,
                 dropout=0.1, query_feature_dim=None,
                 memory_feature_dim=None, fusion_geometry_mode="value_only_raw",
                 fusion_scene_scale=1.0, fusion_key_geo_init=0.0,
                 fusion_refinement_mode="single", fusion_cascade_layers=4,
                 fusion_assembly_mode="concat_mlp", fusion_assembly_gamma_init=0.0,
                 fusion_reread_delta_alpha=1.0,
                 fusion_reread_scalar_gate=False,
                 fusion_reread_gate_init=0.0,
                 fusion_reread_post_norm=True,
                 fusion_reread_geo_lambda=1.0,
                 fusion_reread_geo_sigma=1.0,
                 fusion_reread_geo_sigma_mode="fixed",
                 fusion_reread_geo_sigma_beta=1.0,
                 fusion_reread_geo_sigma_min=0.5,
                 fusion_reread_trust_region_ratio=0.0,
                 fusion_reread_temperature=1.0,
                 fusion_reread_common_scale=1.0,
                 fusion_reread_effective_ratio_cap=0.0,
                 fusion_reread_warmup_mode="none",
                 fusion_reread_warmup_iters=0,
                 fusion_reread_warmup_start=0.0,
                 fusion_reread_qknorm_eps=1e-6,
                 fusion_reread_qknorm_tau_init=0.0,
                 fusion_reread_layerscale_patch_init=0.01,
                 fusion_reread_layerscale_common_init=0.0,
                 fusion_dual_memory_layerscale_patch_init=0.005,
                 fusion_dual_memory_layerscale_common_init=0.0,
                 fusion_ccf_gate_source="first_attn_entropy",
                 fusion_ccf_gate_floor=0.0,
                 fusion_ccf_gate_gamma=1.0,
                 fusion_ccf_detach_gate=True,
                 fusion_single_qknorm_eps=1e-6,
                 fusion_single_qknorm_tau_init=0.0,
                 fusion_single_layerscale_init=1.0,
                 fusion_coord_prior_scale_init=0.10):
        super().__init__()
        self.mode = mode
        self.feature_dim = feature_dim
        self.fusion_geometry_mode = fusion_geometry_mode
        self.fusion_refinement_mode = str(fusion_refinement_mode)
        self.fusion_cascade_layers = int(fusion_cascade_layers)
        self.fusion_assembly_mode = str(fusion_assembly_mode)
        self.fusion_reread_delta_alpha = float(fusion_reread_delta_alpha)
        self.fusion_reread_scalar_gate = bool(fusion_reread_scalar_gate)
        self.fusion_reread_post_norm = bool(fusion_reread_post_norm)
        self.fusion_reread_geo_lambda = float(fusion_reread_geo_lambda)
        self.fusion_reread_geo_sigma = float(fusion_reread_geo_sigma)
        self.fusion_reread_geo_sigma_mode = str(fusion_reread_geo_sigma_mode)
        self.fusion_reread_geo_sigma_beta = float(fusion_reread_geo_sigma_beta)
        self.fusion_reread_geo_sigma_min = float(fusion_reread_geo_sigma_min)
        self.fusion_reread_trust_region_ratio = float(fusion_reread_trust_region_ratio)
        self.fusion_reread_temperature = float(fusion_reread_temperature)
        self.fusion_reread_common_scale = float(fusion_reread_common_scale)
        self.fusion_reread_effective_ratio_cap = float(fusion_reread_effective_ratio_cap)
        self.fusion_reread_warmup_mode = str(fusion_reread_warmup_mode)
        self.fusion_reread_warmup_iters = int(fusion_reread_warmup_iters)
        self.fusion_reread_warmup_start = float(fusion_reread_warmup_start)
        self.fusion_reread_qknorm_eps = float(fusion_reread_qknorm_eps)
        self.fusion_reread_qknorm_tau_init = float(fusion_reread_qknorm_tau_init)
        self.fusion_reread_layerscale_patch_init = float(fusion_reread_layerscale_patch_init)
        self.fusion_reread_layerscale_common_init = float(fusion_reread_layerscale_common_init)
        self.fusion_dual_memory_layerscale_patch_init = float(fusion_dual_memory_layerscale_patch_init)
        self.fusion_dual_memory_layerscale_common_init = float(fusion_dual_memory_layerscale_common_init)
        self.fusion_ccf_gate_source = str(fusion_ccf_gate_source)
        self.fusion_ccf_gate_floor = float(fusion_ccf_gate_floor)
        self.fusion_ccf_gate_gamma = float(fusion_ccf_gate_gamma)
        self.fusion_ccf_detach_gate = bool(fusion_ccf_detach_gate)
        self.fusion_single_qknorm_eps = float(fusion_single_qknorm_eps)
        self.fusion_single_qknorm_tau_init = float(fusion_single_qknorm_tau_init)
        self.fusion_single_layerscale_init = float(fusion_single_layerscale_init)
        self.fusion_coord_prior_scale_init = float(fusion_coord_prior_scale_init)
        self._fusion_reread_warmup_scale = 1.0

        if self.fusion_refinement_mode not in self.VALID_REFINEMENT_MODES:
            raise ValueError(
                f"Unsupported fusion_refinement_mode={fusion_refinement_mode!r}. "
                f"Expected one of {sorted(self.VALID_REFINEMENT_MODES)}."
            )
        if self.fusion_assembly_mode not in self.VALID_ASSEMBLY_MODES:
            raise ValueError(
                f"Unsupported fusion_assembly_mode={fusion_assembly_mode!r}. "
                f"Expected one of {sorted(self.VALID_ASSEMBLY_MODES)}."
            )
        if self.fusion_cascade_layers < 1:
            raise ValueError("fusion_cascade_layers must be >= 1.")
        if self.fusion_reread_delta_alpha < 0.0:
            raise ValueError("fusion_reread_delta_alpha must be >= 0.")
        if self.fusion_reread_geo_lambda < 0.0:
            raise ValueError("fusion_reread_geo_lambda must be >= 0.")
        if self.fusion_reread_geo_sigma <= 0.0:
            raise ValueError("fusion_reread_geo_sigma must be > 0.")
        if self.fusion_reread_geo_sigma_mode not in {"fixed", "adaptive_spread"}:
            raise ValueError("fusion_reread_geo_sigma_mode must be 'fixed' or 'adaptive_spread'.")
        if self.fusion_reread_geo_sigma_beta <= 0.0:
            raise ValueError("fusion_reread_geo_sigma_beta must be > 0.")
        if self.fusion_reread_geo_sigma_min <= 0.0:
            raise ValueError("fusion_reread_geo_sigma_min must be > 0.")
        if self.fusion_reread_trust_region_ratio < 0.0:
            raise ValueError("fusion_reread_trust_region_ratio must be >= 0.")
        if self.fusion_reread_temperature <= 0.0:
            raise ValueError("fusion_reread_temperature must be > 0.")
        if self.fusion_reread_common_scale < 0.0:
            raise ValueError("fusion_reread_common_scale must be >= 0.")
        if self.fusion_reread_effective_ratio_cap < 0.0:
            raise ValueError("fusion_reread_effective_ratio_cap must be >= 0.")
        if self.fusion_reread_warmup_mode not in {"none", "linear", "cosine"}:
            raise ValueError("fusion_reread_warmup_mode must be one of: none, linear, cosine.")
        if self.fusion_reread_warmup_iters < 0:
            raise ValueError("fusion_reread_warmup_iters must be >= 0.")
        if not 0.0 <= self.fusion_reread_warmup_start <= 1.0:
            raise ValueError("fusion_reread_warmup_start must be in [0, 1].")
        if self.fusion_reread_qknorm_eps <= 0.0:
            raise ValueError("fusion_reread_qknorm_eps must be > 0.")
        if self.fusion_reread_qknorm_tau_init < 0.0:
            raise ValueError("fusion_reread_qknorm_tau_init must be >= 0. Use 0 for sqrt(head_dim).")
        if self.fusion_ccf_gate_source not in {"first_attn_entropy"}:
            raise ValueError("fusion_ccf_gate_source must be 'first_attn_entropy'.")
        if not 0.0 <= self.fusion_ccf_gate_floor <= 1.0:
            raise ValueError("fusion_ccf_gate_floor must be in [0, 1].")
        if self.fusion_ccf_gate_gamma < 0.0:
            raise ValueError("fusion_ccf_gate_gamma must be >= 0.")
        if self.fusion_single_qknorm_eps <= 0.0:
            raise ValueError("fusion_single_qknorm_eps must be > 0.")
        if self.fusion_single_qknorm_tau_init < 0.0:
            raise ValueError("fusion_single_qknorm_tau_init must be >= 0. Use 0 for sqrt(head_dim).")
        if self.fusion_coord_prior_scale_init < 0.0:
            raise ValueError("fusion_coord_prior_scale_init must be >= 0.")
        if mode == 'hierarchical' and self.fusion_refinement_mode != "single":
            raise ValueError("Fusion refinement is only supported for non-hierarchical LMC modes.")

        block_kwargs = dict(
            feature_dim=feature_dim,
            num_heads=num_heads,
            dropout=dropout,
            query_feature_dim=query_feature_dim,
            memory_feature_dim=memory_feature_dim,
            fusion_geometry_mode=fusion_geometry_mode,
            fusion_scene_scale=fusion_scene_scale,
            fusion_key_geo_init=fusion_key_geo_init,
            single_qk_norm=self.fusion_refinement_mode == "single_qknorm_layerscale",
            single_qk_norm_eps=self.fusion_single_qknorm_eps,
            single_qk_norm_tau_init=self.fusion_single_qknorm_tau_init,
            single_out_layerscale=self.fusion_refinement_mode == "single_qknorm_layerscale",
            single_out_layerscale_init=self.fusion_single_layerscale_init,
        )

        if mode == 'hierarchical':
            self.fusion_coarse = LMCFusionBlock(**block_kwargs)
            self.fusion_fine = LMCFusionBlock(**block_kwargs)
        else:
            self.fusion_single = LMCFusionBlock(**block_kwargs)
            self.progressive_reread = (
                LMCProgressiveRereadBlock(
                    feature_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    attention_temperature=self.fusion_reread_temperature,
                    qk_norm=self.fusion_refinement_mode in (
                        "centered_reread_qknorm_layerscale",
                        "ccf_centered_reread_qknorm_layerscale",
                        "v3_dual_refine"
                    ),
                    qk_norm_eps=self.fusion_reread_qknorm_eps,
                    qk_norm_tau_init=self.fusion_reread_qknorm_tau_init,
                )
                if self.fusion_refinement_mode in (
                    "progressive_reread", "centered_reread",
                    "centered_reread_qknorm_layerscale",
                    "ccf_centered_reread_qknorm_layerscale",
                    "geometry_reread_lite",
                    "v3_dual_refine"
                )
                else None
            )
            if self.fusion_refinement_mode in (
                "progressive_reread", "centered_reread",
                "centered_reread_qknorm_layerscale",
                "ccf_centered_reread_qknorm_layerscale",
                "geometry_reread_lite",
                "weak_residual_ffn",
            ) and self.fusion_reread_scalar_gate:
                self.fusion_reread_gate_logit = nn.Parameter(
                    torch.tensor(float(fusion_reread_gate_init), dtype=torch.float32)
                )
            else:
                self.register_buffer(
                    "fusion_reread_gate_logit",
                    torch.tensor(float(fusion_reread_gate_init), dtype=torch.float32),
                    persistent=False,
                )
            self.adapter_ffn = (
                LMCAdapterFFNBlock(feature_dim, dropout=dropout)
                if self.fusion_refinement_mode in (
                    "adapter_ffn", "weak_residual_ffn", "v3_adapter_control",
                    "v3_dual_refine"
                )
                else None
            )
            if self.fusion_refinement_mode == "coord_prior_v1":
                self.coord_prior_encoder = FourierPositionEncoding(
                    input_dim=3, output_dim=feature_dim
                )
                self.coord_prior_proj = nn.Sequential(
                    nn.LayerNorm(feature_dim),
                    nn.Linear(feature_dim, feature_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(feature_dim, feature_dim),
                )
                self.coord_prior_gamma = nn.Parameter(
                    torch.tensor(self.fusion_coord_prior_scale_init, dtype=torch.float32)
                )
            else:
                self.coord_prior_encoder = None
                self.coord_prior_proj = None
                self.register_buffer(
                    "coord_prior_gamma",
                    torch.tensor(self.fusion_coord_prior_scale_init, dtype=torch.float32),
                    persistent=False,
                )
            if self.fusion_refinement_mode in (
                "centered_reread_qknorm_layerscale",
                "ccf_centered_reread_qknorm_layerscale",
                "v3_adapter_control", "v3_dual_refine"
            ):
                self.fusion_reread_gamma_patch = nn.Parameter(
                    torch.full((feature_dim,), self.fusion_reread_layerscale_patch_init, dtype=torch.float32)
                )
                self.fusion_reread_gamma_common = nn.Parameter(
                    torch.full((feature_dim,), self.fusion_reread_layerscale_common_init, dtype=torch.float32)
                )
            else:
                self.register_buffer(
                    "fusion_reread_gamma_patch",
                    torch.full((feature_dim,), self.fusion_reread_layerscale_patch_init, dtype=torch.float32),
                    persistent=False,
                )
                self.register_buffer(
                    "fusion_reread_gamma_common",
                    torch.full((feature_dim,), self.fusion_reread_layerscale_common_init, dtype=torch.float32),
                    persistent=False,
                )
            if self.fusion_refinement_mode == "v3_dual_refine":
                self.fusion_dual_memory_gamma_patch = nn.Parameter(
                    torch.full((feature_dim,), self.fusion_dual_memory_layerscale_patch_init, dtype=torch.float32)
                )
                self.fusion_dual_memory_gamma_common = nn.Parameter(
                    torch.full((feature_dim,), self.fusion_dual_memory_layerscale_common_init, dtype=torch.float32)
                )
            else:
                self.register_buffer(
                    "fusion_dual_memory_gamma_patch",
                    torch.full((feature_dim,), self.fusion_dual_memory_layerscale_patch_init, dtype=torch.float32),
                    persistent=False,
                )
                self.register_buffer(
                    "fusion_dual_memory_gamma_common",
                    torch.full((feature_dim,), self.fusion_dual_memory_layerscale_common_init, dtype=torch.float32),
                    persistent=False,
                )
            if self.fusion_refinement_mode == "cascade_internal" and self.fusion_cascade_layers > 1:
                cascade_kwargs = dict(block_kwargs)
                cascade_kwargs["query_feature_dim"] = feature_dim
                self.fusion_cascade = nn.ModuleList([
                    LMCFusionBlock(**cascade_kwargs)
                    for _ in range(self.fusion_cascade_layers - 1)
                ])
                assembly_in_dim = feature_dim * (self.fusion_cascade_layers - 1)
                self.fusion_assembly = nn.Sequential(
                    nn.LayerNorm(assembly_in_dim),
                    nn.Linear(assembly_in_dim, feature_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(feature_dim, feature_dim),
                )
                self.fusion_assembly_gamma = nn.Parameter(
                    torch.tensor(float(fusion_assembly_gamma_init), dtype=torch.float32)
                )
            else:
                self.fusion_cascade = nn.ModuleList()
                self.fusion_assembly = None
                self.register_buffer(
                    "fusion_assembly_gamma",
                    torch.tensor(float(fusion_assembly_gamma_init), dtype=torch.float32),
                    persistent=False,
                )

    def set_reread_warmup_progress(self, iteration_idx=None):
        """Set training-time reread residual warmup scale.

        The default scale is 1.0 so evaluation uses the fully activated model.
        Training code can call this at the start of each outer LMC iteration.
        """
        if self.fusion_reread_warmup_mode == "none" or self.fusion_reread_warmup_iters <= 0:
            self._fusion_reread_warmup_scale = 1.0
            return self._fusion_reread_warmup_scale
        if iteration_idx is None:
            self._fusion_reread_warmup_scale = 1.0
            return self._fusion_reread_warmup_scale

        progress = float(iteration_idx + 1) / float(max(1, self.fusion_reread_warmup_iters))
        progress = max(0.0, min(1.0, progress))
        if self.fusion_reread_warmup_mode == "cosine":
            progress = 0.5 - 0.5 * math.cos(math.pi * progress)
        scale = self.fusion_reread_warmup_start + (
            1.0 - self.fusion_reread_warmup_start
        ) * progress
        self._fusion_reread_warmup_scale = float(max(0.0, min(1.0, scale)))
        return self._fusion_reread_warmup_scale

    def get_reread_warmup_scale(self):
        return float(getattr(self, "_fusion_reread_warmup_scale", 1.0))

    @staticmethod
    def _summarize_routing_change(first_attn, reread_attn, max_pixels=4096):
        """Summarize how the second read reroutes attention relative to the first read."""
        with torch.no_grad():
            if max_pixels is not None and max_pixels > 0 and first_attn.shape[2] > max_pixels:
                step = max(1, (first_attn.shape[2] + max_pixels - 1) // max_pixels)
                first_eval = first_attn[:, :, ::step, :][:, :, :max_pixels, :]
                reread_eval = reread_attn[:, :, ::step, :][:, :, :max_pixels, :]
            else:
                first_eval = first_attn
                reread_eval = reread_attn
            first_eval = first_eval.detach().float()
            reread_eval = reread_eval.detach().float()
            eps = 1e-8
            mix = 0.5 * (first_eval + reread_eval)
            js = 0.5 * (
                (first_eval * ((first_eval + eps).log() - (mix + eps).log())).sum(dim=-1)
                + (reread_eval * ((reread_eval + eps).log() - (mix + eps).log())).sum(dim=-1)
            )
            first_top1 = first_eval.argmax(dim=-1)
            reread_top1 = reread_eval.argmax(dim=-1)
            top1_agree = (first_top1 == reread_top1).float().mean()
            topk = min(5, first_eval.shape[-1])
            first_topk = torch.topk(first_eval, k=topk, dim=-1).indices
            reread_topk = torch.topk(reread_eval, k=topk, dim=-1).indices
            overlap = (first_topk.unsqueeze(-1) == reread_topk.unsqueeze(-2)).any(dim=-1).float().sum(dim=-1)
            overlap = overlap / float(topk)
            return {
                "a1_a2_js_mean": float(js.mean().item()),
                "a1_a2_js_p50": float(torch.quantile(js.flatten(), 0.50).item()),
                "a1_a2_js_p90": float(torch.quantile(js.flatten(), 0.90).item()),
                "a1_a2_top1_agreement": float(top1_agree.item()),
                "a1_a2_top5_overlap": float(overlap.mean().item()),
            }

    @staticmethod
    def _build_reread_geometry_bias(first_attn, memory_points_norm, geo_lambda, geo_sigma,
                                    sigma_mode="fixed", sigma_beta=1.0, sigma_min=0.5):
        """Build a soft 3D neighborhood bias from first-read routing."""
        with torch.no_grad():
            a1 = first_attn.detach().mean(dim=1).float()
            points = memory_points_norm.detach().float()
            p1 = a1 @ points
            dist2 = ((points.unsqueeze(1) - p1.unsqueeze(2)) ** 2).sum(dim=-1)
            if str(sigma_mode) == "adaptive_spread":
                spread1 = (a1 * dist2).sum(dim=-1)
                sigma = torch.sqrt(spread1.clamp(min=1e-12)) * float(sigma_beta)
                sigma = sigma.clamp(min=float(sigma_min)).unsqueeze(-1)
            else:
                sigma = max(float(geo_sigma), 1e-6)
            bias = -float(geo_lambda) * dist2 / (2.0 * sigma * sigma)
        return bias.unsqueeze(1)

    @staticmethod
    def _summarize_geometry_reread(first_attn, reread_attn, memory_points_norm, max_pixels=4096):
        """Summarize first/second read movement in normalized memory-coordinate space."""
        with torch.no_grad():
            if max_pixels is not None and max_pixels > 0 and first_attn.shape[2] > max_pixels:
                step = max(1, (first_attn.shape[2] + max_pixels - 1) // max_pixels)
                first_eval = first_attn[:, :, ::step, :][:, :, :max_pixels, :]
                reread_eval = reread_attn[:, :, ::step, :][:, :, :max_pixels, :]
            else:
                first_eval = first_attn
                reread_eval = reread_attn
            points = memory_points_norm.detach().float()
            a1 = first_eval.detach().float().mean(dim=1)
            a2 = reread_eval.detach().float().mean(dim=1)
            p1 = a1 @ points
            p2 = a2 @ points
            shift = (p2 - p1).norm(dim=-1)
            dist1 = ((points.unsqueeze(1) - p1.unsqueeze(2)) ** 2).sum(dim=-1)
            dist2 = ((points.unsqueeze(1) - p2.unsqueeze(2)) ** 2).sum(dim=-1)
            spread1 = (a1 * dist1).sum(dim=-1)
            spread2 = (a2 * dist2).sum(dim=-1)
            return {
                "reread_geo_p2_p1_shift_mean": float(shift.mean().item()),
                "reread_geo_p2_p1_shift_p50": float(torch.quantile(shift.flatten(), 0.50).item()),
                "reread_geo_p2_p1_shift_p90": float(torch.quantile(shift.flatten(), 0.90).item()),
                "reread_geo_spread_a1_mean": float(spread1.mean().item()),
                "reread_geo_spread_a2_mean": float(spread2.mean().item()),
                "reread_geo_spread_ratio_a2_a1": float(
                    (spread2.mean() / spread1.mean().clamp(min=1e-8)).item()
                ),
            }

    @staticmethod
    def _prefix_stats(prefix, stats):
        return {f"{prefix}{k}": v for k, v in stats.items()}

    def _compute_ccf_gate(self, first_attn):
        """Return a per-query confidence gate from first-read attention concentration."""
        if self.fusion_ccf_gate_source != "first_attn_entropy":
            raise ValueError(f"Unsupported fusion_ccf_gate_source={self.fusion_ccf_gate_source!r}")
        attn = first_attn.float().mean(dim=1)
        eps = 1e-8
        entropy = -(attn * (attn + eps).log()).sum(dim=-1)
        max_entropy = math.log(max(int(first_attn.shape[-1]), 2))
        norm_entropy = (entropy / max_entropy).clamp(0.0, 1.0)
        confidence = (1.0 - norm_entropy).clamp(0.0, 1.0)
        gate = confidence.pow(float(self.fusion_ccf_gate_gamma))
        gate = float(self.fusion_ccf_gate_floor) + (1.0 - float(self.fusion_ccf_gate_floor)) * gate
        gate = gate.clamp(0.0, 1.0).unsqueeze(-1)
        if self.fusion_ccf_detach_gate:
            gate = gate.detach()
            confidence = confidence.detach()
            norm_entropy = norm_entropy.detach()
        return gate.to(device=first_attn.device, dtype=first_attn.dtype), confidence, norm_entropy

    def _forward_single_or_cascade(self, query_feats, latent_z, latent_p, scene_center,
                                   return_stats=False, stats_max_pixels=4096):
        if self.fusion_refinement_mode in (
            "progressive_reread", "centered_reread",
            "centered_reread_qknorm_layerscale",
            "ccf_centered_reread_qknorm_layerscale",
            "geometry_reread_lite",
            "v3_dual_refine"
        ):
            k, v, pe_input = self.fusion_single.encode_memory(latent_z, latent_p, scene_center)
            anchor_feats, first_delta, first_attn = self.fusion_single.fuse_encoded_memory(
                query_feats, k, v
            )
            reread_attention_bias = None
            if self.fusion_refinement_mode == "geometry_reread_lite":
                reread_attention_bias = self._build_reread_geometry_bias(
                    first_attn,
                    pe_input,
                    self.fusion_reread_geo_lambda,
                    self.fusion_reread_geo_sigma,
                    self.fusion_reread_geo_sigma_mode,
                    self.fusion_reread_geo_sigma_beta,
                    self.fusion_reread_geo_sigma_min,
                )
            _, reread_delta, reread_attn = self.progressive_reread(
                anchor_feats,
                k,
                v,
                attention_bias=reread_attention_bias,
                reference_attn=(
                    first_attn if self.fusion_refinement_mode in (
                        "centered_reread", "centered_reread_qknorm_layerscale",
                        "ccf_centered_reread_qknorm_layerscale",
                        "v3_dual_refine"
                    ) else None
                ),
            )
            reread_gate = None
            reread_effective_scale = reread_delta.new_tensor(float(self.fusion_reread_delta_alpha))
            if self.fusion_reread_scalar_gate:
                reread_gate = torch.sigmoid(self.fusion_reread_gate_logit).to(
                    device=reread_delta.device,
                    dtype=reread_delta.dtype,
                )
                reread_effective_scale = reread_effective_scale * reread_gate

            reread_update_delta = reread_delta
            reread_patch_delta = None
            reread_common_delta = None
            if self.fusion_refinement_mode in (
                "centered_reread_qknorm_layerscale",
                "ccf_centered_reread_qknorm_layerscale",
                "v3_dual_refine"
            ):
                reread_common_delta = reread_update_delta.mean(dim=1, keepdim=True)
                reread_patch_delta = reread_update_delta - reread_common_delta
                if self.fusion_refinement_mode == "v3_dual_refine":
                    gamma_patch = self.fusion_dual_memory_gamma_patch.to(
                        device=reread_update_delta.device, dtype=reread_update_delta.dtype
                    ).view(1, 1, -1)
                    gamma_common = self.fusion_dual_memory_gamma_common.to(
                        device=reread_update_delta.device, dtype=reread_update_delta.dtype
                    ).view(1, 1, -1)
                else:
                    gamma_patch = self.fusion_reread_gamma_patch.to(
                        device=reread_update_delta.device, dtype=reread_update_delta.dtype
                    ).view(1, 1, -1)
                    gamma_common = self.fusion_reread_gamma_common.to(
                        device=reread_update_delta.device, dtype=reread_update_delta.dtype
                    ).view(1, 1, -1)
                reread_update_delta = (
                    gamma_patch * reread_patch_delta
                    + gamma_common * float(self.fusion_reread_common_scale) * reread_common_delta
                )
            elif self.fusion_reread_common_scale != 1.0:
                reread_common_delta = reread_update_delta.mean(dim=1, keepdim=True)
                reread_update_delta = (
                    reread_update_delta
                    - reread_common_delta
                    + float(self.fusion_reread_common_scale) * reread_common_delta
                )
            trust_scale = None
            raw_delta_ratio = None
            if self.fusion_reread_trust_region_ratio > 0.0:
                with torch.no_grad():
                    eps = 1e-8
                    delta_norm = reread_update_delta.detach().float().norm(dim=-1, keepdim=True)
                    anchor_norm = anchor_feats.detach().float().norm(dim=-1, keepdim=True)
                    raw_delta_ratio = delta_norm / anchor_norm.clamp(min=eps)
                    max_delta_norm = float(self.fusion_reread_trust_region_ratio) * anchor_norm
                    trust_scale = (max_delta_norm / delta_norm.clamp(min=eps)).clamp(max=1.0)
                    trust_scale = trust_scale.to(device=reread_update_delta.device, dtype=reread_update_delta.dtype)
                reread_update_delta = reread_update_delta * trust_scale

            reread_warmup_scale = reread_delta.new_tensor(self.get_reread_warmup_scale())
            reread_effective_update_before_warmup = reread_effective_scale * reread_update_delta
            reread_effective_update = reread_effective_update_before_warmup * reread_warmup_scale
            ccf_gate = None
            ccf_confidence = None
            ccf_norm_entropy = None
            reread_effective_update_before_ccf = reread_effective_update
            if self.fusion_refinement_mode == "ccf_centered_reread_qknorm_layerscale":
                ccf_gate, ccf_confidence, ccf_norm_entropy = self._compute_ccf_gate(first_attn)
                ccf_gate = ccf_gate.to(device=reread_effective_update.device, dtype=reread_effective_update.dtype)
                reread_effective_update = reread_effective_update * ccf_gate
            reread_effective_update_after_ccf = reread_effective_update

            dual_adapter_delta = None
            dual_adapter_patch_delta = None
            dual_adapter_common_delta = None
            dual_adapter_update_delta = None
            dual_adapter_effective_update_before_warmup = None
            dual_adapter_effective_update = None
            if self.fusion_refinement_mode == "v3_dual_refine":
                dual_adapter_delta = self.adapter_ffn.compute_delta(anchor_feats)
                dual_adapter_common_delta = dual_adapter_delta.mean(dim=1, keepdim=True)
                dual_adapter_patch_delta = dual_adapter_delta - dual_adapter_common_delta
                query_gamma_patch = self.fusion_reread_gamma_patch.to(
                    device=dual_adapter_delta.device, dtype=dual_adapter_delta.dtype
                ).view(1, 1, -1)
                query_gamma_common = self.fusion_reread_gamma_common.to(
                    device=dual_adapter_delta.device, dtype=dual_adapter_delta.dtype
                ).view(1, 1, -1)
                dual_adapter_update_delta = (
                    query_gamma_patch * dual_adapter_patch_delta
                    + query_gamma_common * float(self.fusion_reread_common_scale) * dual_adapter_common_delta
                )
                dual_adapter_effective_update_before_warmup = (
                    reread_effective_scale * dual_adapter_update_delta
                )
                dual_adapter_effective_update = (
                    dual_adapter_effective_update_before_warmup * reread_warmup_scale
                )

            total_effective_update = reread_effective_update
            if dual_adapter_effective_update is not None:
                total_effective_update = total_effective_update + dual_adapter_effective_update
            effective_ratio_scale = None
            if self.fusion_reread_effective_ratio_cap > 0.0:
                eps = 1e-8
                with torch.no_grad():
                    anchor_norm = anchor_feats.detach().float().norm(dim=-1, keepdim=True)
                effective_update_norm = total_effective_update.float().norm(
                    dim=-1, keepdim=True
                )
                effective_ratio = effective_update_norm / anchor_norm.clamp(min=eps)
                image_mean_ratio = effective_ratio.mean(dim=1, keepdim=True)
                effective_ratio_scale = (
                    float(self.fusion_reread_effective_ratio_cap)
                    / image_mean_ratio.clamp(min=eps)
                ).clamp(max=1.0)
                effective_ratio_scale = effective_ratio_scale.to(
                    device=total_effective_update.device,
                    dtype=total_effective_update.dtype,
                )
                reread_effective_update = reread_effective_update * effective_ratio_scale
                if dual_adapter_effective_update is not None:
                    dual_adapter_effective_update = dual_adapter_effective_update * effective_ratio_scale
                total_effective_update = total_effective_update * effective_ratio_scale

            fused = anchor_feats + total_effective_update
            if self.fusion_reread_post_norm:
                fused = self.progressive_reread.output_norm(fused)
            if not return_stats:
                return fused

            pe_eval = pe_input.detach().float()
            stats = self.fusion_single._summarize_attention(
                first_attn,
                query_feats,
                first_delta,
                anchor_feats,
                max_pixels=stats_max_pixels,
                extra_stats={
                    "fusion_geometry_mode": self.fusion_single.fusion_geometry_mode,
                    "fusion_scene_scale": float(
                        self.fusion_single.fusion_scene_scale.detach().cpu().item()
                    ),
                    "key_geo_scale": float(
                        getattr(
                            self.fusion_single,
                            "key_geo_scale",
                            torch.tensor(0.0, device=pe_input.device),
                        ).detach().float().cpu().item()
                    ),
                    "memory_p_norm_std": float(pe_eval.std(unbiased=False).item()),
                    "memory_p_norm_absmax": float(pe_eval.abs().max().item()),
                    "memory_p_norm_finite": bool(torch.isfinite(pe_eval).all().item()),
                },
            )
            reread_stats = self.fusion_single._summarize_attention(
                reread_attn,
                anchor_feats,
                reread_delta,
                fused,
                max_pixels=stats_max_pixels,
            )
            stats.update(self._prefix_stats("reread_", reread_stats))
            stats.update(self._summarize_routing_change(
                first_attn,
                reread_attn,
                max_pixels=stats_max_pixels,
            ))
            with torch.no_grad():
                stats.update({
                    "fusion_refinement_mode": self.fusion_refinement_mode,
                    "fusion_cascade_layers": 2,
                    "reread_delta_alpha": float(self.fusion_reread_delta_alpha),
                    "reread_scalar_gate": bool(self.fusion_reread_scalar_gate),
                    "reread_post_norm": bool(self.fusion_reread_post_norm),
                    "reread_centered_context": self.fusion_refinement_mode in (
                        "centered_reread", "centered_reread_qknorm_layerscale",
                        "ccf_centered_reread_qknorm_layerscale",
                        "v3_dual_refine"
                    ),
                    "reread_qk_norm": bool(
                        getattr(self.progressive_reread, "qk_norm", False)
                        if self.progressive_reread is not None else False
                    ),
                    "reread_qknorm_tau_mean": float(
                        self.progressive_reread.qk_norm_log_tau.detach().float().exp().mean().item()
                        if (
                            self.progressive_reread is not None
                            and getattr(self.progressive_reread, "qk_norm", False)
                            and self.progressive_reread.qk_norm_log_tau.numel() > 0
                        ) else 0.0
                    ),
                    "reread_qknorm_tau_min": float(
                        self.progressive_reread.qk_norm_log_tau.detach().float().exp().min().item()
                        if (
                            self.progressive_reread is not None
                            and getattr(self.progressive_reread, "qk_norm", False)
                            and self.progressive_reread.qk_norm_log_tau.numel() > 0
                        ) else 0.0
                    ),
                    "reread_qknorm_tau_max": float(
                        self.progressive_reread.qk_norm_log_tau.detach().float().exp().max().item()
                        if (
                            self.progressive_reread is not None
                            and getattr(self.progressive_reread, "qk_norm", False)
                            and self.progressive_reread.qk_norm_log_tau.numel() > 0
                        ) else 0.0
                    ),
                    "reread_temperature": float(self.fusion_reread_temperature),
                    "reread_common_scale": float(self.fusion_reread_common_scale),
                    "reread_effective_ratio_cap": float(
                        self.fusion_reread_effective_ratio_cap
                    ),
                    "reread_warmup_mode": self.fusion_reread_warmup_mode,
                    "reread_warmup_iters": int(self.fusion_reread_warmup_iters),
                    "reread_warmup_start": float(self.fusion_reread_warmup_start),
                    "reread_warmup_scale": float(reread_warmup_scale.detach().float().cpu().item()),
                    "reread_effective_ratio_scale_mean": float(
                        effective_ratio_scale.detach().float().mean().item()
                        if effective_ratio_scale is not None else 1.0
                    ),
                    "reread_effective_ratio_clamp_fraction": float(
                        (effective_ratio_scale.detach().float() < 0.999).float().mean().item()
                        if effective_ratio_scale is not None else 0.0
                    ),
                    "reread_effective_ratio_before_warmup_mean": float(
                        (
                            reread_effective_update_before_warmup.detach().float().norm(dim=-1)
                            / anchor_feats.detach().float().norm(dim=-1).clamp(min=1e-8)
                        ).mean().item()
                    ),
                    "reread_effective_ratio_before_warmup_p90": float(
                        torch.quantile(
                            (
                                reread_effective_update_before_warmup.detach().float().norm(dim=-1)
                                / anchor_feats.detach().float().norm(dim=-1).clamp(min=1e-8)
                            ).flatten(),
                            0.90,
                        ).item()
                    ),
                    "reread_effective_ratio_raw_mean": float(
                        (
                            reread_effective_update.detach().float().norm(dim=-1)
                            / anchor_feats.detach().float().norm(dim=-1).clamp(min=1e-8)
                        ).mean().item()
                    ),
                    "reread_effective_ratio_raw_p90": float(
                        torch.quantile(
                            (
                                reread_effective_update.detach().float().norm(dim=-1)
                                / anchor_feats.detach().float().norm(dim=-1).clamp(min=1e-8)
                            ).flatten(),
                            0.90,
                        ).item()
                    ),
                    "reread_effective_ratio_applied_mean": float(
                        (
                            reread_effective_update.detach().float().norm(dim=-1)
                            / anchor_feats.detach().float().norm(dim=-1).clamp(min=1e-8)
                        ).mean().item()
                    ),
                    "reread_effective_ratio_applied_p90": float(
                        torch.quantile(
                            (
                                reread_effective_update.detach().float().norm(dim=-1)
                                / anchor_feats.detach().float().norm(dim=-1).clamp(min=1e-8)
                            ).flatten(),
                            0.90,
                        ).item()
                    ),
                    "reread_effective_update_common_ratio": float(
                        reread_effective_update.detach().float().mean(dim=1).norm(dim=-1).mean().div(
                            reread_effective_update.detach().float().norm(dim=-1).mean().clamp(min=1e-8)
                        ).item()
                    ),
                    "reread_trust_region_ratio": float(self.fusion_reread_trust_region_ratio),
                    "reread_trust_region_scale_mean": float(
                        trust_scale.detach().float().mean().item()
                        if trust_scale is not None else 1.0
                    ),
                    "reread_trust_region_clamp_fraction": float(
                        (trust_scale.detach().float() < 0.999).float().mean().item()
                        if trust_scale is not None else 0.0
                    ),
                    "reread_raw_delta_ratio_mean": float(
                        raw_delta_ratio.mean().item()
                        if raw_delta_ratio is not None else 0.0
                    ),
                    "reread_raw_delta_ratio_p90": float(
                        torch.quantile(raw_delta_ratio.flatten(), 0.90).item()
                        if raw_delta_ratio is not None else 0.0
                    ),
                    "reread_update_delta_norm": float(
                        reread_update_delta.detach().float().norm(dim=-1).mean().item()
                    ),
                    "reread_layerscale_patch_mean": float(
                        (
                            self.fusion_dual_memory_gamma_patch
                            if self.fusion_refinement_mode == "v3_dual_refine"
                            else self.fusion_reread_gamma_patch
                        ).detach().float().mean().item()
                    ),
                    "reread_layerscale_patch_absmax": float(
                        (
                            self.fusion_dual_memory_gamma_patch
                            if self.fusion_refinement_mode == "v3_dual_refine"
                            else self.fusion_reread_gamma_patch
                        ).detach().float().abs().max().item()
                    ),
                    "reread_layerscale_common_mean": float(
                        (
                            self.fusion_dual_memory_gamma_common
                            if self.fusion_refinement_mode == "v3_dual_refine"
                            else self.fusion_reread_gamma_common
                        ).detach().float().mean().item()
                    ),
                    "reread_layerscale_common_absmax": float(
                        (
                            self.fusion_dual_memory_gamma_common
                            if self.fusion_refinement_mode == "v3_dual_refine"
                            else self.fusion_reread_gamma_common
                        ).detach().float().abs().max().item()
                    ),
                    "reread_patch_delta_norm": float(
                        reread_patch_delta.detach().float().norm(dim=-1).mean().item()
                        if reread_patch_delta is not None else 0.0
                    ),
                    "reread_common_delta_norm": float(
                        reread_common_delta.detach().float().norm(dim=-1).mean().item()
                        if reread_common_delta is not None else 0.0
                    ),
                    "reread_effective_update_norm_before_warmup": float(
                        reread_effective_update_before_warmup.detach().float().norm(dim=-1).mean().item()
                    ),
                    "reread_effective_update_norm_after_warmup": float(
                        reread_effective_update.detach().float().norm(dim=-1).mean().item()
                    ),
                    "reread_update_delta_common_ratio": float(
                        reread_update_delta.detach().float().mean(dim=1).norm(dim=-1).mean().div(
                            reread_update_delta.detach().float().norm(dim=-1).mean().clamp(min=1e-8)
                        ).item()
                    ),
                    "reread_geo_lambda": float(
                        self.fusion_reread_geo_lambda
                        if self.fusion_refinement_mode == "geometry_reread_lite" else 0.0
                    ),
                    "reread_geo_sigma": float(
                        self.fusion_reread_geo_sigma
                        if self.fusion_refinement_mode == "geometry_reread_lite" else 0.0
                    ),
                    "reread_geo_sigma_mode": (
                        self.fusion_reread_geo_sigma_mode
                        if self.fusion_refinement_mode == "geometry_reread_lite" else "fixed"
                    ),
                    "reread_geo_sigma_beta": float(
                        self.fusion_reread_geo_sigma_beta
                        if self.fusion_refinement_mode == "geometry_reread_lite" else 0.0
                    ),
                    "reread_geo_sigma_min": float(
                        self.fusion_reread_geo_sigma_min
                        if self.fusion_refinement_mode == "geometry_reread_lite" else 0.0
                    ),
                    "reread_gate_value": float(
                        reread_gate.detach().float().cpu().item()
                        if reread_gate is not None else 1.0
                    ),
                    "reread_gate_logit": float(
                        self.fusion_reread_gate_logit.detach().float().cpu().item()
                    ),
                    "reread_effective_delta_scale": float(
                        reread_effective_scale.detach().float().cpu().item()
                    ),
                    "reread_delta_norm": float(
                        reread_delta.detach().float().norm(dim=-1).mean().item()
                    ),
                    "reread_delta_patch_mean_norm": float(
                        reread_delta.detach().float().mean(dim=1).norm(dim=-1).mean().item()
                    ),
                    "reread_delta_patch_std_norm": float(
                        (
                            reread_delta.detach().float()
                            - reread_delta.detach().float().mean(dim=1, keepdim=True)
                        ).norm(dim=-1).mean().item()
                    ),
                    "reread_delta_common_ratio": float(
                        reread_delta.detach().float().mean(dim=1).norm(dim=-1).mean().div(
                            reread_delta.detach().float().norm(dim=-1).mean().clamp(min=1e-8)
                        ).item()
                    ),
                    "anchor_reread_cosine": float(
                        F.cosine_similarity(
                            anchor_feats.detach().float(),
                            fused.detach().float(),
                            dim=-1,
                        ).mean().item()
                    ),
                    "fused_feature_norm": float(
                        fused.detach().float().norm(dim=-1).mean().item()
                    ),
                })
                before_ccf_ratio = (
                    reread_effective_update_before_ccf.detach().float().norm(dim=-1)
                    / anchor_feats.detach().float().norm(dim=-1).clamp(min=1e-8)
                )
                after_ccf_ratio = (
                    reread_effective_update_after_ccf.detach().float().norm(dim=-1)
                    / anchor_feats.detach().float().norm(dim=-1).clamp(min=1e-8)
                )
                if ccf_gate is not None:
                    ccf_gate_eval = ccf_gate.detach().float().squeeze(-1)
                    ccf_conf_eval = ccf_confidence.detach().float()
                    ccf_entropy_eval = ccf_norm_entropy.detach().float()
                    stats.update({
                        "ccf_gate_enabled": True,
                        "ccf_gate_source": self.fusion_ccf_gate_source,
                        "ccf_gate_floor": float(self.fusion_ccf_gate_floor),
                        "ccf_gate_gamma": float(self.fusion_ccf_gate_gamma),
                        "ccf_detach_gate": bool(self.fusion_ccf_detach_gate),
                        "ccf_gate_mean": float(ccf_gate_eval.mean().item()),
                        "ccf_gate_p10": float(torch.quantile(ccf_gate_eval.flatten(), 0.10).item()),
                        "ccf_gate_p50": float(torch.quantile(ccf_gate_eval.flatten(), 0.50).item()),
                        "ccf_gate_p90": float(torch.quantile(ccf_gate_eval.flatten(), 0.90).item()),
                        "ccf_confidence_mean": float(ccf_conf_eval.mean().item()),
                        "ccf_confidence_p10": float(torch.quantile(ccf_conf_eval.flatten(), 0.10).item()),
                        "ccf_confidence_p50": float(torch.quantile(ccf_conf_eval.flatten(), 0.50).item()),
                        "ccf_confidence_p90": float(torch.quantile(ccf_conf_eval.flatten(), 0.90).item()),
                        "ccf_norm_entropy_mean": float(ccf_entropy_eval.mean().item()),
                        "ccf_norm_entropy_p50": float(torch.quantile(ccf_entropy_eval.flatten(), 0.50).item()),
                        "reread_effective_ratio_before_ccf_mean": float(before_ccf_ratio.mean().item()),
                        "reread_effective_ratio_before_ccf_p90": float(
                            torch.quantile(before_ccf_ratio.flatten(), 0.90).item()
                        ),
                        "reread_effective_ratio_after_ccf_mean": float(after_ccf_ratio.mean().item()),
                        "reread_effective_ratio_after_ccf_p90": float(
                            torch.quantile(after_ccf_ratio.flatten(), 0.90).item()
                        ),
                        "ccf_gate_delta_ratio_mean": float(
                            after_ccf_ratio.mean().div(before_ccf_ratio.mean().clamp(min=1e-8)).item()
                        ),
                    })
                else:
                    stats.update({
                        "ccf_gate_enabled": False,
                        "ccf_gate_source": self.fusion_ccf_gate_source,
                        "ccf_gate_floor": float(self.fusion_ccf_gate_floor),
                        "ccf_gate_gamma": float(self.fusion_ccf_gate_gamma),
                        "ccf_detach_gate": bool(self.fusion_ccf_detach_gate),
                        "ccf_gate_mean": 1.0,
                        "ccf_gate_p10": 1.0,
                        "ccf_gate_p50": 1.0,
                        "ccf_gate_p90": 1.0,
                        "ccf_confidence_mean": 0.0,
                        "ccf_confidence_p10": 0.0,
                        "ccf_confidence_p50": 0.0,
                        "ccf_confidence_p90": 0.0,
                        "ccf_norm_entropy_mean": 0.0,
                        "ccf_norm_entropy_p50": 0.0,
                        "reread_effective_ratio_before_ccf_mean": float(before_ccf_ratio.mean().item()),
                        "reread_effective_ratio_before_ccf_p90": float(
                            torch.quantile(before_ccf_ratio.flatten(), 0.90).item()
                        ),
                        "reread_effective_ratio_after_ccf_mean": float(after_ccf_ratio.mean().item()),
                        "reread_effective_ratio_after_ccf_p90": float(
                            torch.quantile(after_ccf_ratio.flatten(), 0.90).item()
                        ),
                        "ccf_gate_delta_ratio_mean": 1.0,
                    })
                if self.fusion_refinement_mode == "v3_dual_refine":
                    query_effective_ratio = (
                        dual_adapter_effective_update.detach().float().norm(dim=-1)
                        / anchor_feats.detach().float().norm(dim=-1).clamp(min=1e-8)
                    )
                    total_effective_ratio = (
                        total_effective_update.detach().float().norm(dim=-1)
                        / anchor_feats.detach().float().norm(dim=-1).clamp(min=1e-8)
                    )
                    query_update_common_ratio = (
                        dual_adapter_update_delta.detach().float().mean(dim=1).norm(dim=-1).mean()
                        / dual_adapter_update_delta.detach().float().norm(dim=-1).mean().clamp(min=1e-8)
                    )
                    total_update_common_ratio = (
                        total_effective_update.detach().float().mean(dim=1).norm(dim=-1).mean()
                        / total_effective_update.detach().float().norm(dim=-1).mean().clamp(min=1e-8)
                    )
                    stats.update({
                        "dual_query_delta_norm": float(
                            dual_adapter_delta.detach().float().norm(dim=-1).mean().item()
                        ),
                        "dual_query_patch_delta_norm": float(
                            dual_adapter_patch_delta.detach().float().norm(dim=-1).mean().item()
                        ),
                        "dual_query_common_delta_norm": float(
                            dual_adapter_common_delta.detach().float().norm(dim=-1).mean().item()
                        ),
                        "dual_query_update_delta_norm": float(
                            dual_adapter_update_delta.detach().float().norm(dim=-1).mean().item()
                        ),
                        "dual_query_effective_update_norm_before_warmup": float(
                            dual_adapter_effective_update_before_warmup.detach().float().norm(dim=-1).mean().item()
                        ),
                        "dual_query_effective_update_norm_after_warmup": float(
                            dual_adapter_effective_update.detach().float().norm(dim=-1).mean().item()
                        ),
                        "dual_query_effective_ratio_mean": float(query_effective_ratio.mean().item()),
                        "dual_query_effective_ratio_p90": float(
                            torch.quantile(query_effective_ratio.flatten(), 0.90).item()
                        ),
                        "dual_query_update_delta_common_ratio": float(query_update_common_ratio.item()),
                        "dual_query_layerscale_patch_mean": float(
                            self.fusion_reread_gamma_patch.detach().float().mean().item()
                        ),
                        "dual_query_layerscale_patch_absmax": float(
                            self.fusion_reread_gamma_patch.detach().float().abs().max().item()
                        ),
                        "dual_query_layerscale_common_mean": float(
                            self.fusion_reread_gamma_common.detach().float().mean().item()
                        ),
                        "dual_query_layerscale_common_absmax": float(
                            self.fusion_reread_gamma_common.detach().float().abs().max().item()
                        ),
                        "dual_memory_layerscale_patch_init": float(
                            self.fusion_dual_memory_layerscale_patch_init
                        ),
                        "dual_memory_layerscale_common_init": float(
                            self.fusion_dual_memory_layerscale_common_init
                        ),
                        "dual_total_update_norm": float(
                            total_effective_update.detach().float().norm(dim=-1).mean().item()
                        ),
                        "dual_total_effective_ratio_mean": float(total_effective_ratio.mean().item()),
                        "dual_total_effective_ratio_p90": float(
                            torch.quantile(total_effective_ratio.flatten(), 0.90).item()
                        ),
                        "dual_total_update_common_ratio": float(total_update_common_ratio.item()),
                        "dual_memory_query_update_cosine": float(
                            F.cosine_similarity(
                                reread_effective_update.detach().float(),
                                dual_adapter_effective_update.detach().float(),
                                dim=-1,
                            ).mean().item()
                        ),
                    })
                if self.fusion_refinement_mode == "geometry_reread_lite":
                    stats.update(self._summarize_geometry_reread(
                        first_attn,
                        reread_attn,
                        pe_input,
                        max_pixels=stats_max_pixels,
                    ))
            return fused, stats


        if self.fusion_refinement_mode == "coord_prior_v1":
            k, v, pe_input = self.fusion_single.encode_memory(latent_z, latent_p, scene_center)
            anchor_feats, first_delta, first_attn = self.fusion_single.fuse_encoded_memory(
                query_feats, k, v
            )
            prior_attn = first_attn.mean(dim=1)
            memory_points = latent_p.to(device=prior_attn.device, dtype=prior_attn.dtype)
            coord_prior_world = prior_attn @ memory_points
            scene_center_t = scene_center.to(
                device=coord_prior_world.device, dtype=coord_prior_world.dtype
            ).unsqueeze(1)
            coord_scale = self.fusion_single.fusion_scene_scale.to(
                device=coord_prior_world.device, dtype=coord_prior_world.dtype
            ).clamp(min=1e-6)
            coord_prior_norm = (coord_prior_world - scene_center_t) / coord_scale
            coord_prior_embed = self.coord_prior_encoder(coord_prior_norm)
            coord_prior_delta = self.coord_prior_proj(coord_prior_embed)
            coord_prior_gamma = self.coord_prior_gamma.to(
                device=coord_prior_delta.device, dtype=coord_prior_delta.dtype
            )
            coord_prior_update = coord_prior_gamma.view(1, 1, 1) * coord_prior_delta
            fused = anchor_feats + coord_prior_update
            if not return_stats:
                return fused

            pe_eval = pe_input.detach().float()
            stats = self.fusion_single._summarize_attention(
                first_attn,
                query_feats,
                first_delta,
                anchor_feats,
                max_pixels=stats_max_pixels,
                extra_stats={
                    "fusion_geometry_mode": self.fusion_single.fusion_geometry_mode,
                    "fusion_scene_scale": float(
                        self.fusion_single.fusion_scene_scale.detach().cpu().item()
                    ),
                    "key_geo_scale": float(
                        getattr(
                            self.fusion_single,
                            "key_geo_scale",
                            torch.tensor(0.0, device=pe_input.device),
                        ).detach().float().cpu().item()
                    ),
                    "memory_p_norm_std": float(pe_eval.std(unbiased=False).item()),
                    "memory_p_norm_absmax": float(pe_eval.abs().max().item()),
                    "memory_p_norm_finite": bool(torch.isfinite(pe_eval).all().item()),
                    "single_qk_norm": bool(self.fusion_single.single_qk_norm),
                    "single_qknorm_tau_mean": float(
                        self.fusion_single.single_qk_norm_log_tau.detach().float().exp().mean().item()
                        if (
                            self.fusion_single.single_qk_norm
                            and self.fusion_single.single_qk_norm_log_tau.numel() > 0
                        ) else 0.0
                    ),
                    "single_out_layerscale": bool(self.fusion_single.single_out_layerscale),
                },
            )
            with torch.no_grad():
                coord_prior_radius = coord_prior_norm.detach().float().norm(dim=-1)
                coord_prior_update_eval = coord_prior_update.detach().float()
                anchor_eval = anchor_feats.detach().float()
                coord_effective_ratio = (
                    coord_prior_update_eval.norm(dim=-1)
                    / anchor_eval.norm(dim=-1).clamp(min=1e-8)
                )
                coord_common_ratio = (
                    coord_prior_update_eval.mean(dim=1).norm(dim=-1).mean()
                    / coord_prior_update_eval.norm(dim=-1).mean().clamp(min=1e-8)
                )
                stats.update({
                    "fusion_refinement_mode": self.fusion_refinement_mode,
                    "fusion_cascade_layers": 2,
                    "coord_prior_scale_init": float(self.fusion_coord_prior_scale_init),
                    "coord_prior_gamma": float(
                        self.coord_prior_gamma.detach().float().cpu().item()
                    ),
                    "coord_prior_scene_scale": float(
                        self.fusion_single.fusion_scene_scale.detach().float().cpu().item()
                    ),
                    "coord_prior_norm_radius_mean": float(coord_prior_radius.mean().item()),
                    "coord_prior_norm_radius_p90": float(
                        torch.quantile(coord_prior_radius.flatten(), 0.90).item()
                    ),
                    "coord_prior_embed_norm": float(
                        coord_prior_embed.detach().float().norm(dim=-1).mean().item()
                    ),
                    "coord_prior_delta_norm": float(
                        coord_prior_delta.detach().float().norm(dim=-1).mean().item()
                    ),
                    "coord_prior_update_norm": float(
                        coord_prior_update_eval.norm(dim=-1).mean().item()
                    ),
                    "coord_prior_effective_ratio_mean": float(coord_effective_ratio.mean().item()),
                    "coord_prior_effective_ratio_p90": float(
                        torch.quantile(coord_effective_ratio.flatten(), 0.90).item()
                    ),
                    "coord_prior_update_common_ratio": float(coord_common_ratio.item()),
                    "anchor_coord_prior_cosine": float(
                        F.cosine_similarity(anchor_eval, fused.detach().float(), dim=-1).mean().item()
                    ),
                    "fused_feature_norm": float(
                        fused.detach().float().norm(dim=-1).mean().item()
                    ),
                })
            return fused, stats

        first_out = self.fusion_single(
            query_feats,
            latent_z,
            latent_p,
            scene_center,
            return_stats=return_stats,
            stats_max_pixels=stats_max_pixels,
        )
        if return_stats:
            anchor_feats, stats = first_out
        else:
            anchor_feats = first_out
            stats = None

        if self.fusion_refinement_mode == "weak_residual_ffn":
            adapter_delta = self.adapter_ffn.compute_delta(anchor_feats)
            adapter_gate = None
            adapter_effective_scale = adapter_delta.new_tensor(float(self.fusion_reread_delta_alpha))
            if self.fusion_reread_scalar_gate:
                adapter_gate = torch.sigmoid(self.fusion_reread_gate_logit).to(
                    device=adapter_delta.device,
                    dtype=adapter_delta.dtype,
                )
                adapter_effective_scale = adapter_effective_scale * adapter_gate
            fused = anchor_feats + adapter_effective_scale * adapter_delta
            if self.fusion_reread_post_norm:
                fused = self.adapter_ffn.output_norm(fused)
            if return_stats:
                stats = dict(stats)
                with torch.no_grad():
                    stats.update({
                        "fusion_refinement_mode": self.fusion_refinement_mode,
                        "fusion_cascade_layers": 2,
                        "weak_ffn_delta_alpha": float(self.fusion_reread_delta_alpha),
                        "weak_ffn_scalar_gate": bool(self.fusion_reread_scalar_gate),
                        "weak_ffn_post_norm": bool(self.fusion_reread_post_norm),
                        "weak_ffn_gate_value": float(
                            adapter_gate.detach().float().cpu().item()
                            if adapter_gate is not None else 1.0
                        ),
                        "weak_ffn_gate_logit": float(
                            self.fusion_reread_gate_logit.detach().float().cpu().item()
                        ),
                        "weak_ffn_effective_delta_scale": float(
                            adapter_effective_scale.detach().float().cpu().item()
                        ),
                        "weak_ffn_delta_norm": float(
                            adapter_delta.detach().float().norm(dim=-1).mean().item()
                        ),
                        "anchor_weak_ffn_cosine": float(
                            F.cosine_similarity(
                                anchor_feats.detach().float(),
                                fused.detach().float(),
                                dim=-1,
                            ).mean().item()
                        ),
                        "fused_feature_norm": float(
                            fused.detach().float().norm(dim=-1).mean().item()
                        ),
                    })
                return fused, stats
            return fused

        if self.fusion_refinement_mode == "v3_adapter_control":
            adapter_delta = self.adapter_ffn.compute_delta(anchor_feats)
            adapter_common_delta = adapter_delta.mean(dim=1, keepdim=True)
            adapter_patch_delta = adapter_delta - adapter_common_delta
            gamma_patch = self.fusion_reread_gamma_patch.to(
                device=adapter_delta.device, dtype=adapter_delta.dtype
            ).view(1, 1, -1)
            gamma_common = self.fusion_reread_gamma_common.to(
                device=adapter_delta.device, dtype=adapter_delta.dtype
            ).view(1, 1, -1)
            adapter_update_delta = (
                gamma_patch * adapter_patch_delta
                + gamma_common * float(self.fusion_reread_common_scale) * adapter_common_delta
            )
            adapter_effective_scale = adapter_delta.new_tensor(float(self.fusion_reread_delta_alpha))
            adapter_warmup_scale = adapter_delta.new_tensor(self.get_reread_warmup_scale())
            adapter_effective_update_before_warmup = adapter_effective_scale * adapter_update_delta
            adapter_effective_update = adapter_effective_update_before_warmup * adapter_warmup_scale
            effective_ratio_scale = None
            if self.fusion_reread_effective_ratio_cap > 0.0:
                eps = 1e-8
                with torch.no_grad():
                    anchor_norm = anchor_feats.detach().float().norm(dim=-1, keepdim=True)
                effective_update_norm = adapter_effective_update.float().norm(
                    dim=-1, keepdim=True
                )
                effective_ratio = effective_update_norm / anchor_norm.clamp(min=eps)
                image_mean_ratio = effective_ratio.mean(dim=1, keepdim=True)
                effective_ratio_scale = (
                    float(self.fusion_reread_effective_ratio_cap)
                    / image_mean_ratio.clamp(min=eps)
                ).clamp(max=1.0)
                effective_ratio_scale = effective_ratio_scale.to(
                    device=adapter_effective_update.device,
                    dtype=adapter_effective_update.dtype,
                )
                adapter_effective_update = adapter_effective_update * effective_ratio_scale
            fused = anchor_feats + adapter_effective_update
            if return_stats:
                stats = dict(stats)
                with torch.no_grad():
                    adapter_effective_ratio = (
                        adapter_effective_update.detach().float().norm(dim=-1)
                        / anchor_feats.detach().float().norm(dim=-1).clamp(min=1e-8)
                    )
                    adapter_update_common_ratio = (
                        adapter_update_delta.detach().float().mean(dim=1).norm(dim=-1).mean()
                        / adapter_update_delta.detach().float().norm(dim=-1).mean().clamp(min=1e-8)
                    )
                    stats.update({
                        "fusion_refinement_mode": self.fusion_refinement_mode,
                        "fusion_cascade_layers": 2,
                        "v3_adapter_delta_alpha": float(self.fusion_reread_delta_alpha),
                        "v3_adapter_post_norm": False,
                        "v3_adapter_common_scale": float(self.fusion_reread_common_scale),
                        "v3_adapter_effective_ratio_cap": float(self.fusion_reread_effective_ratio_cap),
                        "v3_adapter_warmup_mode": self.fusion_reread_warmup_mode,
                        "v3_adapter_warmup_iters": int(self.fusion_reread_warmup_iters),
                        "v3_adapter_warmup_start": float(self.fusion_reread_warmup_start),
                        "v3_adapter_warmup_scale": float(
                            adapter_warmup_scale.detach().float().cpu().item()
                        ),
                        "v3_adapter_effective_ratio_scale_mean": float(
                            effective_ratio_scale.detach().float().mean().item()
                            if effective_ratio_scale is not None else 1.0
                        ),
                        "v3_adapter_effective_ratio_clamp_fraction": float(
                            (effective_ratio_scale.detach().float() < 0.999).float().mean().item()
                            if effective_ratio_scale is not None else 0.0
                        ),
                        "v3_adapter_effective_ratio_mean": float(adapter_effective_ratio.mean().item()),
                        "v3_adapter_effective_ratio_p90": float(
                            torch.quantile(adapter_effective_ratio.flatten(), 0.90).item()
                        ),
                        "v3_adapter_delta_norm": float(
                            adapter_delta.detach().float().norm(dim=-1).mean().item()
                        ),
                        "v3_adapter_patch_delta_norm": float(
                            adapter_patch_delta.detach().float().norm(dim=-1).mean().item()
                        ),
                        "v3_adapter_common_delta_norm": float(
                            adapter_common_delta.detach().float().norm(dim=-1).mean().item()
                        ),
                        "v3_adapter_update_delta_norm": float(
                            adapter_update_delta.detach().float().norm(dim=-1).mean().item()
                        ),
                        "v3_adapter_effective_update_norm_before_warmup": float(
                            adapter_effective_update_before_warmup.detach().float().norm(dim=-1).mean().item()
                        ),
                        "v3_adapter_effective_update_norm_after_warmup": float(
                            adapter_effective_update.detach().float().norm(dim=-1).mean().item()
                        ),
                        "v3_adapter_update_delta_common_ratio": float(
                            adapter_update_common_ratio.item()
                        ),
                        "v3_adapter_layerscale_patch_mean": float(
                            self.fusion_reread_gamma_patch.detach().float().mean().item()
                        ),
                        "v3_adapter_layerscale_patch_absmax": float(
                            self.fusion_reread_gamma_patch.detach().float().abs().max().item()
                        ),
                        "v3_adapter_layerscale_common_mean": float(
                            self.fusion_reread_gamma_common.detach().float().mean().item()
                        ),
                        "v3_adapter_layerscale_common_absmax": float(
                            self.fusion_reread_gamma_common.detach().float().abs().max().item()
                        ),
                        "anchor_v3_adapter_cosine": float(
                            F.cosine_similarity(
                                anchor_feats.detach().float(),
                                fused.detach().float(),
                                dim=-1,
                            ).mean().item()
                        ),
                        "fused_feature_norm": float(
                            fused.detach().float().norm(dim=-1).mean().item()
                        ),
                    })
                return fused, stats
            return fused

        if self.fusion_refinement_mode == "adapter_ffn":
            fused, adapter_delta = self.adapter_ffn(anchor_feats)
            if return_stats:
                stats = dict(stats)
                with torch.no_grad():
                    stats.update({
                        "fusion_refinement_mode": self.fusion_refinement_mode,
                        "fusion_cascade_layers": 2,
                        "adapter_delta_norm": float(
                            adapter_delta.detach().float().norm(dim=-1).mean().item()
                        ),
                        "anchor_adapter_cosine": float(
                            F.cosine_similarity(
                                anchor_feats.detach().float(),
                                fused.detach().float(),
                                dim=-1,
                            ).mean().item()
                        ),
                        "fused_feature_norm": float(
                            fused.detach().float().norm(dim=-1).mean().item()
                        ),
                    })
                return fused, stats
            return fused

        if self.fusion_refinement_mode != "cascade_internal" or len(self.fusion_cascade) == 0:
            if return_stats:
                stats = dict(stats)
                stats.update({
                    "fusion_refinement_mode": self.fusion_refinement_mode,
                    "fusion_cascade_layers": int(self.fusion_cascade_layers),
                    "fusion_assembly_gamma": float(self.fusion_assembly_gamma.detach().float().cpu().item()),
                    "single_qknorm_eps": float(self.fusion_single_qknorm_eps),
                    "single_qknorm_tau_init": float(self.fusion_single_qknorm_tau_init),
                    "single_layerscale_init": float(self.fusion_single_layerscale_init),
                })
                return anchor_feats, stats
            return anchor_feats

        cascade_states = []
        current = anchor_feats
        cascade_stats = {}
        for layer_idx, block in enumerate(self.fusion_cascade, start=2):
            block_out = block(
                current,
                latent_z,
                latent_p,
                scene_center,
                return_stats=return_stats,
                stats_max_pixels=stats_max_pixels,
            )
            if return_stats:
                current, block_stats = block_out
                cascade_stats.update(self._prefix_stats(f"cascade_l{layer_idx}_", block_stats))
            else:
                current = block_out
            cascade_states.append(current)

        residual = self.fusion_assembly(torch.cat(cascade_states, dim=-1))
        gamma = self.fusion_assembly_gamma.to(dtype=residual.dtype)
        fused = anchor_feats + gamma * residual

        if return_stats:
            stats = dict(stats)
            stats.update(cascade_stats)
            with torch.no_grad():
                stats.update({
                    "fusion_refinement_mode": self.fusion_refinement_mode,
                    "fusion_cascade_layers": int(self.fusion_cascade_layers),
                    "fusion_assembly_mode": self.fusion_assembly_mode,
                    "fusion_assembly_gamma": float(self.fusion_assembly_gamma.detach().float().cpu().item()),
                    "fusion_assembly_residual_norm": float(residual.detach().float().norm(dim=-1).mean().item()),
                    "fusion_cascade_output_norm": float(current.detach().float().norm(dim=-1).mean().item()),
                    "fused_feature_norm": float(fused.detach().float().norm(dim=-1).mean().item()),
                })
            return fused, stats
        return fused

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
                stats.update({
                    "fusion_refinement_mode": self.fusion_refinement_mode,
                    "fusion_cascade_layers": int(self.fusion_cascade_layers),
                    "fusion_assembly_gamma": float(self.fusion_assembly_gamma.detach().float().cpu().item()),
                })
                return feats_final, stats
            return fine_out

        latent_z, latent_p = compressor_out
        return self._forward_single_or_cascade(
            query_feats,
            latent_z,
            latent_p,
            scene_center,
            return_stats=return_stats,
            stats_max_pixels=stats_max_pixels,
        )
