"""Single memory-to-image fusion used by the paper method."""

import torch
from torch import nn

from .contracts import AnchoredSceneTokens
from .geolmc import FourierPositionEncoding


class SingleMemoryFusion(nn.Module):
    """One cross-attention read from anchored scene tokens.

    Token geometry is encoded from ``anchor - scene_center`` and added to the
    value path, matching the active ``value_only_raw`` implementation. This is
    an internal implementation detail, not a separate output or supervision.
    """

    def __init__(
        self,
        query_feature_dim: int,
        token_dim: int = 1024,
        output_dim: int = 1024,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        if output_dim % num_heads:
            raise ValueError("output_dim must be divisible by num_heads")
        self.output_dim = output_dim
        self.num_heads = num_heads
        self.head_dim = output_dim // num_heads
        self.scale = self.head_dim**-0.5

        self.position_encoder = FourierPositionEncoding(token_dim)
        self.query_projection = nn.Linear(query_feature_dim, output_dim)
        self.key_projection = nn.Linear(token_dim, output_dim)
        self.value_projection = nn.Linear(token_dim, output_dim)
        self.residual_projection = (
            nn.Identity() if query_feature_dim == output_dim else nn.Linear(query_feature_dim, output_dim)
        )
        self.output_projection = nn.Linear(output_dim, output_dim)
        self.attention_dropout = nn.Dropout(dropout)
        self.attention_norm = nn.LayerNorm(output_dim)
        self.ffn_norm = nn.LayerNorm(output_dim)
        self.ffn = nn.Sequential(
            nn.Linear(output_dim, 4 * output_dim),
            nn.GELU(),
            nn.Linear(4 * output_dim, output_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        query_features: torch.Tensor,
        scene_tokens: AnchoredSceneTokens,
    ) -> torch.Tensor:
        scene_tokens.validate()
        batch_size, num_queries, _ = query_features.shape
        num_tokens = scene_tokens.features.shape[1]
        if scene_tokens.features.shape[0] == 1 and batch_size > 1:
            token_features = scene_tokens.features.expand(batch_size, -1, -1)
            anchors = scene_tokens.anchors.expand(batch_size, -1, -1)
            scene_center = scene_tokens.scene_center.expand(batch_size, -1)
        else:
            token_features = scene_tokens.features
            anchors = scene_tokens.anchors
            scene_center = scene_tokens.scene_center
        if token_features.shape[0] != batch_size:
            raise ValueError("query and scene-token batch sizes are incompatible")

        geometry = self.position_encoder(anchors - scene_center[:, None])
        key_input = token_features
        value_input = token_features + geometry

        query = self.query_projection(query_features)
        key = self.key_projection(key_input)
        value = self.value_projection(value_input)
        query = query.reshape(batch_size, num_queries, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.reshape(batch_size, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.reshape(batch_size, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)

        attention = self.attention_dropout(
            ((query @ key.transpose(-2, -1)) * self.scale).softmax(dim=-1)
        )
        memory_read = (attention @ value).transpose(1, 2).reshape(
            batch_size, num_queries, self.output_dim
        )
        memory_read = self.output_projection(memory_read)
        fused = self.attention_norm(self.residual_projection(query_features) + memory_read)
        return self.ffn_norm(fused + self.ffn(fused))
