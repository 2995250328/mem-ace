"""Global-only GeoLMC used by the paper method overview.

This module intentionally removes local, hierarchical, learned-token, layer
selection, and scale-token branches. It preserves the common method: sample 3D
anchors, initialize latent queries from anchor geometry, and read the pooled
scene memory with distance-aware cross-attention.
"""

import math

import torch
from torch import nn

from .contracts import AnchoredSceneTokens, SceneMemory


class FourierPositionEncoding(nn.Module):
    def __init__(self, output_dim: int, hidden_dim: int = 256, num_frequencies: int = 10):
        super().__init__()
        self.register_buffer("basis", torch.randn(3, num_frequencies))
        self.mlp = nn.Sequential(
            nn.Linear(2 * num_frequencies, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, coordinates: torch.Tensor) -> torch.Tensor:
        projected = coordinates @ self.basis
        encoded = torch.cat(
            [torch.sin(2 * math.pi * projected), torch.cos(2 * math.pi * projected)],
            dim=-1,
        )
        return self.mlp(encoded)


def farthest_point_sampling(
    points: torch.Tensor,
    count: int,
    scene_center: torch.Tensor,
) -> torch.Tensor:
    """Deterministic FPS, initialized by the point farthest from scene center."""
    batch_size, num_points, _ = points.shape
    count = min(int(count), num_points)
    selected = torch.zeros(batch_size, count, dtype=torch.long, device=points.device)
    min_distance = torch.full(
        (batch_size, num_points),
        torch.finfo(points.dtype).max,
        dtype=points.dtype,
        device=points.device,
    )
    farthest = ((points - scene_center[:, None]) ** 2).sum(dim=-1).argmax(dim=1)
    batch_indices = torch.arange(batch_size, device=points.device)

    for index in range(count):
        selected[:, index] = farthest
        centroid = points[batch_indices, farthest][:, None]
        distance = ((points - centroid) ** 2).sum(dim=-1)
        min_distance = torch.minimum(min_distance, distance)
        farthest = min_distance.argmax(dim=1)
    return selected


class GeometryAwareCrossAttention(nn.Module):
    """Cross-attention with a learned soft bias from 3D anchor-point distance."""

    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        if dim % num_heads:
            raise ValueError("dim must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.query = nn.Linear(dim, dim, bias=False)
        self.key = nn.Linear(dim, dim, bias=False)
        self.value = nn.Linear(dim, dim, bias=False)
        self.distance_bias = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, num_heads),
        )
        self.attention_dropout = nn.Dropout(dropout)
        self.output = nn.Linear(dim, dim)

    def forward(
        self,
        query: torch.Tensor,
        memory: torch.Tensor,
        squared_distances: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, num_queries, dim = query.shape
        num_memory = memory.shape[1]
        q = self.query(query).reshape(batch_size, num_queries, self.num_heads, self.head_dim)
        k = self.key(memory).reshape(batch_size, num_memory, self.num_heads, self.head_dim)
        v = self.value(memory).reshape(batch_size, num_memory, self.num_heads, self.head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        logits = (q @ k.transpose(-2, -1)) * self.scale
        geometry_bias = self.distance_bias(
            torch.log(squared_distances[..., None] + 1e-6)
        ).permute(0, 3, 1, 2)
        attention = self.attention_dropout((logits + geometry_bias).softmax(dim=-1))
        output = (attention @ v).transpose(1, 2).reshape(batch_size, num_queries, dim)
        return self.output(output)


class GeoLMCBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.query_norm = nn.LayerNorm(dim)
        self.attention = GeometryAwareCrossAttention(dim, num_heads, dropout)
        self.ffn_norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Linear(4 * dim, dim),
        )

    def forward(
        self,
        latent: torch.Tensor,
        memory: torch.Tensor,
        squared_distances: torch.Tensor,
    ) -> torch.Tensor:
        latent = latent + self.attention(self.query_norm(latent), memory, squared_distances)
        return latent + self.ffn(self.ffn_norm(latent))


class GlobalGeoLMC(nn.Module):
    """Compress a pooled scene memory into K anchored scene tokens."""

    def __init__(
        self,
        memory_feature_dim: int,
        token_dim: int = 1024,
        num_tokens: int = 64,
        num_attention_layers: int = 4,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_tokens = int(num_tokens)
        self.position_encoder = FourierPositionEncoding(token_dim)
        self.memory_projection = nn.Sequential(
            nn.Linear(memory_feature_dim, token_dim),
            nn.LayerNorm(token_dim),
        )
        self.blocks = nn.ModuleList(
            GeoLMCBlock(token_dim, num_heads, dropout)
            for _ in range(num_attention_layers)
        )

    def forward(self, memory: SceneMemory) -> AnchoredSceneTokens:
        memory = memory.batched()
        anchor_indices = farthest_point_sampling(
            memory.points,
            self.num_tokens,
            memory.scene_center,
        )
        anchors = torch.gather(
            memory.points,
            dim=1,
            index=anchor_indices[..., None].expand(-1, -1, 3),
        )
        latent = self.position_encoder(anchors - memory.scene_center[:, None])
        projected_memory = self.memory_projection(memory.features)
        squared_distances = torch.cdist(anchors, memory.points, p=2).square()

        for block in self.blocks:
            latent = block(latent, projected_memory, squared_distances)

        return AnchoredSceneTokens(
            features=latent,
            anchors=anchors,
            scene_center=memory.scene_center,
        ).validate()
