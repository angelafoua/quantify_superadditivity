"""Vision Transformer (ViT-Tiny) for federated learning experiments.

A minimal ViT implementation with configurable patch size, embedding
dimension, depth, and number of heads. Uses a learnable [CLS] token
and positional embeddings that adapt to any input resolution.

References
----------
* Dosovitskiy et al., "An Image is Worth 16x16 Words", ICLR 2021.
"""

from __future__ import annotations

import logging
import math
from typing import List

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class PatchEmbedding(nn.Module):
    """Convert image into patch embeddings via a strided convolution."""

    def __init__(
        self,
        in_channels: int = 3,
        patch_size: int = 4,
        embed_dim: int = 192,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size, stride=patch_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, C, H, W) -> (B, embed_dim, H/P, W/P) -> (B, N, embed_dim)
        out = self.proj(x)
        out = out.flatten(2).transpose(1, 2)
        return out


class TransformerBlock(nn.Module):
    """Standard pre-norm transformer block with MHA + FFN."""

    def __init__(
        self,
        embed_dim: int = 192,
        num_heads: int = 3,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True,
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        mlp_hidden = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normed = self.norm1(x)
        attn_out, _ = self.attn(normed, normed, normed)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


class VisionTransformer(nn.Module):
    """Vision Transformer (ViT) with configurable architecture.

    Parameters
    ----------
    image_size:
        Expected input spatial resolution (square images).
    patch_size:
        Size of each patch (image_size must be divisible by patch_size).
    in_channels:
        Number of input image channels.
    num_classes:
        Number of output classes.
    embed_dim:
        Transformer embedding dimension.
    depth:
        Number of transformer blocks.
    num_heads:
        Number of attention heads.
    mlp_ratio:
        FFN hidden-dim multiplier.
    dropout:
        Dropout rate.

    Attributes
    ----------
    feature_dim : int
        Dimensionality of the penultimate representation (embed_dim).
    """

    def __init__(
        self,
        image_size: int = 32,
        patch_size: int = 4,
        in_channels: int = 3,
        num_classes: int = 100,
        embed_dim: int = 192,
        depth: int = 12,
        num_heads: int = 3,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        assert image_size % patch_size == 0, (
            f"image_size ({image_size}) must be divisible by patch_size ({patch_size})"
        )
        self.feature_dim: int = embed_dim
        self.image_size = image_size
        self.patch_size = patch_size
        n_patches = (image_size // patch_size) ** 2

        self.patch_embed = PatchEmbedding(in_channels, patch_size, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.fc = nn.Linear(embed_dim, num_classes)

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = self.pos_drop(x + self.pos_embed)

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        cls_out = x[:, 0]
        return self.fc(cls_out)

    @torch.no_grad()
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract [CLS] token features (before the classifier)."""
        B = x.shape[0]
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = self.pos_drop(x + self.pos_embed)

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        return x[:, 0]

    @staticmethod
    def get_layer_names() -> List[str]:
        """Return the canonical ordered list of extractable names.

        For ViT, we extract at evenly spaced transformer blocks and the
        final classifier head.
        """
        return ["blocks.2", "blocks.5", "blocks.8", "blocks.11", "fc"]


def build_vit_tiny(
    num_classes: int = 100,
    in_channels: int = 3,
    image_size: int = 32,
    patch_size: int = 4,
) -> VisionTransformer:
    """Construct a ViT-Tiny (dim=192, depth=12, heads=3)."""
    logger.info(
        "Building ViT-Tiny with %d classes, %d channels, %dx%d input, patch=%d",
        num_classes, in_channels, image_size, image_size, patch_size,
    )
    return VisionTransformer(
        image_size=image_size,
        patch_size=patch_size,
        in_channels=in_channels,
        num_classes=num_classes,
        embed_dim=192,
        depth=12,
        num_heads=3,
        mlp_ratio=4.0,
        dropout=0.0,
    )
