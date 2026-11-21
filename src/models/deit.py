"""DeiT-Small (Subject #6).

Datasets: CIFAR-10, ImageNet-1k (val).

Implemented from scratch in pure PyTorch following
"Training data-efficient image transformers & distillation through attention"
(Touvron et al., 2021).

DeiT shares the ViT encoder architecture but adds a learnable *distillation
token* alongside the class token. At inference the two classifier heads
(``head`` from the CLS token and ``head_dist`` from the distillation token) are
averaged. DeiT-Small uses embed_dim=384, depth=12, heads=6.

The transformer building blocks are reused from :mod:`src.models.vit`.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .vit import PatchEmbed, TransformerEncoderBlock

__all__ = ["DistilledVisionTransformer", "deit_small_patch16"]


class DistilledVisionTransformer(nn.Module):
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        num_classes: int = 1000,
        embed_dim: int = 384,
        depth: int = 12,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # +2 tokens: class token and distillation token
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 2, embed_dim))
        self.pos_drop = nn.Dropout(dropout)

        self.blocks = nn.ModuleList(
            [
                TransformerEncoderBlock(
                    embed_dim, num_heads, mlp_ratio, qkv_bias, dropout
                )
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)

        self.head = nn.Linear(embed_dim, num_classes)
        self.head_dist = nn.Linear(embed_dim, num_classes)

        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.dist_token, std=0.02)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward_features(self, x: torch.Tensor):
        B = x.shape[0]
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        dist_tokens = self.dist_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, dist_tokens, x), dim=1)
        x = self.pos_drop(x + self.pos_embed)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return x[:, 0], x[:, 1]  # (cls feature, distillation feature)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_cls, x_dist = self.forward_features(x)
        out_cls = self.head(x_cls)
        out_dist = self.head_dist(x_dist)
        if self.training:
            # both logits are returned during training (each gets its own loss)
            return out_cls, out_dist
        # at inference the two heads are averaged
        return (out_cls + out_dist) / 2


def deit_small_patch16(
    img_size: int = 224, num_classes: int = 1000, **kwargs
) -> DistilledVisionTransformer:
    """DeiT-Small/16: embed_dim=384, depth=12, heads=6."""
    return DistilledVisionTransformer(
        img_size=img_size,
        patch_size=16,
        num_classes=num_classes,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4.0,
        **kwargs,
    )
