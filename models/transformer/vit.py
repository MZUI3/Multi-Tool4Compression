"""ViT and DeiT.

- ViT  — "An Image is Worth 16x16 Words" (Dosovitskiy et al., 2021)
- DeiT — "Training data-efficient image transformers & distillation through attention" (Touvron et al., 2021)

Patch size is chosen from the input size (16 for ImageNet-sized, 4 for CIFAR-sized inputs).
"""
import torch
import torch.nn as nn

from ..layers import drop_path_rates, init_transformer_weights
from ..registry import register_model
from .common import PatchEmbed, TransformerBlock, auto_patch_size


class VisionTransformer(nn.Module):
    def __init__(
        self,
        img_size=224,
        patch_size=None,
        num_classes=1000,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        drop_rate=0.0,
        drop_path_rate=0.1,
        distilled=False,
    ):
        super().__init__()
        patch_size = patch_size or auto_patch_size(img_size)
        self.patch_embed = PatchEmbed(img_size, patch_size, 3, embed_dim)
        self.num_prefix_tokens = 2 if distilled else 1

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None
        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.num_patches + self.num_prefix_tokens, embed_dim))
        self.pos_drop = nn.Dropout(drop_rate)

        dpr = drop_path_rates(drop_path_rate, depth)
        self.blocks = nn.Sequential(
            *[TransformerBlock(embed_dim, num_heads, mlp_ratio, drop=drop_rate, drop_path=dpr[i]) for i in range(depth)]
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        self.head_dist = nn.Linear(embed_dim, num_classes) if distilled else None

        init_transformer_weights(self)
        for p in (self.pos_embed, self.cls_token, self.dist_token):
            if p is not None:
                nn.init.trunc_normal_(p, std=0.02)

    def forward_features(self, x):
        x = self.patch_embed(x)
        prefix = [self.cls_token.expand(x.shape[0], -1, -1)]
        if self.dist_token is not None:
            prefix.append(self.dist_token.expand(x.shape[0], -1, -1))
        x = torch.cat(prefix + [x], 1)
        x = self.pos_drop(x + self.pos_embed)
        return self.norm(self.blocks(x))

    def forward(self, x):
        x = self.forward_features(x)
        if self.head_dist is None:
            return self.head(x[:, 0])
        # Without a teacher both heads are trained on the labels; their average is the prediction.
        return (self.head(x[:, 0]) + self.head_dist(x[:, 1])) / 2


_vit = dict(category="transformer", family="ViT", paper="Dosovitskiy et al., 2021", input_size_divisor=4)
_deit = dict(category="transformer", family="DeiT", paper="Touvron et al., 2021", input_size_divisor=4)


@register_model(**_vit)
def vit_tiny(num_classes=1000, img_size=224):
    return VisionTransformer(img_size, num_classes=num_classes, embed_dim=192, depth=12, num_heads=3)


@register_model(**_vit)
def vit_small(num_classes=1000, img_size=224):
    return VisionTransformer(img_size, num_classes=num_classes, embed_dim=384, depth=12, num_heads=6)


@register_model(**_deit)
def deit_tiny(num_classes=1000, img_size=224):
    return VisionTransformer(img_size, num_classes=num_classes, embed_dim=192, depth=12, num_heads=3, distilled=True)


@register_model(**_deit)
def deit_small(num_classes=1000, img_size=224):
    return VisionTransformer(img_size, num_classes=num_classes, embed_dim=384, depth=12, num_heads=6, distilled=True)
