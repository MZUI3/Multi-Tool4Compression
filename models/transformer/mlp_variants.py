"""All-MLP architectures besides MLP-Mixer.

ResMLP — "ResMLP: Feedforward networks for image classification with data-efficient training"
          (Touvron et al., 2021): affine normalisation + a linear cross-patch layer + LayerScale.
gMLP   — "Pay Attention to MLPs" (Liu et al., 2021): spatial gating unit instead of self-attention.
"""
import torch
import torch.nn as nn

from ..layers import DropPath, Mlp, drop_path_rates, init_transformer_weights
from ..registry import register_model
from .common import PatchEmbed, auto_patch_size


class Affine(nn.Module):
    """Per-channel affine transform (ResMLP replaces LayerNorm with this)."""

    def __init__(self, dim):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        return x * self.alpha + self.beta


class ResMLPBlock(nn.Module):
    def __init__(self, dim, num_tokens, mlp_ratio=4.0, init_values=1e-4, drop_path=0.0):
        super().__init__()
        self.norm1 = Affine(dim)
        self.cross_patch = nn.Linear(num_tokens, num_tokens)
        self.ls1 = nn.Parameter(init_values * torch.ones(dim))
        self.norm2 = Affine(dim)
        self.mlp = Mlp(dim, int(dim * mlp_ratio))
        self.ls2 = nn.Parameter(init_values * torch.ones(dim))
        self.drop_path = DropPath(drop_path)

    def forward(self, x):
        x = x + self.drop_path(self.ls1 * self.cross_patch(self.norm1(x).transpose(1, 2)).transpose(1, 2))
        return x + self.drop_path(self.ls2 * self.mlp(self.norm2(x)))


class ResMLP(nn.Module):
    def __init__(self, img_size=224, num_classes=1000, dim=384, depth=12, mlp_ratio=4.0, drop_path_rate=0.05):
        super().__init__()
        self.stem = PatchEmbed(img_size, auto_patch_size(img_size), 3, dim)
        dpr = drop_path_rates(drop_path_rate, depth)
        self.blocks = nn.Sequential(*[ResMLPBlock(dim, self.stem.num_patches, mlp_ratio, drop_path=dpr[i]) for i in range(depth)])
        self.norm = Affine(dim)
        self.head = nn.Linear(dim, num_classes)
        init_transformer_weights(self)

    def forward(self, x):
        x = self.norm(self.blocks(self.stem(x)))
        return self.head(x.mean(1))


class SpatialGatingUnit(nn.Module):
    def __init__(self, dim, num_tokens):
        super().__init__()
        self.norm = nn.LayerNorm(dim // 2)
        self.proj = nn.Linear(num_tokens, num_tokens)
        nn.init.normal_(self.proj.weight, std=1e-6)  # start close to identity gating, as in the paper
        nn.init.ones_(self.proj.bias)

    def forward(self, x):
        u, v = x.chunk(2, dim=-1)
        v = self.proj(self.norm(v).transpose(1, 2)).transpose(1, 2)
        return u * v


class GMLPBlock(nn.Module):
    def __init__(self, dim, num_tokens, mlp_ratio=6.0, drop_path=0.0):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.sgu = SpatialGatingUnit(hidden, num_tokens)
        self.fc2 = nn.Linear(hidden // 2, dim)
        self.drop_path = DropPath(drop_path)

    def forward(self, x):
        y = self.fc2(self.sgu(self.act(self.fc1(self.norm(x)))))
        return x + self.drop_path(y)


class GMLP(nn.Module):
    def __init__(self, img_size=224, num_classes=1000, dim=128, depth=30, mlp_ratio=6.0, drop_path_rate=0.05):
        super().__init__()
        self.stem = PatchEmbed(img_size, auto_patch_size(img_size), 3, dim)
        dpr = drop_path_rates(drop_path_rate, depth)
        self.blocks = nn.Sequential(*[GMLPBlock(dim, self.stem.num_patches, mlp_ratio, dpr[i]) for i in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)
        init_transformer_weights(self)
        for m in self.modules():
            if isinstance(m, SpatialGatingUnit):
                nn.init.normal_(m.proj.weight, std=1e-6)
                nn.init.ones_(m.proj.bias)

    def forward(self, x):
        x = self.norm(self.blocks(self.stem(x)))
        return self.head(x.mean(1))


_resmlp = dict(category="transformer", family="ResMLP", paper="Touvron et al., 2021", input_size_divisor=4)
_gmlp = dict(category="transformer", family="gMLP", paper="Liu et al., 2021", input_size_divisor=4)


@register_model(**_resmlp)
def resmlp_12(num_classes=1000, img_size=224):
    return ResMLP(img_size, num_classes, dim=384, depth=12)


@register_model(**_resmlp)
def resmlp_24(num_classes=1000, img_size=224):
    return ResMLP(img_size, num_classes, dim=384, depth=24)


@register_model(**_gmlp)
def gmlp_ti(num_classes=1000, img_size=224):
    return GMLP(img_size, num_classes, dim=128, depth=30)


@register_model(**_gmlp)
def gmlp_s(num_classes=1000, img_size=224):
    return GMLP(img_size, num_classes, dim=256, depth=30)
