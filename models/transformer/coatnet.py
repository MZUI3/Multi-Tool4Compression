"""CoAtNet — "CoAtNet: Marrying Convolution and Attention for All Data Sizes" (Dai et al., 2021).

C-C-T-T layout: conv stem, two MBConv stages, two relative-attention transformer stages.
"""
import math

import torch.nn as nn

from ..layers import ConvBNAct, DropPath, SqueezeExcite, drop_path_rates, init_transformer_weights, is_small_input
from ..registry import register_model
from .common import RelativePositionBias2D


class MBConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, expand_ratio=4, drop_path=0.0):
        super().__init__()
        hidden = in_channels * expand_ratio
        self.shortcut = nn.Identity()
        if stride > 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.MaxPool2d(3, stride, 1) if stride > 1 else nn.Identity(),
                nn.Conv2d(in_channels, out_channels, 1),
            )
        self.conv = nn.Sequential(
            nn.BatchNorm2d(in_channels),  # pre-norm
            ConvBNAct(in_channels, hidden, 1, act_layer=nn.GELU),
            ConvBNAct(hidden, hidden, 3, stride, groups=hidden, act_layer=nn.GELU),
            SqueezeExcite(hidden, rd_channels=max(1, in_channels // 4), act_layer=nn.GELU),
            nn.Conv2d(hidden, out_channels, 1),
        )
        self.drop_path = DropPath(drop_path)

    def forward(self, x):
        return self.shortcut(x) + self.drop_path(self.conv(x))


class RelativeAttention(nn.Module):
    def __init__(self, dim, out_dim, fmap_size, head_dim=32):
        super().__init__()
        self.num_heads = max(1, out_dim // head_dim)
        self.head_dim = out_dim // self.num_heads
        self.scale = self.head_dim**-0.5
        self.qkv = nn.Linear(dim, out_dim * 3)
        self.proj = nn.Linear(out_dim, out_dim)
        self.rel_pos = RelativePositionBias2D(self.num_heads, fmap_size, fmap_size)

    def forward(self, x):
        batch, tokens, _ = x.shape
        qkv = self.qkv(x).reshape(batch, tokens, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = ((q * self.scale) @ k.transpose(-2, -1) + self.rel_pos()).softmax(-1)
        x = (attn @ v).transpose(1, 2).reshape(batch, tokens, -1)
        return self.proj(x)


class CoAtTransformer(nn.Module):
    """Transformer block on NCHW maps; the first block of a stage downsamples with max-pool."""

    def __init__(self, in_channels, out_channels, out_size, downsample=False, mlp_ratio=4, drop_path=0.0):
        super().__init__()
        self.pool = nn.MaxPool2d(3, 2, 1) if downsample else nn.Identity()
        self.shortcut_proj = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        self.norm1 = nn.LayerNorm(in_channels)
        self.attn = RelativeAttention(in_channels, out_channels, out_size)
        self.norm2 = nn.LayerNorm(out_channels)
        self.mlp = nn.Sequential(
            nn.Linear(out_channels, out_channels * mlp_ratio), nn.GELU(), nn.Linear(out_channels * mlp_ratio, out_channels)
        )
        self.drop_path = DropPath(drop_path)

    def forward(self, x):
        x = self.pool(x)
        batch, _, height, width = x.shape
        shortcut = self.shortcut_proj(x).flatten(2).transpose(1, 2)
        tokens = x.flatten(2).transpose(1, 2)
        tokens = shortcut + self.drop_path(self.attn(self.norm1(tokens)))
        tokens = tokens + self.drop_path(self.mlp(self.norm2(tokens)))
        return tokens.transpose(1, 2).reshape(batch, -1, height, width)


class CoAtNet(nn.Module):
    def __init__(self, depths, dims, num_classes=1000, img_size=224, drop_path_rate=0.1):
        super().__init__()
        small = is_small_input(img_size)
        self.stem = nn.Sequential(
            ConvBNAct(3, dims[0], 3, 1 if small else 2, act_layer=nn.GELU),
            ConvBNAct(dims[0], dims[0], 3, 1, act_layer=nn.GELU),
        )
        size = img_size if small else math.ceil(img_size / 2)
        dpr = drop_path_rates(drop_path_rate, sum(depths))
        cursor = 0
        stages = []
        in_channels = dims[0]
        for i, depth in enumerate(depths):
            out_channels = dims[i + 1]
            size = math.ceil(size / 2)
            blocks = []
            for j in range(depth):
                first = j == 0
                if i < 2:
                    blocks.append(MBConv(in_channels, out_channels, 2 if first else 1, drop_path=dpr[cursor]))
                else:
                    blocks.append(CoAtTransformer(in_channels, out_channels, size, downsample=first, drop_path=dpr[cursor]))
                in_channels = out_channels
                cursor += 1
            stages.append(nn.Sequential(*blocks))
        self.stages = nn.Sequential(*stages)
        self.norm = nn.LayerNorm(in_channels)
        self.head = nn.Linear(in_channels, num_classes)
        init_transformer_weights(self)

    def forward(self, x):
        x = self.stages(self.stem(x)).mean((2, 3))
        return self.head(self.norm(x))


_spec = dict(category="transformer", family="CoAtNet", paper="Dai et al., 2021")


@register_model(**_spec)
def coatnet_0(num_classes=1000, img_size=224):
    return CoAtNet((2, 3, 5, 2), (64, 96, 192, 384, 768), num_classes, img_size)


@register_model(**_spec)
def coatnet_1(num_classes=1000, img_size=224):
    return CoAtNet((2, 6, 14, 2), (64, 96, 192, 384, 768), num_classes, img_size)
