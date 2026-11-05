"""PVTv2 — "PVT v2: Improved Baselines with Pyramid Vision Transformer" (Wang et al., 2022)."""
import torch.nn as nn

from ..layers import DropPath, drop_path_rates, init_transformer_weights, is_small_input
from ..registry import register_model


class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_channels, embed_dim, patch_size, stride):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim, patch_size, stride, patch_size // 2)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.proj(x)
        height, width = x.shape[2:]
        return self.norm(x.flatten(2).transpose(1, 2)), height, width


class SpatialReductionAttention(nn.Module):
    """Keys/values are computed on a spatially reduced map, so cost drops by sr_ratio^2."""

    def __init__(self, dim, num_heads, sr_ratio):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, dim * 2)
        self.proj = nn.Linear(dim, dim)
        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, sr_ratio, sr_ratio)
            self.norm = nn.LayerNorm(dim)

    def forward(self, x, height, width):
        batch, tokens, channels = x.shape
        head_dim = channels // self.num_heads
        q = self.q(x).reshape(batch, tokens, self.num_heads, head_dim).transpose(1, 2)
        if self.sr_ratio > 1:
            x = x.transpose(1, 2).reshape(batch, channels, height, width)
            x = self.norm(self.sr(x).flatten(2).transpose(1, 2))
        kv = self.kv(x).reshape(batch, -1, 2, self.num_heads, head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv.unbind(0)
        attn = ((q * self.scale) @ k.transpose(-2, -1)).softmax(-1)
        x = (attn @ v).transpose(1, 2).reshape(batch, tokens, channels)
        return self.proj(x)


class ConvMlp(nn.Module):
    """MLP with a 3x3 depthwise conv between the two linears (PVTv2 positional encoding)."""

    def __init__(self, dim, hidden):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.dwconv = nn.Conv2d(hidden, hidden, 3, 1, 1, groups=hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x, height, width):
        x = self.fc1(x)
        batch, tokens, channels = x.shape
        x = self.dwconv(x.transpose(1, 2).reshape(batch, channels, height, width)).flatten(2).transpose(1, 2)
        return self.fc2(self.act(x))


class PvtBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio, sr_ratio, drop_path):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = SpatialReductionAttention(dim, num_heads, sr_ratio)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = ConvMlp(dim, int(dim * mlp_ratio))
        self.drop_path = DropPath(drop_path)

    def forward(self, x, height, width):
        x = x + self.drop_path(self.attn(self.norm1(x), height, width))
        return x + self.drop_path(self.mlp(self.norm2(x), height, width))


class PyramidVisionTransformerV2(nn.Module):
    def __init__(self, embed_dims, depths, num_classes=1000, img_size=224, num_heads=(1, 2, 5, 8),
                 mlp_ratios=(8, 8, 4, 4), sr_ratios=(8, 4, 2, 1), drop_path_rate=0.1):
        super().__init__()
        dpr = drop_path_rates(drop_path_rate, sum(depths))
        first_stride = 2 if is_small_input(img_size) else 4
        self.patch_embeds = nn.ModuleList()
        self.stages = nn.ModuleList()
        self.norms = nn.ModuleList()
        cursor = 0
        for i, depth in enumerate(depths):
            in_channels = 3 if i == 0 else embed_dims[i - 1]
            patch, stride = (7, first_stride) if i == 0 else (3, 2)
            self.patch_embeds.append(OverlapPatchEmbed(in_channels, embed_dims[i], patch, stride))
            self.stages.append(
                nn.ModuleList(
                    [PvtBlock(embed_dims[i], num_heads[i], mlp_ratios[i], sr_ratios[i], dpr[cursor + j]) for j in range(depth)]
                )
            )
            self.norms.append(nn.LayerNorm(embed_dims[i]))
            cursor += depth
        self.head = nn.Linear(embed_dims[-1], num_classes)
        init_transformer_weights(self)

    def forward(self, x):
        batch = x.shape[0]
        for i, (embed, blocks, norm) in enumerate(zip(self.patch_embeds, self.stages, self.norms)):
            x, height, width = embed(x)
            for block in blocks:
                x = block(x, height, width)
            x = norm(x)
            if i < len(self.stages) - 1:
                x = x.transpose(1, 2).reshape(batch, -1, height, width)
        return self.head(x.mean(1))


_spec = dict(category="transformer", family="PVTv2", paper="Wang et al., 2022")


@register_model(**_spec)
def pvt_v2_b0(num_classes=1000, img_size=224):
    return PyramidVisionTransformerV2((32, 64, 160, 256), (2, 2, 2, 2), num_classes, img_size)


@register_model(**_spec)
def pvt_v2_b1(num_classes=1000, img_size=224):
    return PyramidVisionTransformerV2((64, 128, 320, 512), (2, 2, 2, 2), num_classes, img_size)


@register_model(**_spec)
def pvt_v2_b2(num_classes=1000, img_size=224):
    return PyramidVisionTransformerV2((64, 128, 320, 512), (3, 4, 6, 3), num_classes, img_size)
