"""MobileViTv2 — "Separable Self-attention for Mobile Vision Transformers" (Mehta & Rastegari, 2022)."""
import torch.nn as nn
import torch.nn.functional as F

from ..layers import ConvBNAct, DropPath, init_cnn_weights, is_small_input, make_divisible
from ..registry import register_model


class InvertedResidual(nn.Module):
    def __init__(self, in_channels, out_channels, stride, expand_ratio):
        super().__init__()
        hidden = make_divisible(in_channels * expand_ratio)
        self.use_res = stride == 1 and in_channels == out_channels
        self.block = nn.Sequential(
            ConvBNAct(in_channels, hidden, 1),
            ConvBNAct(hidden, hidden, 3, stride, groups=hidden),
            ConvBNAct(hidden, out_channels, 1, act_layer=None),
        )

    def forward(self, x):
        return x + self.block(x) if self.use_res else self.block(x)


class LinearSelfAttention(nn.Module):
    """Separable self-attention: O(k) context vector instead of O(k^2) attention matrix."""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.qkv_proj = nn.Conv2d(dim, 1 + 2 * dim, 1)
        self.out_proj = nn.Conv2d(dim, dim, 1)

    def forward(self, x):  # x: B, C, P (pixels per patch), N (patches)
        query, key, value = self.qkv_proj(x).split([1, self.dim, self.dim], 1)
        context_scores = query.softmax(-1)
        context_vector = (key * context_scores).sum(-1, keepdim=True)
        return self.out_proj(F.relu(value) * context_vector)


class LinearTransformerBlock(nn.Module):
    def __init__(self, dim, ffn_multiplier=2.0, drop_path=0.0):
        super().__init__()
        hidden = make_divisible(dim * ffn_multiplier, 16)
        self.norm1 = nn.GroupNorm(1, dim)
        self.attn = LinearSelfAttention(dim)
        self.norm2 = nn.GroupNorm(1, dim)
        self.ffn = nn.Sequential(nn.Conv2d(dim, hidden, 1), nn.SiLU(), nn.Conv2d(hidden, dim, 1))
        self.drop_path = DropPath(drop_path)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        return x + self.drop_path(self.ffn(self.norm2(x)))


class MobileViTV2Block(nn.Module):
    def __init__(self, in_channels, attn_dim, depth, patch_size=2):
        super().__init__()
        self.patch_size = patch_size
        self.local_rep = nn.Sequential(
            ConvBNAct(in_channels, in_channels, 3, groups=in_channels),
            nn.Conv2d(in_channels, attn_dim, 1, bias=False),
        )
        self.global_rep = nn.Sequential(*[LinearTransformerBlock(attn_dim) for _ in range(depth)], nn.GroupNorm(1, attn_dim))
        self.conv_proj = ConvBNAct(attn_dim, in_channels, 1, act_layer=None)

    def forward(self, x):
        height, width = x.shape[2:]
        p = self.patch_size
        new_h, new_w = -(-height // p) * p, -(-width // p) * p
        if (new_h, new_w) != (height, width):
            x = F.interpolate(x, size=(new_h, new_w), mode="bilinear", align_corners=False)

        x = self.local_rep(x)
        batch, channels = x.shape[:2]
        # unfold: B, C, H, W -> B, C, P, N
        patches = F.unfold(x, kernel_size=p, stride=p).reshape(batch, channels, p * p, -1)
        patches = self.global_rep(patches)
        # fold back: B, C, P, N -> B, C, H, W
        x = F.fold(patches.reshape(batch, channels * p * p, -1), output_size=(new_h, new_w), kernel_size=p, stride=p)
        x = self.conv_proj(x)
        if (new_h, new_w) != (height, width):
            x = F.interpolate(x, size=(height, width), mode="bilinear", align_corners=False)
        return x


class MobileViTV2(nn.Module):
    def __init__(self, width_multiplier=1.0, num_classes=1000, img_size=224):
        super().__init__()

        def ch(c):
            return make_divisible(c * width_multiplier)

        self.stem = ConvBNAct(3, ch(32), 3, 1 if is_small_input(img_size) else 2)
        self.layer1 = InvertedResidual(ch(32), ch(64), 1, 2)
        self.layer2 = nn.Sequential(InvertedResidual(ch(64), ch(128), 2, 2), InvertedResidual(ch(128), ch(128), 1, 2))
        self.layer3 = nn.Sequential(InvertedResidual(ch(128), ch(256), 2, 2), MobileViTV2Block(ch(256), ch(128), 2))
        self.layer4 = nn.Sequential(InvertedResidual(ch(256), ch(384), 2, 2), MobileViTV2Block(ch(384), ch(192), 4))
        self.layer5 = nn.Sequential(InvertedResidual(ch(384), ch(512), 2, 2), MobileViTV2Block(ch(512), ch(256), 3))
        self.head = nn.Linear(ch(512), num_classes)
        init_cnn_weights(self)

    def forward(self, x):
        x = self.layer2(self.layer1(self.stem(x)))
        x = self.layer5(self.layer4(self.layer3(x)))
        return self.head(x.mean((2, 3)))


_spec = dict(category="transformer", family="MobileViTv2", paper="Mehta & Rastegari, 2022")


@register_model(**_spec)
def mobilevitv2_050(num_classes=1000, img_size=224):
    return MobileViTV2(0.5, num_classes, img_size)


@register_model(**_spec)
def mobilevitv2_100(num_classes=1000, img_size=224):
    return MobileViTV2(1.0, num_classes, img_size)
