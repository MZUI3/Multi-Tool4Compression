"""CvT — "CvT: Introducing Convolutions to Vision Transformers" (Wu et al., 2021).

Convolutional token embedding between stages and depthwise-conv projections for q/k/v.
The class token is only added in the last stage, as in the paper.
"""
import torch
import torch.nn as nn

from ..layers import DropPath, Mlp, drop_path_rates, init_transformer_weights, is_small_input
from ..registry import register_model


class ConvProjection(nn.Sequential):
    def __init__(self, dim, stride):
        super().__init__(nn.Conv2d(dim, dim, 3, stride, 1, groups=dim, bias=False), nn.BatchNorm2d(dim))


class ConvAttention(nn.Module):
    def __init__(self, dim, num_heads, kv_stride=2, has_cls_token=False):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.has_cls_token = has_cls_token
        self.conv_q = ConvProjection(dim, 1)
        self.conv_k = ConvProjection(dim, kv_stride)
        self.conv_v = ConvProjection(dim, kv_stride)
        self.proj_q = nn.Linear(dim, dim)
        self.proj_k = nn.Linear(dim, dim)
        self.proj_v = nn.Linear(dim, dim)
        self.proj = nn.Linear(dim, dim)

    def _split_heads(self, x):
        batch, tokens, channels = x.shape
        return x.reshape(batch, tokens, self.num_heads, channels // self.num_heads).transpose(1, 2)

    def forward(self, x, height, width):
        batch, tokens, channels = x.shape
        cls_token = None
        if self.has_cls_token:
            cls_token, x = x[:, :1], x[:, 1:]
        fmap = x.transpose(1, 2).reshape(batch, channels, height, width)

        def project(conv):
            out = conv(fmap).flatten(2).transpose(1, 2)
            return torch.cat([cls_token, out], 1) if cls_token is not None else out

        q = self._split_heads(self.proj_q(project(self.conv_q)))
        k = self._split_heads(self.proj_k(project(self.conv_k)))
        v = self._split_heads(self.proj_v(project(self.conv_v)))
        attn = ((q * self.scale) @ k.transpose(-2, -1)).softmax(-1)
        x = (attn @ v).transpose(1, 2).reshape(batch, tokens, channels)
        return self.proj(x)


class CvtBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio, drop_path, has_cls_token):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = ConvAttention(dim, num_heads, has_cls_token=has_cls_token)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, int(dim * mlp_ratio))
        self.drop_path = DropPath(drop_path)

    def forward(self, x, height, width):
        x = x + self.drop_path(self.attn(self.norm1(x), height, width))
        return x + self.drop_path(self.mlp(self.norm2(x)))


class CvtStage(nn.Module):
    def __init__(self, in_channels, dim, depth, num_heads, patch_size, stride, mlp_ratio, dpr, has_cls_token):
        super().__init__()
        self.embed = nn.Conv2d(in_channels, dim, patch_size, stride, patch_size // 2)
        self.embed_norm = nn.LayerNorm(dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim)) if has_cls_token else None
        if self.cls_token is not None:
            nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.blocks = nn.ModuleList([CvtBlock(dim, num_heads, mlp_ratio, dpr[i], has_cls_token) for i in range(depth)])

    def forward(self, x):
        x = self.embed(x)
        batch, channels, height, width = x.shape
        x = self.embed_norm(x.flatten(2).transpose(1, 2))
        if self.cls_token is not None:
            x = torch.cat([self.cls_token.expand(batch, -1, -1), x], 1)
        for block in self.blocks:
            x = block(x, height, width)
        if self.cls_token is not None:
            return x
        return x.transpose(1, 2).reshape(batch, channels, height, width)


class CvT(nn.Module):
    def __init__(self, depths, dims=(64, 192, 384), heads=(1, 3, 6), num_classes=1000, img_size=224, drop_path_rate=0.1):
        super().__init__()
        dpr = drop_path_rates(drop_path_rate, sum(depths))
        first_stride = 2 if is_small_input(img_size) else 4
        stages = []
        cursor = 0
        for i, depth in enumerate(depths):
            stages.append(
                CvtStage(
                    in_channels=3 if i == 0 else dims[i - 1],
                    dim=dims[i],
                    depth=depth,
                    num_heads=heads[i],
                    patch_size=7 if i == 0 else 3,
                    stride=first_stride if i == 0 else 2,
                    mlp_ratio=4.0,
                    dpr=dpr[cursor : cursor + depth],
                    has_cls_token=i == len(depths) - 1,
                )
            )
            cursor += depth
        self.stages = nn.Sequential(*stages)
        self.norm = nn.LayerNorm(dims[-1])
        self.head = nn.Linear(dims[-1], num_classes)
        init_transformer_weights(self)

    def forward(self, x):
        x = self.stages(x)
        return self.head(self.norm(x[:, 0]))


_spec = dict(category="transformer", family="CvT", paper="Wu et al., 2021")


@register_model(**_spec)
def cvt_13(num_classes=1000, img_size=224):
    return CvT((1, 2, 10), num_classes=num_classes, img_size=img_size)


@register_model(**_spec)
def cvt_21(num_classes=1000, img_size=224):
    return CvT((1, 4, 16), num_classes=num_classes, img_size=img_size)
