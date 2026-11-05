"""BoTNet — "Bottleneck Transformers for Visual Recognition" (Srinivas et al., 2021).

ResNet-50 whose last stage replaces the 3x3 conv with multi-head self-attention
using 2D relative-position (content-position) logits.
"""
import torch
import torch.nn as nn

from ..layers import ConvBNAct
from ..registry import register_model
from ..resnet_base import Bottleneck, ResNetBase


class BoTMHSA(nn.Module):
    def __init__(self, dim, fmap_size, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.fmap_size = fmap_size
        self.qkv = nn.Conv2d(dim, dim * 3, 1, bias=False)
        self.rel_h = nn.Parameter(torch.randn(num_heads, self.head_dim, fmap_size, 1) * self.scale)
        self.rel_w = nn.Parameter(torch.randn(num_heads, self.head_dim, 1, fmap_size) * self.scale)

    def forward(self, x):
        batch, channels, height, width = x.shape
        if height != self.fmap_size or width != self.fmap_size:
            raise ValueError(
                f"BoTNet attention was built for {self.fmap_size}x{self.fmap_size} maps, got {height}x{width}. "
                "Build the model with the same --resize you feed it."
            )
        q, k, v = self.qkv(x).reshape(batch, 3, self.num_heads, self.head_dim, height * width).unbind(1)
        position = (self.rel_h + self.rel_w).reshape(1, self.num_heads, self.head_dim, height * width)
        content_content = q.transpose(-2, -1) @ k  # B, h, N, N
        content_position = q.transpose(-2, -1) @ position
        attn = ((content_content + content_position) * self.scale).softmax(-1)
        out = v @ attn.transpose(-2, -1)  # B, h, d, N
        return out.reshape(batch, channels, height, width)


class BoTBlock(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, fmap_size=14, num_heads=4, **_):
        super().__init__()
        out_planes = planes * self.expansion
        self.conv1 = ConvBNAct(inplanes, planes, 1, act_layer=nn.ReLU)
        self.mhsa = BoTMHSA(planes, fmap_size, num_heads)
        self.pool = nn.AvgPool2d(2, 2) if stride == 2 else nn.Identity()
        self.post = nn.Sequential(nn.BatchNorm2d(planes), nn.ReLU(inplace=True))
        self.conv3 = ConvBNAct(planes, out_planes, 1, act_layer=None)
        self.downsample = downsample
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        shortcut = x if self.downsample is None else self.downsample(x)
        out = self.post(self.pool(self.mhsa(self.conv1(x))))
        return self.act(self.conv3(out) + shortcut)


_spec = dict(category="transformer", family="BoTNet", paper="Srinivas et al., 2021", input_size_divisor=32)


@register_model(**_spec)
def botnet50(num_classes=1000, img_size=224):
    return ResNetBase(Bottleneck, [3, 4, 6, 3], num_classes, img_size, stage_blocks=[Bottleneck, Bottleneck, Bottleneck, BoTBlock])
