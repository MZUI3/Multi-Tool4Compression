"""ResNets with plug-in channel / spatial attention.

- SENet  — "Squeeze-and-Excitation Networks" (Hu et al., 2018)
- CBAM   — "CBAM: Convolutional Block Attention Module" (Woo et al., 2018)
- ECA-Net — "ECA-Net: Efficient Channel Attention" (Wang et al., 2020)
"""
import math
from functools import partial

import torch
import torch.nn as nn

from ..layers import SqueezeExcite
from ..registry import register_model
from ..resnet_base import BasicBlock, Bottleneck, ResNetBase


class CBAM(nn.Module):
    def __init__(self, channels, reduction=16, kernel_size=7):
        super().__init__()
        hidden = max(channels // reduction, 1)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1, bias=False),
        )
        self.spatial = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False),
            nn.BatchNorm2d(1),
        )

    def forward(self, x):
        channel_attn = self.mlp(x.mean((2, 3), keepdim=True)) + self.mlp(x.amax((2, 3), keepdim=True))
        x = x * torch.sigmoid(channel_attn)
        spatial_attn = self.spatial(torch.cat([x.mean(1, keepdim=True), x.amax(1, keepdim=True)], 1))
        return x * torch.sigmoid(spatial_attn)


class ECA(nn.Module):
    def __init__(self, channels, gamma=2, beta=1):
        super().__init__()
        kernel_size = int(abs(math.log2(channels) / gamma + beta / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1
        self.conv = nn.Conv1d(1, 1, kernel_size, padding=kernel_size // 2, bias=False)

    def forward(self, x):
        y = x.mean((2, 3)).unsqueeze(1)  # B, 1, C
        y = self.conv(y).transpose(1, 2).unsqueeze(-1)  # B, C, 1, 1
        return x * torch.sigmoid(y)


ATTENTION_LAYERS = {
    "se": partial(SqueezeExcite, rd_ratio=1 / 16),
    "cbam": CBAM,
    "eca": ECA,
}


def _attn_resnet(attn, block, layers, num_classes, img_size):
    return ResNetBase(block, layers, num_classes, img_size, block_kwargs=dict(attn_layer=ATTENTION_LAYERS[attn]))


_se = dict(category="lightweight", family="SENet", paper="Hu et al., 2018")
_cbam = dict(category="lightweight", family="CBAM", paper="Woo et al., 2018")
_eca = dict(category="lightweight", family="ECA-Net", paper="Wang et al., 2020")


@register_model(**_se, aliases=("senet",))
def se_resnet18(num_classes=1000, img_size=224):
    return _attn_resnet("se", BasicBlock, [2, 2, 2, 2], num_classes, img_size)


@register_model(**_se)
def se_resnet50(num_classes=1000, img_size=224):
    return _attn_resnet("se", Bottleneck, [3, 4, 6, 3], num_classes, img_size)


@register_model(**_cbam)
def cbam_resnet18(num_classes=1000, img_size=224):
    return _attn_resnet("cbam", BasicBlock, [2, 2, 2, 2], num_classes, img_size)


@register_model(**_cbam)
def cbam_resnet50(num_classes=1000, img_size=224):
    return _attn_resnet("cbam", Bottleneck, [3, 4, 6, 3], num_classes, img_size)


@register_model(**_eca)
def eca_resnet18(num_classes=1000, img_size=224):
    return _attn_resnet("eca", BasicBlock, [2, 2, 2, 2], num_classes, img_size)


@register_model(**_eca)
def eca_resnet50(num_classes=1000, img_size=224):
    return _attn_resnet("eca", Bottleneck, [3, 4, 6, 3], num_classes, img_size)
