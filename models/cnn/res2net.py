"""Res2Net — "Res2Net: A New Multi-scale Backbone Architecture" (Gao et al., 2019)."""
import math

import torch
import torch.nn as nn

from ..layers import ConvBNAct
from ..registry import register_model
from ..resnet_base import ResNetBase


class Bottle2neck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, first=False, base_width=26, scale=4, **_):
        super().__init__()
        width = int(math.floor(planes * (base_width / 64.0)))
        self.width = width
        self.scale = scale
        # The first block of a stage cannot reuse the previous split, so each split is processed independently.
        self.is_stage = first
        self.nums = 1 if scale == 1 else scale - 1

        self.conv1 = ConvBNAct(inplanes, width * scale, 1, act_layer=nn.ReLU)
        self.convs = nn.ModuleList(
            [ConvBNAct(width, width, 3, stride, act_layer=nn.ReLU) for _ in range(self.nums)]
        )
        self.pool = nn.AvgPool2d(3, stride, 1) if self.is_stage and scale != 1 else None
        self.conv3 = ConvBNAct(width * scale, planes * self.expansion, 1, act_layer=None)
        self.downsample = downsample
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        shortcut = x if self.downsample is None else self.downsample(x)
        splits = torch.split(self.conv1(x), self.width, 1)

        outs = []
        sp = None
        for i, conv in enumerate(self.convs):
            sp = splits[i] if i == 0 or self.is_stage else sp + splits[i]
            sp = conv(sp)
            outs.append(sp)
        if self.scale != 1:
            outs.append(self.pool(splits[self.nums]) if self.pool is not None else splits[self.nums])

        out = self.conv3(torch.cat(outs, 1))
        return self.act(out + shortcut)


_spec = dict(category="cnn", family="Res2Net", paper="Gao et al., 2019")


@register_model(**_spec)
def res2net50_26w_4s(num_classes=1000, img_size=224):
    return ResNetBase(Bottle2neck, [3, 4, 6, 3], num_classes, img_size, block_kwargs=dict(base_width=26, scale=4))


@register_model(**_spec)
def res2net101_26w_4s(num_classes=1000, img_size=224):
    return ResNetBase(Bottle2neck, [3, 4, 23, 3], num_classes, img_size, block_kwargs=dict(base_width=26, scale=4))
