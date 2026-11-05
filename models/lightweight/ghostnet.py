"""GhostNet — "GhostNet: More Features from Cheap Operations" (Han et al., 2020)."""
import math

import torch
import torch.nn as nn

from ..layers import ConvBNAct, SqueezeExcite, init_cnn_weights, is_small_input, make_divisible
from ..registry import register_model

# kernel, expansion, out, se_ratio, stride — grouped per stage
_CFG = [
    [[3, 16, 16, 0, 1]],
    [[3, 48, 24, 0, 2]],
    [[3, 72, 24, 0, 1]],
    [[5, 72, 40, 0.25, 2]],
    [[5, 120, 40, 0.25, 1]],
    [[3, 240, 80, 0, 2]],
    [[3, 200, 80, 0, 1], [3, 184, 80, 0, 1], [3, 184, 80, 0, 1], [3, 480, 112, 0.25, 1], [3, 672, 112, 0.25, 1]],
    [[5, 672, 160, 0.25, 2]],
    [[5, 960, 160, 0, 1], [5, 960, 160, 0.25, 1], [5, 960, 160, 0, 1], [5, 960, 160, 0.25, 1]],
]


class GhostModule(nn.Module):
    def __init__(self, in_channels, out_channels, ratio=2, dw_size=3, relu=True):
        super().__init__()
        self.out_channels = out_channels
        init_channels = math.ceil(out_channels / ratio)
        act = nn.ReLU if relu else None
        self.primary_conv = ConvBNAct(in_channels, init_channels, 1, act_layer=act)
        self.cheap_operation = ConvBNAct(init_channels, init_channels * (ratio - 1), dw_size, groups=init_channels, act_layer=act)

    def forward(self, x):
        x1 = self.primary_conv(x)
        out = torch.cat([x1, self.cheap_operation(x1)], 1)
        return out[:, : self.out_channels]


class GhostBottleneck(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, dw_kernel_size=3, stride=1, se_ratio=0.0):
        super().__init__()
        self.ghost1 = GhostModule(in_channels, mid_channels, relu=True)
        self.conv_dw = (
            ConvBNAct(mid_channels, mid_channels, dw_kernel_size, stride, groups=mid_channels, act_layer=None)
            if stride > 1
            else nn.Identity()
        )
        self.se = (
            SqueezeExcite(mid_channels, rd_ratio=se_ratio, rd_divisor=4, gate_layer=nn.Hardsigmoid)
            if se_ratio > 0
            else nn.Identity()
        )
        self.ghost2 = GhostModule(mid_channels, out_channels, relu=False)

        if in_channels == out_channels and stride == 1:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Sequential(
                ConvBNAct(in_channels, in_channels, dw_kernel_size, stride, groups=in_channels, act_layer=None),
                ConvBNAct(in_channels, out_channels, 1, act_layer=None),
            )

    def forward(self, x):
        out = self.ghost2(self.se(self.conv_dw(self.ghost1(x))))
        return out + self.shortcut(x)


class GhostNet(nn.Module):
    def __init__(self, num_classes=1000, img_size=224, width=1.0, drop_rate=0.2):
        super().__init__()
        channels = make_divisible(16 * width, 4)
        self.stem = ConvBNAct(3, channels, 3, 1 if is_small_input(img_size) else 2, act_layer=nn.ReLU)

        stages = []
        for cfg in _CFG:
            blocks = []
            for k, exp, out, se, s in cfg:
                out_channels = make_divisible(out * width, 4)
                mid_channels = make_divisible(exp * width, 4)
                blocks.append(GhostBottleneck(channels, mid_channels, out_channels, k, s, se))
                channels = out_channels
            stages.append(nn.Sequential(*blocks))
        out_channels = make_divisible(exp * width, 4)
        stages.append(ConvBNAct(channels, out_channels, 1, act_layer=nn.ReLU))
        self.blocks = nn.Sequential(*stages)

        self.conv_head = nn.Conv2d(out_channels, 1280, 1)
        self.act = nn.ReLU(inplace=True)
        self.classifier = nn.Sequential(nn.Dropout(drop_rate), nn.Linear(1280, num_classes))
        init_cnn_weights(self)

    def forward(self, x):
        x = self.blocks(self.stem(x)).mean((2, 3), keepdim=True)
        x = self.act(self.conv_head(x)).flatten(1)
        return self.classifier(x)


_spec = dict(category="lightweight", family="GhostNet", paper="Han et al., 2020")


@register_model(**_spec)
def ghostnet_1_0(num_classes=1000, img_size=224):
    return GhostNet(num_classes, img_size, width=1.0)


@register_model(**_spec)
def ghostnet_1_3(num_classes=1000, img_size=224):
    return GhostNet(num_classes, img_size, width=1.3)
