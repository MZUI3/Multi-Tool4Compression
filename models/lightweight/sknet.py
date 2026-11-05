"""SKNet — "Selective Kernel Networks" (Li et al., 2019). SKNet-50 is built on ResNeXt-50 (32x4d)."""
import torch
import torch.nn as nn

from ..layers import ConvBNAct
from ..registry import register_model
from ..resnet_base import ResNetBase


class SKConv(nn.Module):
    def __init__(self, channels, stride=1, branches=2, groups=32, reduction=16, min_dim=32):
        super().__init__()
        hidden = max(channels // reduction, min_dim)
        # branch i: 3x3 conv with dilation i+1 (receptive field of 3x3, 5x5, ...)
        self.convs = nn.ModuleList(
            [ConvBNAct(channels, channels, 3, stride, groups=groups, dilation=i + 1, act_layer=nn.ReLU) for i in range(branches)]
        )
        self.fc = ConvBNAct(channels, hidden, 1, act_layer=nn.ReLU)
        self.fcs = nn.ModuleList([nn.Conv2d(hidden, channels, 1) for _ in range(branches)])

    def forward(self, x):
        feats = torch.stack([conv(x) for conv in self.convs], 1)  # B, M, C, H, W
        z = self.fc(feats.sum(1).mean((2, 3), keepdim=True))
        attn = torch.stack([fc(z) for fc in self.fcs], 1).softmax(1)  # B, M, C, 1, 1
        return (feats * attn).sum(1)


class SKBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=32, base_width=4, **_):
        super().__init__()
        width = int(planes * (base_width / 64.0)) * groups
        self.conv1 = ConvBNAct(inplanes, width, 1, act_layer=nn.ReLU)
        self.conv2 = SKConv(width, stride, groups=groups)
        self.conv3 = ConvBNAct(width, planes * self.expansion, 1, act_layer=None)
        self.downsample = downsample
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        shortcut = x if self.downsample is None else self.downsample(x)
        return self.act(self.conv3(self.conv2(self.conv1(x))) + shortcut)


_spec = dict(category="lightweight", family="SKNet", paper="Li et al., 2019")


@register_model(**_spec)
def sknet50(num_classes=1000, img_size=224):
    return ResNetBase(SKBottleneck, [3, 4, 6, 3], num_classes, img_size)


@register_model(**_spec)
def sknet101(num_classes=1000, img_size=224):
    return ResNetBase(SKBottleneck, [3, 4, 23, 3], num_classes, img_size)
