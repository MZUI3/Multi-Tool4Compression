"""ResNeSt — "ResNeSt: Split-Attention Networks" (Zhang et al., 2020)."""
import torch.nn as nn

from ..layers import ConvBNAct
from ..registry import register_model
from ..resnet_base import ResNetBase


class SplitAttnConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, groups=1, radix=2, reduction=4, min_dim=32):
        super().__init__()
        self.radix = radix
        self.groups = groups
        self.out_channels = out_channels
        hidden = max(in_channels * radix // reduction, min_dim)
        self.conv = ConvBNAct(in_channels, out_channels * radix, 3, stride, groups=groups * radix, act_layer=nn.ReLU)
        self.fc1 = ConvBNAct(out_channels, hidden, 1, groups=groups, act_layer=nn.ReLU)
        self.fc2 = nn.Conv2d(hidden, out_channels * radix, 1, groups=groups)

    def _rsoftmax(self, x):
        batch = x.shape[0]
        if self.radix > 1:
            x = x.view(batch, self.groups, self.radix, -1).transpose(1, 2).softmax(1)
            return x.reshape(batch, -1)
        return x.sigmoid().reshape(batch, -1)

    def forward(self, x):
        x = self.conv(x)
        batch, _, height, width = x.shape
        splits = x.view(batch, self.radix, self.out_channels, height, width)
        gap = self.fc1(splits.sum(1).mean((2, 3), keepdim=True))
        attn = self._rsoftmax(self.fc2(gap)).view(batch, self.radix, self.out_channels, 1, 1)
        return (splits * attn).sum(1)


class ResNeStBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, radix=2, groups=1, base_width=64, **_):
        super().__init__()
        width = int(planes * (base_width / 64.0)) * groups
        self.conv1 = ConvBNAct(inplanes, width, 1, act_layer=nn.ReLU)
        self.conv2 = SplitAttnConv(width, width, 1, groups=groups, radix=radix)
        # "avd": downsample with avg-pool after the split-attention conv
        self.avd = nn.AvgPool2d(3, stride, 1) if stride > 1 else nn.Identity()
        self.conv3 = ConvBNAct(width, planes * self.expansion, 1, act_layer=None)
        self.downsample = downsample
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        shortcut = x if self.downsample is None else self.downsample(x)
        out = self.conv3(self.avd(self.conv2(self.conv1(x))))
        return self.act(out + shortcut)


_spec = dict(category="lightweight", family="ResNeSt", paper="Zhang et al., 2020")


@register_model(**_spec)
def resnest50(num_classes=1000, img_size=224):
    return ResNetBase(
        ResNeStBottleneck, [3, 4, 6, 3], num_classes, img_size, deep_stem=True, stem_width=32, avg_down=True
    )


@register_model(**_spec)
def resnest101(num_classes=1000, img_size=224):
    return ResNetBase(
        ResNeStBottleneck, [3, 4, 23, 3], num_classes, img_size, deep_stem=True, stem_width=64, avg_down=True
    )
