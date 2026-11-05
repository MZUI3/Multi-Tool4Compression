"""ShuffleNetV1 — "ShuffleNet: An Extremely Efficient CNN for Mobile Devices" (Zhang et al., 2018)."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..layers import ConvBNAct, channel_shuffle, init_cnn_weights, is_small_input
from ..registry import register_model

_STAGE_CHANNELS = {1: 144, 2: 200, 3: 240, 4: 272, 8: 384}


class ShuffleV1Unit(nn.Module):
    def __init__(self, in_channels, out_channels, groups, stride, first_group_conv=True):
        super().__init__()
        self.stride = stride
        self.groups = groups
        mid = out_channels // 4
        branch_out = out_channels - in_channels if stride == 2 else out_channels

        self.conv1 = ConvBNAct(in_channels, mid, 1, groups=groups if first_group_conv else 1, act_layer=nn.ReLU)
        self.dwconv = ConvBNAct(mid, mid, 3, stride, groups=mid, act_layer=None)
        self.conv3 = ConvBNAct(mid, branch_out, 1, groups=groups, act_layer=None)
        self.first_group_conv = first_group_conv

    def forward(self, x):
        out = self.conv1(x)
        if self.first_group_conv:
            out = channel_shuffle(out, self.groups)
        out = self.conv3(self.dwconv(out))
        if self.stride == 2:
            return F.relu(torch.cat([F.avg_pool2d(x, 3, 2, 1), out], 1))
        return F.relu(x + out)


class ShuffleNetV1(nn.Module):
    def __init__(self, groups=3, num_classes=1000, img_size=224, repeats=(4, 8, 4)):
        super().__init__()
        small = is_small_input(img_size)
        self.stem = nn.Sequential(
            ConvBNAct(3, 24, 3, 1 if small else 2, act_layer=nn.ReLU),
            nn.Identity() if small else nn.MaxPool2d(3, 2, 1),
        )
        channels = _STAGE_CHANNELS[groups]
        in_channels = 24
        stages = []
        for i, repeat in enumerate(repeats):
            out_channels = channels * (2**i)
            units = []
            for j in range(repeat):
                units.append(
                    ShuffleV1Unit(
                        in_channels,
                        out_channels,
                        groups,
                        stride=2 if j == 0 else 1,
                        # The stem output (24 ch) is too narrow for a grouped 1x1 conv.
                        first_group_conv=not (i == 0 and j == 0),
                    )
                )
                in_channels = out_channels
            stages.append(nn.Sequential(*units))
        self.stages = nn.Sequential(*stages)
        self.fc = nn.Linear(in_channels, num_classes)
        init_cnn_weights(self)

    def forward(self, x):
        return self.fc(self.stages(self.stem(x)).mean((2, 3)))


_spec = dict(category="lightweight", family="ShuffleNetV1", paper="Zhang et al., 2018")


@register_model(**_spec)
def shufflenet_v1_g3(num_classes=1000, img_size=224):
    return ShuffleNetV1(3, num_classes, img_size)


@register_model(**_spec)
def shufflenet_v1_g8(num_classes=1000, img_size=224):
    return ShuffleNetV1(8, num_classes, img_size)
