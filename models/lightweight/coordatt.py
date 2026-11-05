"""Coordinate Attention — "Coordinate Attention for Efficient Mobile Network Design" (Hou et al., 2021).

MobileNetV2 with a coordinate-attention module after every depthwise conv.
"""
import torch
import torch.nn as nn

from ..layers import ConvBNAct, init_cnn_weights, is_small_input, make_divisible
from ..registry import register_model

# expansion t, channels c, repeats n, stride s
_MBV2_CFG = [(1, 16, 1, 1), (6, 24, 2, 2), (6, 32, 3, 2), (6, 64, 4, 2), (6, 96, 3, 1), (6, 160, 3, 2), (6, 320, 1, 1)]


class CoordAtt(nn.Module):
    def __init__(self, channels, reduction=32):
        super().__init__()
        mid = max(8, channels // reduction)
        self.conv1 = ConvBNAct(channels, mid, 1, act_layer=nn.Hardswish)
        self.conv_h = nn.Conv2d(mid, channels, 1)
        self.conv_w = nn.Conv2d(mid, channels, 1)

    def forward(self, x):
        _, _, height, width = x.shape
        x_h = x.mean(3, keepdim=True)  # B, C, H, 1
        x_w = x.mean(2, keepdim=True).transpose(2, 3)  # B, C, W, 1
        y = self.conv1(torch.cat([x_h, x_w], 2))
        y_h, y_w = torch.split(y, [height, width], 2)
        a_h = self.conv_h(y_h).sigmoid()
        a_w = self.conv_w(y_w.transpose(2, 3)).sigmoid()
        return x * a_h * a_w


class CAInvertedResidual(nn.Module):
    def __init__(self, in_channels, out_channels, stride, expand_ratio):
        super().__init__()
        hidden = int(round(in_channels * expand_ratio))
        self.use_res = stride == 1 and in_channels == out_channels
        layers = []
        if expand_ratio != 1:
            layers.append(ConvBNAct(in_channels, hidden, 1, act_layer=nn.ReLU6))
        layers += [
            ConvBNAct(hidden, hidden, 3, stride, groups=hidden, act_layer=nn.ReLU6),
            CoordAtt(hidden),
            ConvBNAct(hidden, out_channels, 1, act_layer=None),
        ]
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        return x + self.conv(x) if self.use_res else self.conv(x)


class CAMobileNetV2(nn.Module):
    def __init__(self, num_classes=1000, img_size=224, width_mult=1.0, drop_rate=0.2):
        super().__init__()
        in_channels = make_divisible(32 * width_mult)
        last_channels = make_divisible(1280 * max(1.0, width_mult))
        layers = [ConvBNAct(3, in_channels, 3, 1 if is_small_input(img_size) else 2, act_layer=nn.ReLU6)]
        for t, c, n, s in _MBV2_CFG:
            out_channels = make_divisible(c * width_mult)
            for i in range(n):
                layers.append(CAInvertedResidual(in_channels, out_channels, s if i == 0 else 1, t))
                in_channels = out_channels
        layers.append(ConvBNAct(in_channels, last_channels, 1, act_layer=nn.ReLU6))
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Sequential(nn.Dropout(drop_rate), nn.Linear(last_channels, num_classes))
        init_cnn_weights(self)

    def forward(self, x):
        return self.classifier(self.features(x).mean((2, 3)))


@register_model(category="lightweight", family="Coordinate Attention", paper="Hou et al., 2021")
def ca_mobilenet_v2(num_classes=1000, img_size=224):
    return CAMobileNetV2(num_classes, img_size)
