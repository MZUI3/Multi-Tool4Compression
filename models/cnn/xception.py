"""Xception — "Xception: Deep Learning with Depthwise Separable Convolutions" (Chollet, 2017)."""
import torch.nn as nn

from ..layers import ConvBNAct, init_cnn_weights, is_small_input
from ..registry import register_model


class SeparableConv2d(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super().__init__(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride, kernel_size // 2, groups=in_channels, bias=False),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
        )


class XceptionBlock(nn.Module):
    def __init__(self, in_channels, out_channels, reps, stride=1, start_with_relu=True, grow_first=True):
        super().__init__()
        self.skip = None
        if out_channels != in_channels or stride != 1:
            self.skip = ConvBNAct(in_channels, out_channels, 1, stride, act_layer=None)

        layers = []
        channels = in_channels
        if grow_first:
            layers += [nn.ReLU(), SeparableConv2d(in_channels, out_channels), nn.BatchNorm2d(out_channels)]
            channels = out_channels
        for _ in range(reps - 1):
            layers += [nn.ReLU(), SeparableConv2d(channels, channels), nn.BatchNorm2d(channels)]
        if not grow_first:
            layers += [nn.ReLU(), SeparableConv2d(in_channels, out_channels), nn.BatchNorm2d(out_channels)]
        if not start_with_relu:
            layers = layers[1:]
        if stride != 1:
            layers.append(nn.MaxPool2d(3, stride, 1))
        self.rep = nn.Sequential(*layers)

    def forward(self, x):
        skip = self.skip(x) if self.skip is not None else x
        return self.rep(x) + skip


class Xception(nn.Module):
    def __init__(self, num_classes=1000, img_size=224, middle_blocks=8):
        super().__init__()
        self.entry = nn.Sequential(
            ConvBNAct(3, 32, 3, 1 if is_small_input(img_size) else 2, act_layer=nn.ReLU),
            ConvBNAct(32, 64, 3, 1, act_layer=nn.ReLU),
            XceptionBlock(64, 128, 2, 2, start_with_relu=False),
            XceptionBlock(128, 256, 2, 2),
            XceptionBlock(256, 728, 2, 2),
        )
        self.middle = nn.Sequential(*[XceptionBlock(728, 728, 3, 1) for _ in range(middle_blocks)])
        self.exit = nn.Sequential(
            XceptionBlock(728, 1024, 2, 2, grow_first=False),
            SeparableConv2d(1024, 1536),
            nn.BatchNorm2d(1536),
            nn.ReLU(inplace=True),
            SeparableConv2d(1536, 2048),
            nn.BatchNorm2d(2048),
            nn.ReLU(inplace=True),
        )
        self.fc = nn.Linear(2048, num_classes)
        init_cnn_weights(self)

    def forward(self, x):
        x = self.exit(self.middle(self.entry(x)))
        return self.fc(x.mean((2, 3)))


@register_model(category="cnn", family="Xception", paper="Chollet, 2017")
def xception(num_classes=1000, img_size=224):
    return Xception(num_classes, img_size)
