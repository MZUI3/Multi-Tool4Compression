"""PyramidNet — "Deep Pyramidal Residual Networks" (Han et al., 2017). CIFAR-style 3-stage network."""
import torch.nn as nn
import torch.nn.functional as F

from ..layers import init_cnn_weights, is_small_input
from ..registry import register_model


def _pad_channels(shortcut, channels):
    extra = channels - shortcut.shape[1]
    return F.pad(shortcut, (0, 0, 0, 0, 0, extra)) if extra > 0 else shortcut


class PyramidBasicBlock(nn.Module):
    outchannel_ratio = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.conv1 = nn.Conv2d(inplanes, planes, 3, stride, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, 1, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.downsample = downsample

    def forward(self, x):
        out = self.conv1(self.bn1(x))
        out = self.bn3(self.conv2(F.relu(self.bn2(out))))
        shortcut = self.downsample(x) if self.downsample is not None else x
        return out + _pad_channels(shortcut, out.shape[1])


class PyramidBottleneck(nn.Module):
    outchannel_ratio = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.outchannel_ratio, 1, bias=False)
        self.bn4 = nn.BatchNorm2d(planes * self.outchannel_ratio)
        self.downsample = downsample

    def forward(self, x):
        out = self.conv1(self.bn1(x))
        out = self.conv2(F.relu(self.bn2(out)))
        out = self.bn4(self.conv3(F.relu(self.bn3(out))))
        shortcut = self.downsample(x) if self.downsample is not None else x
        return out + _pad_channels(shortcut, out.shape[1])


class PyramidNet(nn.Module):
    def __init__(self, depth, alpha, bottleneck, num_classes=1000, img_size=224):
        super().__init__()
        block = PyramidBottleneck if bottleneck else PyramidBasicBlock
        n = (depth - 2) // 9 if bottleneck else (depth - 2) // 6
        self.addrate = alpha / (3 * n)

        self.input_featuremap_dim = 16
        if is_small_input(img_size):
            self.stem = nn.Sequential(nn.Conv2d(3, 16, 3, 1, 1, bias=False), nn.BatchNorm2d(16))
        else:
            self.stem = nn.Sequential(
                nn.Conv2d(3, 16, 7, 2, 3, bias=False), nn.BatchNorm2d(16), nn.ReLU(inplace=True), nn.MaxPool2d(3, 2, 1)
            )

        self.featuremap_dim = float(self.input_featuremap_dim)
        self.layer1 = self._make_layer(block, n, 1)
        self.layer2 = self._make_layer(block, n, 2)
        self.layer3 = self._make_layer(block, n, 2)
        self.bn_final = nn.BatchNorm2d(self.input_featuremap_dim)
        self.fc = nn.Linear(self.input_featuremap_dim, num_classes)
        init_cnn_weights(self)

    def _make_layer(self, block, n, stride):
        downsample = nn.AvgPool2d(2, 2, ceil_mode=True) if stride != 1 else None
        blocks = []
        for i in range(n):
            self.featuremap_dim += self.addrate
            planes = int(round(self.featuremap_dim))
            blocks.append(block(self.input_featuremap_dim, planes, stride if i == 0 else 1, downsample if i == 0 else None))
            self.input_featuremap_dim = planes * block.outchannel_ratio
        return nn.Sequential(*blocks)

    def forward(self, x):
        x = self.layer3(self.layer2(self.layer1(self.stem(x))))
        x = F.relu(self.bn_final(x))
        return self.fc(x.mean((2, 3)))


_spec = dict(category="cnn", family="PyramidNet", paper="Han et al., 2017")


@register_model(**_spec)
def pyramidnet110_a48(num_classes=1000, img_size=224):
    return PyramidNet(110, 48, bottleneck=False, num_classes=num_classes, img_size=img_size)


@register_model(**_spec)
def pyramidnet164_a270(num_classes=1000, img_size=224):
    return PyramidNet(164, 270, bottleneck=True, num_classes=num_classes, img_size=img_size)
