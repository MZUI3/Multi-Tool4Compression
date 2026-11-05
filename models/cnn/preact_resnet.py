"""Pre-activation ResNet — "Identity Mappings in Deep Residual Networks" (He et al., 2016)."""
import torch.nn as nn
import torch.nn.functional as F

from ..layers import init_cnn_weights, is_small_input
from ..registry import register_model


class PreActBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, 1, 1, bias=False)
        self.shortcut = None
        if stride != 1 or in_planes != planes * self.expansion:
            self.shortcut = nn.Conv2d(in_planes, planes * self.expansion, 1, stride, bias=False)

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if self.shortcut is not None else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        return out + shortcut


class PreActBottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.shortcut = None
        if stride != 1 or in_planes != planes * self.expansion:
            self.shortcut = nn.Conv2d(in_planes, planes * self.expansion, 1, stride, bias=False)

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if self.shortcut is not None else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out = self.conv3(F.relu(self.bn3(out)))
        return out + shortcut


class PreActResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000, img_size=224):
        super().__init__()
        if is_small_input(img_size):
            self.stem = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        else:
            self.stem = nn.Sequential(nn.Conv2d(3, 64, 7, 2, 3, bias=False), nn.MaxPool2d(3, 2, 1))

        self.in_planes = 64
        self.layer1 = self._make_layer(block, 64, layers[0], 1)
        self.layer2 = self._make_layer(block, 128, layers[1], 2)
        self.layer3 = self._make_layer(block, 256, layers[2], 2)
        self.layer4 = self._make_layer(block, 512, layers[3], 2)
        self.bn = nn.BatchNorm2d(self.in_planes)
        self.fc = nn.Linear(self.in_planes, num_classes)
        init_cnn_weights(self)

    def _make_layer(self, block, planes, depth, stride):
        blocks = []
        for s in [stride] + [1] * (depth - 1):
            blocks.append(block(self.in_planes, planes, s))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*blocks)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        x = F.relu(self.bn(x))
        return self.fc(x.mean((2, 3)))


_spec = dict(category="cnn", family="PreAct ResNet", paper="He et al., 2016 (Identity Mappings)")


@register_model(**_spec)
def preact_resnet18(num_classes=1000, img_size=224):
    return PreActResNet(PreActBasicBlock, [2, 2, 2, 2], num_classes, img_size)


@register_model(**_spec)
def preact_resnet34(num_classes=1000, img_size=224):
    return PreActResNet(PreActBasicBlock, [3, 4, 6, 3], num_classes, img_size)


@register_model(**_spec)
def preact_resnet50(num_classes=1000, img_size=224):
    return PreActResNet(PreActBottleneck, [3, 4, 6, 3], num_classes, img_size)
