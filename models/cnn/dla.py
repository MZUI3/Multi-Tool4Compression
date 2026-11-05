"""DLA — "Deep Layer Aggregation" (Yu et al., 2018)."""
import torch
import torch.nn as nn

from ..layers import ConvBNAct, init_cnn_weights, is_small_input
from ..registry import register_model


class DlaBasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1):
        super().__init__()
        self.conv1 = ConvBNAct(inplanes, planes, 3, stride, act_layer=nn.ReLU)
        self.conv2 = ConvBNAct(planes, planes, 3, 1, act_layer=None)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x, residual=None):
        residual = x if residual is None else residual
        return self.act(self.conv2(self.conv1(x)) + residual)


class DlaRoot(nn.Module):
    def __init__(self, in_channels, out_channels, residual):
        super().__init__()
        self.conv = ConvBNAct(in_channels, out_channels, 1, act_layer=None)
        self.residual = residual
        self.act = nn.ReLU(inplace=True)

    def forward(self, *children):
        x = self.conv(torch.cat(children, 1))
        if self.residual:
            x = x + children[0]
        return self.act(x)


class DlaTree(nn.Module):
    """Hierarchical deep aggregation: a binary tree of blocks merged by aggregation nodes."""

    def __init__(self, levels, in_channels, out_channels, stride=1, level_root=False, root_dim=0, root_residual=False):
        super().__init__()
        if root_dim == 0:
            root_dim = 2 * out_channels
        if level_root:
            root_dim += in_channels

        if levels == 1:
            self.tree1 = DlaBasicBlock(in_channels, out_channels, stride)
            self.tree2 = DlaBasicBlock(out_channels, out_channels, 1)
            self.root = DlaRoot(root_dim, out_channels, root_residual)
        else:
            self.tree1 = DlaTree(levels - 1, in_channels, out_channels, stride, root_residual=root_residual)
            self.tree2 = DlaTree(
                levels - 1, out_channels, out_channels, root_dim=root_dim + out_channels, root_residual=root_residual
            )
            self.root = None

        self.levels = levels
        self.level_root = level_root
        self.downsample = nn.MaxPool2d(stride, stride) if stride > 1 else None
        self.project = ConvBNAct(in_channels, out_channels, 1, act_layer=None) if in_channels != out_channels else None

    def forward(self, x, residual=None, children=None):
        children = [] if children is None else children
        bottom = self.downsample(x) if self.downsample is not None else x
        residual = self.project(bottom) if self.project is not None else bottom
        if self.level_root:
            children.append(bottom)
        x1 = self.tree1(x, residual)
        if self.levels == 1:
            x2 = self.tree2(x1)
            return self.root(x2, x1, *children)
        children.append(x1)
        return self.tree2(x1, children=children)


class DLA(nn.Module):
    def __init__(self, levels, channels, num_classes=1000, img_size=224, residual_root=False):
        super().__init__()
        small = is_small_input(img_size)
        self.base_layer = ConvBNAct(3, channels[0], 7, 1, act_layer=nn.ReLU)
        self.level0 = self._make_conv_level(channels[0], channels[0], levels[0])
        self.level1 = self._make_conv_level(channels[0], channels[1], levels[1], stride=1 if small else 2)
        self.level2 = DlaTree(levels[2], channels[1], channels[2], 2, level_root=False, root_residual=residual_root)
        self.level3 = DlaTree(levels[3], channels[2], channels[3], 2, level_root=True, root_residual=residual_root)
        self.level4 = DlaTree(levels[4], channels[3], channels[4], 2, level_root=True, root_residual=residual_root)
        self.level5 = DlaTree(levels[5], channels[4], channels[5], 2, level_root=True, root_residual=residual_root)
        self.fc = nn.Linear(channels[5], num_classes)
        init_cnn_weights(self)

    @staticmethod
    def _make_conv_level(inplanes, planes, convs, stride=1):
        layers = []
        for i in range(convs):
            layers.append(ConvBNAct(inplanes, planes, 3, stride if i == 0 else 1, act_layer=nn.ReLU))
            inplanes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.level1(self.level0(self.base_layer(x)))
        x = self.level5(self.level4(self.level3(self.level2(x))))
        return self.fc(x.mean((2, 3)))


_spec = dict(category="cnn", family="DLA", paper="Yu et al., 2018")


@register_model(**_spec)
def dla34(num_classes=1000, img_size=224):
    return DLA([1, 1, 1, 2, 2, 1], [16, 32, 64, 128, 256, 512], num_classes, img_size)
