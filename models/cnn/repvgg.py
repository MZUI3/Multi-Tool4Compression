"""RepVGG — "RepVGG: Making VGG-style ConvNets Great Again" (Ding et al., 2021).

Trains with 3x3 + 1x1 + identity branches; call ``models.reparameterize_model(model)`` after
training to fold them into a single 3x3 conv per block (plain VGG at inference).
"""
from collections import OrderedDict

import torch.nn as nn
import torch.nn.functional as F

from ..layers import fuse_conv_bn, identity_kernel, init_cnn_weights, is_small_input
from ..registry import register_model


def _conv_bn(in_channels, out_channels, kernel_size, stride, padding, groups=1):
    return nn.Sequential(
        OrderedDict(
            conv=nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=False),
            bn=nn.BatchNorm2d(out_channels),
        )
    )


class RepVGGBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, groups=1):
        super().__init__()
        self.in_channels = in_channels
        self.groups = groups
        self.act = nn.ReLU(inplace=True)
        self.rbr_identity = nn.BatchNorm2d(in_channels) if in_channels == out_channels and stride == 1 else None
        self.rbr_dense = _conv_bn(in_channels, out_channels, 3, stride, 1, groups)
        self.rbr_1x1 = _conv_bn(in_channels, out_channels, 1, stride, 0, groups)
        self.rbr_reparam = None

    def forward(self, x):
        if self.rbr_reparam is not None:
            return self.act(self.rbr_reparam(x))
        identity = self.rbr_identity(x) if self.rbr_identity is not None else 0
        return self.act(self.rbr_dense(x) + self.rbr_1x1(x) + identity)

    def _fuse_branch(self, branch):
        if branch is None:
            return 0, 0
        if isinstance(branch, nn.Sequential):
            return fuse_conv_bn(branch.conv.weight, branch.bn)
        kernel = identity_kernel(self.in_channels, self.groups, 3, branch.weight)
        return fuse_conv_bn(kernel, branch)

    def reparameterize(self):
        if self.rbr_reparam is not None:
            return
        k3, b3 = self._fuse_branch(self.rbr_dense)
        k1, b1 = self._fuse_branch(self.rbr_1x1)
        kid, bid = self._fuse_branch(self.rbr_identity)
        conv = self.rbr_dense.conv
        self.rbr_reparam = nn.Conv2d(
            conv.in_channels, conv.out_channels, 3, conv.stride, 1, groups=conv.groups, bias=True
        ).to(conv.weight.device)
        self.rbr_reparam.weight.data = (k3 + F.pad(k1, [1, 1, 1, 1]) + kid).detach()
        self.rbr_reparam.bias.data = (b3 + b1 + bid).detach()
        del self.rbr_dense, self.rbr_1x1, self.rbr_identity
        self.rbr_identity = None


class RepVGG(nn.Module):
    def __init__(self, num_blocks, width_multiplier, num_classes=1000, img_size=224):
        super().__init__()
        in_planes = min(64, int(64 * width_multiplier[0]))
        self.in_planes = in_planes
        self.stage0 = RepVGGBlock(3, in_planes, stride=1 if is_small_input(img_size) else 2)
        self.stage1 = self._make_stage(int(64 * width_multiplier[0]), num_blocks[0], 2)
        self.stage2 = self._make_stage(int(128 * width_multiplier[1]), num_blocks[1], 2)
        self.stage3 = self._make_stage(int(256 * width_multiplier[2]), num_blocks[2], 2)
        self.stage4 = self._make_stage(int(512 * width_multiplier[3]), num_blocks[3], 2)
        self.fc = nn.Linear(self.in_planes, num_classes)
        init_cnn_weights(self)

    def _make_stage(self, planes, depth, stride):
        blocks = []
        for s in [stride] + [1] * (depth - 1):
            blocks.append(RepVGGBlock(self.in_planes, planes, s))
            self.in_planes = planes
        return nn.Sequential(*blocks)

    def forward(self, x):
        x = self.stage0(x)
        x = self.stage4(self.stage3(self.stage2(self.stage1(x))))
        return self.fc(x.mean((2, 3)))


_spec = dict(category="cnn", family="RepVGG", paper="Ding et al., 2021")


@register_model(**_spec)
def repvgg_a0(num_classes=1000, img_size=224):
    return RepVGG([2, 4, 14, 1], [0.75, 0.75, 0.75, 2.5], num_classes, img_size)


@register_model(**_spec)
def repvgg_a1(num_classes=1000, img_size=224):
    return RepVGG([2, 4, 14, 1], [1, 1, 1, 2.5], num_classes, img_size)


@register_model(**_spec)
def repvgg_b0(num_classes=1000, img_size=224):
    return RepVGG([4, 6, 16, 1], [1, 1, 1, 2.5], num_classes, img_size)
