"""MobileOne — "MobileOne: An Improved One millisecond Mobile Backbone" (Vasu et al., 2023).

Multi-branch at train time; call ``models.reparameterize_model(model)`` before deployment.
"""
from collections import OrderedDict

import torch.nn as nn
import torch.nn.functional as F

from ..layers import SqueezeExcite, fuse_conv_bn, identity_kernel, init_cnn_weights, is_small_input
from ..registry import register_model


def _conv_bn(in_channels, out_channels, kernel_size, stride, groups):
    return nn.Sequential(
        OrderedDict(
            conv=nn.Conv2d(in_channels, out_channels, kernel_size, stride, kernel_size // 2, groups=groups, bias=False),
            bn=nn.BatchNorm2d(out_channels),
        )
    )


class MobileOneBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, groups=1, num_conv_branches=1, use_se=False):
        super().__init__()
        self.in_channels = in_channels
        self.groups = groups
        self.kernel_size = kernel_size
        self.se = SqueezeExcite(out_channels, rd_ratio=0.0625) if use_se else nn.Identity()
        self.act = nn.ReLU(inplace=True)

        self.rbr_skip = nn.BatchNorm2d(in_channels) if out_channels == in_channels and stride == 1 else None
        self.rbr_conv = nn.ModuleList(
            [_conv_bn(in_channels, out_channels, kernel_size, stride, groups) for _ in range(num_conv_branches)]
        )
        self.rbr_scale = _conv_bn(in_channels, out_channels, 1, stride, groups) if kernel_size > 1 else None
        self.reparam_conv = None

    def forward(self, x):
        if self.reparam_conv is not None:
            return self.act(self.se(self.reparam_conv(x)))
        out = 0
        if self.rbr_skip is not None:
            out = self.rbr_skip(x)
        if self.rbr_scale is not None:
            out = out + self.rbr_scale(x)
        for branch in self.rbr_conv:
            out = out + branch(x)
        return self.act(self.se(out))

    def reparameterize(self):
        if self.reparam_conv is not None:
            return
        reference = self.rbr_conv[0].conv
        kernel, bias = 0, 0
        for branch in self.rbr_conv:
            k, b = fuse_conv_bn(branch.conv.weight, branch.bn)
            kernel, bias = kernel + k, bias + b
        if self.rbr_scale is not None:
            k, b = fuse_conv_bn(self.rbr_scale.conv.weight, self.rbr_scale.bn)
            pad = self.kernel_size // 2
            kernel, bias = kernel + F.pad(k, [pad, pad, pad, pad]), bias + b
        if self.rbr_skip is not None:
            k = identity_kernel(self.in_channels, self.groups, self.kernel_size, reference.weight)
            k, b = fuse_conv_bn(k, self.rbr_skip)
            kernel, bias = kernel + k, bias + b

        self.reparam_conv = nn.Conv2d(
            reference.in_channels,
            reference.out_channels,
            self.kernel_size,
            reference.stride,
            self.kernel_size // 2,
            groups=self.groups,
            bias=True,
        ).to(reference.weight.device)
        self.reparam_conv.weight.data = kernel.detach()
        self.reparam_conv.bias.data = bias.detach()
        del self.rbr_conv, self.rbr_scale, self.rbr_skip
        self.rbr_conv, self.rbr_scale, self.rbr_skip = nn.ModuleList(), None, None


class MobileOne(nn.Module):
    def __init__(self, width_multipliers, num_conv_branches=1, use_se=False, num_classes=1000, img_size=224,
                 num_blocks=(2, 8, 10, 1)):
        super().__init__()
        self.in_planes = min(64, int(64 * width_multipliers[0]))
        self.stage0 = MobileOneBlock(3, self.in_planes, 3, stride=1 if is_small_input(img_size) else 2)
        self.num_conv_branches = num_conv_branches
        stages = []
        for i, (base, blocks) in enumerate(zip((64, 128, 256, 512), num_blocks)):
            # SE only on the last blocks of stage 3 and all of stage 4 (as in MobileOne-S4)
            num_se = 0 if not use_se else (5 if i == 2 else blocks if i == 3 else 0)
            stages.append(self._make_stage(int(base * width_multipliers[i]), blocks, num_se))
        self.stages = nn.Sequential(*stages)
        self.fc = nn.Linear(self.in_planes, num_classes)
        init_cnn_weights(self)

    def _make_stage(self, planes, num_blocks, num_se):
        blocks = []
        for i, stride in enumerate([2] + [1] * (num_blocks - 1)):
            use_se = i >= num_blocks - num_se
            blocks.append(
                MobileOneBlock(self.in_planes, self.in_planes, 3, stride, self.in_planes, self.num_conv_branches, use_se)
            )
            blocks.append(MobileOneBlock(self.in_planes, planes, 1, 1, 1, self.num_conv_branches, use_se))
            self.in_planes = planes
        return nn.Sequential(*blocks)

    def forward(self, x):
        return self.fc(self.stages(self.stage0(x)).mean((2, 3)))


_spec = dict(category="lightweight", family="MobileOne", paper="Vasu et al., 2023")


@register_model(**_spec)
def mobileone_s0(num_classes=1000, img_size=224):
    return MobileOne((0.75, 1.0, 1.0, 2.0), num_conv_branches=4, num_classes=num_classes, img_size=img_size)


@register_model(**_spec)
def mobileone_s1(num_classes=1000, img_size=224):
    return MobileOne((1.5, 1.5, 2.0, 2.5), num_classes=num_classes, img_size=img_size)


@register_model(**_spec)
def mobileone_s4(num_classes=1000, img_size=224):
    return MobileOne((3.0, 3.5, 3.5, 4.0), use_se=True, num_classes=num_classes, img_size=img_size)
