"""ConvMixer — "Patches Are All You Need?" (Trockman & Kolter, 2022).

A patch-embedding stem followed by depth x (depthwise 9x9 conv with residual, pointwise 1x1 conv), each
with GELU + BatchNorm. Resolution agnostic, so it works with the multi-scale sampler.
"""
import torch.nn as nn

from ..layers import is_small_input
from ..registry import register_model


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return x + self.fn(x)


class ConvMixer(nn.Module):
    def __init__(self, dim=256, depth=8, kernel_size=9, patch_size=7, num_classes=1000):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3, dim, patch_size, stride=patch_size), nn.GELU(), nn.BatchNorm2d(dim))
        self.blocks = nn.Sequential(
            *[
                nn.Sequential(
                    Residual(nn.Sequential(nn.Conv2d(dim, dim, kernel_size, groups=dim, padding=kernel_size // 2), nn.GELU(), nn.BatchNorm2d(dim))),
                    nn.Conv2d(dim, dim, 1),
                    nn.GELU(),
                    nn.BatchNorm2d(dim),
                )
                for _ in range(depth)
            ]
        )
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(dim, num_classes))

    def forward(self, x):
        return self.head(self.blocks(self.stem(x)))


_spec = dict(category="cnn", family="ConvMixer", paper="Trockman & Kolter, 2022")


@register_model(**_spec)
def convmixer_256_8(num_classes=1000, img_size=224):
    # the CIFAR configuration of the paper uses 2x2 patches (1x1 would keep 32x32 maps and is very slow)
    return ConvMixer(256, 8, kernel_size=9 if is_small_input(img_size) else 9, patch_size=2 if is_small_input(img_size) else 7, num_classes=num_classes)


@register_model(**_spec)
def convmixer_512_16(num_classes=1000, img_size=224):
    return ConvMixer(512, 16, kernel_size=9, patch_size=2 if is_small_input(img_size) else 7, num_classes=num_classes)


@register_model(**_spec)
def convmixer_768_32(num_classes=1000, img_size=224):
    return ConvMixer(768, 32, kernel_size=7, patch_size=2 if is_small_input(img_size) else 7, num_classes=num_classes)
