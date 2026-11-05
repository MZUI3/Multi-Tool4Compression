"""Generic ResNet skeleton reused by ResNet variants (SE/CBAM/ECA/SK/ResNeSt/Res2Net/BoTNet).

A block class is called as ``block(inplanes, planes, stride, downsample, first=..., fmap_size=..., **block_kwargs)``
and must expose an ``expansion`` attribute. ``first`` marks the first block of a stage and
``fmap_size`` is the spatial size of the block input (needed by attention blocks).
"""
import math
from typing import Optional, Sequence

import torch.nn as nn

from .layers import ConvBNAct, init_cnn_weights, is_small_input


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, attn_layer=None, **_):
        super().__init__()
        self.conv1 = ConvBNAct(inplanes, planes, 3, stride, act_layer=nn.ReLU)
        self.conv2 = ConvBNAct(planes, planes, 3, 1, act_layer=None)
        self.attn = attn_layer(planes) if attn_layer is not None else nn.Identity()
        self.downsample = downsample
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        shortcut = x if self.downsample is None else self.downsample(x)
        out = self.attn(self.conv2(self.conv1(x)))
        return self.act(out + shortcut)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, attn_layer=None, groups=1, base_width=64, **_):
        super().__init__()
        width = int(planes * (base_width / 64.0)) * groups
        out_planes = planes * self.expansion
        self.conv1 = ConvBNAct(inplanes, width, 1, act_layer=nn.ReLU)
        self.conv2 = ConvBNAct(width, width, 3, stride, groups=groups, act_layer=nn.ReLU)
        self.conv3 = ConvBNAct(width, out_planes, 1, act_layer=None)
        self.attn = attn_layer(out_planes) if attn_layer is not None else nn.Identity()
        self.downsample = downsample
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        shortcut = x if self.downsample is None else self.downsample(x)
        out = self.attn(self.conv3(self.conv2(self.conv1(x))))
        return self.act(out + shortcut)


class ResNetBase(nn.Module):
    def __init__(
        self,
        block,
        layers: Sequence[int],
        num_classes: int = 1000,
        img_size: int = 224,
        block_kwargs: Optional[dict] = None,
        stage_blocks: Optional[Sequence] = None,
        widths: Sequence[int] = (64, 128, 256, 512),
        deep_stem: bool = False,
        stem_width: int = 64,
        avg_down: bool = False,
        drop_rate: float = 0.0,
    ):
        super().__init__()
        block_kwargs = block_kwargs or {}
        stage_blocks = list(stage_blocks) if stage_blocks is not None else [block] * len(layers)
        small = is_small_input(img_size)
        stem_stride = 1 if small else 2

        if deep_stem:
            self.stem = nn.Sequential(
                ConvBNAct(3, stem_width, 3, stem_stride, act_layer=nn.ReLU),
                ConvBNAct(stem_width, stem_width, 3, 1, act_layer=nn.ReLU),
                ConvBNAct(stem_width, 64, 3, 1, act_layer=nn.ReLU),
            )
        else:
            kernel = 3 if small else 7
            self.stem = ConvBNAct(3, 64, kernel, stem_stride, act_layer=nn.ReLU)
        self.maxpool = nn.Identity() if small else nn.MaxPool2d(3, 2, 1)

        fmap_size = img_size if small else math.ceil(math.ceil(img_size / 2) / 2)
        self.inplanes = 64
        self.avg_down = avg_down
        stages = []
        for i, (stage_block, depth, width) in enumerate(zip(stage_blocks, layers, widths)):
            stride = 1 if i == 0 else 2
            stages.append(self._make_stage(stage_block, width, depth, stride, fmap_size, block_kwargs))
            fmap_size = math.ceil(fmap_size / stride)
        self.layer1, self.layer2, self.layer3, self.layer4 = stages

        self.num_features = self.inplanes
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(drop_rate),
            nn.Linear(self.num_features, num_classes),
        )
        init_cnn_weights(self)

    def _make_stage(self, block, planes, depth, stride, fmap_size, block_kwargs):
        downsample = None
        out_planes = planes * block.expansion
        if stride != 1 or self.inplanes != out_planes:
            if self.avg_down and stride != 1:
                downsample = nn.Sequential(
                    nn.AvgPool2d(stride, stride, ceil_mode=True, count_include_pad=False),
                    ConvBNAct(self.inplanes, out_planes, 1, act_layer=None),
                )
            else:
                downsample = ConvBNAct(self.inplanes, out_planes, 1, stride, act_layer=None)

        blocks = [block(self.inplanes, planes, stride, downsample, first=True, fmap_size=fmap_size, **block_kwargs)]
        self.inplanes = out_planes
        fmap_size = math.ceil(fmap_size / stride)
        for _ in range(1, depth):
            blocks.append(block(self.inplanes, planes, 1, None, first=False, fmap_size=fmap_size, **block_kwargs))
        return nn.Sequential(*blocks)

    def forward_features(self, x):
        x = self.maxpool(self.stem(x))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return self.layer4(x)

    def forward(self, x):
        return self.head(self.forward_features(x))
