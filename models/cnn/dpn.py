"""DPN — "Dual Path Networks" (Chen et al., 2017)."""
import torch
import torch.nn as nn

from ..layers import ConvBNAct, init_cnn_weights, is_small_input
from ..registry import register_model


class DualPathBlock(nn.Module):
    def __init__(self, last_planes, in_planes, out_planes, dense_depth, stride, first_layer, groups=32):
        super().__init__()
        self.out_planes = out_planes
        self.conv1 = ConvBNAct(last_planes, in_planes, 1, act_layer=nn.ReLU)
        self.conv2 = ConvBNAct(in_planes, in_planes, 3, stride, groups=groups, act_layer=nn.ReLU)
        self.conv3 = ConvBNAct(in_planes, out_planes + dense_depth, 1, act_layer=None)
        self.shortcut = None
        if first_layer:
            self.shortcut = ConvBNAct(last_planes, out_planes + dense_depth, 1, stride, act_layer=None)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv3(self.conv2(self.conv1(x)))
        x = self.shortcut(x) if self.shortcut is not None else x
        d = self.out_planes
        # residual path (first d channels) + densely connected path (the rest)
        out = torch.cat([x[:, :d] + out[:, :d], x[:, d:], out[:, d:]], 1)
        return self.act(out)


class DPN(nn.Module):
    def __init__(self, in_planes, out_planes, num_blocks, dense_depth, num_classes=1000, img_size=224, stem_width=64):
        super().__init__()
        if is_small_input(img_size):
            self.stem = ConvBNAct(3, stem_width, 3, 1, act_layer=nn.ReLU)
        else:
            self.stem = nn.Sequential(ConvBNAct(3, stem_width, 7, 2, act_layer=nn.ReLU), nn.MaxPool2d(3, 2, 1))

        self.last_planes = stem_width
        stages = []
        for i, stride in enumerate([1, 2, 2, 2]):
            stages.append(self._make_layer(in_planes[i], out_planes[i], num_blocks[i], dense_depth[i], stride))
        self.stages = nn.Sequential(*stages)
        self.fc = nn.Linear(self.last_planes, num_classes)
        init_cnn_weights(self)

    def _make_layer(self, in_planes, out_planes, num_blocks, dense_depth, stride):
        blocks = []
        for i, s in enumerate([stride] + [1] * (num_blocks - 1)):
            blocks.append(DualPathBlock(self.last_planes, in_planes, out_planes, dense_depth, s, i == 0))
            self.last_planes = out_planes + (i + 2) * dense_depth
        return nn.Sequential(*blocks)

    def forward(self, x):
        x = self.stages(self.stem(x))
        return self.fc(x.mean((2, 3)))


_spec = dict(category="cnn", family="DPN", paper="Chen et al., 2017")
_widths = dict(in_planes=(96, 192, 384, 768), out_planes=(256, 512, 1024, 2048))


@register_model(**_spec)
def dpn26(num_classes=1000, img_size=224):
    return DPN(num_blocks=(2, 2, 2, 2), dense_depth=(16, 32, 24, 128), num_classes=num_classes, img_size=img_size, **_widths)


@register_model(**_spec)
def dpn92(num_classes=1000, img_size=224):
    return DPN(num_blocks=(3, 4, 20, 3), dense_depth=(16, 32, 24, 128), num_classes=num_classes, img_size=img_size, **_widths)
