"""MobileNetV1 — "MobileNets: Efficient CNNs for Mobile Vision Applications" (Howard et al., 2017)."""
import torch.nn as nn

from ..layers import ConvBNAct, init_cnn_weights, is_small_input, make_divisible
from ..registry import register_model

# (out_channels, stride) of each depthwise-separable block
_CFG = [(64, 1), (128, 2), (128, 1), (256, 2), (256, 1), (512, 2)] + [(512, 1)] * 5 + [(1024, 2), (1024, 1)]


class MobileNetV1(nn.Module):
    def __init__(self, num_classes=1000, img_size=224, width_mult=1.0, drop_rate=0.2):
        super().__init__()
        in_channels = make_divisible(32 * width_mult)
        layers = [ConvBNAct(3, in_channels, 3, 1 if is_small_input(img_size) else 2, act_layer=nn.ReLU)]
        for out, stride in _CFG:
            out = make_divisible(out * width_mult)
            layers += [
                ConvBNAct(in_channels, in_channels, 3, stride, groups=in_channels, act_layer=nn.ReLU),
                ConvBNAct(in_channels, out, 1, act_layer=nn.ReLU),
            ]
            in_channels = out
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Sequential(nn.Dropout(drop_rate), nn.Linear(in_channels, num_classes))
        init_cnn_weights(self)

    def forward(self, x):
        return self.classifier(self.features(x).mean((2, 3)))


_spec = dict(category="lightweight", family="MobileNetV1", paper="Howard et al., 2017")


@register_model(**_spec, aliases=("mobilenetv1",))
def mobilenet_v1(num_classes=1000, img_size=224):
    return MobileNetV1(num_classes, img_size, width_mult=1.0)


@register_model(**_spec)
def mobilenet_v1_0_5(num_classes=1000, img_size=224):
    return MobileNetV1(num_classes, img_size, width_mult=0.5)
