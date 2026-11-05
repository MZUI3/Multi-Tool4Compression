"""Building blocks shared by the hand-written models."""
import math
from typing import Optional, Type

import torch
import torch.nn as nn
import torch.nn.functional as F

# Inputs at or below this resolution (CIFAR, Tiny-ImageNet) use a stride-1 stem so the
# feature map is not collapsed to 1x1 before the last stage.
SMALL_INPUT_SIZE = 64


def is_small_input(img_size: int) -> bool:
    return img_size <= SMALL_INPUT_SIZE


def make_divisible(value: float, divisor: int = 8, min_value: Optional[int] = None) -> int:
    min_value = min_value or divisor
    new_value = max(min_value, int(value + divisor / 2) // divisor * divisor)
    if new_value < 0.9 * value:
        new_value += divisor
    return new_value


def create_act(act_layer: Optional[Type[nn.Module]]) -> nn.Module:
    if act_layer is None:
        return nn.Identity()
    try:
        return act_layer(inplace=True)
    except TypeError:
        return act_layer()


class ConvBNAct(nn.Sequential):
    """Conv2d -> BatchNorm2d -> activation. ``act_layer=None`` drops the activation."""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=None,
        groups=1,
        dilation=1,
        act_layer: Optional[Type[nn.Module]] = nn.SiLU,
    ):
        if padding is None:
            padding = (kernel_size - 1) // 2 * dilation
        layers = [
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                dilation=dilation,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
        ]
        if act_layer is not None:
            layers.append(create_act(act_layer))
        super().__init__(*layers)


class SqueezeExcite(nn.Module):
    """Squeeze-and-Excitation (Hu et al., 2018)."""

    def __init__(
        self,
        channels,
        rd_ratio=1 / 16,
        rd_channels=None,
        rd_divisor=1,
        act_layer=nn.ReLU,
        gate_layer=nn.Sigmoid,
    ):
        super().__init__()
        if rd_channels is None:
            rd_channels = make_divisible(channels * rd_ratio, rd_divisor, min_value=1)
        self.fc1 = nn.Conv2d(channels, rd_channels, 1)
        self.act = create_act(act_layer)
        self.fc2 = nn.Conv2d(rd_channels, channels, 1)
        self.gate = gate_layer()

    def forward(self, x):
        scale = x.mean((2, 3), keepdim=True)
        scale = self.fc2(self.act(self.fc1(scale)))
        return x * self.gate(scale)


class DropPath(nn.Module):
    """Stochastic depth per sample (Huang et al., 2016)."""

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = x.new_empty(shape).bernoulli_(keep_prob)
        return x * mask / keep_prob

    def extra_repr(self):
        return f"drop_prob={self.drop_prob}"


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0):
        super().__init__()
        hidden_features = hidden_features or in_features
        out_features = out_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))


class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5, channels_first=False):
        super().__init__()
        self.channels_first = channels_first
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        gamma = self.gamma.view(1, -1, 1, 1) if self.channels_first else self.gamma
        return x * gamma


class LayerNorm2d(nn.LayerNorm):
    """LayerNorm over the channel dim of an NCHW tensor."""

    def forward(self, x):
        return F.layer_norm(x.permute(0, 2, 3, 1), self.normalized_shape, self.weight, self.bias, self.eps).permute(
            0, 3, 1, 2
        )


def channel_shuffle(x: torch.Tensor, groups: int) -> torch.Tensor:
    batch, channels, height, width = x.shape
    x = x.view(batch, groups, channels // groups, height, width).transpose(1, 2)
    return x.reshape(batch, channels, height, width)


def drop_path_rates(drop_path_rate: float, depth: int):
    return [drop_path_rate * i / max(depth - 1, 1) for i in range(depth)]


# --- structural re-parameterisation helpers (RepVGG, MobileOne) ---------------------------


def fuse_conv_bn(kernel: torch.Tensor, bn: nn.BatchNorm2d):
    std = (bn.running_var + bn.eps).sqrt()
    t = (bn.weight / std).reshape(-1, 1, 1, 1)
    return kernel * t, bn.bias - bn.running_mean * bn.weight / std


def identity_kernel(channels: int, groups: int, kernel_size: int, like: torch.Tensor) -> torch.Tensor:
    input_dim = channels // groups
    kernel = torch.zeros(channels, input_dim, kernel_size, kernel_size, dtype=like.dtype, device=like.device)
    center = kernel_size // 2
    for i in range(channels):
        kernel[i, i % input_dim, center, center] = 1.0
    return kernel


def reparameterize_model(model: nn.Module) -> nn.Module:
    """Fold multi-branch training blocks into single convs for inference (in place)."""
    for module in model.modules():
        if hasattr(module, "reparameterize") and module is not model:
            module.reparameterize()
    return model


# --- weight init ---------------------------------------------------------------------------


def init_cnn_weights(model: nn.Module) -> None:
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            nn.init.zeros_(m.bias)


def init_transformer_weights(model: nn.Module) -> None:
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels // m.groups
            nn.init.normal_(m.weight, 0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d, nn.GroupNorm)):
            if m.weight is not None:
                nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
