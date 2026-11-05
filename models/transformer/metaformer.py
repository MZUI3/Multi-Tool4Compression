"""MetaFormer family.

- PoolFormer      — "MetaFormer Is Actually What You Need for Vision" (Yu et al., 2022)
- EfficientFormer — "EfficientFormer: Vision Transformers at MobileNet Speed" (Li et al., 2022)
"""
import torch.nn as nn

from ..layers import ConvBNAct, DropPath, LayerScale, Mlp, drop_path_rates, init_transformer_weights, is_small_input
from ..registry import register_model
from .common import Attention


class Pooling(nn.Module):
    """Token mixer of PoolFormer: avg-pool minus identity."""

    def __init__(self, pool_size=3):
        super().__init__()
        self.pool = nn.AvgPool2d(pool_size, 1, pool_size // 2, count_include_pad=False)

    def forward(self, x):
        return self.pool(x) - x


class ConvMlp(nn.Module):
    def __init__(self, dim, hidden, norm=False):
        super().__init__()
        self.fc1 = nn.Conv2d(dim, hidden, 1)
        self.norm1 = nn.BatchNorm2d(hidden) if norm else nn.Identity()
        self.act = nn.GELU()
        self.fc2 = nn.Conv2d(hidden, dim, 1)
        self.norm2 = nn.BatchNorm2d(dim) if norm else nn.Identity()

    def forward(self, x):
        return self.norm2(self.fc2(self.act(self.norm1(self.fc1(x)))))


class PoolFormerBlock(nn.Module):
    def __init__(self, dim, mlp_ratio=4.0, drop_path=0.0, init_values=1e-5):
        super().__init__()
        self.norm1 = nn.GroupNorm(1, dim)
        self.token_mixer = Pooling()
        self.ls1 = LayerScale(dim, init_values, channels_first=True)
        self.norm2 = nn.GroupNorm(1, dim)
        self.mlp = ConvMlp(dim, int(dim * mlp_ratio))
        self.ls2 = LayerScale(dim, init_values, channels_first=True)
        self.drop_path = DropPath(drop_path)

    def forward(self, x):
        x = x + self.drop_path(self.ls1(self.token_mixer(self.norm1(x))))
        return x + self.drop_path(self.ls2(self.mlp(self.norm2(x))))


class PoolFormer(nn.Module):
    def __init__(self, layers, dims, num_classes=1000, img_size=224, mlp_ratio=4.0, drop_path_rate=0.1):
        super().__init__()
        small = is_small_input(img_size)
        self.stem = nn.Conv2d(3, dims[0], 3 if small else 7, 1 if small else 4, 1 if small else 2)
        dpr = drop_path_rates(drop_path_rate, sum(layers))
        stages = []
        cursor = 0
        for i, depth in enumerate(layers):
            if i > 0:
                stages.append(nn.Conv2d(dims[i - 1], dims[i], 3, 2, 1))
            stages += [PoolFormerBlock(dims[i], mlp_ratio, dpr[cursor + j]) for j in range(depth)]
            cursor += depth
        self.stages = nn.Sequential(*stages)
        self.norm = nn.GroupNorm(1, dims[-1])
        self.head = nn.Linear(dims[-1], num_classes)
        init_transformer_weights(self)

    def forward(self, x):
        x = self.norm(self.stages(self.stem(x)))
        return self.head(x.mean((2, 3)))


class Meta4D(nn.Module):
    """EfficientFormer conv-style block: pooling mixer + conv MLP with BN."""

    def __init__(self, dim, mlp_ratio=4.0, drop_path=0.0, init_values=1e-5):
        super().__init__()
        self.token_mixer = Pooling()
        self.ls1 = LayerScale(dim, init_values, channels_first=True)
        self.mlp = ConvMlp(dim, int(dim * mlp_ratio), norm=True)
        self.ls2 = LayerScale(dim, init_values, channels_first=True)
        self.drop_path = DropPath(drop_path)

    def forward(self, x):
        x = x + self.drop_path(self.ls1(self.token_mixer(x)))
        return x + self.drop_path(self.ls2(self.mlp(x)))


class Meta3D(nn.Module):
    """EfficientFormer transformer block on flattened tokens."""

    def __init__(self, dim, num_heads=8, mlp_ratio=4.0, drop_path=0.0, init_values=1e-5):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads)
        self.ls1 = LayerScale(dim, init_values)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, int(dim * mlp_ratio))
        self.ls2 = LayerScale(dim, init_values)
        self.drop_path = DropPath(drop_path)

    def forward(self, x):
        x = x + self.drop_path(self.ls1(self.attn(self.norm1(x))))
        return x + self.drop_path(self.ls2(self.mlp(self.norm2(x))))


class EfficientFormer(nn.Module):
    def __init__(self, layers, dims, num_vit=1, num_classes=1000, img_size=224, drop_path_rate=0.0):
        super().__init__()
        small = is_small_input(img_size)
        self.stem = nn.Sequential(
            ConvBNAct(3, dims[0] // 2, 3, 1 if small else 2, act_layer=nn.ReLU),
            ConvBNAct(dims[0] // 2, dims[0], 3, 2, act_layer=nn.ReLU),
        )
        dpr = drop_path_rates(drop_path_rate, sum(layers))
        stages = []
        cursor = 0
        for i, depth in enumerate(layers):
            if i > 0:
                stages.append(ConvBNAct(dims[i - 1], dims[i], 3, 2, act_layer=None))
            num_meta3d = num_vit if i == len(layers) - 1 else 0
            stages += [Meta4D(dims[i], drop_path=dpr[cursor + j]) for j in range(depth - num_meta3d)]
            cursor += depth - num_meta3d
            if num_meta3d:
                self.meta3d = nn.Sequential(*[Meta3D(dims[i], drop_path=dpr[cursor + j]) for j in range(num_meta3d)])
                cursor += num_meta3d
        self.stages = nn.Sequential(*stages)
        self.norm = nn.LayerNorm(dims[-1])
        self.head = nn.Linear(dims[-1], num_classes)
        init_transformer_weights(self)

    def forward(self, x):
        x = self.stages(self.stem(x))
        x = self.meta3d(x.flatten(2).transpose(1, 2))
        return self.head(self.norm(x).mean(1))


_pool = dict(category="transformer", family="PoolFormer (MetaFormer)", paper="Yu et al., 2022")
_eff = dict(category="transformer", family="EfficientFormer", paper="Li et al., 2022")


@register_model(**_pool)
def poolformer_s12(num_classes=1000, img_size=224):
    return PoolFormer((2, 2, 6, 2), (64, 128, 320, 512), num_classes, img_size)


@register_model(**_pool)
def poolformer_s24(num_classes=1000, img_size=224):
    return PoolFormer((4, 4, 12, 4), (64, 128, 320, 512), num_classes, img_size)


@register_model(**_eff)
def efficientformer_l1(num_classes=1000, img_size=224):
    return EfficientFormer((3, 2, 6, 4), (48, 96, 224, 448), num_vit=1, num_classes=num_classes, img_size=img_size)


@register_model(**_eff)
def efficientformer_l3(num_classes=1000, img_size=224):
    return EfficientFormer((4, 4, 12, 6), (64, 128, 320, 512), num_vit=4, num_classes=num_classes, img_size=img_size)
