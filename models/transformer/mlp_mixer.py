"""MLP-Mixer — "MLP-Mixer: An all-MLP Architecture for Vision" (Tolstikhin et al., 2021)."""
import torch.nn as nn

from ..layers import DropPath, Mlp, drop_path_rates, init_transformer_weights
from ..registry import register_model
from .common import PatchEmbed, auto_patch_size


class MixerBlock(nn.Module):
    def __init__(self, dim, num_tokens, tokens_mlp_dim, channels_mlp_dim, drop_path=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.token_mlp = Mlp(num_tokens, tokens_mlp_dim)
        self.norm2 = nn.LayerNorm(dim)
        self.channel_mlp = Mlp(dim, channels_mlp_dim)
        self.drop_path = DropPath(drop_path)

    def forward(self, x):
        x = x + self.drop_path(self.token_mlp(self.norm1(x).transpose(1, 2)).transpose(1, 2))
        return x + self.drop_path(self.channel_mlp(self.norm2(x)))


class MlpMixer(nn.Module):
    def __init__(self, img_size=224, num_classes=1000, dim=512, depth=8, tokens_mlp_dim=256, channels_mlp_dim=2048,
                 drop_path_rate=0.1):
        super().__init__()
        self.stem = PatchEmbed(img_size, auto_patch_size(img_size), 3, dim)
        dpr = drop_path_rates(drop_path_rate, depth)
        self.blocks = nn.Sequential(
            *[MixerBlock(dim, self.stem.num_patches, tokens_mlp_dim, channels_mlp_dim, dpr[i]) for i in range(depth)]
        )
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)
        init_transformer_weights(self)

    def forward(self, x):
        x = self.norm(self.blocks(self.stem(x)))
        return self.head(x.mean(1))


_spec = dict(category="transformer", family="MLP-Mixer", paper="Tolstikhin et al., 2021", input_size_divisor=4)


@register_model(**_spec)
def mixer_s(num_classes=1000, img_size=224):
    return MlpMixer(img_size, num_classes, dim=512, depth=8, tokens_mlp_dim=256, channels_mlp_dim=2048)


@register_model(**_spec)
def mixer_b(num_classes=1000, img_size=224):
    return MlpMixer(img_size, num_classes, dim=768, depth=12, tokens_mlp_dim=384, channels_mlp_dim=3072)
