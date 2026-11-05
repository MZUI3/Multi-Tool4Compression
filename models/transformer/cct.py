"""CCT — "Escaping the Big Data Paradigm with Compact Transformers" (Hassani et al., 2021).

Conv tokenizer + transformer encoder + sequence pooling. Trains well from scratch on CIFAR.
"""
import torch
import torch.nn as nn

from ..layers import drop_path_rates, init_transformer_weights
from ..registry import register_model
from .common import TransformerBlock


class ConvTokenizer(nn.Module):
    def __init__(self, embed_dim, kernel_size, num_layers, mid_channels=64):
        super().__init__()
        stride = max(1, kernel_size // 2 - 1)
        layers = []
        for i in range(num_layers):
            in_channels = 3 if i == 0 else mid_channels
            out_channels = embed_dim if i == num_layers - 1 else mid_channels
            layers += [
                nn.Conv2d(in_channels, out_channels, kernel_size, stride, kernel_size // 2, bias=False),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(3, 2, 1),
            ]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x).flatten(2).transpose(1, 2)


class CCT(nn.Module):
    def __init__(self, img_size=224, num_classes=1000, embed_dim=256, depth=7, num_heads=4, mlp_ratio=2.0,
                 kernel_size=3, num_conv_layers=1, drop_path_rate=0.1):
        super().__init__()
        self.tokenizer = ConvTokenizer(embed_dim, kernel_size, num_conv_layers)
        with torch.no_grad():
            num_tokens = self.tokenizer(torch.zeros(1, 3, img_size, img_size)).shape[1]
        self.pos_embed = nn.Parameter(torch.zeros(1, num_tokens, embed_dim))
        dpr = drop_path_rates(drop_path_rate, depth)
        self.blocks = nn.Sequential(
            *[TransformerBlock(embed_dim, num_heads, mlp_ratio, drop_path=dpr[i]) for i in range(depth)]
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.attention_pool = nn.Linear(embed_dim, 1)
        self.head = nn.Linear(embed_dim, num_classes)
        init_transformer_weights(self)
        nn.init.trunc_normal_(self.pos_embed, std=0.2)

    def forward(self, x):
        x = self.tokenizer(x) + self.pos_embed
        x = self.norm(self.blocks(x))
        weights = self.attention_pool(x).softmax(1)  # sequence pooling
        x = (weights.transpose(1, 2) @ x).squeeze(1)
        return self.head(x)


_spec = dict(category="transformer", family="CCT", paper="Hassani et al., 2021")


@register_model(**_spec)
def cct_7_3x1(num_classes=1000, img_size=224):
    """CIFAR configuration: 7 layers, one 3x3 conv in the tokenizer."""
    return CCT(img_size, num_classes, embed_dim=256, depth=7, num_heads=4, kernel_size=3, num_conv_layers=1)


@register_model(**_spec)
def cct_14_7x2(num_classes=1000, img_size=224):
    """ImageNet configuration: 14 layers, two 7x7 convs in the tokenizer."""
    return CCT(img_size, num_classes, embed_dim=384, depth=14, num_heads=6, mlp_ratio=3.0, kernel_size=7, num_conv_layers=2)
