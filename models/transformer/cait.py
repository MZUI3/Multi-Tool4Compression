"""CaiT — "Going deeper with Image Transformers" (Touvron et al., 2021).

LayerScale + talking-heads self-attention on patches, then class-attention layers that only
update the class token.
"""
import torch
import torch.nn as nn

from ..layers import DropPath, LayerScale, Mlp, drop_path_rates, init_transformer_weights
from ..registry import register_model
from .common import PatchEmbed, auto_patch_size


class TalkingHeadAttention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj_l = nn.Linear(num_heads, num_heads)  # mixes heads before softmax
        self.proj_w = nn.Linear(num_heads, num_heads)  # mixes heads after softmax
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        batch, tokens, channels = x.shape
        qkv = self.qkv(x).reshape(batch, tokens, 3, self.num_heads, channels // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = (q * self.scale) @ k.transpose(-2, -1)
        attn = self.proj_l(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2).softmax(-1)
        attn = self.proj_w(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x = (attn @ v).transpose(1, 2).reshape(batch, tokens, channels)
        return self.proj(x)


class ClassAttention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        batch, tokens, channels = x.shape
        head_dim = channels // self.num_heads
        q = self.q(x[:, :1]).reshape(batch, 1, self.num_heads, head_dim).transpose(1, 2)
        k = self.k(x).reshape(batch, tokens, self.num_heads, head_dim).transpose(1, 2)
        v = self.v(x).reshape(batch, tokens, self.num_heads, head_dim).transpose(1, 2)
        attn = ((q * self.scale) @ k.transpose(-2, -1)).softmax(-1)
        cls = (attn @ v).transpose(1, 2).reshape(batch, 1, channels)
        return self.proj(cls)


class LayerScaleBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio, drop_path, init_values, attn_layer):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = attn_layer(dim, num_heads)
        self.ls1 = LayerScale(dim, init_values)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, int(dim * mlp_ratio))
        self.ls2 = LayerScale(dim, init_values)
        self.drop_path = DropPath(drop_path)

    def forward(self, x, cls_token=None):
        if cls_token is None:
            x = x + self.drop_path(self.ls1(self.attn(self.norm1(x))))
            return x + self.drop_path(self.ls2(self.mlp(self.norm2(x))))
        u = torch.cat([cls_token, x], 1)
        cls_token = cls_token + self.drop_path(self.ls1(self.attn(self.norm1(u))))
        return cls_token + self.drop_path(self.ls2(self.mlp(self.norm2(cls_token))))


class CaiT(nn.Module):
    def __init__(self, img_size=224, num_classes=1000, embed_dim=192, depth=24, num_heads=4, mlp_ratio=4.0,
                 depth_token_only=2, init_values=1e-5, drop_path_rate=0.1):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, auto_patch_size(img_size), 3, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.num_patches, embed_dim))
        dpr = drop_path_rates(drop_path_rate, depth)
        self.blocks = nn.ModuleList(
            [LayerScaleBlock(embed_dim, num_heads, mlp_ratio, dpr[i], init_values, TalkingHeadAttention) for i in range(depth)]
        )
        self.blocks_token_only = nn.ModuleList(
            [LayerScaleBlock(embed_dim, num_heads, mlp_ratio, 0.0, init_values, ClassAttention) for _ in range(depth_token_only)]
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        init_transformer_weights(self)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x):
        x = self.patch_embed(x) + self.pos_embed
        for block in self.blocks:
            x = block(x)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        for block in self.blocks_token_only:
            cls_token = block(x, cls_token)
        return self.head(self.norm(cls_token)[:, 0])


_spec = dict(category="transformer", family="CaiT", paper="Touvron et al., 2021", input_size_divisor=4)


@register_model(**_spec)
def cait_xxs24(num_classes=1000, img_size=224):
    return CaiT(img_size, num_classes, embed_dim=192, depth=24, num_heads=4)


@register_model(**_spec)
def cait_xs24(num_classes=1000, img_size=224):
    return CaiT(img_size, num_classes, embed_dim=288, depth=24, num_heads=6)
