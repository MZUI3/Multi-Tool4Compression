"""Transformer building blocks shared by ViT-style models."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..layers import DropPath, LayerScale, Mlp


def auto_patch_size(img_size: int) -> int:
    """16x16 patches for ImageNet-sized inputs, 4x4 for CIFAR-sized inputs (keeps 64+ tokens)."""
    return 16 if img_size >= 128 else 4


class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        if img_size % patch_size != 0:
            raise ValueError(f"img_size {img_size} must be divisible by patch_size {patch_size}")
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size**2
        self.proj = nn.Conv2d(in_channels, embed_dim, patch_size, patch_size)

    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)  # B, N, C


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=True, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim {dim} must be divisible by num_heads {num_heads}")
        self.num_heads = num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = attn_drop
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, attn_bias=None):
        batch, tokens, channels = x.shape
        qkv = self.qkv(x).reshape(batch, tokens, 3, self.num_heads, channels // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        x = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_bias, dropout_p=self.attn_drop if self.training else 0.0
        )
        x = x.transpose(1, 2).reshape(batch, tokens, channels)
        return self.proj_drop(self.proj(x))


class TransformerBlock(nn.Module):
    """Pre-norm transformer encoder block with optional LayerScale."""

    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        init_values=None,
        attn_layer=Attention,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = attn_layer(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.ls1 = LayerScale(dim, init_values) if init_values else nn.Identity()
        self.drop_path = DropPath(drop_path)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, int(dim * mlp_ratio), drop=drop)
        self.ls2 = LayerScale(dim, init_values) if init_values else nn.Identity()

    def forward(self, x):
        x = x + self.drop_path(self.ls1(self.attn(self.norm1(x))))
        return x + self.drop_path(self.ls2(self.mlp(self.norm2(x))))


class RelativePositionBias2D(nn.Module):
    """Learned relative position bias for attention over a fixed HxW grid (Swin / CoAtNet style)."""

    def __init__(self, num_heads, height, width):
        super().__init__()
        self.height, self.width = height, width
        self.table = nn.Parameter(torch.zeros((2 * height - 1) * (2 * width - 1), num_heads))
        nn.init.trunc_normal_(self.table, std=0.02)

        coords = torch.stack(torch.meshgrid(torch.arange(height), torch.arange(width), indexing="ij")).flatten(1)
        relative = (coords[:, :, None] - coords[:, None, :]).permute(1, 2, 0)
        relative[:, :, 0] += height - 1
        relative[:, :, 1] += width - 1
        index = relative[:, :, 0] * (2 * width - 1) + relative[:, :, 1]
        self.register_buffer("index", index, persistent=False)

    def forward(self):
        tokens = self.height * self.width
        bias = self.table[self.index.view(-1)].view(tokens, tokens, -1)
        return bias.permute(2, 0, 1).unsqueeze(0)  # 1, heads, N, N
