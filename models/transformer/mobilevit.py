"""MobileViT / ExMobileViT — "MobileViT: Light-weight, General-purpose, and Mobile-friendly Vision Transformer"
(Mehta & Rastegari, 2022). Project-specific compact variants.
"""
import torch
import torch.nn as nn

from ..layers import ConvBNAct
from ..registry import register_model


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class MobileViTTransformer(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim, dropout=0.0):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "norm1": nn.LayerNorm(dim),
                        "attn": nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True),
                        "norm2": nn.LayerNorm(dim),
                        "ffn": FeedForward(dim, mlp_dim, dropout=dropout),
                    }
                )
                for _ in range(depth)
            ]
        )

    def forward(self, x):
        for layer in self.layers:
            residual = x
            x = layer["norm1"](x)
            attn_output, _ = layer["attn"](x, x, x)
            x = residual + attn_output
            x = x + layer["ffn"](layer["norm2"](x))
        return x


class MobileViTBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        transformer_dim,
        num_transformer_blocks,
        num_heads,
        mlp_dim,
        patch_size=2,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.local_rep = nn.Sequential(
            ConvBNAct(in_channels, in_channels, kernel_size=3, stride=1),
            ConvBNAct(in_channels, transformer_dim, kernel_size=1, stride=1, padding=0),
        )
        self.transformer = MobileViTTransformer(transformer_dim, num_transformer_blocks, num_heads, mlp_dim)
        self.project_tokens = ConvBNAct(transformer_dim, in_channels, kernel_size=1, stride=1, padding=0)
        self.fusion = ConvBNAct(in_channels * 2, in_channels, kernel_size=3, stride=1)

    def _unfold_patches(self, x):
        batch, channels, height, width = x.shape
        return x.view(batch, channels, -1).permute(0, 2, 1)

    def _fold_patches(self, x, height, width):
        batch, seq_len, channels = x.shape
        return x.permute(0, 2, 1).view(batch, channels, height, width)

    def forward(self, x):
        local_features = self.local_rep[0](x)
        x = self.local_rep[1](local_features)

        batch, channels, height, width = x.shape
        token_embeddings = self._unfold_patches(x)
        token_embeddings = self.transformer(token_embeddings)
        token_features = self._fold_patches(token_embeddings, height, width)
        token_features = self.project_tokens(token_features)

        fused = torch.cat([local_features, token_features], dim=1)
        return self.fusion(fused)


class MobileViT(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.stem = ConvBNAct(3, 16, stride=2)
        self.conv1 = ConvBNAct(16, 32, stride=2)
        self.conv2 = ConvBNAct(32, 64, stride=2)
        self.mobilevit_block = MobileViTBlock(
            in_channels=64,
            transformer_dim=128,
            num_transformer_blocks=2,
            num_heads=4,
            mlp_dim=256,
            patch_size=2,
        )
        self.conv3 = ConvBNAct(64, 320, kernel_size=1, stride=1, padding=0)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(320, num_classes),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.mobilevit_block(x)
        x = self.conv3(x)
        return self.classifier(x)


class ExMobileViT(MobileViT):
    """MobileViT with ExShortcut ("ExMobileViT: Lightweight Classifier Extension for Mobile Vision
    Transformer", Yang et al., 2023). The backbone is the same as ``MobileViT``. The feature maps that
    enter and leave the MobileViT block each pass through a 1x1 conv and global average pooling, and are
    concatenated to the last stage's pooled output before the linear classifier."""

    def __init__(self, num_classes=1000, shortcut_channels=(128, 128)):
        super().__init__(num_classes)
        self.shortcuts = nn.ModuleList(
            [ConvBNAct(64, c, kernel_size=1, stride=1, padding=0) for c in shortcut_channels]
        )
        self.pool = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.classifier = nn.Linear(320 + sum(shortcut_channels), num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.conv1(x)
        x = self.conv2(x)
        intermediate = [x]
        x = self.mobilevit_block(x)
        intermediate.append(x)
        x = self.conv3(x)
        pooled = [self.pool(x)] + [self.pool(proj(f)) for proj, f in zip(self.shortcuts, intermediate)]
        return self.classifier(torch.cat(pooled, dim=1))


_spec = dict(category="transformer", family="MobileViT", paper="Mehta & Rastegari, 2022")


@register_model("mobilevit", **_spec)
def build_mobilevit(num_classes=1000, img_size=224):
    return MobileViT(num_classes=num_classes)


@register_model("exmobilevit", **_spec)
def build_exmobilevit(num_classes=1000, img_size=224):
    return ExMobileViT(num_classes=num_classes)
