"""Architectures that torchvision already implements well. Trained from scratch (``weights=None``).

torchvision stems are designed for 224px ImageNet images (stride-2 conv + max-pool, or 4x4 patches),
which reduce a 32px CIFAR image to 1x1 before the last stage. For inputs <= 64px each family gets the
usual small-image adaptation (3x3 stride-1 stem, no stem pooling, 2x2 patches) — the same convention
the hand-written models follow.
"""
from functools import partial

import torch.nn as nn
from torchvision import models as tvm

from .layers import is_small_input
from .registry import register_builder


def _conv3x3_like(conv: nn.Conv2d) -> nn.Conv2d:
    new = nn.Conv2d(conv.in_channels, conv.out_channels, 3, 1, 1, bias=conv.bias is not None)
    nn.init.kaiming_normal_(new.weight, mode="fan_out", nonlinearity="relu")
    if new.bias is not None:
        nn.init.zeros_(new.bias)
    return new


def _patch2x2_like(conv: nn.Conv2d) -> nn.Conv2d:
    new = nn.Conv2d(conv.in_channels, conv.out_channels, 2, 2, bias=conv.bias is not None)
    nn.init.trunc_normal_(new.weight, std=0.02)
    if new.bias is not None:
        nn.init.zeros_(new.bias)
    return new


def _stride1(conv: nn.Conv2d) -> None:
    conv.stride = (1, 1)


# --- small-input adapters: (model, num_classes) -> None, applied in place -------------------


def _adapt_resnet(m, _):
    m.conv1 = _conv3x3_like(m.conv1)
    m.maxpool = nn.Identity()


def _adapt_densenet(m, _):
    m.features.conv0 = _conv3x3_like(m.features.conv0)
    m.features.pool0 = nn.Identity()


def _adapt_googlenet(m, _):
    m.conv1.conv = _conv3x3_like(m.conv1.conv)
    m.maxpool1 = nn.Identity()


def _adapt_vgg(m, num_classes):
    # 5 max-pools turn 32px into 1x1; the 7x7 adaptive pool + 4096-wide classifier only add dead weight.
    m.avgpool = nn.AdaptiveAvgPool2d(1)
    m.classifier = nn.Sequential(nn.Linear(512, 512), nn.ReLU(inplace=True), nn.Dropout(0.5), nn.Linear(512, num_classes))
    for layer in m.classifier:
        if isinstance(layer, nn.Linear):
            nn.init.normal_(layer.weight, 0, 0.01)
            nn.init.zeros_(layer.bias)


def _adapt_features_stem(m, _):  # MobileNetV2/V3, EfficientNet(V2): features[0] = Conv2dNormActivation
    _stride1(m.features[0][0])


def _adapt_mnasnet(m, _):
    _stride1(m.layers[0])


def _adapt_shufflenet(m, _):
    _stride1(m.conv1[0])
    m.maxpool = nn.Identity()


def _adapt_squeezenet(m, _):
    m.features[0] = _conv3x3_like(m.features[0])


def _adapt_regnet(m, _):
    _stride1(m.stem[0])


def _adapt_patch_stem(m, _):  # ConvNeXt, Swin: features[0][0] = 4x4 stride-4 patchify conv
    m.features[0][0] = _patch2x2_like(m.features[0][0])


# --- builders ------------------------------------------------------------------------------


def _tv_builder(fn, adapter=None, **fixed_kwargs):
    def build(num_classes=1000, img_size=224):
        model = fn(weights=None, num_classes=num_classes, **fixed_kwargs)
        if adapter is not None and is_small_input(img_size):
            adapter(model, num_classes)
        return model

    build.__name__ = fn.__name__
    return build


def _tv_vit(patch_size, small_patch_size, large=False):
    """torchvision ViT-B / ViT-L; 16/32px patches on ImageNet-sized inputs, 4/8px patches on small inputs."""
    layers, heads, hidden, mlp = (24, 16, 1024, 4096) if large else (12, 12, 768, 3072)

    def build(num_classes=1000, img_size=224):
        patch = small_patch_size if is_small_input(img_size) else patch_size
        return tvm.VisionTransformer(
            image_size=img_size,
            patch_size=patch,
            num_layers=layers,
            num_heads=heads,
            hidden_dim=hidden,
            mlp_dim=mlp,
            num_classes=num_classes,
        )

    build.__name__ = f"vit_{'l' if large else 'b'}_{patch_size}"
    return build


def _tv_maxvit(fn):
    def build(num_classes=1000, img_size=224):
        return fn(weights=None, num_classes=num_classes, input_size=(img_size, img_size))

    build.__name__ = fn.__name__
    return build


def _register_family(names, category, family, paper, adapter=None, aliases=None, builder_factory=None, **spec_kwargs):
    aliases = aliases or {}
    for name in names:
        fn = getattr(tvm, name)
        builder = builder_factory(fn) if builder_factory is not None else _tv_builder(fn, adapter)
        register_builder(
            name,
            builder,
            category=category,
            family=family,
            paper=paper,
            aliases=aliases.get(name, ()),
            **spec_kwargs,
        )


cnn = partial(_register_family, category="cnn")
lightweight = partial(_register_family, category="lightweight")
transformer = partial(_register_family, category="transformer")

# --- CNN & Residual ------------------------------------------------------------------------
cnn(["alexnet"], family="AlexNet", paper="Krizhevsky et al., 2012", min_input_size=64)
cnn(
    ["vgg11", "vgg11_bn", "vgg13", "vgg13_bn", "vgg16", "vgg16_bn", "vgg19", "vgg19_bn"],
    family="VGG",
    paper="Simonyan & Zisserman, 2014",
    adapter=_adapt_vgg,
)
# Auxiliary classifiers would make forward() return a tuple during training, so they are disabled.
cnn(
    ["googlenet"],
    family="GoogLeNet (Inception v1)",
    paper="Szegedy et al., 2015",
    builder_factory=lambda fn: _tv_builder(fn, _adapt_googlenet, aux_logits=False, init_weights=True),
)
cnn(
    ["inception_v3"],
    family="Inception v3",
    paper="Szegedy et al., 2016",
    builder_factory=lambda fn: _tv_builder(fn, None, aux_logits=False, init_weights=True),
    min_input_size=75,
)
cnn(
    ["resnet18", "resnet34", "resnet50", "resnet101", "resnet152"],
    family="ResNet",
    paper="He et al., 2016",
    adapter=_adapt_resnet,
)
cnn(["resnext50_32x4d", "resnext101_32x8d", "resnext101_64x4d"], family="ResNeXt", paper="Xie et al., 2017", adapter=_adapt_resnet)
cnn(
    ["wide_resnet50_2", "wide_resnet101_2"],
    family="Wide ResNet",
    paper="Zagoruyko & Komodakis, 2016",
    adapter=_adapt_resnet,
)
cnn(
    ["regnet_x_400mf", "regnet_x_800mf", "regnet_x_1_6gf", "regnet_x_3_2gf", "regnet_y_400mf", "regnet_y_800mf",
     "regnet_y_1_6gf", "regnet_y_3_2gf", "regnet_y_8gf"],
    family="RegNet",
    paper="Radosavovic et al., 2020",
    adapter=_adapt_regnet,
)
cnn(["convnext_tiny", "convnext_small", "convnext_base", "convnext_large"], family="ConvNeXt", paper="Liu et al., 2022", adapter=_adapt_patch_stem)
cnn(["densenet121", "densenet169", "densenet201"], family="DenseNet", paper="Huang et al., 2017", adapter=_adapt_densenet)

# --- Lightweight & Channel Attention -------------------------------------------------------
lightweight(
    ["mobilenet_v2"],
    family="MobileNetV2",
    paper="Sandler et al., 2018",
    adapter=_adapt_features_stem,
    aliases={"mobilenet_v2": ("mobilenet",)},
)
lightweight(
    ["mobilenet_v3_small", "mobilenet_v3_large"],
    family="MobileNetV3",
    paper="Howard et al., 2019",
    adapter=_adapt_features_stem,
)
lightweight(
    ["shufflenet_v2_x0_5", "shufflenet_v2_x1_0", "shufflenet_v2_x1_5", "shufflenet_v2_x2_0"],
    family="ShuffleNetV2",
    paper="Ma et al., 2018",
    adapter=_adapt_shufflenet,
)
lightweight(["squeezenet1_0", "squeezenet1_1"], family="SqueezeNet", paper="Iandola et al., 2016", adapter=_adapt_squeezenet)
lightweight(["mnasnet0_5", "mnasnet0_75", "mnasnet1_0", "mnasnet1_3"], family="MnasNet", paper="Tan et al., 2019", adapter=_adapt_mnasnet)
lightweight(
    ["efficientnet_b0", "efficientnet_b1", "efficientnet_b2", "efficientnet_b3", "efficientnet_b4"],
    family="EfficientNet",
    paper="Tan & Le, 2019",
    adapter=_adapt_features_stem,
)
lightweight(["efficientnet_v2_s", "efficientnet_v2_m", "efficientnet_v2_l"], family="EfficientNetV2", paper="Tan & Le, 2021", adapter=_adapt_features_stem)

# --- Transformer ---------------------------------------------------------------------------
register_builder(
    "vit_b_16", _tv_vit(16, 4), category="transformer", family="ViT", paper="Dosovitskiy et al., 2021", input_size_divisor=16
)
register_builder(
    "vit_b_32", _tv_vit(32, 8), category="transformer", family="ViT", paper="Dosovitskiy et al., 2021", input_size_divisor=32
)
register_builder(
    "vit_l_16", _tv_vit(16, 4, large=True), category="transformer", family="ViT", paper="Dosovitskiy et al., 2021", input_size_divisor=16
)
register_builder(
    "vit_l_32", _tv_vit(32, 8, large=True), category="transformer", family="ViT", paper="Dosovitskiy et al., 2021", input_size_divisor=32
)
transformer(["swin_t", "swin_s", "swin_b"], family="Swin Transformer", paper="Liu et al., 2021", adapter=_adapt_patch_stem)
transformer(["swin_v2_t", "swin_v2_s", "swin_v2_b"], family="Swin Transformer V2", paper="Liu et al., 2022", adapter=_adapt_patch_stem)
transformer(
    ["maxvit_t"],
    family="MaxViT",
    paper="Tu et al., 2022",
    builder_factory=_tv_maxvit,
    min_input_size=224,
    input_size_divisor=224,
)
