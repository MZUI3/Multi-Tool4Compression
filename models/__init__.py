"""Model hub. Importing this package registers every architecture."""
from . import cnn, lightweight, torchvision_zoo, transformer  # noqa: F401
from .layers import reparameterize_model
from .registry import (
    CATEGORIES,
    ModelSpec,
    create_model,
    get_model_spec,
    list_models,
    register_builder,
    register_model,
    resolve_input_size,
)
from .transformer.mobilevit import ExMobileViT, MobileViT

__all__ = [
    "CATEGORIES",
    "ModelSpec",
    "create_model",
    "get_model_spec",
    "list_models",
    "register_builder",
    "register_model",
    "reparameterize_model",
    "resolve_input_size",
    "MobileViT",
    "ExMobileViT",
]
