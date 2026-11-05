"""Model registry.

Every architecture registers itself with ``@register_model`` (or ``register_builder``),
so ``main.py`` / ``trainer.py`` never need to know about individual models.

All builders share one signature::

    builder(num_classes: int = 1000, img_size: int = 224) -> nn.Module
"""
import difflib
import re
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import torch.nn as nn

CATEGORIES = {
    "cnn": "CNN & Residual",
    "lightweight": "Lightweight & Channel Attention (Squeeze)",
    "transformer": "Transformer & MLP",
}


@dataclass(frozen=True)
class ModelSpec:
    name: str
    builder: Callable[..., nn.Module]
    category: str
    family: str
    paper: str = ""
    # Smallest input resolution the architecture can process.
    min_input_size: int = 32
    # Input resolution must be a multiple of this value (1 = no constraint).
    input_size_divisor: int = 1
    aliases: Tuple[str, ...] = field(default_factory=tuple)

    def build(self, num_classes: int, img_size: int) -> nn.Module:
        return self.builder(num_classes=num_classes, img_size=img_size)


_REGISTRY: Dict[str, ModelSpec] = {}
_LOOKUP: Dict[str, str] = {}


def _normalize(name: str) -> str:
    return re.sub(r"[\s_\-.]", "", str(name).lower())


def _add_lookup(key: str, name: str) -> None:
    normalized = _normalize(key)
    if normalized in _LOOKUP and _LOOKUP[normalized] != name:
        raise KeyError(f"Model name '{key}' collides with already registered '{_LOOKUP[normalized]}'")
    _LOOKUP[normalized] = name


def register_builder(
    name: str,
    builder: Callable[..., nn.Module],
    *,
    category: str,
    family: str,
    paper: str = "",
    min_input_size: int = 32,
    input_size_divisor: int = 1,
    aliases: Tuple[str, ...] = (),
) -> ModelSpec:
    if category not in CATEGORIES:
        raise ValueError(f"Unknown category '{category}'. Expected one of {list(CATEGORIES)}")
    if name in _REGISTRY:
        raise KeyError(f"Model '{name}' is already registered")

    spec = ModelSpec(
        name=name,
        builder=builder,
        category=category,
        family=family,
        paper=paper,
        min_input_size=min_input_size,
        input_size_divisor=input_size_divisor,
        aliases=tuple(aliases),
    )
    _add_lookup(name, name)
    for alias in aliases:
        _add_lookup(alias, name)
    _REGISTRY[name] = spec
    return spec


def register_model(name: Optional[str] = None, **spec_kwargs):
    """Decorator form of :func:`register_builder`. Defaults the name to the function name."""

    def decorator(fn: Callable[..., nn.Module]) -> Callable[..., nn.Module]:
        register_builder(name or fn.__name__, fn, **spec_kwargs)
        return fn

    return decorator


def get_model_spec(name: str) -> ModelSpec:
    key = _normalize(name)
    if key not in _LOOKUP:
        matches = difflib.get_close_matches(key, list(_LOOKUP), n=5, cutoff=0.6)
        suggestions = list(dict.fromkeys(_LOOKUP[m] for m in matches))[:3]
        hint = f" Did you mean: {', '.join(suggestions)}?" if suggestions else ""
        raise ValueError(f"Unsupported model '{name}'.{hint} Run `python main.py --mode list` to see all models.")
    return _REGISTRY[_LOOKUP[key]]


def list_models(category: Optional[str] = None) -> List[ModelSpec]:
    specs = sorted(_REGISTRY.values(), key=lambda s: (list(CATEGORIES).index(s.category), s.family, s.name))
    if category is not None:
        specs = [s for s in specs if s.category == category]
    return specs


def resolve_input_size(spec: ModelSpec, requested: Optional[int], default: int) -> Tuple[int, bool]:
    """Return ``(img_size, adjusted)``.

    An explicitly requested size that the model cannot handle raises; the dataset default is
    rounded up to the closest size the model accepts.
    """
    divisor = max(1, spec.input_size_divisor)
    if requested is not None:
        if requested < spec.min_input_size or requested % divisor != 0:
            raise ValueError(
                f"Model '{spec.name}' needs input size >= {spec.min_input_size}"
                + (f" and divisible by {divisor}" if divisor > 1 else "")
                + f", got --resize {requested}"
            )
        return requested, False

    size = max(default, spec.min_input_size)
    size = -(-size // divisor) * divisor
    return size, size != default


def create_model(name: str, num_classes: int, img_size: int = 224) -> nn.Module:
    return get_model_spec(name).build(num_classes=num_classes, img_size=img_size)
