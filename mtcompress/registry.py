"""Method registry: name -> class, kind, family and paper reference.

Every compression method is a class decorated with ``@register_method``. The runner and the entry points
only talk to the registry, never to individual methods.
"""
import difflib
import re
from dataclasses import dataclass
from typing import Dict, List, Type

KINDS = {
    "prune": "Pruning",
    "quant": "Quantization",
    "pipeline": "Combined pipelines",
}


@dataclass(frozen=True)
class MethodSpec:
    name: str
    cls: Type
    kind: str
    family: str
    paper: str


_REGISTRY: Dict[str, MethodSpec] = {}


def _normalize(name: str) -> str:
    return re.sub(r"[\s\-.]", "_", str(name).strip().lower())


def register_method(name: str, *, kind: str, family: str, paper: str = ""):
    if kind not in KINDS:
        raise ValueError(f"Unknown kind '{kind}'. Expected one of {list(KINDS)}")

    def deco(cls):
        key = _normalize(name)
        if key in _REGISTRY:
            raise KeyError(f"Method '{key}' is already registered")
        cls.name = key
        cls.kind = kind
        _REGISTRY[key] = MethodSpec(key, cls, kind, family, paper)
        return cls

    return deco


def _ensure_loaded() -> None:
    # Importing the subpackages registers every method.
    from . import pipeline, prune, quant  # noqa: F401


def get_method(name: str) -> MethodSpec:
    _ensure_loaded()
    key = _normalize(name)
    if key not in _REGISTRY:
        close = difflib.get_close_matches(key, list(_REGISTRY), n=3)
        hint = f" Did you mean: {', '.join(close)}?" if close else ""
        raise KeyError(f"Unknown method '{name}'.{hint} Run `python compress.py --list`.")
    return _REGISTRY[key]


def list_methods(kind: str = None) -> List[MethodSpec]:
    _ensure_loaded()
    specs = [s for s in _REGISTRY.values() if kind is None or s.kind == kind]
    return sorted(specs, key=lambda s: (list(KINDS).index(s.kind), s.family, s.name))
