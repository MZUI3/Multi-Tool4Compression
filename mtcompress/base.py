"""Base class for compression methods and the context they run in."""
import copy
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import torch
import torch.nn as nn

from .train import evaluate, fit

# Fine-tuning settings every method understands. ``ft_epochs: 0`` turns fine-tuning off.
FINETUNE_DEFAULTS = {
    "ft_epochs": 5,
    "ft_lr": 0.01,
    "ft_optimizer": "sgd",
    "ft_weight_decay": 5e-4,
    "ft_batch_size": 128,
    "ft_warmup_epochs": 0.0,
    "kd_alpha": 0.0,  # > 0 distils from the uncompressed model (Hinton et al., 2015)
    "kd_temperature": 4.0,
}


@dataclass
class Context:
    """Everything a method may use. ``model`` is replaced or changed in place by ``Method.apply``."""

    model: nn.Module
    data: Any
    device: torch.device
    amp: bool = False
    teacher: Optional[nn.Module] = None  # frozen copy of the model before compression
    log: Callable = print
    history: List[dict] = field(default_factory=list)
    limit_batches: int = 0  # smoke tests: cap batches per epoch and per calibration pass
    info: Dict[str, Any] = field(default_factory=dict)  # method-specific results for summary.json
    build_model: Optional[Callable[[], nn.Module]] = None  # fresh, randomly initialised model on the device

    def evaluate(self, model: nn.Module = None) -> Dict[str, float]:
        return evaluate(model or self.model, self.data.val(limit=self.limit_batches), self.amp)

    def calib(self, batch_size: int = 128):
        return self.data.calib(batch_size, limit=self.limit_batches)

    def train_batches(self, batch_size: int = 128):
        return self.data.train(batch_size, limit=self.limit_batches)


def _merged_defaults(cls) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for klass in reversed(cls.__mro__):
        out.update(getattr(klass, "__dict__", {}).get("hparams", {}))
    return out


class Method:
    """A compression method. Subclasses declare ``hparams`` (merged along the class hierarchy) and
    implement ``apply(ctx)``. Unknown hparams are rejected with the list of valid keys."""

    name = "method"
    kind = "prune"
    amp_ok = True  # fake quantisation runs in float32
    hparams: Dict[str, Any] = {}

    def __init__(self, hparams: Optional[Dict[str, Any]] = None):
        defaults = self.default_hparams()
        given = dict(hparams or {})
        unknown = sorted(set(given) - set(defaults))
        if unknown:
            raise KeyError(f"{self.name}: unknown hparams {unknown}. Valid: {sorted(defaults)}")
        self.hp = {**defaults, **given}

    @classmethod
    def default_hparams(cls) -> Dict[str, Any]:
        return copy.deepcopy(_merged_defaults(cls))

    def apply(self, ctx: Context) -> None:
        raise NotImplementedError

    # ------------------------------------------------------------------ shared helpers
    def finetune(self, ctx: Context, hooks=None, epochs: float = None, lr: float = None, extra_groups=None,
                 tag: str = None, model: nn.Module = None) -> List[dict]:
        hp = self.hp
        epochs = hp["ft_epochs"] if epochs is None else epochs
        hist = fit(
            model or ctx.model,
            ctx.data,
            epochs=epochs,
            lr=hp["ft_lr"] if lr is None else lr,
            optimizer=hp["ft_optimizer"],
            weight_decay=hp["ft_weight_decay"],
            warmup_epochs=hp["ft_warmup_epochs"],
            batch_size=hp["ft_batch_size"],
            amp=ctx.amp and self.amp_ok,
            teacher=ctx.teacher if hp["kd_alpha"] > 0 else None,
            kd_alpha=hp["kd_alpha"],
            kd_temperature=hp["kd_temperature"],
            hooks=hooks,
            extra_groups=extra_groups,
            limit_batches=ctx.limit_batches,
            log=ctx.log,
            tag=tag or f"{self.name}/finetune",
        )
        ctx.history.extend(hist)
        return hist


class PruneMethod(Method):
    kind = "prune"
    hparams = {
        **FINETUNE_DEFAULTS,
        # Layers left dense. Names are matched as prefixes of the module name; "first" and "last" mean the
        # first conv/linear and the classifier, which is the convention of most pruning papers.
        "exclude": ["first", "last"],
    }


class QuantMethod(Method):
    kind = "quant"
    amp_ok = False
    hparams = {
        **FINETUNE_DEFAULTS,
        "ft_epochs": 0,
        "w_bits": 8,
        "a_bits": 8,
        # Layers kept in float. 8-bit methods quantise everything; low-bit methods keep the first and last
        # layer in float, the usual convention.
        "exclude": [],
    }


def layer_names(model: nn.Module, types=(nn.Conv2d, nn.Linear)) -> List[str]:
    """Module names of conv/linear layers in registration order (MHA out_proj excluded)."""
    return [n for n, m in model.named_modules()
            if isinstance(m, types) and type(m).__name__ != "NonDynamicallyQuantizableLinear"]


def resolve_exclude(model: nn.Module, exclude) -> set:
    names = layer_names(model)
    out = set()
    for item in exclude or []:
        if item == "first" and names:
            out.add(names[0])
        elif item == "last" and names:
            linears = layer_names(model, (nn.Linear,))
            out.add(linears[-1] if linears else names[-1])
        else:
            out.update(n for n in names if n == item or n.startswith(item + "."))
    return out
