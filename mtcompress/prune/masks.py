"""Weight masks for unstructured and semi-structured pruning.

A ``MaskSet`` holds one boolean mask per prunable weight. During fine-tuning it is passed to ``fit`` as
the hooks object: gradients of pruned weights are zeroed after ``backward`` and the weights are re-masked
after every optimiser step (momentum and weight decay would otherwise revive them).
"""
from typing import Callable, Dict, Optional

import torch
import torch.nn as nn

from ..base import resolve_exclude
from ..measure.profile import weight_tensors


class MaskSet:
    def __init__(self, model: nn.Module, exclude=("first", "last")):
        skip = resolve_exclude(model, exclude)
        self.targets: Dict[str, torch.Tensor] = {}
        for name, module, w in weight_tensors(model):
            mod_name = name.rsplit(".", 1)[0]
            if mod_name in skip:
                continue
            self.targets[name] = w
        self.masks: Dict[str, torch.Tensor] = {n: torch.ones_like(w, dtype=torch.bool) for n, w in self.targets.items()}

    # ---------------------------------------------------------------- state
    @property
    def total(self) -> int:
        return sum(m.numel() for m in self.masks.values())

    def sparsity(self) -> float:
        kept = sum(int(m.sum()) for m in self.masks.values())
        return 1.0 - kept / max(self.total, 1)

    def layer_sparsity(self) -> Dict[str, float]:
        return {n: 1.0 - float(m.float().mean()) for n, m in self.masks.items()}

    @torch.no_grad()
    def apply(self) -> None:
        for n, w in self.targets.items():
            w.mul_(self.masks[n])

    # ---------------------------------------------------------------- fit() hooks
    def after_backward(self, model) -> None:
        for n, w in self.targets.items():
            if w.grad is not None:
                w.grad.mul_(self.masks[n])

    def after_step(self, model) -> None:
        self.apply()

    # ---------------------------------------------------------------- selecting weights
    @torch.no_grad()
    def prune_by_scores(self, scores: Dict[str, torch.Tensor], sparsity: float, scope: str = "global",
                        only_unpruned: bool = True) -> None:
        """Keep the highest-scoring weights so that ``sparsity`` of all target weights are zero.

        ``scope='global'`` ranks all layers together (one threshold), ``'layer'`` prunes every layer to the
        same sparsity. Already pruned weights stay pruned. At least one weight per layer is kept."""
        sparsity = float(min(max(sparsity, 0.0), 1.0))
        if scope == "layer":
            for n, s in scores.items():
                s = s.float().clone()
                if only_unpruned:
                    s[~self.masks[n]] = -float("inf")
                k = int(round(s.numel() * (1 - sparsity)))
                k = max(k, 1)
                idx = torch.topk(s.flatten(), k).indices
                mask = torch.zeros(s.numel(), dtype=torch.bool, device=s.device)
                mask[idx] = True
                self.masks[n] = mask.view_as(s) & (self.masks[n] if only_unpruned else True)
        elif scope == "global":
            flat = []
            for n, s in scores.items():
                s = s.float().flatten().clone()
                if only_unpruned:
                    s[~self.masks[n].flatten()] = -float("inf")
                flat.append(s)
            allscores = torch.cat(flat)
            k = max(int(round(allscores.numel() * (1 - sparsity))), 1)
            threshold = torch.topk(allscores, k).values.min()
            for n, s in scores.items():
                s = s.float().clone()
                if only_unpruned:
                    s[~self.masks[n]] = -float("inf")
                mask = s >= threshold
                if not mask.any():
                    mask.view(-1)[s.flatten().argmax()] = True
                self.masks[n] = mask
        else:
            raise ValueError(f"scope must be global or layer, got '{scope}'")
        self.apply()

    def scores(self, fn: Callable[[str, torch.Tensor], torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {n: fn(n, w) for n, w in self.targets.items()}

    def finalize(self, model: nn.Module, info: Optional[dict] = None) -> dict:
        self.apply()
        out = {"target_sparsity_actual": round(100 * self.sparsity(), 2),
               "layer_sparsity": {n: round(100 * s, 2) for n, s in self.layer_sparsity().items()}}
        if info is not None:
            info.update(out)
        return out


def cubic_sparsity(step: int, total: int, final: float, initial: float = 0.0, start: float = 0.0,
                   end: float = 0.8) -> float:
    """Zhu & Gupta (2017): s_t = s_f + (s_i - s_f) (1 - (t - t0) / (n dt))^3 between ``start`` and ``end``
    (fractions of training)."""
    t0, t1 = start * total, end * total
    if step <= t0:
        return initial
    if step >= t1:
        return final
    frac = (step - t0) / max(t1 - t0, 1)
    return final + (initial - final) * (1 - frac) ** 3


def accumulate_grads(model: nn.Module, batches, fn: Callable[[nn.Module], None], loss_fn=None,
                     amp: bool = False) -> int:
    """Run forward/backward over ``batches`` and call ``fn(model)`` after each backward (gradients are
    zeroed before each batch). Returns the number of batches."""
    loss_fn = loss_fn or nn.functional.cross_entropy
    was_training = model.training
    model.eval()  # fixed BN statistics: scores should not depend on batch composition
    n = 0
    for x, y in batches:
        model.zero_grad(set_to_none=True)
        loss = loss_fn(model(x), y)
        loss.backward()
        fn(model)
        n += 1
    model.zero_grad(set_to_none=True)
    model.train(was_training)
    return n
