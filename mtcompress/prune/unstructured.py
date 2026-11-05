"""Unstructured pruning of a trained model: individual weights are zeroed.

Dense hardware does not run these models faster; the gain is storage (sparse encoding) and the
``sparse_macs`` a sparse kernel could skip. ``summary.json`` reports both next to the dense latency.
"""
import torch

from ..base import PruneMethod
from ..registry import register_method
from .masks import MaskSet, accumulate_grads, cubic_sparsity


class OneShotPrune(PruneMethod):
    """Score once, prune to ``sparsity``, fine-tune with the mask fixed."""

    hparams = {"sparsity": 0.5, "scope": "global"}

    def score(self, ctx, masks: MaskSet):
        raise NotImplementedError

    def apply(self, ctx):
        masks = MaskSet(ctx.model, self.hp["exclude"])
        scores = self.score(ctx, masks)
        masks.prune_by_scores(scores, self.hp["sparsity"], self.hp["scope"])
        ctx.info["after_prune"] = ctx.evaluate()
        ctx.log(f"{self.name}: sparsity {100 * masks.sparsity():.1f}% before fine-tuning: {ctx.info['after_prune']}")
        self.finetune(ctx, hooks=masks)
        masks.finalize(ctx.model, ctx.info)


@register_method("magnitude", kind="prune", family="Unstructured", paper="Han et al., 2015")
class Magnitude(OneShotPrune):
    def score(self, ctx, masks):
        return masks.scores(lambda n, w: w.detach().abs())


@register_method("random", kind="prune", family="Unstructured",
                 paper="Liu et al., 2022 (The Unreasonable Effectiveness of Random Pruning)")
class RandomPrune(OneShotPrune):
    hparams = {"seed": 0}

    def score(self, ctx, masks):
        g = torch.Generator(device=next(iter(masks.targets.values())).device).manual_seed(self.hp["seed"])
        return masks.scores(lambda n, w: torch.rand(w.shape, device=w.device, generator=g))


@register_method("obd", kind="prune", family="Unstructured",
                 paper="LeCun et al., 1990 (Optimal Brain Damage; diagonal Hessian ~ empirical Fisher)")
class OBD(OneShotPrune):
    """Saliency 0.5 * H_ii * w_i^2 with H_ii approximated by the mean squared gradient over the
    calibration set (empirical Fisher), the usual practical stand-in for the exact diagonal."""

    hparams = {"calib_batches": 8}

    def score(self, ctx, masks):
        fisher = {n: torch.zeros_like(w) for n, w in masks.targets.items()}

        def add(model):
            for n, w in masks.targets.items():
                if w.grad is not None:
                    fisher[n] += w.grad.detach() ** 2

        count = accumulate_grads(ctx.model, ctx.data.calib(32, limit=self.hp["calib_batches"]), add)
        return {n: 0.5 * fisher[n] / max(count, 1) * masks.targets[n].detach() ** 2 for n in fisher}


@register_method("gmp", kind="prune", family="Unstructured",
                 paper="Zhu & Gupta, 2017 (To prune, or not to prune)")
class GMP(PruneMethod):
    """Gradual magnitude pruning during fine-tuning on the cubic schedule."""

    hparams = {"sparsity": 0.8, "scope": "global", "start": 0.0, "end": 0.7, "update_every": 50, "ft_epochs": 10}

    def apply(self, ctx):
        masks = MaskSet(ctx.model, self.hp["exclude"])
        hp = self.hp

        class Hooks:
            def on_step_begin(_, step, total):
                if step % hp["update_every"] == 0 or step == int(hp["end"] * total):
                    target = cubic_sparsity(step, total, hp["sparsity"], 0.0, hp["start"], hp["end"])
                    if target > masks.sparsity() + 1e-6:
                        masks.prune_by_scores(masks.scores(lambda n, w: w.detach().abs()), target, hp["scope"])

            after_backward = staticmethod(masks.after_backward)
            after_step = staticmethod(masks.after_step)

        self.finetune(ctx, hooks=Hooks())
        if masks.sparsity() < hp["sparsity"] - 1e-3:  # very short runs never reach the end of the schedule
            masks.prune_by_scores(masks.scores(lambda n, w: w.detach().abs()), hp["sparsity"], hp["scope"])
        masks.finalize(ctx.model, ctx.info)


@register_method("movement", kind="prune", family="Unstructured",
                 paper="Sanh et al., 2020 (Movement Pruning)")
class Movement(PruneMethod):
    """Hard movement pruning: scores S accumulate -dL/dW * W during fine-tuning; on the cubic schedule the
    weights with the lowest S are masked. Weights that move towards zero are pruned even if large."""

    hparams = {"sparsity": 0.8, "scope": "global", "start": 0.1, "end": 0.7, "update_every": 50, "ft_epochs": 10}

    def apply(self, ctx):
        masks = MaskSet(ctx.model, self.hp["exclude"])
        hp = self.hp
        scores = {n: torch.zeros_like(w) for n, w in masks.targets.items()}

        class Hooks:
            def on_step_begin(_, step, total):
                if step % hp["update_every"] == 0 and step > hp["start"] * total:
                    target = cubic_sparsity(step, total, hp["sparsity"], 0.0, hp["start"], hp["end"])
                    if target > masks.sparsity() + 1e-6:
                        masks.prune_by_scores(scores, target, hp["scope"])

            def after_backward(_, model):
                for n, w in masks.targets.items():
                    if w.grad is not None:
                        scores[n] -= w.grad.detach() * w.detach()
                masks.after_backward(model)

            after_step = staticmethod(masks.after_step)

        self.finetune(ctx, hooks=Hooks())
        if masks.sparsity() < hp["sparsity"] - 1e-3:
            masks.prune_by_scores(scores, hp["sparsity"], hp["scope"])
        masks.finalize(ctx.model, ctx.info)
