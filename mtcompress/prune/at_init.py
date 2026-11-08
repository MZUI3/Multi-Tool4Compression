"""Pruning at initialisation: score a randomly initialised network, prune, then train the sparse network
from scratch for ``ft_epochs``.

With ``from_scratch: false`` the same criteria score the trained model instead, which makes them
comparable with the one-shot methods on a pretrained network.
"""
import copy

import torch
import torch.nn.functional as F

from ..base import PruneMethod
from ..registry import register_method
from .masks import MaskSet


class AtInit(PruneMethod):
    hparams = {"sparsity": 0.8, "from_scratch": True, "ft_epochs": 30, "ft_lr": 0.05, "ft_warmup_epochs": 1.0,
               "score_batches": 8}

    def score(self, ctx, masks: MaskSet):
        raise NotImplementedError

    def apply(self, ctx):
        if self.hp["from_scratch"]:
            if ctx.build_model is None:
                raise RuntimeError("from_scratch needs ctx.build_model")
            ctx.model = ctx.build_model()
        masks = MaskSet(ctx.model, self.hp["exclude"])
        self.prune(ctx, masks)
        ctx.info["after_prune"] = ctx.evaluate()
        ctx.log(f"{self.name}: sparsity {100 * masks.sparsity():.1f}%, training {self.hp['ft_epochs']} epochs")
        self.finetune(ctx, hooks=masks)
        masks.finalize(ctx.model, ctx.info)

    def prune(self, ctx, masks):
        masks.prune_by_scores(self.score(ctx, masks), self.hp["sparsity"], "global")


@register_method("random_init", kind="prune", family="At initialisation",
                 paper="Liu et al., 2022 (random pruning at initialisation, the baseline for SNIP/GraSP/SynFlow)")
class RandomInit(AtInit):
    def score(self, ctx, masks):
        g = torch.Generator(device=ctx.device).manual_seed(0)
        return masks.scores(lambda n, w: torch.rand(w.shape, device=w.device, generator=g))


@register_method("snip", kind="prune", family="At initialisation", paper="Lee et al., 2019 (SNIP)")
class SNIP(AtInit):
    """Connection sensitivity |dL/dw * w| summed over a few batches."""

    def score(self, ctx, masks):
        model = ctx.model
        model.train()
        scores = {n: torch.zeros_like(w) for n, w in masks.targets.items()}
        for x, y in ctx.data.calib(32, limit=self.hp["score_batches"]):
            model.zero_grad(set_to_none=True)
            F.cross_entropy(model(x), y).backward()
            for n, w in masks.targets.items():
                scores[n] += (w.grad * w).detach().abs()
        model.zero_grad(set_to_none=True)
        return scores


@register_method("grasp", kind="prune", family="At initialisation", paper="Wang et al., 2020 (GraSP)")
class GraSP(AtInit):
    """Keep the weights whose removal would most reduce gradient flow: S = -w * (H g); the weights with
    the largest S are removed. Hg comes from double back-propagation with temperature-scaled logits."""

    hparams = {"temperature": 200.0}

    def score(self, ctx, masks):
        model = ctx.model
        model.train()
        weights = list(masks.targets.values())
        names = list(masks.targets)
        T = self.hp["temperature"]
        batches = list(ctx.data.calib(32, limit=self.hp["score_batches"]))
        grad_w = None
        for x, y in batches:
            g = torch.autograd.grad(F.cross_entropy(model(x) / T, y), weights)
            grad_w = [gi.detach() for gi in g] if grad_w is None else [a + b.detach() for a, b in zip(grad_w, g)]
        hg = [torch.zeros_like(w) for w in weights]
        for x, y in batches:
            g = torch.autograd.grad(F.cross_entropy(model(x) / T, y), weights, create_graph=True)
            z = sum((gi * gw).sum() for gi, gw in zip(g, grad_w))
            for acc, h in zip(hg, torch.autograd.grad(z, weights)):
                acc += h.detach()
        model.zero_grad(set_to_none=True)
        # remove largest -w*Hg  ==  keep largest w*Hg
        return {n: (w.detach() * h) for n, w, h in zip(names, weights, hg)}


@register_method("synflow", kind="prune", family="At initialisation",
                 paper="Tanaka et al., 2020 (SynFlow, data-free)")
class SynFlow(AtInit):
    """Iterative synaptic flow: all parameters made positive, an all-ones input, R = sum(output) and
    score |dR/dw * w|, pruned on an exponential schedule over ``iterations`` rounds (float64)."""

    hparams = {"iterations": 100}

    def prune(self, ctx, masks):
        model = ctx.model
        probe = copy.deepcopy(model).double().eval()
        with torch.no_grad():
            for p in probe.parameters():
                p.abs_()
        probe_targets = dict((n, w) for n, w in MaskSet(probe, self.hp["exclude"]).targets.items())
        base = {n: w.detach().clone() for n, w in probe_targets.items()}
        x = torch.ones((1, 3, ctx.data.img_size, ctx.data.img_size), dtype=torch.float64, device=ctx.device)
        n_iter = self.hp["iterations"]
        density = 1 - self.hp["sparsity"]
        for k in range(1, n_iter + 1):
            with torch.no_grad():
                for n, w in probe_targets.items():
                    w.copy_(base[n] * masks.masks[n])
            probe.zero_grad(set_to_none=True)
            probe(x).sum().backward()
            scores = {n: (w.grad * w).detach().abs().float() for n, w in probe_targets.items()}
            masks.prune_by_scores(scores, 1 - density ** (k / n_iter), "global")
        del probe
