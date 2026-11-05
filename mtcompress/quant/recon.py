"""Reconstruction-based PTQ: AdaRound (layer-wise), BRECQ (block-wise), QDrop (block-wise with dropped
activation quantisation).

For every unit, in order, the inputs come from the partly quantised model (earlier units already
final) and the targets from a float copy of the model fed with float inputs. Each weight learns whether
to round up or down through a rectified sigmoid h(V); a regulariser pushes h(V) to 0 or 1 as its
exponent beta is annealed from ``b_start`` to ``b_end``. At the end the rounding is fixed and written into
the weight, so the model is again a plain fake-quantised model (and exports to int8 like any other).

BRECQ and QDrop also learn the activation step sizes of the unit during reconstruction. BRECQ weights
the output error with the squared gradient of the loss (Fisher) when ``fisher: true``; the default is
plain MSE. Iteration counts are far below the papers' 20k per unit, to keep runs short.
"""
import copy
import math
import random
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..registry import register_method
from .core import QUANT_TYPES, Quantizer, calibrate, quant_layers
from .ptq import PTQ

ZETA, GAMMA = 1.1, -0.1


class AdaRoundWeight(nn.Module):
    """Weight quantiser with learnable rounding, built from a calibrated uniform quantiser."""

    def __init__(self, base: Quantizer, weight: torch.Tensor):
        super().__init__()
        self.base = base
        s = base._shape(weight, base.scale)
        floor = torch.floor(weight.detach() / s)
        rest = weight.detach() / s - floor
        self.register_buffer("floor", floor)
        self.alpha = nn.Parameter(-torch.log((ZETA - GAMMA) / (rest - GAMMA).clamp(1e-4, ZETA - GAMMA - 1e-4) - 1))
        self.hard = False

    def h(self):
        return torch.clamp(torch.sigmoid(self.alpha) * (ZETA - GAMMA) + GAMMA, 0, 1)

    def forward(self, w):
        b = self.base
        s, z = b._shape(w, b.scale), b._shape(w, b.zero_point)
        r = (self.alpha >= 0).float() if self.hard else self.h()
        q = torch.clamp(self.floor + r + z, b.qmin, b.qmax)
        return (q - z) * s

    def reg(self, beta):
        return (1 - (2 * self.h() - 1).abs().pow(beta)).sum()


def find_units(model: nn.Module, example: torch.Tensor, max_layers: int) -> List[Tuple[str, nn.Module]]:
    """Split the model into called sub-modules with at most ``max_layers`` quantised layers each."""

    def n_layers(m):
        return sum(1 for x in m.modules() if isinstance(x, QUANT_TYPES))

    def called(mods):
        fired = set()
        hooks = [m.register_forward_hook(lambda mod, i, o, key=k: fired.add(key)) for k, m in mods]
        with torch.no_grad():
            model.eval()(example)
        for h in hooks:
            h.remove()
        return fired

    frontier = [(n, m) for n, m in model.named_children()]
    units = []
    for _ in range(10):
        nxt = []
        candidates = [(n, m) for n, m in frontier if n_layers(m) > 0]
        fired = called(candidates)
        for n, m in candidates:
            if n in fired and (n_layers(m) <= max_layers or isinstance(m, QUANT_TYPES)):
                units.append((n, m))
            else:
                nxt += [(f"{n}.{c}", cm) for c, cm in m.named_children()]
        frontier = nxt
        if not frontier:
            break
    order = {n: i for i, (n, _) in enumerate(model.named_modules())}
    return sorted(units, key=lambda u: order[u[0]])


def _capture(model, unit, batches, want="input"):
    store = []

    def hook(m, args, out):
        if want == "input":
            store.append(tuple(a.detach() if torch.is_tensor(a) else a for a in args))
        else:
            store.append(out[0].detach() if isinstance(out, tuple) else out.detach())

    h = unit.register_forward_hook(hook)
    with torch.no_grad():
        model.eval()
        for x, _ in batches:
            model(x)
    h.remove()
    return store


class Reconstruct(PTQ):
    hparams = {"w_bits": 4, "a_bits": 8, "exclude": ["first", "last"], "w_observer": "mse",
               "iters": 1000, "batch": 32, "reg_weight": 0.01, "b_start": 20.0, "b_end": 2.0, "warmup": 0.2,
               "lr": 1e-3, "act_lr": 4e-5, "calib_batches": 8, "max_layers": 1, "learn_act": False,
               "drop_prob": 0.0, "fisher": False}

    def after_weights(self, ctx):
        self._fp = copy.deepcopy(ctx.model)
        for _, m in quant_layers(self._fp):
            m.wq.mode = "off"
            m.aq.mode = "off"

    def after_calibration(self, ctx):
        model = ctx.model
        hp = self.hp
        example = next(iter(ctx.data.calib(2)))[0]
        batches = list(ctx.data.calib(128, limit=hp["calib_batches"]))
        if not hp["learn_act"]:  # AdaRound: weights only, activations calibrated afterwards
            for _, m in quant_layers(model):
                m.aq.mode = "off"
        units = find_units(model, example, hp["max_layers"])
        fp_mods = dict(self._fp.named_modules())
        ctx.info["units"] = [n for n, _ in units]
        losses = {}
        for name, unit in units:
            inputs = _capture(model, unit, batches, "input")
            targets = _capture(self._fp, fp_mods[name], batches, "output")
            grads = self._fisher(ctx, name, batches) if hp["fisher"] else None
            losses[name] = self._optimise(unit, inputs, targets, grads)
            ctx.log(f"{self.name}: {name} recon loss {losses[name][0]:.4g} -> {losses[name][1]:.4g}")
        ctx.info["recon_loss"] = {k: [round(a, 6), round(b, 6)] for k, (a, b) in losses.items()}
        if not hp["learn_act"]:
            calibrate(model, ctx.data.calib(128, limit=hp["calib_batches"]))
        del self._fp

    def _fisher(self, ctx, name, batches):
        unit = dict(self._fp.named_modules())[name]
        store = []

        def hook(m, i, o):
            o = o[0] if isinstance(o, tuple) else o
            o.retain_grad()
            store.append(o)

        h = unit.register_forward_hook(hook)
        grads = []
        for x, y in batches:
            store.clear()
            out = self._fp(x)
            F.cross_entropy(out, y).backward()
            grads.append(store[0].grad.detach() ** 2)
        h.remove()
        self._fp.zero_grad(set_to_none=True)
        return grads

    def _optimise(self, unit, inputs, targets, grads=None):
        hp = self.hp
        layers = [m for m in unit.modules() if isinstance(m, QUANT_TYPES)]
        if isinstance(unit, QUANT_TYPES):
            layers = [unit]
        ada = []
        for layer in layers:
            if isinstance(layer.wq, Quantizer) and layer.wq.mode == "quant":
                layer.wq = AdaRoundWeight(layer.wq, layer.weight)
                ada.append(layer)
        params = [{"params": [l.wq.alpha for l in ada], "lr": hp["lr"]}]
        act_q = []
        if hp["learn_act"]:
            for layer in layers:
                q = layer.aq
                if isinstance(q, Quantizer) and q.mode == "quant":
                    q.scale = nn.Parameter(q.scale.detach().clone())
                    q.drop_prob = hp["drop_prob"]
                    q.train()
                    act_q.append(q)
            if act_q:
                params.append({"params": [q.scale for q in act_q], "lr": hp["act_lr"]})
        for p in unit.parameters():
            p.requires_grad_(False)
        for l in ada:
            l.wq.alpha.requires_grad_(True)
        for q in act_q:
            q.scale.requires_grad_(True)
        opt = torch.optim.Adam(params)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, hp["iters"])

        def loss_at(i, sub):
            args = tuple(a[sub] if torch.is_tensor(a) and a.dim() > 0 else a for a in inputs[i])
            out = unit(*args)
            out = out[0] if isinstance(out, tuple) else out
            err = (out - targets[i][sub]) ** 2
            if grads is not None:
                err = err * grads[i][sub]
            return err.flatten(1).sum(1).mean()

        with torch.no_grad():
            first = sum(loss_at(i, slice(None)).item() for i in range(len(inputs))) / len(inputs)
        warm = int(hp["iters"] * hp["warmup"])
        for it in range(hp["iters"]):
            i = random.randrange(len(inputs))
            n = inputs[i][0].shape[0]
            sub = torch.randint(0, n, (min(hp["batch"], n),), device=targets[i].device)
            loss = loss_at(i, sub)
            if it >= warm and ada:
                frac = (it - warm) / max(1, hp["iters"] - warm)
                beta = hp["b_end"] + 0.5 * (hp["b_start"] - hp["b_end"]) * (1 + math.cos(math.pi * frac))
                loss = loss + hp["reg_weight"] * sum(l.wq.reg(beta) for l in ada)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
        # Fix the rounding into the weights and go back to plain quantisers.
        with torch.no_grad():
            for l in ada:
                l.wq.hard = True
                w_q = l.wq(l.weight)
                base = l.wq.base
                l.weight.copy_(w_q)
                l.wq = base
            for q in act_q:
                value = q._parameters.pop("scale").detach().clone()
                q.register_buffer("scale", value)
                q.drop_prob = 0.0
                q.eval()
            for p in unit.parameters():
                p.requires_grad_(True)
            last = sum(loss_at(i, slice(None)).item() for i in range(len(inputs))) / len(inputs)
        return first, last


@register_method("adaround", kind="quant", family="PTQ reconstruction", paper="Nagel et al., 2020 (AdaRound)")
class AdaRound(Reconstruct):
    pass


@register_method("brecq", kind="quant", family="PTQ reconstruction", paper="Li et al., 2021 (BRECQ)")
class BRECQ(Reconstruct):
    hparams = {"max_layers": 8, "learn_act": True}


@register_method("qdrop", kind="quant", family="PTQ reconstruction", paper="Wei et al., 2022 (QDrop)")
class QDrop(Reconstruct):
    hparams = {"max_layers": 8, "learn_act": True, "drop_prob": 0.5, "a_bits": 4}
