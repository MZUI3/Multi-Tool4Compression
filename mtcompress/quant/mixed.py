"""Mixed-precision weight quantisation: each layer gets its own weight bit width under a size budget.

Both methods score every (layer, bits) pair, then solve the same knapsack exactly by dynamic
programming: minimise the summed score subject to ``sum(n_i * b_i) <= avg_bits * sum(n_i)``.

- ``mixed_sensitivity``: the score is the calibration-loss increase when only that layer is quantised.
- ``hawq``: HAWQ-V2's Omega_i = Tr(H_i) / n_i * ||Q_b(W_i) - W_i||^2, with the Hessian trace of each
  layer's weights estimated by Hutchinson's method. HAWQ-V3 solves an ILP; the DP gives the same optimum
  for this single-constraint problem.

Activations are quantised to ``a_bits`` (8). A short fine-tune with weights quantised on their
current range follows (``ft_epochs``).
"""
import numpy as np
import torch
import torch.nn.functional as F

from ..registry import register_method
from ..train import evaluate
from .core import Quantizer, calibrate, quant_layers, record_meta
from .ptq import PTQ
from .qat import DynamicWeight


def allocate(scores, sizes, candidates, avg_bits, units=20000):
    """scores[i][j]: cost of bits candidates[j] for layer i (size n_i). Returns bits per layer."""
    total = sum(sizes)
    budget = avg_bits * total
    unit = max(1.0, budget / units)
    cap = int(budget // unit)
    INF = float("inf")
    dp = np.full(cap + 1, INF)
    dp[0] = 0.0
    choice = []
    for i, n in enumerate(sizes):
        new = np.full(cap + 1, INF)
        arg = np.full(cap + 1, -1, dtype=np.int64)
        for j, b in enumerate(candidates):
            w = int(np.ceil(n * b / unit))
            if w > cap:
                continue
            shifted = np.full(cap + 1, INF)
            shifted[w:] = dp[: cap + 1 - w] + scores[i][j]
            better = shifted < new
            new[better] = shifted[better]
            arg[better] = j
        choice.append(arg)
        dp = new
    if not np.isfinite(dp).any():
        raise ValueError("budget too small for the smallest bit width")
    c = int(np.nanargmin(dp))
    bits = [0] * len(sizes)
    for i in range(len(sizes) - 1, -1, -1):
        j = int(choice[i][c])
        bits[i] = candidates[j]
        c -= int(np.ceil(sizes[i] * candidates[j] / unit))
    return bits


class Mixed(PTQ):
    hparams = {"candidates": [2, 4, 8], "avg_bits": 4.0, "a_bits": 8, "w_observer": "mse", "ft_epochs": 3,
               "ft_lr": 0.005, "score_batches": 4}

    def scores(self, ctx, layers):
        raise NotImplementedError

    def apply(self, ctx):
        hp = self.hp
        self.quantize(ctx)  # 8-bit everywhere, activations calibrated
        layers = quant_layers(ctx.model)
        for _, m in layers:
            m.aq.mode = "off"
        sizes = [m.weight.numel() for _, m in layers]
        scores = self.scores(ctx, layers)
        bits = allocate(scores, sizes, hp["candidates"], hp["avg_bits"])
        for (name, m), b in zip(layers, bits):
            q = Quantizer(b, symmetric=True, per_channel=True, observer=hp["w_observer"]).to(m.weight.device)
            q.init_from(m.weight)
            m.wq = q
            m.aq.mode = "quant"
        calibrate(ctx.model, ctx.data.calib(128, limit=hp["calib_batches"]))
        ctx.info["bits"] = {n: b for (n, _), b in zip(layers, bits)}
        ctx.info["avg_weight_bits"] = round(sum(s * b for s, b in zip(sizes, bits)) / sum(sizes), 3)
        ctx.info["scores"] = {n: [round(float(v), 6) for v in s] for (n, _), s in zip(layers, scores)}
        ctx.info["after_quant"] = ctx.evaluate()
        ctx.log(f"{self.name}: avg weight bits {ctx.info['avg_weight_bits']} -> {ctx.info['after_quant']}")
        if hp["ft_epochs"] > 0:
            for _, m in layers:
                old = m.wq
                m.wq = DynamicWeight(old.bits, per_channel=True).to(m.weight.device)
                m.wq.init_from(m.weight)
            self.finetune(ctx)
        record_meta(ctx.model)


@register_method("mixed_sensitivity", kind="quant", family="Mixed precision",
                 paper="Hubara et al., 2021 (measured per-layer loss + integer programming)")
class MixedSensitivity(Mixed):
    def scores(self, ctx, layers):
        calib = list(ctx.data.calib(128, limit=self.hp["score_batches"]))
        for _, m in layers:
            m.wq.mode = "off"
        base = evaluate(ctx.model, calib)["loss"]
        out = []
        for name, m in layers:
            row = []
            for b in self.hp["candidates"]:
                q = Quantizer(b, symmetric=True, per_channel=True, observer=self.hp["w_observer"]).to(m.weight.device)
                q.init_from(m.weight)
                m.wq = q
                row.append(max(evaluate(ctx.model, calib)["loss"] - base, 0.0))
                m.wq.mode = "off"
            out.append(row)
        return out


@register_method("hawq", kind="quant", family="Mixed precision",
                 paper="Dong et al., 2020 (HAWQ-V2: Hessian trace); Yao et al., 2021 (HAWQ-V3)")
class HAWQ(Mixed):
    hparams = {"hutchinson_samples": 32}

    def scores(self, ctx, layers):
        model = ctx.model
        for _, m in layers:
            m.wq.mode = "off"
        weights = [m.weight for _, m in layers]
        traces = [0.0] * len(weights)
        model.eval()
        batches = list(ctx.data.calib(128, limit=max(1, self.hp["score_batches"] // 2)))
        for x, y in batches:
            loss = F.cross_entropy(model(x), y)
            grads = torch.autograd.grad(loss, weights, create_graph=True)
            for _ in range(self.hp["hutchinson_samples"]):
                vs = [torch.randint_like(w, 2) * 2 - 1 for w in weights]
                hv = torch.autograd.grad(grads, weights, grad_outputs=vs, retain_graph=True)
                for i, (h, v) in enumerate(zip(hv, vs)):
                    traces[i] += float((h * v).sum())
        n = len(batches) * self.hp["hutchinson_samples"]
        traces = [t / n for t in traces]
        ctx.info["hessian_trace"] = {name: round(t, 6) for (name, _), t in zip(layers, traces)}
        out = []
        for (name, m), tr in zip(layers, traces):
            row = []
            for b in self.hp["candidates"]:
                q = Quantizer(b, symmetric=True, per_channel=True, observer=self.hp["w_observer"]).to(m.weight.device)
                q.init_from(m.weight)
                with torch.no_grad():
                    err = float((q(m.weight) - m.weight).pow(2).sum())
                row.append(max(tr, 0.0) / m.weight.numel() * err)
            out.append(row)
        return out
