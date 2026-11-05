"""Graph helpers for PTQ: BatchNorm folding, cross-layer equalisation and empirical bias correction.

``trace_chains`` runs one forward pass and records, for every leaf module call, which module produced its
input and whether that tensor has any other consumer. Folding and equalisation only touch pairs whose
intermediate tensor has a single consumer.
"""
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.overrides import TorchFunctionMode

from ..nn.attention import SplitAttention

HOMOGENEOUS = (nn.ReLU, nn.LeakyReLU, nn.Identity, nn.PReLU)


def _leaf(m):
    return isinstance(m, SplitAttention) or len(list(m.children())) == 0


class _Recorder(TorchFunctionMode):
    def __init__(self, uses):
        super().__init__()
        self.uses = uses
        self.depth = [0]

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if self.depth[0] == 0:
            for a in list(args) + list(kwargs.values()):
                for t in (a if isinstance(a, (list, tuple)) else [a]):
                    if isinstance(t, torch.Tensor):
                        self.uses[id(t)] = self.uses.get(id(t), 0) + 1
        return func(*args, **kwargs)


@torch.no_grad()
def trace_chains(model: nn.Module, example: torch.Tensor):
    """Returns (calls, producer, uses, transparent): calls = [(module, input id, output id)] in execution
    order, producer = {tensor id: module}, uses = {tensor id: number of consumers}, transparent =
    {tensor id: [modules that returned the tensor itself]} (in-place activations, Identity)."""
    calls, producer, uses, keep, transparent = [], {}, {}, [], {}
    rec = _Recorder(uses)
    hooks = []

    def pre(m, inp):
        rec.depth[0] += 1

    def post(m, inp, out):
        rec.depth[0] -= 1
        if rec.depth[0] == 0 and inp and isinstance(inp[0], torch.Tensor) and isinstance(out, torch.Tensor):
            if out is inp[0]:  # in-place activation or Identity: transparent, not a new tensor
                transparent.setdefault(id(out), []).append(m)
                return
            uses[id(inp[0])] = uses.get(id(inp[0]), 0) + 1
            producer[id(out)] = m
            calls.append((m, id(inp[0]), id(out)))
            keep.extend([inp[0], out])

    for m in model.modules():
        if _leaf(m) and m is not model:
            hooks.append(m.register_forward_pre_hook(pre))
            hooks.append(m.register_forward_hook(post))
    was = model.training
    model.eval()
    try:
        with rec:
            out = model(example)
    finally:
        for h in hooks:
            h.remove()
        model.train(was)
    uses[id(out)] = uses.get(id(out), 0) + 1
    return calls, producer, uses, transparent


def _replace(model: nn.Module, target: nn.Module, new: nn.Module) -> None:
    for name, m in model.named_modules():
        for k, child in m._modules.items():
            if child is target:
                m._modules[k] = new


@torch.no_grad()
def fold_bn(model: nn.Module, example: torch.Tensor) -> int:
    """Fold BatchNorm2d into the preceding Conv2d where the conv output feeds only that BatchNorm."""
    calls, producer, uses, transparent = trace_chains(model, example)
    folded = 0
    for m, in_id, _ in calls:
        if not isinstance(m, nn.BatchNorm2d):
            continue
        conv = producer.get(in_id)
        if not isinstance(conv, nn.Conv2d) or uses.get(in_id, 0) != 1:
            continue
        std = (m.running_var + m.eps).sqrt()
        gamma = m.weight if m.weight is not None else torch.ones_like(std)
        beta = m.bias if m.bias is not None else torch.zeros_like(std)
        factor = gamma / std
        conv.weight.mul_(factor.view(-1, 1, 1, 1))
        bias = conv.bias if conv.bias is not None else torch.zeros_like(m.running_mean)
        new_bias = (bias - m.running_mean) * factor + beta
        conv.bias = nn.Parameter(new_bias)
        _replace(model, m, nn.Identity())
        folded += 1
    return folded


@torch.no_grad()
def equalization_pairs(model: nn.Module, example: torch.Tensor) -> List[Tuple[nn.Conv2d, list, nn.Module]]:
    """(conv A, activations, conv B) with A -> [ReLU-like / Identity] -> B and single-use tensors on the way."""
    calls, producer, uses, transparent = trace_chains(model, example)
    input_of = {o: i for _, i, o in calls}

    def is_conv(m):
        return isinstance(m, nn.Conv2d) and (m.groups == 1 or m.groups == m.in_channels == m.out_channels)

    pairs = []
    for m, in_id, _ in calls:
        if not is_conv(m):
            continue
        cur, acts, ok = in_id, [], True
        for _ in range(8):
            mods = transparent.get(cur, [])
            if not all(isinstance(t, HOMOGENEOUS) for t in mods):
                ok = False
                break
            acts += mods
            prev = producer.get(cur)
            if prev is None or uses.get(cur, 0) != 1:
                ok = False
                break
            if is_conv(prev):
                break
            if isinstance(prev, HOMOGENEOUS):
                acts.append(prev)
                cur = input_of[cur]
                continue
            ok = False
            break
        else:
            ok = False
        if ok:
            pairs.append((producer[cur], acts, m))
    return pairs


@torch.no_grad()
def cross_layer_equalize(model: nn.Module, example: torch.Tensor, iterations: int = 10) -> int:
    """Nagel et al., 2019 (DFQ): rescale channel i of A by 1/s_i and the matching input channel of B by
    s_i, with s_i = sqrt(r1_i / r2_i), so both get the same per-channel range. Valid through ReLU-like
    (positively homogeneous) activations only."""
    pairs = equalization_pairs(model, example)
    for _ in range(iterations):
        for a, _, b in pairs:
            wa = a.weight.reshape(a.weight.shape[0], -1)
            r1 = wa.abs().max(1).values
            if b.groups == 1:
                r2 = b.weight.abs().transpose(0, 1).reshape(b.weight.shape[1], -1).max(1).values
            else:
                r2 = b.weight.abs().reshape(b.weight.shape[0], -1).max(1).values
            s = torch.sqrt(r1 / r2.clamp_min(1e-12)).clamp(1e-4, 1e4)
            s = torch.where((r1 > 0) & (r2 > 0), s, torch.ones_like(s))
            a.weight.div_(s.view(-1, 1, 1, 1))
            if a.bias is not None:
                a.bias.div_(s)
            if b.groups == 1:
                b.weight.mul_(s.view(1, -1, 1, 1))
            else:
                b.weight.mul_(s.view(-1, 1, 1, 1))
    return len(pairs)


@torch.no_grad()
def bias_correction(q_model: nn.Module, fp_model: nn.Module, batches, layers: List[str]) -> int:
    """Empirical bias correction: for each quantised layer in order, subtract the mean output error
    E[y_q - y_fp] (per output channel) from its bias. Inputs to each layer come from the already
    corrected quantised model."""
    batches = list(batches)
    fixed = 0
    for name in layers:
        q_layer = q_model.get_submodule(name)
        f_layer = fp_model.get_submodule(name)
        sums = {}

        def make(key):
            def hook(m, inp, out):
                dims = [0] + list(range(2, out.dim())) if out.dim() == 4 else list(range(out.dim() - 1))
                sums[key] = sums.get(key, 0) + out.float().mean(dims)
            return hook

        h1 = q_layer.register_forward_hook(make("q"))
        h2 = f_layer.register_forward_hook(make("f"))
        for x, _ in batches:
            q_model(x)
            fp_model(x)
        h1.remove()
        h2.remove()
        err = (sums["q"] - sums["f"]) / len(batches)
        if q_layer.bias is None:
            q_layer.bias = nn.Parameter(torch.zeros(err.shape[0], device=err.device))
        q_layer.bias.sub_(err)
        fixed += 1
    return fixed


def relu6_to_relu(model: nn.Module) -> int:
    n = 0
    for m in list(model.modules()):
        for k, child in m._modules.items():
            if isinstance(child, nn.ReLU6):
                m._modules[k] = nn.ReLU(inplace=True)
                n += 1
    return n
