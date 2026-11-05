"""Structured (channel) pruning: whole channel groups are removed from the tensors.

All methods share ``depgraph.build_groups`` and ``remove_channels``; they differ only in how a group's
channels are scored. ``ratio`` is the fraction of channels removed from every prunable group
(``scope: uniform``) or from all of them together after normalising each group's scores by their mean
(``scope: global``, capped at ``max_ratio`` per group).
"""
import math
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base import PruneMethod, resolve_exclude
from ..measure.profile import param_stats
from ..nn.attention import convert_attention
from ..registry import register_method
from .depgraph import Group, build_groups, channel_tensors, per_channel, prunable_groups, remove_channels

PRODUCERS = ("conv_out", "linear_out", "attn_out")


class ChannelPrune(PruneMethod):
    hparams = {"ratio": 0.5, "scope": "uniform", "max_ratio": 0.9, "round_to": 1, "exclude": [],
               "calib_batches": 8}

    def importance(self, ctx, groups: List[Group]) -> List[torch.Tensor]:
        raise NotImplementedError

    def prepare(self, ctx, groups: List[Group]) -> None:
        """Optional step before scoring (sparsity training for Network Slimming)."""

    def plan(self, groups, scores) -> List:
        hp = self.hp
        plan = []
        if hp["scope"] == "uniform":
            for g, s in zip(groups, scores):
                n_keep = g.size - int(math.floor(g.size * hp["ratio"]))
                n_keep = max(hp["round_to"], int(math.ceil(n_keep / hp["round_to"]) * hp["round_to"]))
                n_keep = min(n_keep, g.size)
                keep = torch.zeros(g.size, dtype=torch.bool)
                keep[s.float().cpu().topk(n_keep).indices] = True
                plan.append((g, keep))
        elif hp["scope"] == "global":
            normed = [s.float().cpu() / (s.float().mean().cpu() + 1e-12) for s in scores]
            allscores = torch.cat(normed)
            n_remove = int(allscores.numel() * hp["ratio"])
            threshold = allscores.kthvalue(max(n_remove, 1)).values if n_remove > 0 else -float("inf")
            for g, s in zip(groups, normed):
                keep = s > threshold
                min_keep = max(1, int(math.ceil(g.size * (1 - hp["max_ratio"]))))
                if keep.sum() < min_keep:
                    keep = torch.zeros(g.size, dtype=torch.bool)
                    keep[s.topk(min_keep).indices] = True
                plan.append((g, keep))
        else:
            raise ValueError("scope must be uniform or global")
        return plan

    def apply(self, ctx):
        convert_attention(ctx.model)
        example = next(iter(ctx.data.calib(2)))[0]
        skip = resolve_exclude(ctx.model, self.hp["exclude"])
        groups = prunable_groups(build_groups(ctx.model, example, skip))
        self.prepare(ctx, groups)
        scores = self.importance(ctx, groups)
        plan = self.plan(groups, scores)
        before = param_stats(ctx.model)["params"]
        remove_channels(ctx.model, plan)
        with torch.no_grad():
            ctx.model.eval()(example)
        ctx.info["groups"] = [{"name": g.names[0], "channels": g.size, "kept": int(k.sum()), "members": len(g.members)}
                              for g, k in plan]
        ctx.info["params_before"] = before
        ctx.info["after_prune"] = ctx.evaluate()
        ctx.log(f"{self.name}: params {before} -> {param_stats(ctx.model)['params']}, before fine-tuning "
                f"{ctx.info['after_prune']}")
        self.finetune(ctx)


def _sum_over(group: Group, model, fn, roles=None) -> torch.Tensor:
    total = torch.zeros(group.size)
    for t, dim, off, role in channel_tensors(group, model, roles):
        total += fn(per_channel(t, dim, off, group.size)).float().cpu()
    return total


@register_method("l1_filter", kind="prune", family="Structured channel", paper="Li et al., 2017 (Pruning Filters)")
class L1Filter(ChannelPrune):
    """Sum of absolute filter weights of the producing layers."""

    def importance(self, ctx, groups):
        return [_sum_over(g, ctx.model, lambda m: m.abs().sum(1), PRODUCERS) for g in groups]


@register_method("group_l2", kind="prune", family="Structured channel",
                 paper="Fang et al., 2023 (DepGraph, group-level L2)")
class GroupL2(ChannelPrune):
    """Squared L2 norm over every tensor slice of the group (producers, consumers, norms)."""

    def importance(self, ctx, groups):
        return [_sum_over(g, ctx.model, lambda m: (m ** 2).sum(1)) for g in groups]


@register_method("fpgm", kind="prune", family="Structured channel",
                 paper="He et al., 2019 (Filter Pruning via Geometric Median)")
class FPGM(ChannelPrune):
    """Filters close to the geometric median of their layer are replaceable; the score is a filter's
    summed Euclidean distance to the other filters (small = redundant)."""

    def importance(self, ctx, groups):
        return [_sum_over(g, ctx.model, lambda m: torch.cdist(m.float(), m.float()).sum(1), PRODUCERS) for g in groups]


@register_method("slimming", kind="prune", family="Structured channel",
                 paper="Liu et al., 2017 (Network Slimming)")
class Slimming(ChannelPrune):
    """Train with an L1 penalty on the normalisation scales for ``sparsity_epochs``, then rank channels by
    |gamma|. Groups without a normalisation layer fall back to the L1 norm of the producers."""

    hparams = {"sparsity_epochs": 5, "l1_lambda": 1e-4, "sparsity_lr": 0.01}

    def _gammas(self, groups):
        out = []
        for g in groups:
            for mem in g.members:
                if mem.role == "norm" and getattr(mem.module, "weight", None) is not None:
                    out.append(mem.module.weight)
        return list({id(p): p for p in out}.values())

    def prepare(self, ctx, groups):
        gammas = self._gammas(groups)
        lam = self.hp["l1_lambda"]

        class Hooks:
            def extra_loss(_, model):
                return lam * sum(g.abs().sum() for g in gammas)

        self.finetune(ctx, hooks=Hooks(), epochs=self.hp["sparsity_epochs"], lr=self.hp["sparsity_lr"],
                      tag="slimming/sparsity")

    def importance(self, ctx, groups):
        scores = []
        fallback = []
        for g in groups:
            s = _sum_over(g, ctx.model, lambda m: m.abs().sum(1), ("norm",))
            if s.sum() == 0:
                fallback.append(g.names[0])
                s = _sum_over(g, ctx.model, lambda m: m.abs().sum(1), PRODUCERS)
            scores.append(s)
        ctx.info["l1_fallback_groups"] = fallback
        return scores


@register_method("taylor", kind="prune", family="Structured channel",
                 paper="Molchanov et al., 2019 (Importance Estimation for Neural Network Pruning)")
class Taylor(ChannelPrune):
    """First-order Taylor estimate of the loss change when a channel group is removed:
    (sum over the group's parameters of g * w)^2, averaged over calibration batches."""

    def importance(self, ctx, groups):
        model = ctx.model
        scores = [torch.zeros(g.size) for g in groups]
        model.eval()
        n = 0
        for x, y in ctx.data.calib(64, limit=self.hp["calib_batches"]):
            model.zero_grad(set_to_none=True)
            F.cross_entropy(model(x), y).backward()
            for i, g in enumerate(groups):
                contrib = torch.zeros(g.size)
                for t, dim, off, role in channel_tensors(g, model):
                    if not isinstance(t, nn.Parameter) or t.grad is None:
                        continue
                    gw = per_channel(t.grad * t, dim, off, g.size)
                    contrib += gw.sum(1).float().cpu()
                scores[i] += contrib ** 2
            n += 1
        model.zero_grad(set_to_none=True)
        return [s / max(n, 1) for s in scores]


@register_method("hrank", kind="prune", family="Structured channel",
                 paper="Lin et al., 2020 (HRank: Filter Pruning using High-Rank Feature Map)")
class HRank(ChannelPrune):
    """Average rank of each channel's feature map at the output of the producing conv. Groups produced
    only by linear or attention layers have no 2-D maps and fall back to the L1 norm of the producers."""

    def importance(self, ctx, groups):
        model = ctx.model
        acc: Dict[int, torch.Tensor] = {}
        hooks = []
        convs = {}
        for i, g in enumerate(groups):
            for mem in g.members:
                if mem.role == "conv_out":
                    convs.setdefault(mem.module, []).append(i)

        def make_hook(mod):
            def hook(m, inp, out):
                r = torch.linalg.matrix_rank(out.detach().float()).float().sum(0).cpu()  # (C,)
                acc[id(mod)] = acc.get(id(mod), 0) + r
            return hook

        for mod in convs:
            hooks.append(mod.register_forward_hook(make_hook(mod)))
        model.eval()
        n = 0
        with torch.no_grad():
            for x, _ in ctx.data.calib(64, limit=self.hp["calib_batches"]):
                model(x)
                n += x.shape[0]
        for h in hooks:
            h.remove()
        scores, fallback = [], []
        for i, g in enumerate(groups):
            s = torch.zeros(g.size)
            found = False
            for mod, idxs in convs.items():
                if i in idxs:
                    s += acc[id(mod)] / max(n, 1)
                    found = True
            if not found:
                fallback.append(g.names[0])
                s = _sum_over(g, model, lambda m: m.abs().sum(1), PRODUCERS)
            scores.append(s)
        ctx.info["l1_fallback_groups"] = fallback
        return scores


@register_method("thinet", kind="prune", family="Structured channel",
                 paper="Luo et al., 2017 (ThiNet; greedy selection, no least-squares rescaling)")
class ThiNet(ChannelPrune):
    """Choose the channels to remove so that the next layer's output changes least. For sampled output
    entries of every consuming layer, each input channel's contribution is computed; channels are removed
    greedily, each time picking the one that keeps the summed removed contribution smallest. Uniform
    scope only. The paper's least-squares rescaling of the remaining filters is not applied."""

    hparams = {"samples": 4096}

    def apply(self, ctx):
        if self.hp["scope"] != "uniform":
            raise ValueError("thinet supports scope: uniform only")
        super().apply(ctx)

    def importance(self, ctx, groups):
        model = ctx.model
        inputs: Dict[int, List[torch.Tensor]] = {}
        consumers = {}
        for i, g in enumerate(groups):
            for mem in g.members:
                if mem.role in ("conv_in", "linear_in", "attn_in"):
                    mod = mem.module.qkv if mem.role == "attn_in" else mem.module
                    consumers.setdefault(i, []).append((mod, mem.offset))
        hooks = []
        mods = {mod for lst in consumers.values() for mod, _ in lst}

        def make_hook(mod):
            def hook(m, inp):
                inputs.setdefault(id(mod), []).append(inp[0].detach())
            return hook

        for mod in mods:
            hooks.append(mod.register_forward_pre_hook(make_hook(mod)))
        model.eval()
        with torch.no_grad():
            for x, _ in ctx.data.calib(64, limit=max(1, self.hp["calib_batches"] // 2)):
                model(x)
        for h in hooks:
            h.remove()
        gen = torch.Generator().manual_seed(0)
        scores = []
        for i, g in enumerate(groups):
            mats = []
            for mod, off in consumers.get(i, []):
                X = torch.cat(inputs[id(mod)])
                W = mod.weight.detach()
                S = self.hp["samples"]
                if isinstance(mod, nn.Conv2d) and mod.groups == 1:
                    cols = F.unfold(X, mod.kernel_size, mod.dilation, mod.padding, mod.stride)  # B, Cin*kk, L
                    B, _, L = cols.shape
                    kk = mod.kernel_size[0] * mod.kernel_size[1]
                    b = torch.randint(0, B, (S,), generator=gen)
                    l = torch.randint(0, L, (S,), generator=gen)
                    o = torch.randint(0, W.shape[0], (S,), generator=gen)
                    xs = cols[b.to(X.device), :, l.to(X.device)].view(S, -1, kk)[:, off:off + g.size]
                    ws = W[o.to(W.device)].reshape(S, -1, kk)[:, off:off + g.size]
                    mats.append((xs * ws).sum(-1).float().cpu())
                elif isinstance(mod, nn.Linear):
                    Xf = X.reshape(-1, X.shape[-1])
                    r = torch.randint(0, Xf.shape[0], (S,), generator=gen)
                    o = torch.randint(0, W.shape[0], (S,), generator=gen)
                    mats.append((Xf[r.to(X.device), off:off + g.size] * W[o.to(W.device), off:off + g.size]).float().cpu())
            if not mats:
                scores.append(_sum_over(g, model, lambda m: m.abs().sum(1), PRODUCERS))
                continue
            A = torch.cat([m / (m.norm() + 1e-12) for m in mats])  # samples x channels
            n_remove = int(math.floor(g.size * self.hp["ratio"]))
            removed = []
            r = torch.zeros(A.shape[0])
            col_sq = (A ** 2).sum(0)
            available = torch.ones(g.size, dtype=torch.bool)
            for _ in range(n_remove):
                cost = 2 * (r @ A) + col_sq
                cost[~available] = float("inf")
                c = int(cost.argmin())
                removed.append(c)
                available[c] = False
                r += A[:, c]
            score = torch.full((g.size,), float(g.size))
            for rank, c in enumerate(removed):
                score[c] = rank
            scores.append(score)
        return scores
