"""Pruning that works on transformer structure: attention heads, whole blocks and tokens."""
import math
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base import PruneMethod
from ..nn.attention import SplitAttention, attention_modules, convert_attention
from ..registry import register_method
from ..train import evaluate


# ------------------------------------------------------------------------------------------ heads


@register_method("head_prune", kind="prune", family="Transformer",
                 paper="Michel et al., 2019 (Are Sixteen Heads Really Better than One?)")
class HeadPrune(PruneMethod):
    """Head importance |dL/d gate_h| over the calibration set, normalised per layer (L2), lowest heads
    removed across the network (at least one per layer stays). The q/k/v and output projections shrink."""

    hparams = {"ratio": 0.5, "calib_batches": 8, "exclude": []}

    def apply(self, ctx):
        model = ctx.model
        convert_attention(model)
        attns = attention_modules(model)
        if not attns:
            raise RuntimeError("head_prune: the model has no attention layers")
        for _, a in attns:
            a.gate_param = nn.Parameter(torch.ones(a.num_heads, device=ctx.device))
        imp = {n: torch.zeros(a.num_heads, device=ctx.device) for n, a in attns}
        model.eval()
        for x, y in ctx.data.calib(64, limit=self.hp["calib_batches"]):
            model.zero_grad(set_to_none=True)
            F.cross_entropy(model(x), y).backward()
            for n, a in attns:
                imp[n] += a.gate_param.grad.detach().abs()
        model.zero_grad(set_to_none=True)
        for _, a in attns:
            a.gate_param = None
        for n in imp:
            imp[n] = imp[n] / (imp[n].norm() + 1e-12)
        total = sum(a.num_heads for _, a in attns)
        n_remove = int(total * self.hp["ratio"])
        ranked = sorted(((float(s), n, h) for n, v in imp.items() for h, s in enumerate(v)))
        removed = {n: set() for n, _ in attns}
        heads = dict((n, a.num_heads) for n, a in attns)
        for s, n, h in ranked:
            if n_remove == 0:
                break
            if heads[n] - len(removed[n]) > 1:
                removed[n].add(h)
                n_remove -= 1
        for n, a in attns:
            a.prune_heads([h for h in range(a.num_heads) if h not in removed[n]])
        ctx.info["heads"] = {n: {"kept": a.num_heads, "of": heads[n]} for n, a in attns}
        ctx.info["after_prune"] = ctx.evaluate()
        ctx.log(f"head_prune: {ctx.info['heads']} -> {ctx.info['after_prune']}")
        self.finetune(ctx)


# ------------------------------------------------------------------------------------------ blocks


def block_candidates(model: nn.Module) -> List[Tuple[nn.Module, str, nn.Module]]:
    """(container, key, block) for repeated blocks: entries of a ModuleList, or of a Sequential in which
    the block's class appears at least twice. Leaf layers are not blocks."""
    out = []
    for cname, container in model.named_modules():
        if not isinstance(container, (nn.ModuleList, nn.Sequential)):
            continue
        children = list(container.named_children())
        types = [type(c) for _, c in children]
        for key, child in children:
            if len(list(child.children())) == 0 or not any(p.numel() for p in child.parameters()):
                continue
            if isinstance(container, nn.ModuleList) or types.count(type(child)) >= 2:
                out.append((cname, container, key, child))
    return out


def _drop(container, key):
    if isinstance(container, nn.ModuleList):
        kept = [m for k, m in container.named_children() if k != key]
        container._modules.clear()
        for i, m in enumerate(kept):
            container._modules[str(i)] = m
    else:
        container._modules[key] = nn.Identity()


def _restore(container, snapshot):
    container._modules.clear()
    container._modules.update(snapshot)


@register_method("layer_drop", kind="prune", family="Transformer",
                 paper="Men et al., 2024 (ShortGPT); Fan et al., 2020 (LayerDrop)")
class LayerDrop(PruneMethod):
    """Remove whole residual blocks. Every candidate block is removed once on its own and the calibration
    loss is measured; blocks whose removal costs least are dropped (``num_blocks`` of them). Candidates
    whose removal breaks the forward pass (stage transitions, downsampling blocks) are skipped."""

    hparams = {"num_blocks": 2, "calib_batches": 4, "exclude": []}

    def apply(self, ctx):
        model = ctx.model
        example = next(iter(ctx.data.calib(2)))[0]
        with torch.no_grad():
            ref_shape = model.eval()(example).shape
        calib = list(ctx.data.calib(128, limit=self.hp["calib_batches"]))
        base = evaluate(model, calib)["loss"]
        costs = []
        for cname, container, key, block in block_candidates(model):
            snapshot = dict(container._modules)
            _drop(container, key)
            try:
                with torch.no_grad():
                    ok = model(example).shape == ref_shape
                loss = evaluate(model, calib)["loss"] if ok else None
            except Exception:
                loss = None
            _restore(container, snapshot)
            if loss is not None:
                costs.append((loss - base, cname, key))
        costs.sort()
        chosen = costs[: self.hp["num_blocks"]]
        # Drop from the highest index first so ModuleList indices stay valid.
        by_container = {}
        for _, cname, key in chosen:
            by_container.setdefault(cname, []).append(key)
        modules = dict(model.named_modules())
        for cname, keys in by_container.items():
            for key in sorted(keys, key=lambda k: -int(k) if k.isdigit() else 0):
                _drop(modules[cname], key)
        ctx.info["candidates"] = [{"block": f"{c}.{k}", "loss_increase": round(d, 4)} for d, c, k in costs]
        ctx.info["dropped"] = [f"{c}.{k}" for _, c, k in chosen]
        ctx.info["after_prune"] = ctx.evaluate()
        ctx.log(f"layer_drop: dropped {ctx.info['dropped']} -> {ctx.info['after_prune']}")
        self.finetune(ctx)


# ------------------------------------------------------------------------------------------ tokens


class _TokenState:
    def __init__(self):
        self.size = None


class TokenBlock(nn.Module):
    """Wraps a ViT ``TransformerBlock`` and reduces tokens between attention and MLP."""

    def __init__(self, block: nn.Module, mode: str, state: _TokenState, r: int = 0, keep_rate: float = 1.0,
                 first: bool = False):
        super().__init__()
        self.block = block
        self.mode = mode
        self.state = state
        self.r = r
        self.keep_rate = keep_rate
        self.first = first

    def forward(self, x):
        b = self.block
        attn: SplitAttention = b.attn
        if self.first or self.state.size is None or self.state.size.shape[:2] != x.shape[:2]:
            self.state.size = torch.ones(x.shape[0], x.shape[1], 1, device=x.device, dtype=x.dtype)
        if self.mode == "tome":
            attn.store_keys = self.r > 0
            attn.size_bias = self.state.size
        else:
            attn.keep_attn = self.keep_rate < 1
        x = x + b.drop_path(b.ls1(attn(b.norm1(x))))
        attn.size_bias = None
        if self.mode == "tome" and self.r > 0:
            x, self.state.size = tome_merge(x, attn.last_keys, self.state.size, self.r)
            attn.last_keys = None
        elif self.mode == "evit" and self.keep_rate < 1:
            x = evit_select(x, attn.last_cls_attn, self.keep_rate)
            attn.last_cls_attn = None
        return x + b.drop_path(b.ls2(b.mlp(b.norm2(x))))


def tome_merge(x, keys, size, r):
    """Bipartite soft matching (Bolya et al., 2023). The class token (index 0) is never merged."""
    b, n, c = x.shape
    r = min(r, (n - 1) // 2)
    if r <= 0:
        return x, size
    k = keys / keys.norm(dim=-1, keepdim=True)
    a_idx = torch.arange(1, n, 2, device=x.device)  # set A: odd positions (after the class token)
    b_idx = torch.cat([torch.zeros(1, dtype=torch.long, device=x.device), torch.arange(2, n, 2, device=x.device)])
    scores = k[:, a_idx] @ k[:, b_idx].transpose(1, 2)
    scores[..., 0] = -math.inf  # never merge into the class token
    node_max, node_idx = scores.max(dim=-1)
    order = node_max.argsort(dim=-1, descending=True)
    src = order[:, :r]  # merged A tokens
    unm = order[:, r:]
    dst = node_idx.gather(1, src)
    xa, xb = x[:, a_idx] * size[:, a_idx], x[:, b_idx] * size[:, b_idx]
    sa, sb = size[:, a_idx], size[:, b_idx]
    xb = xb.scatter_reduce(1, dst[..., None].expand(-1, -1, c), xa.gather(1, src[..., None].expand(-1, -1, c)), "sum")
    sb = sb.scatter_reduce(1, dst[..., None], sa.gather(1, src[..., None]), "sum")
    xa_keep = xa.gather(1, unm[..., None].expand(-1, -1, c))
    sa_keep = sa.gather(1, unm[..., None])
    new_x = torch.cat([xb[:, :1], xa_keep, xb[:, 1:]], 1)
    new_s = torch.cat([sb[:, :1], sa_keep, sb[:, 1:]], 1)
    return new_x / new_s, new_s


def evit_select(x, cls_attn, keep_rate):
    """EViT (Liang et al., 2022): keep the patch tokens the class token attends to most and fuse the rest
    into one token weighted by that attention."""
    b, n, c = x.shape
    patches = n - 1
    keep = max(1, int(math.ceil(patches * keep_rate)))
    if keep >= patches:
        return x
    idx = cls_attn.argsort(dim=1, descending=True)
    keep_idx, drop_idx = idx[:, :keep], idx[:, keep:]
    tokens = x[:, 1:]
    kept = tokens.gather(1, keep_idx[..., None].expand(-1, -1, c))
    dropped = tokens.gather(1, drop_idx[..., None].expand(-1, -1, c))
    w = cls_attn.gather(1, drop_idx)[..., None]
    fused = (dropped * w).sum(1, keepdim=True) / (w.sum(1, keepdim=True) + 1e-12)
    return torch.cat([x[:, :1], kept, fused], 1)


def _vit_blocks(model):
    blocks = getattr(model, "blocks", None)
    if blocks is None or not hasattr(model, "cls_token"):
        raise RuntimeError("token reduction needs a ViT with a class token and model.blocks (vit_*, deit_*)")
    return blocks


@register_method("tome", kind="prune", family="Token reduction", paper="Bolya et al., 2023 (Token Merging)")
class ToMe(PruneMethod):
    """Merge ``r`` tokens in every block by bipartite soft matching on the attention keys, with
    proportional attention. Works without training (``ft_epochs: 0``)."""

    hparams = {"r": 8, "ft_epochs": 0, "exclude": []}

    def apply(self, ctx):
        blocks = _vit_blocks(ctx.model)
        convert_attention(ctx.model)
        state = _TokenState()
        for i in range(len(blocks)):
            blocks[i] = TokenBlock(blocks[i], "tome", state, r=self.hp["r"], first=(i == 0))
        ctx.info["after_prune"] = ctx.evaluate()
        self.finetune(ctx)


@register_method("evit", kind="prune", family="Token reduction",
                 paper="Liang et al., 2022 (EViT: Not All Patches are What You Need)")
class EViT(PruneMethod):
    """At blocks ``layers``, keep ``keep_rate`` of the patch tokens by class-token attention and fuse the
    rest into one token. Needs fine-tuning."""

    hparams = {"keep_rate": 0.7, "layers": [3, 6, 9], "ft_epochs": 5, "exclude": []}

    def apply(self, ctx):
        blocks = _vit_blocks(ctx.model)
        convert_attention(ctx.model)
        state = _TokenState()
        for i in range(len(blocks)):
            rate = self.hp["keep_rate"] if i in self.hp["layers"] else 1.0
            blocks[i] = TokenBlock(blocks[i], "evit", state, keep_rate=rate, first=(i == 0))
        ctx.info["after_prune"] = ctx.evaluate()
        self.finetune(ctx)
