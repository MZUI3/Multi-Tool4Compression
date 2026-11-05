"""Weight sharing with k-means and Huffman coding (Deep Compression, Han et al., 2016).

Each layer's non-zero weights are clustered into 2^bits centroids (linear initialisation, the paper's
best choice); zeros from earlier pruning stay exactly zero. During fine-tuning each centroid becomes the
mean of its cluster's updated weights after every step, so all weights of a cluster stay equal.
Storage per layer: a float32 codebook, the Huffman-coded cluster index of every non-zero weight and, if
the layer is sparse, a 1-bit mask (the paper stores relative indices instead). Activations stay in float.
"""
import heapq
from collections import Counter

import torch

from ..base import QuantMethod
from ..prune.masks import MaskSet
from ..registry import register_method


def huffman_bits(counts) -> int:
    """Total bits of the Huffman code for symbols with these counts."""
    counts = [c for c in counts if c > 0]
    if len(counts) <= 1:
        return sum(counts)
    heap = [(c, i, 0) for i, c in enumerate(counts)]  # (count, id, subtree cost)
    heapq.heapify(heap)
    total = 0
    uid = len(counts)
    while len(heap) > 1:
        c1, _, _ = heapq.heappop(heap)
        c2, _, _ = heapq.heappop(heap)
        total += c1 + c2  # every symbol below gets one more bit
        heapq.heappush(heap, (c1 + c2, uid, 0))
        uid += 1
    return total


@torch.no_grad()
def kmeans_1d(values: torch.Tensor, k: int, iters: int = 30):
    lo, hi = values.min(), values.max()
    centroids = torch.linspace(float(lo), float(hi), k, device=values.device)
    for _ in range(iters):
        assign = (values[:, None] - centroids[None]).abs().argmin(1)
        sums = torch.zeros(k, device=values.device).scatter_add_(0, assign, values)
        cnt = torch.zeros(k, device=values.device).scatter_add_(0, assign, torch.ones_like(values))
        centroids = torch.where(cnt > 0, sums / cnt.clamp_min(1), centroids)
    assign = (values[:, None] - centroids[None]).abs().argmin(1)
    return centroids, assign


@register_method("kmeans_share", kind="quant", family="Weight sharing",
                 paper="Han et al., 2016 (Deep Compression: k-means sharing + Huffman)")
class KMeansShare(QuantMethod):
    hparams = {"bits": 5, "iters": 30, "ft_epochs": 3, "huffman": True, "chunk": 1 << 20}

    def apply(self, ctx):
        k = 2 ** self.hp["bits"]
        masks = MaskSet(ctx.model, self.hp["exclude"])
        state = {}
        for name, w in masks.targets.items():
            nz = w.detach() != 0
            vals = w.detach()[nz]
            if vals.numel() == 0:
                continue
            centroids, assign = kmeans_1d(vals, min(k, vals.numel()), self.hp["iters"])
            state[name] = (nz, assign)
            w.data.zero_()
            w.data[nz] = centroids[assign]

        class Hooks:
            def after_backward(_, model):
                for name, w in masks.targets.items():
                    if name in state and w.grad is not None:
                        w.grad.mul_(state[name][0])

            def after_step(_, model):
                with torch.no_grad():
                    for name, w in masks.targets.items():
                        if name not in state:
                            continue
                        nz, assign = state[name]
                        vals = w[nz]
                        kk = int(assign.max()) + 1
                        sums = torch.zeros(kk, device=w.device).scatter_add_(0, assign, vals)
                        cnt = torch.zeros(kk, device=w.device).scatter_add_(0, assign, torch.ones_like(vals))
                        cent = sums / cnt.clamp_min(1)
                        w.zero_()
                        w[nz] = cent[assign]

        ctx.info["after_share"] = ctx.evaluate()
        self.finetune(ctx, hooks=Hooks())
        meta = getattr(ctx.model, "mtc_meta", None) or {}
        sb = meta.setdefault("storage_bytes", {})
        detail = {}
        for name, (nz, assign) in state.items():
            n_codes = int(assign.max()) + 1
            counts = Counter(assign.tolist())
            idx_bits = huffman_bits(list(counts.values())) if self.hp["huffman"] else assign.numel() * self.hp["bits"]
            mask_bits = nz.numel() if int(nz.sum()) < nz.numel() else 0
            total = (idx_bits + mask_bits) / 8 + n_codes * 4
            sb[name] = total
            detail[name] = {"clusters": n_codes, "index_bits_per_weight": round(idx_bits / max(assign.numel(), 1), 3),
                            "bytes": round(total)}
        ctx.model.mtc_meta = meta
        ctx.info["layers"] = detail
