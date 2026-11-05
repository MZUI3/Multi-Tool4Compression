"""N:M semi-structured sparsity (2:4 by default).

In every group of M consecutive weights along the input dimension, the N largest are kept. For a conv
kernel (out, in, kh, kw) the groups run along ``in`` at each kernel position, as in NVIDIA's ASP.
Ampere and newer GPUs run 2:4 matrices on sparse tensor cores. With ``measure_gpu: true`` the run times
the model's Linear layers as dense fp16 and as ``to_sparse_semi_structured`` fp16 on the GPU, at a token
batch of ``gpu_tokens``, and stores both in summary.json.
"""
import torch
import torch.nn as nn

from ..base import PruneMethod
from ..registry import register_method
from .masks import MaskSet


def nm_mask(w: torch.Tensor, n: int, m: int) -> torch.Tensor:
    if w.dim() == 4:
        o, i, kh, kw = w.shape
        flat = w.permute(0, 2, 3, 1).reshape(-1, i)
    else:
        flat = w.reshape(w.shape[0], -1)
    cols = flat.shape[1]
    if cols % m:
        return torch.ones_like(w, dtype=torch.bool)  # cannot be grouped; left dense
    groups = flat.abs().reshape(-1, m)
    idx = groups.topk(n, dim=1).indices
    mask = torch.zeros_like(groups, dtype=torch.bool).scatter_(1, idx, True).reshape(flat.shape)
    if w.dim() == 4:
        return mask.reshape(o, kh, kw, i).permute(0, 3, 1, 2).contiguous()
    return mask.reshape(w.shape)


def _time(fn, iters=50):
    for _ in range(10):
        fn()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


@torch.no_grad()
def semi_structured_linear_timing(model: nn.Module, tokens: int = 4096) -> dict:
    from torch.sparse import to_sparse_semi_structured

    out = {}
    for name, m in model.named_modules():
        if not isinstance(m, nn.Linear) or type(m).__name__ == "NonDynamicallyQuantizableLinear":
            continue
        w = m.weight.detach().half().cuda()
        if w.shape[0] % 64 or w.shape[1] % 64 or float((w == 0).float().mean()) < 0.49:
            continue
        x = torch.randn(tokens, w.shape[1], device="cuda", dtype=torch.half)
        dense = _time(lambda: torch.nn.functional.linear(x, w))
        try:
            ws = to_sparse_semi_structured(w)
            sparse = _time(lambda: torch.nn.functional.linear(x, ws))
        except Exception as exc:  # unsupported shape or build
            out[name] = {"error": str(exc).splitlines()[0]}
            continue
        out[name] = {"shape": list(w.shape), "dense_fp16_ms": round(dense, 4), "sparse_2to4_fp16_ms": round(sparse, 4),
                     "speedup": round(dense / sparse, 3)}
    return out


@register_method("nm_sparsity", kind="prune", family="Semi-structured",
                 paper="Mishra et al., 2021 (Accelerating Sparse Deep Neural Networks, 2:4)")
class NMSparsity(PruneMethod):
    hparams = {"n": 2, "m": 4, "measure_gpu": True, "gpu_tokens": 8192}

    def apply(self, ctx):
        masks = MaskSet(ctx.model, self.hp["exclude"])
        for name, w in masks.targets.items():
            masks.masks[name] = nm_mask(w.detach(), self.hp["n"], self.hp["m"])
        masks.apply()
        ctx.info["after_prune"] = ctx.evaluate()
        self.finetune(ctx, hooks=masks)
        masks.finalize(ctx.model, ctx.info)
        if self.hp["measure_gpu"] and torch.cuda.is_available():
            try:
                ctx.info["semi_structured_gpu"] = semi_structured_linear_timing(ctx.model, self.hp["gpu_tokens"])
            except Exception as exc:
                ctx.info["semi_structured_gpu"] = {"error": str(exc).splitlines()[0]}
