"""An explicit multi-head attention module and converters to it.

``nn.MultiheadAttention`` (MobileViT) and the ViT ``Attention`` of Multi-Tool4Vision (which calls
``scaled_dot_product_attention``) hide the softmax and the head structure. ``SplitAttention`` computes the
same function with every piece exposed:

- ``qkv`` / ``proj`` are ``nn.Linear``, so weight and input quantisers attach to them like any layer.
- ``softmax_q`` / ``post_q`` are identity slots for quantising the attention probabilities (FQ-ViT's
  Log-Int-Softmax, PTQ4ViT's twin uniform quantiser).
- ``head_gate`` scales each head's output; head importance (Michel et al., 2019) is the gradient on it.
- Heads can be removed physically with ``prune_heads``; the projection widths shrink with them.
- ``last_cls_attn`` keeps the class-token attention when ``keep_attn`` is set (EViT token selection).
"""
from typing import Iterable

import torch
import torch.nn as nn


class SplitAttention(nn.Module):
    counts_attention_macs = True

    def __init__(self, dim: int, num_heads: int, head_dim: int, qkv_bias: bool = True, batch_first: bool = True,
                 mha_style: bool = False):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, 3 * num_heads * head_dim, bias=qkv_bias)
        self.proj = nn.Linear(num_heads * head_dim, dim)
        self.softmax_q = nn.Identity()
        self.post_q = nn.Identity()
        self.register_buffer("head_gate", torch.ones(num_heads))
        self.gate_param = None
        self.keep_attn = False
        self.last_cls_attn = None
        self.store_keys = False
        self.last_keys = None
        self.size_bias = None  # b, n, 1 token sizes, set by ToMe blocks
        self.mha_style = mha_style  # forward(q, k, v) -> (out, None) like nn.MultiheadAttention

    @property
    def inner_dim(self) -> int:
        return self.num_heads * self.head_dim

    def attend(self, x: torch.Tensor) -> torch.Tensor:
        b, n, _ = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        if self.store_keys:
            self.last_keys = k.mean(1).detach()  # b, n, d (ToMe similarity metric)
        attn = (q * self.scale) @ k.transpose(-2, -1)
        if self.size_bias is not None:  # ToMe proportional attention: log(token size) on the keys
            attn = attn + self.size_bias.log()[:, None, None, :, 0]
        attn = self.softmax_q(attn.softmax(dim=-1))
        if self.keep_attn:
            self.last_cls_attn = attn[:, :, 0, 1:].mean(1).detach()  # b, n-1
        out = self.post_q(attn @ v)  # b, h, n, d
        gate = self.gate_param if self.gate_param is not None else self.head_gate
        out = out * gate.view(1, -1, 1, 1)
        return self.proj(out.transpose(1, 2).reshape(b, n, self.inner_dim))

    def forward(self, x, *args, **kwargs):
        if self.mha_style:
            return self.attend(x), None
        return self.attend(x)

    @torch.no_grad()
    def prune_heads(self, keep: Iterable[int]) -> None:
        keep = sorted(int(h) for h in keep)
        if not keep:
            raise ValueError("at least one head must stay")
        d = self.head_dim
        rows = torch.cat([torch.arange(s * self.inner_dim + h * d, s * self.inner_dim + (h + 1) * d)
                          for s in range(3) for h in keep])
        cols = torch.cat([torch.arange(h * d, (h + 1) * d) for h in keep])
        qkv = nn.Linear(self.dim, len(rows), bias=self.qkv.bias is not None).to(self.qkv.weight.device)
        qkv.weight.copy_(self.qkv.weight[rows])
        if self.qkv.bias is not None:
            qkv.bias.copy_(self.qkv.bias[rows])
        proj = nn.Linear(len(cols), self.dim).to(self.proj.weight.device)
        proj.weight.copy_(self.proj.weight[:, cols])
        proj.bias.copy_(self.proj.bias)
        self.qkv, self.proj = qkv, proj
        self.head_gate = self.head_gate[keep].clone()
        self.num_heads = len(keep)


@torch.no_grad()
def from_mha(m: nn.MultiheadAttention) -> SplitAttention:
    if not m.batch_first or not m._qkv_same_embed_dim:
        raise ValueError("only batch_first self-attention with a shared embed dim is supported")
    e, h = m.embed_dim, m.num_heads
    out = SplitAttention(e, h, e // h, qkv_bias=m.in_proj_bias is not None, mha_style=True)
    out.qkv.weight.copy_(m.in_proj_weight)
    if m.in_proj_bias is not None:
        out.qkv.bias.copy_(m.in_proj_bias)
    out.proj.weight.copy_(m.out_proj.weight)
    out.proj.bias.copy_(m.out_proj.bias)
    return out.to(m.in_proj_weight.device)


@torch.no_grad()
def from_vit_attention(m: nn.Module) -> SplitAttention:
    dim = m.qkv.in_features
    h = m.num_heads
    out = SplitAttention(dim, h, dim // h, qkv_bias=m.qkv.bias is not None)
    out.qkv.load_state_dict(m.qkv.state_dict())
    out.proj.load_state_dict(m.proj.state_dict())
    return out.to(m.qkv.weight.device)


def convert_attention(model: nn.Module) -> int:
    """Replace every nn.MultiheadAttention and ViT ``Attention`` with an equivalent SplitAttention, in place.
    Returns the number of modules replaced."""
    count = 0
    for name, child in list(model.named_children()):
        if isinstance(child, nn.MultiheadAttention):
            setattr(model, name, from_mha(child))
            count += 1
        elif type(child).__name__ == "Attention" and hasattr(child, "qkv") and hasattr(child, "num_heads"):
            setattr(model, name, from_vit_attention(child))
            count += 1
        else:
            count += convert_attention(child)
    # nn.ModuleDict / ModuleList children are covered by named_children recursion.
    return count


def attention_modules(model: nn.Module):
    return [(n, m) for n, m in model.named_modules() if isinstance(m, SplitAttention)]
