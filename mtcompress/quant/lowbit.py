"""Binary and ternary networks, trained from the full-precision model.

The first and last layers stay in float and BatchNorm is not folded (it restores the scale that
binarisation removes), as in the papers. Storage counts 1 bit (binary) or 2 bits (ternary) per weight
plus one float scale per output channel (XNOR) or per layer (TWN).
"""
import torch
import torch.nn as nn

from ..registry import register_method
from .qat import QAT


def sign_ste(x: torch.Tensor) -> torch.Tensor:
    """sign(x) forward; gradient of hardtanh (1 inside [-1, 1], 0 outside) backward."""
    clipped = torch.clamp(x, -1, 1)
    return (torch.where(x >= 0, torch.ones_like(x), -torch.ones_like(x)) - clipped).detach() + clipped


class _Fixed(nn.Module):
    exportable = False

    def __init__(self, bits):
        super().__init__()
        self.bits = bits
        self.mode = "quant"
        self.export = False
        self.observer = None

    def observe(self):
        pass

    def finish(self):
        pass

    def init_from(self, w):
        pass


class SignWeight(_Fixed):
    """BNN weights (Hubara et al., 2016): sign(w)."""

    def __init__(self, scaled=False):
        super().__init__(1)
        self.scaled = scaled
        self.n_scales = 0

    def init_from(self, w):
        self.n_scales = w.shape[0] if self.scaled else 0

    def forward(self, w):
        b = sign_ste(w)
        if self.scaled:  # XNOR-Net: alpha = mean |w| per output channel
            alpha = w.detach().abs().reshape(w.shape[0], -1).mean(1).view(-1, *([1] * (w.dim() - 1)))
            return b * alpha
        return b


class SignAct(_Fixed):
    """Binary activations. ``scaled`` multiplies by the per-sample mean |x| (a per-sample simplification of
    XNOR-Net's K matrix)."""

    def __init__(self, scaled=False):
        super().__init__(1)
        self.scaled = scaled

    def forward(self, x):
        b = sign_ste(x)
        if self.scaled:
            beta = x.detach().abs().reshape(x.shape[0], -1).mean(1).view(-1, *([1] * (x.dim() - 1)))
            return b * beta
        return b


class TernaryWeight(_Fixed):
    """TWN (Li et al., 2016): delta = 0.7 * mean|w|, alpha = mean of |w| above delta,
    w_t = alpha * sign(w) * [|w| > delta]."""

    def __init__(self):
        super().__init__(2)

    def forward(self, w):
        with torch.no_grad():
            delta = 0.7 * w.abs().mean()
            mask = (w.abs() > delta).float()
            alpha = (w.abs() * mask).sum() / mask.sum().clamp_min(1)
        t = alpha * torch.sign(w) * mask
        return (t - w).detach() + w


class LowBit(QAT):
    hparams = {"exclude": ["first", "last"], "bn_fold": False, "ft_epochs": 15, "ft_optimizer": "adam",
               "ft_lr": 1e-3, "ft_weight_decay": 0.0, "w_bits": 8, "a_bits": 8}


@register_method("bnn", kind="quant", family="Binary / ternary", paper="Hubara et al., 2016 (Binarized Neural Networks)")
class BNN(LowBit):
    def make_weight_quantizer(self, layer, old):
        return SignWeight(scaled=False)

    def make_act_quantizer(self, layer, old, signed):
        return SignAct(scaled=False)


@register_method("xnor", kind="quant", family="Binary / ternary", paper="Rastegari et al., 2016 (XNOR-Net)")
class XNOR(LowBit):
    def make_weight_quantizer(self, layer, old):
        return SignWeight(scaled=True)

    def make_act_quantizer(self, layer, old, signed):
        return SignAct(scaled=True)


@register_method("twn", kind="quant", family="Binary / ternary", paper="Li et al., 2016 (Ternary Weight Networks)")
class TWN(LowBit):
    """Ternary weights, float activations."""

    def make_weight_quantizer(self, layer, old):
        return TernaryWeight()

    def make_act_quantizer(self, layer, old, signed):
        return nn.Identity()
