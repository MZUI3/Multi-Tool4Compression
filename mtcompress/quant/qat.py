"""Quantisation-aware training: the network is fine-tuned through fake quantisers with the
straight-through estimator.

Every QAT method starts like PTQ (BatchNorm folded, min/max calibration) and then swaps in its own
quantisers before fine-tuning for ``ft_epochs``. Activation quantisers are signed when the calibration
data had negative values (SiLU, GELU and LayerNorm outputs); the original papers assume ReLU inputs.
"""
import math

import torch
import torch.nn as nn

from ..registry import register_method
from .core import Quantizer, grad_scale, quant_layers, round_ste
from .ptq import PTQ


class LSQQuantizer(Quantizer):
    """Learned Step Size Quantization (Esser et al., 2020): step size s is a parameter with gradient
    scale 1/sqrt(N * Qp), initialised to 2 * mean|x| / sqrt(Qp). With ``learn_offset`` (LSQ+, Bhalgat et
    al., 2020) an offset beta is learned too and both start from the MSE-optimal range."""

    def __init__(self, bits=8, symmetric=True, per_channel=False, ch_axis=0, learn_offset=False):
        super().__init__(bits, symmetric=symmetric, per_channel=per_channel, ch_axis=ch_axis,
                         observer="mse" if learn_offset else "minmax")
        self.learn_offset = learn_offset
        self.s = None
        self.beta = None
        self.abs_sum = 0.0
        self.n_seen = 0

    def observe(self):
        super().observe()
        self.abs_sum, self.n_seen = 0.0, 0

    def forward(self, x):
        if self.mode == "observe":
            self.abs_sum += float(x.detach().abs().mean()) * 1
            self.n_seen += 1
        return super().forward(x)

    @torch.no_grad()
    def init_from(self, x):
        self.reset_observer()
        self.observer.update(x)
        if self.per_channel:
            rows = x.detach().transpose(0, self.ch_axis).reshape(x.shape[self.ch_axis], -1)
            self.abs_sum, self.n_seen = rows.abs().mean(1), 1
        else:
            self.abs_sum, self.n_seen = x.detach().abs().mean(), 1
        self.finish()

    @torch.no_grad()
    def finish(self):
        lo, hi = self.observer.compute(self.qmin, self.qmax, self.symmetric)
        self.set_range(lo, hi)
        if self.learn_offset:
            s = self.scale.clone()
            beta = (-self.zero_point * self.scale).clone()  # x = s * q + beta
        else:
            mean_abs = self.abs_sum / max(self.n_seen, 1)
            mean_abs = torch.as_tensor(mean_abs, device=self.scale.device, dtype=torch.float32).reshape(-1)
            s = (2 * mean_abs / math.sqrt(self.qmax)).clamp_min(1e-8)
            beta = torch.zeros_like(s)
        self.s = nn.Parameter(s)
        self.beta = nn.Parameter(beta) if self.learn_offset else None
        self.mode = "quant"

    def quantize(self, x):
        n = x[0].numel() if self.per_channel else x.numel()
        g = 1.0 / math.sqrt(max(n, 1) * self.qmax)
        s = self._shape(x, grad_scale(self.s, g)).abs().clamp_min(1e-8)
        if self.learn_offset:
            b = self._shape(x, grad_scale(self.beta, g))
            return torch.clamp(round_ste((x - b) / s), self.qmin, self.qmax) * s + b
        return torch.clamp(round_ste(x / s), self.qmin, self.qmax) * s

    def sync_buffers(self):
        """Copy the learned step (and offset) into scale / zero_point for export."""
        with torch.no_grad():
            self.scale = self.s.detach().abs().clamp_min(1e-8).clone()
            if self.learn_offset:
                self.zero_point = torch.round(-self.beta.detach() / self.scale).clamp(self.qmin, self.qmax)
            else:
                self.zero_point = torch.zeros_like(self.scale)


class PACTQuantizer(Quantizer):
    """PACT (Choi et al., 2018): learnable clipping level alpha, y = clip(x, lo, alpha) quantised
    uniformly on [lo, alpha]; d y_q / d alpha = 1 where x >= alpha. ``lo`` is 0, or -alpha for signed inputs."""

    def __init__(self, bits=8, signed=False):
        super().__init__(bits, symmetric=signed, observer="percentile")
        self.signed = signed
        self.alpha = None

    @torch.no_grad()
    def finish(self):
        lo, hi = self.observer.compute(self.qmin, self.qmax, self.symmetric)
        a = torch.maximum(hi.abs(), lo.abs()) if self.signed else hi.clamp_min(1e-3)
        self.alpha = nn.Parameter(a.reshape(1).float())
        self.mode = "quant"

    def quantize(self, x):
        a = self.alpha.abs().clamp_min(1e-4)
        lo = -a if self.signed else torch.zeros_like(a)
        xc = torch.minimum(torch.maximum(x, lo), a)
        levels = 2 ** self.bits - 1
        step = ((a - lo) / levels).detach()
        return round_ste((xc - lo) / step) * step + lo

    def sync_buffers(self):
        with torch.no_grad():
            a = self.alpha.detach().abs()
            lo = -a if self.signed else torch.zeros_like(a)
            self.scale = ((a - lo) / (self.qmax - self.qmin)).clone()
            self.zero_point = torch.round(self.qmin - lo / self.scale) if not self.signed else torch.zeros_like(a)


class DoReFaWeight(nn.Module):
    """DoReFa-Net weights (Zhou et al., 2016): w_q = 2 Q_k(tanh(w) / (2 max|tanh(w)|) + 1/2) - 1, then
    multiplied by the layer's initial max|w| so layers without a following normalisation keep their
    scale (the paper relies on BatchNorm for that)."""

    exportable = False

    def __init__(self, bits):
        super().__init__()
        self.bits = bits
        self.mode = "quant"
        self.export = False
        self.register_buffer("gain", torch.ones(1))

    def init_from(self, w):
        self.gain = w.detach().abs().max().reshape(1)

    def observe(self):
        pass

    def finish(self):
        pass

    def forward(self, w):
        if self.export:
            raise RuntimeError("DoReFa weights have no ONNX int8 equivalent")
        t = torch.tanh(w)
        r = t / (2 * t.abs().max().detach()) + 0.5
        n = 2 ** self.bits - 1
        return (2 * round_ste(r * n) / n - 1) * self.gain


def apot_levels(bits: int) -> torch.Tensor:
    """Magnitude levels (b-1 bits) made of two additive power-of-two terms, normalised to max 1.
    Term A takes 0 or 2^-1, 2^-3, ...; term B takes 0 or 2^-2, 2^-4, ...; their sizes multiply to
    2^(b-1) levels."""
    mag_bits = bits - 1
    a_bits = (mag_bits + 1) // 2
    b_bits = mag_bits - a_bits
    a = [0.0] + [2.0 ** -(2 * i + 1) for i in range(2 ** a_bits - 1)]
    b = [0.0] + [2.0 ** -(2 * i + 2) for i in range(2 ** b_bits - 1)]
    vals = sorted({x + y for x in a for y in b})
    t = torch.tensor(vals)
    return t / t.max()


class APoTWeight(nn.Module):
    """Additive powers-of-two weights (Li et al., 2020) with a learnable clipping level alpha:
    w_q = alpha * sign(w) * nearest_level(|clip(w / alpha, -1, 1)|), straight-through."""

    exportable = False

    def __init__(self, bits):
        super().__init__()
        self.bits = bits
        self.mode = "quant"
        self.export = False
        self.register_buffer("levels", apot_levels(bits))
        self.alpha = nn.Parameter(torch.ones(1))

    def init_from(self, w):
        with torch.no_grad():
            self.alpha.fill_(float(w.detach().abs().max()))

    def observe(self):
        pass

    def finish(self):
        pass

    def forward(self, w):
        if self.export:
            raise RuntimeError("APoT weights have no ONNX int8 equivalent")
        a = self.alpha.abs().clamp_min(1e-6)
        x = torch.clamp(w / a, -1, 1)
        mag = x.abs()
        idx = (mag.unsqueeze(-1) - self.levels).abs().argmin(-1)
        q = self.levels[idx] * x.sign()
        return a * ((q - x).detach() + x)


# ------------------------------------------------------------------------------------- methods


class QAT(PTQ):
    hparams = {"ft_epochs": 5, "ft_lr": 0.005, "q_lr": 0.001, "exclude": []}

    def make_weight_quantizer(self, layer, old):
        return None  # keep the calibrated uniform quantiser

    def make_act_quantizer(self, layer, old, signed):
        return None

    def apply(self, ctx):
        self.quantize(ctx)
        ctx.info["after_ptq_init"] = ctx.evaluate()
        q_params = []
        for name, layer in quant_layers(ctx.model):
            old_w, old_a = layer.wq, layer.aq
            signed = bool(getattr(old_a, "observer", None) is not None and old_a.observer.lo is not None
                          and float(old_a.observer.lo.min()) < 0) if isinstance(old_a, Quantizer) else False
            new_w = self.make_weight_quantizer(layer, old_w)
            if new_w is not None:
                new_w.to(layer.weight.device).init_from(layer.weight)
                layer.wq = new_w
            new_a = self.make_act_quantizer(layer, old_a, signed)
            if new_a is not None and isinstance(old_a, Quantizer) and old_a.mode == "quant":
                layer.aq = new_a.to(layer.weight.device)
        self.recalibrate(ctx)
        for name, p in ctx.model.named_parameters():
            if name.split(".")[-1] in ("s", "beta", "alpha"):
                q_params.append(p)
        ctx.info["learnable_quant_params"] = len(q_params)
        extra = [{"params": q_params, "lr": self.hp["q_lr"], "weight_decay": self.q_weight_decay()}] if q_params else None
        self.finetune(ctx, extra_groups=extra)
        for _, m in ctx.model.named_modules():
            if hasattr(m, "sync_buffers"):
                m.sync_buffers()
        from .core import record_meta

        record_meta(ctx.model)

    def q_weight_decay(self):
        return 0.0

    def recalibrate(self, ctx):
        """Calibrate the new activation quantisers (they start in ``off`` mode)."""
        from .core import calibrate

        new = [m.aq for _, m in quant_layers(ctx.model) if getattr(m.aq, "mode", "") == "off"]
        if new:
            calibrate(ctx.model, ctx.data.calib(128, limit=self.hp["calib_batches"]), new)


@register_method("qat_ste", kind="quant", family="QAT", paper="Jacob et al., 2018; Bengio et al., 2013 (STE)")
class QATSTE(QAT):
    """Fixed uniform quantisers; weight ranges follow the weights every step and activation ranges an EMA
    of batch min/max while training."""

    def make_weight_quantizer(self, layer, old):
        return DynamicWeight(old.bits, per_channel=old.per_channel)

    def make_act_quantizer(self, layer, old, signed):
        old.ema_training = True
        return None


class DynamicWeight(Quantizer):
    def __init__(self, bits=8, per_channel=True):
        super().__init__(bits, symmetric=True, per_channel=per_channel, ch_axis=0)

    def forward(self, w):
        if self.training or self.mode != "quant":
            with torch.no_grad():
                self.reset_observer()
                self.observer.update(w)
                lo, hi = self.observer.compute(self.qmin, self.qmax, True)
                self.set_range(lo, hi)
            self.mode = "quant"
        return super().forward(w)


@register_method("lsq", kind="quant", family="QAT", paper="Esser et al., 2020 (LSQ)")
class LSQ(QAT):
    hparams = {"w_bits": 4, "a_bits": 4, "exclude": ["first", "last"]}

    def make_weight_quantizer(self, layer, old):
        return LSQQuantizer(self.hp["w_bits"], symmetric=True, per_channel=self.hp["w_per_channel"])

    def make_act_quantizer(self, layer, old, signed):
        return LSQQuantizer(self.hp["a_bits"], symmetric=signed, per_channel=False)


@register_method("lsq_plus", kind="quant", family="QAT", paper="Bhalgat et al., 2020 (LSQ+)")
class LSQPlus(LSQ):
    def make_act_quantizer(self, layer, old, signed):
        return LSQQuantizer(self.hp["a_bits"], symmetric=False, per_channel=False, learn_offset=True)


@register_method("pact", kind="quant", family="QAT", paper="Choi et al., 2018 (PACT)")
class PACT(QAT):
    """PACT activations, per-channel min/max weights (the paper pairs PACT with DoReFa weights; see
    ``dorefa``). alpha gets L2 decay ``alpha_decay``."""

    hparams = {"w_bits": 4, "a_bits": 4, "exclude": ["first", "last"], "alpha_decay": 1e-4, "q_lr": 0.01}

    def make_weight_quantizer(self, layer, old):
        return DynamicWeight(self.hp["w_bits"], per_channel=True)

    def make_act_quantizer(self, layer, old, signed):
        return PACTQuantizer(self.hp["a_bits"], signed=signed)

    def q_weight_decay(self):
        return self.hp["alpha_decay"]


@register_method("dorefa", kind="quant", family="QAT", paper="Zhou et al., 2016 (DoReFa-Net)")
class DoReFa(QAT):
    """DoReFa weights with PACT activations (the PACT paper's setting). BatchNorm is not folded so it can
    absorb the weight scale, as in the paper."""

    hparams = {"w_bits": 4, "a_bits": 4, "exclude": ["first", "last"], "bn_fold": False, "q_lr": 0.01}

    def make_weight_quantizer(self, layer, old):
        return DoReFaWeight(self.hp["w_bits"])

    def make_act_quantizer(self, layer, old, signed):
        return PACTQuantizer(self.hp["a_bits"], signed=signed)


@register_method("apot", kind="quant", family="QAT", paper="Li et al., 2020 (Additive Powers-of-Two)")
class APoT(QAT):
    """APoT weights with learnable clipping; LSQ activations (the paper also uses APoT activations)."""

    hparams = {"w_bits": 4, "a_bits": 4, "exclude": ["first", "last"]}

    def make_weight_quantizer(self, layer, old):
        return APoTWeight(self.hp["w_bits"])

    def make_act_quantizer(self, layer, old, signed):
        return LSQQuantizer(self.hp["a_bits"], symmetric=signed, per_channel=False)
