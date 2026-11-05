"""Fake quantisation: quantiser modules, quantised Conv2d/Linear, attaching them to a model.

Every Conv2d and Linear (after attention layers are made explicit) becomes ``QConv2d`` / ``QLinear`` with
two quantisers: ``wq`` on the weight and ``aq`` on the layer input. This is the usual "weights and
layer inputs" placement; residual additions and normalisation outputs stay in float unless a method adds
quantisers there (FQ-ViT, PTQ4ViT).

Quantisers compute ``x_q = (clamp(round(x / s) + z, qmin, qmax) - z) * s`` with the straight-through
estimator for ``round``. Signed symmetric quantisation uses the restricted range [-(2^(b-1)-1), 2^(b-1)-1];
asymmetric quantisation uses [0, 2^b - 1]. In ``export_mode`` 8-bit uniform quantisers emit
``torch.fake_quantize_per_*_affine``, which the ONNX exporter turns into QuantizeLinear/DequantizeLinear.
"""
import contextlib
from typing import Callable, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base import resolve_exclude
from ..nn.attention import convert_attention
from .observers import Observer, range_to_qparams


class NotExportable(RuntimeError):
    pass


def round_ste(x: torch.Tensor) -> torch.Tensor:
    return (x.round() - x).detach() + x


def grad_scale(x: torch.Tensor, scale: float) -> torch.Tensor:
    return (x - x * scale).detach() + x * scale


class Quantizer(nn.Module):
    """Uniform affine quantiser with an observer for calibration.

    Modes: ``observe`` (collect statistics, pass x through), ``quant`` (fake-quantise), ``off``.
    ``ema_training``: while the module is training, ranges follow an EMA of batch min/max (QAT of
    Jacob et al., 2018)."""

    exportable = True

    def __init__(self, bits: int = 8, symmetric: bool = True, per_channel: bool = False, ch_axis: int = 0,
                 observer: str = "minmax", ema_training: bool = False, **obs_kwargs):
        super().__init__()
        self.bits = bits
        self.symmetric = symmetric
        self.per_channel = per_channel
        self.ch_axis = ch_axis
        self.observer_kind = observer
        self.obs_kwargs = obs_kwargs
        self.ema_training = ema_training
        if symmetric:
            self.qmin, self.qmax = -(2 ** (bits - 1) - 1), 2 ** (bits - 1) - 1
        else:
            self.qmin, self.qmax = 0, 2 ** bits - 1
        self.register_buffer("scale", torch.ones(1))
        self.register_buffer("zero_point", torch.zeros(1))
        self.mode = "off"
        self.observer = None
        self.export = False
        self.drop_prob = 0.0

    def extra_repr(self):
        return f"bits={self.bits}, symmetric={self.symmetric}, per_channel={self.per_channel}, mode={self.mode}"

    def reset_observer(self):
        self.observer = Observer(self.observer_kind, self.ch_axis if self.per_channel else None, **self.obs_kwargs)

    def observe(self):
        self.reset_observer()
        self.mode = "observe"

    @torch.no_grad()
    def init_from(self, x: torch.Tensor):
        """Set the range from one tensor (weights) and switch to ``quant``."""
        self.reset_observer()
        self.observer.update(x)
        self.finish()

    @torch.no_grad()
    def finish(self):
        lo, hi = self.observer.compute(self.qmin, self.qmax, self.symmetric)
        self.set_range(lo, hi)
        self.mode = "quant"

    @torch.no_grad()
    def set_range(self, lo, hi):
        scale, zp = range_to_qparams(lo, hi, self.qmin, self.qmax, self.symmetric)
        self.scale = scale.to(self.scale.device)
        self.zero_point = zp.to(self.scale.device)

    def _shape(self, x, t):
        if t.numel() == 1:
            return t
        shape = [1] * x.dim()
        shape[self.ch_axis] = -1
        return t.view(shape)

    def quantize(self, x):
        s, z = self._shape(x, self.scale).abs().clamp_min(1e-8), self._shape(x, self.zero_point)
        q = torch.clamp(round_ste(x / s) + z, self.qmin, self.qmax)
        return (q - z) * s

    def export_quantize(self, x):
        if not self.exportable or self.bits != 8:
            raise NotExportable(f"{type(self).__name__} with {self.bits} bits has no ONNX int8 equivalent")
        qmin, qmax = (-128, 127) if self.symmetric else (0, 255)
        if self.scale.numel() > 1:
            return torch.fake_quantize_per_channel_affine(x, self.scale.float(), self.zero_point.int(),
                                                          self.ch_axis, qmin, qmax)
        return torch.fake_quantize_per_tensor_affine(x, float(self.scale), int(self.zero_point), qmin, qmax)

    def forward(self, x):
        if self.mode == "off":
            return x
        if self.mode == "observe":
            self.observer.update(x)
            return x
        if self.ema_training and self.training:
            with torch.no_grad():
                if self.observer is None or self.observer.kind != "ema":
                    self.observer = Observer("ema", self.ch_axis if self.per_channel else None)
                    self.observer.lo, self.observer.hi = self._range()
                self.observer.update(x)
                self.set_range(self.observer.lo, self.observer.hi)
        if self.export:
            return self.export_quantize(x)
        q = self.quantize(x)
        if self.drop_prob > 0 and self.training:  # QDrop: keep a random part of the input in float
            return torch.where(torch.rand_like(x) < self.drop_prob, x, q)
        return q

    def _range(self):
        lo = (self.qmin - self.zero_point) * self.scale
        hi = (self.qmax - self.zero_point) * self.scale
        return lo.flatten(), hi.flatten()


class QConv2d(nn.Conv2d):
    @classmethod
    def from_float(cls, m: nn.Conv2d, wq: nn.Module, aq: nn.Module):
        q = cls.__new__(cls)
        nn.Module.__init__(q)
        q.__dict__.update({k: v for k, v in m.__dict__.items() if k not in ("_parameters", "_buffers", "_modules")})
        q._parameters = m._parameters
        q._buffers = m._buffers
        q._modules = dict(m._modules)
        q.wq, q.aq = wq, aq
        return q

    def forward(self, x):
        return self._conv_forward(self.aq(x), self.wq(self.weight), self.bias)


class QLinear(nn.Linear):
    @classmethod
    def from_float(cls, m: nn.Linear, wq: nn.Module, aq: nn.Module):
        q = cls.__new__(cls)
        nn.Module.__init__(q)
        q.__dict__.update({k: v for k, v in m.__dict__.items() if k not in ("_parameters", "_buffers", "_modules")})
        q._parameters = m._parameters
        q._buffers = m._buffers
        q._modules = dict(m._modules)
        q.wq, q.aq = wq, aq
        return q

    def forward(self, x):
        return F.linear(self.aq(x), self.wq(self.weight), self.bias)


QUANT_TYPES = (QConv2d, QLinear)


def quant_layers(model: nn.Module):
    return [(n, m) for n, m in model.named_modules() if isinstance(m, QUANT_TYPES)]


WFactory = Callable[[str, nn.Module], nn.Module]


def default_weight_quantizer(bits=8, per_channel=True, observer="minmax", symmetric=True):
    def make(name, layer):
        return Quantizer(bits, symmetric=symmetric, per_channel=per_channel, ch_axis=0, observer=observer)
    return make


def default_act_quantizer(bits=8, symmetric=False, observer="minmax", **kw):
    def make(name, layer):
        if bits >= 32:
            return Identityq()
        return Quantizer(bits, symmetric=symmetric, per_channel=False, observer=observer, **kw)
    return make


class Identityq(nn.Module):
    """A float placeholder with the quantiser interface."""

    bits = 32
    mode = "off"
    exportable = True

    def observe(self):
        pass

    def finish(self):
        pass

    def init_from(self, x):
        pass

    def forward(self, x):
        return x


def attach(model: nn.Module, wq: WFactory, aq: WFactory, exclude=()) -> Dict[str, nn.Module]:
    """Replace Conv2d/Linear with QConv2d/QLinear in place. ``wq(name, layer)`` and ``aq(name, layer)``
    build the quantisers. Returns {layer name: quantised layer}."""
    convert_attention(model)
    skip = resolve_exclude(model, exclude)
    replaced = {}
    for name, m in list(model.named_modules()):
        if not isinstance(m, (nn.Conv2d, nn.Linear)) or isinstance(m, QUANT_TYPES) or name in skip:
            continue
        if type(m).__name__ == "NonDynamicallyQuantizableLinear":
            continue
        w = wq(name, m)
        a = aq(name, m)
        cls = QConv2d if isinstance(m, nn.Conv2d) else QLinear
        q = cls.from_float(m, w, a).to(m.weight.device)
        parent_name, _, attr = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        parent._modules[attr] = q
        replaced[name] = q
    return replaced


@torch.no_grad()
def init_weights(model: nn.Module) -> None:
    for _, m in quant_layers(model):
        m.wq.init_from(m.weight)


@torch.no_grad()
def calibrate(model: nn.Module, batches, quantizers=None) -> None:
    """Run the calibration set with the given (default: all activation) quantisers observing, then fix
    their ranges. Weight quantisers stay as they are."""
    if quantizers is None:
        quantizers = [m.aq for _, m in quant_layers(model)]
        quantizers += [q for _, q in extra_quantizers(model)]
    for q in quantizers:
        q.observe()
    was_training = model.training
    model.eval()
    for x, _ in batches:
        model(x)
    for q in quantizers:
        if getattr(q, "mode", "off") == "observe":
            if q.observer is not None and q.observer.lo is not None:
                q.finish()
            else:
                q.mode = "off"
    model.train(was_training)


def extra_quantizers(model: nn.Module):
    """Quantisers installed outside QConv/QLinear (attention probabilities, LayerNorm inputs)."""
    out = []
    for n, m in model.named_modules():
        for attr in ("softmax_q", "post_q", "lnq"):
            q = getattr(m, attr, None)
            if q is not None and hasattr(q, "observe") and not isinstance(q, nn.Identity):
                out.append((f"{n}.{attr}", q))
    return out


def record_meta(model: nn.Module) -> None:
    """Store the weight bit widths and scale counts used by the storage calculation."""
    meta = getattr(model, "mtc_meta", None) or {}
    wbits = meta.setdefault("weight_bits", {})
    scales = meta.setdefault("scale_count", {})
    for name, m in quant_layers(model):
        bits = getattr(m.wq, "bits", 32)
        if bits < 32:
            key = f"{name}.weight" if name else "weight"
            wbits[key] = bits
            n = getattr(m.wq, "n_scales", None)
            if n is None:
                n = int(getattr(m.wq, "scale", torch.ones(1)).numel())
                n += 0 if getattr(m.wq, "symmetric", True) else n  # zero points
            scales[key] = int(n)
    model.mtc_meta = meta


@contextlib.contextmanager
def export_mode(model: nn.Module):
    qs = [m for m in model.modules() if hasattr(m, "export")]
    for q in qs:
        q.export = True
    try:
        yield model
    finally:
        for q in qs:
            q.export = False


def quant_summary(model: nn.Module) -> dict:
    layers = quant_layers(model)
    return {
        "quantized_layers": len(layers),
        "weight_bits": sorted({getattr(m.wq, "bits", 32) for _, m in layers}),
        "act_bits": sorted({getattr(m.aq, "bits", 32) for _, m in layers}),
    }
