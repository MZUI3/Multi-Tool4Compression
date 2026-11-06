"""PTQ for vision transformers: quantising the parts uniform quantisers handle badly.

- LayerNorm inputs (the residual stream) have a few channels with ranges far larger than the rest.
  FQ-ViT's Power-of-Two Factor gives channel c the scale s * 2^alpha_c with one shared base scale.
- Attention probabilities are concentrated near 0 with a long tail to 1. FQ-ViT's Log-Int-Softmax
  quantises them on a log2 grid; PTQ4ViT's twin uniform quantiser uses a fine and a coarse range.
- Post-GELU activations are asymmetric (a small negative part). PTQ4ViT gives the negative and positive
  parts separate scales.

The comparison point is ``ptq_minmax`` with ``quant_ln_input`` / ``quant_attn_probs`` turned on, which
puts plain uniform quantisers at the same places. None of these quantisers maps to ONNX int8 ops, so
these runs report no ONNX Runtime latency.
"""
import torch
import torch.nn as nn

from ..nn.attention import SplitAttention
from ..registry import register_method
from .core import round_ste
from .ptq import PTQ

MAX_ROWS = 4096


class QLayerNorm(nn.LayerNorm):
    @classmethod
    def from_float(cls, m: nn.LayerNorm, q: nn.Module):
        new = cls.__new__(cls)
        nn.Module.__init__(new)
        new.__dict__.update({k: v for k, v in m.__dict__.items() if k not in ("_parameters", "_buffers", "_modules")})
        new._parameters = m._parameters
        new._buffers = m._buffers
        new.lnq = q
        return new

    def forward(self, x):
        return super().forward(self.lnq(x))


def install_ln_quantizers(model: nn.Module, factory) -> int:
    n = 0
    for parent in list(model.modules()):
        for k, child in list(parent._modules.items()):
            if type(child) is nn.LayerNorm:
                device = child.weight.device if child.weight is not None else "cpu"
                parent._modules[k] = QLayerNorm.from_float(child, factory().to(device))
                n += 1
    return n


def install_softmax_quantizers(model: nn.Module, factory) -> int:
    n = 0
    for m in model.modules():
        if isinstance(m, SplitAttention):
            m.softmax_q = factory().to(m.qkv.weight.device)
            n += 1
    return n


class _Seen:
    lo = None


class _SampleQuantizer(nn.Module):
    """Base for quantisers calibrated from a sample of rows (tokens x channels)."""

    exportable = False

    def __init__(self, bits):
        super().__init__()
        self.bits = bits
        self.mode = "off"
        self.rows = []
        self.export = False
        self.observer = None

    def observe(self):
        self.rows = []
        self.mode = "observe"
        self.observer = _Seen()  # calibrate() checks observer.lo

    def update(self, x):
        r = x.detach().reshape(-1, x.shape[-1]).float()
        if r.shape[0] > MAX_ROWS:
            r = r[torch.randint(0, r.shape[0], (MAX_ROWS,), device=r.device)]
        self.rows.append(r)
        self.observer.lo = True

    def forward(self, x):
        if self.mode == "observe":
            self.update(x)
            return x
        if self.mode == "off":
            return x
        if self.export:
            raise RuntimeError(f"{type(self).__name__} has no ONNX int8 equivalent")
        return self.quantize(x)


class PTFQuantizer(_SampleQuantizer):
    """Power-of-Two Factor (Lin et al., 2022, FQ-ViT) on the last dimension."""

    def __init__(self, bits=8, max_factor=3):
        super().__init__(bits)
        self.K = max_factor
        self.register_buffer("scale", torch.ones(1))
        self.register_buffer("zero_point", torch.zeros(1))

    @torch.no_grad()
    def finish(self):
        x = torch.cat(self.rows)
        self.rows = []
        qmax = 2 ** self.bits - 1
        lo, hi = x.min(), x.max()
        base = (hi - lo) / qmax / (2 ** self.K)
        best_err = torch.full((x.shape[1],), float("inf"), device=x.device)
        best_scale = torch.full((x.shape[1],), float(base), device=x.device)
        best_zp = torch.zeros(x.shape[1], device=x.device)
        cmin = x.min(0).values
        for a in range(self.K + 1):
            s = base * 2 ** a
            zp = torch.clamp(torch.round(-torch.minimum(cmin, torch.zeros_like(cmin)) / s), 0, qmax)
            q = (torch.clamp(torch.round(x / s) + zp, 0, qmax) - zp) * s
            err = (q - x).pow(2).mean(0)
            better = err < best_err
            best_err = torch.where(better, err, best_err)
            best_scale = torch.where(better, s.expand_as(best_scale), best_scale)
            best_zp = torch.where(better, zp, best_zp)
        self.scale, self.zero_point = best_scale, best_zp
        self.mode = "quant"

    def quantize(self, x):
        qmax = 2 ** self.bits - 1
        return (torch.clamp(round_ste(x / self.scale) + self.zero_point, 0, qmax) - self.zero_point) * self.scale


class Log2Quantizer(_SampleQuantizer):
    """Log-Int-Softmax style log2 quantiser for probabilities in [0, 1] (FQ-ViT): q = round(-log2 p),
    clamped to [0, 2^b - 1], dequantised as 2^-q. Codes at the top of the range map to zero."""

    def finish(self):
        self.rows = []
        self.mode = "quant"

    def quantize(self, p):
        qmax = 2 ** self.bits - 1
        code = torch.clamp(round_ste(-torch.log2(p.clamp_min(1e-30))), 0, qmax)
        out = torch.pow(2.0, -code)
        return torch.where(code >= qmax, torch.zeros_like(out), out)


class TwinUniform(_SampleQuantizer):
    """PTQ4ViT twin uniform quantiser (Yuan et al., 2022). ``kind='softmax'``: two ranges [0, 2^(b-1) d1)
    and [0, 1] with d2 = 2^-(b-1), d1 = d2 / 2^k searched over k. ``kind='gelu'``: the negative part
    uses d1, the positive part d2, both searched. The search minimises the MSE of the quantised values
    (the paper uses a Hessian-guided metric)."""

    def __init__(self, bits=8, kind="softmax"):
        super().__init__(bits)
        self.kind = kind
        self.register_buffer("d1", torch.ones(1))
        self.register_buffer("d2", torch.ones(1))

    def _q(self, x, d1, d2):
        half = 2 ** (self.bits - 1)
        if self.kind == "softmax":
            fine = torch.clamp(round_ste(x / d1), 0, half - 1) * d1
            coarse = torch.clamp(round_ste(x / d2), 0, half - 1) * d2
            return torch.where(x < half * d1, fine, coarse)
        neg = torch.clamp(round_ste(x / d1), -half, 0) * d1
        pos = torch.clamp(round_ste(x / d2), 0, half - 1) * d2
        return torch.where(x < 0, neg, pos)

    @torch.no_grad()
    def finish(self):
        x = torch.cat(self.rows).flatten()
        self.rows = []
        half = 2 ** (self.bits - 1)
        best = (float("inf"), None, None)
        if self.kind == "softmax":
            d2 = torch.tensor(1.0 / (half - 1), device=x.device)
            for k in range(0, 16):
                d1 = d2 / 2 ** k
                err = (self._q(x, d1, d2) - x).pow(2).mean().item()
                if err < best[0]:
                    best = (err, d1, d2)
        else:
            mn, mx = x.min().abs().clamp_min(1e-6), x.max().clamp_min(1e-6)
            for i in range(1, 41):
                d2 = mx * (i / 40) / (half - 1)
                for j in (0.5, 0.75, 1.0):
                    d1 = mn * j / half
                    err = (self._q(x, d1, d2) - x).pow(2).mean().item()
                    if err < best[0]:
                        best = (err, d1, d2)
        self.d1 = best[1].reshape(1).to(x.device)
        self.d2 = best[2].reshape(1).to(x.device)
        self.mode = "quant"

    def quantize(self, x):
        return self._q(x, self.d1, self.d2)


def post_gelu_layers(model: nn.Module, example: torch.Tensor):
    """Quantised Linear/Conv layers whose input comes from a GELU (through Dropout at most)."""
    gelu_out, alias, inputs, keep = set(), {}, {}, []
    hooks = []

    # Hooks must return None: a returned value would replace the module's input or output.
    def on_gelu(mod, i, o):
        gelu_out.add(id(o))
        keep.append(o)

    def on_pass(mod, i, o):
        if o is not i[0]:
            alias[id(o)] = id(i[0])
        keep.append(o)

    def on_layer(mod, i):
        inputs[mod] = id(i[0])
        keep.append(i[0])

    for m in model.modules():
        if isinstance(m, nn.GELU):
            hooks.append(m.register_forward_hook(on_gelu))
        elif isinstance(m, (nn.Dropout, nn.Identity)):
            hooks.append(m.register_forward_hook(on_pass))
        elif hasattr(m, "aq") and hasattr(m, "wq"):
            hooks.append(m.register_forward_pre_hook(on_layer))
    with torch.no_grad():
        model.eval()(example)
    for h in hooks:
        h.remove()
    found = []
    for layer, tid in inputs.items():
        while tid in alias and tid not in gelu_out:
            tid = alias[tid]
        if tid in gelu_out:
            found.append(layer)
    return found


@register_method("fq_vit", kind="quant", family="PTQ transformer",
                 paper="Lin et al., 2022 (FQ-ViT: Power-of-Two Factor, Log-Int-Softmax)")
class FQViT(PTQ):
    """W8A8 with PTF on every LayerNorm input and 4-bit log2 attention probabilities."""

    hparams = {"attn_bits": 4, "ptf_max_factor": 3}

    def install_extra(self, ctx):
        install_ln_quantizers(ctx.model, lambda: PTFQuantizer(self.hp["a_bits"], self.hp["ptf_max_factor"]))
        install_softmax_quantizers(ctx.model, lambda: Log2Quantizer(self.hp["attn_bits"]))


@register_method("ptq4vit", kind="quant", family="PTQ transformer",
                 paper="Yuan et al., 2022 (PTQ4ViT: twin uniform quantisation)")
class PTQ4ViT(PTQ):
    """Twin uniform quantisers on the attention probabilities and on the inputs of layers fed by GELU;
    MSE-searched uniform ranges everywhere else."""

    hparams = {"a_observer": "mse", "w_observer": "mse"}

    def install_extra(self, ctx):
        install_softmax_quantizers(ctx.model, lambda: TwinUniform(self.hp["attn_bits"], "softmax"))
        example = next(iter(ctx.data.calib(2)))[0]
        layers = post_gelu_layers(ctx.model, example)
        for layer in layers:
            layer.aq = TwinUniform(self.hp["a_bits"], "gelu").to(layer.weight.device)
        ctx.info["post_gelu_layers"] = len(layers)
