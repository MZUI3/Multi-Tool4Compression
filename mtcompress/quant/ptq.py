"""Post-training quantisation: no training, a calibration set of 1024 training images.

Every PTQ method folds BatchNorm into the preceding convolution first (``bn_fold``), because a deployed
int8 model runs folded weights; quantising unfolded weights would measure a model nobody ships.
"""
import copy

import torch

from ..base import QuantMethod
from ..nn.attention import convert_attention
from ..registry import register_method
from .core import (attach, calibrate, default_act_quantizer, default_weight_quantizer, init_weights,
                   quant_layers, quant_summary, record_meta, Quantizer)
from .fold import bias_correction, cross_layer_equalize, fold_bn, relu6_to_relu


class PTQ(QuantMethod):
    hparams = {
        "w_observer": "minmax",
        "a_observer": "minmax",
        "w_per_channel": True,
        "a_symmetric": False,
        "bn_fold": True,
        "calib_batches": 8,
        "percentile": 99.99,
        # Extra quantisation points for transformers: the attention probabilities and the inputs of
        # LayerNorm (the residual stream). Off by default; FQ-ViT and PTQ4ViT turn them on with their
        # own quantisers, and turning them on here gives the uniform baseline they are compared with.
        "quant_attn_probs": False,
        "quant_ln_input": False,
        "attn_bits": 8,
    }

    def act_factory(self):
        kw = {"percentile": self.hp["percentile"]} if self.hp["a_observer"] == "percentile" else {}
        return default_act_quantizer(self.hp["a_bits"], self.hp["a_symmetric"], self.hp["a_observer"], **kw)

    def weight_factory(self):
        return default_weight_quantizer(self.hp["w_bits"], self.hp["w_per_channel"], self.hp["w_observer"])

    def preprocess(self, ctx, example):
        if self.hp["bn_fold"]:
            ctx.info["bn_folded"] = fold_bn(ctx.model, example)

    def install_extra(self, ctx):
        from .vit import install_ln_quantizers, install_softmax_quantizers

        if self.hp["quant_attn_probs"]:
            install_softmax_quantizers(ctx.model, lambda: Quantizer(self.hp["attn_bits"], symmetric=False,
                                                                    observer=self.hp["a_observer"]))
        if self.hp["quant_ln_input"]:
            install_ln_quantizers(ctx.model, lambda: Quantizer(self.hp["a_bits"], symmetric=True,
                                                               observer=self.hp["a_observer"]))

    def quantize(self, ctx):
        example = next(iter(ctx.data.calib(2)))[0]
        convert_attention(ctx.model)
        self.preprocess(ctx, example)
        attach(ctx.model, self.weight_factory(), self.act_factory(), self.hp["exclude"])
        self.install_extra(ctx)
        init_weights(ctx.model)
        self.after_weights(ctx)
        calibrate(ctx.model, ctx.data.calib(128, limit=self.hp["calib_batches"]))
        self.after_calibration(ctx)
        record_meta(ctx.model)
        ctx.info.update(quant_summary(ctx.model))

    def after_weights(self, ctx):
        pass

    def after_calibration(self, ctx):
        pass

    def apply(self, ctx):
        self.quantize(ctx)
        ctx.info["after_quant"] = ctx.evaluate()
        ctx.log(f"{self.name}: {ctx.info['after_quant']}")
        if self.hp["ft_epochs"] > 0:
            self.finetune(ctx)


@register_method("ptq_minmax", kind="quant", family="PTQ calibration", paper="Jacob et al., 2018 (min/max ranges)")
class PTQMinMax(PTQ):
    pass


@register_method("ptq_percentile", kind="quant", family="PTQ calibration",
                 paper="McKinstry et al., 2019; Wu et al., 2020 (percentile calibration)")
class PTQPercentile(PTQ):
    hparams = {"a_observer": "percentile"}


@register_method("ptq_mse", kind="quant", family="PTQ calibration", paper="Choukroun et al., 2019 (OMSE)")
class PTQMSE(PTQ):
    hparams = {"a_observer": "mse", "w_observer": "mse"}


@register_method("ptq_kl", kind="quant", family="PTQ calibration",
                 paper="Migacz, 2017 (TensorRT entropy calibration)")
class PTQKL(PTQ):
    """KL calibration is symmetric, so activations use a signed symmetric range."""

    hparams = {"a_observer": "kl", "a_symmetric": True}


@register_method("dfq", kind="quant", family="PTQ data-free",
                 paper="Nagel et al., 2019 (Data-Free Quantization: CLE + bias correction)")
class DFQ(PTQ):
    """BN folding, cross-layer equalisation of conv pairs joined by ReLU-like activations, then
    quantisation and bias correction. DFQ's bias correction is analytic (from BN statistics); this
    implementation uses the empirical variant on the calibration set. SiLU and GELU are not positively
    homogeneous, so pairs joined by them are left alone; ``relu6_to_relu`` swaps ReLU6 for ReLU first,
    as the paper does for MobileNetV2."""

    hparams = {"relu6_to_relu": True, "cle_iterations": 10, "bias_correction": True}

    def preprocess(self, ctx, example):
        if self.hp["relu6_to_relu"]:
            ctx.info["relu6_replaced"] = relu6_to_relu(ctx.model)
        super().preprocess(ctx, example)
        ctx.info["cle_pairs"] = cross_layer_equalize(ctx.model, example, self.hp["cle_iterations"])
        self._fp = copy.deepcopy(ctx.model).eval()

    def after_calibration(self, ctx):
        if self.hp["bias_correction"]:
            layers = [n for n, _ in quant_layers(ctx.model)]
            ctx.info["bias_corrected"] = bias_correction(ctx.model, self._fp, ctx.data.calib(128, limit=4), layers)
        del self._fp
