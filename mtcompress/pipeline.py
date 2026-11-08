"""Pipelines: several methods applied one after another to the same model.

``stages`` is a list of ``{method: name, hparams: {...}}``. The teacher for distillation stays the
original uncompressed model across all stages. Each stage's results go to ``summary.json`` under
``stages``.
"""
from .base import Method
from .registry import get_method, register_method


class Pipeline(Method):
    kind = "pipeline"
    hparams = {"stages": []}

    def apply(self, ctx):
        if not self.hp["stages"]:
            raise ValueError("pipeline needs at least one stage")
        results = []
        outer = ctx.info
        for i, stage in enumerate(self.hp["stages"]):
            spec = get_method(stage["method"])
            method = spec.cls(stage.get("hparams", {}))
            ctx.info = {}
            ctx.log(f"--- stage {i}: {spec.name}")
            method.apply(ctx)
            stage_eval = ctx.evaluate()
            results.append({"method": spec.name, "hparams": method.hp, "eval": stage_eval, "info": ctx.info})
        ctx.info = outer
        ctx.info["stages"] = results


@register_method("pipeline", kind="pipeline", family="Pipeline", paper="")
class GenericPipeline(Pipeline):
    pass


@register_method("deep_compression", kind="pipeline", family="Pipeline",
                 paper="Han et al., 2016 (prune -> k-means sharing -> Huffman)")
class DeepCompression(Pipeline):
    hparams = {"stages": [
        {"method": "gmp", "hparams": {"sparsity": 0.8, "ft_epochs": 8, "exclude": ["first"]}},
        {"method": "kmeans_share", "hparams": {"bits": 5, "ft_epochs": 3}},
    ]}


@register_method("prune_quant", kind="pipeline", family="Pipeline",
                 paper="Channel pruning then int8 PTQ (both visible in ONNX Runtime latency)")
class PruneQuant(Pipeline):
    hparams = {"stages": [
        {"method": "l1_filter", "hparams": {"ratio": 0.3, "ft_epochs": 8}},
        {"method": "ptq_mse", "hparams": {}},
    ]}


@register_method("prune_quant_kd", kind="pipeline", family="Pipeline",
                 paper="As prune_quant, with distillation from the uncompressed model (Hinton et al., 2015)")
class PruneQuantKD(Pipeline):
    hparams = {"stages": [
        {"method": "l1_filter", "hparams": {"ratio": 0.5, "ft_epochs": 10, "kd_alpha": 0.7}},
        {"method": "lsq", "hparams": {"w_bits": 4, "a_bits": 8, "ft_epochs": 5, "kd_alpha": 0.7}},
    ]}
