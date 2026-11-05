"""Write configs/<kind>/<name>.yaml: one default config per method plus the comparison variants used in
results/. Re-run after adding a method; existing files are overwritten.

    python tools/make_configs.py
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import yaml  # noqa: E402

from mtcompress.base import FINETUNE_DEFAULTS  # noqa: E402
from mtcompress.registry import list_methods  # noqa: E402

TRANSFORMERS = ["mobilevit", "vit_tiny"]
VIT = ["vit_tiny"]
APPLIES = {
    "tome": VIT, "evit": VIT, "head_prune": TRANSFORMERS, "fq_vit": TRANSFORMERS, "ptq4vit": TRANSFORMERS,
    "dfq": ["mobilevit", "resnet18", "mobilenet_v2"],
}

# name: (method, hparams, applies_to)
VARIANTS = {
    "prune": {
        "magnitude_90": ("magnitude", {"sparsity": 0.9, "ft_epochs": 10}, None),
        "magnitude_layerwise": ("magnitude", {"scope": "layer"}, None),
        "snip_trained": ("snip", {"from_scratch": False, "ft_epochs": 5, "ft_lr": 0.01, "ft_warmup_epochs": 0.0}, None),
        "imp_weight_rewind": ("imp", {"rewind": "weights"}, None),
        "l1_filter_30": ("l1_filter", {"ratio": 0.3}, None),
        "l1_filter_global": ("l1_filter", {"scope": "global"}, None),
        "l1_filter_kd": ("l1_filter", {"kd_alpha": 0.7}, None),
        "tome_r4": ("tome", {"r": 4}, VIT),
        "tome_r8_ft": ("tome", {"r": 8, "ft_epochs": 3}, VIT),
    },
    "quant": {
        "ptq_minmax_pertensor": ("ptq_minmax", {"w_per_channel": False}, None),
        "ptq_minmax_w4a8": ("ptq_minmax", {"w_bits": 4, "exclude": ["first", "last"]}, None),
        "ptq_mse_w4a8": ("ptq_mse", {"w_bits": 4, "exclude": ["first", "last"]}, None),
        "ptq_mse_w4a4": ("ptq_mse", {"w_bits": 4, "a_bits": 4, "exclude": ["first", "last"]}, None),
        "brecq_w4a4": ("brecq", {"a_bits": 4}, None),
        "ptq_minmax_vit_full": ("ptq_minmax", {"quant_ln_input": True, "quant_attn_probs": True, "attn_bits": 4},
                                TRANSFORMERS),
        "qat_ste_w4a4": ("qat_ste", {"w_bits": 4, "a_bits": 4, "exclude": ["first", "last"]}, None),
        "lsq_w2a2": ("lsq", {"w_bits": 2, "a_bits": 2, "ft_epochs": 10}, None),
        "lsq_w8a8": ("lsq", {"w_bits": 8, "a_bits": 8, "exclude": []}, None),
        "mixed_sensitivity_3bit": ("mixed_sensitivity", {"avg_bits": 3.0}, None),
        "hawq_3bit": ("hawq", {"avg_bits": 3.0}, None),
    },
    "pipeline": {},
}


def own_hparams(cls):
    """The hparams a method declares beyond the shared fine-tuning set, with their defaults."""
    hp = cls.default_hparams()
    return {k: v for k, v in hp.items() if k not in FINETUNE_DEFAULTS or k == "ft_epochs"}


def write(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(yaml.safe_dump(data, sort_keys=False, allow_unicode=True))


def main():
    count = 0
    for spec in list_methods():
        data = {"method": spec.name, "model": "mobilevit", "dataset": "cifar10"}
        if spec.name in APPLIES:
            data["applies_to"] = APPLIES[spec.name]
        if spec.name == "pipeline":
            data["hparams"] = {"stages": [{"method": "magnitude", "hparams": {"sparsity": 0.5}},
                                          {"method": "ptq_minmax", "hparams": {}}]}
        else:
            data["hparams"] = own_hparams(spec.cls)
        write(os.path.join(ROOT, "configs", spec.kind, f"{spec.name}.yaml"), data)
        count += 1
    for kind, variants in VARIANTS.items():
        for name, (method, hp, applies) in variants.items():
            data = {"method": method, "model": "mobilevit", "dataset": "cifar10"}
            if applies or method in APPLIES:
                data["applies_to"] = applies or APPLIES[method]
            data["hparams"] = hp
            write(os.path.join(ROOT, "configs", kind, f"{name}.yaml"), data)
            count += 1
    print(f"wrote {count} configs")


if __name__ == "__main__":
    main()
