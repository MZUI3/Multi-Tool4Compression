"""Build a run from a config, apply the method, measure before and after, save the run directory.

A run directory holds:

- ``config.yaml``: the resolved config with every hparam default filled in.
- ``model.pt``: the compressed model as a pickled module (``torch.load(..., weights_only=False)``),
  because structured pruning and quantisation change the architecture.
- ``progress.jsonl``: one line per fine-tuning epoch.
- ``summary.json``: baseline and final accuracy and cost, method-specific details, hardware.
"""
import copy
import datetime as _dt
import json
import os
import platform
import sys
import time
from typing import Optional

import torch
import torch.nn as nn

from .base import Context
from .common.hwaccel import AccelConfig, Accelerator, seed_everything
from .config import ROOT, save_yaml
from .data import CifarData
from .measure.profile import (count_macs, onnx_latencies, param_stats, storage_bytes, torch_latency_ms)
from .registry import get_method

if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


PROFILE_VERSION = 2  # bump when profile() changes, so cached baseline numbers are re-measured


def checkpoint_path(model: str, dataset: str) -> str:
    return os.path.join(ROOT, "checkpoints", f"{model}_{dataset}.pth")


def build_model(name: str, num_classes: int, img_size: int = 32) -> nn.Module:
    import models  # Multi-Tool4Vision registry, copied into this repository

    return models.create_model(name, num_classes=num_classes, img_size=img_size)


def load_baseline(cfg: dict, data: CifarData, device) -> nn.Module:
    model = build_model(cfg["model"], data.num_classes).to(device)
    path = cfg["checkpoint"]
    if path in (None, "", "none"):
        return model
    if path == "auto":
        path = checkpoint_path(cfg["model"], cfg["dataset"])
    if not os.path.isfile(path):
        raise FileNotFoundError(f"baseline checkpoint {path} not found. Train it first: "
                                f"python train.py --model {cfg['model']} --dataset {cfg['dataset']}")
    with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
        state = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(state.get("model_state", state))
    return model


def has_quantizers(model: nn.Module) -> bool:
    from .quant.core import QUANT_TYPES

    return any(isinstance(m, QUANT_TYPES) for m in model.modules())


def is_uniform_int8(model: nn.Module) -> bool:
    """Every quantiser is an 8-bit uniform one (or float): the model maps onto ONNX Runtime int8 kernels."""
    for m in model.modules():
        if not (hasattr(m, "mode") and hasattr(m, "observe")):
            continue
        if getattr(m, "mode", "off") == "off" or getattr(m, "bits", 32) >= 32:
            continue
        if not getattr(m, "exportable", False) or getattr(m, "bits", 0) != 8:
            return False
    return True


def clear_runtime_state(model: nn.Module) -> None:
    for m in model.modules():
        for attr in ("last_keys", "last_cls_attn"):
            if getattr(m, attr, None) is not None:
                setattr(m, attr, None)
        if hasattr(m, "observer") and not isinstance(getattr(m, "observer"), nn.Module):
            m.observer = None
        if hasattr(m, "rows"):
            m.rows = []
        state = getattr(m, "state", None)
        if state is not None and hasattr(state, "size"):
            state.size = None


def profile(model: nn.Module, data: CifarData, mcfg: dict, log=print) -> dict:
    model.eval()
    clear_runtime_state(model)
    example = data.example_input(1)
    out = {}
    out.update(param_stats(model))
    out.update(storage_bytes(model))
    out.update(count_macs(model, example))
    quantized = has_quantizers(model)
    if mcfg.get("latency"):
        if quantized:
            out["torch_cpu_latency_ms"] = None
            out["torch_cpu_latency_note"] = "fake-quantised model: PyTorch timing would measure emulation"
        else:
            out["torch_cpu_latency_ms"] = round(torch_latency_ms(model, example.cpu(), "cpu", mcfg["threads"],
                                                                 iters=mcfg["iters"]), 4)
    if mcfg.get("gpu_latency") and torch.cuda.is_available() and not quantized:
        xb = data.example_input(mcfg["gpu_batch"])
        out["torch_gpu_latency_ms"] = round(torch_latency_ms(model, xb, "cuda", iters=mcfg["iters"]), 4)
    if mcfg.get("onnx"):
        # Float models: ONNX Runtime fp32, plus the same graph in int8 (static QDQ) for reference.
        # 8-bit uniform quantised models: the int8 graph is what would be deployed, so that is the latency.
        # Other bit widths have no ONNX Runtime CPU kernel and get no latency.
        calib = [x.cpu() for x, _ in data.calib(16, limit=2)]
        int8_ok = (not quantized) or is_uniform_int8(model)
        try:
            if quantized and not int8_ok:
                out["onnx_latency_ms"] = None
                out["onnx_note"] = "bit widths or quantisers without an ONNX Runtime CPU kernel"
            else:
                lat = onnx_latencies(model, example.cpu(), calib, mcfg["threads"], fp32=not quantized, int8=True)
                lat = {k: round(v, 4) if isinstance(v, float) else v for k, v in lat.items()}
                out.update(lat)
                out["onnx_latency_ms"] = lat["onnx_int8_latency_ms"] if quantized else lat["onnx_fp32_latency_ms"]
                out["onnx_int8"] = quantized
        except Exception as exc:  # unsupported ops
            out["onnx_latency_ms"] = None
            out["onnx_note"] = f"{type(exc).__name__}: {str(exc).splitlines()[0][:200]}"
    clear_runtime_state(model)
    return out


def _baseline_profile(cfg, model, data, log):
    """Baseline numbers are cached next to the checkpoint (same file, same measure settings)."""
    ckpt = cfg["checkpoint"] if cfg["checkpoint"] != "auto" else checkpoint_path(cfg["model"], cfg["dataset"])
    cache = None
    key = None
    if ckpt and os.path.isfile(ckpt):
        cache = ckpt + ".profile.json"
        key = {"mtime": os.path.getmtime(ckpt), "measure": cfg["measure"], "torch": str(torch.__version__),
               "profile_version": PROFILE_VERSION}
        if os.path.isfile(cache):
            with open(cache, encoding="utf-8") as f:
                saved = json.load(f)
            if saved.get("key") == json.loads(json.dumps(key)):
                return saved["profile"]
    prof = profile(model, data, cfg["measure"], log)
    if cache:
        with open(cache, "w", encoding="utf-8") as f:
            json.dump({"key": key, "profile": prof}, f, indent=1)
    return prof


def hardware() -> dict:
    info = {"python": platform.python_version(), "torch": torch.__version__, "platform": platform.platform(),
            "cpu": platform.processor()}
    if torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
    try:
        import onnxruntime

        info["onnxruntime"] = onnxruntime.__version__
    except ImportError:
        pass
    return info


def run(cfg: dict, run_dir: Optional[str] = None, log=print) -> dict:
    spec = get_method(cfg["method"])
    method = spec.cls(cfg.get("hparams") or {})
    cfg = copy.deepcopy(cfg)
    cfg["method"] = spec.name
    cfg["hparams"] = method.hp

    acc = Accelerator(AccelConfig(device=cfg["device"], amp=cfg["amp"]), log=log)
    seed_everything(cfg["seed"], rank_offset=False)
    device = acc.device
    data = CifarData(cfg["dataset"], cfg["data_root"], device, seed=cfg["seed"])
    model = load_baseline(cfg, data, device)
    ctx = Context(model=model, data=data, device=device, amp=acc.amp, log=log, limit_batches=cfg["limit_batches"],
                  build_model=lambda: build_model(cfg["model"], data.num_classes).to(device))
    teacher = copy.deepcopy(model).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    ctx.teacher = teacher

    if run_dir is None:
        stamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        run_dir = os.path.join(cfg["save_dir"], f"{cfg.get('variant') or spec.name}_{cfg['model']}_s{cfg['seed']}_{stamp}")
    os.makedirs(run_dir, exist_ok=True)
    save_yaml(cfg, os.path.join(run_dir, "config.yaml"))

    log(f"[{spec.name}] {cfg['model']} on {cfg['dataset']}, {acc.summary()}")
    base_eval = ctx.evaluate(model)
    base_prof = _baseline_profile(cfg, model, data, log) if not cfg["limit_batches"] else profile(model, data, {**cfg["measure"], "latency": False, "onnx": False})
    log(f"baseline: acc {base_eval['acc']:.2f}%, params {base_prof['params']}, MACs {base_prof['macs']:.4g}")

    t0 = time.time()
    method.apply(ctx)
    wall = time.time() - t0
    final_eval = ctx.evaluate()
    measure = cfg["measure"] if not cfg["limit_batches"] else {**cfg["measure"], "latency": False, "onnx": False}
    final_prof = profile(ctx.model, data, measure, log)
    # Relative to the uncompressed baseline (a structurally pruned model is smaller even in fp32).
    final_prof["compression_ratio"] = base_prof["size_bytes"] / max(final_prof["size_bytes"], 1.0)
    log(f"final: acc {final_eval['acc']:.2f}%, size {final_prof['size_bytes'] / 1024:.1f} KiB "
        f"(x{final_prof['compression_ratio']:.2f}), MACs {final_prof['macs']:.4g}")

    summary = {
        "method": spec.name,
        "variant": cfg.get("variant"),
        "kind": spec.kind,
        "family": spec.family,
        "paper": spec.paper,
        "model": cfg["model"],
        "dataset": cfg["dataset"],
        "seed": cfg["seed"],
        "baseline": {**base_eval, **base_prof},
        "final": {**final_eval, **final_prof},
        "delta_acc": round(final_eval["acc"] - base_eval["acc"], 3),
        "wall_time_s": round(wall, 1),
        "info": ctx.info,
        "hardware": hardware(),
    }
    with open(os.path.join(run_dir, "progress.jsonl"), "w", encoding="utf-8") as f:
        for rec in ctx.history:
            f.write(json.dumps(rec) + "\n")
    with open(os.path.join(run_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=1, default=str)
    if cfg.get("save_model"):
        clear_runtime_state(ctx.model)
        torch.save(ctx.model.cpu(), os.path.join(run_dir, "model.pt"))
    summary["run_dir"] = run_dir
    return summary


def load_run(run_dir: str, device="cpu"):
    """(model, config, summary) of a saved run."""
    import yaml

    if ROOT not in sys.path:
        sys.path.insert(0, ROOT)
    model = torch.load(os.path.join(run_dir, "model.pt"), map_location=device, weights_only=False)
    with open(os.path.join(run_dir, "config.yaml"), encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    with open(os.path.join(run_dir, "summary.json"), encoding="utf-8") as f:
        summary = json.load(f)
    return model, cfg, summary
