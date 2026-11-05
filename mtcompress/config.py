"""YAML configs, command-line overrides and defaults.

Precedence: defaults < config file < ``--method/--model/...`` flags < ``--set KEY=VALUE`` pairs.
Values given to ``--set`` are parsed as YAML (``3e-4`` is a number, ``[2,4,8]`` a list).
"""
import copy
import glob
import os
from typing import Any, Dict, List, Optional

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_DIR = os.path.join(ROOT, "configs")

DEFAULTS: Dict[str, Any] = {
    "method": None,
    "model": "mobilevit",
    "dataset": "cifar10",
    "data_root": "./data",
    "checkpoint": "auto",  # checkpoints/<model>_<dataset>.pth, written by train.py
    "seed": 0,
    "device": "auto",
    "amp": True,
    "limit_batches": 0,  # smoke tests: batches per epoch / calibration pass
    "hparams": {},
    "measure": {
        "latency": True,  # PyTorch CPU latency at batch 1 (float models only)
        "onnx": True,  # ONNX Runtime CPU latency at batch 1 (int8 QDQ for 8-bit quantised models)
        "threads": 4,
        "iters": 100,
        "gpu_latency": False,  # PyTorch GPU latency at batch ``gpu_batch``
        "gpu_batch": 256,
    },
    "save_dir": "runs",
    "save_model": True,
}


def _yaml_value(text: str) -> Any:
    try:
        return yaml.safe_load(text)
    except yaml.YAMLError:
        return text


def deep_update(base: dict, other: dict) -> dict:
    for k, v in (other or {}).items():
        if isinstance(v, dict) and isinstance(base.get(k), dict) and k != "hparams":
            deep_update(base[k], v)
        elif k == "hparams" and isinstance(v, dict):
            base.setdefault("hparams", {}).update(v)
        else:
            base[k] = copy.deepcopy(v)
    return base


def set_dotted(cfg: dict, key: str, value: Any) -> None:
    """``a.b.0.c`` sets cfg['a']['b'][0]['c']; integer parts index lists."""
    parts = key.split(".")
    node = cfg
    for p in parts[:-1]:
        node = node[int(p)] if isinstance(node, list) else node.setdefault(p, {})
    last = parts[-1]
    if isinstance(node, list):
        node[int(last)] = value
    else:
        node[last] = value


def find_config(name: str) -> Optional[str]:
    """configs/<kind>/<name>.yaml for a method or variant name, if it exists."""
    hits = glob.glob(os.path.join(CONFIG_DIR, "*", f"{name}.yaml"))
    return hits[0] if hits else None


def load_yaml(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def build_config(config: Optional[str] = None, method: Optional[str] = None, overrides: Optional[dict] = None,
                 sets: Optional[List[str]] = None) -> dict:
    cfg = copy.deepcopy(DEFAULTS)
    path = config
    if path is None and method is not None:
        path = find_config(method)
    if path is not None:
        deep_update(cfg, load_yaml(path))
        cfg["config_file"] = os.path.relpath(path, ROOT) if os.path.isabs(path) else path
    # A variant config found by name (configs/quant/ptq_minmax_w4a8.yaml) names its own method;
    # with an explicit --config, --method overrides the file.
    if method is not None and (config is not None or not cfg.get("method")):
        cfg["method"] = method
    if path is not None and config is None:
        cfg["variant"] = os.path.splitext(os.path.basename(path))[0]
    elif config is not None:
        cfg["variant"] = os.path.splitext(os.path.basename(config))[0]
    else:
        cfg["variant"] = method
    for k, v in (overrides or {}).items():
        if v is not None:
            set_dotted(cfg, k, v)
    for item in sets or []:
        if "=" not in item:
            raise ValueError(f"--set expects KEY=VALUE, got '{item}'")
        k, v = item.split("=", 1)
        set_dotted(cfg, k.strip(), _yaml_value(v))
    if not cfg.get("method"):
        raise ValueError("no method: pass --method NAME or a config with 'method:'")
    return cfg


def save_yaml(cfg: dict, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
