"""Hardware acceleration shared by the CV hub (Mutli-Tool4Vision) and the RL hub (rlzoo).

THIS FILE IS BYTE-IDENTICAL IN BOTH PROJECTS. Edit one copy and copy it over the other
(``utils/hwaccel.py`` in the CV hub, ``rlzoo/common/hwaccel.py`` in rlzoo).

Three execution modes, selected with ``--parallel``:

``none``  one process, one device (``--device auto|cpu|cuda|cuda:N``).
``dp``    one process, ``nn.DataParallel`` over ``--gpu-ids`` (default: every visible GPU). The batch
          is the *total* batch and is split across GPUs. Needs >= 2 CUDA devices, otherwise it
          silently falls back to ``none``.
``ddp``   one process per GPU launched with ``torchrun --nproc_per_node=N``. ``torchrun`` is detected
          automatically, so ``--parallel ddp`` is optional. Batches are *per process*. Works on CPU
          with the gloo backend, which is handy for testing.

Two ways to parallelise a model:

* :meth:`Accelerator.prepare` wraps a model whose ``forward`` is called directly (CV classifiers).
  DDP's gradient hooks need the wrapper's ``forward`` to run, so this is the fast path.
* :meth:`Accelerator.prepare_tree` is for objects that call their networks through other methods
  (``dist()``, ``value()``, ``q_values()`` ... as RL agents do). Under ``dp`` every sub-module that has its
  own ``forward`` is wrapped in a :class:`DataParallelProxy`; under ``ddp`` parameters are broadcast from
  rank 0 and gradients are averaged explicitly with :meth:`Accelerator.sync_gradients` before each
  optimizer step. Either way the wrapped objects keep exposing the original attributes.
"""
from __future__ import annotations

import argparse
import contextlib
import os
import random
import warnings
from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Union

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DataParallel, DistributedDataParallel

try:
    import numpy as _np
except ImportError:  # pragma: no cover - numpy ships with torch, but stay importable without it
    _np = None

PARALLEL_MODES = ("none", "dp", "ddp")
_DEFAULT_MASTER_ADDR = "127.0.0.1"
_DEFAULT_MASTER_PORT = 29500


# --------------------------------------------------------------------------------------- config


def parse_gpu_ids(value) -> Optional[List[int]]:
    """``"0,1"`` / ``[0, 1]`` / ``None`` -> ``[0, 1]`` / ``None``."""
    if value is None or value == "" or value == []:
        return None
    if isinstance(value, str):
        return [int(x) for x in value.replace(";", ",").split(",") if x.strip() != ""]
    return [int(x) for x in value]


@dataclass
class AccelConfig:
    parallel: str = "none"  # none | dp | ddp
    device: str = "auto"  # auto | cpu | cuda | cuda:N  (ignored under ddp: each rank uses its local GPU)
    gpu_ids: Optional[List[int]] = None  # dp only
    sync_bn: bool = False  # ddp only: convert BatchNorm to SyncBatchNorm
    backend: Optional[str] = None  # ddp only: nccl (CUDA) or gloo (CPU); auto when None
    master_addr: str = _DEFAULT_MASTER_ADDR
    master_port: int = _DEFAULT_MASTER_PORT
    amp: bool = False  # mixed precision autocast (CUDA only)
    find_unused_parameters: bool = False  # ddp only

    def __post_init__(self):
        self.parallel = str(self.parallel or "none").lower()
        if self.parallel == "true":  # old scripts passed --ddp true
            self.parallel = "ddp"
        if self.parallel not in PARALLEL_MODES:
            raise ValueError(f"parallel must be one of {PARALLEL_MODES}, got '{self.parallel}'")
        self.device = str(self.device or "auto")
        self.gpu_ids = parse_gpu_ids(self.gpu_ids)
        self.master_port = int(self.master_port)

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "AccelConfig":
        data = dict(data or {})
        if data.pop("ddp", False):
            data["parallel"] = "ddp"
        known = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        unknown = sorted(set(data) - set(known))
        if unknown:
            raise ValueError(f"Unknown accel options {unknown}. Valid: {sorted(cls.__dataclass_fields__)}")
        return cls(**known)

    @classmethod
    def from_namespace(cls, ns: argparse.Namespace, base: Optional["AccelConfig"] = None) -> "AccelConfig":
        """Build from parsed CLI args. ``None`` values keep ``base`` (or dataclass) defaults, so a parser
        created with ``add_accel_args(..., none_defaults=True)`` only overrides what the user typed."""
        data = asdict(base) if base is not None else {}
        for key in cls.__dataclass_fields__:
            value = getattr(ns, key, None)
            if value is not None:
                data[key] = value
        if getattr(ns, "ddp", False):
            data["parallel"] = "ddp"
        return cls.from_dict(data)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def add_accel_args(parser: argparse.ArgumentParser, *, amp: bool = True, none_defaults: bool = False,
                   device_default: str = "auto") -> argparse.ArgumentParser:
    """Register the shared ``--device`` / ``--parallel`` / ... options on an argparse parser.

    With ``none_defaults=True`` every option defaults to ``None`` so that YAML/config values can be kept
    unless the flag is given explicitly (see :meth:`AccelConfig.from_namespace`).
    """
    d = (lambda v: None) if none_defaults else (lambda v: v)
    group = parser.add_argument_group("hardware acceleration (shared CV/RL module)")
    group.add_argument("--device", type=str, default=d(device_default),
                       help="auto (CUDA if available) | cpu | cuda | cuda:N. Ignored under ddp (one GPU per rank)")
    group.add_argument("--parallel", type=str, choices=list(PARALLEL_MODES), default=d("none"),
                       help="none: single device | dp: nn.DataParallel in one process (total batch split over "
                            "--gpu-ids) | ddp: DistributedDataParallel, one process per GPU via torchrun (batch is "
                            "per process). Launching with torchrun selects ddp automatically")
    group.add_argument("--ddp", action="store_true", default=False if not none_defaults else None,
                       help="Alias for --parallel ddp")
    group.add_argument("--gpu-ids", type=str, default=None,
                       help="GPUs for --parallel dp, e.g. 0,1,2 (default: all visible). Use CUDA_VISIBLE_DEVICES for ddp")
    group.add_argument("--sync-bn", action="store_true", default=False if not none_defaults else None,
                       help="Convert BatchNorm to SyncBatchNorm under ddp (CUDA only)")
    group.add_argument("--backend", type=str, default=None, help="ddp backend: nccl (CUDA, default) or gloo (CPU)")
    group.add_argument("--master-addr", type=str, default=d(_DEFAULT_MASTER_ADDR), help="ddp rendezvous address")
    group.add_argument("--master-port", type=int, default=d(_DEFAULT_MASTER_PORT), help="ddp rendezvous port")
    if amp:
        group.add_argument("--amp", action="store_true", default=False if not none_defaults else None,
                           help="Mixed precision (torch.autocast + GradScaler), CUDA only")
    return parser


# ------------------------------------------------------------------------------- process state


def launched_by_torchrun() -> bool:
    return int(os.environ.get("WORLD_SIZE", "1")) > 1 or "TORCHELASTIC_RUN_ID" in os.environ


def ddp_env() -> tuple:
    """(rank, world_size, local_rank) from the torchrun environment (all zero/one otherwise)."""
    return (int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1)),
            int(os.environ.get("LOCAL_RANK", 0)))


def is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    return dist.get_rank() if is_distributed() else 0


def get_world_size() -> int:
    return dist.get_world_size() if is_distributed() else 1


def is_main_process() -> bool:
    return get_rank() == 0


def _visible_gpus() -> int:
    return torch.cuda.device_count() if torch.cuda.is_available() else 0


def seed_everything(seed: int, rank_offset: bool = True) -> int:
    """Seed python / numpy / torch. With ``rank_offset`` each DDP rank gets ``seed + rank``."""
    seed = int(seed) + (get_rank() if rank_offset else 0)
    random.seed(seed)
    if _np is not None:
        _np.random.seed(seed % (2 ** 32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return seed


# -------------------------------------------------------------------------------------- wrappers


class _ProxyMixin:
    """Forward attribute access to the wrapped module so wrapped networks keep their public API."""

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            module = self.__dict__.get("_modules", {}).get("module")
            if module is None or name == "module":
                raise
            return getattr(module, name)


class DataParallelProxy(_ProxyMixin, DataParallel):
    pass


class DistributedDataParallelProxy(_ProxyMixin, DistributedDataParallel):
    pass


def has_own_forward(module: nn.Module) -> bool:
    return type(module).forward is not nn.Module.forward


def unwrap_model(model: nn.Module) -> nn.Module:
    while isinstance(model, (DataParallel, DistributedDataParallel)):
        model = model.module
    return model


def strip_module_prefix(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Remove the ``module.`` prefixes DP/DDP wrappers add to state-dict keys (any nesting depth)."""
    out = {}
    for key, value in state_dict.items():
        while key.startswith("module."):
            key = key[len("module."):]
        out[key] = value
    return out


def wrap_tree_data_parallel(module: nn.Module, device_ids: Sequence[int]) -> nn.Module:
    """Recursively wrap every sub-module that has its own ``forward`` in a :class:`DataParallelProxy`.

    Containers without a forward (``ModuleList``, ``ModuleDict``, plain holders) are descended into, so
    objects such as an actor-critic that call ``self.pi(x)`` / ``self.v(x)`` get parallel sub-networks.
    """
    if isinstance(module, (DataParallel, DistributedDataParallel)):
        return module
    if has_own_forward(module):
        return DataParallelProxy(module, device_ids=list(device_ids), output_device=device_ids[0])
    for name, child in list(module.named_children()):
        module._modules[name] = wrap_tree_data_parallel(child, device_ids)
    return module


# ------------------------------------------------------------------------------------ accelerator


class Accelerator:
    """Owns the process group, the device and the parallel wrappers for one training run.

    >>> acc = Accelerator(AccelConfig.from_namespace(opts), log=print)
    >>> acc.seed(opts.seed)
    >>> model = acc.prepare(model.to(acc.device))      # CV-style model (forward called directly)
    >>> with acc.autocast(): loss = criterion(model(x), y)
    >>> acc.close()
    """

    def __init__(self, cfg: Optional[AccelConfig] = None, log=None):
        self.cfg = cfg or AccelConfig()
        self.log = log or (lambda *_: None)
        self.notes: List[str] = []
        self.rank, self.world_size, self.local_rank = ddp_env()

        mode = self.cfg.parallel
        if mode == "none" and launched_by_torchrun():
            mode = "ddp"
            self.notes.append("torchrun detected; using --parallel ddp")
        self.parallel = mode
        self._owns_process_group = False
        if mode == "ddp":
            self._init_process_group()
        self.device = self._resolve_device()
        if self.device.type == "cuda":
            self._activate_cuda()
        if mode == "dp" and (self.device.type != "cuda" or len(self.gpu_ids) < 2):
            self.notes.append(f"--parallel dp needs >= 2 CUDA devices ({_visible_gpus()} visible); running on {self.device}")
            self.parallel = "none"
        self.amp = bool(self.cfg.amp) and self.device.type == "cuda"
        if self.cfg.amp and not self.amp:
            self.notes.append("--amp is only supported on CUDA; running in full precision")
        for note in self.notes:
            self.print(note)

    # ---------------------------------------------------------------- setup
    def _init_process_group(self) -> None:
        if is_distributed():
            return
        os.environ.setdefault("MASTER_ADDR", self.cfg.master_addr)
        os.environ.setdefault("MASTER_PORT", str(self.cfg.master_port))
        backend = self.cfg.backend or ("nccl" if torch.cuda.is_available() and self.cfg.device != "cpu" else "gloo")
        dist.init_process_group(backend=backend, rank=self.rank, world_size=self.world_size)
        self._owns_process_group = True
        self.rank, self.world_size = dist.get_rank(), dist.get_world_size()

    @property
    def gpu_ids(self) -> List[int]:
        ids = self.cfg.gpu_ids
        visible = _visible_gpus()
        if ids is None or visible == 0:
            return list(range(visible))
        for gpu in ids:
            if gpu >= visible:
                raise ValueError(f"--gpu-ids {ids}: cuda:{gpu} does not exist ({visible} visible)")
        return list(ids)

    def _resolve_device(self) -> torch.device:
        name = self.cfg.device
        cuda_ok = torch.cuda.is_available()
        if name.startswith("cpu") or not cuda_ok:
            if name.startswith("cuda") and not cuda_ok:
                self.notes.append(f"--device {name} requested but CUDA is not available; using cpu")
            return torch.device("cpu")
        visible = _visible_gpus()
        if self.parallel == "ddp":
            return torch.device(f"cuda:{self.local_rank % max(1, visible)}")
        if self.parallel == "dp":
            ids = self.gpu_ids
            return torch.device(f"cuda:{ids[0] if ids else 0}")
        if name == "auto":
            return torch.device("cuda:0")
        try:
            device = torch.device(name)
        except RuntimeError:
            self.notes.append(f"--device {name} is not a valid device; using cuda:0")
            return torch.device("cuda:0")
        if device.type != "cuda":
            return device
        # torch.cuda.set_device needs an index, and the index must exist
        index = 0 if device.index is None else device.index
        if index >= visible:
            self.notes.append(f"--device {name} does not exist ({visible} visible); using cuda:0")
            index = 0
        return torch.device(f"cuda:{index}")

    def _activate_cuda(self) -> None:
        """Select ``self.device`` and run one tiny kernel on it, so a broken CUDA setup (for example a
        PyTorch build without kernels for this GPU) fails here with a clear message instead of mid-training.
        Outside DDP the run falls back to the CPU; under DDP every rank must agree, so it raises."""
        try:
            torch.cuda.set_device(self.device)
            (torch.ones(1, device=self.device) + 1).item()
        except (RuntimeError, ValueError, AssertionError) as exc:
            msg = f"CUDA device {self.device} is not usable ({type(exc).__name__}: {str(exc).splitlines()[0]})"
            if self.parallel == "ddp":
                raise RuntimeError(msg) from exc
            self.notes.append(f"{msg}; using cpu. Check that the PyTorch CUDA build supports this GPU")
            self.device = torch.device("cpu")

    # ---------------------------------------------------------------- properties
    @property
    def is_main(self) -> bool:
        return self.rank == 0

    @property
    def is_distributed(self) -> bool:
        return self.parallel == "ddp" and is_distributed()

    @property
    def is_data_parallel(self) -> bool:
        return self.parallel == "dp"

    def seed(self, seed: int, same_on_all_ranks: bool = False) -> int:
        return seed_everything(seed, rank_offset=not same_on_all_ranks)

    def print(self, *args, **kwargs) -> None:
        if self.is_main:
            self.log(*args, **kwargs)

    def summary(self) -> str:
        text = f"device={self.device} parallel={self.parallel}"
        if self.parallel == "ddp":
            text += f" world_size={self.world_size} backend={dist.get_backend() if is_distributed() else '?'}"
        elif self.parallel == "dp":
            text += f" gpus={self.gpu_ids}"
        if self.amp:
            text += " amp=on"
        return text

    # ---------------------------------------------------------------- model wrapping
    def prepare(self, model: nn.Module, sync_bn: Optional[bool] = None, total_batch: Optional[int] = None) -> nn.Module:
        """Wrap a model whose ``forward`` is called directly (DDP hooks / DP scatter happen there)."""
        sync_bn = self.cfg.sync_bn if sync_bn is None else sync_bn
        if self.parallel == "ddp":
            if sync_bn:
                if self.device.type == "cuda":
                    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
                else:
                    self.print("--sync-bn needs CUDA; keeping regular BatchNorm")
            return DistributedDataParallelProxy(
                model,
                device_ids=[self.device.index] if self.device.type == "cuda" else None,
                find_unused_parameters=self.cfg.find_unused_parameters,
            )
        if self.parallel == "dp":
            if sync_bn:
                self.print("--sync-bn only applies to ddp; ignored")
            ids = self.gpu_ids
            self.print(f"DataParallel over cuda:{ids}"
                       + (f" (total batch {total_batch}, {total_batch // len(ids)} per GPU)" if total_batch else ""))
            return DataParallelProxy(model, device_ids=ids, output_device=ids[0])
        return model

    def prepare_tree(self, target: Union[nn.Module, Any]) -> Any:
        """Parallelise networks reached through arbitrary methods.

        ``target`` is either a module (returned wrapped) or any object whose ``nn.Module`` attributes are
        wrapped in place. Under ``ddp`` parameters/buffers are broadcast from rank 0 (so every rank starts
        from the same weights) and gradients must be averaged with :meth:`sync_gradients`.
        """
        if isinstance(target, nn.Module):
            if self.parallel == "dp":
                return wrap_tree_data_parallel(target, self.gpu_ids)
            if self.parallel == "ddp":
                self.broadcast(target)
            return target
        for name, value in list(vars(target).items()):
            if isinstance(value, nn.Module):
                setattr(target, name, self.prepare_tree(value))
            elif self.parallel == "ddp" and isinstance(value, torch.Tensor) and value.is_floating_point():
                self.broadcast(value)
        return target

    def unwrap(self, model: nn.Module) -> nn.Module:
        return unwrap_model(model)

    def state_dict(self, model: nn.Module) -> Dict[str, Any]:
        return strip_module_prefix(unwrap_model(model).state_dict())

    # ---------------------------------------------------------------- collectives
    @torch.no_grad()
    def broadcast(self, target: Union[nn.Module, torch.Tensor], src: int = 0) -> None:
        if not self.is_distributed:
            return
        if isinstance(target, nn.Module):
            for tensor in list(target.parameters()) + list(target.buffers()):
                dist.broadcast(tensor.data, src=src)
        else:
            dist.broadcast(target.data, src=src)

    def broadcast_object(self, obj: Any, src: int = 0) -> Any:
        if not self.is_distributed:
            return obj
        box = [obj]
        dist.broadcast_object_list(box, src=src)
        return box[0]

    @torch.no_grad()
    def sync_gradients(self, params: Union[nn.Module, Iterable[torch.Tensor]]) -> None:
        """Average ``.grad`` across ranks (no-op outside ddp). Call between ``backward()`` and ``step()``."""
        if not self.is_distributed:
            return
        if isinstance(params, nn.Module):
            params = params.parameters()
        grads = [p.grad for p in params if p.grad is not None]
        if not grads:
            return
        flat = torch.cat([g.reshape(-1) for g in grads])
        dist.all_reduce(flat, op=dist.ReduceOp.SUM)
        flat.div_(self.world_size)
        offset = 0
        for g in grads:
            n = g.numel()
            g.copy_(flat[offset:offset + n].view_as(g))
            offset += n

    def reduce(self, tensor: torch.Tensor, op: str = "sum") -> torch.Tensor:
        """All-reduce a tensor across ranks (returned as is outside ddp)."""
        if not self.is_distributed:
            return tensor
        ops = {"sum": dist.ReduceOp.SUM, "mean": dist.ReduceOp.SUM, "max": dist.ReduceOp.MAX, "min": dist.ReduceOp.MIN}
        out = tensor.detach().clone()
        dist.all_reduce(out, op=ops[op])
        if op == "mean":
            out = out / self.world_size
        return out

    def reduce_scalar(self, value: float, op: str = "mean") -> float:
        if not self.is_distributed:
            return float(value)
        return float(self.reduce(torch.tensor(float(value), device=self.device, dtype=torch.float64), op).item())

    def agree(self, flag: bool) -> bool:
        """True only if every rank passes True (keeps collective call counts identical across ranks)."""
        if not self.is_distributed:
            return bool(flag)
        return bool(self.reduce(torch.tensor(1.0 if flag else 0.0, device=self.device), "min").item() > 0.5)

    def any(self, flag: bool) -> bool:
        if not self.is_distributed:
            return bool(flag)
        return bool(self.reduce(torch.tensor(1.0 if flag else 0.0, device=self.device), "max").item() > 0.5)

    def gather_objects(self, obj: Any) -> List[Any]:
        if not self.is_distributed:
            return [obj]
        out = [None] * self.world_size
        dist.all_gather_object(out, obj)
        return out

    def barrier(self) -> None:
        if self.is_distributed:
            dist.barrier()

    # ---------------------------------------------------------------- mixed precision
    def autocast(self, enabled: Optional[bool] = None):
        enabled = self.amp if enabled is None else (enabled and self.device.type == "cuda")
        if not enabled:
            return contextlib.nullcontext()
        return torch.autocast(device_type=self.device.type, dtype=torch.float16)

    def grad_scaler(self, enabled: Optional[bool] = None) -> "torch.amp.GradScaler":
        enabled = self.amp if enabled is None else enabled
        return torch.amp.GradScaler(self.device.type if self.device.type == "cuda" else "cpu", enabled=bool(enabled))

    # ---------------------------------------------------------------- teardown
    def close(self) -> None:
        if self._owns_process_group and is_distributed():
            dist.destroy_process_group()
            self._owns_process_group = False

    def __enter__(self) -> "Accelerator":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


# ------------------------------------------------------------------------------------- launcher


def launch_processes(nproc: int, command: Sequence[str], master_addr: str = _DEFAULT_MASTER_ADDR,
                     master_port: int = _DEFAULT_MASTER_PORT, env: Optional[Dict[str, str]] = None) -> int:
    """Minimal ``torchrun`` replacement: start ``nproc`` copies of ``command`` with the DDP environment set.

    Useful where ``torchrun`` is unavailable or broken (e.g. Windows wheels built without libuv). Returns
    the first non-zero exit code (0 when every process succeeded).
    """
    import subprocess
    import sys

    base_env = dict(os.environ)
    base_env.update(env or {})
    base_env.update({"MASTER_ADDR": master_addr, "MASTER_PORT": str(master_port), "WORLD_SIZE": str(nproc),
                     "USE_LIBUV": base_env.get("USE_LIBUV", "0")})
    procs = []
    for rank in range(int(nproc)):
        rank_env = dict(base_env, RANK=str(rank), LOCAL_RANK=str(rank))
        procs.append(subprocess.Popen(list(command), env=rank_env))
    code = 0
    try:
        for p in procs:
            rc = p.wait()
            code = code or rc
    except KeyboardInterrupt:
        for p in procs:
            p.terminate()
        code = 130
    return code


def launch_main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry: ``python launch_ddp.py --nproc 2 [--master-port 29500] script.py --flag ...``"""
    import sys

    parser = argparse.ArgumentParser(description="Launch N DDP processes of a script (torchrun stand-in).")
    parser.add_argument("--nproc", "--nproc_per_node", "--nproc-per-node", type=int, default=_visible_gpus() or 2)
    parser.add_argument("--master-addr", type=str, default=_DEFAULT_MASTER_ADDR)
    parser.add_argument("--master-port", type=int, default=_DEFAULT_MASTER_PORT)
    parser.add_argument("script", type=str)
    parser.add_argument("args", nargs=argparse.REMAINDER)
    ns = parser.parse_args(argv)
    cmd = [sys.executable, ns.script] + list(ns.args)
    return launch_processes(ns.nproc, cmd, ns.master_addr, ns.master_port)


__all__ = [
    "PARALLEL_MODES", "AccelConfig", "Accelerator", "DataParallelProxy", "DistributedDataParallelProxy",
    "add_accel_args", "ddp_env", "get_rank", "get_world_size", "has_own_forward", "is_distributed",
    "is_main_process", "launch_main", "launch_processes", "launched_by_torchrun", "parse_gpu_ids",
    "seed_everything", "strip_module_prefix", "unwrap_model", "wrap_tree_data_parallel",
]
