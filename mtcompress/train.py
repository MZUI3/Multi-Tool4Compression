"""Training, fine-tuning and evaluation loops shared by every method.

``fit`` takes an optional ``hooks`` object. Any of these methods may be defined on it:

- ``on_step_begin(step, total_steps)``: before the forward pass (gradual pruning schedules).
- ``extra_loss(model) -> Tensor``: added to the loss (sparsity regularisers).
- ``after_backward(model)``: between ``backward`` and ``optimizer.step`` (gradient masking, score updates).
- ``after_step(model)``: after ``optimizer.step`` (re-applying pruning masks, clamping).
- ``on_epoch_end(epoch, model)``.
"""
import math
import time
from typing import Callable, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _autocast(device: torch.device, enabled: bool):
    return torch.autocast(device_type=device.type, dtype=torch.float16, enabled=enabled and device.type == "cuda")


@torch.no_grad()
def evaluate(model: nn.Module, batches, amp: bool = False) -> Dict[str, float]:
    was_training = model.training
    model.eval()
    correct, total, loss_sum = 0, 0, 0.0
    device = next(iter(model.parameters())).device
    for x, y in batches:
        with _autocast(device, amp):
            out = model(x)
        loss_sum += F.cross_entropy(out.float(), y, reduction="sum").item()
        correct += (out.argmax(1) == y).sum().item()
        total += y.numel()
    model.train(was_training)
    return {"acc": 100.0 * correct / max(total, 1), "loss": loss_sum / max(total, 1)}


def param_groups(model: nn.Module, weight_decay: float, extra: Optional[List[dict]] = None) -> List[dict]:
    """Weight decay on weight matrices and kernels only. Parameters listed in ``extra`` groups
    (for example learnable quantisation step sizes) are taken out of the default groups."""
    taken = set()
    groups = []
    for group in extra or []:
        params = [p for p in group["params"] if p.requires_grad]
        taken.update(id(p) for p in params)
        groups.append({**group, "params": params})
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad or id(p) in taken:
            continue
        (no_decay if p.ndim <= 1 or name.endswith(("cls_token", "pos_embed", "dist_token")) else decay).append(p)
    groups += [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}]
    return [g for g in groups if g["params"]]


def kd_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """Hinton et al., 2015: KL between softened distributions, scaled by T^2."""
    t = temperature
    return F.kl_div(F.log_softmax(student_logits / t, 1), F.softmax(teacher_logits / t, 1), reduction="batchmean") * t * t


def fit(
    model: nn.Module,
    data,
    epochs: float,
    lr: float,
    optimizer: str = "sgd",
    weight_decay: float = 5e-4,
    momentum: float = 0.9,
    warmup_epochs: float = 0.0,
    min_lr: float = 0.0,
    label_smoothing: float = 0.0,
    batch_size: int = 128,
    amp: bool = False,
    teacher: Optional[nn.Module] = None,
    kd_alpha: float = 0.0,
    kd_temperature: float = 4.0,
    grad_clip: float = 0.0,
    hooks=None,
    extra_groups: Optional[List[dict]] = None,
    limit_batches: int = 0,
    eval_every: int = 1,
    log: Callable = print,
    tag: str = "train",
) -> List[dict]:
    """Train ``model`` in place with a cosine schedule (per step) and optional distillation.

    ``epochs`` may be fractional (0.5 = half an epoch). Returns one record per evaluated epoch.
    """
    if epochs <= 0:
        return []
    device = next(iter(model.parameters())).device
    groups = param_groups(model, weight_decay, extra_groups)
    if optimizer == "sgd":
        opt = torch.optim.SGD(groups, lr=lr, momentum=momentum, nesterov=momentum > 0)
    elif optimizer == "adamw":
        opt = torch.optim.AdamW(groups, lr=lr)
    elif optimizer == "adam":
        opt = torch.optim.Adam(groups, lr=lr)
    else:
        raise ValueError(f"optimizer must be sgd, adamw or adam, got '{optimizer}'")
    base_lrs = [g.get("lr", lr) for g in opt.param_groups]

    loader = data.train(batch_size, limit_batches)
    steps_per_epoch = len(loader)
    total = max(1, int(round(epochs * steps_per_epoch)))
    warmup = int(round(warmup_epochs * steps_per_epoch))
    scaler = torch.amp.GradScaler(device.type, enabled=amp and device.type == "cuda")
    if teacher is not None:
        teacher.eval()
    history = []
    step, epoch = 0, 0
    start = time.time()
    model.train()
    while step < total:
        loss_sum, correct, seen = 0.0, 0, 0
        for x, y in loader:
            if step >= total:
                break
            if step < warmup:
                factor = (step + 1) / warmup
            else:
                progress = (step - warmup) / max(1, total - warmup)
                factor = None
                for g, base in zip(opt.param_groups, base_lrs):
                    g["lr"] = min_lr + (base - min_lr) * 0.5 * (1 + math.cos(math.pi * progress))
            if factor is not None:
                for g, base in zip(opt.param_groups, base_lrs):
                    g["lr"] = base * factor
            if hooks is not None and hasattr(hooks, "on_step_begin"):
                hooks.on_step_begin(step, total)

            with _autocast(device, amp):
                out = model(x)
                loss = F.cross_entropy(out, y, label_smoothing=label_smoothing)
                if teacher is not None and kd_alpha > 0:
                    with torch.no_grad():
                        t_out = teacher(x)
                    loss = (1 - kd_alpha) * loss + kd_alpha * kd_loss(out.float(), t_out.float(), kd_temperature)
            if hooks is not None and hasattr(hooks, "extra_loss"):
                loss = loss + hooks.extra_loss(model)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            if hooks is not None and hasattr(hooks, "after_backward"):
                hooks.after_backward(model)
            if grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(opt)
            scaler.update()
            if hooks is not None and hasattr(hooks, "after_step"):
                hooks.after_step(model)

            loss_sum += loss.item() * y.numel()
            correct += (out.argmax(1) == y).sum().item()
            seen += y.numel()
            step += 1
        epoch += 1
        if hooks is not None and hasattr(hooks, "on_epoch_end"):
            hooks.on_epoch_end(epoch, model)
        last = step >= total
        record = {"tag": tag, "epoch": epoch, "step": step, "train_loss": loss_sum / max(seen, 1),
                  "train_acc": 100.0 * correct / max(seen, 1), "lr": opt.param_groups[0]["lr"],
                  "time": round(time.time() - start, 1)}
        if last or (eval_every and epoch % eval_every == 0):
            record.update({f"val_{k}": v for k, v in evaluate(model, data.val(), amp).items()})
            model.train()
        history.append(record)
        log(" ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in record.items()))
    model.eval()
    return history
