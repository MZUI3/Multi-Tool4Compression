"""Channel dependency groups and physical channel removal (in the spirit of DepGraph, Fang et al., 2023).

One forward pass is recorded with module hooks and a ``TorchFunctionMode``, so models that unpack
``x.shape`` or reshape tensors (MobileViT) are traced as they run, with real shapes. Every tensor gets a
channel annotation: the axis that carries channels and the channel groups along it. Producers (conv,
linear and attention output projections) open a new group. Element-wise ops such as residual additions
merge the groups of their inputs, concatenation along the channel axis lays groups side by side, and
reshapes, permutes and pooling move the axis. Anything the tracer does not understand pins the groups it
touches, so they are never pruned. The model output and the input image are pinned as well.

Removing channel ``j`` of a group removes, in every member, the matching slice: filters of the
producer, input channels of every consumer, BatchNorm/LayerNorm entries, and parameters added
element-wise (for example a position embedding). The tensors really shrink, so MACs and latency drop.
"""
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.overrides import TorchFunctionMode

from ..nn.attention import SplitAttention

POOL_TYPES = (nn.AdaptiveAvgPool2d, nn.AdaptiveMaxPool2d, nn.AvgPool2d, nn.MaxPool2d, nn.AdaptiveAvgPool1d)
PASS_TYPES = (nn.ReLU, nn.ReLU6, nn.SiLU, nn.GELU, nn.Hardswish, nn.Hardsigmoid, nn.Sigmoid, nn.Tanh,
              nn.LeakyReLU, nn.ELU, nn.Mish, nn.Dropout, nn.Dropout2d, nn.Identity, nn.Softmax)
ELEMENTWISE = {"add", "__add__", "__radd__", "add_", "__iadd__", "sub", "__sub__", "__rsub__", "mul", "__mul__",
               "__rmul__", "mul_", "__imul__", "div", "__truediv__", "__rtruediv__", "where", "maximum", "minimum"}
UNARY = {"relu", "relu_", "gelu", "silu", "sigmoid", "tanh", "hardswish", "hardsigmoid", "dropout", "contiguous",
         "clone", "float", "half", "to", "neg", "__neg__", "abs", "exp", "sqrt", "pow", "__pow__", "softmax",
         "log_softmax", "detach", "type_as", "leaky_relu", "relu6", "hardtanh", "elu", "mish", "layer_norm_noop"}
RESHAPE = {"view", "reshape", "flatten", "unflatten", "squeeze", "unsqueeze", "expand", "expand_as"}


@dataclass
class Member:
    module: nn.Module
    role: str  # conv_out conv_in dwconv linear_out linear_in norm attn_in attn_out param
    offset: int = 0
    param_name: str = ""
    axis: int = 0


@dataclass
class Group:
    size: int
    members: List[Member] = field(default_factory=list)
    pinned: bool = False
    names: List[str] = field(default_factory=list)


class _UF:
    def __init__(self):
        self.parent: Dict[int, int] = {}

    def find(self, a):
        self.parent.setdefault(a, a)
        while self.parent[a] != a:
            self.parent[a] = self.parent[self.parent[a]]
            a = self.parent[a]
        return a

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra
        return ra


Ann = Tuple[int, Tuple[Tuple[int, int], ...]]  # (axis, ((gid, length), ...))


def _flat_tensors(obj) -> List[torch.Tensor]:
    if isinstance(obj, torch.Tensor):
        return [obj]
    if isinstance(obj, (list, tuple)):
        return [t for o in obj for t in _flat_tensors(o)]
    if isinstance(obj, dict):
        return [t for o in obj.values() for t in _flat_tensors(o)]
    return []


def _prod(xs) -> int:
    out = 1
    for x in xs:
        out *= int(x)
    return out


class ChannelTracer:
    def __init__(self, model: nn.Module):
        self.model = model
        self.uf = _UF()
        self.groups: Dict[int, Group] = {}
        self.ann: Dict[int, Ann] = {}
        self.keep: List[torch.Tensor] = []  # keeps traced tensors alive so ids are unique
        self.param_names = {id(p): n for n, p in model.named_parameters()}
        self.derived: Dict[int, Tuple[str, int]] = {}  # tensor id -> (param name, ndim of param) for expand()
        self.depth = 0
        self.module_names = {m: n for n, m in model.named_modules()}

    # ------------------------------------------------------------------ groups
    def new_group(self, size: int, name: str) -> int:
        gid = len(self.groups)
        self.groups[gid] = Group(size, names=[name])
        self.uf.find(gid)
        return gid

    def group(self, gid: int) -> Group:
        return self.groups[self.uf.find(gid)]

    def add_member(self, gid: int, member: Member):
        self.group(gid).members.append(member)

    def pin(self, ann: Optional[Ann]):
        if ann is None:
            return
        for gid, _ in ann[1]:
            self.group(gid).pinned = True

    def merge(self, a: Ann, b: Ann) -> Optional[Ann]:
        if a[0] != b[0] or [l for _, l in a[1]] != [l for _, l in b[1]]:
            self.pin(a)
            self.pin(b)
            return None
        for (ga, _), (gb, _) in zip(a[1], b[1]):
            ra, rb = self.uf.find(ga), self.uf.find(gb)
            if ra != rb:
                root = self.uf.union(ra, rb)
                other = rb if root == ra else ra
                keep, gone = self.groups[root], self.groups[other]
                keep.members += gone.members
                keep.names += gone.names
                keep.pinned = keep.pinned or gone.pinned
        return a

    def resolve_groups(self) -> List[Group]:
        roots = {}
        for gid in self.groups:
            r = self.uf.find(gid)
            roots[r] = self.groups[r]
        return list(roots.values())

    # ------------------------------------------------------------------ annotations
    def get(self, t) -> Optional[Ann]:
        return self.ann.get(id(t)) if isinstance(t, torch.Tensor) else None

    def set(self, t, ann: Optional[Ann]):
        if isinstance(t, torch.Tensor):
            self.keep.append(t)
            if ann is not None:
                self.ann[id(t)] = ann

    @staticmethod
    def segs_with_offsets(ann: Ann):
        off = 0
        for gid, length in ann[1]:
            yield gid, off, length
            off += length

    def _param_of(self, t) -> Optional[Tuple[str, int]]:
        if id(t) in self.param_names:
            return self.param_names[id(t)], t.dim()
        return self.derived.get(id(t))

    def _move_axis(self, ann: Ann, src: torch.Tensor, out: torch.Tensor) -> Optional[Ann]:
        """Locate the channel axis of ``out`` after a reshape-like op, by size and prefix product."""
        axis = ann[0]
        c = src.shape[axis]
        prefix = _prod(src.shape[:axis])
        for j in range(out.dim()):
            if out.shape[j] == c and _prod(out.shape[:j]) == prefix:
                return (j, ann[1])
        return None

    # ------------------------------------------------------------------ modules
    def on_module(self, m: nn.Module, inputs, output):
        name = self.module_names.get(m, "?")
        x = inputs[0] if inputs else None
        a = self.get(x)
        outs = _flat_tensors(output)
        y = outs[0] if outs else None
        if isinstance(m, nn.Conv2d):
            if m.groups == 1:
                if a is not None and a[0] == 1:
                    for gid, off, _ in self.segs_with_offsets(a):
                        self.add_member(gid, Member(m, "conv_in", off))
                else:
                    self.pin(a)
                g = self.new_group(m.out_channels, name)
                self.add_member(g, Member(m, "conv_out"))
                self.set(y, (1, ((g, m.out_channels),)))
            elif m.groups == m.in_channels == m.out_channels and a is not None and a[0] == 1:
                for gid, off, _ in self.segs_with_offsets(a):
                    self.add_member(gid, Member(m, "dwconv", off))
                self.set(y, a)
            else:
                self.pin(a)
                g = self.new_group(m.out_channels, name)
                self.groups[g].pinned = True
                self.set(y, (1, ((g, m.out_channels),)))
        elif isinstance(m, nn.Linear):
            if a is not None and a[0] == x.dim() - 1:
                for gid, off, _ in self.segs_with_offsets(a):
                    self.add_member(gid, Member(m, "linear_in", off))
            else:
                self.pin(a)
            g = self.new_group(m.out_features, name)
            self.add_member(g, Member(m, "linear_out"))
            self.set(y, (y.dim() - 1, ((g, m.out_features),)))
        elif isinstance(m, SplitAttention):
            if a is not None and a[0] == x.dim() - 1:
                for gid, off, _ in self.segs_with_offsets(a):
                    self.add_member(gid, Member(m, "attn_in", off))
            else:
                self.pin(a)
            g = self.new_group(m.dim, name)
            self.add_member(g, Member(m, "attn_out"))
            self.set(y, (y.dim() - 1, ((g, m.dim),)))
            for extra in outs[1:]:
                self.set(extra, None)
        elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
            if a is not None and a[0] == 1:
                for gid, off, _ in self.segs_with_offsets(a):
                    self.add_member(gid, Member(m, "norm", off))
                self.set(y, a)
            else:
                self.pin(a)
        elif isinstance(m, nn.LayerNorm):
            if a is not None and a[0] == x.dim() - 1 and len(m.normalized_shape) == 1:
                for gid, off, _ in self.segs_with_offsets(a):
                    self.add_member(gid, Member(m, "norm", off))
                self.set(y, a)
            else:
                self.pin(a)
        elif isinstance(m, POOL_TYPES) or isinstance(m, PASS_TYPES):
            if a is not None and y is not None and y.dim() == x.dim() and y.shape[a[0]] == x.shape[a[0]]:
                self.set(y, a)
            else:
                self.pin(a)
        elif isinstance(m, nn.Flatten):
            if a is not None:
                moved = self._move_axis(a, x, y)
                if moved is None:
                    self.pin(a)
                self.set(y, moved)
        elif (not any(True for _ in m.parameters(recurse=False)) and a is not None and y is not None
              and len(outs) == 1 and y.shape == x.shape):
            self.set(y, a)  # parameter-free, shape-preserving leaf (DropPath and the like)
        else:
            for t in _flat_tensors(inputs):
                self.pin(self.get(t))
            for t in outs:
                g = self.new_group(t.shape[1] if t.dim() > 1 else 1, name)
                self.groups[g].pinned = True
                self.set(t, None)
        for t in outs:
            self.keep.append(t)

    # ------------------------------------------------------------------ functions
    def on_function(self, func, args, kwargs, out):
        name = getattr(func, "__name__", str(func))
        ins = _flat_tensors(args) + _flat_tensors(kwargs)
        outs = _flat_tensors(out)
        if not outs:
            return
        anns = [self.get(t) for t in ins]
        annotated = [(t, a) for t, a in zip(ins, anns) if a is not None]

        # expand()/repeat of a parameter keeps a link to it (ViT class token)
        if name in ("expand", "expand_as", "repeat") and ins and self._param_of(ins[0]) is not None and not annotated:
            self.derived[id(outs[0])] = self._param_of(ins[0])
            self.keep.append(outs[0])
            return
        if not annotated:
            for t in outs:
                self.keep.append(t)
            return
        y = outs[0]

        if name in ELEMENTWISE:
            ann = None
            for t, a in annotated:
                ann = a if ann is None else self.merge(ann, a)
                if ann is None:
                    break
            if ann is not None:
                axis = ann[0]
                for t in ins:
                    if self.get(t) is not None:
                        continue
                    shift = y.dim() - t.dim()
                    p_axis = axis - shift
                    if p_axis < 0 or t.shape[p_axis] == 1:
                        continue  # broadcast along the channel axis
                    p = self._param_of(t)
                    if p is not None and t.dim() == p[1]:
                        for gid, off, _ in self.segs_with_offsets(ann):
                            self.add_member(gid, Member(None, "param", off, p[0], p_axis))
                    else:
                        self.pin(ann)
                        ann = None
                        break
            self.set(y, ann)
        elif name == "cat":
            tensors = args[0] if args else kwargs.get("tensors")
            dim = kwargs.get("dim", args[1] if len(args) > 1 else 0)
            dim = dim % y.dim()
            tanns = [self.get(t) for t in tensors]
            if all(a is not None for a in tanns) and all(a[0] == dim for a in tanns):
                segs = tuple(s for a in tanns for s in a[1])
                self.set(y, (dim, segs))
            elif all(a is not None for a in tanns) and all(a[0] != dim for a in tanns):
                ann = tanns[0]
                for a in tanns[1:]:
                    ann = self.merge(ann, a) if ann is not None else None
                self.set(y, ann)
            else:
                ann = next(a for a in tanns if a is not None)
                ok = True
                for t, a in zip(tensors, tanns):
                    if a is None:
                        p = self._param_of(t)
                        if p is not None and ann[0] != dim:
                            for gid, off, _ in self.segs_with_offsets(ann):
                                self.add_member(gid, Member(None, "param", off, p[0], ann[0]))
                        else:
                            ok = False
                for a in tanns:
                    if a is not None and a != ann:
                        ok = ok and self.merge(ann, a) is not None
                if not ok:
                    for a in tanns:
                        self.pin(a)
                self.set(y, ann if ok else None)
        elif name in UNARY and len(annotated) == 1:
            t, a = annotated[0]
            if y.shape == t.shape:
                self.set(y, a)
            else:
                self.pin(a)
        elif name in ("permute",):
            t, a = annotated[0]
            dims = args[1:] if len(args) > 1 and isinstance(args[1], int) else (args[1] if len(args) > 1 else kwargs["dims"])
            dims = [d % t.dim() for d in dims]
            self.set(y, (dims.index(a[0]), a[1]))
        elif name in ("transpose", "swapaxes"):
            t, a = annotated[0]
            d0, d1 = (args[1] % t.dim(), args[2] % t.dim())
            axis = d1 if a[0] == d0 else d0 if a[0] == d1 else a[0]
            self.set(y, (axis, a[1]))
        elif name in RESHAPE:
            t, a = annotated[0]
            moved = self._move_axis(a, t, y)
            if moved is None:
                self.pin(a)
            self.set(y, moved)
        elif name in ("mean", "sum", "amax", "amin", "adaptive_avg_pool2d", "avg_pool2d", "max_pool2d"):
            t, a = annotated[0]
            if name.endswith("pool2d"):
                self.set(y, a if y.dim() == t.dim() else None)
                return
            dim = kwargs.get("dim", args[1] if len(args) > 1 else None)
            keepdim = kwargs.get("keepdim", args[2] if len(args) > 2 else False)
            if dim is None:
                self.pin(a)
                return
            dims = [d % t.dim() for d in (dim if isinstance(dim, (list, tuple)) else [dim])]
            if a[0] in dims:
                self.pin(a)
                return
            axis = a[0] if keepdim else a[0] - sum(1 for d in dims if d < a[0])
            self.set(y, (axis, a[1]))
        elif name == "__getitem__":
            t, a = annotated[0]
            index = args[1]
            index = index if isinstance(index, tuple) else (index,)
            axis_in, axis_out, ok = 0, 0, True
            new_axis = None
            for item in index:
                if item is None:
                    axis_out += 1
                    continue
                if item is Ellipsis:
                    ok = False
                    break
                if axis_in == a[0]:
                    if isinstance(item, slice) and item == slice(None):
                        new_axis = axis_out
                    else:
                        ok = False
                        break
                if isinstance(item, int):
                    axis_in += 1
                elif isinstance(item, slice):
                    axis_in += 1
                    axis_out += 1
                else:
                    ok = False
                    break
            if ok and new_axis is None and a[0] >= axis_in:
                new_axis = axis_out + (a[0] - axis_in)
            if not ok or new_axis is None:
                self.pin(a)
                self.set(y, None)
            else:
                self.set(y, (new_axis, a[1]))
        else:
            for _, a in annotated:
                self.pin(a)
        for t in outs:
            self.keep.append(t)


class _Mode(TorchFunctionMode):
    def __init__(self, tracer: ChannelTracer):
        super().__init__()
        self.tracer = tracer

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        out = func(*args, **kwargs)
        if self.tracer.depth == 0:
            self.tracer.on_function(func, args, kwargs, out)
        return out


def _is_leaf(m: nn.Module) -> bool:
    return isinstance(m, SplitAttention) or len(list(m.children())) == 0


@torch.no_grad()
def build_groups(model: nn.Module, example: torch.Tensor, exclude=()) -> List[Group]:
    tracer = ChannelTracer(model)
    hooks = []

    def pre(m, inputs):
        tracer.depth += 1

    def post(m, inputs, output):
        tracer.depth -= 1
        if tracer.depth == 0:
            tracer.on_module(m, inputs, output)

    for m in model.modules():
        if _is_leaf(m) and m is not model:
            hooks.append(m.register_forward_pre_hook(pre))
            hooks.append(m.register_forward_hook(post))
    was_training = model.training
    model.eval()
    g_in = tracer.new_group(example.shape[1], "<input>")
    tracer.groups[g_in].pinned = True
    tracer.set(example, (1, ((g_in, example.shape[1]),)))
    try:
        with _Mode(tracer):
            out = model(example)
    finally:
        for h in hooks:
            h.remove()
        model.train(was_training)
    for t in _flat_tensors(out):
        tracer.pin(tracer.get(t))
    groups = tracer.resolve_groups()
    names = {m: n for n, m in model.named_modules()}
    for g in groups:
        for mem in g.members:
            if mem.module is not None and names.get(mem.module, "") in exclude:
                g.pinned = True
    return groups


def prunable_groups(groups: List[Group], min_size: int = 2) -> List[Group]:
    return [g for g in groups if not g.pinned and g.size >= min_size and g.members]


# ---------------------------------------------------------------------- slices of a group's channels


def _param_slices(member: Member, model: nn.Module):
    """(tensor, dim, offset) for every tensor touched by this member."""
    m, off = member.module, member.offset
    r = member.role
    if r == "conv_out":
        yield m.weight, 0, off
        if m.bias is not None:
            yield m.bias, 0, off
    elif r == "conv_in":
        yield m.weight, 1, off
    elif r == "dwconv":
        yield m.weight, 0, off
        if m.bias is not None:
            yield m.bias, 0, off
    elif r == "linear_out":
        yield m.weight, 0, off
        if m.bias is not None:
            yield m.bias, 0, off
    elif r == "linear_in":
        yield m.weight, 1, off
    elif r == "norm":
        for t in (m.weight, m.bias, getattr(m, "running_mean", None), getattr(m, "running_var", None)):
            if t is not None:
                yield t, 0, off
    elif r == "attn_in":
        yield m.qkv.weight, 1, off
    elif r == "attn_out":
        yield m.proj.weight, 0, off
        yield m.proj.bias, 0, off
    elif r == "param":
        yield model.get_parameter(member.param_name), member.axis, off


def channel_tensors(group: Group, model: nn.Module, roles=None):
    """Yield (tensor, dim, offset, role) for the group's members (optionally filtered by role)."""
    for mem in group.members:
        if roles is not None and mem.role not in roles:
            continue
        for t, dim, off in _param_slices(mem, model):
            yield t, dim, off, mem.role


def per_channel(t: torch.Tensor, dim: int, off: int, size: int) -> torch.Tensor:
    """The ``size`` channels of ``t`` starting at ``off`` along ``dim``, as a (size, -1) matrix."""
    return t.detach().narrow(dim, off, size).transpose(0, dim).reshape(size, -1)


# ---------------------------------------------------------------------- surgery


@torch.no_grad()
def remove_channels(model: nn.Module, plan: List[Tuple[Group, torch.Tensor]]) -> None:
    """Remove the channels marked ``False`` in each group's keep mask from every member, then fix the
    module attributes. Offsets refer to the original layout, so all groups are applied together."""
    masks: Dict[Tuple[int, int], torch.Tensor] = {}
    tensors: Dict[int, torch.Tensor] = {}
    owners = {}
    for group, keep in plan:
        for mem in group.members:
            for t, dim, off in _param_slices(mem, model):
                key = (id(t), dim)
                if key not in masks:
                    masks[key] = torch.ones(t.shape[dim], dtype=torch.bool, device=t.device)
                    tensors[id(t)] = t
                masks[key][off:off + group.size] &= keep.to(t.device)
            owners.setdefault(id(mem.module), mem)
    # Map every tensor back to (owner, attribute name)
    locations = {}
    for mod_name, mod in model.named_modules():
        for attr, p in list(mod._parameters.items()) + list(mod._buffers.items()):
            if p is not None:
                locations[id(p)] = (mod, attr)
    by_tensor = defaultdict(list)
    for (tid, dim), mask in masks.items():
        by_tensor[tid].append((dim, mask))
    for tid, items in by_tensor.items():
        t = tensors[tid]
        new = t.detach()
        for dim, mask in items:
            new = new.index_select(dim, mask.nonzero().flatten())
        mod, attr = locations[tid]
        if attr in mod._parameters:
            mod._parameters[attr] = nn.Parameter(new.clone(), requires_grad=t.requires_grad)
        else:
            mod._buffers[attr] = new.clone()
    for mod in model.modules():
        _fix_attributes(mod)


def _fix_attributes(m: nn.Module) -> None:
    if isinstance(m, nn.Conv2d):
        if m.groups > 1 and m.groups == m.in_channels:
            m.out_channels = m.weight.shape[0]
            m.in_channels = m.groups = m.out_channels
        else:
            m.out_channels = m.weight.shape[0]
            m.in_channels = m.weight.shape[1] * m.groups
    elif isinstance(m, nn.Linear):
        m.out_features, m.in_features = m.weight.shape
    elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
        m.num_features = m.weight.shape[0] if m.weight is not None else m.running_mean.shape[0]
    elif isinstance(m, nn.LayerNorm):
        m.normalized_shape = (m.weight.shape[0],)
    elif isinstance(m, SplitAttention):
        m.dim = m.qkv.in_features
