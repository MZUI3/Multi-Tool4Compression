"""What a compressed model costs: parameters, non-zeros, storage bytes, MACs and latency.

Storage is computed per tensor of the state dict, so it is the number a deployment would ship:

- ``bits``: 32 for float tensors, the quantiser bit width for quantised weights (recorded in
  ``model.mtc_meta["weight_bits"]``), or an explicit byte count from ``model.mtc_meta["storage_bytes"]``
  (k-means codebooks with Huffman codes).
- Sparse tensors: ``min(dense, nnz * bits + numel)``: the non-zero values plus a 1-bit mask.
- Per-channel scales and zero points of quantised weights are counted as 32-bit floats.
- Buffers that exist only for compression (masks, observers, quantiser state) are not counted.

MACs are counted with forward hooks on Conv2d, Linear and the attention modules, at the shapes the
model actually has. ``sparse_macs`` scales each layer by the density of its weight, which is the
work a sparse kernel could skip, not what dense hardware does.
"""
import copy
import os
import statistics
import tempfile
import time
from typing import Dict

import torch
import torch.nn as nn

SKIP_BUFFER_SUFFIXES = ("_mask", "num_batches_tracked")


def _meta(model: nn.Module) -> dict:
    return getattr(model, "mtc_meta", {}) or {}


def is_compression_state(name: str) -> bool:
    parts = name.split(".")
    return (name.endswith(SKIP_BUFFER_SUFFIXES) or any(p in QUANT_PARTS for p in parts)
            or "observer" in name or name.endswith("head_gate"))


QUANT_PARTS = ("wq", "aq", "lnq", "softmax_q", "post_q")


def weight_tensors(model: nn.Module):
    """(name, module, tensor) of every prunable/quantisable weight: Conv2d, Linear, MHA in_proj."""
    for mod_name, m in model.named_modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            yield f"{mod_name}.weight" if mod_name else "weight", m, m.weight
        elif isinstance(m, nn.MultiheadAttention) and m.in_proj_weight is not None:
            yield f"{mod_name}.in_proj_weight", m, m.in_proj_weight


def param_stats(model: nn.Module) -> Dict[str, float]:
    """Model parameters, not counting the learnable state of quantisers (step sizes, clipping levels)."""
    params = [p for n, p in model.named_parameters() if not is_compression_state(n)]
    total = sum(p.numel() for p in params)
    nnz_total = sum(int((p != 0).sum()) for p in params)
    w_total = sum(w.numel() for _, _, w in weight_tensors(model))
    w_nnz = sum(int((w != 0).sum()) for _, _, w in weight_tensors(model))
    return {
        "params": total,
        "params_nonzero": nnz_total,
        "weights": w_total,
        "weights_nonzero": w_nnz,
        "weight_sparsity": 100.0 * (1 - w_nnz / max(w_total, 1)),
    }


def storage_bytes(model: nn.Module) -> Dict[str, float]:
    meta = _meta(model)
    wbits = meta.get("weight_bits", {})
    explicit = meta.get("storage_bytes", {})
    scales = meta.get("scale_count", {})
    dense_fp32 = 0.0
    compressed = 0.0
    for name, t in model.state_dict().items():
        if is_compression_state(name) or not torch.is_tensor(t):
            continue
        n = t.numel()
        dense_fp32 += n * 4
        if name in explicit:
            compressed += explicit[name]
            continue
        bits = wbits.get(name, 32 if t.is_floating_point() else t.element_size() * 8)
        nnz = int((t != 0).sum()) if t.is_floating_point() else n
        size_bits = n * bits
        if nnz < n:
            size_bits = min(size_bits, nnz * bits + n)
        compressed += size_bits / 8 + scales.get(name, 0) * 4
    return {"size_fp32_bytes": dense_fp32, "size_bytes": compressed,
            "compression_ratio": dense_fp32 / max(compressed, 1.0)}


def _attention_macs(m: nn.Module, x: torch.Tensor) -> int:
    if x.dim() != 3:
        return 0
    b, n, c = x.shape
    return 2 * b * n * n * c  # q @ k^T and attn @ v; heads * head_dim = c


@torch.no_grad()
def count_macs(model: nn.Module, example: torch.Tensor) -> Dict[str, float]:
    dense = [0]
    sparse = [0.0]
    hooks = []

    def density(w):
        return float((w != 0).float().mean()) if w is not None else 1.0

    def conv_hook(m, inp, out):
        k = m.weight[0].numel()  # in/groups * kh * kw
        macs = out.numel() * k
        dense[0] += macs
        sparse[0] += macs * density(m.weight)

    def linear_hook(m, inp, out):
        macs = out.numel() * m.in_features
        dense[0] += macs
        sparse[0] += macs * density(m.weight)

    def mha_hook(m, inp, out):
        x = inp[0]
        b, n, e = x.shape
        proj = b * n * e * 3 * e + b * n * e * e
        att = _attention_macs(m, x)
        dense[0] += proj + att
        sparse[0] += proj * density(m.in_proj_weight) + att

    def attn_hook(m, inp, out):
        x = inp[0]
        c = getattr(m, "inner_dim", None)
        if c is None:
            c = x.shape[-1]
        b, n = x.shape[0], x.shape[1]
        macs = 2 * b * n * n * c
        dense[0] += macs
        sparse[0] += macs

    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            hooks.append(m.register_forward_hook(conv_hook))
        elif isinstance(m, nn.Linear) and not type(m).__name__ == "NonDynamicallyQuantizableLinear":
            hooks.append(m.register_forward_hook(linear_hook))
        elif isinstance(m, nn.MultiheadAttention):
            hooks.append(m.register_forward_hook(mha_hook))
        elif getattr(m, "counts_attention_macs", False) or (
                type(m).__name__ == "Attention" and hasattr(m, "qkv") and hasattr(m, "num_heads")):
            hooks.append(m.register_forward_hook(attn_hook))
    was_training = model.training
    model.eval()
    model(example)
    model.train(was_training)
    for h in hooks:
        h.remove()
    return {"macs": dense[0] / example.shape[0], "sparse_macs": sparse[0] / example.shape[0]}


@torch.no_grad()
def torch_latency_ms(model: nn.Module, example: torch.Tensor, device: str = "cpu", threads: int = 4,
                     warmup: int = 20, iters: int = 100) -> float:
    """Median latency of one forward pass at the example's batch size."""
    m = copy.deepcopy(model).to(device).eval()
    x = example.to(device)
    old_threads = torch.get_num_threads()
    if device == "cpu":
        torch.set_num_threads(threads)
    try:
        for _ in range(warmup):
            m(x)
        times = []
        for _ in range(iters):
            if device != "cpu":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            m(x)
            if device != "cpu":
                torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000)
    finally:
        torch.set_num_threads(old_threads)
    return statistics.median(times)


def export_onnx(model: nn.Module, example: torch.Tensor, path: str) -> str:
    from ..nn.attention import convert_attention
    from ..quant.core import export_mode

    m = copy.deepcopy(model).cpu().eval()
    # nn.MultiheadAttention's fused kernel has no ONNX export; the explicit module computes the same.
    convert_attention(m)
    with export_mode(m), torch.no_grad():
        torch.onnx.export(m, (example.cpu(),), path, input_names=["input"], output_names=["logits"],
                          opset_version=17, dynamo=False)
    return path


def float_graph(model: nn.Module) -> nn.Module:
    """A copy with every quantiser switched off: the same architecture in float."""
    m = copy.deepcopy(model).cpu().eval()
    for mod in m.modules():
        if hasattr(mod, "mode") and hasattr(mod, "observe"):
            mod.mode = "off"
    return m


def ort_int8(model: nn.Module, example: torch.Tensor, calib, path: str) -> str:
    """Quantise the float graph with ONNX Runtime's static quantisation (QDQ format, per-channel int8
    weights, uint8 activations, Q/DQ around every quantisable op). This is the graph ONNX Runtime fuses
    into QLinearConv / QGemm / QLinearAdd kernels; Q/DQ only on layer inputs (the fake-quant layout)
    leaves most convolutions in float."""
    from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_static
    from onnxruntime.quantization.shape_inference import quant_pre_process

    fp = path.replace(".onnx", "_fp32.onnx")
    pre = path.replace(".onnx", "_pre.onnx")
    export_onnx(float_graph(model), example, fp)
    quant_pre_process(fp, pre)

    class Reader(CalibrationDataReader):
        def __init__(self):
            self.items = iter([{"input": x[i:i + 1].cpu().numpy()} for x in calib for i in range(x.shape[0])])

        def get_next(self):
            return next(self.items, None)

    quantize_static(pre, path, Reader(), quant_format=QuantFormat.QDQ, per_channel=True,
                    activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8)
    return path


def _time_ort(path: str, example: torch.Tensor, threads: int, warmup: int, iters: int) -> float:
    import numpy as np
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = threads
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(path, opts, providers=["CPUExecutionProvider"])
    x = example.cpu().numpy().astype(np.float32)
    for _ in range(warmup):
        sess.run(None, {"input": x})
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        sess.run(None, {"input": x})
        times.append((time.perf_counter() - t0) * 1000)
    return statistics.median(times)


def onnx_latencies(model: nn.Module, example: torch.Tensor, calib, threads: int = 4, fp32: bool = True,
                   int8: bool = True, warmup: int = 30, iters: int = 200) -> Dict[str, float]:
    """ONNX Runtime CPU latency at batch 1 of the float graph and of its int8 static-QDQ version."""
    tmp = tempfile.mkdtemp()
    out = {}
    if fp32:
        p = os.path.join(tmp, "fp32.onnx")
        export_onnx(float_graph(model), example, p)
        out["onnx_fp32_latency_ms"] = _time_ort(p, example, threads, warmup, iters)
    if int8:
        p = ort_int8(model, example, calib, os.path.join(tmp, "int8.onnx"))
        out["onnx_int8_latency_ms"] = _time_ort(p, example, threads, warmup, iters)
        out["onnx_int8_file_bytes"] = os.path.getsize(p)
    return out


def onnx_latency_ms(model: nn.Module, example: torch.Tensor, threads: int = 4, warmup: int = 20,
                    iters: int = 100, keep_path: str = None) -> Dict[str, float]:
    """Export the model as it is (fake quantisers become Q/DQ on layer inputs) and time it."""
    tmp = tempfile.mkdtemp()
    path = keep_path or os.path.join(tmp, "model.onnx")
    export_onnx(model, example, path)
    return {"onnx_latency_ms": _time_ort(path, example, threads, warmup, iters), "onnx_file_bytes": os.path.getsize(path)}

