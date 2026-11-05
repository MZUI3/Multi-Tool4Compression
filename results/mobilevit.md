## mobilevit

Baseline: 84.24% top-1, 440,602 parameters, 1727.0 KiB fp32, 7.39M MACs, PyTorch CPU 1.347 ms, ONNX Runtime CPU 0.255 ms fp32 / 0.317 ms int8 (batch 1, 4 threads).

| Variant | Method | Acc (Δ) | Weight sparsity | Size KiB (×) | MACs M | Sparse MACs M | PyTorch CPU ms | ORT fp32 ms | ORT int8 ms | Time s |
|---|---|---|---|---|---|---|---|---|---|---|
| fpgm | fpgm | 72.32 (-11.92) | 0.0% | 570.5 (×1.00) | 2.50 | 2.50 | 1.318 | 0.209 | 0.282 | 57 |
