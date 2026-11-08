# Multi-Tool4Compression

English | [한국어](README.ko.md)

A PyTorch toolkit of 48 model-compression methods (22 pruning, 22 quantisation, 4 pipelines), each applied from a YAML config by one script to a trained CIFAR-10 baseline. Every run reports the same cost numbers next to accuracy: parameters and non-zeros, storage bytes, MACs, PyTorch CPU latency and ONNX Runtime CPU latency in fp32 and int8. That way it is clear which compression actually makes a model faster and which only makes it smaller. The models come from [Multi-Tool4Vision](https://github.com/MZUI3/Multi-Tool4Vision), and `mtcompress/common/hwaccel.py` is the same file used in that repository and in [Multi-Tool4RL](https://github.com/MZUI3/Multi-Tool4RL).

## Results

The setup:

- **Data:** CIFAR-10 at 32×32.
- **Fine-tuning:** one RTX 3060.
- **Latency:** an AMD Ryzen 7 5800X at batch 1 with 4 threads, using PyTorch 2.11 and ONNX Runtime 1.31.
- **Seed:** one (0).
- **Budget:** most methods fine-tune for 5 epochs, and pruning at initialisation trains from scratch for 30. Absolute numbers are therefore below the papers'. What matters is how the methods compare under the same budget.

The full tables (108 runs) are in [results/mobilevit.md](results/mobilevit.md), [results/resnet18.md](results/resnet18.md) and [results/vit_tiny.md](results/vit_tiny.md). The raw summaries are in the matching `.json` files, and accuracy-vs-cost plots are in `results/*.png`.

| Baseline | Top-1 | Params | fp32 size | MACs | PyTorch CPU | ORT fp32 | ORT int8 |
|---|---|---|---|---|---|---|---|
| `resnet18` (30 epochs) | 93.89% | 11.17M | 42.7 MiB | 555.4M | 12.06 ms | 4.99 ms | 2.01 ms |
| `mobilevit` (60 epochs) | 84.24% | 0.44M | 1.69 MiB | 7.39M | 1.35 ms | 0.255 ms | 0.317 ms |
| `vit_tiny` (60 epochs) | 83.37% | 5.36M | 20.5 MiB | 365.1M | 9.11 ms | 5.13 ms | 5.55 ms |

**What actually gets faster: ResNet18**

| Method | Top-1 | Size | MACs | ORT fp32 | ORT int8 |
|---|---|---|---|---|---|
| baseline | 93.89 | 42.7 MiB | 555M | 4.99 ms | 2.01 ms |
| `gmp`: 80% unstructured | 93.72 | 9.9 MiB | 555M | 4.73 ms | 1.97 ms |
| `nm_sparsity`: 2:4 | 93.71 | 22.7 MiB | 555M | 4.98 ms | 2.01 ms |
| `ptq_mse`: int8 PTQ | 93.88 | 10.7 MiB | 555M | — | **2.00 ms** |
| `l1_filter_30`: 30% of channels | 93.38 | 21.0 MiB | 275M | 2.99 ms | 1.47 ms |
| `prune_quant`: 30% channels + int8 PTQ | 93.75 | 5.3 MiB | 275M | — | **1.49 ms** |
| `slimming`: 50% of channels | 92.53 | 10.7 MiB | 139M | 1.07 ms | **0.60 ms** |
| `deep_compression`: GMP 80% + 5-bit k-means + Huffman | 93.74 | **2.4 MiB** | 555M | 4.85 ms | 2.06 ms |

- Unstructured and 2:4 sparsity shrink storage but leave dense CPU latency where it was. The 2:4 GPU kernels (cuSPARSELt) are not in the Windows build of PyTorch, so the run records that it could not measure them.
- Removing channels and quantising to int8 stack. At 50% of channels plus int8, latency drops from 4.99 ms to 0.60 ms (8.3×) for 1.4 points of accuracy.
- The seven channel criteria land within 0.4 points of each other at 50% (92.18 to 92.53). At this budget, how much is pruned matters more than how channels are ranked.

**Small models behave differently: MobileViT (7.4M MACs)**

- int8 is *slower* than fp32 in ONNX Runtime (0.317 vs 0.255 ms). At this size, quantise/dequantise overhead outweighs the cheaper arithmetic. The same happens on `vit_tiny` (5.55 vs 5.13 ms), where attention matmuls and LayerNorm stay in float.
- `layer_drop` measured each block's removal on the calibration set and removed **both transformer layers** of the MobileViT block. The result is 83.71% (−0.53) at 3.06M MACs, and ORT latency drops from 0.255 to 0.090 ms. On CIFAR-10, this compact MobileViT gets little from its attention, which is something the per-block measurement reveals and channel pruning does not. Channel pruning to 3.38M MACs (`l1_filter_global`) reaches 79.32%.
- `head_prune` removes half the heads at 83.97%.

**Unstructured and lottery-ticket pruning (MobileViT)**

- At 50%, `magnitude` (83.96) and `obd` (83.96) tie, and `random` (78.88) is 5 points lower.
- At 80%, gradual magnitude pruning (`gmp`, 83.68) beats `movement` (64.21) by a wide margin. Movement pruning was designed for fine-tuning to a new task; on the task the network was already trained for, magnitude is the better signal, as the paper itself notes.
- `imp` reaches 86% sparsity at 82.69 with learning-rate rewinding and 82.99 with weight rewinding.
- Pruning at initialisation (80%, then 30 epochs from scratch): `snip` scores 79.81, `synflow` 79.77 and `grasp` 77.99, all above `random_init` (72.88). Applied to the trained model instead, SNIP gives 83.72 (`snip_trained`).

**Quantisation**

- int8 PTQ costs at most 0.17 points (ResNet18 93.85 to 93.91, MobileViT 84.07 to 84.28).
- `dfq` found 8 cross-layer equalisation pairs in ResNet18 and none in MobileViT. MobileViT's SiLU is not positively homogeneous, so equalisation would change the function there.
- 4-bit weights and 8-bit activations on ResNet18: `ptq_minmax_w4a8` scores 93.61, `adaround` 93.70 and `brecq` 93.79.
- 4-bit weights and 4-bit activations on ResNet18: `brecq_w4a4` scores 93.34 and `qdrop` 93.50.
- On MobileViT, W4A4 PTQ collapses: `ptq_mse_w4a4` scores 10.52, `brecq_w4a4` 37.45 and `qdrop` 26.18. QAT holds up (`lsq` W4A4 81.30, `lsq_plus` 82.08). For a 0.44M-parameter network with SiLU and attention, 4-bit activations need training.
- **Transformers:** with 4-bit attention probabilities and quantised LayerNorm inputs on `vit_tiny`, uniform quantisers score 78.13 (`ptq_minmax_vit_full`). FQ-ViT's power-of-two factor and log2 quantiser score **82.96** at the same places. `ptq4vit` scores 83.28 with 8-bit twin uniform quantisers.
- Mixed precision on MobileViT: `hawq` averages 3.0 bits per weight at 83.61%, with 183 KiB of storage (9.4×).
- The lowest bit widths on MobileViT: `twn` (ternary weights) scores 81.96 in 143 KiB, `xnor` 56.07, and `bnn` (binary weights and activations) 45.51.

**Tokens (ViT-tiny)**

`tome` with r = 8 merges tokens down to a third of the MACs (123.5M). Without training it scores 77.89, and after 3 epochs of fine-tuning 81.17. At batch 1 on the CPU, though, it runs slower than the baseline (11.5 ms vs 9.1 ms) because merging is small, irregular work. `evit` keeps 70% of tokens at three layers: 82.10 at 246.9M MACs. Fewer MACs did not translate into lower latency here, which is why every run times the model instead of reporting MACs alone.

## Measurement notes

These five problems came up while building the benchmark. Each one would have produced wrong numbers without an error message.

- **Fake-quant QDQ is not an int8 model.** Exporting the fake-quantised model (Q/DQ on layer inputs only) gave 1.34 ms for MobileViT against 0.26 ms in fp32, because ONNX Runtime fused none of the convolutions. Int8 latency now comes from ONNX Runtime's static QDQ quantisation of the same graph. With it, all 20 ResNet18 convolutions run as `QLinearConv`.
- **KL calibration and ReLU.** Half of a ReLU output is exactly zero. In the histogram of |x| that spike made every wider threshold look worse, so KL calibration always picked about 1/10 of the range: ResNet18 int8 fell to 32.6%. Zeros are now left out, and the result is 93.91%.
- **In-place activations and stored targets.** Reconstruction targets stored with `detach()` were overwritten by the `ReLU(inplace=True)` that follows each convolution, so AdaRound fitted pre-activation outputs to post-activation targets. It scored 87.8% on ResNet18, below plain rounding. Storing a `clone()` brings it to 93.70%.
- **`nn.MultiheadAttention`** exports through a fused kernel that has no ONNX symbol. Converting it to `SplitAttention` first gives the same outputs and exports.
- **The TorchScript ONNX exporter** corrupts the heap on ToMe's graph and kills the process. ONNX measurement now runs in a child process, so one crash costs one measurement, not the benchmark.

## Installation

```bash
pip install -r requirements.txt   # torch, torchvision, numpy, pyyaml, onnx, onnxruntime, pytest, matplotlib
```

CIFAR-10 is downloaded to `./data` on first use. The three baselines used in `results/` are tracked in `checkpoints/`.

## Quick start

```bash
git clone https://github.com/MZUI3/Multi-Tool4Compression.git && cd Multi-Tool4Compression
pip install -r requirements.txt
python compress.py --list                                   # every method with family and paper
python compress.py --method l1_filter --model resnet18      # configs/prune/l1_filter.yaml on the ResNet18 baseline
python compress.py --method ptq_mse --model resnet18        # int8 PTQ; reports ONNX Runtime int8 latency
python evaluate.py runs/l1_filter_resnet18_s0_*             # re-measure a saved run
```

## Usage

- `train.py` trains a baseline and writes `checkpoints/<model>_<dataset>.pth`. Any model name of Multi-Tool4Vision works (`resnet18`, `mobilenet_v2`, `mobilevit`, `vit_tiny`, ...). `tools/train_baselines.sh` holds the three runs behind `results/`.
- `compress.py` applies one method. `--method` names a method or a variant config (`configs/<kind>/<name>.yaml`); `--config` takes any YAML file; `--set` takes dotted `KEY=VALUE` pairs parsed as YAML (`--set hparams.sparsity=0.9 hparams.exclude=[]`, list items by index: `hparams.stages.0.hparams.ratio=0.5`). `--hparams` prints a method's hparams with their defaults. `--limit-batches N` caps every pass at N batches for debugging.
- `evaluate.py` reloads run directories and measures them again.
- `benchmark.py` runs a suite (`prune`, `quant`, `pipeline`, `all`) or a list of configs on one or more baselines and writes `results/<name>.json` and `.md`. `--resume` skips runs already in the JSON. `tools/run_benchmarks.sh` holds the runs behind `results/`.
- `plot.py` plots accuracy against storage and MACs from a results file.

A run directory `runs/<variant>_<model>_s<seed>_<time>/` contains `config.yaml` (resolved, every default filled in), `model.pt` (the compressed model as a pickled module, since pruning and quantisation change the architecture; load with `torch.load(path, weights_only=False)` from the repository root), `progress.jsonl` (one line per fine-tuning epoch) and `summary.json` (baseline and final accuracy and cost, method details such as per-layer sparsity, kept channels or chosen bit widths, and the hardware).

```python
from mtcompress.runner import load_run
model, cfg, summary = load_run("runs/l1_filter_resnet18_s0_20261106-101500")
print(summary["final"]["acc"], summary["final"]["macs"], summary["info"]["groups"][:3])
```

## What is measured

| Field | Meaning |
|---|---|
| `acc` | Top-1 on the 10,000 CIFAR-10 test images. |
| `params`, `params_nonzero`, `weight_sparsity` | Model parameters (quantiser step sizes and clipping levels are not counted) and the share of zero conv/linear weights. |
| `size_bytes`, `compression_ratio` | Storage of the state dict (the ratio is against the fp32 baseline): 32 bits per float, the quantiser bit width per quantised weight plus 32-bit scales, `min(dense, nnz·bits + 1-bit mask)` for sparse tensors, codebook plus Huffman codes for k-means sharing. Buffers that exist only for compression are not counted. |
| `macs`, `sparse_macs` | Multiply-accumulates per image at the shapes the model actually has (attention matmuls included); `sparse_macs` scales each layer by its weight density, which is what a sparse kernel could skip. |
| `torch_cpu_latency_ms` | Median PyTorch CPU latency, batch 1, 4 threads. Not reported for fake-quantised models, whose PyTorch timing would measure the emulation. |
| `onnx_fp32_latency_ms` | ONNX Runtime CPU latency of the float graph, batch 1, 4 threads. |
| `onnx_int8_latency_ms` | The same graph quantised with ONNX Runtime's static quantisation (QDQ, per-channel int8 weights, uint8 activations), so it runs on `QLinearConv` / `QGemm` / `QLinearAdd` kernels. For 8-bit uniform methods this is the deployment latency; the accuracy is that of the method's own fake-quantised model. Methods below 8 bits, or with log2 / twin / power-of-two-factor quantisers, have no ONNX Runtime CPU kernel and get no latency. |

The fake-quantised layout used for accuracy puts Q/DQ only on layer inputs. Exported as it is, ONNX Runtime fuses almost nothing: on MobileViT no convolution became `QLinearConv`, and on ResNet18 eight of twenty did. That is why int8 latency comes from the static-QDQ graph.

## Methods

Names are as used by `--method`; `python compress.py --list` adds the paper reference for each.

| Kind | Family | Methods |
|---|---|---|
| **Pruning (22)** | Unstructured | `magnitude`, `random`, `obd`, `gmp`, `movement` |
| | At initialisation | `snip`, `grasp`, `synflow`, `random_init` |
| | Lottery ticket | `imp` (rewinding: `weights`, `lr`, `none`) |
| | Semi-structured | `nm_sparsity` (2:4) |
| | Structured channel | `l1_filter`, `group_l2`, `fpgm`, `slimming`, `taylor`, `hrank`, `thinet` |
| | Transformer | `head_prune`, `layer_drop` |
| | Token reduction | `tome`, `evit` |
| **Quantisation (22)** | PTQ calibration | `ptq_minmax`, `ptq_percentile`, `ptq_mse`, `ptq_kl` |
| | PTQ data-free | `dfq` |
| | PTQ reconstruction | `adaround`, `brecq`, `qdrop` |
| | PTQ transformer | `fq_vit`, `ptq4vit` |
| | QAT | `qat_ste`, `lsq`, `lsq_plus`, `pact`, `dorefa`, `apot` |
| | Binary / ternary | `bnn`, `xnor`, `twn` |
| | Mixed precision | `mixed_sensitivity`, `hawq` |
| | Weight sharing | `kmeans_share` |
| **Pipelines (4)** | | `pipeline` (any stages), `deep_compression`, `prune_quant`, `prune_quant_kd` |

How the pieces fit:

- **Structured pruning really removes channels.** `mtcompress/prune/depgraph.py` records one forward pass with module hooks and a `TorchFunctionMode`, so models that unpack `x.shape` or reshape tensors (MobileViT) are traced with real shapes. Residual additions merge channel groups, concatenation lays them side by side, reshapes and permutes move the channel axis, and anything unknown pins its groups. All seven channel criteria share this graph; removing a channel slices the producer's filters, every consumer's input channels, the norm entries and parameters added element-wise (a ViT's position embedding). It handles ResNet, MobileNetV2, MobileViT, ExMobileViT and ViT without per-model code.
- **Attention is made explicit when needed.** `nn.MultiheadAttention` and the ViT `Attention` (which calls `scaled_dot_product_attention`) are replaced with `SplitAttention`, which computes the same function (max difference 4e-7) with the softmax, the heads and the projections exposed. Head pruning, token merging, softmax quantisers and ONNX export use it.
- **Masks are kept through fine-tuning.** Unstructured masks zero gradients after `backward` and weights after every optimiser step, so momentum and weight decay cannot revive pruned weights.
- **PTQ folds BatchNorm first,** because a deployed int8 model runs folded weights. `dfq` adds cross-layer equalisation (only through ReLU-like activations, which it checks) and empirical bias correction.
- **AdaRound, BRECQ and QDrop share one reconstruction engine.** Units are single layers (AdaRound) or called sub-modules with up to eight quantised layers (BRECQ, QDrop). Inputs come from the partly quantised model and targets from a float copy. At the end the learned rounding is written into the weights, so the result is a plain fake-quantised model again.
- **Every method can distil** from the uncompressed model during fine-tuning (`kd_alpha`, `kd_temperature`).

Where an implementation departs from its paper, the class docstring says so. These are the departures:

- `thinet` selects channels greedily without the least-squares rescaling.
- `obd` uses the empirical Fisher for the Hessian diagonal.
- `dfq` uses empirical rather than analytic bias correction.
- `ptq4vit` searches its scales by MSE instead of the Hessian-guided metric.
- `dorefa` multiplies weights by the layer's initial max|w|.
- `xnor` scales activations per sample instead of using the K matrix.
- `apot` builds its levels from two additive power-of-two terms with LSQ activations.
- `hawq` solves the bit allocation with an exact DP instead of an ILP.
- QAT activation quantisers are signed when the calibration data has negative values (the papers assume ReLU inputs).
- Reconstruction runs 1,000 iterations per unit rather than 20,000.

## Configuration

```yaml
method: l1_filter          # registry name
model: mobilevit           # baseline, checkpoints/<model>_<dataset>.pth
dataset: cifar10           # cifar10 | cifar100
applies_to: [mobilevit]    # optional: benchmark.py skips other models
hparams:                   # only what differs from the method's defaults
  ratio: 0.3
  kd_alpha: 0.7
```

- Every method shares the fine-tuning hparams `ft_epochs`, `ft_lr`, `ft_optimizer` (`sgd`, `adamw`, `adam`), `ft_weight_decay`, `ft_batch_size`, `ft_warmup_epochs`, `kd_alpha` and `kd_temperature`, plus `exclude` (layer-name prefixes, or `first` / `last` for the first conv and the classifier). Each class declares the rest, and they are merged along the class hierarchy. An unknown hparam stops the run with the list of valid keys.
- Top-level keys and their defaults: `seed: 0`, `device: auto`, `amp: true` (fine-tuning of float models only; fake quantisation runs in float32), `checkpoint: auto`, `save_dir: runs`, `measure: {latency: true, onnx: true, threads: 4, iters: 100, gpu_latency: false}`.
- Precedence: the defaults, then the YAML file, then the `--model`, `--seed` and other flags, then `--set`.
- `tools/make_configs.py` writes `configs/`: one config per method with its own hparams spelled out, plus the comparison variants (`magnitude_90`, `ptq_minmax_w4a8`, `ptq_minmax_vit_full`, `lsq_w2a2`, ...).

## Adding a method

Create a class in `mtcompress/prune/` or `mtcompress/quant/`, register it, and import its module in that package's `__init__.py`.

```python
from ..base import PruneMethod
from ..registry import register_method
from .masks import MaskSet


@register_method("my_prune", kind="prune", family="Unstructured", paper="Author et al., Year")
class MyPrune(PruneMethod):
    hparams = {"sparsity": 0.5}

    def apply(self, ctx):
        masks = MaskSet(ctx.model, self.hp["exclude"])
        masks.prune_by_scores(masks.scores(lambda name, w: w.detach().abs()), self.hp["sparsity"])
        self.finetune(ctx, hooks=masks)
        masks.finalize(ctx.model, ctx.info)
```

`ctx` holds the model, the data (`train`, `val`, `calib` iterables on the device), the frozen teacher and an `info` dict that goes to `summary.json`. Then run `python tools/make_configs.py` and `python tests/test_smoke.py my_prune`.

## Project structure

```
train.py, compress.py, evaluate.py, benchmark.py, plot.py   # entry points
configs/prune/, configs/quant/, configs/pipeline/           # one config per method + comparison variants
mtcompress/registry.py, base.py, config.py, runner.py       # method registry, Method/Context, YAML, run/measure/save
mtcompress/data.py, train.py                                # CIFAR on the device, fit/evaluate with KD and hooks
mtcompress/measure/profile.py                               # parameters, storage, MACs, PyTorch and ONNX Runtime latency
mtcompress/nn/attention.py                                  # SplitAttention and converters
mtcompress/prune/   masks, unstructured, at_init, lottery, nm, depgraph, structured, transformer (.py)
mtcompress/quant/   observers, core, fold, ptq, recon, vit, qat, lowbit, mixed, kmeans (.py)
mtcompress/pipeline.py                                      # stages of methods
mtcompress/common/hwaccel.py                                # shared with Multi-Tool4Vision / Multi-Tool4RL
models/                                                     # Multi-Tool4Vision model registry (136 models)
checkpoints/                                                # baselines used in results/
results/                                                    # benchmark tables (.md), raw summaries (.json), plots
tools/   train_baselines.sh, run_benchmarks.sh, make_configs.py
tests/test_smoke.py
```

## Tests

```bash
python tests/test_smoke.py              # every method, randomly initialised model, two batches per pass
python tests/test_smoke.py lsq brecq    # a subset; name:model picks the model, e.g. thinet:resnet18
pytest tests/
```

The smoke test checks that a method runs, saves and reloads, not that it compresses well; `results/` is where that shows.
