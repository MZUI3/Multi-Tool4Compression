# Multi-Tool4Compression

English | [한국어](README.ko.md)

A PyTorch toolkit of 48 model-compression methods (22 pruning, 22 quantisation, 4 pipelines), each applied from a YAML config by one script to a trained CIFAR-10 baseline. Every run reports the same cost numbers next to accuracy: parameters and non-zeros, storage bytes, MACs, PyTorch CPU latency and ONNX Runtime CPU latency in fp32 and int8. That way it is clear which compression actually makes a model faster and which only makes it smaller. The models come from [Multi-Tool4Vision](https://github.com/MZUI3/Multi-Tool4Vision), and `mtcompress/common/hwaccel.py` is the same file used in that repository and in [Multi-Tool4RL](https://github.com/MZUI3/Multi-Tool4RL).

<!-- RESULTS -->

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
