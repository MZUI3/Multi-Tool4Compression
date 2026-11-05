#!/usr/bin/env bash
# The runs behind results/. Baselines first: bash tools/train_baselines.sh
set -e
PY=${PY:-python}

# MobileViT: every config that applies to it.
$PY benchmark.py --suite all --models mobilevit --name mobilevit --resume

# ResNet18: structured pruning, where channel removal pays off, and int8 PTQ, where ONNX Runtime has kernels.
$PY benchmark.py --name resnet18 --models resnet18 --resume --configs \
  magnitude gmp nm_sparsity l1_filter l1_filter_30 fpgm group_l2 taylor slimming hrank thinet layer_drop \
  ptq_minmax ptq_minmax_pertensor ptq_percentile ptq_mse ptq_kl dfq adaround brecq ptq_minmax_w4a8 \
  qat_ste lsq lsq_plus hawq prune_quant deep_compression

# ViT-tiny: methods that work on attention heads, tokens and transformer activations.
$PY benchmark.py --name vit_tiny --models vit_tiny --resume --configs \
  tome tome_r4 tome_r8_ft evit head_prune layer_drop l1_filter magnitude \
  ptq_minmax ptq_minmax_vit_full ptq_mse fq_vit ptq4vit lsq
