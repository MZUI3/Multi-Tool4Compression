#!/usr/bin/env bash
# Baselines used by results/: CIFAR-10, 32x32, one RTX 3060.
set -e
PY=${PY:-python}
$PY train.py --model mobilevit --epochs 60 --lr 0.1 --amp
$PY train.py --model resnet18 --epochs 30 --lr 0.1 --amp
$PY train.py --model vit_tiny --epochs 60 --optimizer adamw --lr 1e-3 --weight-decay 0.05 --warmup-epochs 5 --amp
