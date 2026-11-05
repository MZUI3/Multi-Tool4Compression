# Multi-Tool4Compression

[English](README.md) | 한국어

모델 압축 기법 48개(Pruning 22, Quantization 22, 파이프라인 4)를 모은 PyTorch 툴킷입니다. 모든 기법을 하나의 스크립트에서 YAML config로 실행하며, 학습된 CIFAR-10 baseline에 적용합니다. 실행할 때마다 정확도 옆에 같은 비용 지표를 남깁니다: 파라미터 수와 0이 아닌 가중치 수, 저장 용량(byte), MACs, PyTorch CPU latency, ONNX Runtime CPU latency(fp32와 int8). 그래서 어떤 압축이 모델을 실제로 빠르게 하고 어떤 압축이 크기만 줄이는지 구분됩니다. 모델은 [Multi-Tool4Vision](https://github.com/MZUI3/Multi-Tool4Vision)의 registry를 그대로 쓰고, `mtcompress/common/hwaccel.py`는 그 레포 및 [Multi-Tool4RL](https://github.com/MZUI3/Multi-Tool4RL)과 같은 파일입니다.

<!-- RESULTS -->

## 설치

```bash
pip install -r requirements.txt   # torch, torchvision, numpy, pyyaml, onnx, onnxruntime, pytest, matplotlib
```

CIFAR-10은 처음 쓸 때 `./data`에 내려받습니다. `results/`에 쓴 baseline 세 개는 `checkpoints/`에 들어 있습니다.

## 빠른 시작

```bash
git clone https://github.com/MZUI3/Multi-Tool4Compression.git && cd Multi-Tool4Compression
pip install -r requirements.txt
python compress.py --list                                   # 모든 기법과 계열, 논문
python compress.py --method l1_filter --model resnet18      # configs/prune/l1_filter.yaml을 ResNet18 baseline에 적용
python compress.py --method ptq_mse --model resnet18        # int8 PTQ, ONNX Runtime int8 latency까지 측정
python evaluate.py runs/l1_filter_resnet18_s0_*             # 저장된 run을 다시 측정
```

## 사용법

- `train.py`: baseline을 학습해 `checkpoints/<model>_<dataset>.pth`에 저장합니다. Multi-Tool4Vision의 모델 이름이면 무엇이든 됩니다(`resnet18`, `mobilenet_v2`, `mobilevit`, `vit_tiny` 등). `results/`에 쓴 세 번의 학습은 `tools/train_baselines.sh`에 있습니다.
- `compress.py`: 기법 하나를 적용합니다.
  - `--method`: 기법 이름이나 변형 config 이름(`configs/<kind>/<name>.yaml`)
  - `--config`: 임의의 YAML 파일
  - `--set`: 점으로 이은 `KEY=VALUE`를 YAML로 해석합니다(`--set hparams.sparsity=0.9 hparams.exclude=[]`). 리스트 항목은 번호로 지정합니다(`hparams.stages.0.hparams.ratio=0.5`).
  - `--hparams`: 기법의 hparam과 기본값을 출력합니다.
  - `--limit-batches N`: 디버깅용으로 모든 pass를 N batch로 자릅니다.
- `evaluate.py`: run 디렉터리를 다시 불러와 측정합니다.
- `benchmark.py`: suite(`prune`, `quant`, `pipeline`, `all`)나 config 목록을 하나 이상의 baseline에 돌리고 `results/<name>.json`과 `.md`를 씁니다. `--resume`은 JSON에 이미 있는 run을 건너뜁니다. `results/`에 쓴 실행 목록은 `tools/run_benchmarks.sh`에 있습니다.
- `plot.py`: 결과 파일로 정확도-저장 용량, 정확도-MACs 그래프를 그립니다.

run 디렉터리 `runs/<variant>_<model>_s<seed>_<time>/`에는 다음 파일이 있습니다.

- `config.yaml`: 모든 기본값까지 채운 config
- `model.pt`: 압축된 모델을 모듈째 pickle한 파일. pruning과 양자화가 구조를 바꾸기 때문입니다. 레포 루트에서 `torch.load(path, weights_only=False)`로 불러옵니다.
- `progress.jsonl`: fine-tuning epoch마다 한 줄
- `summary.json`: baseline과 최종 정확도·비용, 층별 sparsity·남은 채널·고른 비트 폭 같은 기법별 정보, 하드웨어

```python
from mtcompress.runner import load_run
model, cfg, summary = load_run("runs/l1_filter_resnet18_s0_20261106-101500")
print(summary["final"]["acc"], summary["final"]["macs"], summary["info"]["groups"][:3])
```

## 측정 항목

| 필드 | 의미 |
|---|---|
| `acc` | CIFAR-10 test 10,000장 top-1 |
| `params`, `params_nonzero`, `weight_sparsity` | 모델 파라미터 수(양자화기의 step size·clipping 값은 빼고 셈)와 conv/linear 가중치 중 0의 비율 |
| `size_bytes`, `compression_ratio` | state dict 저장 용량(비율은 fp32 baseline 대비). float는 32비트, 양자화된 가중치는 양자화 비트 폭 + 32비트 scale, sparse 텐서는 `min(dense, nnz·bits + 1비트 mask)`, k-means 공유는 codebook + Huffman 코드. 압축용으로만 있는 buffer는 세지 않습니다. |
| `macs`, `sparse_macs` | 모델이 실제로 가진 shape에서 이미지 한 장당 곱셈-누산 수(attention 행렬곱 포함). `sparse_macs`는 층마다 가중치 밀도를 곱한 값으로, sparse 커널이 건너뛸 수 있는 양입니다. |
| `torch_cpu_latency_ms` | PyTorch CPU latency 중앙값, batch 1, 4 스레드. fake-quant 모델은 에뮬레이션 시간을 재게 되므로 기록하지 않습니다. |
| `onnx_fp32_latency_ms` | float 그래프의 ONNX Runtime CPU latency, batch 1, 4 스레드 |
| `onnx_int8_latency_ms` | 같은 그래프를 ONNX Runtime static quantization(QDQ, 채널별 int8 가중치, uint8 activation)으로 양자화해 `QLinearConv` / `QGemm` / `QLinearAdd` 커널로 돌린 latency. 8비트 uniform 기법에서는 이것이 배포 latency이고, 정확도는 그 기법의 fake-quant 모델 정확도입니다. 8비트 미만이거나 log2·twin·power-of-two-factor 양자화기를 쓰는 기법은 ONNX Runtime CPU 커널이 없어 latency를 기록하지 않습니다. |

정확도를 재는 fake-quant 배치는 Q/DQ를 층 입력에만 둡니다. 이대로 내보내면 ONNX Runtime이 거의 합치지 못합니다. MobileViT에서는 conv가 하나도 `QLinearConv`가 되지 않았고, ResNet18은 20개 중 8개만 합쳐졌습니다. 그래서 int8 latency는 static-QDQ 그래프로 잽니다.

## 기법

이름은 `--method`에 쓰는 이름입니다. `python compress.py --list`로 논문까지 볼 수 있습니다.

| 종류 | 계열 | 기법 |
|---|---|---|
| **Pruning (22)** | 비구조적 | `magnitude`, `random`, `obd`, `gmp`, `movement` |
| | 초기화 시점 | `snip`, `grasp`, `synflow`, `random_init` |
| | Lottery ticket | `imp` (rewinding: `weights`, `lr`, `none`) |
| | 반구조적 | `nm_sparsity` (2:4) |
| | 구조적 채널 | `l1_filter`, `group_l2`, `fpgm`, `slimming`, `taylor`, `hrank`, `thinet` |
| | Transformer | `head_prune`, `layer_drop` |
| | 토큰 축소 | `tome`, `evit` |
| **Quantization (22)** | PTQ calibration | `ptq_minmax`, `ptq_percentile`, `ptq_mse`, `ptq_kl` |
| | PTQ data-free | `dfq` |
| | PTQ reconstruction | `adaround`, `brecq`, `qdrop` |
| | PTQ transformer | `fq_vit`, `ptq4vit` |
| | QAT | `qat_ste`, `lsq`, `lsq_plus`, `pact`, `dorefa`, `apot` |
| | Binary / ternary | `bnn`, `xnor`, `twn` |
| | 혼합 정밀도 | `mixed_sensitivity`, `hawq` |
| | 가중치 공유 | `kmeans_share` |
| **파이프라인 (4)** | | `pipeline` (임의 단계), `deep_compression`, `prune_quant`, `prune_quant_kd` |

구현이 맞물리는 방식:

- **구조적 pruning은 채널을 실제로 지웁니다.** `mtcompress/prune/depgraph.py`가 module hook과 `TorchFunctionMode`로 forward를 한 번 기록합니다. 그래서 `x.shape`를 풀거나 텐서를 reshape하는 모델(MobileViT)도 실제 shape으로 추적됩니다. residual 덧셈은 채널 그룹을 합치고, concat은 그룹을 나란히 붙이며, reshape·permute는 채널 축을 옮깁니다. 모르는 연산은 해당 그룹을 고정합니다. 채널 기준 7개가 모두 이 그래프를 함께 씁니다. 채널 하나를 지우면 producer의 filter, 모든 consumer의 입력 채널, norm 항목, element-wise로 더해지는 파라미터(ViT의 position embedding)를 함께 자릅니다. ResNet, MobileNetV2, MobileViT, ExMobileViT, ViT를 모델별 코드 없이 처리합니다.
- **필요할 때 attention을 명시적인 모듈로 바꿉니다.** `nn.MultiheadAttention`과 ViT의 `Attention`(`scaled_dot_product_attention` 호출)을 같은 함수를 계산하는 `SplitAttention`으로 바꿉니다(최대 오차 4e-7). softmax, head, projection이 드러나므로 head pruning, 토큰 병합, softmax 양자화기, ONNX export가 이를 씁니다.
- **fine-tuning 중에도 mask를 유지합니다.** 비구조적 mask는 `backward` 뒤에 gradient를, optimizer step 뒤에 가중치를 0으로 만듭니다. 그래서 momentum이나 weight decay가 지운 가중치를 되살리지 못합니다.
- **PTQ는 먼저 BatchNorm을 folding합니다.** 배포되는 int8 모델은 folding된 가중치로 돌기 때문입니다. `dfq`는 여기에 cross-layer equalisation과 empirical bias correction을 더합니다. CLE는 ReLU 계열 activation을 지나는 경우에만 적용하며, 이를 확인합니다.
- **AdaRound, BRECQ, QDrop은 reconstruction 엔진 하나를 함께 씁니다.** 단위는 층 하나(AdaRound) 또는 양자화 층이 8개 이하인 호출 단위 서브모듈(BRECQ, QDrop)입니다. 입력은 일부가 양자화된 모델에서, 목표값은 float 복사본에서 가져옵니다. 끝나면 학습한 rounding을 가중치에 써 넣어 다시 일반 fake-quant 모델로 만듭니다.
- **모든 기법이 fine-tuning 중에 압축 전 모델로부터 distillation할 수 있습니다** (`kd_alpha`, `kd_temperature`).

논문과 다르게 구현한 곳은 클래스 docstring에 적었습니다. 다른 점은 다음과 같습니다.

- `thinet`: greedy 선택만 하고 least-squares 재조정은 하지 않습니다.
- `obd`: Hessian 대각을 empirical Fisher로 근사합니다.
- `dfq`: analytic 대신 empirical bias correction을 씁니다.
- `ptq4vit`: Hessian-guided 지표 대신 MSE로 scale을 찾습니다.
- `dorefa`: 가중치에 층의 초기 max|w|를 곱합니다.
- `xnor`: K 행렬 대신 샘플별 scale을 씁니다.
- `apot`: 두 개의 additive power-of-two 항으로 level을 만들고, activation은 LSQ로 양자화합니다.
- `hawq`: 비트 배분을 ILP 대신 정확한 DP로 풉니다.
- QAT의 activation 양자화기는 calibration 데이터에 음수가 있으면 signed로 둡니다(논문은 ReLU 입력을 가정합니다).
- reconstruction은 단위당 20,000회가 아니라 1,000회 반복합니다.

## Config

```yaml
method: l1_filter          # registry 이름
model: mobilevit           # baseline, checkpoints/<model>_<dataset>.pth
dataset: cifar10           # cifar10 | cifar100
applies_to: [mobilevit]    # 선택: benchmark.py는 다른 모델에서 건너뜀
hparams:                   # 기법 기본값과 다른 것만
  ratio: 0.3
  kd_alpha: 0.7
```

- 공통 fine-tuning hparam: 모든 기법이 다음을 함께 씁니다.
  - `ft_epochs`, `ft_lr`, `ft_optimizer`(`sgd`, `adamw`, `adam`), `ft_weight_decay`, `ft_batch_size`, `ft_warmup_epochs`, `kd_alpha`, `kd_temperature`
  - `exclude`: 층 이름 prefix, 또는 첫 conv와 classifier를 가리키는 `first` / `last`
- 나머지 hparam은 각 클래스가 선언하고, 클래스 계층을 따라 합쳐집니다. 모르는 hparam이 들어오면 유효한 키 목록을 보여 주며 멈춥니다.
- 최상위 키와 기본값:
  - `seed: 0`, `device: auto`, `checkpoint: auto`, `save_dir: runs`
  - `amp: true`: float 모델의 fine-tuning에만 적용되고, fake quantisation은 float32로 돕니다.
  - `measure: {latency: true, onnx: true, threads: 4, iters: 100, gpu_latency: false}`
- 우선순위: 기본값 → YAML 파일 → `--model`, `--seed` 등 플래그 → `--set`
- `tools/make_configs.py`가 `configs/`를 씁니다. 기법마다 고유 hparam을 모두 적은 config 하나와 비교용 변형(`magnitude_90`, `ptq_minmax_w4a8`, `ptq_minmax_vit_full`, `lsq_w2a2` 등)입니다.

## 기법 추가

`mtcompress/prune/`이나 `mtcompress/quant/`에 클래스를 만들어 등록하고, 그 패키지의 `__init__.py`에서 모듈을 import합니다.

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

`ctx`에는 다음이 들어 있습니다.

- 모델
- 데이터: device에 올라간 `train`, `val`, `calib` iterable
- 고정된 teacher
- `summary.json`으로 들어가는 `info` dict

그다음 `python tools/make_configs.py`와 `python tests/test_smoke.py my_prune`을 실행합니다.

## 프로젝트 구조

```
train.py, compress.py, evaluate.py, benchmark.py, plot.py   # 진입점
configs/prune/, configs/quant/, configs/pipeline/           # 기법별 config + 비교용 변형
mtcompress/registry.py, base.py, config.py, runner.py       # 기법 registry, Method/Context, YAML, 실행·측정·저장
mtcompress/data.py, train.py                                # device에 올린 CIFAR, KD와 hook을 지원하는 fit/evaluate
mtcompress/measure/profile.py                               # 파라미터, 저장 용량, MACs, PyTorch·ONNX Runtime latency
mtcompress/nn/attention.py                                  # SplitAttention과 변환기
mtcompress/prune/   masks, unstructured, at_init, lottery, nm, depgraph, structured, transformer (.py)
mtcompress/quant/   observers, core, fold, ptq, recon, vit, qat, lowbit, mixed, kmeans (.py)
mtcompress/pipeline.py                                      # 기법 단계 연결
mtcompress/common/hwaccel.py                                # Multi-Tool4Vision / Multi-Tool4RL과 공유
models/                                                     # Multi-Tool4Vision 모델 registry (136개)
checkpoints/                                                # results/에 쓴 baseline
results/                                                    # 벤치마크 표(.md), 원본 summary(.json), 그래프
tools/   train_baselines.sh, run_benchmarks.sh, make_configs.py
tests/test_smoke.py
```

## 테스트

```bash
python tests/test_smoke.py              # 모든 기법, 무작위 초기화 모델, pass마다 2 batch
python tests/test_smoke.py lsq brecq    # 일부만; name:model로 모델 지정, 예: thinet:resnet18
pytest tests/
```

smoke test는 기법이 돌고, 저장되고, 다시 불러와지는지만 확인합니다. 압축이 잘 되는지는 `results/`에서 확인합니다.
