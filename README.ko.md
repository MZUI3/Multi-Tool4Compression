# Multi-Tool4Compression

[English](README.md) | 한국어

모델 압축 기법 48개(Pruning 22, Quantization 22, 파이프라인 4)를 모은 PyTorch 툴킷입니다. 모든 기법을 하나의 스크립트에서 YAML config로 실행하며, 학습된 CIFAR-10 baseline에 적용합니다. 실행할 때마다 정확도 옆에 같은 비용 지표를 남깁니다: 파라미터 수와 0이 아닌 가중치 수, 저장 용량(byte), MACs, PyTorch CPU latency, ONNX Runtime CPU latency(fp32와 int8). 그래서 어떤 압축이 모델을 실제로 빠르게 하고 어떤 압축이 크기만 줄이는지 구분됩니다. 모델은 [Multi-Tool4Vision](https://github.com/MZUI3/Multi-Tool4Vision)의 registry를 그대로 쓰고, `mtcompress/common/hwaccel.py`는 그 레포 및 [Multi-Tool4RL](https://github.com/MZUI3/Multi-Tool4RL)과 같은 파일입니다.

## 결과

실험 조건:

- **데이터**: CIFAR-10, 32×32
- **fine-tuning**: RTX 3060 한 장
- **latency**: AMD Ryzen 7 5800X, batch 1, 4 스레드(PyTorch 2.11, ONNX Runtime 1.31)
- **seed**: 하나(0)
- **예산**: 대부분 기법은 5 epoch만 fine-tune하고, 초기화 시점 pruning은 처음부터 30 epoch 학습합니다. 그래서 절대 수치는 논문보다 낮습니다. 보려는 것은 같은 예산에서 기법끼리의 비교입니다.

전체 표(108회 실행)는 [results/mobilevit.md](results/mobilevit.md), [results/resnet18.md](results/resnet18.md), [results/vit_tiny.md](results/vit_tiny.md)에 있습니다. 원본 summary는 같은 이름의 `.json`, 정확도-비용 그래프는 `results/*.png`입니다.

| Baseline | Top-1 | 파라미터 | fp32 용량 | MACs | PyTorch CPU | ORT fp32 | ORT int8 |
|---|---|---|---|---|---|---|---|
| `resnet18` (30 epoch) | 93.89% | 11.17M | 42.7 MiB | 555.4M | 12.06 ms | 4.99 ms | 2.01 ms |
| `mobilevit` (60 epoch) | 84.24% | 0.44M | 1.69 MiB | 7.39M | 1.35 ms | 0.255 ms | 0.317 ms |
| `vit_tiny` (60 epoch) | 83.37% | 5.36M | 20.5 MiB | 365.1M | 9.11 ms | 5.13 ms | 5.55 ms |

**실제로 빨라지는 것: ResNet18**

| 기법 | Top-1 | 용량 | MACs | ORT fp32 | ORT int8 |
|---|---|---|---|---|---|
| baseline | 93.89 | 42.7 MiB | 555M | 4.99 ms | 2.01 ms |
| `gmp`: 비구조적 80% | 93.72 | 9.9 MiB | 555M | 4.73 ms | 1.97 ms |
| `nm_sparsity`: 2:4 | 93.71 | 22.7 MiB | 555M | 4.98 ms | 2.01 ms |
| `ptq_mse`: int8 PTQ | 93.88 | 10.7 MiB | 555M | — | **2.00 ms** |
| `l1_filter_30`: 채널 30% 제거 | 93.38 | 21.0 MiB | 275M | 2.99 ms | 1.47 ms |
| `prune_quant`: 채널 30% + int8 PTQ | 93.75 | 5.3 MiB | 275M | — | **1.49 ms** |
| `slimming`: 채널 50% 제거 | 92.53 | 10.7 MiB | 139M | 1.07 ms | **0.60 ms** |
| `deep_compression`: GMP 80% + 5비트 k-means + Huffman | 93.74 | **2.4 MiB** | 555M | 4.85 ms | 2.06 ms |

- 비구조적 sparsity와 2:4 sparsity는 저장 용량은 줄이지만 dense CPU latency는 그대로입니다. 2:4용 GPU 커널(cuSPARSELt)은 Windows용 PyTorch에 없어서, 측정할 수 없었다는 사실만 run에 기록됩니다.
- 채널 제거와 int8은 효과가 겹칩니다. 채널 50% + int8이면 latency가 4.99 → 0.60 ms(8.3배)이고, 정확도는 1.4%p 떨어집니다.
- 채널 기준 7개는 50%에서 서로 0.4%p 안에 모입니다(92.18~92.53). 이 예산에서는 채널을 어떻게 고르느냐보다 얼마나 지우느냐가 결과를 좌우합니다.

**작은 모델은 다릅니다: MobileViT (MACs 7.4M)**

- ONNX Runtime에서 int8이 fp32보다 **느립니다**(0.317 vs 0.255 ms). 이 크기에서는 양자화·역양자화 오버헤드가 싼 연산으로 얻는 이득보다 큽니다. `vit_tiny`도 같습니다(5.55 vs 5.13 ms). attention 행렬곱과 LayerNorm이 float로 남기 때문입니다.
- `layer_drop`은 calibration set에서 블록마다 빼 봤을 때의 손실을 재고, MobileViT 블록의 **transformer 층 2개를 모두** 뺐습니다. 결과는 83.71%(−0.53), MACs 3.06M이고, ORT latency는 0.255 → 0.090 ms입니다. 이 작은 MobileViT는 CIFAR-10에서 attention의 기여가 작다는 뜻입니다. 블록 단위로 재 보면 드러나지만 채널 pruning으로는 보이지 않습니다. 채널 pruning으로 비슷한 MACs(3.38M, `l1_filter_global`)까지 줄이면 79.32%입니다.
- `head_prune`은 head 절반을 지우고 83.97%입니다.

**비구조적 pruning과 lottery ticket (MobileViT)**

- 50%에서 `magnitude`(83.96)와 `obd`(83.96)는 같고, `random`(78.88)은 5%p 낮습니다.
- 80%에서는 점진적 magnitude pruning(`gmp`, 83.68)이 `movement`(64.21)를 크게 앞섭니다. movement pruning은 다른 태스크로 fine-tuning할 때를 위한 기법이고, 이미 학습한 태스크에서는 magnitude가 더 나은 신호라고 논문도 말합니다.
- `imp`는 sparsity 86%에서 LR rewinding 82.69, weight rewinding 82.99입니다.
- 초기화 시점 pruning(80%, 처음부터 30 epoch): `snip` 79.81, `synflow` 79.77, `grasp` 77.99로 모두 `random_init`(72.88)보다 높습니다. 같은 SNIP을 학습된 모델에 적용하면 83.72입니다(`snip_trained`).

**양자화**

- int8 PTQ의 손실은 최대 0.17%p입니다(ResNet18 93.85~93.91, MobileViT 84.07~84.28).
- `dfq`는 ResNet18에서 cross-layer equalisation 쌍 8개를 찾았고, MobileViT에서는 0개입니다. SiLU는 양의 동차성이 없어 equalisation을 하면 함수가 바뀌기 때문입니다.
- ResNet18, 가중치 4비트·activation 8비트: `ptq_minmax_w4a8` 93.61, `adaround` 93.70, `brecq` 93.79
- ResNet18, 가중치·activation 모두 4비트: `brecq_w4a4` 93.34, `qdrop` 93.50
- MobileViT에서는 W4A4 PTQ가 무너집니다: `ptq_mse_w4a4` 10.52, `brecq_w4a4` 37.45, `qdrop` 26.18. QAT는 버팁니다(`lsq` W4A4 81.30, `lsq_plus` 82.08). SiLU와 attention이 있는 0.44M 파라미터 모델에서 4비트 activation은 학습이 필요합니다.
- **Transformer**: `vit_tiny`에서 4비트 attention 확률과 LayerNorm 입력을 uniform 양자화기로 양자화하면 78.13입니다(`ptq_minmax_vit_full`). 같은 자리에 FQ-ViT의 power-of-two factor와 log2 양자화기를 쓰면 **82.96**입니다. `ptq4vit`는 8비트 twin uniform으로 83.28입니다.
- 혼합 정밀도(MobileViT): `hawq`가 가중치 평균 3.0비트에서 83.61%, 저장 용량 183 KiB(9.4배)입니다.
- 가장 낮은 비트 폭(MobileViT): `twn`(ternary 가중치) 143 KiB에서 81.96, `xnor` 56.07, `bnn`(가중치·activation 이진화) 45.51

**토큰 (ViT-tiny)**

`tome`은 r = 8로 토큰을 병합해 MACs를 3분의 1(123.5M)로 줄입니다. 학습 없이 77.89이고, 3 epoch fine-tuning 후 81.17입니다. 하지만 CPU batch 1에서는 baseline보다 느립니다(11.5 ms vs 9.1 ms). 병합 연산이 작고 불규칙하기 때문입니다. `evit`는 세 층에서 토큰 70%만 남기고 82.10, 246.9M MACs입니다. MACs가 줄어도 latency가 줄지는 않았습니다. 그래서 모든 run에서 MACs만 보고하지 않고 실제 시간을 잽니다.

## 측정하면서 발견한 문제

벤치마크를 만들면서 다섯 가지 문제를 만났습니다. 모두 오류 메시지 없이 틀린 숫자를 내는 종류였습니다.

- **fake-quant QDQ는 int8 모델이 아닙니다.** fake-quant 모델(Q/DQ가 층 입력에만 있음)을 그대로 내보내면 MobileViT가 fp32 0.26 ms 대비 1.34 ms였습니다. ONNX Runtime이 conv를 하나도 합치지 못했기 때문입니다. 지금은 int8 latency를 같은 그래프의 ONNX Runtime static QDQ 양자화로 재며, 이때 ResNet18의 conv 20개가 모두 `QLinearConv`로 돕니다.
- **KL calibration과 ReLU.** ReLU 출력은 절반이 정확히 0입니다. |x| 히스토그램에서 이 spike 때문에 넓은 threshold가 항상 나빠 보였고, KL이 늘 범위의 약 1/10을 골라 ResNet18 int8이 32.6%까지 떨어졌습니다. 0을 빼고 나면 93.91%입니다.
- **in-place activation과 저장한 목표값.** `detach()`로 저장한 reconstruction 목표값을 각 conv 뒤의 `ReLU(inplace=True)`가 덮어썼습니다. 그래서 AdaRound가 활성화 전 출력을 활성화 후 목표에 맞췄고, ResNet18에서 87.8%로 단순 반올림보다 낮았습니다. `clone()`으로 저장하면 93.70%입니다.
- **`nn.MultiheadAttention`**은 ONNX symbol이 없는 fused kernel로 export됩니다. 먼저 `SplitAttention`으로 바꾸면 출력이 같고 export도 됩니다.
- **TorchScript ONNX exporter**는 ToMe 그래프에서 heap을 깨뜨려 프로세스를 죽입니다. ONNX 측정을 자식 프로세스에서 돌리게 바꿔, 한 번의 crash는 측정 하나만 잃고 벤치마크는 계속됩니다.

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
