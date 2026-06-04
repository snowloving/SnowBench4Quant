# SnowBench4Quant

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**A unified benchmark for full-precision, binary, and quantized CNNs on standard image classification datasets.**

</div>

---

## 📌 Overview

SnowBench4Quant is a reproducible benchmark suite designed for training and evaluating **full-precision**, **binary**, and **quantized** neural networks on standard image classification datasets. It serves as a baseline framework for research on binary neural networks (BNNs), model quantization, model compression, and efficient deep learning.

**Key features:**
- ✅ Multiple datasets: CIFAR-10, CIFAR-100, Tiny-ImageNet, ImageNet
- ✅ Multiple backbones: VGG, ResNet, AlexNet, BiRealNet
- ✅ Training modes: Full-precision, Binary (XNOR-style), and Quantized (DoReFa-style)
- ✅ Multiple optimizers: SGD, Adam, Bop, Bop2ndOrder, SGDAT
- ✅ Reproducible configurations for fair comparisons
  
---

## 📂 Repository Structure
```text
SnowBench4Quant/
├── README.md                   
├── requirements.txt            
├── data.py                     # Dataset loader: CIFAR-10/100, Tiny-ImageNet, ImageNet (with path configuration)
├── preprocess.py               # Data augmentation & preprocessing pipelines
├── utils.py                    # Logger, metrics, checkpointing, optimizer adjustment
│
├── datasets/
├── results/
├── optimizers/                 # Bop/Bop2ndOrder/SGDAT
│
├── models_sgdat/               # Binary models (SGDAT-style)
│   ├── __init__.py.py/         # __all__ = ['binarynet', 'resnet_binary']
│   ├── binarized_modules.py/   # BinarizeLinear, BinarizeConv2d (1w1a / 1w32a)
│   ├── binarynet.py/          
│   └── resnet_binary.py/      
│  
├── models_full_cifar/          # Full-precision models (CIFAR-scale)
│   ├── __init__.py.py/         # __all__ = ['vgg_small', 'vgg16', 'resnet18', 'resnet20', 'resnet56', 'resnet18_preact', 'resnet20_preact', 'resnet56_preact']
│   ├── vgg.py/                 # Full-precision versions of VGG-family
│   ├── vgg_opt.py/             # Same architecture as vgg.py, but with optimized code style (cleaner implementation)
│   ├── resnet.py/              # Full-precision versions of ResNet-family
│   └── resnet_preact.py/       # Full-precision versions of PreActResNet-family
│
├── models_binarynet/           # Binary & Quantized (BinaryNet-style)
│   ├── __init__.py.py/         # __all__ = ['vgg_small_binary', 'vgg16_binary', 'resnet18_binary', 'resnet20_binary', 'resnet56_binary', 'resnet_binary', 'vgg_small_quant', 'vgg16_quant', 'resnet18_quant', 'resnet20_quant', 'resnet56_quant']
│   ├── binarized_modules.py/   # Binarize / quantize layers & functions
│   ├── vgg.py/                 # Binary VGG
│   ├── resnet.py/              # Binary ResNet
│   ├── resnet_opt.py/          # Binary ResNet: optimized code structure with slightly different architectural details compared to the standard `resnet.py`
│   ├── vgg_quant.py/           # Quantized VGG
│   └── resnet_quant.py/        # Quantized ResNet
│
├── models_dorefanet/           # Binary & Quantized (DoreFaNet-style)
│   ├── __init__.py.py/         # __all__ = ['vgg_small_quant', 'vgg16_quant', 'resnet18_quant', 'resnet20_quant', 'resnet56_quant']
│   ├── binarized_modules.py/   # Binarize / quantize layers & functions
│   ├── vgg.py/                 # Binary & Quantized VGG
│   └── resnet.py/              # Binary & Quantized ResNet
│
├── models_qat/                 # Quantized-Aware Training
│   ├── __init__.py/            # __all__ = ['resnet18_preact_quant', 'resnet20_preact_quant', 'resnet56_preact_quant']
│   ├── resnet_preact.py        # [Clean] Pure network architecture definition — completely agnostic to quantization method
│   ├── q_layers.py             # [Hub] Defines QConv2d, QLinear — dispatches to different quantizers
│   │
│   └── quantizers/             # [Core] Algorithm factory — one file per top-conference method!
│       ├── __init__.py         # Contains get_weight_quantizer() and get_act_quantizer()
│       ├── dorefa.py           # DoReFa-Net (Zhou et al., 2016)
│       ├── pact.py             # PACT — PArameterized Clipping acTivation (Choi et al., ICML 2018)
│       ├── lsq.py              # LSQ — Learned Step Size Quantization (Esser et al., ICLR 2020)
│       ├── lsq_plus.py         # LSQ+ — LSQ with learnable offsets (Bhalgat et al., ICLR 2021)
│       ├── dsq.py              # DSQ — Differentiable Soft Quantization (Gong et al., AAAI 2021)
│       └── ...
│
├── models_full_imagenet/       # Full-precision models (ImageNet-scale)
│   ├── __init__.py.py/         # __all__ = []
│   ├── alexnet.py/             # Full-precision versions of AlexNet
│   ├── birealnet.py/           # Full-precision versions of BiRealNet
│   └── resnet.py/              # Full-precision versions of ResNet18
│
├── main_binary_sgdat.py        # Entry: binary models (models_sgdat)
├── main_full_cifar.py          # Entry: full-precision on cifar (models_full_cifar)
├── main_binary_binarynet.py    # Entry: binary & full-precision (models_binarynet)
├── main_quant_dorefa.py     # Entry: quantized & full-precision (models_dorefanet) — nearly identical to main_binary_binarynet.py
├── main_modern_qat.py          # Entry: qat methods (models_qat) — nearly identical to main_binary_binarynet.py
├── main_full_imagenet.py       # Entry: full-precision on imagenet (models_full_imagenet)
└
```

---

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/snowloving/SnowBench4Quant.git
cd SnowBench4Quant
pip install -r requirements.txt
```

---


## 🖥️ Experiments

This benchmark supports following main experimental tracks:

### 🎯 Experiments Navigator
 
- [🔬 Experiment 1](#exp1): Binary Network Optimizer Comparison (SGDAT-style) — SGD vs Adam vs Bop vs Bop2ndOrder vs SGDAT
- [⚗️ Experiment 2](#exp2): Full-Precision Network Accuracy Comparison on CIFAR — FP32 baselines as reference ceiling
- [🔭 Experiment 3](#exp3): Binary & Quantized Network Accuracy Comparison on CIFAR (BinaryNet-style) — 1-bit & Multi-bit compression
- [🧬 Experiment 4](#exp4): Binary & Quantized Network Accuracy Comparison on CIFAR (DoReFaNet-style) — 1-bit & Multi-bit compression
- [🩺 Experiment 5](#exp5): Modern QAT SOTAs for Extreme Low-Bit (LSQ, EWGS, QDrop, etc.)
- [🧫 Experiment N](#expN): Full-Precision Network Accuracy Comparison on ImageNet — Large-scale datasets ceiling

---


<a id="exp1"></a>
### 🔬 Experiment 1: Binary Network Optimizer Comparison

 **Goal:** Compare different optimizers (SGD, Adam, Bop, Bop2ndOrder, SGDAT) for training binary neural networks.  

#### 📊 Results on BinaryNet

| Optimizer | CIFAR-10 | CIFAR-100 | Tiny-ImageNet |
|-----------|:--------:|:---------:|:-------------:|
| SGD | 90.04 | 64.42 | 46.69 |
| SGDM | 89.88 | 65.23 | 45.88  |
| Adam | 89.74 | 65.27 | 45.64 |
| Bop | 89.08 | 63.91 | 44.95 |
| Bop2ndOrder | 89.74 | 64.70 | 46.17 |
| SGDT | 89.17 | 64.57 | 44.94 |
| SGDAT | 90.15 | 65.50 | 46.67 |
> 📝 **Notes:**
> All results are from a single run with a fixed random seed (`seed_value=2020`). No hyperparameter tuning was performed.
> 
> ℹ️ **Note:** The "BinaryNet" used throughout this experiment refers to a compact VGG-style architecture (a.k.a. **VGG-Small**), implemented as `vgg_small` in `models_full_cifar/` and the "ResNet" used throughout this experiment refers to a ResNet18 architecture modified for ImageNet, implemented as `resnet18` in `models_full_cifar/`.

#### 📋 Quick Example Command

```bash
python main_binary_sgdat.py \
  --model binarynet \
  --save binarynet_cifar10_SGD \
  --dataset cifar10 \
  --bin_regime "{0: {'optimizer': 'SGD', 'lr': 1e-4}}" \
  --binarization det \
  --gpus 0
```

<details> <summary>🔁 All Reproducible Commands on BinaryNet</summary>

---

**CIFAR-10 with SGD** 
```bash
python main_binary_sgdat.py --model binarynet --save binarynet_cifar10_SGD --dataset cifar10 --bin_regime "{0: {'optimizer': 'SGD','lr':1e-4}}" --binarization det --input_size 32 --epochs 200 -b 256 --gpus 0
```

**CIFAR-10 with SGDM** 
```bash
python main_binary_sgdat.py --model binarynet --save binarynet_cifar10_SGDM --dataset cifar10 --bin_regime "{0: {'optimizer': 'SGD','lr':1e-4,'momentum':0.9}}" --binarization det --input_size 32 --epochs 200 -b 256 --gpus 2
```

**CIFAR-10 with Adam** 
```bash
python main_binary_sgdat.py --model binarynet --save binarynet_cifar10_Adam --dataset cifar10 --bin_regime "{0: {'optimizer': 'Adam','lr':1e-3}}" --binarization det --input_size 32 --epochs 200 -b 256 --gpus 2
```

**CIFAR-10 with Bop** 
```bash
python main_binary_sgdat.py --model binarynet --save binarynet_cifar10_Bop --dataset cifar10 --bin_regime "{0: {'optimizer': 'Bop','gamma':1e-4,'threshold':1e-8}}" --binarization det --input_size 32 --epochs 200 -b 256 --gpus 1
```

**CIFAR-10 with Bop2ndOrder** 
```bash
python main_binary_sgdat.py --model binarynet --save binarynet_cifar10_Bop2ndOrder --dataset cifar10 --bin_regime "{0: {'optimizer': 'Bop2ndOrder','gamma':1e-7,'sigma':1e-3,'threshold':1e-6}}" --binarization det --input_size 32 --epochs 200 -b 256 --gpus 0
```

**CIFAR-10 with SGDT** 
```bash
python main_binary_sgdat.py --model binarynet --save binarynet_cifar10_SGDT --dataset cifar10 --bin_regime "{0: {'optimizer': 'SGD','lr':1e-4}}" --binarization threshold --threshold 1e-8 --input_size 32 --epochs 200 -b 256 --gpus 1
```

**CIFAR-10 with SGDAT** 
```bash
python main_binary_sgdat.py --model binarynet --save binarynet_cifar10_SGDAT --dataset cifar10 --bin_regime "{0: {'optimizer':'SGDAT','lr':1e-4,'threshold':1e-7}}" --binarization det --input_size 32 --epochs 200 -b 256 --gpus 2
```

**CIFAR-100 with SGD** 
```bash
python main_binary_sgdat.py --model binarynet --save binarynet_cifar100_SGD --dataset cifar100 --bin_regime "{0: {'optimizer': 'SGD','lr':1e-4}}" --binarization det --input_size 32 --epochs 200 -b 256 --gpus 0
```

**CIFAR-100 with SGDM** 
```bash
python main_binary_sgdat.py --model binarynet --save binarynet_cifar100_SGDM --dataset cifar100 --bin_regime "{0: {'optimizer': 'SGD','lr':1e-4,'momentum':0.9}}" --binarization det --input_size 32 --epochs 200 -b 256 --gpus 1
```

**CIFAR-100 with Adam** 
```bash
python main_binary_sgdat.py --model binarynet --save binarynet_cifar100_Adam --dataset cifar100 --bin_regime "{0: {'optimizer': 'Adam','lr':1e-3}}" --binarization det --input_size 32 --epochs 200 -b 256 --gpus 0
```

**CIFAR-100 with Bop** 
```bash
python main_binary_sgdat.py --model binarynet --save binarynet_cifar100_Bop --dataset cifar100 --bin_regime "{0: {'optimizer': 'Bop','gamma':1e-4,'threshold':1e-8}}" --binarization det --input_size 32 --epochs 200 -b 256 --gpus 3
```

**CIFAR-100 with Bop2ndOrder** 
```bash
python main_binary_sgdat.py --model binarynet --save binarynet_cifar100_Bop2ndOrder --dataset cifar100 --bin_regime "{0: {'optimizer': 'Bop2ndOrder','gamma':1e-7,'sigma':1e-3,'threshold':1e-6}}" --binarization det --input_size 32 --epochs 200 -b 256 --gpus 0
```

**CIFAR-100 with SGDT** 
```bash
python main_binary_sgdat.py --model binarynet --save binarynet_cifar100_SGDT --dataset cifar100 --bin_regime "{0: {'optimizer': 'SGD','lr':1e-4}}" --binarization threshold --threshold 1e-8 --input_size 32 --epochs 200 -b 256 --gpus 2
```

**CIFAR-100 with SGDAT** 
```bash
python main_binary_sgdat.py --model binarynet --save binarynet_cifar100_SGDAT --dataset cifar100 --bin_regime "{0: {'optimizer':'SGDAT','lr':1e-4,'threshold':1e-7}}" --binarization det --input_size 32 --epochs 200 -b 256 --gpus 3
```

**tiny-imagenet with SGD** 
```bash
python main_binary_sgdat.py --model binarynet --save binarynet_tiny_imagenet_SGD --dataset tiny_imagenet --bin_regime "{0: {'optimizer': 'SGD','lr':1e-4}}" --binarization det --input_size 64 --epochs 100 -b 256 --gpus 0
```

**tiny-imagenet with SGDM** 
```bash
python main_binary_sgdat.py --model binarynet --save binarynet_tiny_imagenet_SGDM --dataset tiny_imagenet --bin_regime "{0: {'optimizer': 'SGD','lr':1e-4,'momentum':0.9}}" --binarization det --input_size 64 --epochs 100 -b 256 --gpus 1
```

**tiny-imagenet with Adam** 
```bash
python main_binary_sgdat.py --model binarynet --save binarynet_tiny_imagenet_Adam --dataset tiny_imagenet --bin_regime "{0: {'optimizer': 'Adam','lr':1e-3}}" --binarization det --input_size 64 --epochs 100 -b 256 --gpus 0
```

**tiny-imagenet with Bop** 
```bash
python main_binary_sgdat.py --model binarynet --save binarynet_tiny_imagenet_Bop --dataset tiny_imagenet --bin_regime "{0: {'optimizer': 'Bop','gamma':1e-4,'threshold':1e-8}}" --binarization det --input_size 64 --epochs 100 -b 256 --gpus 1
```

**tiny-imagenet with Bop2ndOrder** 
```bash
python main_binary_sgdat.py --model binarynet --save binarynet_tiny_imagenet_Bop2ndOrder --dataset tiny_imagenet --bin_regime "{0: {'optimizer': 'Bop2ndOrder','gamma':1e-7,'sigma':1e-3,'threshold':1e-6}}" --binarization det --input_size 64 --epochs 100 -b 256 --gpus 2
```

**tiny-imagenet with SGDT** 
```bash
python main_binary_sgdat.py --model binarynet --save binarynet_tiny_imagenet_SGDT --dataset tiny_imagenet --bin_regime "{0: {'optimizer': 'SGD','lr':1e-4}}" --binarization threshold --threshold 1e-8 --input_size 64 --epochs 100 -b 256 --gpus 2
```

**tiny-imagenet with SGDAT** 
```bash
python main_binary_sgdat.py --model binarynet --save binarynet_tiny_imagenet_SGDAT --dataset tiny_imagenet --bin_regime "{0: {'optimizer':'SGDAT','lr':1e-4,'threshold':1e-7}}" --binarization det --input_size 64 --epochs 100 -b 256 --gpus 3
```

</details>

#### ⚙️ Key Arguments
| Argument | Description | Options in this experiment |
|----------|-------------|---------------------------|
| `--bin_regime` | Optimizer & hyperparams for **binary layers** | `SGD`, `Adam`, `Bop`, `Bop2ndOrder`, `SGDAT` |
| `--fp_regime` | Optimizer & hyperparams for **full-precision layers** | Defaults to `Adam` |
| `--model` | Model architecture | `binarynet`, `resnet_binary` |
| `--dataset` | Dataset | `cifar10`, `cifar100`, `tiny-imagenet` |
| `--binarization` | Binarization method | `det` or `threshold` |

#### 🔨 Optimizer Configuration Reference

| Optimizer | `--bin_regime` Configuration |
|-----------|---------------------------|
| SGD | `"{0: {'optimizer': 'SGD','lr':1e-4}}"` |
| SGDM | `"{0: {'optimizer': 'SGD','momentum':0.9,'lr':1e-4}}"` |
| Adam | `"{0: {'optimizer': 'Adam','lr':1e-4}}"` |
| Bop | `"{0: {'optimizer': 'Bop','lr':1e-4}}"` |
| Bop2ndOrder | `"{0: {'optimizer': 'Bop2ndOrder','lr':1e-4}}"` |
| SGDT | `"{0: {'optimizer': 'SGDT','lr':1e-4}}"` & --binarization `threshold` |
| SGDAT | `"{0: {'optimizer': 'SGDAT','lr':1e-4}}"` |

#### 🔧 Optimizer-Specific Initialization

Binary networks in this benchmark use different latent weight (`weight.org`) initialization strategies depending on the optimizer:

| Optimizer Type | Initialization |
|:--------------:|---------------|
| SGD-like (SGD, SGDM, SGDT, SGDAT) | `self.weight.org = torch.zeros_like(self.weight)` |
| Others (Adam, Bop, Bop2ndOrder) | `self.weight.org = self.weight.data.clone().detach()` |

> **Why?** `zeros_like` makes SGD-like optimizers infinitely sensitive to early gradients — the first batch's gradient direction alone decides the sign of every binary weight, acting as a data-driven initialization. Adam and Bop don't need this: Adam's adaptive learning rate naturally amplifies small gradients, easily crossing the zero axis from any starting point.

---

<a id="exp2"></a>
### ⚗️ Experiment 2: Full-Precision Network Accuracy Comparison on CIFAR

**Goal:** Establish the FP32 upper-bound accuracy for standard CIFAR architectures. These results serve as the "ceiling" references when calculating the accuracy drop of binary and quantized models.

#### 📊 Results with SGD

| Architecture | Model Name | CIFAR-10 | CIFAR-100 | Structural Notes |
|-----------|:---------:|:---------:|:---------:|---------|
| VGG | vgg_small | 91.37 | 68.75 | BinaryNet baseline (7-layer) |
|  | vgg16 | 90.62 | 64.96 | CIFAR modified (No 4096-FCs) |
| ResNet | resnet20 | 89.15 | 61.75 | He et al. (16-32-64 channels) |
|  | resnet56 | 90.06 | 63.54 | He et al. (16-32-64 channels) |
|  | resnet18 | 92.08 | 68.42 | ImageNet modified (3x3 conv1) |
| PreActResNet | resnet20 | 89.28 | 60.75 | Pre-activation: ReLU *before* conv |
|  | resnet56 | 90.60 | 63.35 | Pre-activation: ReLU *before* conv |
|  | resnet18 | 92.18 | 68.73 | Pre-activation: ReLU *before* conv |

> ℹ️ **Default SGD configuration:** Unless otherwise noted, all full-precision SGD results use **lr = 0.1, momentum = 0.9, weight decay = 1e-4**, trained for **200 epochs**. These serve as the standard baseline shared across all full-precision experiments. Note that some models may benefit from longer training (e.g., 300–400 epochs); these results represent a fair but not fully converged comparison point.
>
> ⚠️ **On PreActResNet:** PreActResNet results are summarized here as they will serve as the backbone for upcoming quantization and binarization experiments. This architecture places activation functions (ReLU) *before* the convolution (rather than after, as in standard ResNet), preserving negative activations prior to binarization/quantization — maintaining the representational capacity of discrete features that would otherwise be clipped to zero by a post-conv ReLU.

#### 📋 Quick Example Command

```bash
python main_full_cifar.py \
  --model vgg_small \
  --save full_vgg_small_cifar10 \
  --dataset cifar10 \
  --gpus 0
```

<details> <summary>🔁 All Reproducible Commands </summary>
  
---

**CIFAR-10 on VGG_Small** 
```bash
python main_full_cifar.py --model vgg_small --save full_vgg_small_cifar10 --dataset cifar10 --epochs 200 -b 256 --gpus 0
```

**CIFAR-10 on VGG16** 
```bash
python main_full_cifar.py --model vgg16 --save full_vgg16_cifar10 --dataset cifar10 --epochs 200 -b 256 --gpus 2
```

**CIFAR-10 on ResNet20** 
```bash
python main_full_cifar.py --model resnet20 --save full_resnet20_cifar10 --dataset cifar10 --epochs 200 -b 256 --gpus 0
```

**CIFAR-10 on ResNet56** 
```bash
python main_full_cifar.py --model resnet56 --save full_resnet56_cifar10 --dataset cifar10 --epochs 200 -b 256 --gpus 1
```

**CIFAR-10 on ResNet18** 
```bash
python main_full_cifar.py --model resnet18 --save full_resnet18_cifar10 --dataset cifar10 --epochs 200 -b 256 --gpus 2
```

**CIFAR-100 on VGG_Small** 
```bash
python main_full_cifar.py --model vgg_small --save full_vgg_small_cifar100 --dataset cifar100 --epochs 200 -b 256 --gpus 1
```

**CIFAR-100 on VGG16** 
```bash
python main_full_cifar.py --model vgg16 --save full_vgg16_cifar100 --dataset cifar100 --epochs 200 -b 256 --gpus 3
```

**CIFAR-100 on ResNet20** 
```bash
python main_full_cifar.py --model resnet20 --save full_resnet20_cifar100 --dataset cifar100 --epochs 200 -b 256 --gpus 3
```

**CIFAR-100 on ResNet56** 
```bash
python main_full_cifar.py --model resnet56 --save full_resnet56_cifar100 --dataset cifar100 --epochs 200 -b 256 --gpus 2
```

**CIFAR-100 on ResNet18** 
```bash
python main_full_cifar.py --model resnet18 --save full_resnet18_cifar100 --dataset cifar100 --epochs 200 -b 256 --gpus 3
```

</details>

---

<a id="exp3"></a>
### 🔭 Experiment 3: Binary & Quantized Networks on CIFAR (BinaryNet-style) 

**Goal:** Evaluate extreme model compression techniques (1-bit BNNs & low-bit QNNs) using the standardized CIFAR architectures defined in Experiment 2.

#### 📊 Results of Binary Networks with SGD (1w1a)

| Architecture | Model Name | CIFAR-10 | CIFAR-100 |
|-----------|:---------:|:---------:|:---------:|
| VGG | vgg_small | 69.96 | 48.89 |
|  | vgg16 | 50.59 | 22.78 |
| ResNet | resnet20 | 74.64 | 39.86 |
|  | resnet56 | 74.22 | 41.86 |
|  | resnet18 | 81.24 | 54.50 |
> ℹ️ **Note on SGD:** The SGD hyperparameters used here (lr, momentum, weight decay, eopch) are identical to those in [Experiment 2](#exp2) for full-precision networks. However, SGD performs noticeably worse on binary neural networks compared to full-precision — this gap highlights the inherent difficulty of optimizing 1-bit weights with standard first-order methods, motivating the optimizer comparison in [Experiment 1](#exp1).
> 
> ℹ️ **Epoch:** Note that some models may benefit from longer training (e.g., 500–600 epochs); these results represent a fair but not fully converged comparison point.

#### 📋 Quick Example Command

```bash
python main_binary_binarynet.py \
  --model vgg_small_binary \
  --save vgg_small_binary_cifar10 \
  --dataset cifar10 \
  --gpus 0
```

<details> <summary>🔁 Results and Reproducible Commands for Binary Networks with SGD</summary>

---

**CIFAR-10 on VGG_Small** 
```bash
python main_binary_binarynet.py --model vgg_small_binary --save vgg_small_binary_cifar10 --dataset cifar10 --epochs 200 -b 256 --gpus 0
```

**CIFAR-10 on VGG16** 
```bash
python main_binary_binarynet.py --model vgg16_binary --save vgg16_binary_cifar10 --dataset cifar10 --epochs 200 -b 256 --gpus 1
```

**CIFAR-10 on ResNet20** 
```bash
python main_binary_binarynet.py --model resnet20_binary --save resnet20_binary_cifar10 --dataset cifar10 --epochs 200 -b 256 --gpus 1
```

**CIFAR-10 on ResNet56** 
```bash
python main_binary_binarynet.py --model resnet56_binary --save resnet56_binary_cifar10 --dataset cifar10 --epochs 200 -b 256 --gpus 2
```

**CIFAR-10 on ResNet18** 
```bash
python main_binary_binarynet.py --model resnet18_binary --save resnet18_binary_cifar10 --dataset cifar10 --epochs 200 -b 256 --gpus 0
```

**CIFAR-100 on VGG_Small** 
```bash
python main_binary_binarynet.py --model vgg_small_binary --save vgg_small_binary_cifar100 --dataset cifar100 --epochs 200 -b 256 --gpus 2
```

**CIFAR-100 on VGG16** 
```bash
python main_binary_binarynet.py --model vgg16_binary --save vgg16_binary_cifar100 --dataset cifar100 --epochs 200 -b 256 --gpus 3
```

**CIFAR-100 on ResNet20** 
```bash
python main_binary_binarynet.py --model resnet20_binary --save resnet20_binary_cifar010 --dataset cifar100 --epochs 200 -b 256 --gpus 0
```

**CIFAR-100 on ResNet56** 
```bash
python main_binary_binarynet.py --model resnet56_binary --save resnet56_binary_cifar100 --dataset cifar100 --epochs 200 -b 256 --gpus 2
```

**CIFAR-100 on ResNet18** 
```bash
python main_binary_binarynet.py --model resnet18_binary --save resnet18_binary_cifar100 --dataset cifar100 --epochs 200 -b 256 --gpus 1
```
</details>

---

#### 📊 Results of Binary Networks with Adam

| Architecture | Model Name | CIFAR-10 | CIFAR-100 |
|-----------|:---------:|:---------:|:---------:|
| VGG | vgg_small | 87.01 | 60.78 |
|  | vgg16 | 83.77 | 54.44 |
|  | vgg_small <sup>*</sup> | 89.59 | 65.06 |
|  | vgg16 <sup>*</sup> | 87.29 | 58.81 |
| ResNet | resnet20 | 67.75 | 29.50 |
|  | resnet56 | 62.89 | 24.88 |
|  | resnet18 | 82.87 | 52.61 |
|  | resnet20 <sup>$</sup> | 68.30 | 30.23 |
|  | resnet56 <sup>$</sup> | 62.76 | 26.38 |
|  | resnet18 <sup>$</sup> | 83.80 | 53.58 |
|  | resnet20 <sup>†</sup> | 84.26 | 53.21 |
> ℹ️ **Note on Adam:** These results use vanilla Adam without learning rate scheduling, gradient clipping, or step decay — the same basic configuration as the full-precision [Experiment 2](#exp2). No optimization tricks (warmup, cosine annealing, etc.) were applied, which may leave room for further improvement on binary networks.
>
> **Symbols:**
> - <sup>*</sup> Uses `infl_ratio = 3` (inflation ratio for latent weight initialization scaling). Other models use the default `infl_ratio = 1`.
> - <sup>$</sup> Uses **Option A** (parameter-free shortcut — no 1×1 conv, only identity mapping). Preferred for strict compression benchmarks and top-tier submissions where every parameter counts. Unmarked ResNet models use **Option B** (full-precision 1×1 conv in shortcut, a common practice in BNN literature to stabilize training and boost accuracy, though technically it keeps the residual projection uncompressed).
> - <sup>†</sup> Implementation from `resnet_opt.py` — optimized code structure with slightly different architectural details compared to the standard `resnet_binary.py`.


#### 📋 Quick Example Command

```bash
python main_binary_binarynet.py \
  --model vgg_small_binary \
  --save vgg_small_binary_cifar10 \
  --dataset cifar10 \
  --optimizer Adam \
   --lr 1e-4 \
   --momentum 0 \
   --weight-decay 0 \
  --gpus 0
```

<details> <summary>🔁 Results and Reproducible Commands for Binary Networks with Adam</summary>

---

**CIFAR-10 on VGG_Small** 
```bash
python main_binary_binarynet.py --model vgg_small_binary --save vgg_small_binary_cifar10_Adam --dataset cifar10 --optimizer Adam --lr 1e-4 --momentum 0 --weight-decay 0  --epochs 200 -b 256 --gpus 2
```

**CIFAR-10 on VGG16** 
```bash
python main_binary_binarynet.py --model vgg16_binary --save vgg16_binary_cifar10_Adam --dataset cifar10 --optimizer Adam --lr 1e-4 --momentum 0 --weight-decay 0  --epochs 200 -b 256 --gpus 0
```

**CIFAR-10 on ResNet20** -
```bash
python main_binary_binarynet.py --model resnet20_binary --save resnet20_binary_cifar10_Adam --dataset cifar10 --optimizer Adam --lr 1e-4 --momentum 0 --weight-decay 0  --epochs 200 -b 256 --gpus 1
```

**CIFAR-10 on ResNet56** 
```bash
python main_binary_binarynet.py --model resnet56_binary --save resnet56_binary_cifar10_Adam --dataset cifar10 --optimizer Adam --lr 1e-4 --momentum 0 --weight-decay 0  --epochs 200 -b 256 --gpus 1
```

**CIFAR-10 on ResNet18** 
```bash
python main_binary_binarynet.py --model resnet18_binary --save resnet18_binary_cifar10_Adam --dataset cifar10 --optimizer Adam --lr 1e-4 --momentum 0 --weight-decay 0  --epochs 200 -b 256 --gpus 1
```

**CIFAR-100 on VGG_Small** 
```bash
python main_binary_binarynet.py --model vgg_small_binary --save vgg_small_binary_cifar100_Adam --dataset cifar100 --optimizer Adam --lr 1e-4 --momentum 0 --weight-decay 0  --epochs 200 -b 256 --gpus 2
```

**CIFAR-100 on VGG16** 
```bash
python main_binary_binarynet.py --model vgg16_binary --save vgg16_binary_cifar100_Adam --dataset cifar100 --optimizer Adam --lr 1e-4 --momentum 0 --weight-decay 0  --epochs 200 -b 256 --gpus 0
```

**CIFAR-100 on ResNet20** -
```bash
python main_binary_binarynet.py --model resnet20_binary --save resnet20_binary_cifar100_Adam --dataset cifar100 --optimizer Adam --lr 1e-4 --momentum 0 --weight-decay 0  --epochs 200 -b 256 --gpus 1
```

**CIFAR-100 on ResNet56** -
```bash
python main_binary_binarynet.py --model resnet56_binary --save resnet56_binary_cifar100_Adam --dataset cifar100 --optimizer Adam --lr 1e-4 --momentum 0 --weight-decay 0  --epochs 200 -b 256 --gpus 2
```

**CIFAR-100 on ResNet18** -
```bash
python main_binary_binarynet.py --model resnet18_binary --save resnet18_binary_cifar100_Adam --dataset cifar100 --optimizer Adam --lr 1e-4 --momentum 0 --weight-decay 0  --epochs 200 -b 256 --gpus 2
```

**CIFAR-10 on VGG_Small with infl_ratio=3** 
```bash
python main_binary_binarynet.py --model vgg_small_binary --save vgg_small_binary_infl_cifar10_Adam --dataset cifar10 --optimizer Adam --lr 1e-4 --momentum 0 --weight-decay 0  --epochs 200 -b 256 --gpus 0
```

**CIFAR-10 on VGG16_infl with infl_ratio=3** 
```bash
python main_binary_binarynet.py --model vgg16_binary --save vgg16_binary_infl_cifar10_Adam --dataset cifar10 --optimizer Adam --lr 1e-4 --momentum 0 --weight-decay 0  --epochs 200 -b 256 --gpus 1
```

**CIFAR-100 on VGG_Small with infl_ratio=3** 
```bash
python main_binary_binarynet.py --model vgg_small_binary --save vgg_small_binary_infl_cifar100_Adam --dataset cifar100 --optimizer Adam --lr 1e-4 --momentum 0 --weight-decay 0  --epochs 200 -b 256 --gpus 2
```

**CIFAR-100 on VGG16_infl with infl_ratio=3** 
```bash
python main_binary_binarynet.py --model vgg16_binary --save vgg16_binary_infl_cifar100_Adam --dataset cifar100 --optimizer Adam --lr 1e-4 --momentum 0 --weight-decay 0  --epochs 200 -b 256 --gpus 1
```

**CIFAR-10 on ResNet20_opt** 
```bash
python main_binary_binarynet.py --model resnet_binary --save resnet_binary_cifar10_Adam --dataset cifar10 --optimizer Adam --lr 1e-4 --momentum 0 --weight-decay 0  --epochs 200 -b 256 --gpus 0
```

**CIFAR-100 on ResNet20_opt**
```bash
python main_binary_binarynet.py --model resnet_binary --save resnet_binary_cifar100_Adam --dataset cifar100 --optimizer Adam --lr 1e-4 --momentum 0 --weight-decay 0  --epochs 200 -b 256 --gpus 0
```
</details>

---

#### 📊 Results of Quantized Networks

| Model | Wbits | Abits | CIFAR-10 | CIFAR-100 |
|-----------|:---------:|:---------:|:---------:|:---------:|
| vgg_small | 1 | 1 | 87.01 | 60.78 |
|  | 8 | 8 | 90.18 | 65.96 |
|  | 4 | 4 | 89.56 | 65.72 |
|  | 2 | 2 | 46.16 | 22.30 |
| * | 2 | 2 | 88.75 | 61.82 |
| resnet18 | 1 | 1 | 83.80 | 53.58 |
|  | 8 | 8 | 88.87 | 63.17 |
|  | 4 | 4 | 88.96 | 63.70 |
|  | 2 | 2 | 86.05 | 59.65 |
| * | 2 | 2 | 85.50 | 56.53 |

> ⚠️ **Note:** Quantization experiments are conducted on **VGG-Small** (uses `infl_ratio = 1`) and **ResNet18** (Option A) as representative models from each architecture family.
>
> **Optimizer legend:**
> - \* Uses **Adam** (all 1-bit and asterisk-marked rows). Adam is the default for binary/very low-bit regimes where SGD struggles.
> - Unmarked rows use **SGD** (lr = 0.1, momentum = 0.9, weight decay = 1e-4), which performs well at higher bit-widths (4-bit and above).

#### 📋 Quick Example Command

```bash
python main_binary_binarynet.py \
  --model vgg_small_quant \
  --save vgg_small_quant_cifar10 \
  --dataset cifar10 \
  --gpus 1
```

<details> <summary>🔁 All Reproducible Commands for Quantized Networks</summary>

---

**CIFAR-10 on VGG_Small**
```bash
python main_binary_binarynet.py --model vgg_small_quant --save vgg_small_quant_cifar10_8w8a --dataset cifar10 --wbits 8 --abits 8 --epochs 200 -b 256 --gpus 0
```

**CIFAR-100 on VGG_Small** 
```bash
python main_binary_binarynet.py --model vgg_small_quant --save vgg_small_quant_cifar100_8w8a --dataset cifar100 --wbits 8 --abits 8 --epochs 200 -b 256 --gpus 2
```

**CIFAR-10 on ResNet18** 
```bash
python main_binary_binarynet.py --model resnet18_quant --save resnet18_quant_cifar10_8w8a --dataset cifar10 --wbits 8 --abits 8 --epochs 200 -b 256 --gpus 1
```

**CIFAR-100 on ResNet18** 
```bash
python main_binary_binarynet.py --model resnet18_quant --save resnet18_quant_cifar100_8w8a --dataset cifar100 --wbits 8 --abits 8 --epochs 200 -b 256 --gpus 1
```
</details>

---

<a id="exp4"></a>
### 🧬 Experiment 4: Binary & Quantized Networks on CIFAR  (DoReFaNet-style) 

**Goal:** Evaluate extreme model compression (1-bit & low-bit) using the **DoReFa-Net** quantization scheme. Unlike the generic uniform quantization implemented in [Experiment 3](#exp3), DoReFa-Net provides a mathematically rigorous and academically standardized framework. It strictly separates the quantization logic for weights (e.g., employing `tanh` for soft-clipping outliers and introducing XNOR-style scaling factors for 1-bit) and activations. 

> 💡 **Ablation & Scope Note:** 
> - **Architectures:** Since Experiment 3 already demonstrated the general trends of quantization across various depths, we omit redundant architectures (e.g., VGG16, ResNet56) in this section. We focus exclusively on highly representative compact models (**VGG-Small** and **ResNet18/20**) to highlight the algorithmic superiority of the DoReFa-Net scheme.
> - **Datasets:** This section currently focuses on the CIFAR datasets to rapidly validate the quantization algorithms. Extensive results on large-scale datasets (i.e., **ImageNet-1K**) will be updated in future releases.
> It's worth noting that VGG-Small use `nn.ReLU` while ResNet user `nn.Hardtanh`.

#### 📊 Results of DoReFa-Net Quantized Networks
| Architecture | Model Name | Wbits | Abits | CIFAR-10 | CIFAR-100 |
|-----------|:---------:|:---------:|:---------:|:---------:|:---------:|
| VGG | vgg_small | 2 | 2 | 90.03 | 64.75 |
|  |  | 4 | 4 | 90.50 | 61.56 |
|  |  | 8 | 8 | 91.32 | 61.66 |
| ResNet | resnet20 | 2 | 2 | 72.24 | 34.29 |
|  |  | 4 | 4 | 85.32 | 51.39 |
|  |  | 8 | 8 | 85.77 | 53.19 |
|  | resnet18 | 2 | 2 | 85.09 | 55.45 |
|  |  | 4 | 4 | 90.88 | 64.62 |
|  |  | 8 | 8 | 91.21 | 65.50 |
|  | resnet18_preact_quant | 2 | 2 | 90.66 | ⌛️ |
|  |  | 4 | 4 | 91.38 | ⌛️ |

> ℹ️ **Activation function note:** All results above use `nn.ReLU` by default. However, for 2w2a ResNet on CIFAR-10, switching to `nn.Hardtanh` yields significant improvements:
> - **resnet20 (2w2a):** 72.24 → **83.01** ✅
> - **resnet18 (2w2a):** 85.09 → **89.63** ✅
>
> This suggests that Hardtanh's bounded output range `[0,1]` provides a better match for low-bit activation quantization than ReLU's unbounded `[0,∞)`, especially at extremely low bit-widths. Other configurations (4w4a, 8w8a, VGG models) showed no consistent benefit from Hardtanh.


#### 📋 Quick Example Command

```bash
python main_quant_dorefa.py \
  --model vgg_small_quant \
  --save vgg_small_quant_dorefa_cifar10_2w2a \
  --dataset cifar10 \
  --wbits 2 \
  --abits 2 \
  --gpus 0
```

<details> <summary>🔁 All Reproducible Commands for DoReFa-Net Quantization</summary>

---

✂️ **Note:** 1-bit and 4-bit quantization commands are omitted for brevity. Full command sets available upon request.

**CIFAR-10 on VGG_Small**
```bash
python main_quant_dorefa.py --model vgg_small_quant --save vgg_small_quant_cifar10_2w2a --dataset cifar10 --wbits 2 --abits 2 --epochs 200 -b 256 --gpus 1
```

**CIFAR-10 on ResNet20** 
```bash
python main_quant_dorefa.py --model resnet20_quant --save resnet20_quant_cifar10_2w2a --dataset cifar10 --wbits 2 --abits 2 --epochs 200 -b 256 --gpus 2
```

**CIFAR-10 on ResNet18** 
```bash
python main_quant_dorefa.py --model resnet18_quant --save resnet18_quant_cifar10_2w2a --dataset cifar10 --wbits 2 --abits 2 --epochs 200 -b 256 --gpus 0
```

**CIFAR-10 on PreActResNet18** 
```bash
python main_quant_dorefa.py --model resnet18_preact_quant --save resnet18_preact_quant_cifar10_2w2a --dataset cifar10 --wbits 2 --abits 2 --epochs 200 -b 256 --gpus 2
```

</details>
---

<a id="exp5"></a>
### 🩺 Experiment 5: Modern QAT SOTAs for Extreme Low-Bit (LSQ, EWGS, QDrop, etc.)

**Goal:** Provide fast, reproducible implementations of modern State-of-the-Art (SOTA) Quantization-Aware Training (QAT) algorithms. While [Experiment 4](#exp4) established the classical DoReFa-Net baseline, this section benchmarks advanced algorithms designed specifically to overcome the severe gradient mismatch (STE approximation errors) and weight oscillation problems inherent in extreme low-bit regimes.

> 💡 **Ablation & Scope Note:** 
> - **Bit-width Focus (2w2a & 3w3a):** Modern QAT methods achieve near-lossless accuracy at 4-bit, making it difficult to observe algorithmic superiority. Conversely, 1-bit (BNN) often requires specialized architectural modifications. Therefore, we strictly constrain this experiment to **2-bit and 3-bit** settings, which are the true battlegrounds for evaluating modern QAT algorithms.
> - **Architecture & Dataset:** We isolate variables by standardizing the backbone to **PreActResNet18** (the optimal structure for retaining negative activations) and the dataset to **CIFAR-100**. CIFAR-100's higher complexity compared to CIFAR-10 serves as an excellent, computationally efficient proxy for evaluating quantization robustness before scaling to ImageNet.

#### ℹ️ Supported QAT Algorithms
This benchmark integrates several milestone QAT algorithms, including but not limited to:
- **DoReFa-Net (arXiv 2016):** The foundational baseline that utilizes deterministic, statistics-based scaling and introduces a `tanh` soft-clipping mechanism to mitigate the impact of weight outliers.
- **PACT (ICLR 2018):** Introduces a parameterized, learnable clipping bound ($\alpha$) for activations.
- **LSQ (ICLR 2020) & LSQ+ (CVPRW 2020):** The modern gold standards that treat the quantization step size (and zero-point) as learnable parameters optimized via backpropagation.
- **DSQ (ICCV 2019):** Differentiable Soft Quantization, which employs a series of hyperbolic tangent functions to smoothly approximate the discrete step function.
- **EWGS (CVPR 2021):** Element-Wise Gradient Scaling, an elegant method that adaptively scales the backpropagated gradients based on the forward quantization error.
- **QDrop (ICLR 2022):** Randomly drops the quantization of certain weights/activations during training to smooth the loss landscape (Flat Minima) and alleviate weight oscillation.

#### 📊 Results of SOTA QAT Methods on CIFAR-100

|Backbone | Method | 2w2a | 4w4a |
|--------|--------|:----:|:----:|
| PreActResNet18  | DoReFaNet | 70.48 | 71.43 |
|  | PACT | ❌️ | ⏳ |
|  | LSQ | ⏳ | ⏳ |
|  | LSQ+ | ⏳ | ⏳ |
|  | DSQ | ⏳ | ⏳ |
|  | EWGS | ⏳ | ⏳ |
|  | QDrop | ⏳ | ⏳ |

> ⚠️ **Optimizer Note:** Following common practices in modern QAT, these methods are trained using **SGD** (with momentum=0.9, weight_decay=1e-4) and Cosine Annealing learning rate schedules unless the specific algorithm strictly dictates otherwise (e.g., LSQ and DSQ implementations inherently employ gradient scaling for their learnable quantization parameters). 
> 
> 🚨 **Experimental Anomaly Note:** 
> Readers may notice an anomalous trend in the current CIFAR-100 results: 
> 1) **Sub-optimal Multi-bit Scaling:** Modern learnable methods (LSQ, LSQ+) underperform their 2-bit counterparts when scaled to 3-bit. 
> 2) **Inverted Hierarchy:** The classical heuristic baseline (DoReFa-Net) surprisingly outperforms modern optimization-based SOTAs (LSQ/PACT).
> 
> **Analysis:** These paradoxes are a known phenomenon when deploying highly parameterized QAT algorithms on small-scale datasets like CIFAR-100. The rigid statistical scaling of DoReFa-Net acts as an implicit regularizer, preventing overfitting. Conversely, the extra learnable parameters in LSQ and PACT ($s$, $\alpha$, $\beta$) provide excess capacity, causing the model to severely overfit the dataset noise or suffer from hyperparameter sensitivity (e.g., learning rate mismatch leading to severe weight oscillation at 3-bit).
> 
> **Status:** The current implementations serve as rapid, structurally verified baselines. To observe the true algorithmic superiority of modern QAT methods (where LSQ/PACT typically dominate), we will conduct rigorous evaluations on the large-scale **ImageNet-1K** dataset in future updates, where the abundance of complex data prevents such benign overfitting.


#### 📋 Quick Example Command

```bash
# Example: Training PreActResNet18 on CIFAR-100 with DoRaFaNet (2w2a)
python main_modern_qat.py \
  --model resnet18_preact_quant \
  --qat_method dorefa \
  --save resnet18_preact_dorefa_cifar100_2w2a \
  --dataset cifar100 \
  --wbits 2 \
  --abits 2 \
  --epochs 200 \
  -b 256 \
  --gpus 1 
```

<details> <summary>🔁 All Reproducible Commands for all supported QAT methods</summary>

---

**DoReFaNet** 

```bash
python main_modern_qat.py --model resnet18_preact_quant --qat_method dorefa --save resnet18_preact_dorefa_cifar100_2w2a --dataset cifar100 --wbits 2 --abits 2 --epochs 200 -b 256 --gpus 1
```

**PACT** 

```bash
python main_modern_qat.py --model resnet18_preact_quant --qat_method pact --save resnet18_preact_pact_cifar100_2w2a --dataset cifar100 --wbits 2 --abits 2 --epochs 200 -b 256 --gpus 1
```

**LSQ** 

```bash
python main_modern_qat.py --model resnet18_preact_quant --qat_method lsq --save resnet18_preact_lsq_cifar100_2w2a --dataset cifar100 --wbits 2 --abits 2 --epochs 200 -b 256 --gpus 2
```

**LSQ+** 

```bash
python main_modern_qat.py --model resnet18_preact_quant --qat_method lsq_plus --save resnet18_preact_lsq_plus_cifar100_2w2a --dataset cifar100 --wbits 2 --abits 2 --epochs 200 -b 256 --gpus 1
```

**DSQ** 

```bash
python main_modern_qat.py --model resnet18_preact_quant --qat_method dsq --save resnet18_preact_dsq_cifar100_2w2a --dataset cifar100 --wbits 2 --abits 2 --epochs 200 -b 256 --gpus 1
```

</details>

#### 🔨 QAT Method Configuration Reference

| Optimizer | `-qat_method` Configuration |
|-----------|---------------------------|
| DoReFaNet | `"dorefa"` |
| PACT | `"pact"` |
| LSQ | `"lsq"` |
| LSQ+ | `"lsq_plus"` |
| DAQ | `"dsq"` |


---

<a id="expN"></a>
### 🧫 Experiment N: Full-Precision Network Accuracy Comparison on ImageNet

**Goal:** Provide baseline accuracy on large-scale, high-resolution datasets (ImageNet-1K) for standard full-precision models.

#### 📊 Results

| Model | Top1 (%) | Top5 (%) |
|-------|:--------:|:---------:|
| AlexNet | ⏳ | ⏳ |
| ResNet | ⏳ | ⏳ |
| BiRealNet | ⏳ | ⏳ |

#### 📋 Quick Example Command

```bash
```

<details> <summary>🔁 All Reproducible Commands </summary>
</details>

---


## 📝 Citation
If you use SnowBench in your research, please cite:

```bibtex
@misc{snowbench2024,
  author = {Xue He},
  title = {SnowBench4Quant: A Unified Benchmark for Full-Precision and Binary CNNs},
  year = {2026},
  publisher = {GitHub},
  url = {https://github.com/snowloving/SnowBench}
}
```

## 🙏 Acknowledgements

This project is built upon the excellent work of the following open-source projects (listed in order of reference, not contribution):

- [SGDAT](https://github.com/gushan/SGDAT) — SGD with Adaptive Threshold for binary neural networks
- [BinaryNet](https://github.com/itayhubara/BinaryNet.pytorch) — Training deep neural networks with weights and activations constrained to +1 or -1
- [DoReFa-Net](https://github.com/zzzxxxttt/pytorch_DoReFaNet) — Training low bitwidth convolutional neural networks with low bitwidth gradients (PyTorch implementation)
- [LSQ](https://github.com/hustzxd/LSQuantization/tree/master) — Learned Step Size Quantization (PyTorch implementation)
- [DSQ](https://github.com/ricky40403/DSQ/) — Differentiable Soft Quantization (PyTorch implementation)

We are grateful to all the researchers and developers who have made their code publicly available, enabling this benchmark to exist.

## 📧 Contact
For questions or suggestions, please open an issue or contact a1311965600@gmail.com.

