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
SnowBench/
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
│   ├── __init__.py.py/         # __all__ = ['vgg_small', 'vgg16', 'resnet18', 'resnet20', 'resnet56']
│   ├── vgg.py/                 # Full-precision versions of VGG-family
│   ├── vgg_opt.py/             # Same architecture as vgg.py, but with optimized code style (cleaner implementation)
│   └── resnet.py/              # Full-precision versions of ResNet-family
├
├── models_binarynet/           # Full-precision & Binary & Quantized (BinaryNet-style)
│   ├── __init__.py.py/         # __all__ = ['vgg_small_binary', 'vgg16_binary', 'resnet18_binary', 'resnet20_binary', 'resnet56_binary']
│   ├── binarized_modules.py/   # Binarize / quantize layers & functions
│   ├── vgg.py/                 # Binary VGG
│   └── resnet.py/              # Binary ResNet
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
├── main_full_imagenet.py       # Entry: full-precision on imagenet (models_full_imagenet)
├── main_quant.py               # Entry: quantization (to be added)
└
```

---

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/yourusername/SnowBench4Quant.git
cd SnowBench4Quant
pip install -r requirements.txt
```

---


## 🖥️ Experiments

This benchmark supports two main experimental tracks:

### 🎯 Experiments Navigator

- [🔬 Experiment 1](#exp1): Binary Network Optimizer Comparison (SGDAT-style) — SGD vs Adam vs Bop vs Bop2ndOrder vs SGDAT
- [⚗️ Experiment 2](#exp2): Full-Precision Network Accuracy Comparison on CIFAR — FP32 baselines as reference ceiling
- [🔭 Experiment 3](#exp3): Binary & Quantized Network Accuracy Comparison on CIFAR (BynaryNet-style) — 1-bit & Multi-bit compression
- [🧫 Experiment 4](#exp4): Full-Precision Network Accuracy Comparison on ImageNet — Large-scale datasets ceiling

---

<a id="exp1"></a>
### 🔬 Experiment 1: Binary Network Optimizer Comparison

> **Goal:** Compare different optimizers (SGD, Adam, Bop, Bop2ndOrder, SGDAT) for training binary neural networks.  
> ℹ️ **Note:** The "BinaryNet" used throughout this experiment refers to a compact VGG-style architecture (a.k.a. **VGG-Small**), implemented as `vgg_small` in `models_full_cifar/` and the "ResNet" used throughout this experiment refers to a ResNet18 architecture modified for ImageNet, implemented as `resnet18` in `models_full_cifar/`.

#### 📊 Results (Take BinaryNet as example)

**BinaryNet**

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

#### 📋 Quick Example Command

````bash
python main_binary_sgdat.py \
  --model binarynet \
  --save binarynet_cifar10_SGD \
  --dataset cifar10 \
  --bin_regime "{0: {'optimizer': 'SGD', 'lr': 1e-4}}" \
  --binarization det \
  --gpus 0
````

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

---

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

<details> <summary>⌛️ The results on ResNet18 will coming soon~</summary>
</details>
---

<a id="exp2"></a>
### ⚗️ Experiment 2: Full-Precision Network Accuracy Comparison on CIFAR

**Goal:** Establish the FP32 upper-bound accuracy for standard CIFAR architectures. These results serve as the "ceiling" references when calculating the accuracy drop of binary and quantized models.

#### 📊 Results (Take SGD as example)

**SGD**

| Architecture | Model Name | CIFAR-10 | CIFAR-100 | Structural Notes |
|-----------|:---------:|:---------:|:---------:|---------|
| VGG | vgg_small | 91.37 | 68.75 | BinaryNet baseline (7-layer) |
|  | vgg16 | 90.62 | 64.96 | CIFAR modified (No 4096-FCs) |
| ResNet | resnet20 | 89.15 | 61.75 | He et al. (16-32-64 channels)  |
|  | resnet56 | 90.06 | 63.54 | He et al. (16-32-64 channels) |
|  | resnet18 | 92.08 | 68.42 | ImageNet modified (3x3 conv1) |

#### 📋 Quick Example Command

````bash
python main_full_cifar.py \
  --model vgg_small \
  --save full_vgg_small_cifar10 \
  --dataset cifar10 \
  --gpus 0
````

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
### 🔭 Experiment 3: Binary & Quantized Networks on CIFAR

**Goal:** Evaluate extreme model compression techniques (1-bit BNNs & low-bit QNNs) using the standardized CIFAR architectures defined in Experiment 2.

#### 📊 Results

| Architecture | Model Name | Precision (W/A) | CIFAR-10 | CIFAR-100 |
|-----------|:---------:|:---------:|:---------:|:---------:|
| VGG | vgg_small | 1/32 | ⌛️ | ⌛️ |
|  | vgg16 | 1/32 | ⌛️ | ⌛️ |
| ResNet | resnet20 | 1/32 | ⌛️ | ⌛️ |
|  | resnet56 | 1/32 | ⌛️ | ⌛️ |
|  | resnet18 | 1/32 | ⌛️ | ⌛️ |

#### 📋 Quick Example Command

````bash
python main_binary_binarynet.py \
  --model vgg_small_binary \
  --save vgg_small_binary_cifar10 \
  --dataset cifar10 \
  --gpus 0
````

<details> <summary>🔁 All Reproducible Commands </summary>
</details>

---

<a id="exp4"></a>
### 🧫 Experiment 4: Full-Precision Network Accuracy Comparison on ImageNet

**Goal:** Provide baseline accuracy on large-scale, high-resolution datasets (ImageNet-1K) for standard full-precision models.

#### 📊 Results

| Model | Top1 (%) | Top5 (%) |
|-------|:--------:|:---------:|
| AlexNet | ⏳ | ⏳ |
| ResNet | ⏳ | ⏳ |
| BiRealNet | ⏳ | ⏳ |

#### 📋 Quick Example Command

````bash
````

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
- [VISPA](https://github.com/snownus/bnn_vi) - BNN training and variational inference

We are grateful to all the researchers and developers who have made their code publicly available, enabling this benchmark to exist.

## 📧 Contact
For questions or suggestions, please open an issue or contact a1311965600@gmail.com.

