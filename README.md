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
- ✅ Multiple backbones: BinaryNet, ResNet18
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
├── preprocess.py               # Data augmentation & preprocessing: transforms, Lighting, ColorJitter, and dataset-specific pipelines
├── utils.py                    # Logger, metrics, checkpointing utilities, model binarization, and optimizer adjustment (SGD/Adam/Bop/Bop2ndOrder/SGDAT)
│
├── datasets/
├── results/
├── optimizers/                 # Bop/Bop2ndOrder/SGDAT
│
├── models_sgdat/               # Model definitions derived from SGDAT: BinaryNet, ResNet18
│   ├── __init__.py.py/         
│   ├── binarized_modules.py/   # Binary quantization layers: BinarizeLinear, BinarizeConv2d (weights + activations), and 1w32a variants (weights only binarized)
│   ├── binarynet.py/           # Binary versions of BinaryNet
│   └── resnet_binary.py/       # Binary versions of ResNet18
│
├── models_full/                # Full-precision models (to be added)
│   ├── __init__.py.py/         
│
├── main_binary_sgdat.py        # Training entry: binary models (corresponding to models_sgdat)
├── main_full.py                # Training entry: full-precision models (to be added)
├── main_quant.py               # Training entry: other quantization bit-widths (to be added)
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

- [🔬 Experiment 1](#exp1): Binary Network Optimizer Comparison — SGD vs Adam vs Bop vs Bop2ndOrder vs SGDAT
- [🧫 Experiment 2](#exp2): Full-Precision Network Accuracy Comparison — FP32 baselines as reference ceiling for binary & quantization benchmarks


---

<a id="exp1"></a>
### 🔬 Experiment 1: Binary Network Optimizer Comparison


**Goal:** Compare different optimizers (SGD, Adam, Bop, Bop2ndOrder, SGDAT) for training binary neural networks.  

#### 📊 Results

**BinaryNet**

| Optimizer | CIFAR-10 | CIFAR-100 | Tiny-ImageNet |
|-----------|:--------:|:---------:|:-------------:|
| SGD | 90.04 | 64.42 | ⏳ |
| SGDM | 89.88 | 65.23 | |
| Adam | 89.97 | 64.94 | |
| Bop | 89.11 | 64.61 | |
| Bop2ndOrder | 89.74 | 65.19 | |
| SGDT | 89.17 | 64.23 | |
| SGDAT | 90.04 | 65.77 | |
> 📝 **Notes:**
> - Results reported are from a **single run** with fix seed_value (not averaged over 5 trials). Multi-run results with mean ± std are on the way.
> - Entries marked with hourglass are currently being benchmarked and will be updated soon.

#### 📋 Example Command

````bash
python main_binary_sgdat.py \
  --model binarynet \
  --save binarynet_cifar10_SGD \
  --dataset cifar10 \
  --bin_regime "{0: {'optimizer': 'SGD', 'lr': 1e-4}}" \
  --fp_regime "{0: {'optimizer': 'Adam','lr':1e-3}}" \
  --binarization det \
  --input_size 32 \
  --epochs 200 \
  -b 256 \
  --gpus 0
````

<details> <summary>🔁 All Reproducible Commands on BinaryNet</summary>

---

**CIFAR-10 with SGD** 
```bash
python main_binary_sgdat.py --model binarynet --save binarynet_cifar10_SGD --dataset cifar10 --bin_regime "{0: {'optimizer': 'SGD','lr':1e-4}}" --binarization det --input_size 32 --epochs 200 -b 256 --gpus 1
```

**CIFAR-10 with SGDM** 
```bash
python main_binary_sgdat.py --model binarynet --save binarynet_cifar10_SGDM --dataset cifar10 --bin_regime "{0: {'optimizer': 'SGD','lr':1e-4,'momentum':0.9}}" --binarization det --input_size 32 --epochs 200 -b 256 --gpus 1
```

**CIFAR-10 with Adam** 
```bash
python main_binary_sgdat.py --model binarynet --save binarynet_cifar10_Adam --dataset cifar10 --bin_regime "{0: {'optimizer': 'Adam','lr':1e-3}}" --binarization det --input_size 32 --epochs 200 -b 256 --gpus 2
```

**CIFAR-10 with Bop** 
```bash
python main_binary_sgdat.py --model binarynet --save binarynet_cifar10_Bop --dataset cifar10 --bin_regime "{0: {'optimizer': 'Bop','gamma':1e-4,'threshold':1e-8}}" --binarization det --input_size 32 --epochs 200 -b 256 --gpus 3
```

**CIFAR-10 with Bop2ndOrder** 
```bash
python main_binary_sgdat.py --model binarynet --save binarynet_cifar10_Bop2ndOrder --dataset cifar10 --bin_regime "{0: {'optimizer': 'Bop2ndOrder','gamma':1e-7,'sigma':1e-3,'threshold':1e-6}}" --binarization det --input_size 32 --epochs 200 -b 256 --gpus 1
```

**CIFAR-10 with SGDT** 
```bash
python main_binary_sgdat.py --model binarynet --save binarynet_cifar10_SGDT --dataset cifar10 --bin_regime "{0: {'optimizer': 'SGD','lr':1e-4}}" --binarization threshold --threshold 1e-8 --input_size 32 --epochs 200 -b 256 --gpus 2
```

**CIFAR-10 with SGDAT** 
```bash
python main_binary_sgdat.py --model binarynet --save binarynet_cifar10_SGDAT --dataset cifar10 --bin_regime "{0: {'optimizer':'SGDAT','lr':1e-4,'threshold':1e-7}}" --binarization det --input_size 32 --epochs 200 -b 256 --gpus 3
```

**CIFAR-100 with SGD** 
```bash
python main_binary_sgdat.py --model binarynet --save binarynet_cifar100_SGD --dataset cifar100 --bin_regime "{0: {'optimizer': 'SGD','lr':1e-4}}" --binarization det --input_size 32 --epochs 200 -b 256 --gpus 1
```

**tiny-imagenet with SGD** 
```bash
python main_binary_sgdat.py --model binarynet --save binarynet_tiny_imagenet_SGD --dataset tiny_imagenet --bin_regime "{0: {'optimizer': 'SGD','lr':1e-4}}" --binarization det --input_size 64 --epochs 100 -b 256 --gpus 1
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


| Optimizer | `--bin_regime` Configuration |
|-----------|---------------------------|
| SGD | `"{0: {'optimizer': 'SGD','lr':1e-4}}"` |
| SGDM | `"{0: {'optimizer': 'SGD','momentum':0.9,'lr':1e-4}}"` |
| Adam | `"{0: {'optimizer': 'Adam','lr':1e-4}}"` |
| Bop | `"{0: {'optimizer': 'Bop','lr':1e-4}}"` |
| Bop2ndOrder | `"{0: {'optimizer': 'Bop2ndOrder','lr':1e-4}}"` |
| SGDT | `"{0: {'optimizer': 'SGDT','lr':1e-4}}"` & --binarization `threshold` |
| SGDAT | `"{0: {'optimizer': 'SGDAT','lr':1e-4}}"` |

<a id="exp2"></a>
### 🧫 Experiment 2: Full-Precision Network Accuracy Comparison


**Goal:** Establish full-precision (FP32) accuracy baselines across multiple architectures to serve as **upper-bound references** for subsequent binary and quantization experiments. The accuracy gap between these baselines and compressed models quantifies the cost of binarization and low-bit quantization.

#### 📊 Results

| Model | CIFAR-10 | CIFAR-100 |
|-------|:--------:|:---------:|
| AlexNet | ⏳ | ⏳ |
| VGG (CIFAR) | ⏳ | ⏳ |
| ResNet18 | ⏳ | ⏳ |

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
- [BinaryNet](https://github.com/itayhubara/BinaryNet) — Training deep neural networks with weights and activations constrained to +1 or -1

We are grateful to all the researchers and developers who have made their code publicly available, enabling this benchmark to exist.

## 📧 Contact
For questions or suggestions, please open an issue or contact a1311965600@gmail.com.

