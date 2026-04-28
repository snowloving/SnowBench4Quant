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
- ✅ Multiple backbones: ResNet18, VGG16, AlexNet
- ✅ Training modes: Full-precision, Binary (XNOR-style), and Quantized (DoReFa-style)
- ✅ Multiple optimizers: SGD, Adam, Bop, Bop2ndOrder, SGDAT
- ✅ Reproducible configurations for fair comparisons

---

```text
SnowBench4Quant/
├── configs/              # Configuration files for datasets/models
├── data/                 # Dataset loading & preprocessing
├── models/               # Model definitions
│   ├── full_precision/   # ResNet18, VGG16, AlexNet
│   └── binary/           # Binary versions of the above
├── optimizers/           # Custom optimizer wrappers
├── trainers/             # Training loops (full & binary)
├── utils/                # Logger, metrics, checkpointing
├── scripts/              # Run scripts for experiments
├── results/              # Output logs & checkpoints
└── config.py             # Global configuration
```

---

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/yourusername/SnowBench4Quant.git
cd SnowBench4Quant
pip install -r requirements.txt
```

## Basic Usage

### Train ResNet18 (full-precision) on CIFAR-10:

```bash
python train.py --dataset cifar10 --model resnet18 --mode full --epochs 200
```

### Train binary AlexNet on Tiny-ImageNet:

```bash
python train.py --dataset tinyimagenet --model alexnet --mode binary --epochs 100 --optimizer adamw
```

## 📊 Supported Components

| Dataset | Classes | Image Size | Training Size |
| --- | --- | --- | --- |
| CIFAR-10 | 10 | 32 × 32 | 50,000 |
| CIFAR-100 | 100 | 32 × 32 | 50,000 |
| Tiny-ImageNet | 200 | 64 × 64 | 100,000 |
| ImageNet | 1,000 | 224 × 224 | 1,200,000 |


| Model | Full-precision | Binary |
|-------|:--------------:|:------:|
| ResNet18 | ✅ | ✅ |
| VGG16 | ✅ | ✅ |
| AlexNet | ✅ | ✅ |

Optimizers
-SGD (with momentum & weight decay)
-Adam
-AdamW
-RMSprop

## 🧪 Experiments
To reproduce baseline results:

```bash
# Full-precision baselines
python train.py --config configs/experiments/full_precision.yaml

# Binary baselines  
python train.py --config configs/experiments/binary.yaml

# Optimizer comparison
python run_optimizer_sweep.py --dataset cifar10 --model resnet18
Expected results (CIFAR-10, ResNet18):
```

Expected results (CIFAR-10, ResNet18):

| Mode | Optimizer | Top-1 Acc |
|------|-----------|-----------|
| Full-precision (FP32) | SGD | ~94.5% |
| Binary (XNOR) | SGD | ~89.0% |
| Full-precision | AdamW | ~93.8% |

## 📝 Citation
If you use SnowBench in your research, please cite:

```bibtex
@misc{snowbench2024,
  author = {Your Name},
  title = {SnowBench4Quant: A Unified Benchmark for Full-Precision and Binary CNNs},
  year = {2026},
  publisher = {GitHub},
  url = {https://github.com/yourusername/SnowBench}
}
```


## 🙏 Acknowledgements
PyTorch for the deep learning framework
torchvision for datasets and pre-trained models
Binary neural network implementations inspired by XNOR-Net and DoReFa-Net

## 📧 Contact
For questions or suggestions, please open an issue or contact a1311965600@gmai.com.

