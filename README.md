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

## 🎯 Experiments Navigator

- [🔬](#exp1) Experiment 1: Binary Network Optimizer Comparison — SGD vs Adam vs Bop vs Bop2ndOrder vs SGDAT
- 📊 Experiment 2: Full-Precision Network Accuracy Comparison — BinaryNet & ResNet18 at FP32

---

## 🧪 Experiments

<a id="exp1"></a>
### 🔬 Experiment 1: Binary Network Optimizer Comparison


#### BinaryNet on CIFAR-10 with SGD* (* full-precision parameters presents no default optimizer):

```bash
python main_binary_sgdat.py --model binarynet --save binarynet_cifar10_SGD --dataset cifar10 --bin_regime "{0: {'optimizer': 'SGD','lr':1e-4}}" --fp_regime "{0: {'optimizer': 'Adam','lr':1e-3}}" --binarization det --input_size 32 --epochs 200 -b 256 --gpus 1
```

### BinaryNet on CIFAR-10 with SGD
```bash
python main_binary_sgdat.py --model binarynet --save binarynet_cifar10_SGD --dataset cifar10 --bin_regime "{0: {'optimizer': 'SGD','lr':1e-4}}" --binarization det --input_size 32 --epochs 200 -b 256 --gpus 1
```

### BinaryNet on CIFAR-10 with SGDM
```bash
python main_binary_sgdat.py --model binarynet --save binarynet_cifar10_SGD --dataset cifar10 --bin_regime "{0: {'optimizer': 'SGD','lr':1e-4,'momentum':0.9}}" --binarization det --input_size 32 --epochs 200 -b 256 --gpus 1
```

### BinaryNet on CIFAR-10 with Adam

### BinaryNet on CIFAR-10 with Bop

### BinaryNet on CIFAR-10 with Bop2ndOrder

### BinaryNet on CIFAR-10 with SGDT

### BinaryNet on CIFAR-10 with SGDAT

### BinaryNet on CIFAR-100 with SGD

### BinaryNet on tiny-imagenet with SGD

### Resnet-18 on CIFAR-10 with SGDAT

### 加个图标吧 Experiment 2: Full-Precision Network Accuracy Comparison

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
  author = {Xue He},
  title = {SnowBench4Quant: A Unified Benchmark for Full-Precision and Binary CNNs},
  year = {2026},
  publisher = {GitHub},
  url = {https://github.com/snowloving/SnowBench}
}
```


## 🙏 Acknowledgements
The code is based on [SGDAT](https://github.com/gushan/SGDAT/blob/main/README.md).

## 📧 Contact
For questions or suggestions, please open an issue or contact a1311965600@gmai.com.

