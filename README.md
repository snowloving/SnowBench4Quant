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

## 📂 Repository Structure

## File Branch

**Main**

- main\_binary.py (models\_sgdat)
- main\_probs.py (models\_sgdat)
- main\_full.py (models\_full)
- main\_sdp\_wo\_gaussian\_qudrature.py (models\_sdp)
- main\_imagenet.py (models\_sdp+imagenet)


**Models_SGDAT**

*Binary*

- binarized\_modules.py

*Model*

1w1a + didn't binarize the first and the last layers

- binarynet: binarynet.py
	- vgg\_cifar10\_binary    
	- vgg\_cifar100\_binary
	- vgg\_tiny\_imagenet\_binary
- resnet_binary: resnet\_binary.py
	- resnet\_binary
- vggsmall (1w1a): vgg\_small.py
	- vgg\_small\_cifar10\_binary
	- vgg\_small\_cifar100\_binary
- vgg16 (1w32a): vgg16.py
	- vgg16\_cifar10\_binary
	- vgg16\_cifar100\_binary
- resnet\_cifar: resnet\_cifar.py
	- resnet18\_1w1a\_cifar\_binary 
	- resnet18\_1w32a\_cifar\_binary 
- resnet18\_1w32a
	- resnet18\_1w32a
	- resnet34\_1w32a
- resnet18\_1w1a
	-  resnet18\_1w1a_recu
	-  resnet18\_1w32a_recu
	-  resnet34\_1w1a_recu
	-  resnet34\_1w32a_recu
	
## Optimizer
- Adam
- SGD
- Bop
- Bop2ndOrder
- SGDAT
- ProbBop
- ProbBop2ndOrder

## Results1

Let's take cifar100 as example!!!

epoch=600 

Adam: lr=1e-3 

SGD: lr=1e-1 weight decay=0 (Resnet/BinaryNet momentum=0.4, VGG momentum=0, Resnet18 momentum=0.8)

|Number        | Model           | Optimizer         | Acc@1(e600)      |
| :---         | :---            |    :----          |         :---     |
| 1.1.1        | VGG-Small(full) | Adam              | 68.16(68.53)     | 
| 1.1.2        | VGG-Small(full) | SGD               | 72.59(73.13)     | 
| 1.2.1        | VGG16(full)     | Adam              | 67.19(69.69)     | 
| 1.2.2        | VGG16(full)     | SGD               | 67.51(69.43)     | 
| 1.3.1        | Resnet18(full)  | Adam              | 70.35(71.08)     |
| 1.3.2        | Resnet18(full)  | SGD               | 67.54(67.16)     |
| 1.4.1        | binarynet(full) | Adam              | 67.53(67.53)     |
| 1.4.2        | binarynet(full) | SGD               | 66.28(67.23)     |
| 1.5.1        | Resnet(full)    | Adam              | 62.26(64.51)     |
| 1.5.2        | Resnet(full)    | SGD               | 61.17(63.93)     |



**Run** ❤

*Resnet18(full) with Adam* 
```
python main_full.py --model resnet18_cifar_full --save resnet18_full_cifar100_Adam --dataset cifar100 --input_size 32 --epochs 200 -b 256 -j 20 --opt adam --lr 1e-3 --gpus 0
```

*Resnet18(full) with SGD lr=1e-1* 
```
python main_full.py --model resnet18_cifar_full --save resnet18_full_cifar100_SGD --dataset cifar100 --input_size 32 --epochs 200 -b 256 -j 20 --opt sgd --lr 1e-1 --wd 0 --momentum 0 --gpus 0
```

*VGG-Small(full) with Adam* 
```
python main_full.py --model vgg_small_cifar100_full --save vgg_small_full_cifar100_Adam --dataset cifar100 --input_size 32 --epochs 200 -b 256 -j 20 --opt adam --lr 1e-3 --gpus 1
```

*VGG-Small(full) with SGD lr=1e-1* 
```
python main_full.py --model vgg_small_cifar100_full --save vgg_small_full_cifar100_SGD --dataset cifar100 --input_size 32 --epochs 200 -b 256 -j 20 --opt sgd --lr 1e-1 --wd 0 --momentum 0 --gpus 1
```

*VGG16(full) with Adam* 
```
python main_full.py --model vgg16_cifar100_full --save vgg16_full_cifar100_Adam --dataset cifar100 --input_size 32 --epochs 200 -b 256 -j 20 --opt adam --lr 1e-3 --gpus 2
```

*VGG16(full) with SGD lr=1e-1* 
```
python main_full.py --model vgg16_cifar100_full --save vgg16_full_cifar100_SGD --dataset cifar100 --input_size 32 --epochs 200 -b 256 -j 20 --opt sgd --lr 1e-1 --wd 0 --momentum 0 --gpus 2
```

*binarynet(full) with Adam* 
```
python main_full.py --model vgg_cifar100_full --save vgg_full_cifar100_Adam --dataset cifar100 --input_size 32 --epochs 200 -b 256 -j 20 --opt adam --lr 1e-3 --gpus 3
```

*binarynet(full) with SGD lr=1e-1* 
```
python main_full.py --model vgg_cifar100_full --save vgg_full_cifar100_SGD --dataset cifar100 --input_size 32 --epochs 200 -b 256 -j 20 --opt sgd --lr 1e-1 --wd 0 --momentum 0 --gpus 3
```

*Resnet(full) with Adam* 
```
python main_full.py --model resnet_full --save resnet_full_cifar100_Adam --dataset cifar100 --input_size 32 --epochs 200 -b 256 -j 20 --opt adam --lr 1e-3 --gpus 0
```

*Resnet(full) with SGD lr=1e-1* 
```
python main_full.py --model resnet_full --save resnet_full_cifar100_SGD --dataset cifar100 --input_size 32 --epochs 200 -b 256 -j 20 --opt sgd --lr 1e-1 --wd 0 --momentum 0 --gpus 0
```

***Example for epoch 600**
```
python main_full.py --model resnet18_cifar_full --save resnet18_full_cifar100_Adam_e600 --dataset cifar100 --input_size 32 --epochs 600 -b 256 -j 20 --opt adam --lr 1e-3 --gpus 0
```



## Results2

Let's take cifar100 + binarynet & resnet18 as example!!!

**Setting**

- On CIFAR-10, CIFAR-100 and Tiny-ImageNet: models with both binarized weights and activations (denoted as 1W1A)
- Use Adam with a larger learning rate (1e−3) for full precision weights and set a smaller learning rate (1e-4) for binary weights optimizer SGD(T)
- The firat convd layer and the last linear layer is full precision


1. experiment1 \- cifar10:
  BinaryNet, Resnet
2. experiment2 \- cifar100:
  BinaryNet, Resnet
3. experiment3 \- tiny-imagenet:
  BinaryNet, Resnet



<!-- **Command** 
- Benchmark\_cifar100\_resnet18\_Adam.sh (2.2.1)
- Benchmark\_cifar100\_resnet18\_SGD.sh (2.2.2)
- Benchmark\_cifar100\_resnet18\_Bop.sh (2.2.3)
- Benchmark\_cifar100\_resnet18\_Bop2ndOrder.sh (2.2.4)
- Benchmark\_cifar100\_resnet18\_SGDAT.sh (2.2.5)
-->

**Note**

When the initialization is the below format, the performance of SGD optimizer is bad.

***initializtion 1***

```
self.weight.org = self.weight.data.clone()
```

|Number        | Model         | Optimizer         | Acc@1(e600)   |
| :---         | :---          |    :----          |         :---  |
| 2.1.1        | binarynet     | Adam              | 64.91(65.49)  |
| 2.1.2        | binarynet     | SGD               | 21.12(22.63)  | 
| 2.1.3        | binarynet     | Bop               | 64.31(64.53)  |
| 2.1.4        | binarynet     | Bop2ndOrder       | 65.17(65.01)  |
| 2.1.5        | binarynet     | SGDAT             | 65.59(65.65)  |
| 2.2.1        | Resnet        | Adam              | 57.97(59.27)  |
| 2.2.2        | Resnet        | SGD               | 10.94(13.78)  |
| 2.2.3        | Resnet        | Bop               | 56.74(57.91)  |
| 2.2.4        | Resnet        | Bop2ndOrder       | 58.80(59.65)  |
| 2.2.5        | Resnet        | SGDAT             | 59.00(60.18)  |
| 2.2.6        | Resnet        | ProbBop_v4        | 59.302±0.491  | 
| 2.2.7        | Resnet        | ProbBop_v4+f      | 59.354±0.259  |                
| 2.2.8        | Resnet        | ProbBop2ndOrder   | 59.484±0.211  |               
| 2.2.9        | Resnet        | ProbBop2ndOrder+f | 59.894±0.306  |

But when the initialization is the below format, the performance of Probs- optimizer is bad.

***initializtion 2***

```
self.weight.org=torch.zeros_like(self.weight)
```

There is only results with epoch=200

|Number        | Model         | Optimizer         | Acc@1         |
| :---         | :---          |    :----          |         :---  |
| 2.0.1        | binarynet     | Adam              | 64.64         |
| 2.0.2        | binarynet     | SGD               | 64.47         | 
| 2.0.3        | binarynet     | Bop               | 64.31(√)      |
| 2.0.4        | binarynet     | Bop2ndOrder       | 65.07         |
| 2.0.5        | binarynet     | SGDAT             | 65.59(√)      |
| 2.0.6        | Resnet        | Adam              | 56.94         |
| 2.0.7        | Resnet        | SGD               | 53.05         |
| 2.0.8        | Resnet        | Bop               | 56.74(√)      |
| 2.0.9        | Resnet        | Bop2ndOrder       | 60.17         |
| 2.0.10       | Resnet        | SGDAT             | 59.00(√)      |

**Run** 

*BinaryNet with Adam*
```
python main_binary.py --model vgg_cifar100_binary --save vgg_cifar100_Adam --dataset cifar100 --bin_regime "{0: {'optimizer': 'Adam','lr':1e-3}}" --binarization det --input_size 32 --epochs 200 -b 256 -j 20 --gpus 3
```

*BinaryNet with SGD*
```
python main_binary.py --model vgg_cifar100_binary --save vgg_cifar100_SGD --dataset cifar100 --bin_regime "{0: {'optimizer': 'SGD','lr':1e-4}}" --binarization det --input_size 32 --epochs 200 -b 256 --gpus 0
```

*BinaryNet with Bop*
```
python main_binary.py --model vgg_cifar100_binary --save vgg_cifar100_Bop --dataset cifar100 --bin_regime "{0: {'optimizer': 'Bop','gamma':1e-4,'threshold':1e-8}}" --binarization det --input_size 32 --epochs 200 -b 256 --gpus 1
```

*BinaryNet with Bop2ndOrder*
```
python main_binary.py --model vgg_cifar100_binary --save vgg_cifar100_Bop2ndOrder --dataset cifar100 --bin_regime "{0: {'optimizer': 'Bop2ndOrder','gamma':1e-7,'sigma':1e-3,'threshold':1e-6}}" --binarization det --input_size 32 --epochs 200 -b 256 --gpus 2
```

*BinaryNet with SGDAT*
```
python main_binary.py --model vgg_cifar100_binary --save vgg_cifar100_SGDAT --dataset cifar100 --bin_regime "{0: {'optimizer':'SGDAT','lr':1e-4,'threshold':1e-7}}" --binarization det --input_size 32 --epochs 200 -b 256 --gpus 3
```

*Resnet with Adam*
```
python main_binary.py --model resnet_binary --save resnet_cifar100_Adam --dataset cifar100 --bin_regime "{0: {'optimizer': 'Adam','lr':1e-3}}" --binarization det --input_size 32 --epochs 200 -b 256 --gpus 0
```

*Resnet with SGD*
```
python main_binary.py --model resnet_binary --save resnet_cifar100_SGD --dataset cifar100 --bin_regime "{0: {'optimizer': 'SGD','lr':1e-4}}" --binarization det --input_size 32 --epochs 200 -b 256 --gpus 2
```

*Resnet with Bop*
```
python main_binary.py --model resnet_binary --save resnet_cifar100_Bop --dataset cifar100 --bin_regime "{0: {'optimizer': 'Bop','gamma':1e-4,'threshold':1e-8}}" --binarization det --input_size 32 --epochs 200 -b 256 --gpus 3
```

*Resnet with Bop2ndOrder*
```
python main_binary.py --model resnet_binary --save resnet_cifar100_Bop2ndOrder --dataset cifar100 --bin_regime "{0: {'optimizer': 'Bop2ndOrder','gamma':1e-7,'sigma':1e-3,'threshold':1e-6}}" --binarization det --input_size 32 --epochs 200 -b 256 --gpus 0
```

*Resnet with SGDAT* 
```
python main_binary.py --model resnet_binary --save resnet_cifar100_SGDAT --dataset cifar100 --bin_regime "{0: {'optimizer':'SGDAT','lr':1e-4,'threshold':1e-7}}" --binarization det --input_size 32 --epochs 200 -b 256 --gpus 0
```

*Resnet with ProbBop_v4* ❤❤❤
```
python main_probs.py --model resnet_binary --save resnet_cifar100_ProbBop_v4_f11 --dataset cifar100 --bin_regime "{0: {'optimizer': 'ProbBop_v4','gamma':1e-4,'alpha':0.5,'threshold':1e-8, 'formula':11}}"  --binarization det --input_size 32 --epochs 200 -b 256 -j 20 --gpus 0
```

*Resnet with ProbBop_v4* ❤❤❤
```
python main_probs.py --model resnet_binary --save resnet_cifar100_ProbBop_v4_f14 --dataset cifar100 --bin_regime "{0: {'optimizer': 'ProbBop_v4','gamma':1e-4,'alpha':0.5,'threshold':1e-8, 'formula':14}}"  --binarization det --input_size 32 --epochs 200 -b 256 -j 20 --gpus 3
```

*Resnet with ProbBop2ndOrder* ❤❤❤
```
python main_probs.py --model resnet_binary --save resnet_cifar100_ProbBop2ndOrder_f11 --dataset cifar100 --bin_regime "{0: {'optimizer': 'ProbBop2ndOrder', 'gamma':1e-8,'sigma':1e-3,'threshold':1e-8,'alpha':0.6, 'formula':11}}"  --binarization det --input_size 32 --epochs 200 -b 256 -j 20 --gpus 2
```

*Resnet with ProbBop2ndOrder* ❤❤❤
```
python main_probs.py --model resnet_binary --save resnet_cifar100_ProbBop2ndOrder_f14 --dataset cifar100 --bin_regime "{0: {'optimizer': 'ProbBop2ndOrder', 'gamma':1e-8,'sigma':1e-3,'threshold':1e-8,'alpha':0.5, 'formula':14}}"  --binarization det --input_size 32 --epochs 200 -b 256 -j 20 --gpus 3
```

***Resnet with ProbBop_v4 + epoch=600***
```
python main_probs.py --model resnet_binary --save resnet_cifar100_ProbBop_v4_g1e-4_t1e-9_a09_f1_e600 --dataset cifar100 --bin_regime "{0: {'optimizer': 'ProbBop_v4','gamma':1e-3,'alpha':0.5,'threshold':1e-8, 'formula':1}}"  --binarization det --input_size 32 --epochs 600 -b 256 -j 20 --gpus 1
```

***Adam in cifar100 with binarynet + initialization2***
```
python main_binary.py --model vgg_cifar100_binary --save vgg_cifar100_Adam2 --dataset cifar100 --bin_regime "{0: {'optimizer': 'Adam','lr':1e-3}}" --binarization det --input_size 32 --epochs 200 -b 256 -j 20 --gpus 0
```

**Resnet + CIFAR10**

|Number        | Model         | Optimizer         | Acc@1         |
| :---         | :---          |    :----          |   :---        |
| 2.3.1        | Resnet        | ProbBop_v4        | 87.9±0.138    |
| 2.3.2        | Resnet        | ProbBop_v4+f      | 87.806±0.073  |
| 2.3.3        | Resnet        | ProbBop2ndOrder   | 88.37±0.167   |
| 2.3.4        | Resnet        | ProbBop2ndOrder+f | 88.374±0.113  |

***Resnet with ProbBop_v4 on CIFAR10*** ❤
```
python main_probs.py --model resnet_binary --save resnet_cifar10_ProbBop_v4_f11 --dataset cifar10 --bin_regime "{0: {'optimizer': 'ProbBop_v4','gamma':1e-4,'alpha':0.5,'threshold':1e-8, 'formula':11}}"  --binarization det --input_size 32 --epochs 200 -b 256 -j 20 --gpus 0
```

***Resnet with ProbBop_v4 + f on CIFAR10*** ❤
```
python main_probs.py --model resnet_binary --save resnet_cifar10_ProbBop_v4_f14 --dataset cifar10 --bin_regime "{0: {'optimizer': 'ProbBop_v4','gamma':1e-4,'alpha':0.5,'threshold':1e-8, 'formula':14}}"  --binarization det --input_size 32 --epochs 200 -b 256 -j 20 --gpus 0
```

***Resnet with ProbBop2ndOrder on CIFAR10*** ❤
```
python main_probs.py --model resnet_binary --save resnet_cifar10_ProbBop2ndOrder_f11 --dataset cifar10 --bin_regime "{0: {'optimizer': 'ProbBop2ndOrder', 'gamma':1e-8,'sigma':1e-3,'threshold':1e-8,'alpha':0.6,'formula':11}}"  --binarization det --input_size 32 --epochs 200 -b 256 -j 20 --gpus 2
```

***Resnet with ProbBop2ndOrder + f on CIFAR10*** ❤
```
python main_probs.py --model resnet_binary --save resnet_cifar10_ProbBop2ndOrder_f14 --dataset cifar10 --bin_regime "{0: {'optimizer': 'ProbBop2ndOrder', 'gamma':1e-8,'sigma':1e-3,'threshold':1e-8,'alpha':0.6,'formula':14}}"  --binarization det --input_size 32 --epochs 200 -b 256 -j 20 --gpus 3
```

**BinaryNet + CIFAR100**

|Number        | Model         | Optimizer         | Acc@1         |
| :---         | :---          |    :----          |   :---        |
| 2.4.1        | BinaryNet     | ProbBop_v4        | 63.97±0.166   | 65.134±0.104
| 2.4.2        | BinaryNet     | ProbBop_v4+f      | 63.846±0.200  | 65.16±0.065
| 2.4.3        | BinaryNet     | ProbBop2ndOrder   | 65.362±0.149  |
| 2.4.4        | BinaryNet     | ProbBop2ndOrder+f | 65.404±0.149  |

***BinaryNet with ProbBop_v4 on CIFAR100*** ❤ 
```
python main_probs.py --model vgg_cifar100_binary --save vgg_cifar100_ProbBop_v4_f11 --dataset cifar100 --bin_regime "{0: {'optimizer': 'ProbBop_v4','gamma':1e-4,'alpha':0.5,'threshold':1e-9, 'formula':11}}"  --binarization det --input_size 32 --epochs 200 -b 256 -j 20 --gpus 0
```

***BinaryNet with ProbBop_v4 + f on CIFAR100*** ❤
```
python main_probs.py --model vgg_cifar100_binary --save vgg_cifar100_ProbBop_v4_f14 --dataset cifar100 --bin_regime "{0: {'optimizer': 'ProbBop_v4','gamma':1e-4,'alpha':0.5,'threshold':1e-9, 'formula':14}}"  --binarization det --input_size 32 --epochs 200 -b 256 -j 20 --gpus 1
```

***BinaryNet with ProbBop2ndOrder on CIFAR100*** ❤
```
python main_probs.py --model vgg_cifar100_binary --save vgg_cifar100_ProbBop2ndOrder_f11 --dataset cifar100 --bin_regime "{0: {'optimizer': 'ProbBop2ndOrder', 'gamma':1e-8,'sigma':1e-3,'threshold':1e-8,'alpha':0.6,'formula':11}}"  --binarization det --input_size 32 --epochs 200 -b 256 -j 20 --gpus 2
```

***BinaryNet with ProbBop2ndOrder + f on CIFAR100*** ❤
```
python main_probs.py --model vgg_cifar100_binary --save vgg_cifar100_ProbBop2ndOrder_f14 --dataset cifar100 --bin_regime "{0: {'optimizer': 'ProbBop2ndOrder', 'gamma':1e-8,'sigma':1e-3,'threshold':1e-8,'alpha':0.6,'formula':14}}"  --binarization det --input_size 32 --epochs 200 -b 256 -j 20 --gpus 3
```

**BinaryNet + CIFAR10**

|Number        | Model         | Optimizer         | Acc@1         |
| :---         | :---          |    :----          |   :---        |
| 2.5.1        | BinaryNet     | ProbBop_v4        | 89.124±0.064  |
| 2.5.2        | BinaryNet     | ProbBop_v4+f      | 89.022±0.148  |
| 2.5.3        | BinaryNet     | ProbBop2ndOrder   | 89.872±0.131  |
| 2.5.4        | BinaryNet     | ProbBop2ndOrder+f | 89.92±0.143   |

***BinaryNet with ProbBop_v4 on CIFAR10*** ❤
```
python main_probs.py --model vgg_cifar10_binary --save vgg_cifar10_ProbBop_v4_f11 --dataset cifar10 --bin_regime "{0: {'optimizer': 'ProbBop_v4','gamma':1e-4,'alpha':0.5,'threshold':1e-8, 'formula':11}}"  --binarization det --input_size 32 --epochs 200 -b 256 -j 20 --gpus 0
```

***BinaryNet with ProbBop_v4 + f on CIFAR10*** ❤
```
python main_probs.py --model vgg_cifar10_binary --save vgg_cifar10_ProbBop_v4_f14 --dataset cifar10 --bin_regime "{0: {'optimizer': 'ProbBop_v4','gamma':1e-4,'alpha':0.5,'threshold':1e-8, 'formula':14}}"  --binarization det --input_size 32 --epochs 200 -b 256 -j 20 --gpus 1
```

***BinaryNet with ProbBop2ndOrder on CIFAR10*** ❤
```
python main_probs.py --model vgg_cifar10_binary --save vgg_cifar10_ProbBop2ndOrder_f11 --dataset cifar10 --bin_regime "{0: {'optimizer': 'ProbBop2ndOrder', 'gamma':1e-8,'sigma':1e-3,'threshold':1e-8,'alpha':0.6,'formula':11}}"  --binarization det --input_size 32 --epochs 200 -b 256 -j 20 --gpus 2
```

***BinaryNet with ProbBop2ndOrder + f on CIFAR10*** ❤
```
python main_probs.py --model vgg_cifar10_binary --save vgg_cifar10_ProbBop2ndOrder_f14 --dataset cifar10 --bin_regime "{0: {'optimizer': 'ProbBop2ndOrder', 'gamma':1e-8,'sigma':1e-3,'threshold':1e-8,'alpha':0.6,'formula':14}}"  --binarization det --input_size 32 --epochs 200 -b 256 -j 20 --gpus 3
```

**Resnet + Tiny-imagenet**

epoch=100

|Number        | Model         | Optimizer         | Acc@1         |
| :---         | :---          |    :----          |  :---         |
| 2.6.1        | Resnet        | ProbBop_v4        | 44.78         |
| 2.6.2        | Resnet        | ProbBop_v+f       | 44.43         |
| 2.6.3        | Resnet        | ProbBop2ndOrder   | 44.81  |
| 2.6.4        | Resnet        | ProbBop2ndOrder+f | 45.01         |

***Resnet with ProbBop_v4 on tiny-imagenet*** ❤
```
python main_probs.py --model resnet_binary --save resnet_tiny_imagenet_ProbBop_v4_f11 --dataset tiny_imagenet --bin_regime "{0: {'optimizer': 'ProbBop_v4','gamma':1e-4,'alpha':0.5,'threshold':1e-8, 'formula':11}}"  --binarization det --input_size 64 --epochs 100 -b 256 -j 20 --gpus 0
```

***Resnet with ProbBop_v4 on tiny-imagenet*** ❤
```
python main_probs.py --model resnet_binary --save resnet_tiny_imagenet_ProbBop_v4_f14 --dataset tiny_imagenet --bin_regime "{0: {'optimizer': 'ProbBop_v4','gamma':1e-4,'alpha':0.5,'threshold':1e-8, 'formula':14}}"  --binarization det --input_size 64 --epochs 100 -b 256 -j 20 --gpus 1
```

***Resnet with ProbBop2ndOrder on tiny-imagenet*** ❤
```
python main_probs.py --model resnet_binary --save resnet_tiny_imagenet_ProbBop2ndOrder_f11 --dataset tiny_imagenet --bin_regime "{0: {'optimizer': 'ProbBop2ndOrder', 'gamma':1e-8,'sigma':1e-3,'threshold':1e-8,'alpha':0.6, 'formula':11}}" --binarization det --input_size 64 --epochs 100 -b 256 -j 20 --gpus 2
```

***Resnet with ProbBop2ndOrder on tiny-imagenet*** ❤
```
python main_probs.py --model resnet_binary --save resnet_tiny_imagenet_ProbBop2ndOrder_f14 --dataset tiny_imagenet --bin_regime "{0: {'optimizer': 'ProbBop2ndOrder', 'gamma':1e-8,'sigma':1e-3,'threshold':1e-8,'alpha':0.6, 'formula':14}}" --binarization det --input_size 64 --epochs 100 -b 256 -j 20 --gpus 3
```

**BinaryNet + Tiny-imagenet**

epoch=100

|Number        | Model         | Optimizer         | Acc@1         |
| :---         | :---          |    :----          |   :---        |
| 2.7.1        | BinaryNet     | ProbBop_v4        | 45.05         |
| 2.7.2        | BinaryNet     | ProbBop_v4+f      | 45.48         |
| 2.7.3        | BinaryNet     | ProbBop2ndOrder   | 46.91         |
| 2.7.4        | BinaryNet     | ProbBop2ndOrder+f | 47.06         |

***BinaryNet with ProbBop_v4 on tiny-imagenet*** ❤
```
python main_probs.py --model vgg_tiny_imagenet_binary --save vgg_tiny_imagenet_ProbBop_v4_f11 --dataset tiny_imagenet --bin_regime "{0: {'optimizer': 'ProbBop_v4','gamma':1e-4,'alpha':0.5,'threshold':1e-8, 'formula':11}}" --binarization det --input_size 64 --epochs 100 -b 256 --gpus 0
```

***BinaryNet with ProbBop_v4 on tiny-imagenet*** ❤
```
python main_probs.py --model vgg_tiny_imagenet_binary --save vgg_tiny_imagenet_ProbBop_v4_f14 --dataset tiny_imagenet --bin_regime "{0: {'optimizer': 'ProbBop_v4','gamma':1e-4,'alpha':0.5,'threshold':1e-8, 'formula':14}}" --binarization det --input_size 64 --epochs 100 -b 256 --gpus 0
```

***BinaryNet with ProbBop2ndOrder on tiny-imagenet*** ❤
```
python main_probs.py --model vgg_tiny_imagenet_binary --save vgg_tiny_imagenet_ProbBop2ndOrder_f11 --dataset tiny_imagenet --bin_regime "{0: {'optimizer': 'ProbBop2ndOrder', 'gamma':1e-8,'sigma':1e-3,'threshold':1e-8,'alpha':0.6, 'formula':11}}" --binarization det --input_size 64 --epochs 100 -b 256 --gpus 1
```

***BinaryNet with ProbBop2ndOrder on tiny-imagenet*** ❤
```
python main_probs.py --model vgg_tiny_imagenet_binary --save vgg_tiny_imagenet_ProbBop2ndOrder_f14 --dataset tiny_imagenet --bin_regime "{0: {'optimizer': 'ProbBop2ndOrder', 'gamma':1e-8,'sigma':1e-3,'threshold':1e-8,'alpha':0.6, 'formula':14}}" --binarization det --input_size 64 --epochs 100 -b 256 --gpus 3
```

**Resnet + imagenet**

|Number        | Model         | Optimizer         | Acc@1(e600) |
| :---         | :---          |    :----          |   :---      |
| 2.8.1        | Resnet18_1w32a| ProbBop_v4        | 64.046      |
| 2.8.2        | Resnet18_1w1a | ProbBop_v4        | 44.46       |

24h+ 
sdp      63.9/68.2
baseline 61.6/67.5

***Resnet18_1w32a with ProbBop_v4 on imagenet*** ❤
```
python main_probs.py --model resnet18_1w32a --save resnet18_1w32a_imagenet --dataset imagenet --bin_regime "{0: {'optimizer': 'ProbBop2ndOrder','gamma':1e-8,'sigma':1e-3,'threshold':1e-8,'alpha':0.6}}" --binarization det --epochs 150 -b 256 -j 20 --gpus 2
```

***Resnet18_1w1a_recu with ProbBop_v4 on imagenet*** ❤
```
python main_probs.py --model resnet18_1w1a_recu --save resnet18_1w1a_recu_imagenet --dataset imagenet --bin_regime "{0: {'optimizer': 'ProbBop2ndOrder','gamma':1e-8,'sigma':1e-3,'threshold':1e-8,'alpha':0.6}}" --binarization det --epochs 150 -b 256 -j 20 --gpus 3
```


## Results3

Let's take cifar100 as example!!!

- VGG-Small(1w1a)
- VGG16(1w32a)
- Resnet18(1w1a)
- Resnet18(1w32a)

 Adam/SGD/Bop/Bop2ndOrder

 1w1a model: lr=1e-4

|Number        | Model           | Optimizer         | Acc@1        |
| :---         | :---            |    :----          |     :---     |
| 3.1.1        | VGG-Small(1w1a) | Adam              | 62.67        |
| 3.1.2        | VGG-Small(1w1a) | Adam              | 63.05        | *initialization 2*
| 3.1.3        | VGG-Small(1w1a) | SGD               | 62.85        | *initialization 2*
| 3.1.4        | VGG-Small(1w1a) | Bop               | 62.82        |
| 3.1.5        | VGG-Small(1w1a) | Bop2ndOrder       | 63.55        |
| 3.1.6        | VGG-Small(1w1a) | Bop2ndOrder       | 63.69        | *initialization 2*
| 3.1.7        | VGG-Small(1w1a) | SGDAT             | 64.02        |
| 3.2.1        | VGG16(1w32a)    | Adam              | 65.14        | 
| 3.2.2        | VGG16(1w32a)    | Adam              | 65.12        | *initialization 2*
| 3.2.3        | VGG16(1w32a)    | SGD               | 65.39        | *initialization 2*
| 3.2.4        | VGG16(1w32a)    | Bop               | 64.64        |
| 3.2.5        | VGG16(1w32a)    | Bop2ndOrder       | 66.87        |
| 3.2.6        | VGG16(1w32a)    | Bop2ndOrder       | 66.92        | *initialization 2*
| 3.2.7        | VGG16(1w32a)    | SGDAT             | 66.95        |
| 3.3.1        | Resnet18(1w1a)  | Adam              | 53.96        |
| 3.3.2        | Resnet18(1w1a)  | Adam              | 57.65        | *initialization 2*
| 3.3.3        | Resnet18(1w1a)  | SGD               | 57.01        | *initialization 2*
| 3.3.4        | Resnet18(1w1a)  | Bop               | 58.26        |
| 3.3.5        | Resnet18(1w1a)  | Bop2ndOrder       | 58.85        |
| 3.3.6        | Resnet18(1w1a)  | Bop2ndOrder       | 58.38        | *initialization 2*
| 3.3.7        | Resnet18(1w1a)  | SGDAT             | 58.85        |
| 3.4.1        | Resnet18(1w32a) | Adam              | 68.43        |
| 3.4.2        | Resnet18(1w32a) | Adam              | 68.33        | *initialization 2*
| 3.4.3        | Resnet18(1w32a) | SGD               | 68.83        | *initialization 2*
| 3.4.4        | Resnet18(1w32a) | Bop               | 66.32        |
| 3.4.5        | Resnet18(1w32a) | Bop2ndOrder       | 69.41        |
| 3.4.6        | Resnet18(1w32a) | Bop2ndOrder       | 69.61        | *initialization 2*
| 3.4.7        | Resnet18(1w32a) | SGDAT             | 69.70        |


**Run**  

*VGG-Small with Adam lr=1e-4*
```
python main_binary.py --model vgg_small_cifar100_binary --save vgg_small_cifar100_Adam --dataset cifar100 --bin_regime "{0: {'optimizer': 'Adam','lr':1e-4}}" --binarization det --input_size 32 --epochs 200 -b 256 --gpus 0
```

*VGG-Small with SGD*
```
python main_binary.py --model vgg_small_cifar100_binary --save vgg_small_cifar100_SGD --dataset cifar100 --bin_regime "{0: {'optimizer': 'SGD','lr':1e-4}}" --binarization det --input_size 32 --epochs 200 -b 256 --gpus 1
```

*VGG-Small with Bop*
```
python main_binary.py --model vgg_small_cifar100_binary --save vgg_small_cifar100_Bop --dataset cifar100 --bin_regime "{0: {'optimizer': 'Bop','gamma':1e-4,'threshold':1e-8}}" --binarization det --input_size 32 --epochs 200 -b 256 --gpus 2
```

*VGG-Small with Bop2ndOrder*
```
python main_binary.py --model vgg_small_cifar100_binary --save vgg_small_cifar100_Bop2ndOrder --dataset cifar100 --bin_regime "{0: {'optimizer': 'Bop2ndOrder','gamma':1e-7,'sigma':1e-3,'threshold':1e-6}}" --binarization det --input_size 32 --epochs 200 -b 256 --gpus 3
```

*VGG-Small with SGDAT* 
```
python main_binary.py --model vgg_small_cifar100_binary --save vgg_small_cifar100_SGDAT --dataset cifar100 --bin_regime "{0: {'optimizer':'SGDAT','lr':1e-4,'threshold':1e-7}}" --binarization det --input_size 32 --epochs 200 -b 256 --gpus 0
```


*VGG16 with Adam*
```
python main_binary.py --model vgg16_cifar100_binary --save vgg16_cifar100_Adam --dataset cifar100 --bin_regime "{0: {'optimizer': 'Adam','lr':1e-3}}" --binarization det --input_size 32 --epochs 200 -b 256 --gpus 0
```


*VGG16 with SGD*
```
python main_binary.py --model vgg16_cifar100_binary --save vgg16_cifar100_SGD --dataset cifar100 --bin_regime "{0: {'optimizer': 'SGD','lr':1e-4}}" --binarization det --input_size 32 --epochs 200 -b 256 --gpus 1
```

*VGG16 with Bop*
```
python main_binary.py --model vgg16_cifar100_binary --save vgg16_cifar100_Bop --dataset cifar100 --bin_regime "{0: {'optimizer': 'Bop','gamma':1e-4,'threshold':1e-8}}" --binarization det --input_size 32 --epochs 200 -b 256 --gpus 2
```

*VGG16 with Bop2ndOrder*
```
python main_binary.py --model vgg16_cifar100_binary --save vgg16_cifar100_Bop2ndOrder --dataset cifar100 --bin_regime "{0: {'optimizer': 'Bop2ndOrder','gamma':1e-7,'sigma':1e-3,'threshold':1e-6}}" --binarization det --input_size 32 --epochs 200 -b 256 --gpus 3
```

*VGG16 with SGDAT* 
```
python main_binary.py --model vgg16_cifar100_binary --save vgg16_cifar100_SGDAT --dataset cifar100 --bin_regime "{0: {'optimizer':'SGDAT','lr':1e-4,'threshold':1e-7}}" --binarization det --input_size 32 --epochs 200 -b 256 --gpus 0
```

*Resnet18(1w1a) with Adam lr=1e-4*
```
python main_binary.py --model resnet18_1w1a_cifar_binary --save resnet18_1w1a_cifar100_Adam --dataset cifar100 --bin_regime "{0: {'optimizer': 'Adam','lr':1e-4}}" --binarization det --input_size 32 --epochs 200 -b 256 --gpus 0
```

*Resnet18(1w1a) with SGD lr=1e-4*
```
python main_binary.py --model resnet18_1w1a_cifar_binary --save resnet18_1w1a_cifar100_SGD --dataset cifar100 --bin_regime "{0: {'optimizer': 'SGD','lr':1e-4}}" --binarization det --input_size 32 --epochs 200 -b 256 --gpus 3
```

*Resnet18(1w1a) with Bop*
```
python main_binary.py --model resnet18_1w1a_cifar_binary --save resnet18_1w1a_cifar100_Bop --dataset cifar100 --bin_regime "{0: {'optimizer': 'Bop','gamma':1e-4,'threshold':1e-8}}" --binarization det --input_size 32 --epochs 200 -b 256 --gpus 1
```

*Resnet18(1w1a) with Bop2ndOrder*
```
python main_binary.py --model resnet18_1w1a_cifar_binary --save resnet18_1w1a_cifar100_Bop2ndOrder --dataset cifar100 --bin_regime "{0: {'optimizer': 'Bop2ndOrder','gamma':1e-7,'sigma':1e-3,'threshold':1e-6}}" --binarization det --input_size 32 --epochs 200 -b 256 --gpus 2
```

*Resnet18(1w1a) with SGDAT* 
```
python main_binary.py --model resnet18_1w1a_cifar_binary --save resnet18_1w1a_cifar100_SGDAT --dataset cifar100 --bin_regime "{0: {'optimizer':'SGDAT','lr':1e-4,'threshold':1e-7}}" --binarization det --input_size 32 --epochs 200 -b 256 --gpus 0
```

*Resnet18(1w32a) with Adam*
```
python main_binary.py --model resnet18_1w32a_cifar_binary --save resnet18_1w32a_cifar100_Adam --dataset cifar100 --bin_regime "{0: {'optimizer': 'Adam','lr':1e-3}}" --binarization det --input_size 32 --epochs 200 -b 256 --gpus 2
```

*Resnet18(1w32a) with SGD*
```
python main_binary.py --model resnet18_1w32a_cifar_binary --save resnet18_1w32a_cifar100_SGD --dataset cifar100 --bin_regime "{0: {'optimizer': 'SGD','lr':1e-4}}" --binarization det --input_size 32 --epochs 200 -b 256 --gpus 3
```

*Resnet18(1w32a) with Bop*
```
python main_binary.py --model resnet18_1w32a_cifar_binary --save resnet18_1w32a_cifar100_Bop --dataset cifar100 --bin_regime "{0: {'optimizer': 'Bop','gamma':1e-4,'threshold':1e-8}}" --binarization det --input_size 32 --epochs 200 -b 256 --gpus 2
```

*Resnet18(1w32a) with Bop2ndOrder*
```
python main_binary.py --model resnet18_1w32a_cifar_binary --save resnet18_1w32a_cifar100_Bop2ndOrder --dataset cifar100 --bin_regime "{0: {'optimizer': 'Bop2ndOrder','gamma':1e-7,'sigma':1e-3,'threshold':1e-6}}" --binarization det --input_size 32 --epochs 200 -b 256 --gpus 3
```

*Resnet18(1w32a) with SGDAT* 
```
python main_binary.py --model resnet18_1w32a_cifar_binary --save resnet18_1w32a_cifar100_SGDAT --dataset cifar100 --bin_regime "{0: {'optimizer':'SGDAT','lr':1e-4,'threshold':1e-7}}" --binarization det --input_size 32 --epochs 200 -b 256 --gpus 0
```

## Notes
vgg16 in this setting didn't binarize the first and the last layers.


