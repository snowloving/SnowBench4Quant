import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torchvision.transforms as transforms
from torch.autograd import Function
import math
from .binarized_modules import  QuantizeLinear,QuantizeConv2d

__all__ = ['resnet18_quant', 'resnet20_quant', 'resnet56_quant']


# Option A（补零）
class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


def init_model(model):
    for m in model.modules():
        if isinstance(m, QuantizeConv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()


class BasicBlockQuant(nn.Module):
    """
    标准的 ResNet 基础残差块 (包含两个 3x3 卷积)。
    适用于 ResNet-18, 20, 56。
    """
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, wbits=8, abits=8):
        super(BasicBlockQuant, self).__init__()
        self.conv1 = QuantizeConv2d(in_planes, planes, kernel_size=3, stride=stride,
                     padding=1, bias=False, wbits=wbits, abits=abits)
        self.bn1 = nn.BatchNorm2d(planes)
        self.tanh1 = nn.Hardtanh(inplace=True)
        
        self.conv2 = QuantizeConv2d(planes, planes, kernel_size=3, stride=1,
                     padding=1, bias=False, wbits=wbits, abits=abits)
        self.bn2 = nn.BatchNorm2d(planes)
        self.tanh2 = nn.Hardtanh(inplace=True)

        self.shortcut = nn.Sequential()
        # Option A（补零）
        pad = 0 if planes == self.expansion*in_planes else planes // 4

        if stride != 1 or in_planes != self.expansion*planes:
            # Option B（1x1 卷积）
            '''
            self.shortcut = nn.Sequential(
                # 【极其核心的细节】：Shortcut 里的 1x1 降维卷积，强烈建议保持全精度 (nn.Conv2d)！
                # 学术界管这个叫 Option B。因为 1x1 卷积二值化会导致极其灾难性的信息丢失。
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )
            '''

            # Option A（补零）
            
            self.shortcut = nn.Sequential(
                        nn.AvgPool2d((2,2)), 
                        LambdaLayer(lambda x:
                        F.pad(x, (0, 0, 0, 0, pad, pad), "constant", 0)))
            


    def forward(self, x):
        out = self.tanh1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.tanh2(out)
        return out


class ResNet_CIFAR_quant(nn.Module):
    """
    何恺明原论文 (ResNet, 2016) 中专为 CIFAR 提出的轻量级架构。
    特点：3 个 stage，通道数分别为 16, 32, 64。
    """
    def __init__(self, block, num_blocks, num_classes=10, wbits=8, abits=8):
        super(ResNet_CIFAR_quant, self).__init__()
        self.in_planes = 16

        # 【潜规则 1】：第一层必须保持全精度 (nn.Conv2d)！负责高精度提取原始 RGB 边缘特征。
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.tanh1 = nn.Hardtanh(inplace=True)
        
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        
        # 【潜规则 2】：最后一层分类器必须保持全精度 (nn.Linear)！否则 Logits 会严重降维。
        self.linear = nn.Linear(64, num_classes)

        init_model(self)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.tanh1(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        
        # 使用自适应平均池化，不再硬编码特征图大小，更加灵活 (这里 CIFAR 会 pooling 8x8 的特征图)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


class ResNet_ImageNet_Modified_quant(nn.Module):
    """
    ImageNet 缩水版架构 (ResNet-18)。
    特点：4 个 stage，通道数分别为 64, 128, 256, 512。
    """
    def __init__(self, block, num_blocks, num_classes=10, wbits=8, abits=8):
        super(ResNet_ImageNet_Modified_quant, self).__init__()
        self.in_planes = 64

        # 针对 CIFAR 的修改：原版的 7x7 stride=2 替换为 3x3 stride=1，并且去掉了紧随其后的 MaxPool
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.tanh1 = nn.Hardtanh(inplace=True)
        
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        
        self.linear = nn.Linear(512, num_classes)

        init_model(self)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.tanh1(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        
        # 同样使用自适应池化 (这里 CIFAR 会 pooling 4x4 的特征图)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


# ================= 工厂函数 =================

def _get_num_classes(kwargs):
    """辅助函数：根据数据集决定分类数"""
    dataset = kwargs.get('dataset', 'cifar10')
    if dataset == 'cifar10':
        return kwargs.get('num_classes', 10)
    elif dataset == 'cifar100':
        return kwargs.get('num_classes', 100)
    else:
        raise ValueError(f"Model only supports cifar10/cifar100, got {dataset}")


def resnet20_quant(**kwargs):
    """ 
    CIFAR 专用 ResNet-20 
    公式：6*n + 2 = 20 -> n = 3
    """
    num_classes = _get_num_classes(kwargs)
    wbits = kwargs.get('wbits', 8)
    abits = kwargs.get('abits', 8)
    return ResNet_CIFAR_quant(BasicBlockQuant, [3, 3, 3], num_classes=num_classes, wbits=wbits, abits=abits)


def resnet56_quant(**kwargs):
    """ 
    CIFAR 专用 ResNet-56 
    公式：6*n + 2 = 56 -> n = 9
    """
    num_classes = _get_num_classes(kwargs)
    wbits = kwargs.get('wbits', 8)
    abits = kwargs.get('abits', 8)
    return ResNet_CIFAR_quant(BasicBlockQuant, [9, 9, 9], num_classes=num_classes, wbits=wbits, abits=abits)


def resnet18_quant(**kwargs):
    """ 
    适配 CIFAR (32x32 输入) 的魔改版 ResNet-18
    拥有远超 resnet20/56 的参数量 (约 11M)
    """
    num_classes = _get_num_classes(kwargs)
    wbits = kwargs.get('wbits', 8)
    abits = kwargs.get('abits', 8)
    return ResNet_ImageNet_Modified_quant(BasicBlockQuant, [2, 2, 2, 2], num_classes=num_classes, wbits=wbits, abits=abits)
