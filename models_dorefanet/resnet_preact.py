import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torchvision.transforms as transforms
from torch.autograd import Function
import math
from .binarized_modules import QuantizeLinear,QuantizeConv2d


__all__ = ['resnet18_preact_quant', 'resnet20_preact_quant', 'resnet56_preact_quant']

class PreActBasicBlockQuant(nn.Module):
    """
    标准的 Pre-Activation ResNet 基础残差块 (ResNet v2)。
    完美保住了 Shortcut 作为纯净的恒等映射。
    """
    def __init__(self, in_planes, planes, stride=1):
        super(PreActBasicBlockQuant, self).__init__()
        
        # 核心修改 1：第一个 BN 接收的是 in_planes
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.tanh1 = nn.Hardtanh(inplace=True)
        self.conv1 = QuantizeConv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.tanh1 = nn.Hardtanh(inplace=True)
        
        self.bn2 = nn.BatchNorm2d(planes)
        self.tanh2 = nn.Hardtanh(inplace=True)
        self.conv2 = QuantizeConv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)

        # 核心修改 2：Shortcut 处理
        self.use_shortcut = stride != 1 or in_planes != planes
        if self.use_shortcut:
            # 【细节 A：如果是 1x1 卷积 (Option B)】
            # 必须加 BN！并且 1x1 卷积保持全精度
            # DoreFaNet这里也用了二值，我们就不用了
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes) # 调节方差的保命神装
            )
            
            # 【细节 B：如果是 AvgPool 补零 (Option A)】
            # Option A 通常不需要 BN，因为它没有线性变换，不会剧烈改变方差
            '''
            pad = planes // 4
            self.shortcut = nn.Sequential(
                nn.AvgPool2d((2, 2)), 
                LambdaLayer(lambda x: F.pad(x, (0, 0, 0, 0, pad, pad), "constant", 0))
            )
            '''

    def forward(self, x):
        # 1. 对整个块的输入进行预激活
        out = self.tanh1(self.bn1(x))
        
        # 2. 如果 Shortcut 需要降维，使用【预激活后】的特征 out；否则原样传递 x
        shortcut = self.shortcut(out) if self.use_shortcut else x
        
        # 3. 走主干卷积网络
        out = self.conv1(out)
        out = self.conv2(self.tanh2(self.bn2(out)))
        
        # 4. 纯净相加！后面绝对不加 ReLU！
        out += shortcut
        return out


class PreActResNet_CIFAR(nn.Module):
    """
    PreActResNet-20/56 for CIFAR
    """
    def __init__(self, block, num_blocks, num_classes=10):
        super(PreActResNet_CIFAR, self).__init__()
        self.in_planes = 16

        # 核心修改 3（头）：只保留 Conv，去掉原本这里的 BN 和 ReLU
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        
        # 核心修改 4（尾）：在进入 Pooling 之前，必须补一个全局预激活
        self.bn_final = nn.BatchNorm2d(64)
        self.tanh1 = nn.Hardtanh(inplace=True)
        self.linear = nn.Linear(64, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        # 开头只有 Conv
        out = self.conv1(x)
        
        # 走完所有的残差块 (每个块都是以 Conv 结尾的)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        
        # 必须在这里做最后一次 BN + ReLU
        # 在量化/二值化网络中不加ReLU或者换成Hardtanh
        out = self.tanh1(self.bn_final(out))
        
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


class PreActResNet_ImageNet_Modified(nn.Module):
    """
    PreActResNet-18 for CIFAR
    """
    def __init__(self, block, num_blocks, num_classes=10):
        super(PreActResNet_ImageNet_Modified, self).__init__()
        self.in_planes = 64

        # 核心修改 3（头）：只保留 Conv
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        
        # 核心修改 4（尾）：末尾补全局 BN
        self.bn_final = nn.BatchNorm2d(512)
        self.tanh1 = nn.Hardtanh(inplace=True)
        self.linear = nn.Linear(512, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        
        # 必须在这里做最后一次 BN + ReLU
        out = self.tanh1(self.bn_final(out))
        
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


# ================= 工厂函数 =================

def _get_num_classes(kwargs):
    dataset = kwargs.get('dataset', 'cifar10')
    if dataset == 'cifar10':
        return kwargs.get('num_classes', 10)
    elif dataset == 'cifar100':
        return kwargs.get('num_classes', 100)
    else:
        raise ValueError(f"Model only supports cifar10/cifar100, got {dataset}")


def resnet20_preact_quant(**kwargs):
    num_classes = _get_num_classes(kwargs)
    return PreActResNet_CIFAR(PreActBasicBlockQuant, [3, 3, 3], num_classes=num_classes)


def resnet56_preact_quant(**kwargs):
    num_classes = _get_num_classes(kwargs)
    return PreActResNet_CIFAR(PreActBasicBlockQuant, [9, 9, 9], num_classes=num_classes)


def resnet18_preact_quant(**kwargs):
    num_classes = _get_num_classes(kwargs)
    return PreActResNet_ImageNet_Modified(PreActBasicBlockQuant, [2, 2, 2, 2], num_classes=num_classes)
