# models_qat/resnet_preact.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from .q_layers import QConv2d # 引入我们写好的中枢层

__all__ = ['resnet18_preact_quant', 'resnet20_preact_quant', 'resnet56_preact_quant']

# Option A（补零）
class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class PreActBasicBlockQuant(nn.Module):
    """
    标准的 Pre-Activation ResNet 基础残差块 (ResNet v2)。
    完美保住了 Shortcut 作为纯净的恒等映射。
    """
    def __init__(self, in_planes, planes, stride=1, wbits=8, abits=8, qat_method='lsq'):
        super(PreActBasicBlockQuant, self).__init__()
        
        # 核心修改 1：第一个 BN 接收的是 in_planes
        self.bn1 = nn.BatchNorm2d(in_planes)
        # 注意：不要加 Hardtanh，交给量化层的 act_quantizer 处理截断
        
        # 🚨 修复点：使用 QConv2d 而不是 QuantizeConv2d
        self.conv1 = QConv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False, 
                             wbits=wbits, abits=abits, qat_method=qat_method)
        
        self.bn2 = nn.BatchNorm2d(planes)
        # 🚨 修复点：使用 QConv2d
        self.conv2 = QConv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False, 
                             wbits=wbits, abits=abits, qat_method=qat_method)

        # 核心修改 2：Shortcut 处理
        self.use_shortcut = stride != 1 or in_planes != planes
        if self.use_shortcut:
            # 必须加 BN！并且 1x1 卷积保持全精度
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes) # 调节方差的保命神装
            )

    def forward(self, x):
        # 1. 对整个块的输入进行预激活 (只有 BN)
        out = self.bn1(x)
        
        # 2. Shortcut 需要降维时，使用预激活特征 out；否则原样传递 x
        shortcut = self.shortcut(out) if self.use_shortcut else x
        
        # 3. 走主干卷积网络 (截断和量化由 conv1 内部的 act_quantizer 自动完成)
        out = self.conv1(out)
        
        out = self.bn2(out)
        out = self.conv2(out)
        
        # 4. 纯净相加！后面绝对不加 ReLU！
        out += shortcut
        return out


class PreActResNet_CIFAR(nn.Module):
    """
    PreActResNet-20/56 for CIFAR
    """
    def __init__(self, block, num_blocks, num_classes=10, wbits=8, abits=8, qat_method='lsq'):
        super(PreActResNet_CIFAR, self).__init__()
        self.in_planes = 16

        # 核心修改 3（头）：只保留全精度 Conv
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1, wbits=wbits, abits=abits, qat_method=qat_method)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2, wbits=wbits, abits=abits, qat_method=qat_method)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2, wbits=wbits, abits=abits, qat_method=qat_method)
        
        # 核心修改 4（尾）：在进入 Pooling 之前，必须补一个全局预激活
        self.bn_final = nn.BatchNorm2d(64)
        self.tanh1 = nn.Hardtanh(inplace=True)
        self.linear = nn.Linear(64, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride, wbits, abits, qat_method):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s, wbits, abits, qat_method))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        # 开头只有 Conv
        out = self.conv1(x)
        
        # 走完所有的残差块 (每个块都是以 Conv 结尾的)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        
        # 尾部 BN + Hardtanh
        out = self.tanh1(self.bn_final(out))
        
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


class PreActResNet_ImageNet_Modified(nn.Module):
    """
    PreActResNet-18 for CIFAR
    """
    def __init__(self, block, num_blocks, num_classes=10, wbits=8, abits=8, qat_method='lsq'):
        super(PreActResNet_ImageNet_Modified, self).__init__()
        self.in_planes = 64

        # 核心修改 3（头）：只保留全精度 Conv
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1, wbits=wbits, abits=abits, qat_method=qat_method)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2, wbits=wbits, abits=abits, qat_method=qat_method)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2, wbits=wbits, abits=abits, qat_method=qat_method)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2, wbits=wbits, abits=abits, qat_method=qat_method)
        
        # 核心修改 4（尾）：末尾补全局 BN
        self.bn_final = nn.BatchNorm2d(512)
        self.tanh1 = nn.Hardtanh(inplace=True)
        self.linear = nn.Linear(512, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride, wbits, abits, qat_method):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s, wbits, abits, qat_method))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        
        # 尾部 BN + Hardtanh
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
    wbits = kwargs.get('wbits', 8)
    abits = kwargs.get('abits', 8)
    qat_method = kwargs.get('qat_method', 'dorefa')
    return PreActResNet_CIFAR(PreActBasicBlockQuant, [3, 3, 3], num_classes=num_classes, wbits=wbits, abits=abits, qat_method=qat_method)


def resnet56_preact_quant(**kwargs):
    num_classes = _get_num_classes(kwargs)
    wbits = kwargs.get('wbits', 8)
    abits = kwargs.get('abits', 8)
    qat_method = kwargs.get('qat_method', 'dorefa')
    return PreActResNet_CIFAR(PreActBasicBlockQuant, [9, 9, 9], num_classes=num_classes, wbits=wbits, abits=abits, qat_method=qat_method)


def resnet18_preact_quant(**kwargs):
    num_classes = _get_num_classes(kwargs)
    wbits = kwargs.get('wbits', 8)
    abits = kwargs.get('abits', 8)
    qat_method = kwargs.get('qat_method', 'dorefa')
    return PreActResNet_ImageNet_Modified(PreActBasicBlockQuant, [2, 2, 2, 2], num_classes=num_classes, wbits=wbits, abits=abits, qat_method=qat_method)
