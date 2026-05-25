import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torchvision.transforms as transforms
from torch.autograd import Function
from .binarized_modules import  BinarizeLinear,BinarizeConv2d

__all__ = ['resnet18_binary', 'resnet20_binary', 'resnet56_binary']

def Binaryconv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return BinarizeConv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def _weights_init(m):
    if isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


def init_model(model):
    for m in model.modules():
        if isinstance(m, BinarizeConv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()

class BasicBlock(nn.Module):
    """
    标准的 ResNet 基础残差块 (包含两个 3x3 卷积)。
    适用于 ResNet-18, 20, 56。
    """
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = Binaryconv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.tanh1 = nn.Hardtanh(inplace=True)
        
        self.conv2 = Binaryconv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.tanh2 = nn.Hardtanh(inplace=True)

        self.shortcut = nn.Sequential()
        pad = 0 if planes == self.expansion*in_planes else planes // 4
        # 当步长不为 1，或者输入输出通道数不一致时，需要用 1x1 卷积调整 shortcut 的维度
        if stride != 1 or in_planes != self.expansion*planes:
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


class ResNet_CIFAR(nn.Module):
    """
    何恺明原论文 (ResNet, 2016) 中专为 CIFAR 提出的轻量级架构。
    特点：3 个 stage，通道数分别为 16, 32, 64。
    """
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet_CIFAR, self).__init__()
        self.in_planes = 16

        # CIFAR 专用开头：3x3 卷积，没有 MaxPool
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        
        self.linear = nn.Linear(64, num_classes)

        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        
        # 使用自适应平均池化，不再硬编码特征图大小，更加灵活 (这里 CIFAR 会 pooling 8x8 的特征图)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


class ResNet_ImageNet_Modified(nn.Module):
    """
    ImageNet 缩水版架构 (ResNet-18)。
    特点：4 个 stage，通道数分别为 64, 128, 256, 512。
    """
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet_ImageNet_Modified, self).__init__()
        self.in_planes = 64

        # 针对 CIFAR 的修改：原版的 7x7 stride=2 替换为 3x3 stride=1，并且去掉了紧随其后的 MaxPool
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        
        self.linear = nn.Linear(512, num_classes)

        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
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


def resnet20_binary(**kwargs):
    """ 
    CIFAR 专用 ResNet-20 
    公式：6*n + 2 = 20 -> n = 3
    """
    num_classes = _get_num_classes(kwargs)
    return ResNet_CIFAR(BasicBlock, [3, 3, 3], num_classes=num_classes)


def resnet56_binary(**kwargs):
    """ 
    CIFAR 专用 ResNet-56 
    公式：6*n + 2 = 56 -> n = 9
    """
    num_classes = _get_num_classes(kwargs)
    return ResNet_CIFAR(BasicBlock, [9, 9, 9], num_classes=num_classes)


def resnet18_binary(**kwargs):
    """ 
    适配 CIFAR (32x32 输入) 的魔改版 ResNet-18
    拥有远超 resnet20/56 的参数量 (约 11M)
    """
    num_classes = _get_num_classes(kwargs)
    return ResNet_ImageNet_Modified(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)
