import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.autograd import Function
from .binarized_modules import  QuantizeLinear,QuantizeConv2d

__all__ = ['vgg_small_quant', 'vgg16_quant']


class VGG_Small(nn.Module):
    """
    BNN (BinaryNet) 开山之作中使用的经典直筒状 VGG 变体。
    常被量化/二值化论文作为 Baseline。
    """
    def __init__(self, num_classes=10, wbits=8, abits=8):
        super(VGG_Small, self).__init__()
        self.infl_ratio=3
        self.features = nn.Sequential(
            QuantizeConv2d(3, 128*self.infl_ratio, kernel_size=3, stride=1, padding=1,
                      bias=True, wbits=wbits, abits=abits),
            nn.BatchNorm2d(128*self.infl_ratio),
            nn.Hardtanh(inplace=True),

            QuantizeConv2d(128*self.infl_ratio, 128*self.infl_ratio, kernel_size=3, padding=1, bias=True, wbits=wbits, abits=abits),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(128*self.infl_ratio),
            nn.Hardtanh(inplace=True),


            QuantizeConv2d(128*self.infl_ratio, 256*self.infl_ratio, kernel_size=3, padding=1, bias=True, wbits=wbits, abits=abits),
            nn.BatchNorm2d(256*self.infl_ratio),
            nn.Hardtanh(inplace=True),


            QuantizeConv2d(256*self.infl_ratio, 256*self.infl_ratio, kernel_size=3, padding=1, bias=True, wbits=wbits, abits=abits),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(256*self.infl_ratio),
            nn.Hardtanh(inplace=True),


            QuantizeConv2d(256*self.infl_ratio, 512*self.infl_ratio, kernel_size=3, padding=1, bias=True, wbits=wbits, abits=abits),
            nn.BatchNorm2d(512*self.infl_ratio),
            nn.Hardtanh(inplace=True),


            QuantizeConv2d(512*self.infl_ratio, 512, kernel_size=3, padding=1, bias=True, wbits=wbits, abits=abits),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(512),
            nn.Hardtanh(inplace=True)
        )

        
        # 维度计算：CIFAR 输入为 32x32
        # vgg_small 有 3 个 'M' (MaxPool)，所以尺寸变为 32 -> 16 -> 8 -> 4
        # 最终特征图大小为 512 * 4 * 4
        self.classifier = nn.Sequential(
            QuantizeLinear(512 * 4 * 4, 1024, bias=True, wbits=wbits, abits=abits),
            nn.BatchNorm1d(1024),
            nn.Hardtanh(inplace=True),
            #nn.Dropout(0.5),
            QuantizeLinear(1024, 1024, bias=True, wbits=wbits, abits=abits),
            nn.BatchNorm1d(1024),
            nn.Hardtanh(inplace=True),
            #nn.Dropout(0.5),
            QuantizeLinear(1024, num_classes, bias=True, wbits=wbits, abits=abits),
            nn.BatchNorm1d(num_classes, affine=False),
            nn.LogSoftmax()
        )


        '''
        self.regime = {
            0: {'optimizer': 'Adam', 'betas': (0.9, 0.999),'lr': 5e-3},
            40: {'lr': 1e-3},
            80: {'lr': 5e-4},
            100: {'lr': 1e-4},
            120: {'lr': 5e-5},
            140: {'lr': 1e-5}
        }
        '''


    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 512 * 4 * 4)
        x = self.classifier(x)
        return x


class VGG16(nn.Module):
    """
    CIFAR 专属的 VGG-16 架构。
    与 ImageNet 版的区别在于：去掉了最后两个庞大的 4096 全连接层，防止在小图上严重过拟合。
    """
    def __init__(self, num_classes=10, wbits=8, abits=8):
        super(VGG16, self).__init__()
        self.infl_ratio=3
        self.features = nn.Sequential(
            QuantizeConv2d(3, 64*self.infl_ratio, kernel_size=3, stride=1, padding=1,
                      bias=True, wbits=wbits, abits=abits),
            nn.BatchNorm2d(64*self.infl_ratio),
            nn.Hardtanh(inplace=True),

            QuantizeConv2d(64*self.infl_ratio, 64*self.infl_ratio, kernel_size=3, padding=1, bias=True, wbits=wbits, abits=abits),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(64*self.infl_ratio),
            nn.Hardtanh(inplace=True),


            QuantizeConv2d(64*self.infl_ratio, 128*self.infl_ratio, kernel_size=3, stride=1, padding=1,bias=True, wbits=wbits, abits=abits),
            nn.BatchNorm2d(128*self.infl_ratio),
            nn.Hardtanh(inplace=True),

            QuantizeConv2d(128*self.infl_ratio, 128*self.infl_ratio, kernel_size=3, padding=1, bias=True, wbits=wbits, abits=abits),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(128*self.infl_ratio),
            nn.Hardtanh(inplace=True),


            QuantizeConv2d(128*self.infl_ratio, 256*self.infl_ratio, kernel_size=3, padding=1, bias=True, wbits=wbits, abits=abits),
            nn.BatchNorm2d(256*self.infl_ratio),
            nn.Hardtanh(inplace=True),


            QuantizeConv2d(256*self.infl_ratio, 256*self.infl_ratio, kernel_size=3, padding=1, bias=True, wbits=wbits, abits=abits),
            nn.BatchNorm2d(256*self.infl_ratio),
            nn.Hardtanh(inplace=True),

            QuantizeConv2d(256*self.infl_ratio, 256*self.infl_ratio, kernel_size=3, padding=1, bias=True, wbits=wbits, abits=abits),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(256*self.infl_ratio),
            nn.Hardtanh(inplace=True),


            QuantizeConv2d(256*self.infl_ratio, 512*self.infl_ratio, kernel_size=3, padding=1, bias=True, wbits=wbits, abits=abits),
            nn.BatchNorm2d(512*self.infl_ratio),
            nn.Hardtanh(inplace=True),


            QuantizeConv2d(512*self.infl_ratio, 512*self.infl_ratio, kernel_size=3, padding=1, bias=True, wbits=wbits, abits=abits),
            nn.BatchNorm2d(512*self.infl_ratio),
            nn.Hardtanh(inplace=True),

            QuantizeConv2d(512*self.infl_ratio, 512*self.infl_ratio, kernel_size=3, padding=1, bias=True, wbits=wbits, abits=abits),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(512*self.infl_ratio),
            nn.Hardtanh(inplace=True),


            QuantizeConv2d(512*self.infl_ratio, 512*self.infl_ratio, kernel_size=3, padding=1, bias=True, wbits=wbits, abits=abits),
            nn.BatchNorm2d(512*self.infl_ratio),
            nn.Hardtanh(inplace=True),

            QuantizeConv2d(512*self.infl_ratio, 512*self.infl_ratio, kernel_size=3, padding=1, bias=True, wbits=wbits, abits=abits),
            nn.BatchNorm2d(512*self.infl_ratio),
            nn.Hardtanh(inplace=True),


            QuantizeConv2d(512*self.infl_ratio, 512, kernel_size=3, padding=1, bias=True, wbits=wbits, abits=abits),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(512),
            nn.Hardtanh(inplace=True),
        )
        
        # 维度计算：CIFAR 输入为 32x32
        # vgg16 有 5 个 'M' (MaxPool)，所以尺寸变为 32 -> 16 -> 8 -> 4 -> 2 -> 1
        # 最终特征图大小为 512 * 1 * 1 = 512
        self.classifier = nn.Sequential(
            QuantizeLinear(512, 512, bias=True, wbits=wbits, abits=abits),
            nn.BatchNorm1d(512),
            nn.Hardtanh(inplace=True),
            #nn.Dropout(0.5),
            QuantizeLinear(512, num_classes, bias=True, wbits=wbits, abits=abits),
            nn.BatchNorm1d(num_classes, affine=False),
            nn.LogSoftmax()
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 512)  # 展平
        x = self.classifier(x)
        return x


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


def vgg_small_quant(**kwargs):
    """ BNN 经典的 VGG-Small (BinaryNet) 架构 """
    num_classes = _get_num_classes(kwargs)
    wbits = kwargs.get('wbits', 8)
    abits = kwargs.get('abits', 8)

    return VGG_Small(num_classes=num_classes, wbits=wbits, abits=abits)


def vgg16_quant(**kwargs):
    """ CIFAR 专属的 VGG-16 架构 """
    num_classes = _get_num_classes(kwargs)
    wbits = kwargs.get('wbits', 8)
    abits = kwargs.get('abits', 8)

    return VGG16(num_classes=num_classes, wbits=wbits, abits=abits)
