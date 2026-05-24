import torch.nn as nn
import torch.nn.functional as F

__all__ = ['vgg_small', 'vgg16']

# VGG 系列架构配置字典
# 数字代表通道数，'M' 代表 MaxPool 层
cfg = {
    'vgg_small': [128, 128, 'M', 256, 256, 'M', 512, 512, 'M'],
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
}

def make_layers(cfg_list):
    """根据配置列表生成特征提取层 (Conv + BN + ReLU)"""
    layers = []
    in_channels = 3
    for v in cfg_list:
        if v == 'M':
            # CIFAR 图小，MaxPool 使用标准的 2x2 缩小一倍
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            # 采用 3x3 卷积，padding=1 保证卷积时不改变特征图大小
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1, bias=False)
            layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


class VGG_Small(nn.Module):
    """
    BNN (BinaryNet) 开山之作中使用的经典直筒状 VGG 变体。
    常被量化/二值化论文作为 Baseline。
    """
    def __init__(self, num_classes=10):
        super(VGG_Small, self).__init__()
        self.features = make_layers(cfg['vgg_small'])
        
        # 维度计算：CIFAR 输入为 32x32
        # vgg_small 有 3 个 'M' (MaxPool)，所以尺寸变为 32 -> 16 -> 8 -> 4
        # 最终特征图大小为 512 * 4 * 4
        self.classifier = nn.Sequential(
            nn.Linear(512 * 4 * 4, 1024, bias=False),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, 1024, bias=False),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # 展平
        x = self.classifier(x)
        return x


class VGG16(nn.Module):
    """
    CIFAR 专属的 VGG-16 架构。
    与 ImageNet 版的区别在于：去掉了最后两个庞大的 4096 全连接层，防止在小图上严重过拟合。
    """
    def __init__(self, num_classes=10):
        super(VGG16, self).__init__()
        self.features = make_layers(cfg['vgg16'])
        
        # 维度计算：CIFAR 输入为 32x32
        # vgg16 有 5 个 'M' (MaxPool)，所以尺寸变为 32 -> 16 -> 8 -> 4 -> 2 -> 1
        # 最终特征图大小为 512 * 1 * 1 = 512
        self.classifier = nn.Sequential(
            nn.Linear(512, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # 展平
        x = self.classifier(x)
        return x


# ================= 工厂函数 =================

def vgg_small(**kwargs):
    """ BNN 经典的 VGG-Small (BinaryNet) 架构 """
    dataset = kwargs.get('dataset', 'cifar10')
    if dataset == 'cifar10':
        # input_size = kwargs.get('input_size', 32) # 保留接口，但CIFAR强制按32算
        num_classes = kwargs.get('num_classes', 10)
        return VGG_Small(num_classes=num_classes)
        
    elif dataset == 'cifar100':
        num_classes = kwargs.get('num_classes', 100)
        return VGG_Small(num_classes=num_classes)
    else:
        raise ValueError(f"vgg_small only supports cifar10/cifar100, got {dataset}")


def vgg16(**kwargs):
    """ CIFAR 专属的 VGG-16 架构 """
    dataset = kwargs.get('dataset', 'cifar10')
    if dataset == 'cifar10':
        num_classes = kwargs.get('num_classes', 10)
        return VGG16(num_classes=num_classes)

    elif dataset == 'cifar100':
        num_classes = kwargs.get('num_classes', 100)
        return VGG16(num_classes=num_classes)
    else:
        raise ValueError(f"vgg16 only supports cifar10/cifar100, got {dataset}")
