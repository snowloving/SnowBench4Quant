import torch.nn as nn
import torch.nn.functional as F

__all__ = ['vgg_small', 'vgg16']


class VGG_Small(nn.Module):
    """
    BNN (BinaryNet) 开山之作中使用的经典直筒状 VGG 变体。
    常被量化/二值化论文作为 Baseline。
    """
    def __init__(self, num_classes=10):
        super(VGG_Small, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 128, kernel_size=3, stride=1, padding=1,bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=2, stride=2),


            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),


            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=2, stride=2),


            nn.Conv2d(256, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),


            nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        
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
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1,bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1,bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=2, stride=2),


            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),


            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=2, stride=2),


            nn.Conv2d(256, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),


            nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),


            nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
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

def _get_num_classes(kwargs):
    """辅助函数：根据数据集决定分类数"""
    dataset = kwargs.get('dataset', 'cifar10')
    if dataset == 'cifar10':
        return kwargs.get('num_classes', 10)
    elif dataset == 'cifar100':
        return kwargs.get('num_classes', 100)
    else:
        raise ValueError(f"Model only supports cifar10/cifar100, got {dataset}")


def vgg_small(**kwargs):
    """ BNN 经典的 VGG-Small (BinaryNet) 架构 """
    num_classes = _get_num_classes(kwargs)
    return VGG_Small(num_classes=num_classes)


def vgg16(**kwargs):
    """ CIFAR 专属的 VGG-16 架构 """
    num_classes = _get_num_classes(kwargs)
    return VGG16(num_classes=num_classes)
