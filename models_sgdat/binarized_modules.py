import torch
import pdb
import torch.nn as nn
import math
from torch.autograd import Variable
from torch.autograd import Function

import numpy as np
from utils import binarize

              

class BinarizeLinear(nn.Linear):

    def __init__(self, *kargs, **kwargs):
        super(BinarizeLinear, self).__init__(*kargs, **kwargs)

    def forward(self, input):
        if input.size(1) != 784: # 784 维的输入（如 MNIST 原始像素）保持不变
            input.data=binarize(input.data)
        if not hasattr(self.weight,'org'):
            # self.weight.org = self.weight.data.clone() # there is something I don't know why
            self.weight.org=torch.zeros_like(self.weight)  
        if not hasattr(self.weight,'pre_binary_data'):
            self.weight.pre_binary_data = binarize(self.weight.data)

        self.weight.data=binarize(self.weight)
        out = nn.functional.linear(input, self.weight)
        if not self.bias is None:
            # self.bias.org=self.bias.data.clone()
            out += self.bias.view(1, -1).expand_as(out)

        return out

class BinarizeConv2d(nn.Conv2d):

    def __init__(self, *kargs, **kwargs):
        super(BinarizeConv2d, self).__init__(*kargs, **kwargs)


    def forward(self, input):
        if input.size(1) != 3: # 3 通道的输入（如 RGB 图像）保持不变, 其他情况（如中间层特征）则进行二值化处理
            input.data = binarize(input.data)
        if not hasattr(self.weight,'org'):
            # self.weight.org = self.weight.data.clone() # there is something I don't know why
            self.weight.org=torch.zeros_like(self.weight)    
        if not hasattr(self.weight,'pre_binary_data'):
            self.weight.pre_binary_data = binarize(self.weight.data)

        self.weight.data=binarize(self.weight)

        out = nn.functional.conv2d(input, self.weight, None, self.stride,
                                   self.padding, self.dilation, self.groups)

        if not self.bias is None:
            # self.bias.org=self.bias.data.clone()
            out += self.bias.view(1, -1, 1, 1).expand_as(out)

        return out



# It doesn't binarize activation, discard input.data=binarize(input.data)
class BinarizeLinear_1w32a(nn.Linear):

    def __init__(self, *kargs, **kwargs):
        super(BinarizeLinear_1w32a, self).__init__(*kargs, **kwargs)

    def forward(self, input):
        if not hasattr(self.weight,'org'):
            # self.weight.org = self.weight.data.clone()
            self.weight.org=torch.zeros_like(self.weight)      
        if not hasattr(self.weight,'pre_binary_data'):
            self.weight.pre_binary_data = binarize(self.weight.data)

        self.weight.data=binarize(self.weight)
        out = nn.functional.linear(input, self.weight)
        if not self.bias is None:
            # self.bias.org=self.bias.data.clone()
            out += self.bias.view(1, -1).expand_as(out)

        return out

# It doesn't binarize activation, discard input.data=binarize(input.data)
class BinarizeConv2d_1w32a(nn.Conv2d):

    def __init__(self, *kargs, **kwargs):
        super(BinarizeConv2d_1w32a, self).__init__(*kargs, **kwargs)


    def forward(self, input):
        if not hasattr(self.weight,'org'):
            # self.weight.org = self.weight.data.clone()
            self.weight.org=torch.zeros_like(self.weight)      
        if not hasattr(self.weight,'pre_binary_data'):
            self.weight.pre_binary_data = binarize(self.weight.data)

        self.weight.data=binarize(self.weight)

        out = nn.functional.conv2d(input, self.weight, None, self.stride,
                                   self.padding, self.dilation, self.groups)

        if not self.bias is None:
            # self.bias.org=self.bias.data.clone()
            out += self.bias.view(1, -1, 1, 1).expand_as(out)

        return out
