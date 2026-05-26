import torch
import pdb
import torch.nn as nn
import math
from torch.autograd import Variable
from torch.autograd.function  import Function, InplaceFunction

import numpy as np



class Quantize(InplaceFunction):
    def forward(ctx,input, numBits=4,inplace=False):
        assert numBits <= 8 or numBits == 32
        ctx.inplace = inplace
        if ctx.inplace:
            ctx.mark_dirty(input)
            output = input
        else:
            output = input.clone()

        if numBits == 32:
            output = input
        elif numBits == 1:
            output = torch.sign(input)
        else:
            n = float(2 ** numBits - 1)
            output = torch.round(input * n) / n
        return output

    
    def backward(ctx, grad_output):
        #STE 
        grad_input = grad_output.clone()
        return grad_input,None,None



def weight_quantize_fn(input,numBits):
    if numBits == 32:
      weight_q = input

    elif numBits == 1:
      E = torch.mean(torch.abs(input)).detach()
      weight_q = Quantize.apply((input / E),numBits) * E

    else:
      weight = torch.tanh(input)
      max_w = torch.max(torch.abs(weight)).detach()
      weight = weight / 2 / max_w + 0.5
      weight_q = max_w * (2 * Quantize.apply(weight,numBits) - 1)

    return weight_q

def activation_quantize_fn(input,numBits):
    if numBits == 32:
      activation_q = input
    else:
      activation_q = Quantize.apply(torch.clamp(input, 0, 1),numBits)
    return activation_q



class QuantizeLinear(nn.Linear):

    def __init__(self, *args, **kwargs):
        # 1. 从 kwargs 中提取自定义参数，并提供默认值（比如默认 8 bit）
        # 使用 pop 可以将这两个键值对从 kwargs 中移除
        self.wbits = kwargs.pop('wbits', 8)
        self.abits = kwargs.pop('abits', 8)

        # 2. 将剩余的、nn.Conv2d 认识的参数传给父类
        super(QuantizeLinear, self).__init__(*args, **kwargs)


    def forward(self, input):

        if input.size(1) != 784:
            input_b=activation_quantize_fn(input, numBits=self.abits)

        weight_b=weight_quantize_fn(self.weight, numBits=self.wbits)
        out = nn.functional.linear(input_b,weight_b)
        if not self.bias is None:
            self.bias.org=self.bias.data.clone()
            out += self.bias.view(1, -1).expand_as(out)

        return out

class QuantizeConv2d(nn.Conv2d):

    def __init__(self, *args, **kwargs):
        # 1. 从 kwargs 中提取自定义参数，并提供默认值（比如默认 8 bit）
        # 使用 pop 可以将这两个键值对从 kwargs 中移除
        self.wbits = kwargs.pop('wbits', 8)
        self.abits = kwargs.pop('abits', 8)

        # 2. 将剩余的、nn.Conv2d 认识的参数传给父类
        super(QuantizeConv2d, self).__init__(*args, **kwargs)



    def forward(self, input):
        if input.size(1) != 3:
            input_b = activation_quantize_fn(input, numBits=self.abits)
        else:
            input_b=input

        weight_b=weight_quantize_fn(self.weight, numBits=self.wbits)

        out = nn.functional.conv2d(input_b, weight_b, None, self.stride,
                                   self.padding, self.dilation, self.groups)

        if not self.bias is None:
            self.bias.org=self.bias.data.clone()
            out += self.bias.view(1, -1, 1, 1).expand_as(out)

        return out
