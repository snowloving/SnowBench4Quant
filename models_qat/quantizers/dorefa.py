# 文件路径：models_qat/quantizers/dorefa.py
import torch
import torch.nn as nn
from torch.autograd.function import InplaceFunction

class UniformQuantize(InplaceFunction):
    @staticmethod
    def forward(ctx, input, numBits=8):
        if numBits == 32:
            return input
        elif numBits == 1:
            return torch.sign(input)
        else:
            # DoReFa 原版均匀量化核心：切分 [0, 1] 空间
            n = float(2 ** numBits - 1)
            output = torch.round(input * n) / n
            return output

    @staticmethod
    def backward(ctx, grad_output):
        # STE 直通估计器
        return grad_output, None

def uniform_quantize(input, numBits):
    return UniformQuantize.apply(input, numBits)

# ==========================================
# 权重 DoReFa 量化器
# ==========================================
class DoReFaWeightQuantizer(nn.Module):
    def __init__(self, wbits):
        super().__init__()
        self.wbits = wbits

    def forward(self, weight):
        if self.wbits == 32:
            return weight

        if self.wbits == 1:
            # XNOR-Net 风格的 1-bit 缩放因子
            E = torch.mean(torch.abs(weight)).detach()
            weight_q = uniform_quantize(weight / E, self.wbits) * E
        else:
            # 100% 纯正的 DoReFa-Net Tanh 截断法
            w_tanh = torch.tanh(weight)
            max_w = torch.max(torch.abs(w_tanh)).detach()
            # 压缩到 [0, 1]
            w_0_1 = w_tanh / (2 * max_w) + 0.5
            # 量化后再映射回 [-max_w, max_w]
            weight_q = max_w * (2 * uniform_quantize(w_0_1, self.wbits) - 1)
            
        return weight_q

# ==========================================
# 激活值 DoReFa 量化器 (完美适配 PreAct 结构)
# ==========================================
class DoReFaActQuantizer(nn.Module):
    def __init__(self, abits):
        super().__init__()
        self.abits = abits

    def forward(self, activation):
        if self.abits == 32:
            return activation
            
        act_0_1 = torch.clamp(activation, 0, 1)
        
        # 用 DoReFa 核心函数量化
        act_q_0_1 = uniform_quantize(act_0_1, self.abits)
        
        return activation_q
