# 文件路径：models_qat/quantizers/dsq.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class RoundWithGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return torch.round(x)
    @staticmethod
    def backward(ctx, g):
        return g

def round_with_grad(x):
    return RoundWithGradient.apply(x)


def clipping(x, upper, lower):
    x = x + F.relu(lower - x)
    x = x - F.relu(x - upper)
    return x

def phi_function(x, mi, alpha, delta):
    alpha = torch.clamp(alpha, min=1e-4, max=1.999)
    s = 1.0 / (1.0 - alpha)
    k = (2.0 / alpha - 1.0).log() * (1.0 / delta)
    out = (((x - mi) * k).tanh()) * s
    return out


class DSQWeightQuantizer(nn.Module):
    def __init__(self, wbits, momentum=0.1):
        super().__init__()
        self.wbits = wbits
        self.momentum = momentum
        self.bit_range = 2 ** self.wbits - 1

        if self.wbits not in [1, 32]:
            # 初始化为 0，等待第一批数据激活
            self.uW = nn.Parameter(data=torch.zeros(1))
            self.lW = nn.Parameter(data=torch.zeros(1))
            
            self.register_buffer('running_uw', torch.zeros(1))
            self.register_buffer('running_lw', torch.zeros(1))
            self.register_buffer('initialized', torch.zeros(1))
            
            self.alphaW = nn.Parameter(data=torch.tensor(0.2).float())

    def forward(self, weight):
        if self.wbits == 32: return weight
        if self.wbits == 1: return torch.sign(weight)

        # 🚨 修复 1：第一批数据到来时，合理初始化上下界！
        if self.initialized == 0:
            std, mean = torch.std_mean(weight)
            # 用均值 +/- 3倍标准差作为安全边界
            init_max = torch.max(weight.min().abs(), (mean + 3 * std).abs()).detach()
            self.uW.data.fill_(init_max.item())
            self.lW.data.fill_(-init_max.item())
            self.running_uw.fill_(init_max.item())
            self.running_lw.fill_(-init_max.item())
            self.initialized.fill_(1)

        if self.training:
            cur_running_lw = self.running_lw.mul(1 - self.momentum).add(self.momentum * self.lW)
            cur_running_uw = self.running_uw.mul(1 - self.momentum).add(self.momentum * self.uW)
            self.running_lw.copy_(cur_running_lw.detach())
            self.running_uw.copy_(cur_running_uw.detach())
        else:
            cur_running_lw = self.running_lw
            cur_running_uw = self.running_uw

        Qweight = clipping(weight, cur_running_uw, cur_running_lw)
        
        cur_max = torch.max(Qweight)
        cur_min = torch.min(Qweight)
        delta = (cur_max - cur_min).clamp(min=1e-5) / self.bit_range
        
        # 🚨 修复 2：防止越界产生非法的第 5 个台阶！
        interval = torch.floor((Qweight - cur_min) / delta).clamp(0, self.bit_range - 1)
        mi = (interval + 0.5) * delta + cur_min

        Qweight_soft = phi_function(Qweight, mi, self.alphaW, delta)
        
        Qweight_scaled = (Qweight_soft + 1.0) / 2.0 + interval
        Qweight_int = round_with_grad(Qweight_scaled)
        
        Qweight_final = Qweight_int * delta + cur_min

        return Qweight_final


class DSQActQuantizer(nn.Module):
    def __init__(self, abits, momentum=0.1):
        super().__init__()
        self.abits = abits
        self.momentum = momentum
        self.bit_range = 2 ** self.abits - 1

        if self.abits not in [1, 32]:
            self.uA = nn.Parameter(data=torch.zeros(1))
            self.lA = nn.Parameter(data=torch.zeros(1))
            
            self.register_buffer('running_uA', torch.zeros(1))
            self.register_buffer('running_lA', torch.zeros(1))
            self.register_buffer('initialized', torch.zeros(1))
            
            self.alphaA = nn.Parameter(data=torch.tensor(0.2).float())

    def forward(self, activation):
        if self.abits == 32: return activation
        if self.abits == 1: return torch.sign(activation)

        # 🚨 修复 1：合理初始化激活值的上下界
        if self.initialized == 0:
            # 激活值通常比权重动态范围大，用直接的最大最小值即可，也可截断离群值
            init_max = activation.abs().max().detach().clamp(min=1e-3)
            self.uA.data.fill_(init_max.item())
            self.lA.data.fill_(-init_max.item()) # 针对 PreAct 的对称网络
            self.running_uA.fill_(init_max.item())
            self.running_lA.fill_(-init_max.item())
            self.initialized.fill_(1)

        if self.training:
            cur_running_lA = self.running_lA.mul(1 - self.momentum).add(self.momentum * self.lA)
            cur_running_uA = self.running_uA.mul(1 - self.momentum).add(self.momentum * self.uA)
            self.running_lA.copy_(cur_running_lA.detach())
            self.running_uA.copy_(cur_running_uA.detach())
        else:
            cur_running_lA = self.running_lA
            cur_running_uA = self.running_uA

        Qactivation = clipping(activation, cur_running_uA, cur_running_lA)
        
        cur_max = torch.max(Qactivation)
        cur_min = torch.min(Qactivation)
        delta = (cur_max - cur_min).clamp(min=1e-5) / self.bit_range
        
        # 🚨 修复 2：严格限制区间，防止激活值溢出崩溃
        interval = torch.floor((Qactivation - cur_min) / delta).clamp(0, self.bit_range - 1)
        mi = (interval + 0.5) * delta + cur_min
        
        Qactivation_soft = phi_function(Qactivation, mi, self.alphaA, delta)
        
        Qactivation_scaled = (Qactivation_soft + 1.0) / 2.0 + interval
        Qactivation_int = round_with_grad(Qactivation_scaled)
        
        Qactivation_final = Qactivation_int * delta + cur_min

        return Qactivation_final
