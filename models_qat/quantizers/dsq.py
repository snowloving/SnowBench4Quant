# 文件路径：models_qat/quantizers/dsq.py
import torch
import torch.nn as nn
import torch.nn.functional as F

# =======================================================
# 核心工具：带梯度的直通取整 (STE Rounding)
# 修复了原版开源代码中奇怪的归一化，使用最纯正的 STE 保证梯度无损
# =======================================================
class RoundWithGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return torch.round(x)
    @staticmethod
    def backward(ctx, g):
        return g

def round_with_grad(x):
    return RoundWithGradient.apply(x)


# =======================================================
# DSQ 核心数学函数
# =======================================================
def clipping(x, upper, lower):
    # DSQ 原版截断：使用 ReLU 来保证梯度在边界处的特殊传导
    x = x + F.relu(lower - x)
    x = x - F.relu(x - upper)
    return x

def phi_function(x, mi, alpha, delta):
    # alpha 控制 tanh 的陡峭程度，必须严格小于 2.0 否则 log 会报 NaN
    alpha = torch.clamp(alpha, min=1e-4, max=1.999)
    s = 1.0 / (1.0 - alpha)
    k = (2.0 / alpha - 1.0).log() * (1.0 / delta)
    # DSQ 的平滑滑梯逼近
    out = (((x - mi) * k).tanh()) * s
    return out


# =======================================================
# 权重 DSQ 量化器 (完全还原官方可学习边界 + EMA)
# =======================================================
class DSQWeightQuantizer(nn.Module):
    def __init__(self, wbits, momentum=0.1):
        super().__init__()
        self.wbits = wbits
        self.momentum = momentum
        self.bit_range = 2 ** self.wbits - 1

        if self.wbits not in [1, 32]:
            # 初始化可学习的上下界为极大值 (2^31-1)，让网络自己去收缩它
            self.uW = nn.Parameter(data=torch.tensor(2 ** 31 - 1).float())
            self.lW = nn.Parameter(data=torch.tensor((-1) * (2 ** 32)).float())
            
            # EMA 滑动平均 Buffer，用于稳定训练
            self.register_buffer('running_uw', torch.tensor([self.uW.data]))
            self.register_buffer('running_lw', torch.tensor([self.lW.data]))
            
            # 可学习的平滑参数 alpha
            self.alphaW = nn.Parameter(data=torch.tensor(0.2).float())

    def forward(self, weight):
        if self.wbits == 32:
            return weight
        if self.wbits == 1:
            return torch.sign(weight)

        # 1. EMA 滑动平均更新边界
        if self.training:
            # cur_running_lw/uw 带有梯度，用于当前这一步的软截断 clipping (保证 lW, uW 能学到东西)
            cur_running_lw = self.running_lw.mul(1 - self.momentum).add(self.momentum * self.lW)
            cur_running_uw = self.running_uw.mul(1 - self.momentum).add(self.momentum * self.uW)
            
            # 🚨 修复Bug：往 running_buffer 里存历史记录时，必须斩断计算图 detach()！
            self.running_lw.copy_(cur_running_lw.detach())
            self.running_uw.copy_(cur_running_uw.detach())
        else:
            cur_running_lw = self.running_lw
            cur_running_uw = self.running_uw

        # 2. 软截断 (Clipping)
        Qweight = clipping(weight, cur_running_uw, cur_running_lw)
        
        # 3. 计算当前的步长 Delta 和中心点 mi
        cur_max = torch.max(Qweight)
        cur_min = torch.min(Qweight)
        # 防止全 0 导致除以 0
        delta = (cur_max - cur_min).clamp(min=1e-5) / self.bit_range
        interval = torch.floor((Qweight - cur_min) / delta)
        mi = (interval + 0.5) * delta + cur_min

        # 4. 执行 DSQ 的 Tanh 软逼近
        Qweight_soft = phi_function(Qweight, mi, self.alphaW, delta)
        
        # 5. 为了获得绝对的离散值进行真实卷积，我们需要将其映射并做 STE Round
        # 官方代码这里写得极其绕，这里做等价的简化映射：
        # 将软化的值映射到 [0, bit_range] 的整数空间
        Qweight_scaled = (Qweight_soft + 1.0) / 2.0 + interval
        Qweight_int = round_with_grad(Qweight_scaled)
        
        # 6. Dequantize 反量化回真实量级
        Qweight_final = Qweight_int * delta + cur_min

        return Qweight_final


# =======================================================
# 激活值 DSQ 量化器 (带有 EMA 稳定器)
# =======================================================
class DSQActQuantizer(nn.Module):
    def __init__(self, abits, momentum=0.1):
        super().__init__()
        self.abits = abits
        self.momentum = momentum
        self.bit_range = 2 ** self.abits - 1

        if self.abits not in [1, 32]:
            self.uA = nn.Parameter(data=torch.tensor(2 ** 31 - 1).float())
            self.lA = nn.Parameter(data=torch.tensor((-1) * (2 ** 32)).float())
            
            self.register_buffer('running_uA', torch.tensor([self.uA.data]))
            self.register_buffer('running_lA', torch.tensor([self.lA.data]))
            
            self.alphaA = nn.Parameter(data=torch.tensor(0.2).float())

    def forward(self, activation):
        if self.abits == 32:
            return activation
        if self.abits == 1:
            return torch.sign(activation)

        if self.training:
            cur_running_lA = self.running_lA.mul(1 - self.momentum).add(self.momentum * self.lA)
            cur_running_uA = self.running_uA.mul(1 - self.momentum).add(self.momentum * self.uA)
            
            # 🚨 修复Bug：斩断计算图 detach()
            self.running_lA.copy_(cur_running_lA.detach())
            self.running_uA.copy_(cur_running_uA.detach())
        else:
            cur_running_lA = self.running_lA
            cur_running_uA = self.running_uA

        Qactivation = clipping(activation, cur_running_uA, cur_running_lA)
        
        cur_max = torch.max(Qactivation)
        cur_min = torch.min(Qactivation)
        delta = (cur_max - cur_min).clamp(min=1e-5) / self.bit_range
        interval = torch.floor((Qactivation - cur_min) / delta)
        mi = (interval + 0.5) * delta + cur_min
        
        Qactivation_soft = phi_function(Qactivation, mi, self.alphaA, delta)
        
        Qactivation_scaled = (Qactivation_soft + 1.0) / 2.0 + interval
        Qactivation_int = round_with_grad(Qactivation_scaled)
        
        Qactivation_final = Qactivation_int * delta + cur_min

        return Qactivation_final
