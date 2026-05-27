# 文件路径：models_qat/quantizers/lsq.py
import torch
import torch.nn as nn
import math

# =======================================================
# 💡 LSQ 核心工具函数
# =======================================================

class GradScale(torch.autograd.Function):
    """
    LSQ 论文核心技巧 1：梯度缩放 (Gradient Scaling)
    为了防止层数和通道数大小影响步长 s 的更新速度，
    需要将传给 s 的梯度乘以 1 / sqrt(N * Qmax)
    """
    @staticmethod
    def forward(ctx, tensor, scale):
        ctx.scale = scale
        return tensor

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output * ctx.scale, None

def grad_scale(tensor, scale):
    return GradScale.apply(tensor, scale)


def round_pass(input):
    """
    LSQ 论文核心技巧 2：STE 直通估计器
    前向传播做 round (四舍五入)
    反向传播时，假装没有做 round，梯度直接穿透 (即局部导数为 1)
    """
    return (input.round() - input).detach() + input


# =======================================================
# 权重 LSQ 量化器
# =======================================================
class LSQWeightQuantizer(nn.Module):
    def __init__(self, wbits):
        super().__init__()
        self.wbits = wbits
        
        if self.wbits not in [1, 32]:
            # 有符号对称量化范围，例如 2-bit: [-2, 1] 或 8-bit: [-128, 127]
            self.qmin = -2 ** (self.wbits - 1)
            self.qmax = 2 ** (self.wbits - 1) - 1
            
            # 步长 s 作为一个可学习的模型参数！
            self.s = nn.Parameter(torch.ones(1))
            # 记录是否已经完成初始化的 Flag
            self.register_buffer('initialized', torch.zeros(1))

    def forward(self, weight):
        if self.wbits == 32:
            return weight
        if self.wbits == 1:
            return torch.sign(weight) # 1-bit 依然退化为符号函数

        # LSQ 原论文初始化：第一批数据到来时，根据权重的绝对值均值初始化 s
        if self.initialized == 0:
            s_init = 2 * weight.abs().mean() / math.sqrt(self.qmax)
            self.s.data.copy_(s_init)
            self.initialized.fill_(1)

        # 根据 LSQ 论文，计算步长 s 的梯度缩放比例
        g_scale = 1.0 / math.sqrt(weight.numel() * self.qmax)
        s_scaled = grad_scale(self.s, g_scale)

        # 核心量化公式：v_q = round(clamp(v / s)) * s
        w_scaled = weight / s_scaled
        w_clipped = torch.clamp(w_scaled, self.qmin, self.qmax)
        w_q = round_pass(w_clipped) * s_scaled
        
        return w_q


# =======================================================
# 激活值 LSQ 量化器
# =======================================================
class LSQActQuantizer(nn.Module):
    def __init__(self, abits):
        super().__init__()
        self.abits = abits
        
        if self.abits not in [1, 32]:
            # 💡 注意：因为我们在 PreActResNet 里去掉了 Hardtanh，
            # 传进来的 activation 是包含正负数的！所以这里必须用有符号量化范围。
            self.qmin = -2 ** (self.abits - 1)
            self.qmax = 2 ** (self.abits - 1) - 1
            
            self.s = nn.Parameter(torch.ones(1))
            self.register_buffer('initialized', torch.zeros(1))

    def forward(self, activation):
        if self.abits == 32:
            return activation
        if self.abits == 1:
            return torch.sign(activation)

        if self.initialized == 0:
            s_init = 2 * activation.abs().mean() / math.sqrt(self.qmax)
            self.s.data.copy_(s_init)
            self.initialized.fill_(1)

        g_scale = 1.0 / math.sqrt(activation.numel() * self.qmax)
        s_scaled = grad_scale(self.s, g_scale)

        # LSQ 前向传播
        a_scaled = activation / s_scaled
        a_clipped = torch.clamp(a_scaled, self.qmin, self.qmax)
        a_q = round_pass(a_clipped) * s_scaled
        
        return a_q
