# 文件路径：models_qat/quantizers/lsq_plus.py
import torch
import torch.nn as nn
import math
from .lsq import grad_scale, round_pass # 复用 LSQ 的底层工具函数

# =======================================================
# 权重 LSQ+ 量化器
# 引入可学习的步长 s 和 偏移量 beta (offset)
# =======================================================
class LSQPlusWeightQuantizer(nn.Module):
    def __init__(self, wbits):
        super().__init__()
        self.wbits = wbits
        
        if self.wbits not in [1, 32]:
            # 有符号对称量化范围
            self.qmin = -2 ** (self.wbits - 1)
            self.qmax = 2 ** (self.wbits - 1) - 1
            
            # LSQ+ 核心：双重可学习参数
            self.s = nn.Parameter(torch.ones(1))
            self.beta = nn.Parameter(torch.zeros(1)) # 初始平移量为 0
            
            self.register_buffer('initialized', torch.zeros(1))

    def forward(self, weight):
        if self.wbits == 32:
            return weight
        if self.wbits == 1:
            return torch.sign(weight)

        # ==========================================
        # LSQ+ 原版初始化策略 (极其重要)
        # ==========================================
        if self.initialized == 0:
            # 找到权重的真实最小和最大值（去除 1% 的极端离群值更稳定，这里用 3倍标准差粗略估计）
            std, mean = torch.std_mean(weight)
            min_val = torch.max(weight.min(), mean - 3 * std)
            max_val = torch.min(weight.max(), mean + 3 * std)
            
            # 初始化 s：真实跨度 / 量化刻度总数
            s_init = (max_val - min_val) / (self.qmax - self.qmin)
            # 初始化 beta：使得零点对齐
            beta_init = min_val - self.qmin * s_init
            
            self.s.data.copy_(torch.clamp(s_init, min=1e-5))
            self.beta.data.copy_(beta_init)
            self.initialized.fill_(1)

        # ==========================================
        # 前向传播与梯度缩放
        # ==========================================
        # s 和 beta 都需要梯度缩放，防止在大层中更新过快
        g_scale = 1.0 / math.sqrt(weight.numel() * self.qmax)
        s_scaled = grad_scale(self.s, g_scale)
        beta_scaled = grad_scale(self.beta, g_scale)

        # 核心公式：先减去偏移量，再除以步长
        w_scaled = (weight - beta_scaled) / s_scaled
        
        # 截断与直通取整
        w_clipped = torch.clamp(w_scaled, self.qmin, self.qmax)
        w_q_int = round_pass(w_clipped)
        
        # 反向恢复量级，再加上偏移量
        w_q = w_q_int * s_scaled + beta_scaled
        
        return w_q


# =======================================================
# 激活值 LSQ+ 量化器
# =======================================================
class LSQPlusActQuantizer(nn.Module):
    def __init__(self, abits):
        super().__init__()
        self.abits = abits
        
        if self.abits not in [1, 32]:
            # 针对 PreAct 的对称区间
            self.qmin = -2 ** (self.abits - 1)
            self.qmax = 2 ** (self.abits - 1) - 1
            
            self.s = nn.Parameter(torch.ones(1))
            self.beta = nn.Parameter(torch.zeros(1))
            
            self.register_buffer('initialized', torch.zeros(1))

    def forward(self, activation):
        if self.abits == 32:
            return activation
        if self.abits == 1:
            return torch.sign(activation)

        if self.initialized == 0:
            std, mean = torch.std_mean(activation)
            min_val = torch.max(activation.min(), mean - 3 * std)
            max_val = torch.min(activation.max(), mean + 3 * std)
            
            s_init = (max_val - min_val) / (self.qmax - self.qmin)
            beta_init = min_val - self.qmin * s_init
            
            self.s.data.copy_(torch.clamp(s_init, min=1e-5))
            self.beta.data.copy_(beta_init)
            self.initialized.fill_(1)

        g_scale = 1.0 / math.sqrt(activation.numel() * self.qmax)
        s_scaled = grad_scale(self.s, g_scale)
        beta_scaled = grad_scale(self.beta, g_scale)

        # LSQ+ 量化计算
        a_scaled = (activation - beta_scaled) / s_scaled
        a_clipped = torch.clamp(a_scaled, self.qmin, self.qmax)
        a_q_int = round_pass(a_clipped)
        a_q = a_q_int * s_scaled + beta_scaled
        
        return a_q
