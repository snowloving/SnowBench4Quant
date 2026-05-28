# 文件路径：models_qat/quantizers/dsq.py
import torch
import torch.nn as nn
import math
from .lsq import grad_scale # 复用梯度缩放，防止 alpha 跑飞

# =======================================================
# DSQ 核心前向逻辑 (软量化逼近)
# =======================================================
def dsq_function(x, alpha, qmin, qmax):
    """
    DSQ 的平滑逼近核心：
    x: 已经被缩放到量化刻度区间的浮点数 (如 1.2, 2.7)
    alpha: 控制 tanh 陡峭程度的可学习参数
    """
    # 找到每个数值距离最近的两个合法台阶
    floor_val = torch.floor(x)
    ceil_val = torch.ceil(x)
    
    # 限制在量化边界内
    floor_val = torch.clamp(floor_val, qmin, qmax)
    ceil_val = torch.clamp(ceil_val, qmin, qmax)
    
    # 防止刚好在整数点时 floor == ceil 导致除以 0
    # 由于浮点误差，通常使用微小量兜底
    interval = (ceil_val - floor_val).clamp(min=1e-5)
    
    # 将 x 映射到它所在的区间内部的相对位置 (-1 到 1 之间)
    # 例如 x=2.4，在 [2, 3] 区间内，相对位置就是 (2.4 - 2.5) * 2 = -0.2
    center = (ceil_val + floor_val) / 2.0
    x_centered = (x - center) / (interval / 2.0)
    
    # 核心魔法：使用 tanh 进行软截断逼近！
    # alpha 越大，tanh 越陡峭，越接近阶梯函数
    soft_x = torch.tanh(alpha * x_centered)
    
    # 重新映射回真实的台阶高度
    out = center + soft_x * (interval / 2.0)
    
    # 最后兜底截断，保证不越出整体量化范围
    out = torch.clamp(out, qmin, qmax)
    
    return out


# =======================================================
# 权重 DSQ 量化器
# =======================================================
class DSQWeightQuantizer(nn.Module):
    def __init__(self, wbits):
        super().__init__()
        self.wbits = wbits
        
        if self.wbits not in [1, 32]:
            self.qmin = -2 ** (self.wbits - 1)
            self.qmax = 2 ** (self.wbits - 1) - 1
            
            # DSQ 需要统计当前层的量级进行整体缩放 (类似 DoReFa 的做法)
            # 也可以学习步长，但这里使用经典的固定极大值法，将精力集中在 alpha 上
            self.scale = None 
            
            # DSQ 核心：可学习的平滑参数 alpha
            # 初始值设为 0.2，让曲线初期平缓，提供丰满的真实梯度
            self.alpha = nn.Parameter(torch.tensor(0.2))

    def forward(self, weight):
        if self.wbits == 32:
            return weight
        if self.wbits == 1:
            return torch.sign(weight)

        # 1. 动态确定比例尺 (类似 DoReFa，但更温和)
        # 用 3 倍标准差作为界限，防止极端离群值干扰
        std, mean = torch.std_mean(weight)
        max_val = torch.max(weight.min().abs(), (mean + 3 * std).abs()).detach()
        max_val = torch.clamp(max_val, min=1e-5)
        
        s = max_val / self.qmax

        # 2. 将权重缩放到整数刻度附近
        w_scaled = weight / s
        
        # 3. 对 alpha 稍微进行梯度缩放，防止震荡
        g_scale = 1.0 / math.sqrt(weight.numel())
        alpha_scaled = grad_scale(self.alpha, g_scale)
        
        # 为了保证 tanh 方向正确，alpha 必须恒为正
        alpha_pos = torch.abs(alpha_scaled) + 1e-4

        # 4. 执行 DSQ 平滑逼近！没有 round()，处处可导！
        w_q_soft = dsq_function(w_scaled, alpha_pos, self.qmin, self.qmax)
        
        # 5. 反缩放回真实量级
        w_q = w_q_soft * s
        
        return w_q


# =======================================================
# 激活值 DSQ 量化器
# =======================================================
class DSQActQuantizer(nn.Module):
    def __init__(self, abits):
        super().__init__()
        self.abits = abits
        
        if self.abits not in [1, 32]:
            self.qmin = -2 ** (self.abits - 1)
            self.qmax = 2 ** (self.abits - 1) - 1
            
            self.alpha = nn.Parameter(torch.tensor(0.2))
            
            # 对于激活值，我们通常采用类似 EMA (指数移动平均) 来稳定 scale
            # 但这里为了简洁且避免状态冲突，采用批内统计
            
    def forward(self, activation):
        if self.abits == 32:
            return activation
        if self.abits == 1:
            return torch.sign(activation)

        # 1. 寻找当前 batch 的最大量级 (针对 PreAct 的对称截断)
        max_val = activation.abs().max().detach()
        max_val = torch.clamp(max_val, min=1e-5)
        s = max_val / self.qmax

        a_scaled = activation / s

        g_scale = 1.0 / math.sqrt(activation.numel())
        alpha_scaled = grad_scale(self.alpha, g_scale)
        alpha_pos = torch.abs(alpha_scaled) + 1e-4

        # 2. 执行 DSQ 平滑量化
        a_q_soft = dsq_function(a_scaled, alpha_pos, self.qmin, self.qmax)
        
        a_q = a_q_soft * s
        
        return a_q
