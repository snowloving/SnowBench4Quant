# 文件路径：models_qat/quantizers/pact.py
import torch
import torch.nn as nn
from .dorefa import uniform_quantize, DoReFaWeightQuantizer

# =======================================================
# PACT 核心工具：可学习的截断阈值 Alpha 的前向与反向
# PACT 论文规定了 \alpha 受到损失函数的梯度约束
# =======================================================
class PACTFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        # 保存上下文用于反向传播
        ctx.save_for_backward(x, alpha)
        
        # 前向传播：将输入严格截断在 [0, alpha] 之间 (假设基于 ReLU)
        # 现代改良版 (支持带负数的 PreAct 结构)：截断在 [-alpha, alpha] 之间
        out = torch.clamp(x, -alpha.item(), alpha.item())
        return out

    @staticmethod
    def backward(ctx, grad_output):
        x, alpha = ctx.saved_tensors
        
        # 1. 对应输入 x 的梯度 (只有在 [-alpha, alpha] 内部的数据才传回梯度)
        grad_x = grad_output.clone()
        grad_x[x < -alpha] = 0
        grad_x[x > alpha] = 0
        
        # 2. 对应截断阈值 alpha 的梯度 (极其核心！)
        # PACT 原文：对于大于 alpha 的部分，梯度为 1 (使得 alpha 倾向于增大)
        # 对于带负数的对称结构：大于 alpha 或小于 -alpha 的部分都会贡献梯度
        grad_alpha = grad_output.clone()
        grad_alpha[(x >= -alpha) & (x <= alpha)] = 0
        grad_alpha[x < -alpha] = -1.0 * grad_alpha[x < -alpha] # 负数侧的修正
        
        return grad_x, grad_alpha.sum().view_as(alpha)

def pact_clip(x, alpha):
    return PACTFunction.apply(x, alpha)


# =======================================================
# 权重 PACT 量化器 (通常沿用 DoReFa 的权重处理方式)
# =======================================================
class PACTWeightQuantizer(DoReFaWeightQuantizer):
    """
    PACT 论文本身只提出了针对激活值 (Activation) 的改进。
    对于权重 (Weights)，原论文采用了类似 DoReFa-Net 的统计学均匀量化。
    因此这里我们直接继承已经写好的 DoReFaWeightQuantizer 即可。
    """
    def __init__(self, wbits):
        super().__init__(wbits)


# =======================================================
# 激活值 PACT 量化器
# =======================================================
class PACTActQuantizer(nn.Module):
    def __init__(self, abits):
        super().__init__()
        self.abits = abits
        
        if self.abits not in [1, 32]:
            # PACT 引入的可学习截断参数 alpha！
            # 初始化通常给一个常数（比如 ReLU6 的 6，或者预激活中我们通常设为 1.0）
            self.alpha = nn.Parameter(torch.tensor(1.0))

    def forward(self, activation):
        if self.abits == 32:
            return activation
        if self.abits == 1:
            return torch.sign(activation)

        # 1. 使用 PACT 算法带梯度的截断函数：限制在 [-alpha, alpha]
        act_clipped = pact_clip(activation, self.alpha)
        
        # 2. 将数据平移、压缩到 [0, 1] 以备 DoReFa 核心函数切割
        # 因为我们截断在 [-alpha, alpha]，所以最大幅度就是 alpha
        act_0_1 = act_clipped / (2.0 * self.alpha) + 0.5
        
        # 3. 使用底层的 uniform_quantize 切片
        act_q_0_1 = uniform_quantize(act_0_1, self.abits)
        
        # 4. 反向映射回 [-alpha, alpha] 空间交还给卷积
        activation_q = act_q_0_1 * (2.0 * self.alpha) - self.alpha
        
        return activation_q
