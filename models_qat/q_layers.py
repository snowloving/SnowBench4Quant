# models_qat/q_layers.py
import torch.nn as nn
from .quantizers import get_weight_quantizer, get_act_quantizer

class QConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False, 
                 wbits=8, abits=8, qat_method='dorefa'):
        super(QConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        
        # 魔法在这里：动态获取对应的量化器！
        self.weight_quantizer = get_weight_quantizer(qat_method, wbits)
        self.act_quantizer = get_act_quantizer(qat_method, abits)

    def forward(self, x):
        # 1. 量化激活值 (第一层输入图片时不量化)
        if x.size(1) != 3:
            x = self.act_quantizer(x)
            
        # 2. 量化权重
        w_q = self.weight_quantizer(self.weight)
        
        # 3. 执行卷积
        out = nn.functional.conv2d(x, w_q, self.bias, self.stride, self.padding, self.dilation, self.groups)
        return out
