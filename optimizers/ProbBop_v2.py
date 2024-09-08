import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer, required
from torch import Tensor
from typing import List, Optional
import torch.optim._functional as F
import math
import sys

class ProbBop_v2(Optimizer):
    def __init__(
        self, 
        params, 
        gamma: float = 1e-4,
        threshold: float = 1e-8,
        alpha: float = 0.5,
        formula: int = 1,
        name="ProbBop_v2", 
        **kwargs
    ):
        if threshold < 0:
            raise ValueError(
                'Invalid threshold value: {}'.format(threshold)
            )


        defaults = dict(
            gamma=gamma,
            alpha=alpha,
            threshold=threshold,
            formula=formula
        )

        super(ProbBop_v2, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, last_step=196, closure=None):
        """Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                if hasattr(p,'org'):
                    grad = p.grad.data

                    state = self.state[p]
                    if len(state) == 0:
                        state['step'] = 0

                    if not hasattr(p,'m'):
                        p.m = torch.zeros_like(p, memory_format=torch.preserve_format).cuda()
                    
                    if not hasattr(p, 'epochs'):
                        p.epochs = 1

                    if not hasattr(p, 'each_flip_num'):
                        p.each_flip_num = torch.zeros_like(p, memory_format=torch.preserve_format)+1

                    gamma= group['gamma']
                    threshold = group['threshold']
                    alpha = group['alpha']
                    formula = group['formula']


                    thresholds = 1 - pow(alpha, int(state['step']))/2

                    state['step'] += 1
                    bigger_thresholds = (1.0/thresholds)
                    
                    p.m.mul_(1-gamma).add_(grad, alpha=gamma)


                    m_t = p.m
                    delta = abs(m_t)/threshold

                    if int(formula/10)==0:
                        indices = torch.where((torch.sign(m_t) == torch.sign(p.data)) & (delta>bigger_thresholds))
                    else:
                        indices = torch.where((torch.sign(m_t) == torch.sign(p.data)) & (delta/p.each_flip_num>bigger_thresholds))

                    p.data[indices] = -p.data[indices]
                    p.each_flip_num[indices] +=1

                    if int(formula/10)==0:
                        indices = torch.where((torch.sign(m_t) == torch.sign(p.data)) & (delta>thresholds) & (delta<bigger_thresholds))
                    else:
                        indices = torch.where((torch.sign(m_t) == torch.sign(p.data)) & (delta/p.each_flip_num>thresholds) & (delta/p.each_flip_num<bigger_thresholds))

                    if formula%10==1:
                        prob = torch.exp(-delta[indices])
                    elif formula%10==2:
                        prob = torch.exp(-delta[indices]/p.each_flip_num[indices])
                    elif formula%10==3:
                        prob = torch.exp(-delta[indices]/pow(p.each_flip_num[indices],2))
                    elif formula%10==4:
                        prob = torch.exp(-delta[indices]/(p.epochs * p.each_flip_num[indices]))

                    tmp = torch.sign(2*torch.bernoulli(prob) - 1)

                    p.data[indices] *= tmp
                    p.each_flip_num[indices] += (1-tmp)/2
                   
                    if state['step'] == last_step:
                        p.epochs += 1

        return loss