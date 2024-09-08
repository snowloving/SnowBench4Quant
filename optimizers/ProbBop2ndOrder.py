import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer, required
from torch import Tensor
from typing import List, Optional
import torch.optim._functional as F
import math
import sys

class ProbBop2ndOrder(Optimizer):
    def __init__(
        self, 
        params, 
        gamma: float = 1e-4,
        threshold: float = 1e-8,
        sigma: float = 1e-2,
        alpha: float = 0.5,
        formula: int = 1,
        name="ProbBop2ndOrder", 
        **kwargs
    ):
        if threshold < 0:
            raise ValueError(
                'Invalid threshold value: {}'.format(threshold)
            )


        defaults = dict(
            gamma=gamma,
            threshold=threshold,
            sigma=sigma,
            alpha=alpha,
            formula=formula
        )

        super(ProbBop2ndOrder, self).__init__(params, defaults)

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
                # Bop2ndOrder optimizer
                if hasattr(p,'org'):
                    grad = p.grad.data

                    state = self.state[p]
                    if len(state) == 0:
                        state['step'] = 0

                    if not hasattr(p,'m'):
                        p.m = torch.zeros_like(p, memory_format=torch.preserve_format).cuda()
                    if not hasattr(p,'v'):
                        p.v = torch.zeros_like(p, memory_format=torch.preserve_format).cuda()
                    if not hasattr(p, 'flip_num'):
                        p.flip_num = []
                    if not hasattr(p, 'epochs'):
                        p.epochs = 1

                    if not hasattr(p, 'each_flip_num'):
                        p.each_flip_num = torch.zeros_like(p, memory_format=torch.preserve_format)+1

                    gamma= group['gamma']
                    alpha = group['alpha']
                    threshold = group['threshold']
                    sigma = group['sigma']
                    formula = group['formula']

                    state['step'] += 1

                    #p.m.add_(gamma * (grad - m))
                    #p.v.add_(sigma * (grad*grad - v))

                    p.m.mul_(1-gamma).add_(grad, alpha=gamma)
                    p.v.mul_(1-sigma).add_(grad*grad,alpha=sigma)


                    m_t = p.m / (p.v.sqrt() + 1e-10)
                    
                    delta = abs(p.m) / (p.v.sqrt() + 1e-10) /threshold


                    thresholds = 1 - pow(alpha, p.epochs)/2
                    bigger_thresholds = (1/thresholds)

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

                    prev_data = p.data[indices]
                    p.data[indices] *= tmp
                    p.each_flip_num[indices] += (1-tmp)/2                   
                    p.flip_num.append(torch.sum(tmp == -1).item())
                    if state['step'] == last_step:
                        p.epochs += 1

        return loss
        