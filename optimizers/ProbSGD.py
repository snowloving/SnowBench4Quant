import math
import torch
from torch.optim.optimizer import Optimizer
from torch import Tensor, threshold
from typing import List, Optional


class ProbSGD(Optimizer):
    def __init__(
        self, 
        params, 
        lr: float = 1e-4,
        threshold: float = 1e-8,
        alpha: float = 0.5,
        formula: int = 1,
        name="ProbSGD", 
        **kwargs
    ):

        if threshold < 0:
            raise ValueError(
                'Invalid threshold value: {}'.format(threshold)
            )


        defaults = dict(
            lr=lr,
            alpha=alpha,
            threshold=threshold,
            formula=formula
        )
        super(ProbSGD, self).__init__(params, defaults)

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

                    lr = group['lr']
                    alpha = group['alpha']
                    formula = group['formula']
                    threshold = group['threshold']

                    state['step'] += 1


                    delta = abs(grad)/threshold
                    '''print(delta)
                    raise ValueError(
                        'Invalid threshold value: {}'.format(threshold)
                    )'''


                    thresholds = 1 - pow(alpha, p.epochs)/2
                    bigger_thresholds = (1/thresholds)

                    if formula/10==0:
                        indices = torch.where((torch.sign(grad) == torch.sign(p.data)) & (delta>bigger_thresholds))
                    else:
                        indices = torch.where((torch.sign(grad) == torch.sign(p.data)) & (delta/p.each_flip_num>bigger_thresholds))

                    p.data[indices] = -p.data[indices]
                    p.each_flip_num[indices] +=1

                    if formula/10==0:
                        indices = torch.where((torch.sign(grad) == torch.sign(p.data)) & (delta>thresholds) & (delta<bigger_thresholds))
                    else:
                        indices = torch.where((torch.sign(grad) == torch.sign(p.data)) & (delta/p.each_flip_num>thresholds) & (delta<1))

                    
                    '''
                    if delta[indices].numel()!=0:
                        print(delta)
                        raise ValueError(
                            'Invalid threshold value: {}'.format(thresholds)
                        )
                    '''
                    

                    if formula%10==1:
                        prob = torch.exp(-delta[indices])
                    elif formula%10==2:
                        prob = torch.exp(-delta[indices]/p.each_flip_num[indices])
                    elif formula%10==3:
                        prob = torch.exp(-delta[indices]/pow(p.each_flip_num[indices],2))
                    elif formula%10==4:
                        prob = torch.exp(-delta[indices]/(p.epochs * p.each_flip_num[indices]))

                    tmp = torch.sign(2*torch.bernoulli(prob) - 1)

                    indices2 = torch.where(tmp==-1)

                    p.data[indices2] = p.data[indices2] - lr * grad[indices2]
                    p.each_flip_num[indices] += (1-tmp)/2
                   
                    if state['step'] == last_step:
                        p.epochs += 1
                       
        return loss
