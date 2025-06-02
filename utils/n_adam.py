import math
import torch
from torch.optim import Optimizer


class NAdam(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, schedule_decay=0.004):
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, schedule_decay=schedule_decay)
        super(NAdam, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(NAdam, self).__setstate__(state)

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('NAdam does not support sparse gradients')

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    # \mu^{t} 
                    state['m_schedule'] = 1.

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']

                beta1, beta2 = group['betas']
                
                schedule_decay = group['schedule_decay']

                state['step'] += 1

                if group['weight_decay'] != 0:
                    grad = grad.add(group['weight_decay'], p.data)

                # calculate the momentum cache \mu^{t} and \mu^{t+1}
                momentum_cache_t = beta1 * ( \
                    1. - 0.5 * (pow(0.96, state['step'] * schedule_decay)))
                momentum_cache_t_1 = beta1 * ( \
                    1. - 0.5 * (pow(0.96, (state['step'] + 1) * schedule_decay)))
                m_schedule_new = state['m_schedule'] * momentum_cache_t
                m_schedule_next = state['m_schedule'] * momentum_cache_t * momentum_cache_t_1

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                        
                g_prime = torch.div( grad, 1. - m_schedule_new)
                exp_avg_prime = torch.div( exp_avg,  1. - m_schedule_next )
                exp_avg_sq_prime = torch.div(exp_avg_sq,  1. - pow(beta2, state['step']))
                
                exp_avg_bar = torch.add( (1. - momentum_cache_t) * g_prime, \
                                         momentum_cache_t_1,  exp_avg_prime )

                denom = exp_avg_sq_prime.sqrt().add_(group['eps'])

                step_size = group['lr'] 

                p.data.addcdiv_(-step_size, exp_avg_bar, denom)
                                      
        return loss