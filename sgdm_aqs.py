
import torch
from torch.optim.optimizer import Optimizer

class SGDMAQS(Optimizer):
    """Sparse-Gradient Dual-Momentum Descent with Adaptive Quantization Scheduling (optimizer part).
    - Gradient filtering: scales grads by a gate factor (0..1) exposed by the model's head.
    - Dual momentum buffers (short and long): combine with mixing gamma.
    """
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.98), gamma=0.5, weight_decay=0.0):
        defaults = dict(lr=lr, betas=betas, gamma=gamma, weight_decay=weight_decay, gate_factor=1.0)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None, gate_factor: float = None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            beta1, beta2 = group['betas']
            gamma = group['gamma']
            lr = group['lr']
            wd = group['weight_decay']
            gf = group['gate_factor'] if gate_factor is None else gate_factor

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad
                # Gradient filtering (scaled by gate factor from sparse gates)
                if gf is not None:
                    d_p = d_p.mul(gf)
                state = self.state[p]
                if len(state) == 0:
                    state['mshort'] = torch.zeros_like(p)
                    state['mlong'] = torch.zeros_like(p)
                mshort = state['mshort']
                mlong = state['mlong']
                # weight decay
                if wd != 0:
                    d_p = d_p.add(p, alpha=wd)
                # update moments
                mshort.mul_(beta1).add_(d_p, alpha=(1 - beta1))
                mlong.mul_(beta2).add_(d_p, alpha=(1 - beta2))
                # combine
                u = mshort.mul(gamma).add(mlong, alpha=(1 - gamma))
                p.add_(u, alpha=-lr)
        return loss
