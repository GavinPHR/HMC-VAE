from typing import Callable

import torch
from torch import nn


class HMC(nn.Module):
    """A basic implementation of Hamiltonian Monte Carlo.

    T iterations of HMC and L leapfrog steps per iteration.
    Step size is tuned to have 95% global acceptance rate.
    """

    def __init__(
        self,
        dim: int,
        T: int,
        L: int,
    ):
        super().__init__()
        self.dim = dim
        self.log_prob: Callable = None  # type: ignore
        self.T = T
        self.L = L
        self.register_buffer("step_size", torch.tensor(0.1))
        self.step_size = self.step_size  # type: ignore

    def register_log_prob(self, log_prob: Callable):
        self.log_prob = log_prob

    def grad_log_prob(self, x):
        with torch.enable_grad():
            x = x.detach()
            x.requires_grad = True
            logprob = self.log_prob(x).sum()
            grad = torch.autograd.grad(logprob, x)[0]
            return grad

    def leapfrog(self, x, p):
        eps = self.step_size
        p = p + 0.5 * eps * self.grad_log_prob(x)
        for _ in range(self.L - 1):
            x = x + eps * p
            p = p + eps * self.grad_log_prob(x)
        x = x + eps * p
        p = p + 0.5 * eps * self.grad_log_prob(x)
        return x, p

    def HMC_step(self, x_old):
        def H(x, p):
            return -self.log_prob(x) + 0.5 * torch.sum(p.pow(2), dim=-1)

        p_old = torch.randn_like(x_old)
        x_new, p_new = self.leapfrog(x_old.clone(), p_old.clone())
        log_accept_prob = -(H(x_new, p_new) - H(x_old, p_old))
        log_accept_prob[log_accept_prob > 0] = 0

        accept = torch.log(torch.rand_like(log_accept_prob)) < log_accept_prob
        accept = accept.unsqueeze(dim=-1)
        ret = x_new * accept + x_old * torch.logical_not(accept), accept.sum() / accept.numel()
        return ret

    def forward(self, x):
        accept_probs = []
        for _ in range(self.T):
            x, accept_prob = self.HMC_step(x)
            accept_probs.append(accept_prob)
        accept_prob = torch.mean(torch.tensor(accept_probs))
        if self.training:
            if accept_prob > 0.65:
                self.step_size = self.step_size * 1.005
            else:
                self.step_size = self.step_size * 0.995
        return x, accept_prob
