from typing import List

import torch
import torch.distributions as D

from models.vae import VAE
from models.hmc import HMC
import utils


class HMCVAE(VAE):
    def __init__(
        self,
        in_channels: int,
        latent_dims: List[int],
        hidden_channels: int,
        T: int,
        L: int,
    ):
        super().__init__(in_channels, latent_dims, hidden_channels)
        self.hmc = HMC(sum(latent_dims), T, L)

    def register_log_prob(self, x):
        def log_prob(z):
            with utils.EnableOnly(self):
                z_list = torch.split(z, self.latent_dims, dim=-1)
                x_logits = self.decode(z_list)
                return self.HMC_bound(x, x_logits, z_list)

        self.hmc.register_log_prob(log_prob)

    def run_hmc(self, x, z_list):
        self.register_log_prob(x)
        z = torch.cat(z_list, dim=-1)
        z, accept_prob = self.hmc(z.detach().clone())
        return torch.split(z, self.latent_dims, dim=-1), accept_prob

    def HMC_bound(self, x, x_logits, z_list):
        logp_x = D.Categorical(logits=x_logits).log_prob(x.int()).sum(dim=(-1, -2, -3))
        prior = D.Normal(0, 1)
        logp_z = sum(prior.log_prob(z).sum(dim=-1) for z in z_list)
        return logp_x + logp_z

    def encoding_parameters(self):
        params = []
        for name, param in self.named_parameters():
            if "conv_normal" in name:
                params.append(param)
        return params

    def decoding_parameters(self):
        params = []
        for name, param in self.named_parameters():
            if "conv_transpose" in name:
                params.append(param)
        return params
