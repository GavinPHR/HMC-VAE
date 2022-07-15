import torch.distributions as D

from vae import VAE
from hmc import HMC
import utils


class HMCVAE(VAE):
    def __init__(
        self,
        in_channels: int,
        latent_dim: int,
        T: int,
        L: int,
    ):
        super().__init__(in_channels, latent_dim)
        self.hmc = HMC(latent_dim, T, L)

    def register_log_prob(self, x):
        def log_prob(z):
            with utils.EnableOnly(self):
                x_logits = self.decoder(z).view(*x.shape)
                logp_x = D.ContinuousBernoulli(logits=x_logits).log_prob(x).sum(dim=(-1, -2, -3))
                logp_z = D.Normal(0, 1).log_prob(z).sum(dim=-1)
                return logp_x + logp_z

        self.hmc.register_log_prob(log_prob)

    def run_hmc(self, x, z):
        self.register_log_prob(x)
        z, accept_prob = self.hmc(z.detach().clone())
        return z, accept_prob

    def HMC_bound(self, x, x_logits, z):
        logp_x = D.ContinuousBernoulli(logits=x_logits).log_prob(x).sum(dim=(-1, -2, -3))
        logp_z = D.Normal(0, 1).log_prob(z).sum(dim=-1)
        return logp_x + logp_z
