import torch
from torch import nn
import torch.nn.functional as F
import torch.distributions as D


class VAE(nn.Module):
    class View(nn.Module):
        def __init__(self, shape):
            super().__init__()
            self.shape = shape

        def forward(self, x):
            return x.view(-1, *self.shape)

    def __init__(
        self,
        in_channels: int,
        latent_dim: int,
    ):
        super().__init__()
        self.encoder = nn.Sequential(
            # 1, 32, 32
            nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=3, stride=2),
            # 32, 15, 15
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2),
            # 64, 7, 7
            nn.ReLU(),
            nn.Flatten(),
            # 3136
            nn.Linear(3136, 2 * latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32 * 8 * 8),
            self.View((32, 8, 8)),
            nn.ConvTranspose2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1),
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=3, stride=2, padding=1),
            nn.ConvTranspose2d(
                in_channels=32, out_channels=in_channels, kernel_size=3, stride=1, padding=1
            ),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32 * 8 * 8),
            self.View((32, 8, 8)),
            nn.ConvTranspose2d(in_channels=32, out_channels=64, kernel_size=2, stride=2),
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=2, stride=2),
            nn.ConvTranspose2d(in_channels=32, out_channels=in_channels, kernel_size=1),
        )

    def encode(self, x):
        mu, std_ = torch.chunk(self.encoder(x), chunks=2, dim=-1)
        pz = D.Normal(mu, F.softplus(std_))
        return pz

    def decode(self, z):
        x_logits = self.decoder(z)
        return x_logits

    def ELBO(self, x, x_logits, pz):
        logp_x = D.ContinuousBernoulli(logits=x_logits).log_prob(x).sum(dim=(-1, -2, -3))
        kl = D.kl_divergence(pz, D.Normal(0, 1)).sum(dim=-1)
        return logp_x - kl
