import torch
from torch import nn
import torch.nn.functional as F
import torch.distributions as D


class VAE(nn.Module):
    def __init__(
        self,
        in_channels: int,
        latent_dim: int,
        hidden_channels: int,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.latent_dim = latent_dim
        c = hidden_channels
        self.encoder = nn.Sequential(
            # in_channels, 32, 32
            nn.Conv2d(in_channels=in_channels, out_channels=c, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(c),
            nn.ReLU(),
            # *, 16, 16
            nn.Conv2d(in_channels=c, out_channels=2 * c, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(2 * c),
            nn.ReLU(),
            # *, 8, 8
            nn.Conv2d(in_channels=2 * c, out_channels=4 * c, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(4 * c),
            nn.ReLU(),
            # *, 4, 4
            nn.Conv2d(in_channels=4 * c, out_channels=2 * latent_dim, kernel_size=4, stride=1),
            # 200, 1, 1
        )
        self.decoder = nn.Sequential(
            # 100, 1, 1
            nn.ConvTranspose2d(in_channels=latent_dim, out_channels=4 * c, kernel_size=4, stride=1),
            nn.BatchNorm2d(4 * c),
            nn.ReLU(),
            # *, 4, 4
            nn.ConvTranspose2d(
                in_channels=4 * c, out_channels=2 * c, kernel_size=4, stride=2, padding=1
            ),
            nn.BatchNorm2d(2 * c),
            nn.ReLU(),
            # *, 8, 8
            nn.ConvTranspose2d(
                in_channels=2 * c, out_channels=c, kernel_size=4, stride=2, padding=1
            ),
            nn.BatchNorm2d(c),
            nn.ReLU(),
            # *, 16, 16
            nn.ConvTranspose2d(
                in_channels=c, out_channels=256 * in_channels, kernel_size=4, stride=2, padding=1
            ),
            # 256 * in_channels, 32, 32
        )

    def encode(self, x):
        x = self.encoder(x).view(-1, 2 * self.latent_dim)
        mu, std_ = torch.chunk(x, chunks=2, dim=-1)
        pz = D.Normal(mu, F.softplus(std_))
        return pz

    def decode(self, z):
        z = z.view(-1, self.latent_dim, 1, 1)
        x_logits = self.decoder(z)
        N, C, H, W = x_logits.shape
        x_logits = x_logits.view(N, self.in_channels, 256, H, W)
        return x_logits.permute(0, 1, 3, 4, 2)

    def ELBO(self, x, x_logits, pz):
        logp_x = D.Categorical(logits=x_logits).log_prob(x.int()).sum(dim=(-1, -2, -3))
        kl = D.kl_divergence(pz, D.Normal(0, 1)).sum(dim=-1)
        return logp_x - kl
