import torch
from torch import nn
import torch.distributions as D
import torch.nn.functional as F


class EncoderBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
    ):
        super().__init__()
        self.conv_normal = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.conv_normal(x)


class DecoderBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
    ):
        super().__init__()
        self.conv_transpose = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.conv_transpose(x)


class LatentBlock(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        in_channels: int,
        kernel_size: int,
    ):
        super().__init__()
        self.conv_normal = nn.Conv2d(in_channels, 2 * latent_dim, kernel_size, stride=1)
        self.conv_transpose = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, in_channels, kernel_size, stride=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
        )

    def encode(self, x):
        x = self.conv_normal(x)
        x = x.view(*x.shape[:-2])
        mu, std_ = torch.chunk(x, chunks=2, dim=-1)
        pz = D.Normal(mu, F.softplus(std_))
        return pz

    def decode(self, z):
        z = z.unsqueeze(-1).unsqueeze(-1)
        return self.conv_transpose(z)
