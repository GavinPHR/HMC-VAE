from typing import List

from torch import nn
import torch.distributions as D

from models.blocks import EncoderBlock, LatentBlock, DecoderBlock


class VAE(nn.Module):
    def __init__(
        self,
        in_channels: int,
        latent_dims: List[int],
        hidden_channels: int,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.latent_dims = latent_dims
        c = hidden_channels
        self.encoder_blocks = nn.ModuleList(
            [
                EncoderBlock(in_channels=self.in_channels, out_channels=c),
                EncoderBlock(in_channels=c, out_channels=2 * c),
                EncoderBlock(in_channels=2 * c, out_channels=4 * c),
            ]
        )
        self.latent_blocks = nn.ModuleList([])
        for i, latent_dim in enumerate(latent_dims):
            in_channels = 2 ** (2 - i) * c
            kernel_size = 4 * 2**i
            self.latent_blocks.append(LatentBlock(latent_dim, in_channels, kernel_size))
        self.final_conv_transpose = nn.ConvTranspose2d(
            c, 256 * self.in_channels, kernel_size=4, stride=2, padding=1
        )
        self.decoder_blocks = nn.ModuleList(
            [
                DecoderBlock(in_channels=4 * c, out_channels=2 * c),
                DecoderBlock(in_channels=2 * c, out_channels=c),
                self.final_conv_transpose,
            ]
        )

    def encode(self, x):
        pz_list = []
        for i, encoder in enumerate(self.encoder_blocks):
            x = encoder(x)
            if 2 - i < len(self.latent_blocks):
                pz_list.append(self.latent_blocks[2 - i].encode(x))
        return pz_list[::-1]

    def decode(self, z_list):
        h = self.latent_blocks[0].decode(z_list[0])
        for i, decoder in enumerate(self.decoder_blocks):
            h = decoder(h)
            if i + 1 < len(self.latent_blocks):
                h = h + self.latent_blocks[i + 1].decode(z_list[i + 1])
        N, C, H, W = h.shape
        x_logits = h.view(N, self.in_channels, 256, H, W)
        return x_logits.permute(0, 1, 3, 4, 2)

    def ELBO(self, x, x_logits, pz_list):
        logp_x = D.Categorical(logits=x_logits).log_prob(x.int()).sum(dim=(-1, -2, -3))
        prior = D.Normal(0, 1)
        kl = sum(D.kl_divergence(pz, prior).sum(dim=-1) for pz in pz_list)
        return logp_x - kl
