import torch
import torch.nn as nn
import torch.nn.functional as F
from math import log2


def _norm(channels):
    return nn.GroupNorm(min(8, channels), channels)


class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.net = nn.Sequential(
            _norm(channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            _norm(channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
        )

    def forward(self, x):
        return x + self.net(x)


class Encoder(nn.Module):
    def __init__(self, in_channels, latent_channels, base_channels, num_downs, max_channels):
        super().__init__()
        layers = [nn.Conv2d(in_channels, base_channels, 3, padding=1)]
        ch = base_channels
        for _ in range(num_downs):
            ch_out = min(ch * 2, max_channels)
            layers += [
                nn.Conv2d(ch, ch_out, 4, stride=2, padding=1),
                _norm(ch_out),
                nn.SiLU(),
                ResBlock(ch_out),
            ]
            ch = ch_out
        layers.append(nn.Conv2d(ch, latent_channels * 2, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        mu, logvar = self.net(x).chunk(2, dim=1)
        return mu, logvar


class Decoder(nn.Module):
    def __init__(self, out_channels, latent_channels, base_channels, num_downs, max_channels):
        super().__init__()
        ch_list = [base_channels]
        for _ in range(num_downs):
            ch_list.append(min(ch_list[-1] * 2, max_channels))
        ch_list = ch_list[::-1]

        layers = [nn.Conv2d(latent_channels, ch_list[0], 1)]
        for i in range(num_downs):
            ch_in, ch_out = ch_list[i], ch_list[i + 1]
            layers += [
                ResBlock(ch_in),
                nn.ConvTranspose2d(ch_in, ch_out, 4, stride=2, padding=1),
                _norm(ch_out),
                nn.SiLU(),
            ]
        layers.append(nn.Conv2d(ch_list[-1], out_channels, 3, padding=1))
        self.net = nn.Sequential(*layers)

    def forward(self, z):
        return self.net(z)


class VAE(nn.Module):
    def __init__(
        self,
        in_channels: int = 2,
        latent_channels: int = 8,
        base_channels: int = 64,
        scale_factor: int = 8,
        max_channels: int = 512,
    ):
        super().__init__()
        assert scale_factor & (scale_factor - 1) == 0, "scale_factor must be a power of 2"
        num_downs = int(log2(scale_factor))

        self.scale_factor = scale_factor
        self.encoder = Encoder(in_channels, latent_channels, base_channels, num_downs, max_channels)
        self.decoder = Decoder(in_channels, latent_channels, base_channels, num_downs, max_channels)

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def reparameterize(self, mu, logvar):
        std = (0.5 * logvar).exp()
        return mu + std * torch.randn_like(std)

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z)
        return recon, mu, logvar


def vae_loss(recon, x, mu, logvar, kl_weight=1e-3):
    recon_loss = F.mse_loss(recon, x)
    kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).mean()
    return recon_loss + kl_weight * kl, recon_loss, kl
