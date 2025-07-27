from typing import Union
import torch
from torch import nn
import torch.nn.functional as F
from transformers.activations import ACT2FN

from .config import TIEDModelConfig


class Encoder2LatentProjector(nn.Module):
    def __init__(self, config: TIEDModelConfig):
        super().__init__()

        num_down_blocks = len(config.vae_config["down_block_types"])
        out_spatial = config.image_size // (2 * num_down_blocks)
        out_size = out_spatial * out_spatial  # 16 * 16 = 256

        self.linear_1 = nn.Linear(config.text_encoder_config.hidden_size, (config.text_encoder_config.hidden_size + out_size)//2)
        self.act = ACT2FN[config.projector_hidden_act]
        self.linear_2 = nn.Linear((config.text_encoder_config.hidden_size + out_size)//2, out_size)

    def forward(self, features):  # [B, T, hidden]
        x = self.linear_1(features)
        x = self.act(x)
        return self.linear_2(x)  # [B, T, out_size]
    
class Encoder2ChanelsProjector(nn.Module):
    def __init__(self, config: TIEDModelConfig):
        super().__init__()
        self.latent_channels = config.vae_config["latent_channels"]
        self.spatial_size = config.image_size // (2 * len(config.vae_config["down_block_types"]))

        self.projector = nn.ModuleList([
            Encoder2LatentProjector(config) for _ in range(self.latent_channels)
        ])

    def forward(self, features):
        latents = [p(features) for p in self.projector]  # each: [B, T, H*W]
        latents = torch.stack(latents, dim=2)  # [B, C, T, H*W]
        latents = latents.view(latents.size(0), latents.size(1), latents.size(2), self.spatial_size, self.spatial_size)  # [B, C, T, H, W]

        B, C, T, H, W = latents.shape
        D = min(C, T)

        batch_idx = torch.arange(B, device=latents.device).unsqueeze(1)  # [B, 1]
        diag_idx = torch.arange(D, device=latents.device).unsqueeze(0)   # [1, D]

        diag_latents = latents[batch_idx, diag_idx, diag_idx]  # [B, D, H, W]

        # Вместо in-place присваивания — сформировать нужный тензор напрямую
        padding = T - D
        if padding > 0:
            pad = torch.zeros(B, padding, H, W, device=latents.device, dtype=latents.dtype)
            diag_latents = torch.cat([diag_latents, pad], dim=1)  # [B, T, H, W]

        zero_latents = diag_latents.unsqueeze(1)  # [B, 1, T, H, W]

        return zero_latents


class FeaturesProjector(nn.Module):
    def __init__(self, config: TIEDModelConfig, in_dim, out_dim, hidden_dim):
        super().__init__()

        self.linear_1 = nn.Linear(in_dim, hidden_dim, bias=True)
        self.act = ACT2FN[config.projector_hidden_act]
        self.linear_2 = nn.Linear(hidden_dim, out_dim, bias=True)

    def forward(self, features):
        hidden_states = self.linear_1(features)
        hidden_states = self.act(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        return hidden_states


# class SpatialSelfAttention(nn.Module):
#     def __init__(self, in_channels):
#         super().__init__()
#         self.query = nn.Conv2d(in_channels, in_channels, 1)
#         self.key   = nn.Conv2d(in_channels, in_channels, 1)
#         self.value = nn.Conv2d(in_channels, in_channels, 1)
#         self.scale = in_channels ** -0.5

#     def forward(self, x):  # [B, C, H, W]
#         B, C, H, W = x.shape
#         q = self.query(x).reshape(B, C, -1)  # [B, C, HW]
#         k = self.key(x).reshape(B, C, -1)    # [B, C, HW]
#         v = self.value(x).reshape(B, C, -1)  # [B, C, HW]

#         attn = torch.bmm(q.transpose(1, 2), k) * self.scale  # [B, HW, HW]
#         attn = attn.softmax(dim=-1)
#         out = torch.bmm(v, attn.transpose(1, 2))  # [B, C, HW]
#         out = out.reshape(B, C, H, W)
#         return out + x  # residual

class MultiHeadSpatialAttention(nn.Module):
    def __init__(self, in_channels, num_heads=4, ffn_expansion=2):
        super().__init__()
        assert in_channels % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = in_channels // num_heads
        self.scale = self.head_dim ** -0.5

        self.query = nn.Conv2d(in_channels, in_channels, 1)
        self.key   = nn.Conv2d(in_channels, in_channels, 1)
        self.value = nn.Conv2d(in_channels, in_channels, 1)
        self.out_proj = nn.Conv2d(in_channels, in_channels, 1)

        self.norm1 = nn.BatchNorm2d(in_channels)
        self.norm2 = nn.BatchNorm2d(in_channels)

        hidden_dim = in_channels * ffn_expansion
        self.ffn = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, 1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, in_channels, 1),
        )

    def forward(self, x):
        B, C, H, W = x.shape
        q = self.query(x).reshape(B, self.num_heads, self.head_dim, H * W)
        k = self.key(x).reshape(B, self.num_heads, self.head_dim, H * W)
        v = self.value(x).reshape(B, self.num_heads, self.head_dim, H * W)

        attn = torch.einsum("bnch,bnck->bnhk", q, k) * self.scale
        attn = attn.softmax(dim=-1)

        out = torch.einsum("bnhk,bnck->bnch", attn, v)
        out = out.reshape(B, C, H, W)
        x = x + self.out_proj(out)
        x = self.norm1(x)
        x = x + self.ffn(x)
        x = self.norm2(x)
        return x

import math

# class DownBlock(nn.Module):
#     def __init__(self, in_ch, out_ch):
#         super().__init__()
#         self.block = nn.Sequential(
#             nn.Conv2d(in_ch, out_ch, 4, 2, 1),
#             MultiHeadSpatialAttention(out_ch),
#             nn.BatchNorm2d(out_ch),
#             nn.ReLU()
#         )

#     def forward(self, x):
#         return self.block(x)


# class UpBlock(nn.Module):
#     def __init__(self, in_ch, out_ch):
#         super().__init__()
#         self.block = nn.Sequential(
#             nn.ConvTranspose2d(in_ch, out_ch, 4, 2, 1),
#             SpatialSelfAttention(out_ch),
#             nn.BatchNorm2d(out_ch),
#             nn.ReLU()
#         )

#     def forward(self, x):
#         return self.block(x)

class GaussianBlur2d(nn.Module):
    def __init__(self, channels, kernel_size=9, sigma=1.0):
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd")

        self.padding = kernel_size // 2

        coords = torch.arange(kernel_size) - self.padding
        g = torch.exp(-(coords**2) / (2 * sigma**2))
        g = g / g.sum()

        kernel_2d = torch.outer(g, g)
        kernel_2d = kernel_2d[None, None, :, :]               # [1, 1, K, K]
        kernel_2d = kernel_2d.repeat(channels, 1, 1, 1)       # [C, 1, K, K]

        self.register_buffer("weight", kernel_2d)
        self.groups = channels

    def forward(self, x):
        return F.conv2d(x, self.weight, padding=self.padding, groups=self.groups)



# class UpBlock(nn.Module):
#     def __init__(self, in_ch, out_ch, upsample_mode="nearest"):
#         super().__init__()
#         self.block = nn.Sequential(
#             nn.Upsample(scale_factor=2, mode=upsample_mode),
#             nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1),
#             MultiHeadSpatialAttention(out_ch),
#             nn.BatchNorm2d(out_ch),
#             nn.ReLU()
#         )

#     def forward(self, x):
#         return self.block(x)

# class DownBlock(nn.Module):
#     def __init__(self, in_ch, out_ch):
#         super().__init__()
#         self.block = nn.Sequential(
#             GaussianBlur2d(in_ch, kernel_size=3, sigma=1.0),
#             nn.Conv2d(in_ch, out_ch, 4, 2, 1),
#             MultiHeadSpatialAttention(out_ch),
#             nn.BatchNorm2d(out_ch),
#             nn.ReLU()
#         )

#     def forward(self, x):
#         return self.block(x)

class DownBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            # GaussianBlur2d(in_ch, kernel_size=5, sigma=1.0),
            nn.Conv2d(in_ch, out_ch, 4, 2, 1),
            MultiHeadSpatialAttention(out_ch),
            nn.BatchNorm2d(out_ch),
            nn.GELU()
        )

    def forward(self, x):
        return self.block(x)


class UpBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, 4, 2, 1),  # апсемплинг
            # GaussianBlur2d(out_ch, kernel_size=5, sigma=1.0),
            MultiHeadSpatialAttention(out_ch),
            nn.BatchNorm2d(out_ch),
            nn.GELU()
        )

    def forward(self, x):
        return self.block(x)



class LatentVAE(nn.Module):
    def __init__(self, config: TIEDModelConfig):
        super().__init__()
        in_channels = 4
        latent_dim = config.hidden_size
        input_size = config.image_size // (2 * len(config.vae_config["down_block_types"]))
        num_blocks = int(math.log2(input_size)) - 2

        channels = [in_channels] + [input_size * (2 ** i) for i in range(num_blocks)]
        rev_channels = list(reversed(channels))

        self.encoder = nn.Sequential(*[
            DownBlock(channels[i], channels[i + 1]) for i in range(num_blocks)
        ])

        final_ch = channels[-1]
        self.flat_dim = final_ch * 4 * 4
        self.flatten = nn.Flatten()

        self.fc_mu = nn.Linear(self.flat_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.flat_dim, latent_dim)
        self.decoder_input = nn.Linear(latent_dim, self.flat_dim)
        self.decoder_init_ch = rev_channels[0]

        self.decoder_blocks = nn.Sequential(*[
            UpBlock(rev_channels[i], rev_channels[i + 1]) for i in range(num_blocks)
        ])
        self.output_conv = nn.ConvTranspose2d(in_channels, in_channels, 3, 1, 1)

        # self.scale_factor = nn.Parameter(torch.tensor(1.0), requires_grad=True)

    def encode(self, x):
        x = self.encoder(x)
        x = self.flatten(x)
        return self.fc_mu(x), self.fc_logvar(x)

    def decode(self, z):
        x = self.decoder_input(z).view(-1, self.decoder_init_ch, 4, 4)
        x = self.decoder_blocks(x)
        x = self.output_conv(x)
        # x = x * self.scale_factor
        return x

    def reparameterize(self, mu, logvar):
        std = (0.5 * logvar).exp()
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar, z
