import torch
import math
import torch
import torch.nn as nn
from diffusers.models.unets.unet_2d_blocks import DownEncoderBlock2D, UpDecoderBlock2D, UNetMidBlock2D
from torch import nn
import torch.nn.functional as F
from transformers.activations import ACT2FN

from .config import TIEDModelConfig


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

def gn_common(a, b, cap=32):
    for g in (32,16,8,4,2,1):
        if a % g == 0 and b % g == 0 and g <= cap:
            return g
    return 1

def gn_single(c, cap=32):
    for g in (32,16,8,4,2,1):
        if c % g == 0 and g <= cap:
            return g
    return 1

class LowLevelVectorLatentVAE(nn.Module):
    def __init__(self, config):
        super().__init__()
        in_channels = 4
        latent_dim = config.hidden_size
        latent_stride = 2 * len(config.vae_config["down_block_types"])
        input_size = config.image_size // latent_stride
        num_blocks = int(math.log2(input_size)) - 2
        if 2 ** (num_blocks + 2) != input_size:
            raise ValueError("image_size must yield 4x4 bottleneck with given config")

        bot_hw = input_size // (2 ** num_blocks)
        if latent_dim % (bot_hw * bot_hw) != 0:
            raise ValueError(f"hidden_size={latent_dim} not divisible by bottleneck area {bot_hw*bot_hw}")
        bot_ch = latent_dim // (bot_hw * bot_hw)

        core = [input_size * (2 ** i) for i in range(max(0, num_blocks - 1))]
        channels = [in_channels] + core + [bot_ch]
        rev_channels = list(reversed(channels))

        self.down = nn.ModuleList([
            DownEncoderBlock2D(
                in_channels=channels[i],
                out_channels=channels[i + 1],
                num_layers=1,
                resnet_eps=1e-6,
                resnet_act_fn="silu",
                resnet_groups=gn_common(channels[i], channels[i + 1]),
                add_downsample=True
            ) for i in range(num_blocks)
        ])

        self.bot_ch = bot_ch
        self.bot_hw = bot_hw
        self.mid = UNetMidBlock2D(
            in_channels=self.bot_ch,
            resnet_eps=1e-6,
            resnet_act_fn="silu",
            resnet_groups=gn_single(self.bot_ch),
            add_attention=True,
            temb_channels=None
        )

        self.flat_dim = latent_dim
        self.flatten = nn.Flatten()
        self.fc_mu = nn.Linear(self.flat_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.flat_dim, latent_dim)
        self.fc_seed = nn.Linear(latent_dim, self.flat_dim)

        self.up = nn.ModuleList([
            UpDecoderBlock2D(
                in_channels=rev_channels[i],
                out_channels=rev_channels[i + 1],
                num_layers=1,
                resnet_eps=1e-6,
                resnet_act_fn="silu",
                resnet_groups=gn_common(rev_channels[i], rev_channels[i + 1]),
                add_upsample=True
            ) for i in range(num_blocks)
        ])

        self.conv_norm_out = nn.GroupNorm(gn_single(rev_channels[-1]), rev_channels[-1], eps=1e-6, affine=True)
        self.conv_act = nn.Identity()
        self.conv_out = nn.Conv2d(rev_channels[-1], in_channels, 3, 1, 1)

        print(f"[Init] in_channels={in_channels}, hidden_size={latent_dim}, image_size={config.image_size}, "
              f"latent_stride={latent_stride}, input_size={input_size}, num_blocks={num_blocks}, "
              f"bot_hw={self.bot_hw}, bot_ch={self.bot_ch}, flat_dim={self.flat_dim}")

        s = input_size
        for i in range(num_blocks - 1):
            print(f"[Down {i}] {channels[i]}x{s}x{s} -> {channels[i+1]}x{s//2}x{s//2}")
            s //= 2
        if num_blocks > 0:
            print(f"[Down {num_blocks-1}] {channels[num_blocks-1]}x{s}x{s} -> {channels[num_blocks]}x{s//2}x{s//2}")
            s //= 2

        print(f"[Mid] {self.bot_ch}x{self.bot_hw}x{self.bot_hw} = {self.flat_dim}")

        s = self.bot_hw
        for i in range(num_blocks):
            ns = min(input_size, s * 2)
            print(f"[Up {i}] {rev_channels[i]}x{s}x{s} -> {rev_channels[i+1]}x{ns}x{ns}")
            s = ns

        print(f"[FC mu] {self.flat_dim} -> {latent_dim}")
        print(f"[FC logvar] {self.flat_dim} -> {latent_dim}")
        print(f"[FC seed] {latent_dim} -> {self.flat_dim}")
        print(f"[Conv out] {rev_channels[-1]} -> {in_channels} at {s}x{s}")

    def encode(self, x):
        h = x
        for b in self.down:
            h = b(h)
        h = self.mid(h)
        u = self.flatten(h)
        mu = self.fc_mu(u)
        logvar = self.fc_logvar(u)
        return u, mu, logvar

    def reparameterize(self, mu, logvar):
        std = (0.5 * logvar).exp()
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.fc_seed(z).view(-1, self.bot_ch, self.bot_hw, self.bot_hw)
        for b in self.up:
            h = b(h)
        h = self.conv_norm_out(h)
        h = self.conv_act(h)
        x = self.conv_out(h)
        return x

    def forward(self, x):
        u, mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return u, recon, mu, logvar, z