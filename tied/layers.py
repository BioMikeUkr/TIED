from typing import Union
import torch
from torch import nn
import torch.nn.functional as F
from transformers.activations import ACT2FN

from .config import TIEDModelConfig

class Encoder2DecoderProjector(nn.Module):
    def __init__(self, config: TIEDModelConfig):
        super().__init__()

        self.linear_1 = nn.Linear(config.text_encoder_config.hidden_size, config.hidden_size, bias=True)
        self.act = ACT2FN[config.projector_hidden_act]
        self.linear_2 = nn.Linear(config.hidden_size, config.decoder_config.hidden_size, bias=True)

    def forward(self, features):
        hidden_states = self.linear_1(features)
        hidden_states = self.act(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        return hidden_states
    

class Decoder2LatentProjector(nn.Module):
    def __init__(self, config: TIEDModelConfig):
        super().__init__()

        num_down_blocks = len(config.vae_config["down_block_types"])
        out_spatial = config.image_size // (2 * num_down_blocks)
        out_size = out_spatial * out_spatial  # 16 * 16 = 256

        self.linear_1 = nn.Linear(config.decoder_config.hidden_size, config.hidden_size)
        self.act = ACT2FN[config.projector_hidden_act]
        self.linear_2 = nn.Linear(config.hidden_size, out_size)

    def forward(self, features):  # [B, T, hidden]
        x = self.linear_1(features)
        x = self.act(x)
        return self.linear_2(x)  # [B, T, out_size]

    
class Chanels2DecoderProjector(nn.Module):
    def __init__(self, config: TIEDModelConfig):
        super().__init__()

        num_down_blocks = len(config.vae_config["down_block_types"])

        latent_channels = config.vae_config.get("latent_channels", 4)
        in_size = (config.image_size // (2 * num_down_blocks)) ** 2 * latent_channels
        self.out_size = config.decoder_config.hidden_size

        self.linear_1 = nn.Linear(in_size, config.hidden_size)
        self.act = ACT2FN[config.projector_hidden_act]
        self.linear_2 = nn.Linear(config.hidden_size, config.decoder_config.hidden_size)

    def forward(self, features):
        B, Z, C, H, W = features.shape
        x = features.view(B, Z, C * H * W)  # [B, Z, in_size]
        x = self.linear_1(x)
        x = self.act(x)
        return self.linear_2(x)  # [B, Z, hidden]

    
class Decoder2ChanelsProjector(nn.Module):
    def __init__(self, config: TIEDModelConfig):
        super().__init__()
        self.latent_channels = config.vae_config["latent_channels"]
        self.spatial_size = config.image_size // (2 * len(config.vae_config["down_block_types"]))

        self.projector = nn.ModuleList([
            Decoder2LatentProjector(config) for _ in range(self.latent_channels)
        ])

    def forward(self, features):  # [B, T, hidden]
        latents = [p(features) for p in self.projector]  # each: [B, T, H*W]
        latents = torch.stack(latents, dim=2)  # [B, C, T, H*W]
        latents = latents.view(latents.size(0), latents.size(1), latents.size(2), self.spatial_size, self.spatial_size)        
        return latents

