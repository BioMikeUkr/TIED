import os
import tempfile
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, kl_divergence

from typing import Optional, Union
from dataclasses import dataclass
from transformers import (
    PreTrainedModel, AutoModel, AutoModelForCausalLM,
)
from transformers.modeling_outputs import BaseModelOutput
from transformers.activations import ACT2FN
from safetensors.torch import save_file, load_file
from huggingface_hub import create_repo, upload_folder, hf_hub_download

from .config import TIEDModelConfig
from .layers import Encoder2ChanelsProjector, FeaturesProjector, LatentVAE
from .pooling import POOLING2OBJECT

@dataclass
class TIEDModelOutput(BaseModelOutput):
    encoded_latents: torch.Tensor = None
    loss: Optional[torch.Tensor] = None
    recon_loss: Optional[torch.Tensor] = None
    kld: Optional[torch.Tensor] = None
    mse_align: Optional[torch.Tensor] = None


class TIEDModel(PreTrainedModel):
    config_class = TIEDModelConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True

    def __init__(self, config: TIEDModelConfig, device="cpu"):
        super().__init__(config)

        self.config = config
        self.text_encoder = AutoModel.from_config(config.text_encoder_config)

        from diffusers import AutoencoderKL
        if not config.vae_model:
            raise ValueError("vae_model must be specified in config")

        self.vae = AutoencoderKL.from_pretrained(config.vae_model)
        for p in self.vae.parameters():
            p.requires_grad = False

        self.vocab_size = config.vocab_size
        self.text_prompt_pooling_type = config.text_prompt_pooling_type
        self.projector_hidden_act = ACT2FN[config.projector_hidden_act]
        
        self.visual_tokens_projector = FeaturesProjector(
            config, in_dim=config.text_encoder_config.hidden_size,
            out_dim=config.hidden_size,
            hidden_dim=config.hidden_size * 2
        )

        self.pooler  = POOLING2OBJECT[config.text_prompt_pooling_type](config.n_pooling_tokens)

        self.dropout = torch.nn.Dropout(0.0)

        self.inner_vae = LatentVAE(config)


    def save_pretrained(self, save_directory, **kwargs):
        os.makedirs(save_directory, exist_ok=True)
        self.config.save_pretrained(save_directory)

        # Save only custom weights (excluding backbone submodules)
        filtered_state_dict = {
            k: v for k, v in self.state_dict().items()
            if not (k.startswith("text_encoder.") or k.startswith("decoder.") or k.startswith("vae."))
        }
        save_file(filtered_state_dict, os.path.join(save_directory, "model.safetensors"))

        # Save components
        self.text_encoder.save_pretrained(os.path.join(save_directory, "text_encoder"))
        self.vae.save_pretrained(os.path.join(save_directory, "vae"))

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        config = TIEDModelConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)
        model = cls(config)

        model.text_encoder = AutoModel.from_pretrained(
            pretrained_model_name_or_path, subfolder="text_encoder"
        )

        from diffusers import AutoencoderKL
        try:
            model.vae = AutoencoderKL.from_pretrained(
                pretrained_model_name_or_path, subfolder="vae"
            )
        except Exception:
            if config.vae_model:
                model.vae = AutoencoderKL.from_pretrained(config.vae_model)
            else:
                raise ValueError("VAE not found and no fallback specified")

        for p in model.vae.parameters():
            p.requires_grad = False

        # Load model.safetensors
        if os.path.isdir(pretrained_model_name_or_path):
            safetensor_path = os.path.join(pretrained_model_name_or_path, "model.safetensors")
        else:
            safetensor_path = hf_hub_download(
                repo_id=pretrained_model_name_or_path,
                filename="model.safetensors",
                repo_type="model"
            )

        state_dict = load_file(safetensor_path)
        model.load_state_dict(state_dict, strict=False)
        return model

    def push_to_hub(self, repo_id, token=None, private=False):
        create_repo(repo_id, token=token, private=private, exist_ok=True)
        tmpdir = tempfile.mkdtemp()
        self.save_pretrained(tmpdir)
        upload_folder(
            folder_path=tmpdir,
            repo_id=repo_id,
            token=token,
            repo_type="model"
        )
    def resize_token_embeddings(self, new_num_tokens: Optional[int] = None, pad_to_multiple_of=None) -> nn.Embedding:
        model_embeds = self.text_encoder.resize_token_embeddings(new_num_tokens, pad_to_multiple_of)

        self.config.text_encoder_config.vocab_size = model_embeds.num_embeddings
        return model_embeds
    
    def get_loss(self, text_z, z, recon, mu, logvar, true_latents, input_images=None):
        cfg = self.config
        alpha = getattr(cfg, "inner_recon_weight", 0.3)
        beta = getattr(cfg, "beta", 0.5)
        gamma = getattr(cfg, "mse_weight", 6.0)
        delta = getattr(cfg, "main_recon_weight", 1.0)

        batch_size = recon.size(0)
        
        decoded = self.vae.decode(recon).sample
        main_recon_loss = F.mse_loss(decoded, input_images, reduction="sum") / batch_size
        inner_recon_loss = F.mse_loss(recon, true_latents, reduction="sum") / batch_size
        mse_align = F.mse_loss(text_z, z, reduction="sum") / batch_size

        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / batch_size

        inner_recon_loss = alpha * inner_recon_loss
        main_recon_loss = delta * main_recon_loss
        kld = beta * kld
        mse_align = gamma * mse_align

        loss = inner_recon_loss + main_recon_loss + kld + mse_align

        return loss, {
            "recon_loss": main_recon_loss.detach() + inner_recon_loss.detach(),
            "kld": kld.detach(),
            "mse_align": mse_align.detach()
        }


    def get_diffused_latents(self, input_images: torch.Tensor) -> torch.Tensor:
        if input_images.dim() != 4:
            raise ValueError("Expected input_images of shape (B, 3, H, W)")

        latents = self.vae.encode(input_images).latent_dist.sample()

        return latents
    
    def vaes_forward(self, input_images: torch.Tensor) -> torch.Tensor:
        if input_images.dim() != 4:
            raise ValueError("Expected input_images of shape (B, 3, H, W)")

        true_latents = self.get_diffused_latents(input_images)
        recon, mu, logvar, z = self.inner_vae(true_latents)
        batch_size = recon.size(0)
        recon_loss = F.mse_loss(recon, true_latents, reduction="sum") / batch_size

        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / batch_size
        loss = recon_loss + kld

        return TIEDModelOutput(
                    encoded_latents=None,
                    loss=loss,
                    recon_loss=recon_loss,
                    kld=kld,
                    mse_align=kld
                )
    
    def forward(self, input_ids=None, attention_mask=None, input_images=None, **kwargs):

        if input_ids is None:
            raise ValueError("input_ids must be provided")
        
        if self.config.train_vae_only and input_images is not None:
            return self.vaes_forward(input_images)

        # Encode text
        text_features = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        
        # Pool text features
        pooled_features = self.pooler(text_features)
        text_z = self.visual_tokens_projector(pooled_features)
        encoded_latents = self.inner_vae.decode(text_z)

        loss = None
        if input_images is not None:
            true_latents = self.get_diffused_latents(input_images)
            recon, mu, logvar, z = self.inner_vae(true_latents)
            loss, metrics = self.get_loss(text_z, z, recon, mu, logvar, true_latents, input_images)
            return TIEDModelOutput(
                encoded_latents=encoded_latents,
                loss=loss,
                recon_loss=metrics["recon_loss"],
                kld=metrics["kld"],
                mse_align=metrics["mse_align"]
            )
        else:
            return TIEDModelOutput(
                encoded_latents=encoded_latents
            )

