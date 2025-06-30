import os
import tempfile
import torch
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
from .layers import Encoder2DecoderProjector, Decoder2ChanelsProjector, Chanels2DecoderProjector
from .pooling import POOLING2OBJECT

@dataclass
class TIEDModelOutput(BaseModelOutput):
   decoded_latents: torch.Tensor = None
   loss: Optional[torch.Tensor] = None


class TIEDModel(PreTrainedModel):
    config_class = TIEDModelConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True

    def __init__(self, config: TIEDModelConfig, device="cpu"):
        super().__init__(config)
        self.config = config
        self.text_encoder = AutoModel.from_config(config.text_encoder_config)
        self.decoder = AutoModelForCausalLM.from_config(config.decoder_config)

        from diffusers import AutoencoderKL
        if not config.vae_model:
            raise ValueError("vae_model must be specified in config")

        self.vae = AutoencoderKL.from_pretrained(config.vae_model)
        for p in self.vae.parameters():
            p.requires_grad = False

        self.vocab_size = config.vocab_size
        self.text_prompt_pooling_type = config.text_prompt_pooling_type
        self.projector_hidden_act = ACT2FN[config.projector_hidden_act]

        self.encoder2decoder_projector = Encoder2DecoderProjector(config)
        self.decoder2chanels_projector = Decoder2ChanelsProjector(config)
        self.chanels2decoder_projector = Chanels2DecoderProjector(config)

        self.pooler  = POOLING2OBJECT[config.text_prompt_pooling_type]()


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
        self.decoder.save_pretrained(os.path.join(save_directory, "decoder"))
        self.vae.save_pretrained(os.path.join(save_directory, "vae"))

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        config = TIEDModelConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)
        model = cls(config)

        model.text_encoder = AutoModel.from_pretrained(
            pretrained_model_name_or_path, subfolder="text_encoder"
        )

        model.decoder = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path, config=config.decoder_config, subfolder="decoder"
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

    def get_diffused_latents(self, input_images: torch.Tensor) -> torch.Tensor:
        steps = self.config.z_step

        if input_images.dim() != 4:
            raise ValueError("Expected input_images of shape (B, 3, H, W)")

        with torch.no_grad():
            latents = self.vae.encode(input_images).latent_dist.sample()
        
        betas = torch.linspace(1e-4, 0.3, steps, device=latents.device, dtype=latents.dtype)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)

        all_noised_latents = []

        for i in range(steps):
            alpha_bar = alpha_bars[i]
            noise = torch.randn_like(latents)
            noised_latent = torch.sqrt(alpha_bar) * latents + torch.sqrt(1 - alpha_bar) * noise
            all_noised_latents.append(noised_latent)

        return torch.stack(all_noised_latents, dim=1)
    
    def get_prompt_embeddings(self, input_ids, attention_mask=None):
        if input_ids is None:
            raise ValueError("input_ids must be provided")

        # Encode text
        text_features = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state

        # Pool text features
        pooled_features = self.pooler(text_features)

        # Project to decoder space
        decoder_features = self.encoder2decoder_projector(pooled_features)

        return decoder_features
    
    def construct_decoder_inputs(self, prompts, input_images):
        if input_images is not None:
            if input_images.dim() != 4:
                raise ValueError("Expected input_images of shape (B, 3, H, W)")
            latents = self.get_diffused_latents(input_images)
        else:
            raise ValueError("Either input_images or input_latents must be provided")

        # Project latents to decoder space
        projected_latents = self.chanels2decoder_projector(latents)

        zero_embeds = torch.zeros(
            (projected_latents.shape[0], self.config.z_step, self.config.decoder_config.hidden_size),
            device=projected_latents.device,
            dtype=projected_latents.dtype
        )
        zero_attention_mask = torch.ones(
            (zero_embeds.shape[0], self.config.z_step),
            device=zero_embeds.device,
            dtype=torch.long
        )

        zero_embeds[:, 0, :] = prompts
        zero_embeds[:, 1:, :] = projected_latents[:, -1:, :]
        

        return {
            "inputs_embeds": zero_embeds,
            "attention_mask": zero_attention_mask,
            "latent_labels": latents
        }
    

    def decode_prompts(self, inputs_embeds, attention_mask=None):
        if inputs_embeds is None:
            raise ValueError("inputs_embeds must be provided")

        # Decode using the decoder model
        decoder_outputs = self.decoder(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            return_dict=True,
            output_hidden_states=True
        )

        # Project to latent space
        latent_features = self.decoder2chanels_projector(decoder_outputs.hidden_states[-1])

        return latent_features

    def get_loss(self, x, y):
        if x is None or y is None:
            raise ValueError("Both x and y must be provided for loss calculation")

        # Calculate the loss between the decoded latents and the input images
        loss = torch.nn.functional.mse_loss(x, y, reduction=self.config.reduction)
        return loss
    
    def forward(self, input_ids=None, attention_mask=None, input_images=None, **kwargs):
        if input_ids is None:
            raise ValueError("input_ids must be provided")
        prompts = self.get_prompt_embeddings(input_ids, attention_mask)
        decoder_inputs = self.construct_decoder_inputs(prompts, input_images)

        decoded_latents = self.decode_prompts(
            inputs_embeds=decoder_inputs["inputs_embeds"],
            attention_mask=decoder_inputs["attention_mask"]
        )

        loss = None
        if input_images is not None:
            loss = self.get_loss(decoder_inputs["latent_labels"], decoded_latents)
        return TIEDModelOutput(
            decoded_latents=decoded_latents,
            loss=loss
        )
    
    def generate(self, input_ids=None, attention_mask=None):
        if input_ids is None:
            raise ValueError("input_ids must be provided")

        prompt_embed = self.get_prompt_embeddings(input_ids, attention_mask)
        batch_size, hidden_size = prompt_embed.shape
        z_steps = self.config.z_step

        inputs_embeds = prompt_embed.unsqueeze(1)
        attention_mask = torch.ones((batch_size, 1), device=prompt_embed.device, dtype=torch.long)
        decoded_latents = []

        for _ in range(1, z_steps):
            decoder_outputs = self.decoder(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
                output_hidden_states=True
            )
            decoded_latent = self.decoder2chanels_projector(decoder_outputs.hidden_states[:, -1, :])

            next_input_embed = self.chanels2decoder_projector(decoded_latent)

            inputs_embeds = torch.cat([inputs_embeds, next_input_embed], dim=1)
            attention_mask = torch.cat([attention_mask, torch.ones((batch_size, 1), device=attention_mask.device)], dim=1)
            decoded_latents.append(decoded_latent)

        # Final decoding to get latents
        final_decoder_outputs = self.decoder(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            return_dict=True,
            output_hidden_states=True
        )
        decoded_latent = self.decoder2chanels_projector(final_decoder_outputs.hidden_states[:, -1, :])
        decoded_latents.append(decoded_latent)
        decoded_latents = torch.stack(decoded_latents, dim=1)

        return decoded_latents

