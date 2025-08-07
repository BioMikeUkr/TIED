import os
import tempfile
import torch
import torch.nn.functional as F
from typing import Optional, Union
from dataclasses import dataclass
from transformers import (
    PreTrainedModel, AutoModel, AutoModelForCausalLM,
)
from transformers.modeling_outputs import BaseModelOutput
from transformers.activations import ACT2FN
from diffusers import AutoencoderKL

import lpips
import piq

from safetensors.torch import save_file, load_file
from huggingface_hub import create_repo, upload_folder, hf_hub_download

from .config import TIEDModelConfig
from .layers import LowLevelVectorLatentVAE, FeaturesProjector
from .pooling import POOLING2OBJECT

@dataclass
class TIEDModelOutput(BaseModelOutput):
    generated_latents: torch.Tensor = None
    loss: Optional[torch.Tensor] = None


class TIEDModel(PreTrainedModel):
    config_class = TIEDModelConfig
    base_model_prefix = "tied model"
    supports_gradient_checkpointing = True

    def __init__(self, config: TIEDModelConfig, device="cpu"):
        super().__init__(config)
        self.config = config
        self.text_encoder = AutoModel.from_config(config.text_encoder_config)
        self.decoder = AutoModelForCausalLM.from_config(config.decoder_config)

        if not config.vae_model:
            raise ValueError("vae_model must be specified in config")

        self.vae = AutoencoderKL.from_pretrained(config.vae_model)

        self.vocab_size = config.vocab_size
        self.text_prompt_pooling_type = config.text_prompt_pooling_type
        self.projector_hidden_act = ACT2FN[config.projector_hidden_act]

        self.encoder2latent_proj = FeaturesProjector(
            config, in_dim=config.text_encoder_config.hidden_size,
            out_dim=config.hidden_size,
            hidden_dim=config.hidden_size * 2
        )

        self.decoder2latent_proj = FeaturesProjector(
            config, in_dim=config.decoder_hidden_size,
            out_dim=1,
            hidden_dim=config.hidden_size // 2
        )

        self.pooler  = POOLING2OBJECT[config.text_prompt_pooling_type]()
        self.dropout = torch.nn.Dropout(0.0)

        self.inner_vae = LowLevelVectorLatentVAE(config)

        self._lpips_fn = None

    @property
    def lpips_fn(self):
        if self._lpips_fn is None:
            self._lpips_fn = lpips.LPIPS(net='vgg').to(self.device)
            self._lpips_fn.eval()
            for p in self._lpips_fn.parameters():
                p.requires_grad = False
        return self._lpips_fn

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

    def get_latents(self, input_images: torch.Tensor) -> torch.Tensor:
        if input_images.dim() != 4:
            raise ValueError("Expected input_images of shape (B, 3, H, W)")

        latents = self.vae.encode(input_images).latent_dist.sample()

        return latents
    
    def get_prompt_embeddings(self, input_ids, attention_mask=None):
        if input_ids is None:
            raise ValueError("input_ids must be provided")

        text_features = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state

        pooled_features = self.pooler(text_features)

        return self.encoder2latent_proj(pooled_features)
    
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
    
    def prepare_latent_for_decoder(self, latent: torch.Tensor):
        b, d = latent.size()
        
        random_start_spans = torch.zeros(
            (b, self.config.span_size),
            device=latent.device,
            dtype=latent.dtype
        )
        latent = torch.cat([random_start_spans, latent], dim=1)

        spans = latent.unfold(
            dimension=1, 
            size=self.config.span_size,
            step=1
        )
        
        return spans

    def forward(self, input_ids=None, attention_mask=None, input_images=None, **kwargs):
        text_latents = self.get_prompt_embeddings(input_ids, attention_mask)
        image_latent = self.get_latents(input_images)

        u, recon, mu, logvar, z = self.inner_vae(image_latent)
        image_latent = u

        prompt = self.prepare_latent_for_decoder(text_latents)
        latent = self.prepare_latent_for_decoder(image_latent)
        # print("text_latents", prompt.size())
        # print("image_latent", latent.size())
        input_embeds = torch.cat([prompt, latent[:,:-1, : ]], dim=1)
        # print("input_embeds", input_embeds.size())
        attention_mask = torch.ones(
            (input_embeds.size(0), input_embeds.size(1)),
            device=input_embeds.device
        )

        decoder_outputs = self.decoder(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            return_dict=True,
            output_hidden_states=True
        )
        b = latent.size(0)
        generated_latent = self.decoder2latent_proj(decoder_outputs.hidden_states[-1][:,self.config.hidden_size + 1:, :]).view(b, -1)
        # print("u ", u.size())
        # print("generated_latent ", generated_latent.size())
        loss = self.get_loss(generated_latent, u)

        return TIEDModelOutput(
                generated_latents=generated_latent,
                loss=loss,
            )

    def get_loss(self, x, y):
        return F.smooth_l1_loss(x, y, reduction="sum") / x.size(0)


    def generate(self, input_ids=None, attention_mask=None, target_length=2048, max_new_tokens=None):

        if input_ids is None:
            raise ValueError("input_ids must be provided")
        
        device = input_ids.device
        batch_size = input_ids.size(0)
        
        text_latents = self.get_prompt_embeddings(input_ids, attention_mask)
        prompt_spans = self.prepare_latent_for_decoder(text_latents)
        
        if max_new_tokens is not None:
            target_length = max_new_tokens
        
        print(f"Generating latent of length {target_length}")
        print(f"prompt_spans.shape: {prompt_spans.shape}")
        
        generated_values = []
        
        current_latent_spans = torch.zeros(
            (batch_size, 0, self.config.span_size),
            device=device,
            dtype=text_latents.dtype
        )
        
        for step in range(target_length):
            new_span = torch.zeros(
                (batch_size, 1, self.config.span_size),
                device=device,
                dtype=text_latents.dtype
            )
            current_latent_spans = torch.cat([current_latent_spans, new_span], dim=1)
            
            input_embeds = torch.cat([prompt_spans, current_latent_spans[:, :-1, :]], dim=1)
            
            attention_mask_tensor = torch.ones(
                (input_embeds.size(0), input_embeds.size(1)),
                device=device
            )
            
            with torch.no_grad():
                decoder_outputs = self.decoder(
                    inputs_embeds=input_embeds,
                    attention_mask=attention_mask_tensor,
                    return_dict=True,
                    output_hidden_states=True
                )
            
            latent_hidden = decoder_outputs.hidden_states[-1][:, self.config.hidden_size + 1:, :]
            
            if latent_hidden.size(1) > 0:
                last_hidden = latent_hidden[:, -1:, :]  # [batch, 1, decoder_hidden_size]
                generated_value = self.decoder2latent_proj(last_hidden).squeeze(-1)  # [batch, 1]
                generated_values.append(generated_value)
            else:
                generated_values.append(torch.zeros((batch_size, 1), device=device))
            
            if (step + 1) % 100 == 0:
                print(f"Generated {step + 1}/{target_length} values")
        
        if generated_values:
            final_latent = torch.cat(generated_values, dim=1)  # [batch, target_length]
            print(f"Generated latent shape: {final_latent.shape}")
            return final_latent
        else:
            print("No values generated")
            return torch.zeros((batch_size, target_length), device=device, dtype=text_latents.dtype)

    def generate_image(self, input_ids=None, attention_mask=None, target_length=2048):
        generated_latent = self.generate(
            input_ids=input_ids,
            attention_mask=attention_mask, 
            target_length=target_length
        )
        
        print(f"Generated latent for image: {generated_latent.shape}")
        
        with torch.no_grad():
            if hasattr(self.inner_vae, 'decode'):
                vae_latents = self.inner_vae.decode(generated_latent)
            elif hasattr(self.inner_vae, 'decoder'):
                vae_latents = self.inner_vae.decoder(generated_latent)
            else:
                _, vae_latents, _, _, _ = self.inner_vae(generated_latent)
            
            print(f"VAE latents shape: {vae_latents.shape}")
            
            generated_image = self.vae.decode(vae_latents).sample
            
            print(f"Generated image shape: {generated_image.shape}")
        
        return generated_image