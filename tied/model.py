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
from .layers import Encoder2DecoderProjector, Decoder2ChanelsProjector, Chanels2DecoderProjector, SelfAttentionBlock, Encoder2ChanelsProjector
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

        from diffusers import AutoencoderKL
        if not config.vae_model:
            raise ValueError("vae_model must be specified in config")

        self.vae = AutoencoderKL.from_pretrained(config.vae_model)
        for p in self.vae.parameters():
            p.requires_grad = False

        self.vocab_size = config.vocab_size
        self.text_prompt_pooling_type = config.text_prompt_pooling_type
        self.projector_hidden_act = ACT2FN[config.projector_hidden_act]

        self.encoder2chanels_projector = Encoder2ChanelsProjector(config)

        self.pooler  = POOLING2OBJECT[config.text_prompt_pooling_type]()

        self.dropout = torch.nn.Dropout(0.2)


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

    def get_loss(self, x, y):
        if x is None or y is None:
            raise ValueError("Both x and y must be provided for loss calculation")
        
        loss = torch.nn.functional.mse_loss(x, y, reduction=self.config.reduction)

        return loss

    def get_diffused_latents(self, input_images: torch.Tensor) -> torch.Tensor:
        if input_images.dim() != 4:
            raise ValueError("Expected input_images of shape (B, 3, H, W)")

        latents = self.vae.encode(input_images).latent_dist.sample().unsqueeze(1)

        return latents
    
    def forward(self, input_ids=None, attention_mask=None, input_images=None, **kwargs):

        if input_ids is None:
            raise ValueError("input_ids must be provided")

        # Encode text
        text_features = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state

        # Pool text features
        pooled_features = self.dropout(self.pooler(text_features))

        encoded_latents = self.encoder2chanels_projector(pooled_features)

        loss = None
        if input_images is not None:
            true_latents = self.get_diffused_latents(input_images)
            loss = self.get_loss(encoded_latents, true_latents)
        return TIEDModelOutput(
            decoded_latents=encoded_latents,
            loss=loss
        )

