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
from .layers import Encoder2ChanelsProjector, FeaturesProjector, LatentVAE, LowLevelVectorLatentVAE
from .pooling import POOLING2OBJECT

import lpips
import piq

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

    def __init__(self, config: TIEDModelConfig, device="cuda"):
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
        if not (k.startswith("text_encoder.") or 
                k.startswith("decoder.") or 
                k.startswith("vae.") or
                k.startswith("_lpips_fn.") or
                k == "_lpips_fn") 
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
            p.requires_grad = True

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
    
    # def get_loss(self, text_u, u, recon, mu, logvar, true_latents, input_images=None):
    #     cfg = self.config
    #     alpha = getattr(cfg, "inner_recon_weight", 1.0)
    #     beta = getattr(cfg, "beta", 1.0)
    #     gamma = getattr(cfg, "mse_weight", 30.0)
    #     delta = getattr(cfg, "main_recon_weight", 1.0)


    #     batch_size = recon.size(0)
        
    #     decoded = self.vae.decode(recon).sample
    #     main_recon_loss = F.mse_loss(decoded, input_images, reduction="sum") / batch_size
    #     inner_recon_loss = F.mse_loss(recon, true_latents, reduction="sum") / batch_size
    #     mse_align = F.mse_loss(text_u, u, reduction="sum") / batch_size

    #     kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / batch_size

    #     inner_recon_loss = alpha * inner_recon_loss
    #     main_recon_loss = delta * main_recon_loss
    #     kld = beta * kld
    #     mse_align = gamma * mse_align

    #     loss = inner_recon_loss + kld + mse_align + main_recon_loss

    #     return loss, {
    #         "recon_loss": inner_recon_loss.detach() + main_recon_loss.detach(),
    #         "kld": kld.detach(),
    #         "mse_align": mse_align.detach()
    #     }
    # def get_loss(self, text_u, u, recon, mu, logvar, true_latents, input_images=None):
    #     cfg = self.config
    #     alpha = getattr(cfg, "inner_recon_weight", 1.0)
    #     beta = getattr(cfg, "beta", 1.0)
    #     gamma = getattr(cfg, "mse_weight", 30.0)
    #     delta = getattr(cfg, "main_recon_weight", 1.0)

    #     batch_size = recon.size(0)

    #     decoded = self.vae.decode(recon).sample  # ∈ [−1, 1]

    #     lpips_loss = self.lpips_fn(decoded, input_images).mean()

    #     decoded_01 = ((decoded + 1) / 2).clamp(0, 1)
    #     input_01 = ((input_images + 1) / 2).clamp(0, 1)

    #     # SSIM и MSE
    #     ssim_loss = 1.0 - piq.ssim(decoded_01, input_01, data_range=1.0)
    #     mse_loss = F.mse_loss(decoded_01, input_01, reduction="mean")

    #     main_recon_loss = delta * (1.0 * lpips_loss + 0.2 * ssim_loss + 0.1 * mse_loss)

    #     inner_recon_loss = F.mse_loss(recon, true_latents, reduction="sum") / batch_size
    #     inner_recon_loss = alpha * inner_recon_loss

    #     mse_align = F.mse_loss(text_u, u, reduction="sum") / batch_size
    #     mse_align = gamma * mse_align

    #     kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / batch_size
    #     kld = beta * kld

    #     loss = inner_recon_loss + kld + mse_align + main_recon_loss

    #     return loss, {
    #         "recon_loss": inner_recon_loss.detach() + main_recon_loss.detach(),
    #         "kld": kld.detach(),
    #         "mse_align": mse_align.detach()
    #     }
    def get_loss(self, text_u, u, recon, mu, logvar, true_latents, input_images=None):
        cfg = self.config
        alpha = getattr(cfg, "inner_recon_weight", 1.0)
        beta = getattr(cfg, "beta", 1.0)
        gamma = getattr(cfg, "mse_weight", 60.0)
        delta = getattr(cfg, "main_recon_weight", 1.0)
        
        # New parameters for inner loss
        inner_mse_weight = getattr(cfg, "inner_mse_weight", 0.5)
        inner_ssim_weight = getattr(cfg, "inner_ssim_weight", 0.5)
        
        batch_size = recon.size(0)
        
        # === INNER RECONSTRUCTION LOSS with SSIM ===
        # 1. L1 component with summation
        inner_mse = F.l1_loss(recon, true_latents, reduction="sum") / batch_size
        # print("inner_mse ", inner_mse)
        
        # 2. SSIM component
        # Normalize latents for SSIM
        recon_norm = torch.zeros_like(recon)
        true_norm = torch.zeros_like(true_latents)
        
        # Get number of channels dynamically
        num_channels = recon.size(1)  # should be 4 for SD VAE
        
        for ch in range(num_channels):
            recon_ch = recon[:, ch:ch+1, :, :]
            true_ch = true_latents[:, ch:ch+1, :, :]
            
            # Min-max normalization per channel
            recon_min = recon_ch.min()
            recon_max = recon_ch.max()
            true_min = true_ch.min()
            true_max = true_ch.max()
            
            recon_norm[:, ch:ch+1, :, :] = (recon_ch - recon_min) / (recon_max - recon_min + 1e-5)
            true_norm[:, ch:ch+1, :, :] = (true_ch - true_min) / (true_max - true_min + 1e-5)
        
        # Calculate SSIM for each channel
        inner_ssim = 0
        for ch in range(num_channels):
            ssim_value = piq.ssim(
                recon_norm[:, ch:ch+1, :, :], 
                true_norm[:, ch:ch+1, :, :], 
                data_range=1.0
            )
            inner_ssim += ssim_value.mean()  # Average over batch
        inner_ssim = inner_ssim / num_channels
        inner_ssim_loss = 1.0 - inner_ssim
        # print("inner_ssim_loss ", inner_ssim_loss)
        
        # Scale SSIM loss to match the scale of L1 loss
        latent_elements = recon.size(1) * recon.size(2) * recon.size(3)
        inner_ssim_loss = inner_ssim_loss * latent_elements
        # print("inner_ssim_loss scaled ", inner_ssim_loss)
        
        # Combine L1 and SSIM
        inner_recon_loss = alpha * (inner_mse_weight * inner_mse + inner_ssim_weight * inner_ssim_loss)
        # print("inner_recon_loss ", inner_recon_loss)
        
        # === MAIN RECONSTRUCTION LOSS ===
        decoded = self.vae.decode(recon).sample  # in [-1, 1]
        
        # LPIPS loss - returns mean, convert to sum scale
        lpips_per_element = self.lpips_fn(decoded, input_images).mean()
        # Convert mean to sum: mean * total_elements / batch_size
        image_elements = decoded.size(1) * decoded.size(2) * decoded.size(3)  # C*H*W
        lpips_loss = lpips_per_element * image_elements
        # print("lpips_loss ", lpips_loss)
        
        # Normalize images to [0, 1] for SSIM and pixel loss
        decoded_01 = ((decoded + 1) / 2).clamp(0, 1)
        input_01 = ((input_images + 1) / 2).clamp(0, 1)
        
        # SSIM loss for images - convert to sum scale
        ssim_value = piq.ssim(decoded_01, input_01, data_range=1.0)
        ssim_per_element = 1.0 - ssim_value.mean()
        # Convert to sum scale
        ssim_loss = ssim_per_element * image_elements  # C*H*W
        # print("ssim_loss ", ssim_loss)
        
        # MSE loss for images - already sum/batch_size
        mse_loss = F.mse_loss(decoded_01, input_01, reduction="sum") / batch_size
        # print("mse_loss ", mse_loss)
        
        # Combine perceptual losses
        main_recon_loss = delta * (1.0 * lpips_loss + 0.2 * ssim_loss + 0.1 * mse_loss)
        # print("main_recon_loss ", main_recon_loss)
        
        # === ALIGNMENT LOSS ===
        mse_align = F.mse_loss(text_u, u, reduction="sum") / batch_size
        mse_align = gamma * mse_align
        # print("mse_align ", mse_align)
        
        # === KL DIVERGENCE ===
        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / batch_size
        kld = beta * kld
        # print("kld ", kld)
        
        # === TOTAL LOSS ===
        loss = inner_recon_loss + kld + mse_align + main_recon_loss
        # print("total loss ", loss)
        # print("="*50)
        
        return loss, {
            "recon_loss": inner_recon_loss.detach() + main_recon_loss.detach(),
            "kld": kld.detach(),
            "mse_align": mse_align.detach(),
            "inner_ssim": inner_ssim.detach(),  # for monitoring (0-1 range)
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
        u, recon, mu, logvar, z = self.inner_vae(true_latents)
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
    
    def text_encoder_forward(self, input_images: torch.Tensor, input_ids=None, attention_mask=None) -> torch.Tensor:
        if input_images.dim() != 4:
            raise ValueError("Expected input_images of shape (B, 3, H, W)")

        true_latents = self.get_diffused_latents(input_images)
        u, recon, mu, logvar, z = self.inner_vae(true_latents)
        batch_size = recon.size(0)

        # Pool text features
        text_features = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        pooled_features = self.pooler(text_features)
        text_u = self.visual_tokens_projector(pooled_features)

        # Normalize embeddings for cosine similarity
        u_norm = F.normalize(u, p=2, dim=-1)
        text_u_norm = F.normalize(text_u, p=2, dim=-1)
        
        # Compute similarity matrix (batch_size x batch_size)
        # similarity[i,j] = cosine_similarity(u[i], text_u[j])
        similarity_matrix = torch.matmul(u_norm, text_u_norm.t())
        
        # Option 1: Contrastive loss (like CLIP)
        # Diagonal elements are positive pairs, off-diagonal are negatives
        labels = torch.arange(batch_size, device=u.device)
        temperature = 0.07  # температурный параметр (можно сделать learnable)
        
        # Cross-entropy loss в обе стороны
        loss_i2t = F.cross_entropy(similarity_matrix / temperature, labels)
        loss_t2i = F.cross_entropy(similarity_matrix.t() / temperature, labels)
        contrastive_loss = (loss_i2t + loss_t2i) / 2
        
        # Option 2: Triplet loss с margin
        margin = 0.2 
        triplet_loss = 0
        
        for i in range(batch_size):
            # Positive: u[i] и text_u[i]
            pos_sim = similarity_matrix[i, i]
            
            # Negatives: u[i] и все text_u[j] где j != i
            neg_sims = torch.cat([similarity_matrix[i, :i], similarity_matrix[i, i+1:]])
            
            # Triplet loss с hard negative mining
            hardest_negative = neg_sims.max()
            loss_i = torch.relu(margin - pos_sim + hardest_negative)
            triplet_loss += loss_i
        
        triplet_loss = triplet_loss / batch_size
        
        # Можно комбинировать с MSE loss
        u_norm = F.normalize(u, p=2, dim=-1)
        text_u_norm = F.normalize(text_u, p=2, dim=-1)
        mse_align = F.smooth_l1_loss(u_norm, text_u_norm, reduction="sum") / batch_size
        # mse_align = F.l1_loss(text_u, u, reduction="sum") / batch_size
        
        # loss = contrastive_loss  # Option 1
        # loss = triplet_loss      # Option 2
        loss = contrastive_loss +  mse_align
        
        return TIEDModelOutput(
            encoded_latents=None,
            loss=loss,
            recon_loss=triplet_loss,
            kld=contrastive_loss,
            mse_align=mse_align,
        )
    
    def forward(self, input_ids=None, attention_mask=None, input_images=None, **kwargs):

        if input_ids is None:
            raise ValueError("input_ids must be provided")
        
        # if self.config.train_vae_only and input_images is not None:
        #     return self.vaes_forward(input_images)

        if self.config.train_vae_only and input_images is not None:
            return self.text_encoder_forward(input_images, input_ids, attention_mask)

        # Encode text
        text_features = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        
        # Pool text features
        pooled_features = self.pooler(text_features)
        text_u = self.visual_tokens_projector(pooled_features)
        mu = self.inner_vae.fc_mu(text_u)
        logvar = self.inner_vae.fc_logvar(text_u)
        text_z = self.inner_vae.reparameterize(mu, logvar)
        encoded_latents = self.inner_vae.decode(text_z)

        loss = None
        if input_images is not None:
            true_latents = self.get_diffused_latents(input_images)
            u, recon, mu, logvar, z = self.inner_vae(true_latents)
            loss, metrics = self.get_loss(text_u, u, recon, mu, logvar, true_latents, input_images)
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

