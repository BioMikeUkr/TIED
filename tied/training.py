from transformers import Trainer
from typing import Dict, Union, Any
from transformers import TrainingArguments
from transformers.trainer import get_parameter_names, ALL_LAYERNORM_LAYERS
from dataclasses import dataclass, field
from typing import Optional
import torch


@dataclass
class TrainingArguments(TrainingArguments):
    text_encoder_lr: Optional[float] = field(default=None)
    inner_vae_lr: Optional[float] = field(default=None)
    decoder_lr: Optional[float] = field(default=None)
    others_lr: Optional[float] = field(default=None)

    train_vae: bool = True
    train_text_encoder: bool = True
    train_decoder: bool = True
    train_vae_decoder: bool = False


class TIEDTrainer(Trainer):
    # ───────────────────── compute_loss ─────────────────────
    def compute_loss(
        self,
        model: torch.nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        return_outputs: bool = False,
    ):
        input_ids      = inputs.get("input_ids")
        attention_mask = inputs.get("attention_mask")
        pixel_values   = inputs.get("pixel_values")

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            input_images=pixel_values,
        )
        loss = outputs.loss
        return (loss, outputs) if return_outputs else loss

    # ───────────────────── training_step ─────────────────────
    def training_step(
        self,
        model: torch.nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        *args,
        **kwargs,
    ):
        model.train()
        loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
        self.accelerator.backward(loss, **kwargs)

        if self.state.global_step % self.args.logging_steps == 0:
            # self.log(
            #     {
            #         "recon_loss": outputs.recon_loss.item(),
            #         "kld": outputs.kld.item(),
            #         "mse_align": outputs.mse_align.item(),
            #     },
            # )
            pass

        return loss.detach()

    # ─────────────────── helper: grad‑norm ───────────────────
    @staticmethod
    def _grad_norm(model: torch.nn.Module) -> float:
        total = 0.0
        for p in model.parameters():
            if p.grad is not None:
                total += p.grad.data.norm(2).item() ** 2
        return total ** 0.5
    
    def create_optimizer(self):
        if self.optimizer is not None:
            return self.optimizer

        model = self.model
        args = self.args
        
        # Set requires_grad based on training flags - do this first and clearly
        # VAE parameters (except decoder if train_vae_decoder is True)
        for param in model.vae.parameters():
            param.requires_grad = False
        
        if args.train_vae_decoder:
            for param in model.vae.decoder.parameters():
                param.requires_grad = True
        
        # Inner VAE parameters
        for param in model.inner_vae.parameters():
            param.requires_grad = args.train_vae
        
        # Text encoder parameters
        for param in model.text_encoder.parameters():
            param.requires_grad = args.train_text_encoder
        
        # Decoder parameters (assuming this is model.decoder, not model.vae.decoder)
        for param in model.decoder.parameters():
            param.requires_grad = args.train_decoder

        # Get parameters that should have weight decay
        decay_params = get_parameter_names(model, ALL_LAYERNORM_LAYERS)
        decay_params = [n for n in decay_params if "bias" not in n]

        # More precise parameter matching functions
        def is_text_encoder(name):
            return "text_encoder" in name and "inner_vae" not in name and "vae.decoder" not in name
        
        def is_inner_vae(name):
            return "inner_vae" in name
        
        def is_decoder(name):
            # Match model.decoder but not model.vae.decoder
            return "decoder" in name and "vae.decoder" not in name and "inner_vae" not in name
        
        def is_vae_decoder(name):
            return "vae.decoder" in name
        
        def is_other(name):
            return not (is_text_encoder(name) or is_inner_vae(name) or is_decoder(name) or is_vae_decoder(name) or "vae" in name)

        param_groups = []

        # Text encoder parameters
        if args.text_encoder_lr is not None:
            text_encoder_decay = [p for n, p in model.named_parameters()
                                if n in decay_params and is_text_encoder(n) and p.requires_grad]
            text_encoder_no_decay = [p for n, p in model.named_parameters()
                                if n not in decay_params and is_text_encoder(n) and p.requires_grad]
            
            if text_encoder_decay:
                param_groups.append({
                    "params": text_encoder_decay,
                    "weight_decay": args.weight_decay,
                    "lr": args.text_encoder_lr,
                })
            
            if text_encoder_no_decay:
                param_groups.append({
                    "params": text_encoder_no_decay,
                    "weight_decay": 0.0,
                    "lr": args.text_encoder_lr,
                })

        # Inner VAE parameters
        if args.inner_vae_lr is not None and args.train_vae:
            inner_vae_decay = [p for n, p in model.named_parameters()
                            if n in decay_params and is_inner_vae(n) and p.requires_grad]
            inner_vae_no_decay = [p for n, p in model.named_parameters()
                                if n not in decay_params and is_inner_vae(n) and p.requires_grad]
            
            if inner_vae_decay:
                param_groups.append({
                    "params": inner_vae_decay,
                    "weight_decay": args.weight_decay,
                    "lr": args.inner_vae_lr,
                })
            
            if inner_vae_no_decay:
                param_groups.append({
                    "params": inner_vae_no_decay,
                    "weight_decay": 0.0,
                    "lr": args.inner_vae_lr,
                })

        # Decoder parameters
        if args.decoder_lr is not None and args.train_decoder:
            decoder_decay = [p for n, p in model.named_parameters()
                        if n in decay_params and is_decoder(n) and p.requires_grad]
            decoder_no_decay = [p for n, p in model.named_parameters()
                            if n not in decay_params and is_decoder(n) and p.requires_grad]
            
            if decoder_decay:
                param_groups.append({
                    "params": decoder_decay,
                    "weight_decay": args.weight_decay,
                    "lr": args.decoder_lr,
                })
            
            if decoder_no_decay:
                param_groups.append({
                    "params": decoder_no_decay,
                    "weight_decay": 0.0,
                    "lr": args.decoder_lr,
                })

        # VAE decoder parameters (if training VAE decoder separately)
        if args.train_vae_decoder:
            vae_decoder_lr = args.inner_vae_lr if args.inner_vae_lr is not None else args.learning_rate
            vae_decoder_decay = [p for n, p in model.named_parameters()
                            if n in decay_params and is_vae_decoder(n) and p.requires_grad]
            vae_decoder_no_decay = [p for n, p in model.named_parameters()
                                if n not in decay_params and is_vae_decoder(n) and p.requires_grad]
            
            if vae_decoder_decay:
                param_groups.append({
                    "params": vae_decoder_decay,
                    "weight_decay": args.weight_decay,
                    "lr": vae_decoder_lr,
                })
            
            if vae_decoder_no_decay:
                param_groups.append({
                    "params": vae_decoder_no_decay,
                    "weight_decay": 0.0,
                    "lr": vae_decoder_lr,
                })

        # Other parameters
        if args.others_lr is not None:
            other_decay = [p for n, p in model.named_parameters()
                        if n in decay_params and is_other(n) and p.requires_grad]
            other_no_decay = [p for n, p in model.named_parameters()
                            if n not in decay_params and is_other(n) and p.requires_grad]
            
            if other_decay:
                param_groups.append({
                    "params": other_decay,
                    "weight_decay": args.weight_decay,
                    "lr": args.others_lr,
                })
            
            if other_no_decay:
                param_groups.append({
                    "params": other_no_decay,
                    "weight_decay": 0.0,
                    "lr": args.others_lr,
                })

        # Debug: Print parameter group info
        print(f"Created {len(param_groups)} parameter groups:")
        for i, group in enumerate(param_groups):
            print(f"  Group {i}: {len(group['params'])} params, lr={group['lr']}, wd={group['weight_decay']}")

        # Verify no parameter appears in multiple groups
        all_params_in_groups = set()
        for group in param_groups:
            for param in group['params']:
                if param in all_params_in_groups:
                    raise ValueError("Parameter appears in multiple groups!")
                all_params_in_groups.add(param)

        optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(args)
        self.optimizer = optimizer_cls(param_groups, **optimizer_kwargs)
        return self.optimizer