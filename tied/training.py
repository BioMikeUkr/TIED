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
    others_lr: Optional[float] = field(default=None)


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
            self.log(
                {
                    "recon_loss": outputs.recon_loss.item(),
                    "kld": outputs.kld.item(),
                    "mse_align": outputs.mse_align.item(),
                },
            )

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

        decay_params = get_parameter_names(model, ALL_LAYERNORM_LAYERS)
        decay_params = [n for n in decay_params if "bias" not in n]

        def is_in(n, key): return key in n

        param_groups = []

        if args.text_encoder_lr is not None:
            text_params = [n for n, _ in model.named_parameters() if is_in(n, "text_encoder")]
            param_groups += [
                {
                    "params": [p for n, p in model.named_parameters()
                            if n in decay_params and is_in(n, "text_encoder") and p.requires_grad],
                    "weight_decay": args.weight_decay,
                    "lr": args.text_encoder_lr,
                },
                {
                    "params": [p for n, p in model.named_parameters()
                            if n not in decay_params and is_in(n, "text_encoder") and p.requires_grad],
                    "weight_decay": 0.0,
                    "lr": args.text_encoder_lr,
                }
            ]

        if args.inner_vae_lr is not None:
            vae_params = [n for n, _ in model.named_parameters() if is_in(n, "inner_vae")]
            param_groups += [
                {
                    "params": [p for n, p in model.named_parameters()
                            if n in decay_params and is_in(n, "inner_vae") and p.requires_grad],
                    "weight_decay": args.weight_decay,
                    "lr": args.inner_vae_lr,
                },
                {
                    "params": [p for n, p in model.named_parameters()
                            if n not in decay_params and is_in(n, "inner_vae") and p.requires_grad],
                    "weight_decay": 0.0,
                    "lr": args.inner_vae_lr,
                }
            ]

        # Remaining parameters (others)
        if args.others_lr is not None:
            def is_other(n):
                return not is_in(n, "text_encoder") and not is_in(n, "inner_vae") and not is_in(n, "vae")

            param_groups += [
                {
                    "params": [p for n, p in model.named_parameters()
                            if n in decay_params and is_other(n) and p.requires_grad],
                    "weight_decay": args.weight_decay,
                    "lr": args.others_lr,
                },
                {
                    "params": [p for n, p in model.named_parameters()
                            if n not in decay_params and is_other(n) and p.requires_grad],
                    "weight_decay": 0.0,
                    "lr": args.others_lr,
                }
            ]

        optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(args)
        self.optimizer = optimizer_cls(param_groups, **optimizer_kwargs)
        return self.optimizer