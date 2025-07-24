from transformers import Trainer
from typing import Dict, Union, Any
import os
import torch

class TIEDTrainer(Trainer):
    def compute_loss(
        self,
        model: torch.nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        return_outputs: bool = False
    ):
        input_ids = inputs.get("input_ids")
        attention_mask = inputs.get("attention_mask")
        pixel_values = inputs.get("pixel_values")
        
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            input_images=pixel_values,
        )

        loss = outputs.loss

        return (loss, outputs) if return_outputs else loss
    
    def training_step(
        self,
        model: torch.nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        *args,
        **kwargs
    ):
        # try:
            model.train()
            loss = self.compute_loss(model, inputs)
            self.accelerator.backward(loss, **kwargs)
            return loss.detach()
        # except Exception as e:
        #     print(f"Skipping iteration due to error: {e}")
        #     model.zero_grad(set_to_none=True)
        #     torch.cuda.empty_cache()
        #     return torch.tensor(0.0, requires_grad=True).to(model.device)
            

    # def _save_checkpoint(self, model, step=None):
    #     checkpoint_dir = f"checkpoint-{step}" if step else "final_model"
    #     output_dir = os.path.join(self.args.output_dir, checkpoint_dir)
    #     os.makedirs(output_dir, exist_ok=True)
    #     model.save_pretrained(output_dir)
    #     if self.tokenizer is not None:
    #         self.tokenizer.save_pretrained(output_dir)
    #     print(f"Checkpoint saved to {output_dir}")
