from tied import TIEDModel, TIEDModelConfig
from transformers import AutoConfig, LlamaConfig, AutoTokenizer
from transformers import TrainingArguments
from diffusers import AutoencoderKL
from tied.data_processing import TIEDDataset
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate
import torch
import json
import random
import os
from datasets import load_dataset
import torchvision.transforms as transforms
import argparse
from tied.training import TIEDTrainer

def safe_collate(batch):
    batch = [x for x in batch if isinstance(x, dict) and "pixel_values" in x]
    if len(batch) == 0:
        raise RuntimeError("All examples in batch missing 'pixel_values'")

    return {
        "input_ids": torch.stack([x["input_ids"] for x in batch]),
        "attention_mask": torch.stack([x["attention_mask"] for x in batch]),
        "pixel_values": torch.stack([x["pixel_values"].squeeze(0) for x in batch])  # remove unsqueeze(0)
    }



def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Load VAE model
    vae = AutoencoderKL.from_pretrained(args.vae_model)

    # Define decoder configuration
    decoder_config = LlamaConfig(
        vocab_size=1,
        hidden_size=args.decoder_hidden_size,
        num_hidden_layers=args.decoder_num_hidden_layers,
        num_attention_heads=args.decoder_num_attention_heads,
        intermediate_size=args.decoder_intermediate_size,
        max_position_embeddings=args.decoder_max_position_embeddings,
        rope_theta=args.decoder_rope_theta
    )

    # Load text encoder configuration
    text_encoder_config = AutoConfig.from_pretrained(args.text_encoder_model).to_dict()

    tokenizer = AutoTokenizer.from_pretrained(args.text_encoder_model)
    # Create TIED model configuration
    config = TIEDModelConfig(
        text_encoder_model=args.text_encoder_model,
        text_encoder_config=text_encoder_config,
        decoder_config=decoder_config.to_dict(),
        vae_model=args.vae_model,
        vae_config=vae.config,
        vocab_size=1,
        image_size=args.image_size,
        hidden_size=args.hidden_size,
        z_step=args.z_step,
        text_prompt_pooling_type=args.text_prompt_pooling_type,
        projector_hidden_act=args.projector_hidden_act,
        reduction=args.reduction,
    )

    # Initialize the TIED model
    model = TIEDModel(config).to(device)

    # Load dataset
    try:
        dataset = json.load(open(args.train_data, 'r'))
    except Exception as e:
        dataset = load_dataset(args.train_data, split='train')
        dataset = list(dataset)

    random.seed(42)
    random.shuffle(dataset)

    # Define image transformations
    image_transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    train_dataset = TIEDDataset(
        data=dataset,
        image_transform=image_transform,
        tokenizer=tokenizer,
        max_length=args.max_length,
    )
    # Create DataLoader
    training_args = TrainingArguments(
        remove_unused_columns=False,
        output_dir=args.save_path,
        per_device_train_batch_size=args.batch_size,
        num_train_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        evaluation_strategy="no",  #  "steps"/"epoch"
        save_total_limit=args.save_total_limit,
        fp16=args.fp16,
    )

    trainer = TIEDTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        data_collator=safe_collate
    )

    # Start training
    trainer.train()
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", type=str, help="Path to training data file", default= "dataset/dataset.json")
    parser.add_argument("--save_path", type=str, help="Directory to save the model", default="models")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for training")
    parser.add_argument("--num_epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate for the optimizer")
    parser.add_argument("--max_length", type=int, default=128, help="Maximum length of text sequences")
    parser.add_argument("--image_size", type=int, default=128, help="Size of input images")
    parser.add_argument("--text_encoder_model", type=str, default="microsoft/deberta-v3-base", help="Pretrained text encoder model")
    parser.add_argument("--vae_model", type=str, default="stabilityai/sdxl-vae", help="Pretrained VAE model")
    parser.add_argument("--hidden_size", type=int, default=1024, help="Hidden size for the decoder")
    parser.add_argument("--z_step", type=int, default=32, help="Z step for decoder")
    parser.add_argument("--text_prompt_pooling_type", type=str, default="first", help="Pooling type for text prompts")
    parser.add_argument("--projector_hidden_act", type=str, default="gelu", help="Activation function for projectors")
    parser.add_argument("--reduction", type=str, default="mean", help="Reduction method for loss calculation")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for DataLoader")
    parser.add_argument("--decoder_hidden_size", type=int, default=512, help="Hidden size for the decoder")
    parser.add_argument("--decoder_num_hidden_layers", type=int, default=6, help="Number of hidden layers in the decoder")
    parser.add_argument("--decoder_num_attention_heads", type=int, default=8, help="Number of attention heads in the decoder")
    parser.add_argument("--decoder_intermediate_size", type=int, default=1024, help="Intermediate size for the decoder")
    parser.add_argument("--decoder_max_position_embeddings", type=int, default=32, help="Max position embeddings for the decoder")
    parser.add_argument("--decoder_rope_theta", type=int, default=4000, help="RoPE theta for the decoder")
    parser.add_argument("--fp16", type=bool, default=False, help="Use mixed precision training if available")
    parser.add_argument("--logging_steps", type=int, default=1, help="Number of steps between logging")
    parser.add_argument("--save_steps", type=int, default=1000, help="Number of steps between saving checkpoints")
    parser.add_argument("--eval_steps", type=int, default=1000, help="Number of steps between evaluations")
    parser.add_argument("--save_total_limit", type=int, default=2, help="Maximum number of checkpoints to keep")
    args = parser.parse_args()

    # Ensure save path exists
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    # Start the training process
    main(args)


