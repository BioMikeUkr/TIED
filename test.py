from tied import TIEDModel, TIEDModelConfig
from transformers import AutoConfig, LlamaConfig
from diffusers import AutoencoderKL
from transformers import AutoTokenizer

# vae = AutoencoderKL.from_pretrained("stabilityai/sdxl-vae")

# decoder_config = LlamaConfig(
#     vocab_size=1,
#     hidden_size=128,
#     num_hidden_layers=6,
#     num_attention_heads=8,
#     intermediate_size=256,
#     max_position_embeddings=64,
#     rope_theta=4000
# )

# config = TIEDModelConfig(
#     text_encoder_model="microsoft/deberta-v3-base",
#     text_encoder_config=AutoConfig.from_pretrained("microsoft/deberta-v3-base").to_dict(),
#     decoder_config=decoder_config.to_dict(),
#     vae_model="stabilityai/sdxl-vae",
#     vae_config=vae.config,
#     vocab_size=1,
# )

model = TIEDModel.from_pretrained("models/final_model")
# model = TIEDModel(config)
# model.save_pretrained("my_model_dir")
token = ""
tokenizer = AutoTokenizer.from_pretrained("models/final_model")
tokenizer.push_to_hub("BioMike/TIDE-deberta-v3-small-sdxl-vae", token=token)
model.push_to_hub("BioMike/TIDE-deberta-v3-small-sdxl-vae", token=token)

model = TIEDModel.from_pretrained("BioMike/TIDE-deberta-v3-small-sdxl-vae")
