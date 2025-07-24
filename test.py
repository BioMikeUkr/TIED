from tied import TIEDModel
from transformers import AutoTokenizer
import torch
import matplotlib.pyplot as plt

# Load model and tokenizer
model = TIEDModel.from_pretrained("models/checkpoint-30000")
tokenizer = AutoTokenizer.from_pretrained("models/checkpoint-30000")


prompt = """
The image shows a painting of a red house in the middle of a snowy field, with a horse cart in front of it.
"""
# Tokenize input
inputs = tokenizer("This is a test input", return_tensors="pt")
input_ids = inputs.input_ids
attention_mask = inputs.attention_mask

# Generate latents
with torch.no_grad():
    generated_latents = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        z_step = 32
    )  # shape: [B, T, C, H, W]

print("Generated Latents Shape:", generated_latents.shape)

# Take the last latent frame
last_latent = generated_latents[:, -1]  # shape: [B, C, H, W]
print("Last Latent Shape:", last_latent.shape)

# Decode
decoded = model.vae.decode(last_latent).sample  # shape: [B, 3, H, W]
decoded = (decoded / 2 + 0.5).clamp(0, 1)

# Convert to numpy image
img_np = decoded.squeeze().permute(1, 2, 0).cpu().numpy()
plt.imshow(img_np)
plt.axis('off')
plt.savefig("output_image.png", bbox_inches='tight', pad_inches=0)
# plt.show()
