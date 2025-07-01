from datasets import load_dataset
from PIL import Image
from tqdm import tqdm
import os
import json

output_dir = "artbench"
os.makedirs(output_dir, exist_ok=True)

dataset = load_dataset("alfredplpl/artbench-pd-256x256", split="train")
formatted_data = []

for idx, item in tqdm(enumerate(dataset)):
    prompt = item.get("caption", "Untitled")

    image = item["image"]
    filename = f"image_{idx+1}.jpg"
    image_path = os.path.join(output_dir, filename)
    image.save(image_path)

    formatted_data.append({
        "prompt": prompt,
        "image": os.path.abspath(image_path)
    })

with open("wikiart_dataset.json", "w", encoding="utf-8") as f:
    json.dump(formatted_data, f, ensure_ascii=False, indent=2)
