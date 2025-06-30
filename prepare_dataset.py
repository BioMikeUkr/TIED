from datasets import load_dataset
import json

dataset = load_dataset("Artificio/WikiArt", split="train")

data = []

for item in dataset:
    artist = item.get("artist", "Unknown artist")
    title = item.get("title", "Untitled")
    style = item.get("style", "Unknown style")

    prompt = f"A painting titled '{title}' in the style of {style} by {artist}."
    image_url = item["image"]

    data.append({
        "prompt": prompt,
        "image": image_url
    })

with open("wikiart_dataset.json", "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

