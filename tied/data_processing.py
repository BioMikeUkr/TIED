from torch.utils.data import Dataset
from PIL import Image
from transformers import PreTrainedTokenizer
import requests
import random
from io import BytesIO
import os

class TIEDDataset(Dataset):
    def __init__(
        self,
        data: list,
        tokenizer: PreTrainedTokenizer,
        image_transform,
        max_length: int = 128
    ):
        """
        Args:
            data (list): List of dictionaries [{"prompt": ..., "image": ...}, ...]
                         where "image" is a local path or URL.
            tokenizer (PreTrainedTokenizer): Tokenizer for processing the text prompt.
            image_transform: torchvision transform applied to the image (e.g., resize, normalize).
            max_length (int): Maximum token sequence length for the prompt.
        """
        self.data = data
        self.tokenizer = tokenizer
        self.image_transform = image_transform
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def _load_image(self, path_or_url: str) -> Image.Image:
        if path_or_url.startswith("http://") or path_or_url.startswith("https://"):
            response = requests.get(path_or_url)
            image = Image.open(BytesIO(response.content)).convert("RGB")
        else:
            image = Image.open(path_or_url).convert("RGB")
        return image

    def __getitem__(self, idx):

        entry = self.data[idx]
        prompt = entry["prompt"]
        actions = ["none", "upper", "lower"]
        modifications = ["none", "add_space", "add_tab", "add_newline"]
        cuts = ["none", "cut_05", "cut_025", "random_cut"]
        cut = random.choice(cuts)
        action = random.choice(actions)
        modification = random.choice(modifications)
        if action == "none":
            prompt = prompt
        elif action == "upper":
            prompt = prompt.upper()
        elif action == "lower":
            prompt = prompt.lower()
        if modification == "none":
            prompt = prompt
        elif modification == "add_space":
            prompt = " " + prompt + " "
        elif modification == "add_tab":
            prompt = "\t" + prompt + "\t"
        elif modification == "add_newline":
            prompt = "\n" + prompt + "\n"
        if cut == "cut_05":
            prompt = prompt[:len(prompt) // 2]
        elif cut == "cut_025":
            prompt = prompt[:len(prompt) // 4]
        elif cut == "random_cut":
            cut_length = random.randint(1, len(prompt))
            start_index = random.randint(0, len(prompt) - cut_length)
            prompt = prompt[:start_index] + prompt[start_index + cut_length:]
        
        image = self._load_image(entry["image"])
        pixel_values = self.image_transform(image)

        encoded = self.tokenizer(
            prompt,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        result = {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "pixel_values": pixel_values.unsqueeze(0)
        }
        return result
