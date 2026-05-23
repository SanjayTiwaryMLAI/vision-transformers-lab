"""CLIP Zero-Shot Image Classification"""
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import requests
from io import BytesIO


class CLIPZeroShot:
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32"):
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model     = CLIPModel.from_pretrained(model_name)
        self.model.eval()

    def load_image(self, source: str) -> Image.Image:
        if source.startswith("http"):
            return Image.open(BytesIO(requests.get(source).content)).convert("RGB")
        return Image.open(source).convert("RGB")

    def classify(self, image_source: str, labels: list[str]) -> dict:
        image   = self.load_image(image_source)
        prompts = [f"a photo of a {label}" for label in labels]
        inputs  = self.processor(text=prompts, images=image, return_tensors="pt", padding=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs   = outputs.logits_per_image.softmax(dim=1).squeeze()
        return {label: round(prob.item(), 4) for label, prob in zip(labels, probs)}

    def image_similarity(self, img1: str, img2: str) -> float:
        images = [self.load_image(img1), self.load_image(img2)]
        inputs = self.processor(images=images, return_tensors="pt", padding=True)
        with torch.no_grad():
            feats = self.model.get_image_features(**inputs)
            feats = feats / feats.norm(dim=-1, keepdim=True)
        return round(torch.cosine_similarity(feats[0:1], feats[1:2]).item(), 4)


if __name__ == "__main__":
    model  = CLIPZeroShot()
    result = model.classify(
        "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/1200px-Cat03.jpg",
        labels=["cat", "dog", "car", "airplane", "bird"]
    )
    print(result)
