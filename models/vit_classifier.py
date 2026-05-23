"""ViT Fine-Tuning for Image Classification"""
import torch
from transformers import ViTForImageClassification, ViTImageProcessor, TrainingArguments, Trainer
from datasets import load_dataset
import numpy as np
import evaluate


class ViTClassifier:
    def __init__(self, model_name: str = "google/vit-base-patch16-224", num_labels: int = 10):
        self.processor = ViTImageProcessor.from_pretrained(model_name)
        self.model     = ViTForImageClassification.from_pretrained(
            model_name, num_labels=num_labels, ignore_mismatched_sizes=True
        )

    def preprocess(self, examples):
        inputs = self.processor(images=examples["img"], return_tensors="pt")
        inputs["labels"] = examples["label"]
        return inputs

    def predict(self, image) -> dict:
        self.model.eval()
        with torch.no_grad():
            inputs = self.processor(images=image, return_tensors="pt")
            logits = self.model(**inputs).logits
            probs  = torch.softmax(logits, dim=-1).squeeze()
        top5_idx = probs.argsort(descending=True)[:5]
        return {self.model.config.id2label[i.item()]: round(probs[i].item(), 4) for i in top5_idx}

    def fine_tune(self, dataset_name: str = "cifar10", output_dir: str = "./vit-finetuned"):
        ds      = load_dataset(dataset_name)
        ds      = ds.map(self.preprocess, batched=True, remove_columns=ds["train"].column_names)
        metric  = evaluate.load("accuracy")
        trainer = Trainer(
            model=self.model,
            args=TrainingArguments(output_dir=output_dir, num_train_epochs=3,
                                   per_device_train_batch_size=32, evaluation_strategy="epoch",
                                   remove_unused_columns=False),
            train_dataset=ds["train"].select(range(5000)),
            eval_dataset=ds["test"].select(range(1000)),
            compute_metrics=lambda p: metric.compute(
                predictions=np.argmax(p.predictions, axis=1), references=p.label_ids
            ),
        )
        trainer.train()
        print(f"Model saved to {output_dir}")
