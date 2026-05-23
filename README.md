# 🖼️ Vision Transformers Lab

ViT, CLIP, DINO, and Swin Transformer experiments — image classification, zero-shot recognition, self-supervised feature extraction, and transfer learning.

## 🚀 Models Covered
| Model | Task | Paper |
|-------|------|-------|
| ViT-B/16 | Image Classification | Dosovitskiy et al. 2020 |
| CLIP | Zero-Shot Recognition | Radford et al. 2021 |
| DINO | Self-Supervised Features | Caron et al. 2021 |
| Swin Transformer | Object Detection | Liu et al. 2021 |

## 📁 Structure
```
vision-transformers-lab/
├── models/
│   ├── vit_classifier.py     # ViT fine-tuning
│   ├── clip_zero_shot.py     # CLIP zero-shot classification
│   └── dino_features.py      # DINO feature extraction
├── requirements.txt
└── README.md
```

## ⚡ Quick Start
```python
from models.clip_zero_shot import CLIPZeroShot
model = CLIPZeroShot()
result = model.classify("cat.jpg", labels=["cat", "dog", "car", "airplane"])
print(result)  # {"cat": 0.92, "dog": 0.05, ...}
```
