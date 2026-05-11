"""
model.py - ResNet18 Transfer Learning for Brain Tumor Classification
Classes: glioma, meningioma, notumor, pituitary
"""

import torch
import torch.nn as nn
from torchvision import models


def build_model(num_classes=4, freeze_backbone=True):
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    in_features = model.fc.in_features  # 512
    model.fc = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(p=0.3),
        nn.Linear(256, num_classes)
    )

    return model


def load_trained_model(checkpoint_path, device, num_classes=4):
    model = build_model(num_classes=num_classes, freeze_backbone=False)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()
    return model


if __name__ == "__main__":
    model = build_model()
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model          : ResNet18")
    print(f"Total params   : {total:,}")
    print(f"Trainable      : {trainable:,}")