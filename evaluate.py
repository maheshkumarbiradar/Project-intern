"""
evaluate.py - Full test set evaluation for ResNet18
Usage:
    python evaluate.py
    python evaluate.py --data data --model best_model.pth
"""

import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

from model import load_trained_model
from dataset import get_dataloaders, CLASS_NAMES


def evaluate(data_dir="data", model_path="best_model.pth", batch_size=32):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nModel  : ResNet18")
    print(f"Device : {device}")

    model = load_trained_model(model_path, device)
    dataloaders, sizes = get_dataloaders(data_dir, batch_size=batch_size)
    print(f"Test samples: {sizes['test']}\n")

    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in dataloaders["test"]:
            images  = images.to(device)
            outputs = model(images)
            preds   = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)

    accuracy = (all_preds == all_labels).mean() * 100
    print(f"{'='*50}")
    print(f"  ResNet18 Test Accuracy : {accuracy:.2f}%")
    print(f"{'='*50}\n")

    print("Classification Report:")
    print(classification_report(all_labels, all_preds,
                                target_names=CLASS_NAMES, zero_division=0))

    print("Per-Class Accuracy:")
    print(f"{'─'*40}")
    for i, cls in enumerate(CLASS_NAMES):
        mask    = all_labels == i
        cls_acc = (all_preds[mask] == all_labels[mask]).mean() * 100
        bar     = "█" * int(cls_acc // 5)
        print(f"  {cls:12s}: {cls_acc:6.2f}%  {bar}")
    print(f"{'─'*40}\n")

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.title("Confusion Matrix — ResNet18")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=150)
    print("Confusion matrix saved → confusion_matrix.png")
    plt.show()

    return accuracy


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",  default="data")
    parser.add_argument("--model", default="best_model.pth")
    args = parser.parse_args()
    evaluate(data_dir=args.data, model_path=args.model)