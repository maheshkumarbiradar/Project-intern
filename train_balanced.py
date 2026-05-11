"""
train_balanced.py - Retrain with class weights to fix prediction bias
Run: python train_balanced.py
"""

import os
import time
import copy
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torchvision import datasets
from model import build_model
from dataset import get_dataloaders, get_transforms

ORIGINAL_CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]
DATA_DIR    = "data"
SAVE_PATH   = "best_model_balanced.pth"
BATCH_SIZE  = 32
EPOCHS      = 7
LR          = 1e-3
FREEZE_EPOCHS = 3


def compute_class_weights(data_dir):
    """Compute inverse-frequency weights so rare classes get higher loss penalty."""
    train_path = os.path.join(data_dir, "Training")
    dataset    = datasets.ImageFolder(train_path, transform=get_transforms("train"))
    
    class_counts = np.zeros(4)
    for _, label in dataset.samples:
        class_counts[label] += 1

    print("Training image counts per class:")
    for i, cls in enumerate(dataset.classes):
        print(f"  {cls:15s}: {int(class_counts[i])} images")

    # Weight = total / (num_classes * count)  — standard inverse frequency
    total   = class_counts.sum()
    weights = total / (len(class_counts) * class_counts)
    weights = weights / weights.sum() * len(class_counts)  # normalize

    print("\nClass weights (higher = more penalty for mistakes):")
    for i, cls in enumerate(dataset.classes):
        print(f"  {cls:15s}: {weights[i]:.3f}")

    return torch.FloatTensor(weights), dataset.classes


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}\n")

    # Compute class weights from training data
    class_weights, class_names = compute_class_weights(DATA_DIR)
    class_weights = class_weights.to(device)

    dataloaders, sizes = get_dataloaders(DATA_DIR, batch_size=BATCH_SIZE)
    print(f"\nTrain: {sizes['train']} | Val: {sizes['val']} | Test: {sizes['test']}\n")

    model = build_model(num_classes=4, freeze_backbone=True).to(device)

    # ★ KEY FIX: weighted loss — penalizes misclassifying rare classes more
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)
    scheduler = StepLR(optimizer, step_size=3, gamma=0.5)

    history    = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_acc   = 0.0
    best_weights = copy.deepcopy(model.state_dict())

    for epoch in range(1, EPOCHS + 1):
        print(f"Epoch {epoch}/{EPOCHS}  {'─'*40}")

        if epoch == FREEZE_EPOCHS + 1:
            print("  → Unfreezing backbone for fine-tuning")
            for param in model.parameters():
                param.requires_grad = True
            optimizer = Adam(model.parameters(), lr=LR * 0.1)
            scheduler = StepLR(optimizer, step_size=2, gamma=0.5)

        for phase in ["train", "val"]:
            model.train() if phase == "train" else model.eval()
            running_loss = running_correct = 0
            t0 = time.time()

            for images, labels in dataloaders[phase]:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(images)
                    loss    = criterion(outputs, labels)
                    preds   = outputs.argmax(dim=1)
                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                running_loss    += loss.item() * images.size(0)
                running_correct += (preds == labels).sum().item()

            epoch_loss = running_loss    / sizes[phase]
            epoch_acc  = running_correct / sizes[phase]
            elapsed    = time.time() - t0
            print(f"  {phase.upper():5s}  loss={epoch_loss:.4f}  acc={epoch_acc:.4f}  ({elapsed:.1f}s)")

            history[f"{phase}_loss"].append(epoch_loss)
            history[f"{phase}_acc"].append(epoch_acc)

            if phase == "val" and epoch_acc > best_acc:
                best_acc     = epoch_acc
                best_weights = copy.deepcopy(model.state_dict())
                torch.save(best_weights, SAVE_PATH)
                print(f"  ✔ Best model saved (val_acc={best_acc:.4f})")

        scheduler.step()
        print()

    print(f"Training complete. Best val accuracy: {best_acc:.4f}")
    print(f"Model saved → {SAVE_PATH}")
    print(f"\nNow test external accuracy with:")
    print(f'  python evaluate_external.py --data "C:\\project intern\\data1\\testing" --model {SAVE_PATH}')

    # Plot
    x = range(1, EPOCHS + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(x, history["train_acc"], label="Train", marker="o")
    ax1.plot(x, history["val_acc"],   label="Val",   marker="s")
    ax1.set_title("Accuracy"); ax1.legend(); ax1.grid(True)
    ax2.plot(x, history["train_loss"], label="Train", marker="o")
    ax2.plot(x, history["val_loss"],   label="Val",   marker="s")
    ax2.set_title("Loss"); ax2.legend(); ax2.grid(True)
    plt.tight_layout()
    plt.savefig("training_curves_balanced.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    train()
