"""
train.py - ResNet18 Training for Brain Tumor Classification
Run: python train.py
"""

import os
import time
import copy
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt
from model import build_model
from dataset import get_dataloaders

DATA_DIR      = "data"
SAVE_PATH     = "best_model.pth"
BATCH_SIZE    = 32
EPOCHS        = 7
LEARNING_RATE = 1e-3
NUM_CLASSES   = 4


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pin    = device.type == "cuda"

    print(f"{'='*50}")
    print(f"  Model  : ResNet18")
    print(f"  Device : {device}")
    print(f"  Epochs : {EPOCHS}")
    print(f"{'='*50}\n")

    dataloaders, sizes = get_dataloaders(DATA_DIR, batch_size=BATCH_SIZE, pin_memory=pin)
    print(f"Train : {sizes['train']} | Val : {sizes['val']} | Test : {sizes['test']}\n")

    history      = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_acc     = 0.0
    best_weights = None

    # ── PHASE 1: Train head only (3 epochs) ───────────────────────────────────
    print("PHASE 1: Classifier head only (backbone frozen)")
    print("─" * 50)
    model      = build_model(num_classes=NUM_CLASSES, freeze_backbone=True).to(device)
    criterion  = nn.CrossEntropyLoss()
    optimizer1 = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)
    scheduler1 = StepLR(optimizer1, step_size=2, gamma=0.5)

    try:
        best_acc, best_weights = _run_epochs(
            model, dataloaders, sizes, criterion, optimizer1, scheduler1,
            device, epochs=3, start=0, best_acc=best_acc,
            best_weights=best_weights, history=history
        )
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted! Saving best model so far...")
        if best_weights: torch.save(best_weights, SAVE_PATH)
        plot_history(history); return

    # ── PHASE 2: Fine-tune full network (remaining epochs) ────────────────────
    print("\nPHASE 2: Full network fine-tuning")
    print("─" * 50)
    for param in model.parameters():
        param.requires_grad = True

    optimizer2 = Adam(model.parameters(), lr=LEARNING_RATE * 0.1)
    scheduler2 = StepLR(optimizer2, step_size=2, gamma=0.5)

    try:
        best_acc, best_weights = _run_epochs(
            model, dataloaders, sizes, criterion, optimizer2, scheduler2,
            device, epochs=EPOCHS - 3, start=3, best_acc=best_acc,
            best_weights=best_weights, history=history
        )
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted! Saving best model so far...")

    if best_weights:
        torch.save(best_weights, SAVE_PATH)

    print(f"\n{'='*50}")
    print(f"  Training complete!")
    print(f"  Best Val Accuracy : {best_acc*100:.2f}%")
    print(f"  Model saved       : {SAVE_PATH}")
    print(f"{'='*50}")
    plot_history(history)


def _run_epochs(model, dataloaders, sizes, criterion, optimizer, scheduler,
                device, epochs, start, best_acc, best_weights, history):
    for epoch in range(1, epochs + 1):
        print(f"\nEpoch {start + epoch}  {'─'*40}")
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
            print(f"  {phase.upper():5s}  loss={epoch_loss:.4f}  "
                  f"acc={epoch_acc:.4f}  ({time.time()-t0:.1f}s)")

            history[f"{phase}_loss"].append(epoch_loss)
            history[f"{phase}_acc"].append(epoch_acc)

            if phase == "val" and epoch_acc > best_acc:
                best_acc     = epoch_acc
                best_weights = model.state_dict().copy()
                torch.save(best_weights, SAVE_PATH)
                print(f"  ✔ Best model saved  (val_acc={best_acc:.4f})")

        scheduler.step()
    return best_acc, best_weights


def plot_history(history):
    n = len(history["train_acc"])
    if n == 0: return
    x = range(1, n + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(x, history["train_acc"], label="Train", marker="o", color="navy")
    ax1.plot(x, history["val_acc"],   label="Val",   marker="s", color="orange")
    ax1.set_title("ResNet18 — Accuracy"); ax1.legend(); ax1.grid(True, alpha=0.3)
    ax2.plot(x, history["train_loss"], label="Train", marker="o", color="navy")
    ax2.plot(x, history["val_loss"],   label="Val",   marker="s", color="orange")
    ax2.set_title("ResNet18 — Loss"); ax2.legend(); ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("training_curves.png", dpi=150)
    print("Training curves saved → training_curves.png")
    plt.show()


if __name__ == "__main__":
    train() 