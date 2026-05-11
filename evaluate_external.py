r"""
evaluate_external.py - Evaluate model on any external dataset
Optional --normalize flag fixes brightness mismatch between datasets

Usage:
    python evaluate_external.py --data "C:\project intern\data2\Testing"
    python evaluate_external.py --data "C:\project intern\data3\Testing" --normalize
"""

import argparse
import csv
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image, ImageOps
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import classification_report, confusion_matrix
from model import load_trained_model
import os

ORIGINAL_CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]
TRAIN_BRIGHTNESS = 31.7   # average brightness of your training data
IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')


def is_image_file(path):
    return path.lower().endswith(IMAGE_EXTENSIONS)


class MRIDataset(Dataset):
    """
    Loads MRI images with optional brightness normalization.
    normalize=True: scales image brightness to match training data distribution
    normalize=False: standard preprocessing (use for same-source datasets)
    """
    def __init__(self, root_dir, transform=None, normalize_brightness=False):
        self.transform   = transform
        self.normalize   = normalize_brightness
        if not os.path.isdir(root_dir):
            raise FileNotFoundError(f"Dataset folder not found: {root_dir}")

        self.classes     = sorted([
            d for d in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, d))
        ])
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.samples      = []

        for cls in self.classes:
            cls_dir = os.path.join(root_dir, cls)
            for fname in sorted(os.listdir(cls_dir)):
                path = os.path.join(cls_dir, fname)
                if is_image_file(fname) and os.path.getsize(path) > 0:
                    self.samples.append(
                        (path, self.class_to_idx[cls])
                    )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")

        if self.normalize:
            img = self._normalize_brightness(img)

        if self.transform:
            img = self.transform(img)
        return img, label

    def _normalize_brightness(self, img):
        """
        Scale pixel values so the image brightness matches training data.
        Training brightness = 31.7
        If data3 brightness = 68.7, scale factor = 31.7 / 68.7 = 0.46
        This makes bright external images look like training images to the model.
        """
        arr          = np.array(img).astype(np.float32)
        current_mean = arr.mean()

        if current_mean > 1.0:   # avoid division by zero on black images
            scale = TRAIN_BRIGHTNESS / current_mean
            arr   = np.clip(arr * scale, 0, 255).astype(np.uint8)

        return Image.fromarray(arr)


def list_root_images(data_path):
    if not os.path.isdir(data_path):
        raise FileNotFoundError(f"Dataset folder not found: {data_path}")

    paths = []
    for fname in sorted(os.listdir(data_path)):
        path = os.path.join(data_path, fname)
        if os.path.isfile(path) and is_image_file(fname) and os.path.getsize(path) > 0:
            paths.append(path)
    return paths


def predict_unlabeled_images(model, image_paths, transform, device, normalize=False):
    rows = []
    print("No class subfolders found. Running prediction-only mode.\n")

    for path in image_paths:
        try:
            img = Image.open(path).convert("RGB")
            if normalize:
                arr = np.array(img).astype(np.float32)
                current_mean = arr.mean()
                if current_mean > 1.0:
                    arr = np.clip(arr * (TRAIN_BRIGHTNESS / current_mean), 0, 255).astype(np.uint8)
                    img = Image.fromarray(arr)

            tensor = transform(img).unsqueeze(0).to(device)
            with torch.no_grad():
                probs = torch.softmax(model(tensor), dim=1)[0].cpu().numpy()

            pred_idx = int(np.argmax(probs))
            row = {
                "file": os.path.basename(path),
                "predicted_class": ORIGINAL_CLASSES[pred_idx],
                "confidence": round(float(probs[pred_idx]) * 100, 2),
            }
            for idx, cls in enumerate(ORIGINAL_CLASSES):
                row[f"{cls}_prob"] = round(float(probs[idx]) * 100, 2)
            rows.append(row)
            print(f"  {row['file']:45s} -> {row['predicted_class']:12s} {row['confidence']:6.2f}%")
        except Exception as exc:
            print(f"  Skipped {os.path.basename(path)}: {exc}")

    if not rows:
        print("No readable images found for prediction.")
        return None

    output_path = "external_predictions.csv"
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nPredictions saved -> {output_path}\n")
    return rows


def evaluate_external(data_path, model_path="best_model.pth",
                      batch_size=32, normalize=False):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice    : {device}")
    print(f"Model     : {model_path}")
    print(f"Data      : {data_path}")
    print(f"Normalize : {'ON  - brightness correction applied' if normalize else 'OFF'}\n")

    model = load_trained_model(model_path, device, num_classes=4)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    dataset       = MRIDataset(data_path, transform=transform,
                               normalize_brightness=normalize)
    loader        = DataLoader(dataset, batch_size=batch_size,
                               shuffle=False, num_workers=0)
    found_classes = dataset.classes

    print(f"Classes found : {found_classes}")
    print(f"Total images  : {len(dataset)}\n")

    if len(dataset) == 0:
        root_images = list_root_images(data_path)
        if root_images:
            predict_unlabeled_images(model, root_images, transform, device,
                                     normalize=normalize)
            return None

        print("No images found.")
        print("For accuracy evaluation, use this folder structure:")
        for cls in ORIGINAL_CLASSES:
            print(f"  {data_path}/{cls}/image.jpg")
        return None

    # Show brightness info if normalizing
    if normalize:
        print("Brightness check (first 10 images per class):")
        for cls in found_classes:
            cls_dir = os.path.join(data_path, cls)
            imgs    = [img for img in sorted(os.listdir(cls_dir)) if is_image_file(img)][:10]
            means   = []
            for img in imgs:
                try:
                    arr = np.array(Image.open(os.path.join(cls_dir, img)).convert("L"))
                    means.append(arr.mean())
                except Exception:
                    continue
            print(f"  {cls:12s}: raw brightness={np.mean(means):.1f}  "
                  f"-> normalized to ~{TRAIN_BRIGHTNESS:.1f}")
        print()

    valid_model_indices = [
        ORIGINAL_CLASSES.index(cls)
        for cls in found_classes if cls in ORIGINAL_CLASSES
    ]
    local_to_model = {
        local_idx: ORIGINAL_CLASSES.index(cls)
        for local_idx, cls in enumerate(found_classes)
        if cls in ORIGINAL_CLASSES
    }

    # Inference
    all_preds  = []
    all_labels = []

    with torch.no_grad():
        for images, local_labels in loader:
            images = images.to(device)
            logits = model(images)

            if len(valid_model_indices) < 4:
                mask = torch.full((4,), float('-inf'))
                for idx in valid_model_indices:
                    mask[idx] = 0.0
                logits = logits + mask.to(device)

            preds = logits.argmax(dim=1).cpu().numpy()

            for i, local_lbl in enumerate(local_labels):
                local_lbl = local_lbl.item()
                if local_lbl not in local_to_model:
                    continue
                all_preds.append(preds[i])
                all_labels.append(local_to_model[local_lbl])

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)

    if len(all_labels) == 0:
        print("No evaluable labels found.")
        print(f"Class folder names must match: {', '.join(ORIGINAL_CLASSES)}")
        return None

    accuracy = (all_preds == all_labels).mean() * 100
    print(f"{'='*50}")
    print(f"  Overall Accuracy : {accuracy:.2f}%")
    print(f"{'='*50}\n")

    report_names = [ORIGINAL_CLASSES[i] for i in valid_model_indices]
    print("Classification Report:")
    print(classification_report(
        all_labels, all_preds,
        labels=valid_model_indices,
        target_names=report_names,
        zero_division=0
    ))

    cm = confusion_matrix(all_labels, all_preds, labels=valid_model_indices)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=report_names, yticklabels=report_names)
    plt.title(f"Confusion Matrix - External Dataset {'(normalized)' if normalize else ''}")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig("confusion_matrix_external.png", dpi=150)
    print("Confusion matrix saved -> confusion_matrix_external.png\n")
    plt.show()

    print("Per-Class Accuracy:")
    print(f"{'-'*40}")
    for idx, cls in zip(valid_model_indices, report_names):
        mask    = all_labels == idx
        cls_acc = (all_preds[mask] == all_labels[mask]).mean() * 100 if mask.sum() > 0 else 0
        count   = mask.sum()
        bar     = "=" * int(cls_acc // 5)
        print(f"  {cls:12s}: {cls_acc:6.2f}%  ({count} images)  {bar}")
    print(f"{'-'*40}\n")

    return accuracy


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",      required=True)
    parser.add_argument("--model",     default="best_model.pth")
    parser.add_argument("--normalize", action="store_true",
                        help="Normalize brightness to match training data (use for bright datasets)")
    args = parser.parse_args()
    evaluate_external(data_path=args.data, model_path=args.model,
                      normalize=args.normalize)
