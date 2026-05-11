"""
dataset.py - Data loading and augmentation for Brain Tumor MRI dataset
"""

import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

CLASS_NAMES   = ["glioma", "meningioma", "notumor", "pituitary"]
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def get_transforms(split="train"):
    if split == "train":
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.2),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])


def get_dataloaders(data_dir, batch_size=32, num_workers=0, pin_memory=False):
    """
    pin_memory=False on CPU (avoids warning)
    pin_memory=True  on GPU (faster data transfer)
    """
    splits = {
        "train": os.path.join(data_dir, "Training"),
        "val":   os.path.join(data_dir, "Validation"),
        "test":  os.path.join(data_dir, "Testing"),
    }

    if not os.path.isdir(splits["val"]):
        splits["val"] = splits["test"]

    datasets_dict = {
        split: datasets.ImageFolder(path, transform=get_transforms(split))
        for split, path in splits.items()
    }

    dataloaders = {
        "train": DataLoader(datasets_dict["train"], batch_size=batch_size,
                            shuffle=True,  num_workers=num_workers,
                            pin_memory=pin_memory),
        "val":   DataLoader(datasets_dict["val"],   batch_size=batch_size,
                            shuffle=False, num_workers=num_workers,
                            pin_memory=pin_memory),
        "test":  DataLoader(datasets_dict["test"],  batch_size=batch_size,
                            shuffle=False, num_workers=num_workers,
                            pin_memory=pin_memory),
    }

    sizes = {split: len(ds) for split, ds in datasets_dict.items()}
    return dataloaders, sizes