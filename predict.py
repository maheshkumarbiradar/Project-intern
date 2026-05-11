"""
predict.py - Single image prediction using EfficientNetB0
Usage:
    python predict.py --image "C:\path\to\mri.jpg"
    python predict.py --image "C:\path\to\mri.jpg" --model best_model_efficientnet.pth
"""

import argparse
import torch
from PIL import Image
from dataset import get_transforms, CLASS_NAMES
from model import load_trained_model


def predict_image(image_path, model_path="best_model_efficientnet.pth"):
    """
    Run inference on a single MRI image using EfficientNetB0.

    Returns:
        dict with label, confidence, all_probs
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model     = load_trained_model(model_path, device)
    transform = get_transforms("test")

    image  = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs  = torch.softmax(logits, dim=1)[0]

    all_probs = {name: round(probs[i].item() * 100, 2)
                 for i, name in enumerate(CLASS_NAMES)}

    top_idx    = probs.argmax().item()
    top_label  = CLASS_NAMES[top_idx]
    confidence = round(probs[top_idx].item() * 100, 2)

    return {
        "label":       top_label,
        "class_index": top_idx,
        "confidence":  confidence,
        "all_probs":   all_probs,
    }


def main():
    parser = argparse.ArgumentParser(description="Brain Tumor Prediction - EfficientNetB0")
    parser.add_argument("--image", required=True, help="Path to MRI image")
    parser.add_argument("--model", default="best_model_efficientnet.pth",
                        help="Path to model checkpoint")
    args = parser.parse_args()

    result = predict_image(args.image, args.model)

    print(f"\n{'─'*45}")
    print(f"  Model           : EfficientNetB0")
    print(f"  Predicted Class : {result['label'].upper()}")
    print(f"  Confidence      : {result['confidence']}%")
    print(f"{'─'*45}")
    print("  Class Probabilities:")
    for cls, prob in sorted(result["all_probs"].items(),
                            key=lambda x: x[1], reverse=True):
        bar = "█" * int(prob // 5)
        print(f"    {cls:15s}: {prob:6.2f}%  {bar}")
    print(f"{'─'*45}\n")


if __name__ == "__main__":
    main()