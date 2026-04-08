import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from models import HanziCNN


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict handwritten Hanzi class.")
    parser.add_argument("--image", type=str, required=True, help="Path to a bmp image.")
    parser.add_argument("--checkpoint", type=str, default="./checkpoints/best_model.pt", help="Model checkpoint.")
    parser.add_argument("--class-names", type=str, default="", help="Optional class_names.json path.")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="Inference device.")
    return parser.parse_args()


def load_model(checkpoint_path: Path, device: torch.device):
    ckpt = torch.load(checkpoint_path, map_location=device)
    class_names = ckpt.get("class_names", None)
    img_size = int(ckpt.get("img_size", 64))
    num_classes = len(class_names) if class_names else 12

    model = HanziCNN(num_classes=num_classes)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model, class_names, img_size


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")

    checkpoint_path = Path(args.checkpoint).resolve()
    model, class_names, img_size = load_model(checkpoint_path, device)

    if args.class_names:
        with open(args.class_names, "r", encoding="utf-8") as f:
            class_names = json.load(f)

    image_path = Path(args.image).resolve()
    image_np = plt.imread(image_path)
    if image_np.ndim == 3:
        image_np = image_np.mean(axis=2)
    image_np = image_np.astype(np.float32)
    if image_np.max() > 1.0:
        image_np = image_np / 255.0

    tensor = torch.from_numpy(image_np).unsqueeze(0).unsqueeze(0)
    tensor = F.interpolate(tensor, size=(img_size, img_size), mode="bilinear", align_corners=False)
    tensor = (tensor - 0.5) / 0.5
    tensor = tensor.to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)
        conf, pred_idx = probs.max(dim=1)

    pred_idx = int(pred_idx.item())
    confidence = float(conf.item())
    pred_label = class_names[pred_idx] if class_names is not None else str(pred_idx)

    print(f"Image: {image_path}")
    print(f"Predicted class index: {pred_idx}")
    print(f"Predicted class label: {pred_label}")
    print(f"Confidence: {confidence:.4f}")


if __name__ == "__main__":
    main()
