import argparse
import os
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Inference with manual-CNN model.")
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default="./checkpoints/best_model.npz")
    parser.add_argument("--meta", type=str, default="./checkpoints/meta.json")
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "gpu"],
        default="gpu",
        help="Backend: 'gpu' uses cupy, 'cpu' uses numpy.",
    )
    parser.add_argument("--gpu-id", type=int, default=4)
    return parser.parse_args()


# Backend must be selected before ``backend`` / ``model`` / ``mynn`` get
# imported.
_ARGS = parse_args()
os.environ["UNPYTORCHED_DEVICE"] = _ARGS.device

import json

import matplotlib.pyplot as plt

from backend import xp as cp, is_gpu, set_gpu_device
from model import HanziCNN


def resize_nearest(img: cp.ndarray, size: int) -> cp.ndarray:
    h, w = img.shape
    ys = cp.linspace(0, h - 1, size).astype(cp.int32)
    xs = cp.linspace(0, w - 1, size).astype(cp.int32)
    return img[ys][:, xs]


def main():
    args = _ARGS
    if is_gpu:
        set_gpu_device(args.gpu_id)
        print(f"[device] Using cupy on GPU{args.gpu_id}")
    else:
        print("[device] Using numpy on CPU")

    with open(Path(args.meta).resolve(), "r", encoding="utf-8") as f:
        meta = json.load(f)
    class_names = meta.get("class_names", [str(i) for i in range(12)])
    img_size = int(meta.get("img_size", 64))

    model = HanziCNN(num_classes=len(class_names))
    weights = cp.load(Path(args.checkpoint).resolve(), allow_pickle=False)
    state = {k: weights[k] for k in weights.files}
    model.load_state_dict(state)
    model.eval()

    img = plt.imread(Path(args.image).resolve())
    img = cp.asarray(img)
    if img.ndim == 3:
        img = img.mean(axis=2)
    img = img.astype(cp.float32)
    if img.max() > 1.0:
        img = img / 255.0

    img = resize_nearest(img, img_size)
    x = img[None, None, :, :].astype(cp.float32)
    x = (x - 0.5) / 0.5

    logits = model.forward(x)
    shifted = logits - cp.max(logits, axis=1, keepdims=True)
    probs = cp.exp(shifted) / cp.sum(cp.exp(shifted), axis=1, keepdims=True)
    idx = int(cp.argmax(probs, axis=1)[0].item())
    conf = float(probs[0, idx].item())
    label = class_names[idx] if idx < len(class_names) else str(idx)
    print(f"Predicted index: {idx}")
    print(f"Predicted label: {label}")
    print(f"Confidence: {conf:.4f}")


if __name__ == "__main__":
    main()
