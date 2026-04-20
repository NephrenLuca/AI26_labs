"""Load a classification checkpoint (.npz) and run inference / validation."""

from __future__ import annotations

import argparse
import csv
import os
from datetime import datetime
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image

try:
    from .nn import NeuralNetwork, init_backend
    from .train_classification import (
        confusion_matrix,
        evaluate_classification,
        load_image_dataset,
        load_model_state,
        save_confusion_matrix_image,
        train_val_split,
    )
except ImportError:
    from nn import NeuralNetwork, init_backend
    from train_classification import (
        confusion_matrix,
        evaluate_classification,
        load_image_dataset,
        load_model_state,
        save_confusion_matrix_image,
        train_val_split,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load a saved classification checkpoint (.npz) and evaluate or predict.",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        required=True,
        help="Path to a classification_best_*.npz checkpoint saved by train_classification.py",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "train"),
        help="Dataset root with class subfolders. Used both for labels and as the evaluation set.",
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="If set, predict a single image path instead of evaluating a dataset.",
    )
    parser.add_argument("--img-size", type=str, default="28,28")
    parser.add_argument(
        "--activation",
        type=str,
        default="relu",
        choices=["relu", "tanh", "sigmoid"],
        help="Hidden activation used during training (must match; not stored in ckpt).",
    )
    parser.add_argument(
        "--batchnorm",
        type=str,
        default="auto",
        choices=["auto", "on", "off"],
        help="Whether the ckpt was trained with BatchNorm. 'auto' detects from running stats.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="all",
        choices=["all", "val", "train"],
        help="Which subset to evaluate on: all images, or the same val/train split used during training.",
    )
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Must match training to reproduce split.")
    parser.add_argument("--seed", type=int, default=42, help="Must match training to reproduce split.")
    parser.add_argument("--batch-size", type=int, default=512, help="Inference batch size.")
    parser.add_argument("--gpu", type=int, default=4)
    parser.add_argument("--no-cuda", action="store_true")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.path.dirname(__file__),
        help="Directory to write confusion matrix and misclassified CSV.",
    )
    parser.add_argument(
        "--save-misclassified",
        action="store_true",
        help="If set, write a CSV listing misclassified samples (only in dataset-evaluation mode).",
    )
    parser.add_argument("--no-figure", action="store_true", help="Skip writing confusion matrix PNG.")
    parser.add_argument("--topk", type=int, default=3, help="Show top-k probabilities for --image mode.")
    return parser.parse_args()


def _parse_hw(s: str) -> Tuple[int, int]:
    h, w = s.split(",")
    return int(h), int(w)


def load_checkpoint(ckpt_path: str) -> dict:
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    data = np.load(ckpt_path, allow_pickle=True)
    required = {"weights", "biases", "bn_gamma", "bn_beta", "bn_running_mean", "bn_running_var"}
    missing = required - set(data.files)
    if missing:
        raise RuntimeError(f"Checkpoint missing keys: {sorted(missing)}")
    state = {k: [np.asarray(t, dtype=np.float32) for t in data[k]] for k in required}
    return state


def infer_layer_sizes(state: dict) -> List[int]:
    weights = state["weights"]
    if not weights:
        raise RuntimeError("Checkpoint has no weight matrices.")
    sizes = [int(weights[0].shape[0])]
    for w in weights:
        sizes.append(int(w.shape[1]))
    return sizes


def detect_batchnorm(state: dict) -> bool:
    for rv in state["bn_running_var"]:
        if not np.allclose(rv, 1.0, atol=1e-5):
            return True
    for rm in state["bn_running_mean"]:
        if not np.allclose(rm, 0.0, atol=1e-5):
            return True
    for g in state["bn_gamma"]:
        if not np.allclose(g, 1.0, atol=1e-5):
            return True
    for b in state["bn_beta"]:
        if not np.allclose(b, 0.0, atol=1e-5):
            return True
    return False


def build_model_from_ckpt(
    state: dict,
    num_classes: int,
    input_dim: int,
    xp,
    rng,
    activation: str,
    use_batchnorm: bool,
) -> NeuralNetwork:
    sizes = infer_layer_sizes(state)
    if sizes[0] != input_dim:
        raise RuntimeError(
            f"Checkpoint input dim {sizes[0]} does not match image input dim {input_dim}. "
            f"Adjust --img-size so that H*W == {sizes[0]}."
        )
    if sizes[-1] != num_classes:
        raise RuntimeError(
            f"Checkpoint output dim {sizes[-1]} != number of classes in data dir ({num_classes})."
        )
    model = NeuralNetwork(
        layer_sizes=sizes,
        hidden_activation=activation,
        output_activation="linear",
        xp=xp,
        rng=rng,
        use_batchnorm=use_batchnorm,
        dropout=0.0,
        optimizer="sgd",
    )
    load_model_state(model, state)
    return model


def _softmax_np(logits_np: np.ndarray) -> np.ndarray:
    shifted = logits_np - logits_np.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def _batched_forward(model: NeuralNetwork, x_np: np.ndarray, batch_size: int, xp, to_cpu) -> np.ndarray:
    n = x_np.shape[0]
    out_parts: List[np.ndarray] = []
    for start in range(0, n, batch_size):
        xb = xp.asarray(x_np[start : start + batch_size], dtype=xp.float32)
        logits = model.forward(xb, training=False)
        out_parts.append(np.asarray(to_cpu(logits)))
    return np.concatenate(out_parts, axis=0) if out_parts else np.zeros((0, 0), dtype=np.float32)


def _per_class_accuracy(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    acc = np.zeros(num_classes, dtype=np.float64)
    for c in range(num_classes):
        mask = y_true == c
        total = int(mask.sum())
        acc[c] = float((y_pred[mask] == c).sum()) / total if total > 0 else float("nan")
    return acc


def _load_single_image(path: str, image_size: Tuple[int, int]) -> np.ndarray:
    with Image.open(path) as img:
        gray = img.convert("L").resize((image_size[1], image_size[0]))
    arr = np.asarray(gray, dtype=np.float32) / 255.0
    return arr.reshape(1, -1)


def _collect_image_paths(data_dir: str, class_names: List[str]) -> List[str]:
    valid_ext = {".bmp", ".png", ".jpg", ".jpeg"}
    paths: List[str] = []
    for cls in class_names:
        cls_dir = os.path.join(data_dir, cls)
        for name in sorted(os.listdir(cls_dir)):
            if os.path.splitext(name)[1].lower() in valid_ext:
                paths.append(os.path.join(cls_dir, name))
    return paths


def _write_misclassified_csv(
    out_path: str,
    paths: List[str],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probs: np.ndarray,
    class_names: List[str],
) -> int:
    n_written = 0
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "true_label", "pred_label", "pred_confidence"])
        for path, t, p, prob in zip(paths, y_true, y_pred, probs):
            if t != p:
                writer.writerow([path, class_names[int(t)], class_names[int(p)], f"{float(prob[int(p)]):.4f}"])
                n_written += 1
    return n_written


def run_single_image(args: argparse.Namespace) -> None:
    h, w = _parse_hw(args.img_size)
    if not os.path.isdir(args.data_dir):
        raise FileNotFoundError(f"--data-dir not found (needed for class names): {args.data_dir}")
    class_names = sorted(
        [n for n in os.listdir(args.data_dir) if os.path.isdir(os.path.join(args.data_dir, n))]
    )
    if len(class_names) == 0:
        raise RuntimeError(f"No class subfolders found in {args.data_dir}.")

    use_cuda = not args.no_cuda
    xp, to_cpu, rng = init_backend(use_cuda=use_cuda, gpu_id=args.gpu, seed=args.seed)

    state = load_checkpoint(args.ckpt)
    if args.batchnorm == "auto":
        use_bn = detect_batchnorm(state)
    else:
        use_bn = args.batchnorm == "on"
    model = build_model_from_ckpt(
        state=state,
        num_classes=len(class_names),
        input_dim=h * w,
        xp=xp,
        rng=rng,
        activation=args.activation,
        use_batchnorm=use_bn,
    )

    x = _load_single_image(args.image, image_size=(h, w))
    logits = model.forward(xp.asarray(x, dtype=xp.float32), training=False)
    logits_np = np.asarray(to_cpu(logits))
    probs = _softmax_np(logits_np)[0]
    order = np.argsort(-probs)
    k = max(1, min(args.topk, len(class_names)))

    print(f"Image: {args.image}")
    print(f"Checkpoint: {args.ckpt}")
    print(f"Batchnorm: {use_bn} (resolved from '{args.batchnorm}')")
    print(f"Predicted class: {class_names[int(order[0])]} (p={probs[int(order[0])]:.4f})")
    print(f"Top-{k}:")
    for i in range(k):
        idx = int(order[i])
        print(f"  {i + 1}. {class_names[idx]:<12s} p={probs[idx]:.4f}")


def run_dataset_eval(args: argparse.Namespace) -> None:
    h, w = _parse_hw(args.img_size)
    x, y, class_names = load_image_dataset(args.data_dir, image_size=(h, w))

    if args.split == "all":
        x_eval, y_eval = x, y
        subset_desc = "all"
    else:
        x_train, y_train, x_val, y_val = train_val_split(x, y, args.val_ratio, args.seed)
        if args.split == "val":
            x_eval, y_eval = x_val, y_val
            subset_desc = f"val (val-ratio={args.val_ratio}, seed={args.seed})"
        else:
            x_eval, y_eval = x_train, y_train
            subset_desc = f"train (val-ratio={args.val_ratio}, seed={args.seed})"

    if len(y_eval) == 0:
        raise RuntimeError(f"Evaluation subset '{args.split}' is empty.")

    use_cuda = not args.no_cuda
    xp, to_cpu, rng = init_backend(use_cuda=use_cuda, gpu_id=args.gpu, seed=args.seed)

    state = load_checkpoint(args.ckpt)
    if args.batchnorm == "auto":
        use_bn = detect_batchnorm(state)
    else:
        use_bn = args.batchnorm == "on"

    model = build_model_from_ckpt(
        state=state,
        num_classes=len(class_names),
        input_dim=h * w,
        xp=xp,
        rng=rng,
        activation=args.activation,
        use_batchnorm=use_bn,
    )

    logits_np = _batched_forward(model, x_eval, args.batch_size, xp=xp, to_cpu=to_cpu)
    probs = _softmax_np(logits_np)
    y_pred = probs.argmax(axis=1).astype(np.int64)
    y_true = y_eval.astype(np.int64)

    overall_acc = float((y_pred == y_true).mean())
    per_class = _per_class_accuracy(y_true, y_pred, num_classes=len(class_names))
    cm = confusion_matrix(y_true, y_pred, num_classes=len(class_names))

    print(f"Checkpoint : {args.ckpt}")
    print(f"Data dir   : {args.data_dir}")
    print(f"Subset     : {subset_desc}")
    print(f"Num samples: {len(y_true)}")
    print(f"Batchnorm  : {use_bn} (resolved from '{args.batchnorm}')")
    print(f"Activation : {args.activation}")
    print(f"Layer sizes: {infer_layer_sizes(state)}")
    print(f"Overall accuracy: {overall_acc:.4f}")
    print("Per-class accuracy:")
    for cls_name, acc in zip(class_names, per_class):
        if np.isnan(acc):
            print(f"  {cls_name:<12s}: (no samples)")
        else:
            print(f"  {cls_name:<12s}: {acc:.4f}")

    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if not args.no_figure:
        fig_path = os.path.join(args.output_dir, f"classification_infer_confusion_{timestamp}.png")
        save_confusion_matrix_image(cm, class_names, overall_acc, fig_path)
        print(f"Saved confusion matrix figure: {fig_path}")

    if args.save_misclassified:
        paths: Optional[List[str]] = _collect_image_paths(args.data_dir, class_names)
        if paths is not None and len(paths) == len(y) and args.split == "all":
            target_paths = paths
            target_true, target_pred, target_probs = y_true, y_pred, probs
        else:
            print(
                "Note: --save-misclassified with --split != 'all' is not supported "
                "(sample order cannot be reproduced). Skipping CSV."
            )
            target_paths = None

        if target_paths is not None:
            csv_path = os.path.join(args.output_dir, f"classification_infer_misclassified_{timestamp}.csv")
            n = _write_misclassified_csv(csv_path, target_paths, target_true, target_pred, target_probs, class_names)
            print(f"Saved misclassified list ({n} samples): {csv_path}")


def main() -> None:
    args = parse_args()
    if args.image is not None:
        run_single_image(args)
    else:
        run_dataset_eval(args)


if __name__ == "__main__":
    main()
