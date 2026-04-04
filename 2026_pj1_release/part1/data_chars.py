"""
Load 12-class handwritten character images from a directory tree.

Expected layout (flexible class names)
----------------------------------------
::

    data_root/
      <class_a>/  *.png | *.jpg | *.jpeg | *.bmp
      <class_b>/
      ...

Class indices are assigned by **lexicographic order** of subdirectory names,
so you get stable label 0..K-1. For numeric folder names ``0``..``11`` this
matches integer order.

All images are converted to single-channel (``L``), resized to ``image_size``,
flattened to row vectors, and scaled to ``[0, 1]`` as ``float32``.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from PIL import Image


def _list_image_files(folder: Path) -> list[Path]:
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".gif"}
    out = []
    for p in sorted(folder.iterdir()):
        if p.is_file() and p.suffix.lower() in exts:
            out.append(p)
    return out


def load_char_dataset(
    root: str | Path,
    *,
    image_size: tuple[int, int] = (28, 28),
    num_classes: int | None = 12,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Load dataset from ``root``.

    Parameters
    ----------
    root
        Root directory containing one subdirectory per class.
    image_size
        ``(H, W)`` after resize (PIL bilinear).
    num_classes
        If set, enforce exactly this many subdirectories; raises if mismatch.

    Returns
    -------
    X : ndarray, shape (N, H*W), float32
        Flattened pixels in row-major order, values in ``[0, 1]``.
    y : ndarray, shape (N,), int64
        Labels ``0 .. K-1``.
    class_names : list[str]
        Folder name for each label index.
    """
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"data root not found: {root}")

    subdirs = sorted([p for p in root.iterdir() if p.is_dir()])
    if num_classes is not None and len(subdirs) != num_classes:
        raise ValueError(
            f"expected {num_classes} class folders under {root}, found {len(subdirs)}"
        )
    if len(subdirs) == 0:
        raise ValueError(f"no class subdirectories under {root}")

    h, w = image_size
    xs: list[np.ndarray] = []
    ys: list[int] = []
    class_names = [p.name for p in subdirs]

    for label, folder in enumerate(subdirs):
        files = _list_image_files(folder)
        if not files:
            continue
        for fp in files:
            with Image.open(fp) as im:
                im = im.convert("L").resize((w, h), Image.Resampling.BILINEAR)
                arr = np.asarray(im, dtype=np.float32) / 255.0
            xs.append(arr.reshape(-1))
            ys.append(label)

    if not xs:
        raise ValueError(f"no images found under {root}")

    X = np.stack(xs, axis=0)
    y = np.asarray(ys, dtype=np.int64)
    return X, y, class_names


def make_synthetic_char_dataset(
    *,
    n_per_class: int = 200,
    image_size: tuple[int, int] = (28, 28),
    num_classes: int = 12,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Random noise images for pipeline smoke tests (not meaningful for accuracy).

    Labels are balanced 0..num_classes-1.
    """
    rng = np.random.default_rng(seed)
    h, w = image_size
    d = h * w
    X = rng.random((num_classes * n_per_class, d), dtype=np.float32)
    y = np.repeat(np.arange(num_classes, dtype=np.int64), n_per_class)
    names = [str(i) for i in range(num_classes)]
    return X, y, names
