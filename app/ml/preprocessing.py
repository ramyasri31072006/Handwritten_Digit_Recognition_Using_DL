"""Input preparation.

`for_sklearn` and `for_keras` reproduce what the notebooks feed to the models
(pixels / 255.0; the CNN additionally reshapes to 28x28x1).

`image_to_mnist_vector` is only used for drawn or uploaded pictures. It converts them to the
MNIST look: a white digit on black, scaled into a 20x20 box and centred by centre of mass
inside a 28x28 frame.
"""
import numpy as np
import pandas as pd
from PIL import Image

from . import config

PIXEL_COLUMNS = [f"pixel{i}" for i in range(config.N_PIXELS)]


def for_sklearn(pixels_uint8, feature_names=None):
    """(n, 784) uint8 -> DataFrame of float64 in [0, 1] using the column names the model was fitted with."""
    columns = list(feature_names) if feature_names is not None else PIXEL_COLUMNS
    return pd.DataFrame(np.asarray(pixels_uint8, dtype=np.float64) / 255.0, columns=columns)


def for_keras(pixels_uint8):
    """(n, 784) uint8 -> float32 array shaped (n, 28, 28, 1) in [0, 1]."""
    x = np.asarray(pixels_uint8, dtype=np.float32) / 255.0
    return x.reshape(-1, config.IMG_SIDE, config.IMG_SIDE, 1)


def image_to_mnist_vector(img, invert=None):
    """Convert a PIL image of one digit into a (784,) uint8 vector in MNIST style.

    invert=None auto-detects the background: a bright border means dark ink on light paper.
    Raises ValueError when the picture is blank.
    """
    if img.mode in ("RGBA", "LA", "P"):
        rgba = img.convert("RGBA")
        img = Image.alpha_composite(Image.new("RGBA", rgba.size, (255, 255, 255, 255)), rgba)
    gray = np.asarray(img.convert("L"), dtype=np.float32)

    if invert is None:
        border = np.concatenate([gray[0], gray[-1], gray[:, 0], gray[:, -1]])
        invert = float(np.median(border)) > 127
    if invert:
        gray = 255.0 - gray

    lo, hi = float(gray.min()), float(gray.max())
    if hi - lo < 40:
        raise ValueError("The image looks blank or has too little contrast to find a digit.")
    gray = (gray - lo) / (hi - lo) * 255.0

    ys, xs = np.where(gray > 40)
    if ys.size == 0:
        raise ValueError("No digit strokes were found in the image.")
    crop = gray[ys.min(): ys.max() + 1, xs.min(): xs.max() + 1]

    digit = Image.fromarray(crop.astype(np.uint8))
    w, h = digit.size
    scale = 20.0 / max(w, h)
    new_size = (max(1, round(w * scale)), max(1, round(h * scale)))
    digit = digit.resize(new_size, Image.LANCZOS)

    frame = np.zeros((config.IMG_SIDE, config.IMG_SIDE), dtype=np.float32)
    ox, oy = (config.IMG_SIDE - new_size[0]) // 2, (config.IMG_SIDE - new_size[1]) // 2
    frame[oy: oy + new_size[1], ox: ox + new_size[0]] = np.asarray(digit, dtype=np.float32)

    total = float(frame.sum())
    if total > 0:
        idx = np.arange(config.IMG_SIDE)
        cy = float((frame.sum(axis=1) * idx).sum() / total)
        cx = float((frame.sum(axis=0) * idx).sum() / total)
        frame = np.roll(frame, (int(round(13.5 - cy)), int(round(13.5 - cx))), axis=(0, 1))

    return np.clip(frame, 0, 255).astype(np.uint8).reshape(-1)
