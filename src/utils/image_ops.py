from __future__ import annotations

import numpy as np
from PIL import Image


def load_image_rgb(path) -> Image.Image:
    """Load image as RGB (PIL)."""
    img = Image.open(path)
    return img.convert("RGB")


def pil_to_np_float01(img: Image.Image) -> np.ndarray:
    """RGB uint8 -> float32 [0,1]. shape (H,W,3)."""
    return (np.asarray(img).astype(np.float32) / 255.0).clip(0.0, 1.0)


def np_float01_to_pil(arr: np.ndarray) -> Image.Image:
    """float [0,1] -> RGB uint8 PIL. shape (H,W,3)."""
    arr = np.clip(arr, 0.0, 1.0)
    u8 = (arr * 255.0).round().astype(np.uint8)
    return Image.fromarray(u8, mode="RGB")


def jet_colormap(values01: np.ndarray) -> np.ndarray:
    """
    Simple jet-like colormap without matplotlib.
    Input: float array in [0,1], shape (H,W)
    Output: RGB float array in [0,1], shape (H,W,3)
    """
    v = np.clip(values01.astype(np.float32), 0.0, 1.0)
    r = np.clip(1.5 - np.abs(4.0 * v - 3.0), 0.0, 1.0)
    g = np.clip(1.5 - np.abs(4.0 * v - 2.0), 0.0, 1.0)
    b = np.clip(1.5 - np.abs(4.0 * v - 1.0), 0.0, 1.0)
    return np.stack([r, g, b], axis=-1).astype(np.float32)


def alpha_blend(base_rgb01: np.ndarray, top_rgb01: np.ndarray, alpha: float) -> np.ndarray:
    a = float(np.clip(alpha, 0.0, 1.0))
    return (1.0 - a) * base_rgb01 + a * top_rgb01


def rgb_to_gray01(rgb01: np.ndarray) -> np.ndarray:
    r, g, b = rgb01[..., 0], rgb01[..., 1], rgb01[..., 2]
    return (0.299 * r + 0.587 * g + 0.114 * b).astype(np.float32)


def laplacian_variance(gray01: np.ndarray) -> float:
    """
    Rough blur metric without OpenCV.
    Higher variance => sharper.
    """
    k = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)
    g = gray01.astype(np.float32)
    gp = np.pad(g, ((1, 1), (1, 1)), mode="edge")
    out = (
        k[0, 0] * gp[:-2, :-2]
        + k[0, 1] * gp[:-2, 1:-1]
        + k[0, 2] * gp[:-2, 2:]
        + k[1, 0] * gp[1:-1, :-2]
        + k[1, 1] * gp[1:-1, 1:-1]
        + k[1, 2] * gp[1:-1, 2:]
        + k[2, 0] * gp[2:, :-2]
        + k[2, 1] * gp[2:, 1:-1]
        + k[2, 2] * gp[2:, 2:]
    )
    return float(np.var(out))

