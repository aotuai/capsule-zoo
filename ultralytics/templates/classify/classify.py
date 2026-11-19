"""Preparing a detection's image for an image classification model, and reading its answer (numpy and OpenCV only)."""
from typing import Tuple

import cv2
import numpy as np

from .preprocess import letterbox


def _resize(image: np.ndarray, width: int, height: int) -> np.ndarray:
    """Averaging when the image gets smaller (it matches the antialiased resize Ultralytics uses when it tests a
    classifier; plain bilinear is noticeably different for a big crop), bilinear when it gets larger."""
    shrinking = width <= image.shape[1] and height <= image.shape[0]
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA if shrinking else cv2.INTER_LINEAR)


def fit_to_model(crop: np.ndarray, width: int, height: int, mode: str) -> np.ndarray:
    """Make the cropped detection (BGR) exactly ``width`` x ``height``.

    * ``center_crop``: scale so the short side fits, keep the middle (what Ultralytics does when testing a classifier)
    * ``pad``: keep the whole image, fill the rest with grey
    * ``stretch``: squeeze the whole image to the size
    """
    crop_h, crop_w = crop.shape[:2]
    if mode == "pad":  # the same fitting as the detectors use
        return letterbox(crop, width, height)[0]
    if mode == "stretch":
        return _resize(crop, width, height)
    scale = max(width / crop_w, height / crop_h)
    new_w, new_h = max(int(round(crop_w * scale)), width), max(int(round(crop_h * scale)), height)
    resized = _resize(crop, new_w, new_h)
    left, top = (new_w - width) // 2, (new_h - height) // 2
    return resized[top:top + height, left:left + width]


def best_class(output: np.ndarray, num_classes: int) -> Tuple[int, float]:
    """(class number, probability) of the best class. The model normally gives probabilities; if it gives raw
    scores, they are turned into probabilities first."""
    scores = np.asarray(output, dtype=np.float64).reshape(-1)
    if scores.size != num_classes:
        raise ValueError(f"Model output has {scores.size} values, expected {num_classes} (one per class name)")
    if scores.min() < 0 or scores.max() > 1 or abs(scores.sum() - 1) > 1e-2:
        scores = np.exp(scores - scores.max())
        scores = scores / scores.sum()
    best = int(scores.argmax())
    return best, float(scores[best])
