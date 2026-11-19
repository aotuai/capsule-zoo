"""Decode the output of RT-DETR.

Shape (1, 300, 6): up to 300 boxes, each [center x, center y, width, height, score, class id]. The box
values are fractions of the input image (0 to 1), not pixels. RT-DETR has no duplicate boxes by design,
so the capsule does not run NMS.
"""
from typing import Tuple

import numpy as np

NEEDS_NMS = False


def decode(raw: np.ndarray, threshold: float, num_classes: int, input_size: Tuple[int, int] = (640, 640)) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """:returns: (xyxy boxes in input pixels, scores, class ids) of the boxes above the threshold. ``input_size`` is (height, width)."""
    rows = raw.reshape(-1, raw.shape[-1])
    if rows.shape[1] != 6:
        raise ValueError(f"Model output has {rows.shape[1]} values per box, expected 6 (x, y, w, h, score, class)")
    rows = rows[rows[:, 4] >= threshold]
    class_ids = rows[:, 5].astype(np.int64)
    if class_ids.size and (class_ids.min() < 0 or class_ids.max() >= num_classes):
        raise ValueError(f"Model output has class number {int(class_ids.max())} but only {num_classes} class names were given")
    height, width = input_size
    xyxy = np.empty((rows.shape[0], 4), dtype=np.float32)
    xyxy[:, 0] = (rows[:, 0] - rows[:, 2] / 2) * width
    xyxy[:, 1] = (rows[:, 1] - rows[:, 3] / 2) * height
    xyxy[:, 2] = (rows[:, 0] + rows[:, 2] / 2) * width
    xyxy[:, 3] = (rows[:, 1] + rows[:, 3] / 2) * height
    return xyxy, rows[:, 4].copy(), class_ids
