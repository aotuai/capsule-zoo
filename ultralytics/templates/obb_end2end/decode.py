"""Decode the output of an end-to-end oriented bounding box model (YOLO26-obb exported without NMS).

Shape (1, 300, 7): up to 300 objects, each [center x, center y, width, height, score, class id, angle] in input
pixels and radians. The model has already removed duplicates, so the capsule does not run NMS.
"""
from typing import Tuple

import numpy as np

NEEDS_NMS = False


def decode(raw: np.ndarray, threshold: float, num_classes: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """:returns: (rotated boxes (n, 5) [cx, cy, w, h, angle], scores, class ids) of the objects above the threshold."""
    rows = raw.reshape(-1, raw.shape[-1])
    if rows.shape[1] != 7:
        raise ValueError(f"Model output has {rows.shape[1]} values per object, expected 7 "
                         "(center x, center y, width, height, score, class, angle)")
    rows = rows[rows[:, 4] >= threshold]
    class_ids = rows[:, 5].astype(np.int64)
    if class_ids.size and (class_ids.min() < 0 or class_ids.max() >= num_classes):
        raise ValueError(f"Model output has class number {int(class_ids.max())} but only {num_classes} class names were given")
    return np.concatenate([rows[:, :4], rows[:, 6:7]], axis=1), rows[:, 4].copy(), class_ids
