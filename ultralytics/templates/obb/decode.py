"""Decode the output of an oriented bounding box model (YOLOv8-obb, YOLO11-obb).

Shape (1, 4 + classes + 1, N), N = 21504 at 1024x1024. Each column is [center x, center y, width, height,
class scores..., angle] in input pixels and radians.
"""
from typing import Tuple

import numpy as np

NEEDS_NMS = True


def decode(raw: np.ndarray, threshold: float, num_classes: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """:returns: (rotated boxes (n, 5) [cx, cy, w, h, angle], scores, class ids) of the candidates above the threshold."""
    values = raw.reshape(raw.shape[-2], raw.shape[-1])
    if values.shape[0] != 5 + num_classes:
        raise ValueError(f"Model output has {values.shape[0]} values per box, expected {5 + num_classes} "
                         f"for {num_classes} classes")
    rows = values.T
    class_scores = rows[:, 4:4 + num_classes]
    class_ids = class_scores.argmax(axis=1)
    scores = class_scores[np.arange(rows.shape[0]), class_ids]
    keep = scores >= threshold
    rboxes = np.concatenate([rows[keep, :4], rows[keep, 4 + num_classes:5 + num_classes]], axis=1)
    return rboxes, scores[keep], class_ids[keep]
