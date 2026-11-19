"""Decode the output of an original (anchor-based) YOLOv5 model.

Shape (1, N, 5 + classes), N = 25200 at 640x640. Each row is
[center x, center y, width, height, objectness, class scores...] in input pixels.
The anchors and strides are already applied inside the exported model.
"""
from typing import Tuple

import numpy as np


def xywh_to_xyxy(xywh: np.ndarray) -> np.ndarray:
    xyxy = np.empty_like(xywh)
    xyxy[:, 0] = xywh[:, 0] - xywh[:, 2] / 2
    xyxy[:, 1] = xywh[:, 1] - xywh[:, 3] / 2
    xyxy[:, 2] = xywh[:, 0] + xywh[:, 2] / 2
    xyxy[:, 3] = xywh[:, 1] + xywh[:, 3] / 2
    return xyxy


NEEDS_NMS = True


def decode(raw: np.ndarray, threshold: float, num_classes: int, input_size: Tuple[int, int] = (640, 640)) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """:returns: (xyxy boxes, scores, class ids) of the candidates above the threshold. ``input_size`` is (height, width)."""
    rows = raw.reshape(-1, raw.shape[-1])
    if rows.shape[1] != 5 + num_classes:
        raise ValueError(
            f"Model output has {rows.shape[1]} values per box, expected {5 + num_classes} "
            f"for {num_classes} classes")
    class_scores = rows[:, 5:] * rows[:, 4:5]
    class_ids = class_scores.argmax(axis=1)
    scores = class_scores[np.arange(rows.shape[0]), class_ids]
    keep = scores >= threshold
    return xywh_to_xyxy(rows[keep, :4]), scores[keep], class_ids[keep]
