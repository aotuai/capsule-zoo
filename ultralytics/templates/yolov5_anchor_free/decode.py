"""Decode the output of an anchor-free YOLO model (YOLOv5u, YOLOv8, YOLO11, YOLOv9).

Shape (1, 4 + classes, N), N = 8400 at 640x640. Each column is
[center x, center y, width, height, class scores...] in input pixels. There is no
objectness value: the score of a box is its best class score.
"""
from typing import Tuple

import numpy as np


NEEDS_NMS = True


def decode(raw: np.ndarray, threshold: float, num_classes: int, input_size: Tuple[int, int] = (640, 640)) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """:returns: (xyxy boxes, scores, class ids) of the candidates above the threshold. ``input_size`` is (height, width)."""
    values = raw.reshape(raw.shape[-2], raw.shape[-1])
    if values.shape[0] != 4 + num_classes:
        raise ValueError(
            f"Model output has {values.shape[0]} values per box, expected {4 + num_classes} "
            f"for {num_classes} classes")
    rows = values.T
    class_scores = rows[:, 4:]
    class_ids = class_scores.argmax(axis=1)
    scores = class_scores[np.arange(rows.shape[0]), class_ids]
    keep = scores >= threshold
    xywh = rows[keep, :4]
    xyxy = np.empty_like(xywh)
    xyxy[:, 0] = xywh[:, 0] - xywh[:, 2] / 2
    xyxy[:, 1] = xywh[:, 1] - xywh[:, 3] / 2
    xyxy[:, 2] = xywh[:, 0] + xywh[:, 2] / 2
    xyxy[:, 3] = xywh[:, 1] + xywh[:, 3] / 2
    return xyxy, scores[keep], class_ids[keep]
