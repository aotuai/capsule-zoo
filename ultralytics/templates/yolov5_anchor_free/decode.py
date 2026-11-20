"""Decode the output of an anchor-free YOLO model (YOLOv5u, YOLOv8, YOLO11, YOLOv9).

Shape (1, 4 + classes, N), N = 8400 at 640x640. Each column is
[center x, center y, width, height, class scores...] in input pixels. There is no
objectness value: the score of a box is its best class score.
"""
from typing import Optional, Tuple

import numpy as np


NEEDS_NMS = True


def decode(raw: np.ndarray, threshold: float, num_classes: int, input_size: Tuple[int, int] = (640, 640),
           selected: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """:returns: (xyxy boxes, scores, class ids) of the candidates above the threshold. ``input_size`` is (height, width).

    ``selected`` (class numbers) limits the choice of the best class to these classes; ``None`` means all of them.
    """
    values = raw.reshape(raw.shape[-2], raw.shape[-1])
    if values.shape[0] != 4 + num_classes:
        raise ValueError(
            f"Model output has {values.shape[0]} values per box, expected {4 + num_classes} "
            f"for {num_classes} classes")
    # Work on the values as the model gives them (one row per value, one column per box), without transposing them: a model with
    # thousands of classes (YOLOE prompt-free) has an output of over 150 MB, and the best score of each box is found much faster
    # by a maximum along the class rows than by an argmax along strided columns. The class is only looked up for the boxes
    # above the threshold.
    class_scores = values[4:]
    if selected is not None:
        if selected.size == 0:
            return np.zeros((0, 4), values.dtype), np.zeros(0, values.dtype), np.zeros(0, np.int64)
        class_scores = class_scores[selected]  # the other classes cannot take the box
    best = class_scores.max(axis=0)
    keep = np.nonzero(best >= threshold)[0]
    class_ids = class_scores[:, keep].argmax(axis=0)
    if selected is not None:
        class_ids = selected[class_ids]
    xywh = values[:4, keep].T
    xyxy = np.empty_like(xywh)
    xyxy[:, 0] = xywh[:, 0] - xywh[:, 2] / 2
    xyxy[:, 1] = xywh[:, 1] - xywh[:, 3] / 2
    xyxy[:, 2] = xywh[:, 0] + xywh[:, 2] / 2
    xyxy[:, 3] = xywh[:, 1] + xywh[:, 3] / 2
    return xyxy, best[keep], class_ids
