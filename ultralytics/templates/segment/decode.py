"""Decode the output of an instance segmentation model (YOLOv8-seg, YOLOv9-seg, YOLO11-seg).

Shape (1, 4 + classes + masks, N), N = 8400 at 640x640. Each column is [center x, center y, width, height,
class scores..., mask coefficients...] in input pixels. The mask coefficients are combined with the model's
second output (the mask prototypes) to make each object's mask: see masks.py.
"""
from typing import Optional, Tuple

import numpy as np

NEEDS_NMS = True


def decode(raw: np.ndarray, threshold: float, num_classes: int, num_masks: int,
           selected: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """:returns: (xyxy boxes, scores, class ids, mask coefficients) of the candidates above the threshold."""
    values = raw.reshape(raw.shape[-2], raw.shape[-1])
    if values.shape[0] != 4 + num_classes + num_masks:
        raise ValueError(
            f"Model output has {values.shape[0]} values per box, expected {4 + num_classes + num_masks} "
            f"for {num_classes} classes and {num_masks} mask values")
    # Work on the values as the model gives them (one row per value, one column per box), without transposing them: a model with
    # thousands of classes (YOLOE prompt-free) has an output of over 150 MB, and the best score of each box is found much faster
    # by a maximum along the class rows than by an argmax along strided columns. The class is only looked up for the boxes
    # above the threshold.
    class_scores = values[4:4 + num_classes]
    if selected is not None:  # the other classes cannot take the box
        if selected.size == 0:
            return (np.zeros((0, 4), values.dtype), np.zeros(0, values.dtype), np.zeros(0, np.int64),
                    np.zeros((0, num_masks), values.dtype))
        class_scores = class_scores[selected]
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
    return xyxy, best[keep], class_ids, np.ascontiguousarray(values[4 + num_classes:, keep].T)
