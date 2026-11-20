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
    rows = values.T
    class_scores = rows[:, 4:4 + num_classes]
    if selected is not None:  # the other classes cannot take the box
        if selected.size == 0:
            return (np.zeros((0, 4), rows.dtype), np.zeros(0, rows.dtype), np.zeros(0, np.int64),
                    np.zeros((0, num_masks), rows.dtype))
        class_scores = class_scores[:, selected]
    class_ids = class_scores.argmax(axis=1)
    scores = class_scores[np.arange(rows.shape[0]), class_ids]
    if selected is not None:
        class_ids = selected[class_ids]
    keep = scores >= threshold
    xywh = rows[keep, :4]
    xyxy = np.empty_like(xywh)
    xyxy[:, 0] = xywh[:, 0] - xywh[:, 2] / 2
    xyxy[:, 1] = xywh[:, 1] - xywh[:, 3] / 2
    xyxy[:, 2] = xywh[:, 0] + xywh[:, 2] / 2
    xyxy[:, 3] = xywh[:, 1] + xywh[:, 3] / 2
    return xyxy, scores[keep], class_ids[keep], rows[keep, 4 + num_classes:]
