"""Decode the output of an end-to-end instance segmentation model (YOLO26-seg exported without NMS).

Shape (1, 300, 6 + masks): up to 300 objects, each [x1, y1, x2, y2, score, class id, mask coefficients...], the
box corners in input pixels. The model has already removed duplicate boxes, so the capsule does not run NMS.
The mask coefficients are combined with the model's second output (the mask prototypes): see masks.py.
"""
from typing import Tuple

import numpy as np

NEEDS_NMS = False


def decode(raw: np.ndarray, threshold: float, num_classes: int, num_masks: int
           ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """:returns: (xyxy boxes, scores, class ids, mask coefficients) of the objects above the threshold."""
    rows = raw.reshape(-1, raw.shape[-1])
    if rows.shape[1] != 6 + num_masks:
        raise ValueError(f"Model output has {rows.shape[1]} values per object, expected {6 + num_masks} "
                         f"(box, score, class and {num_masks} mask values)")
    rows = rows[rows[:, 4] >= threshold]
    class_ids = rows[:, 5].astype(np.int64)
    if class_ids.size and (class_ids.min() < 0 or class_ids.max() >= num_classes):
        raise ValueError(f"Model output has class number {int(class_ids.max())} but only {num_classes} class names were given")
    return rows[:, :4].copy(), rows[:, 4].copy(), class_ids, rows[:, 6:].copy()
