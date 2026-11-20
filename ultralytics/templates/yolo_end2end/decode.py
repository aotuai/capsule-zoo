"""Decode the output of an end-to-end YOLO model (YOLO26, YOLOv10 exported without non-max-suppression).

Shape (1, 300, 6): up to 300 boxes, each [x1, y1, x2, y2, score, class id], the box corners in input
pixels. The model has already removed duplicate boxes, so the capsule does not run NMS.
"""
from typing import Optional, Tuple

import numpy as np

NEEDS_NMS = False


def decode(raw: np.ndarray, threshold: float, num_classes: int, input_size: Tuple[int, int] = (640, 640),
           selected: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """:returns: (xyxy boxes, scores, class ids) of the boxes above the threshold. ``input_size`` is (height, width)."""
    rows = raw.reshape(-1, raw.shape[-1])
    if rows.shape[1] != 6:
        raise ValueError(f"Model output has {rows.shape[1]} values per box, expected 6 (x1, y1, x2, y2, score, class)")
    keep = rows[:, 4] >= threshold
    rows = rows[keep]
    if selected is not None:  # the model already chose one class per box: drop the boxes of the other classes
        rows = rows[np.isin(rows[:, 5].astype(np.int64), selected)]
    class_ids = rows[:, 5].astype(np.int64)
    if class_ids.size and (class_ids.min() < 0 or class_ids.max() >= num_classes):
        raise ValueError(f"Model output has class number {int(class_ids.max())} but only {num_classes} class names were given")
    return rows[:, :4].copy(), rows[:, 4].copy(), class_ids
