"""Decode the output of an end-to-end pose estimation model (YOLO26-pose exported without NMS).

Shape (1, 300, 6 + keypoints * dim): up to 300 people, each [x1, y1, x2, y2, score, class id, keypoint values...],
the box corners and keypoints in input pixels. The model has already removed duplicates, so the capsule does not
run NMS.
"""
from typing import Tuple

import numpy as np

from .poses import keypoints_of

NEEDS_NMS = False


def decode(raw: np.ndarray, threshold: float, num_classes: int, num_keypoints: int, keypoint_dim: int
           ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """:returns: (xyxy boxes, scores, class ids, keypoints (people, keypoints, 3)) of the people above the threshold."""
    rows = raw.reshape(-1, raw.shape[-1])
    expected = 6 + num_keypoints * keypoint_dim
    if rows.shape[1] != expected:
        raise ValueError(f"Model output has {rows.shape[1]} values per person, expected {expected} "
                         f"(box, score, class and {num_keypoints} keypoints)")
    rows = rows[rows[:, 4] >= threshold]
    class_ids = rows[:, 5].astype(np.int64)
    if class_ids.size and (class_ids.min() < 0 or class_ids.max() >= num_classes):
        raise ValueError(f"Model output has class number {int(class_ids.max())} but only {num_classes} class names were given")
    return rows[:, :4].copy(), rows[:, 4].copy(), class_ids, keypoints_of(rows[:, 6:], num_keypoints, keypoint_dim)
