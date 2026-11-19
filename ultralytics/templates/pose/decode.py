"""Decode the output of a pose estimation model (YOLOv8-pose, YOLO11-pose).

Shape (1, 4 + classes + keypoints * dim, N), N = 8400 at 640x640 (dim is 3: x, y, visibility; or 2: x, y).
Each column is [center x, center y, width, height, class scores..., keypoint values...] in input pixels.
"""
from typing import Tuple

import numpy as np

from .poses import keypoints_of

NEEDS_NMS = True


def decode(raw: np.ndarray, threshold: float, num_classes: int, num_keypoints: int, keypoint_dim: int
           ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """:returns: (xyxy boxes, scores, class ids, keypoints) of the candidates above the threshold.
    ``keypoints`` is (candidates, keypoints, 3): x, y in input pixels and a confidence (1 when the model has none)."""
    values = raw.reshape(raw.shape[-2], raw.shape[-1])
    expected = 4 + num_classes + num_keypoints * keypoint_dim
    if values.shape[0] != expected:
        raise ValueError(f"Model output has {values.shape[0]} values per box, expected {expected} "
                         f"for {num_classes} classes and {num_keypoints} keypoints")
    rows = values.T
    class_scores = rows[:, 4:4 + num_classes]
    class_ids = class_scores.argmax(axis=1)
    scores = class_scores[np.arange(rows.shape[0]), class_ids]
    keep = scores >= threshold
    xywh = rows[keep, :4]
    xyxy = np.empty_like(xywh)
    xyxy[:, 0] = xywh[:, 0] - xywh[:, 2] / 2
    xyxy[:, 1] = xywh[:, 1] - xywh[:, 3] / 2
    xyxy[:, 2] = xywh[:, 0] + xywh[:, 2] / 2
    xyxy[:, 3] = xywh[:, 1] + xywh[:, 3] / 2
    return xyxy, scores[keep], class_ids[keep], keypoints_of(rows[keep, 4 + num_classes:], num_keypoints, keypoint_dim)

