"""Keypoints of the people a pose model found, as numpy arrays and as the data a capsule reports."""
from typing import Dict, List

import numpy as np


def keypoints_of(values: np.ndarray, num_keypoints: int, keypoint_dim: int) -> np.ndarray:
    """(people, keypoints * dim) -> (people, keypoints, 3): x, y and a confidence (1 when the model has none)."""
    points = values.reshape(-1, num_keypoints, keypoint_dim)
    if keypoint_dim == 3:
        return points
    return np.concatenate([points, np.ones((points.shape[0], num_keypoints, 1), dtype=points.dtype)], axis=2)


def to_frame(points: np.ndarray, scale: float, pad_left: int, pad_top: int, frame_w: int, frame_h: int) -> np.ndarray:
    """Map keypoints (keypoints, 3) from model input pixels to the original frame, clipped to it. The confidence stays."""
    mapped = points.astype(np.float32).copy()
    mapped[:, 0] = ((mapped[:, 0] - pad_left) / scale).clip(0, frame_w)
    mapped[:, 1] = ((mapped[:, 1] - pad_top) / scale).clip(0, frame_h)
    return mapped


def keypoint_data(points: np.ndarray, names: List[str], min_confidence: float) -> Dict[str, List[float]]:
    """``{"nose": [x, y, confidence], ...}`` with whole pixels and the confidence rounded to two decimals.
    Keypoints the model is not sure about (below ``min_confidence``) are left out."""
    return {
        name: [int(round(float(x))), int(round(float(y))), round(float(confidence), 2)]
        for name, (x, y, confidence) in zip(names, points)
        if confidence >= min_confidence
    }
