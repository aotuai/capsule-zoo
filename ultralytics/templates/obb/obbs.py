"""Oriented (rotated) boxes with numpy and OpenCV only (the BrainFrame core has numpy 1.19, OpenCV 4)."""
from typing import Tuple

import cv2
import numpy as np


def corners(rboxes: np.ndarray) -> np.ndarray:
    """(n, 5) boxes [center x, center y, width, height, angle in radians] -> (n, 4, 2) corner points.

    The angle turns the box clockwise on the picture (y points down), as Ultralytics and OpenCV do."""
    cx, cy, w, h, angle = (rboxes[:, i] for i in range(5))
    cos, sin = np.cos(angle), np.sin(angle)
    along = np.stack([w / 2 * cos, w / 2 * sin], axis=1)       # half of the width, turned
    across = np.stack([-h / 2 * sin, h / 2 * cos], axis=1)     # half of the height, turned
    centre = np.stack([cx, cy], axis=1)
    return np.stack([centre + along + across, centre + along - across,
                     centre - along - across, centre - along + across], axis=1)


def rotated_nms(rboxes: np.ndarray, scores: np.ndarray, class_ids: np.ndarray, iou_threshold: float,
                max_detections: int = 300) -> np.ndarray:
    """Indices to keep, best score first. A box is only removed because of a box of the same class."""
    keep = []
    for class_id in np.unique(class_ids):
        members = np.flatnonzero(class_ids == class_id)
        shapes = [((float(r[0]), float(r[1])), (float(r[2]), float(r[3])), float(np.degrees(r[4])))
                  for r in rboxes[members]]
        kept = cv2.dnn.NMSBoxesRotated(shapes, [float(s) for s in scores[members]], 0.0, float(iou_threshold))
        keep.extend(members[np.asarray(kept, dtype=np.int64).reshape(-1)].tolist())
    keep = np.array(keep, dtype=np.int64)
    return keep[np.argsort(-scores[keep])][:max_detections] if keep.size else keep


def to_frame(points: np.ndarray, scale: float, pad_left: int, pad_top: int, frame_w: int, frame_h: int) -> np.ndarray:
    """Map points (..., 2) from model input pixels back to the original frame, clipped to it."""
    mapped = points.astype(np.float32).copy()
    mapped[..., 0] = ((mapped[..., 0] - pad_left) / scale).clip(0, frame_w)
    mapped[..., 1] = ((mapped[..., 1] - pad_top) / scale).clip(0, frame_h)
    return mapped
