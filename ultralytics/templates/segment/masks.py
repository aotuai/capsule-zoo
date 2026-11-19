"""From mask coefficients to polygons, with numpy and OpenCV only (the BrainFrame core has numpy 1.19, OpenCV 4)."""
from typing import List, Optional, Tuple

import cv2
import numpy as np

MAX_POINTS = 40
"""A polygon is kept small: it is stored and sent with every detection of every frame."""


def mask_polygons(protos: np.ndarray, coefficients: np.ndarray, boxes: np.ndarray, input_size: Tuple[int, int],
                  mask_threshold: float) -> List[Optional[np.ndarray]]:
    """The outline of each object's mask, in model input pixels.

    :param protos: the mask prototypes, (masks, H, W), usually at a quarter of the input size
    :param coefficients: (objects, masks), one row per object
    :param boxes: (objects, 4) xyxy in input pixels; the mask is cut at its box
    :param input_size: (height, width) of the model input
    :returns: one (points, 2) array per object, or None where there is no usable mask (the caller uses the box)
    """
    if len(boxes) == 0:
        return []
    count, proto_h, proto_w = protos.shape
    logits = coefficients.astype(np.float32) @ protos.reshape(count, -1).astype(np.float32)
    masks = (1.0 / (1.0 + np.exp(-np.clip(logits, -30, 30)))).reshape(-1, proto_h, proto_w)
    scale_x, scale_y = proto_w / input_size[1], proto_h / input_size[0]
    return [_outline(mask, box, scale_x, scale_y, input_size, mask_threshold) for mask, box in zip(masks, boxes)]


def _outline(mask, box, scale_x, scale_y, input_size, threshold) -> Optional[np.ndarray]:
    height, width = input_size
    x1, y1 = max(float(box[0]), 0.0), max(float(box[1]), 0.0)
    x2, y2 = min(float(box[2]), float(width)), min(float(box[3]), float(height))
    proto_h, proto_w = mask.shape
    px1, py1 = int(np.floor(x1 * scale_x)), int(np.floor(y1 * scale_y))
    px2, py2 = min(int(np.ceil(x2 * scale_x)), proto_w), min(int(np.ceil(y2 * scale_y)), proto_h)
    if px2 <= px1 or py2 <= py1:
        return None
    # the part of the mask inside the box, enlarged to input pixels
    origin_x, origin_y = px1 / scale_x, py1 / scale_y
    out_w = max(int(round((px2 - px1) / scale_x)), 1)
    out_h = max(int(round((py2 - py1) / scale_y)), 1)
    crop = cv2.resize(mask[py1:py2, px1:px2], (out_w, out_h), interpolation=cv2.INTER_LINEAR)
    binary = crop > threshold
    xs = np.arange(out_w) + origin_x
    ys = np.arange(out_h) + origin_y
    binary &= ((xs >= x1) & (xs <= x2))[None, :] & ((ys >= y1) & (ys <= y2))[:, None]  # nothing outside the box
    contours = cv2.findContours(binary.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) < 4:
        return None
    epsilon = 0.003 * cv2.arcLength(contour, True)
    points = cv2.approxPolyDP(contour, epsilon, True)
    while len(points) > MAX_POINTS:
        epsilon = max(epsilon, 0.5) * 1.5
        points = cv2.approxPolyDP(contour, epsilon, True)
    if len(points) < 3:
        return None
    return points.reshape(-1, 2).astype(np.float32) + np.array([origin_x, origin_y], dtype=np.float32)


def unletterbox_points(points: np.ndarray, scale: float, pad_left: int, pad_top: int,
                       frame_w: int, frame_h: int) -> np.ndarray:
    """Map polygon points from model input space back to the original frame, clipped to it."""
    mapped = np.empty_like(points)
    mapped[:, 0] = ((points[:, 0] - pad_left) / scale).clip(0, frame_w)
    mapped[:, 1] = ((points[:, 1] - pad_top) / scale).clip(0, frame_h)
    return mapped
