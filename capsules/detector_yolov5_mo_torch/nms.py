"""Non-maximum suppression with numpy only (the BrainFrame core has numpy 1.19, no torch needed)."""
import numpy as np


def nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> np.ndarray:
    """Greedy NMS on xyxy boxes. Returns the indices to keep, best score first."""
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        best = order[0]
        keep.append(best)
        if order.size == 1:
            break
        rest = order[1:]
        x1 = np.maximum(boxes[best, 0], boxes[rest, 0])
        y1 = np.maximum(boxes[best, 1], boxes[rest, 1])
        x2 = np.minimum(boxes[best, 2], boxes[rest, 2])
        y2 = np.minimum(boxes[best, 3], boxes[rest, 3])
        inter = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
        area_best = (boxes[best, 2] - boxes[best, 0]) * (boxes[best, 3] - boxes[best, 1])
        area_rest = (boxes[rest, 2] - boxes[rest, 0]) * (boxes[rest, 3] - boxes[rest, 1])
        iou = inter / np.maximum(area_best + area_rest - inter, 1e-9)
        order = rest[iou <= iou_threshold]
    return np.array(keep, dtype=np.int64)


def class_aware_nms(boxes: np.ndarray, scores: np.ndarray, class_ids: np.ndarray,
                    iou_threshold: float, max_detections: int = 300) -> np.ndarray:
    """NMS that never suppresses a box because of a box of another class."""
    if boxes.shape[0] == 0:
        return np.zeros((0,), dtype=np.int64)
    # Shift each class far away from the others so boxes of different classes never overlap
    offset = class_ids.astype(np.float32)[:, None] * (boxes.max() + 1.0)
    return nms(boxes + offset, scores, iou_threshold)[:max_detections]
