"""Letterbox resize: fit the frame into the model input and remember how to undo it."""
from typing import Tuple

import cv2
import numpy as np

PAD_COLOR = (114, 114, 114)


def letterbox(frame: np.ndarray, width: int, height: int) -> Tuple[np.ndarray, float, int, int]:
    """Scale the frame to fit (width, height), keep its aspect ratio, pad the rest.

    :returns: (image, scale, pad_left, pad_top)
    """
    frame_h, frame_w = frame.shape[:2]
    scale = min(width / frame_w, height / frame_h)
    new_w, new_h = int(round(frame_w * scale)), int(round(frame_h * scale))
    if (new_w, new_h) != (frame_w, frame_h):
        frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    pad_left = (width - new_w) // 2
    pad_top = (height - new_h) // 2
    image = cv2.copyMakeBorder(
        frame, pad_top, height - new_h - pad_top, pad_left, width - new_w - pad_left,
        cv2.BORDER_CONSTANT, value=PAD_COLOR,
    )
    return image, scale, pad_left, pad_top


def to_model_input(image: np.ndarray, in_model: bool = True) -> np.ndarray:
    """HWC BGR uint8 -> 1x3xHxW float32.

    ``in_model=True``: the IR has the division by 255 and the BGR to RGB swap built in (the usual
    ``--scale 255 --reverse_input_channels`` conversion), so the raw 0..255 BGR values are passed.
    ``in_model=False``: the model expects RGB in 0..1 and this does the swap and the scaling.
    """
    if not in_model:
        image = image[:, :, ::-1].astype(np.float32) / 255.0
    return np.ascontiguousarray(image.transpose(2, 0, 1)[None], dtype=np.float32)


def unletterbox(boxes: np.ndarray, scale: float, pad_left: int, pad_top: int,
                frame_w: int, frame_h: int) -> np.ndarray:
    """Map xyxy boxes from model input space back to the original frame, clipped to it."""
    boxes = boxes.copy()
    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad_left) / scale
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad_top) / scale
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, frame_w)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, frame_h)
    return boxes
