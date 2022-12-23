from copy import deepcopy

import numpy as np
import cv2
from vcap import (
    BaseStreamState,
)

FRAME_RESIZE_FACTOR: float = 1 / 10
"""The factor to scale frames by before finding their difference. This removes
noise and should speed up the algorithm.
"""
DIFFERENCE_THRESHOLD: float = 0.7
"""The minimum average distance between pixels in two frames for those frames to
be considered distinct.
"""


def frames_different(frame1: np.ndarray, frame2: np.ndarray):
    """Uses a simple algorithm to find the difference between the two frames.

    :param frame1: The first frame to compare
    :param frame2: The second frame to compare
    :return: True if the frame difference is above a threshold
    """

    if frame1 is None or frame2 is None:
        return True

    height, width = frame1.shape[:2]
    new_height = int(height * FRAME_RESIZE_FACTOR)
    new_width = int(width * FRAME_RESIZE_FACTOR)
    resized_size = new_width, new_height

    # Preprocess frames 1 and 2
    preprocessed_frame1 = cv2.cvtColor(
        cv2.resize(frame1, resized_size, interpolation=cv2.INTER_NEAREST),
        code=cv2.COLOR_BGR2GRAY,
    )
    preprocessed_frame2 = cv2.cvtColor(
        cv2.resize(frame2, resized_size, interpolation=cv2.INTER_NEAREST),
        code=cv2.COLOR_BGR2GRAY,
    )

    # Get the pixel difference between frames 1 and 2
    diff_frame = cv2.absdiff(preprocessed_frame1, preprocessed_frame2)

    # Check the average difference in pixels for each frame against a threshold
    pixel_delta_sum = cv2.sumElems(diff_frame)[0]

    # Divide the number of pixels over the area of the resized image
    per_pixel_avg_delta = pixel_delta_sum / (resized_size[0] * resized_size[1])
    # if per_pixel_avg_delta > DIFFERENCE_THRESHOLD:
    #     print(f"per_pixel_avg_delta={per_pixel_avg_delta}")
    return per_pixel_avg_delta > DIFFERENCE_THRESHOLD


class StreamState(BaseStreamState):
    def __init__(self):
        self.last_cell_frame = None
        self.last_detections = []

    def is_similar_frame(self, cell_frame):
        return not frames_different(self.last_cell_frame, cell_frame)

    def get_last_detections(self):
        return self.last_detections

    def update_last_frame(self, last_cell_frame, last_detections=None):
        self.last_cell_frame = last_cell_frame
        if last_detections is not None:
            self.last_detections = deepcopy(last_detections)
