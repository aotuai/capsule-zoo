from typing import Dict
from time import time

import numpy as np

from vcap import (
    BaseBackend,
    DETECTION_NODE_TYPE,
    OPTION_TYPE)
from .stream_state import StreamState


class Backend(BaseBackend):

    def process_frame(self, frame: np.ndarray,
                      detection_node: DETECTION_NODE_TYPE,
                      options: Dict[str, OPTION_TYPE],
                      state: StreamState) -> DETECTION_NODE_TYPE:
        capacity = options["cache_capacity"]
        last_tstamp, last_coord = state.get(detection_node.track_id)
        new_tstamp, new_coord = time(), detection_node.bbox.center

        if last_coord is not None:
            # Calculate speed if it's not the first track
            pixel_distance = ((new_coord[0] - last_coord[0]) ** 2
                              + (new_coord[1] - last_coord[1]) ** 2) ** 0.5
            time_elapsed = (new_tstamp - last_tstamp)
            pixel_speed = pixel_distance / time_elapsed
            detection_node.extra_data["pixels_per_second_speed"] = pixel_speed
            detection_node.extra_data["pixel_travel"] = pixel_distance
            detection_node.extra_data["time_elapsed"] = time_elapsed
        else:
            detection_node.extra_data["pixels_per_second_speed"] = None
            detection_node.extra_data["pixel_travel"] = None
            detection_node.extra_data["time_elapsed"] = None

        # Update the state with the latest information
        state.put(detection_node.track_id, (new_tstamp, new_coord),
                  capacity=capacity)

        return detection_node
