from collections import namedtuple
from typing import Dict

import numpy as np

from vcap import (
    Resize,
    DETECTION_NODE_TYPE,
    OPTION_TYPE,
    BaseStreamState)
from vcap_utils import BaseOpenVINOBackend
from . import config


class Backend(BaseOpenVINOBackend):

    def process_frame(self, frame: np.ndarray,
                      detection_node: DETECTION_NODE_TYPE,
                      options: Dict[str, OPTION_TYPE],
                      state: BaseStreamState) -> DETECTION_NODE_TYPE:
        resize = Resize(frame).crop_bbox(detection_node.bbox)

        prediction = self.send_to_batch(resize.frame).get()

        detection_node.attributes["color"] = prediction.color

        # Unused
        # detection_node.attributes["type"] = prediction.type

    def parse_results(self, results: np.ndarray, resize: Resize):
        max_color = config.colors[results["color"].argmax()]
        max_type = config.vehicle_types[results["type"].argmax()]

        return namedtuple("Attributes", "color type")(max_color, max_type)
