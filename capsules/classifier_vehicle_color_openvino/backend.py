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
        crop = Resize(frame).crop_bbox(detection_node.bbox).frame

        input_dict, _ = self.prepare_inputs(crop)
        prediction = self.send_to_batch(input_dict).result()

        max_color = config.colors[prediction["color"].argmax()]
        max_type = config.vehicle_classifications[prediction["type"].argmax()]

        detection_node.attributes["color"] = max_color
        detection_node.attributes["vehicle_type"] = max_type
