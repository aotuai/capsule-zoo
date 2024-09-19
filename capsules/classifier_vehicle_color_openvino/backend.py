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

        vcolor_key = next(key for key in prediction.keys() if 'color' in key.names)
        vtype_key = next(key for key in prediction.keys() if 'type' in key.names)

        vcolor_data = prediction[vcolor_key]
        if isinstance(vcolor_data, np.ndarray):
            max_color = config.colors[vcolor_data.flatten().argmax()]
        else:
            max_color = config.colors[vcolor_data.argmax()]

        vtype_data = prediction[vtype_key]
        if isinstance(vtype_data, np.ndarray):
            max_type = config.vehicle_classifications[vtype_data.flatten().argmax()]
        else:
            max_type = config.vehicle_classifications[vtype_data.argmax()]

        detection_node.attributes["color"] = max_color
        detection_node.attributes["vehicle_type"] = max_type
