from typing import Dict

import numpy as np

from vcap import (
    Resize,
    DETECTION_NODE_TYPE,
    OPTION_TYPE,
    BaseStreamState)
from vcap_utils import BaseOpenVINOBackend
from vcap import __version__ as vcap_version
def openvino2024_compatible():
    from packaging import version
    from vcap import __version__ as vcap_version
    openvino2024_compatible_vcap_version = '0.3.8'
    return version.parse(vcap_version) >= version.parse(openvino2024_compatible_vcap_version)
__openvino2024__ = openvino2024_compatible()

from . import config


class Backend(BaseOpenVINOBackend):

    def process_frame(self, frame: np.ndarray,
                      detection_node: DETECTION_NODE_TYPE,
                      options: Dict[str, OPTION_TYPE],
                      state: BaseStreamState) -> DETECTION_NODE_TYPE:
        crop = Resize(frame).crop_bbox(detection_node.bbox).frame

        input_dict, _ = self.prepare_inputs(crop)
        prediction = self.send_to_batch(input_dict).result()

        if __openvino2024__:
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
        else:
            max_color = config.colors[prediction["color"].argmax()]
            max_type = config.vehicle_classifications[prediction["type"].argmax()]

        detection_node.attributes["color"] = max_color
        detection_node.attributes["vehicle_type"] = max_type
