from typing import Dict, List

import numpy as np

from vcap import (
    DetectionNode,
    Resize,
    DETECTION_NODE_TYPE,
    OPTION_TYPE,
    BaseStreamState)
from vcap_utils import BaseOpenVINOBackend


class Backend(BaseOpenVINOBackend):
    label_map: Dict[int, str] = {1: "person"}

    def process_frame(self, frame: np.ndarray,
                      detection_node: DETECTION_NODE_TYPE,
                      options: Dict[str, OPTION_TYPE],
                      state: BaseStreamState) -> DETECTION_NODE_TYPE:
        input_dict, resize = self.prepare_inputs(frame)
        prediction = self.send_to_batch(frame).get()
        detections = self.parse_detection_results(prediction, resize)
        return [d for d in detections
                if d.extra_data["confidence"] > options["threshold"]]
