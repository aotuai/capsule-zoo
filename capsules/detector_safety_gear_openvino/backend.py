from typing import Dict

import numpy as np

from vcap import (
    DETECTION_NODE_TYPE,
    OPTION_TYPE, BaseStreamState)
from vcap_utils import BaseOpenVINOBackend


class Backend(BaseOpenVINOBackend):
    label_map: Dict[int, str] = {1: "unknown", 2: "safety vest",
                                 3: "unknown",
                                 4: "safety hat"}

    def process_frame(self, frame: np.ndarray,
                      detection_node: DETECTION_NODE_TYPE,
                      options: Dict[str, OPTION_TYPE],
                      state: BaseStreamState) -> DETECTION_NODE_TYPE:
        input_dict, resize = self.prepare_inputs(frame)
        prediction = self.send_to_batch(input_dict).get()
        detections = self.parse_detection_results(
            prediction, resize, self.label_map,
            min_confidence=options["threshold"])
        detections = [detection for detection in detections if
                      detection.class_name != "unknown"]

        return detections
