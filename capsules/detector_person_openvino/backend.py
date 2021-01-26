from typing import Dict

import numpy as np

from vcap import (
    DETECTION_NODE_TYPE,
    OPTION_TYPE,
    BaseStreamState,
    SizeFilter)
from vcap_utils import BaseOpenVINOBackend, non_max_suppression


class Backend(BaseOpenVINOBackend):
    label_map: Dict[int, str] = {0: "person"}

    def process_frame(self, frame: np.ndarray,
                      detection_node: DETECTION_NODE_TYPE,
                      options: Dict[str, OPTION_TYPE],
                      state: BaseStreamState) -> DETECTION_NODE_TYPE:
        input_dict, resize = self.prepare_inputs(frame)
        prediction = self.send_to_batch(input_dict).result()
        detections = self.parse_detection_results(
            prediction, resize, self.label_map,
            min_confidence=options["threshold"])

        # Remove overlapping detections
        max_detection_overlap = options["max_detection_overlap"]
        detections = non_max_suppression(detections, max_detection_overlap)

        # Remove detections that are too small or too big
        min_detection_area = options["min_detection_area"]
        max_detection_area = options["max_detection_area"]
        detections = (SizeFilter(detections)
                      .min_area(min_detection_area)
                      .max_area(max_detection_area)
                      .apply())
        return detections
