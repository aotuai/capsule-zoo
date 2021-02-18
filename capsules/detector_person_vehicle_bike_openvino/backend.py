from typing import Dict

import numpy as np

from vcap import (
    DETECTION_NODE_TYPE,
    OPTION_TYPE,
    BaseStreamState,
    SizeFilter)
from vcap_utils import BaseOpenVINOBackend


class Backend(BaseOpenVINOBackend):
    label_map: Dict[int, str] = {1: "vehicle", 2: "person", 3: "bike"}

    def process_frame(self, frame: np.ndarray,
                      detection_node: DETECTION_NODE_TYPE,
                      options: Dict[str, OPTION_TYPE],
                      state: BaseStreamState) -> DETECTION_NODE_TYPE:
        input_dict, resize = self.prepare_inputs(frame)
        prediction = self.send_to_batch(input_dict).result()
        detections = self.parse_detection_results(
            prediction, resize, self.label_map,
            min_confidence=options["threshold"])

        # This capsule has an option to filter out anything that's not "person"
        people_only = options["only_person_detections"]
        if people_only:
            detections = [d for d in detections
                          if d.class_name == "person"]

        # Remove too small and too big detections
        min_detection_area = options["min_detection_area"]
        max_detection_area = options["max_detection_area"]
        detections = (SizeFilter(detections)
                      .min_area(min_detection_area)
                      .max_area(max_detection_area)
                      .apply())
        return detections
