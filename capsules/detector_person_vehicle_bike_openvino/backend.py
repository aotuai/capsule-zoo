from typing import Dict

import numpy as np

from vcap import (
    DETECTION_NODE_TYPE,
    OPTION_TYPE,
    BaseStreamState,
    SizeFilter)
from vcap_utils import BaseOpenVINOBackend, non_max_suppression


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

        # Filter out detections based on capsule options
        include_person_detections = options["include_person_detections"]
        include_vehicle_detections = options["include_vehicle_detections"]
        include_bike_detections = options["include_bike_detections"]
        detections = [
            d for d in detections
            if (include_person_detections and d.class_name == "person")
               or (include_vehicle_detections and d.class_name == "vehicle")
               or (include_bike_detections and d.class_name == "bike")
        ]

        # Remove too small and too big detections
        min_detection_area = options["min_detection_area"]
        max_detection_area = options["max_detection_area"]
        detections = (SizeFilter(detections)
                      .min_area(min_detection_area)
                      .max_area(max_detection_area)
                      .apply())

        # Remove overlapping detections
        max_detection_overlap = options["max_detection_overlap"]
        if len(detections) > 0:
            detections = non_max_suppression(detections, max_detection_overlap)
        return detections
