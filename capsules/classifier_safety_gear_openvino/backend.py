from typing import Dict, Union

import numpy as np

from vcap import (
    DETECTION_NODE_TYPE,
    OPTION_TYPE,
    BaseStreamState,
)

from vcap_utils import (
    BaseOpenVINOBackend,
    linear_assignment,
    iou_cost_matrix,
)

from .config import (
    safety_hat,
    safety_vest,
    safety_gears,
    attributes,
)

from . import config

Num = Union[int, float]


class Backend(BaseOpenVINOBackend):
    label_map: Dict[int, str] = {1: "unknown", 2: safety_vest,
                                 3: "unknown", 4: safety_hat}

    def process_frame(self, frame: np.ndarray,
                      detection_nodes: DETECTION_NODE_TYPE,
                      options: Dict[str, OPTION_TYPE],
                      state: BaseStreamState) -> DETECTION_NODE_TYPE:
        if len(detection_nodes) == 0:
            return detection_nodes

        confidence_threshold = options[config.confidence_threshold]

        input_dict, resize = self.prepare_inputs(frame)
        prediction = self.send_to_batch(input_dict).get()
        detections = self.parse_detection_results(
            prediction, resize, self.label_map,
            min_confidence=confidence_threshold)

        safety_gear_detections = {}
        for gear_type in safety_gears:
            safety_gear_detections[gear_type] = \
                [detection for detection in detections if
                 detection.class_name == gear_type]

        for det in detection_nodes:
            for safety_gear in safety_gears:
                det.attributes[attributes[safety_gear]["attribute_name"]] = \
                    attributes[safety_gear]["possible_values"][0]
                det.extra_data[attributes[safety_gear]["iou"]] = 0
                det.extra_data[attributes[safety_gear]["confidence"]] = 0

        if len(detections) == 0:
            return detection_nodes

        for gear_type in safety_gears:
            dets = safety_gear_detections[gear_type]
            if len(dets) == 0:
                continue
            iou_threshold = options[attributes[gear_type]["iou_threshold"]]
            iou_cost = iou_cost_matrix(detection_nodes, dets)
            iou_cost[iou_cost > (1 - iou_threshold)] = 1
            indices = linear_assignment(iou_cost)

            for det_index, gear_index in indices:
                det = detection_nodes[det_index]
                best_match = dets[gear_index]
                cost_iou = iou_cost[det_index][gear_index]
                if cost_iou >= 1:
                    continue
                det.extra_data[attributes[gear_type]["confidence"]] = \
                    best_match.extra_data["detection_confidence"]
                det.extra_data[attributes[gear_type]["iou"]] = 1 - cost_iou
                det.attributes[attributes[gear_type]["attribute_name"]] = \
                    attributes[gear_type]["possible_values"][1]

        return detection_nodes
