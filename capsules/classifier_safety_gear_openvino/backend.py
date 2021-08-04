from typing import Dict, List

import numpy as np

from vcap import DETECTION_NODE_TYPE, OPTION_TYPE, BaseStreamState

from vcap_utils import BaseOpenVINOBackend, linear_assignment, iou_cost_matrix

from .config import safety_hat, safety_vest, gear_types

from . import config


class Backend(BaseOpenVINOBackend):
    label_map: Dict[int, str] = {1: "unknown", 2: safety_vest,
                                 3: "unknown", 4: safety_hat}

    def process_frame(self, frame: np.ndarray,
                      detection_node: DETECTION_NODE_TYPE,
                      options: Dict[str, OPTION_TYPE],
                      state: BaseStreamState) -> DETECTION_NODE_TYPE:
        if len(detection_node) == 0:
            return detection_node

        confidence_threshold = options[config.confidence_threshold]

        input_dict, resize = self.prepare_inputs(frame)
        prediction = self.send_to_batch(input_dict).result()
        detections = self.parse_detection_results(
            prediction, resize, self.label_map,
            min_confidence=confidence_threshold)

        for gear_type in gear_types:
            assign_gear_attributes(
                detection_node, detections, gear_type, options)

        return detection_node


def assign_gear_attributes(person_detections: List[DETECTION_NODE_TYPE],
                           gear_detections: List[DETECTION_NODE_TYPE],
                           gear_type: str,
                           options: Dict[str, OPTION_TYPE]):
    """This function assigns DetectionNode.attributes[gear_type] to person
    detections, based on the location of gear detection bounding boxes.
    """
    gear_detections = [g for g in gear_detections if g.class_name == gear_type]

    for person_det in person_detections:
        person_det.attributes[gear_type] = f"without_{gear_type}"
        person_det.extra_data[f"{gear_type}_iou"] = 0
        person_det.extra_data[f"{gear_type}_confidence"] = 0

    if len(gear_detections) == 0:
        return

    # Calculate the 'cost matrix' of every permutation of IOU to behavior
    iou_threshold = options[f"{gear_type}_iou_threshold"]
    iou_cost = iou_cost_matrix(person_detections, gear_detections)
    iou_cost[iou_cost > (1 - iou_threshold)] = 1
    indices = linear_assignment(iou_cost)

    for det_index, gear_index in indices:
        person_det = person_detections[det_index]
        best_match = gear_detections[gear_index]
        cost_iou = iou_cost[det_index][gear_index]
        # People bboxes and safety gear bboxes are aligned with linear
        # assignment. However, this algorithm might align two bboxes with high
        # IoU cost. We want to filter them out.
        if cost_iou >= 1:
            continue
        person_det.attributes[gear_type] = f"with_{gear_type}"
        person_det.extra_data[f"{gear_type}_iou"] = 1 - cost_iou
        person_det.extra_data[f"{gear_type}_confidence"] = \
            best_match.extra_data["detection_confidence"]
