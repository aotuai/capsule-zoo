import numpy as np
from typing import Dict

from vcap import (
    DetectionNode,
    rect_to_coords,
    DETECTION_NODE_TYPE,
    BaseStreamState)
from vcap.options import OPTION_TYPE
from vcap_utils import (
    TFObjectDetector,
    detection_iou,
    linear_assignment,
    iou_cost_matrix)

from . import config


class Backend(TFObjectDetector):
    def process_frame(self, frame: np.ndarray,
                      detection_nodes: DETECTION_NODE_TYPE,
                      options: Dict[str, OPTION_TYPE],
                      state: BaseStreamState) -> DETECTION_NODE_TYPE:
        if len(detection_nodes) == 0:
            return detection_nodes

        confidence_threshold = options[config.confidence_threshold]
        iou_threshold = options[config.iou_threshold]

        prediction = self.send_to_batch(frame).result()

        behavior_detections = []
        for pred in prediction:
            if pred.confidence < confidence_threshold:
                continue
            if pred.name in config.ignore:
                continue
            det = DetectionNode(
                name=pred.name,
                coords=rect_to_coords(pred.rect),
                extra_data={config.pose_confidence: pred.confidence})
            behavior_detections.append(det)

        # Fill all detections with 'unknown' data
        for det in detection_nodes:
            det.attributes[config.pose] = "unknown"
            det.extra_data[config.pose_confidence] = 0
            det.extra_data[config.pose_iou] = 0

        # Exit early if there are no behavior detections (also empty lists cause
        # later lines to fail)
        if len(behavior_detections) == 0:
            return detection_nodes

        # Calculate the 'cost matrix' of every permutation of IOU to behavior
        iou_cost = iou_cost_matrix(detection_nodes, behavior_detections)
        iou_cost[iou_cost > (1 - iou_threshold)] = 1
        indices = linear_assignment(iou_cost)

        for det_index, beh_index in indices:
            det = detection_nodes[det_index]
            best_match = behavior_detections[beh_index]
            pose_confidence = best_match.extra_data[config.pose_confidence]
            if det.extra_data[config.pose_confidence] < pose_confidence:
                pose_iou = detection_iou(det, [best_match])
                cost_iou = iou_cost[det_index][beh_index]
                if cost_iou >= 1:
                    continue
                det.attributes[config.pose] = best_match.class_name
                det.extra_data[config.pose_confidence] = pose_confidence
                det.extra_data[config.pose_iou] = pose_iou

        # If you want to see the behavior detections as well, uncomment this
        # for b in behavior_detections:
        #     b.attributes[opts.pose] = b.class_name
        #     b.extra_data[opts.pose_iou] = 0
        # self.detection_nodes += behavior_detections
        return detection_nodes
