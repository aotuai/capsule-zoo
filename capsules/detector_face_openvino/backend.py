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
    label_map: Dict[int, str] = {1: "face"}

    def process_frame(self, frame: np.ndarray,
                      detection_node: DETECTION_NODE_TYPE,
                      options: Dict[str, OPTION_TYPE],
                      state: BaseStreamState) -> DETECTION_NODE_TYPE:
        prediction = self.send_to_batch(frame).get()
        return [pred for pred in prediction
                if pred.extra_data["confidence"] > options["threshold"]]

    def parse_results(self, results: np.ndarray, resize: Resize) -> object:

        output_blob_name = list(self.net.outputs.keys())[0]
        inference_results = results[output_blob_name]

        input_blob_name = list(self.net.inputs.keys())[0]
        _, _, h, w = self.net.inputs[input_blob_name].shape

        nodes: List[DetectionNode] = []
        for result in inference_results[0][0]:
            # If the first index == 0, that's the end of real predictions
            # The network always outputs an array of length 200 even if it does
            # not have that many predictions
            if result[0] != 0:
                break

            class_id = round(result[1])

            class_name = self.label_map[class_id]

            x_min, y_min, x_max, y_max = result[3:7]
            # x and y in res are in terms of percent of image width/height
            x_min, x_max = x_min * w, x_max * w
            y_min, y_max = y_min * h, y_max * h
            coords = [[x_min, y_min], [x_max, y_min],
                      [x_max, y_max], [x_min, y_max]]

            confidence = float(result[2])

            res = DetectionNode(name=class_name,
                                coords=coords,
                                extra_data={"confidence": confidence})
            nodes.append(res)

        resize.scale_and_offset_detection_nodes(nodes)
        return nodes
