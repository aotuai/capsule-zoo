import json
from vcap import BaseBackend, DETECTION_NODE_TYPE, OPTION_TYPE, DetectionNode, BoundingBox, Resize, rect_to_coords
from typing import Dict
import numpy as np
import logging
from .stream_state import StreamState

out_class_name = "filter_confirmed"

class Backend(BaseBackend):

    def process_frame(self, frame: np.ndarray,
                      detection_node: DETECTION_NODE_TYPE,
                      options: Dict[str, OPTION_TYPE],
                      state: StreamState) -> DETECTION_NODE_TYPE:

        if options["attributes_category"] is None or options["attributes_values"] is None:
            return []
        try:
            values = json.loads(options['attributes_values'])
            state.set_window(options['time_window_duration'], options['true_counter'],
                             options['attributes_category'], values)
        except Exception as e:
            logging.error(f"{e}")
            return []

        detections = []
        if detection_node:
            try:
                for det in detection_node:
                    if type(det) != dict:
                        class_name = det.class_name
                        attributes = det.attributes
                        coords = det.coords
                    else:
                        class_name = det["class_name"]
                        attributes = det["attributes"]
                        coords = det["coords"]
                    if class_name == options["class_name"]:
                        if state.category in attributes.keys():
                            value = attributes[state.category]
                            if value in state.values:
                                if state.add_true_counter(value):
                                    extra_data = {state.category: value}
                                    node = DetectionNode(name=out_class_name, coords=coords,
                                                         extra_data={out_class_name: extra_data})
                                    logging.info(f"classification_detections_filter: {node}")
                                    detections.append(node)

            except Exception as e:
                logging.error(f"{e}")

        return detections
