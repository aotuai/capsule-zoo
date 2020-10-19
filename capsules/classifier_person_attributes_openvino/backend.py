from typing import Dict

import numpy as np
import cv2

from vcap import (
    Resize,
    DETECTION_NODE_TYPE,
    OPTION_TYPE,
    BaseStreamState, FloatOption)
from vcap_utils import BaseOpenVINOBackend

ATTRIBUTES = {
    'gender': ['masculine', 'feminine', 'unknown'],
    'bag': ['has_bag', 'no_bag', 'unknown'],
    'backpack': ['has_backpack', 'no_backpack', 'unknown'],
    'hat': ['has_hat', 'no_hat', 'unknown'],
    'sleeves': ['has_long_sleeves', 'has_short_sleeves', 'unknown'],
    'pants': ['has_long_pants', 'has_short_pants', 'unknown'],
    'hair': ['has_long_hair', 'has_short_hair', 'unknown'],
    'coat_jacket': ['has_coat_jacket', 'no_coat_jacket', 'unknown']
}

# Generate the options based on the attributes
options = {
    f"{attribute}_confidence": FloatOption(
        default=0.3,
        min_val=0.0,
        max_val=1.0,
        description=f"If a person is detected as possibly having the "
                    f"attribute {attribute}, this threshold determines how "
                    f"confident it must be to return a detection.")
    for category, attributes in ATTRIBUTES.items()
    for attribute in attributes
    if attribute != "unknown"
}


class Backend(BaseOpenVINOBackend):

    def process_frame(self, frame: np.ndarray,
                      detection_node: DETECTION_NODE_TYPE,
                      options: Dict[str, OPTION_TYPE],
                      state: BaseStreamState) -> DETECTION_NODE_TYPE:
        crop = Resize(frame).crop_bbox(detection_node.bbox).frame
        input_dict, _ = self.prepare_inputs(crop)
        prediction = self.send_to_batch(input_dict).get()

        prediction = prediction['453'].flatten()
        att = list(ATTRIBUTES.keys())

        # Iterate over predictions and add attributes accordingly
        for i, confidence in enumerate(prediction):
            attribute_name = att[i]
            attribute = ATTRIBUTES[attribute_name][
                0 if confidence >= 0.5 else 1
            ]
            option = f"{attribute}_confidence"

            # The confidence value is remapped to create 2 confidence
            # thresholds for the attribute; one for how confident it is in the
            # upper range, the other for the confidence in the lower range.
            remapped_confidence = abs(confidence - 0.5) * 2
            float_option = options[option]
            detection_node.attributes[attribute_name] = (
                attribute
                if remapped_confidence > float_option else
                ATTRIBUTES[attribute_name][2]
            )
