from typing import Dict

import numpy as np

from vcap import (
    Crop,
    DETECTION_NODE_TYPE,
    OPTION_TYPE,
    BaseStreamState)
from vcap_utils import BaseOpenVINOBackend


class Backend(BaseOpenVINOBackend):
    LABELS = ["not_wearing_mask", "wearing_mask"]

    def process_frame(self, frame: np.ndarray,
                      detection_node: DETECTION_NODE_TYPE,
                      options: Dict[str, OPTION_TYPE],
                      state: BaseStreamState) -> DETECTION_NODE_TYPE:
        crop = (Crop.from_detection(detection_node)
                .pad_percent(top=10, bottom=10, left=10, right=10)
                .apply(frame))

        input_dict, _ = self.prepare_inputs(crop)
        prediction = self.send_to_batch(input_dict).result()

        # Convert prediction to a label
        probability = prediction["fc5"].flatten()[0]
        threshold = options["threshold"]
        label = self.LABELS[int(probability > threshold)]

        detection_node.attributes["mask"] = label
        detection_node.extra_data["mask_confidence"] = float(probability)
