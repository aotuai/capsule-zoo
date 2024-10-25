from typing import Dict

import numpy as np

from vcap import (
    Crop,
    DETECTION_NODE_TYPE,
    OPTION_TYPE,
    BaseStreamState)
from vcap_utils import BaseOpenVINOBackend
from vcap import __version__ as vcap_version
def openvino2024_compatible():
    from packaging import version
    from vcap import __version__ as vcap_version
    openvino2024_compatible_vcap_version = '0.3.8'
    return version.parse(vcap_version) >= version.parse(openvino2024_compatible_vcap_version)
__openvino2024__ = openvino2024_compatible()


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
        if __openvino2024__:
            prob_key = next(key for key in prediction.keys() if 'fc5' in key.names)
            prob_data = prediction[prob_key]
            if isinstance(prob_data, np.ndarray):
                probability = float(prob_data.flatten()[0])
            else:
                probability = float(prob_data[0])
        else:
            probability = prediction["fc5"].flatten()[0]

        threshold = options["threshold"]
        label = self.LABELS[int(probability > threshold)]

        detection_node.attributes["mask"] = label
        detection_node.extra_data["mask_confidence"] = float(probability)
