from typing import Dict

import numpy as np

from vcap import (
   DETECTION_NODE_TYPE,
   OPTION_TYPE,
   BaseStreamState)
from vcap_utils import BaseOpenVINOBackend


class Backend(BaseOpenVINOBackend):
   label_map: Dict[int, str] = {1: "vehicle", 2: "person", 3: "bike"}

   def process_frame(self, frame: np.ndarray,
                     detection_node: DETECTION_NODE_TYPE,
                     options: Dict[str, OPTION_TYPE],
                     state: BaseStreamState) -> DETECTION_NODE_TYPE:
       input_dict, resize = self.prepare_inputs(frame)
       prediction = self.send_to_batch(input_dict).get()
       detections = self.parse_detection_results(
           prediction, resize, self.label_map,
           min_confidence=options["threshold"])
       return detections
