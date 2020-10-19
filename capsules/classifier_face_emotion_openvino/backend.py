from collections import namedtuple
from typing import Dict, List, Any

import numpy as np

from vcap import (
    Resize,
    DETECTION_NODE_TYPE,
    OPTION_TYPE,
    BaseStreamState)
from vcap_utils import BaseOpenVINOBackend

EMOTION_TYPES = ['neutral', 'happy', 'sad', 'surprise', 'anger']


class Backend(BaseOpenVINOBackend):

    def process_frame(self, frame: np.ndarray,
                      detection_node: DETECTION_NODE_TYPE,
                      options: Dict[str, OPTION_TYPE],
                      state: BaseStreamState) -> DETECTION_NODE_TYPE:
        crop = Resize(frame).crop_bbox(detection_node.bbox).frame

        input_dict, _ = self.prepare_inputs(crop)
        prediction = self.send_to_batch(input_dict).get()

        emotion_id = prediction["prob_emotion"].argmax()
        emotion = EMOTION_TYPES[emotion_id]
        emotion_score = prediction["prob_emotion"].flatten()[emotion_id]

        detection_node.attributes["emotion"] = emotion
        detection_node.extra_data["emotion_confidence"] = emotion_score
