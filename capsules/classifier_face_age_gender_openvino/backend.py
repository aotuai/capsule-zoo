from typing import Dict

import numpy as np

from vcap import (
    Resize,
    DETECTION_NODE_TYPE,
    OPTION_TYPE,
    BaseStreamState)
from vcap_utils import BaseOpenVINOBackend
from . import config


def _get_age_bin(age: float) -> str:
    bins = config.age_bins
    age_bin = next(iter(bins.keys()))  # set to first bin
    for min_age, category in bins.items():
        if age < min_age:  # if less than current bin
            return age_bin  # return the last bin
        age_bin = category  # otherwise bin to current bin

    return age_bin


class Backend(BaseOpenVINOBackend):

    def process_frame(self, frame: np.ndarray,
                      detection_node: DETECTION_NODE_TYPE,
                      options: Dict[str, OPTION_TYPE],
                      state: BaseStreamState) -> DETECTION_NODE_TYPE:
        crop = Resize(frame).crop_bbox(detection_node.bbox).frame

        input_dict, _ = self.prepare_inputs(crop)
        prediction = self.send_to_batch(input_dict).result()

        age_key = next(key for key in prediction.keys() if 'age_conv3' in key.names)
        prob_key = next(key for key in prediction.keys() if 'prob' in key.names)

        age_data = prediction[age_key]
        if isinstance(age_data, np.ndarray):
            age = int(age_data.flatten()[0] * 100)
        else:
            age = int(age_data * 100)

        prob_data = prediction[prob_key]
        if isinstance(prob_data, np.ndarray):
            gender_id = prob_data.flatten().argmax()
            gender_confidence = float(prob_data.flatten()[gender_id])
        else:
            gender_id = 0 if prob_data[0] > prob_data[1] else 1
            gender_confidence = float(prob_data[gender_id])

        gender = config.genders[gender_id]

        detection_node.extra_data['age'] = age
        detection_node.attributes['gender'] = gender
        detection_node.extra_data['gender_confidence'] = gender_confidence
        detection_node.attributes['age'] = _get_age_bin(age)

