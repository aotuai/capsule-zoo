from typing import Dict

import numpy as np
from vcap import DETECTION_NODE_TYPE, OPTION_TYPE, BaseStreamState, Resize
from vcap_utils import BaseOpenVINOBackend

from .classify import best_class, fit_to_model
from .labels import labels
from .settings import model_includes_preprocessing
from .preprocess import to_model_input
from .selection import selected_classes


class Backend(BaseOpenVINOBackend):
    def __init__(self, *args, attribute_name: str, **kwargs):
        super().__init__(*args, **kwargs)
        self.attribute_name = attribute_name

    def process_frame(
        self,
        frame: np.ndarray,
        detection_node: DETECTION_NODE_TYPE,
        options: Dict[str, OPTION_TYPE],
        state: BaseStreamState,
    ) -> DETECTION_NODE_TYPE:
        if frame is None or detection_node is None:
            return None

        selected = selected_classes(options.get("classes"), labels)  # only models with the classes option
        if selected is not None and selected.size == 0:
            return None  # every name typed is unknown: no class can be chosen

        crop = Resize(frame).crop_bbox(detection_node.bbox).frame
        if crop.size == 0:
            return None

        input_name = self.input_blob_names[0]
        _, _, input_h, input_w = self.compiled_model.input(input_name).shape
        image = fit_to_model(crop, input_w, input_h, options["fit"])
        outputs = self.send_to_batch({input_name: to_model_input(image, model_includes_preprocessing)}).result()
        output = next(np.asarray(o) for o in outputs.values())

        class_id, confidence = best_class(output, len(labels), selected)
        if confidence >= options["min_confidence"]:
            detection_node.attributes[self.attribute_name] = labels[class_id]
            detection_node.extra_data[self.attribute_name + "_confidence"] = confidence

    def close(self):
        super().close()
