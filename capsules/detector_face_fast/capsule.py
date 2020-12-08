from typing import Dict

import numpy as np

from vcap import (
    BaseCapsule,
    NodeDescription,
    DetectionNode,
    common_detector_options,
    rect_to_coords,
    DETECTION_NODE_TYPE,
    BaseStreamState,
    OPTION_TYPE)
from vcap_utils import TFObjectDetector

detection_confidence = "detection_confidence"


class Backend(TFObjectDetector):
    def process_frame(self, frame: np.ndarray,
                      detection_node: DETECTION_NODE_TYPE,
                      options: Dict[str, OPTION_TYPE],
                      state: BaseStreamState) -> DETECTION_NODE_TYPE:
        prediction = self.send_to_batch(frame).result()
        return [DetectionNode(
            name=det.name,
            coords=rect_to_coords(det.rect),
            extra_data={detection_confidence: det.confidence})
            for det in prediction
            if det.name == "face" and det.confidence >= options["threshold"]]


class Capsule(BaseCapsule):
    name = "detector_face_fast"
    description = "✨ Efficiently detect faces in most environments."
    version = 1
    input_type = NodeDescription(size=NodeDescription.Size.NONE)
    output_type = NodeDescription(
        size=NodeDescription.Size.ALL,
        detections=["face"],
        extra_data=[detection_confidence])
    backend_loader = lambda capsule_files, device: Backend(
        device=device,
        model_bytes=capsule_files["detector.pb"],
        metadata_bytes=capsule_files["dataset_metadata.json"])
    options = common_detector_options
