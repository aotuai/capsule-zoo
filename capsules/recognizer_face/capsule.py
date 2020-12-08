from typing import Dict

import numpy as np

from vcap import (
    BaseCapsule,
    NodeDescription,
    Crop,
    FloatOption,
    DETECTION_NODE_TYPE,
    BaseStreamState,
    OPTION_TYPE)
from vcap_utils import OpenFaceEncoder


class Backend(OpenFaceEncoder):
    def process_frame(self, frame: np.ndarray,
                      detection_node: DETECTION_NODE_TYPE,
                      options: Dict[str, OPTION_TYPE],
                      state: BaseStreamState) -> DETECTION_NODE_TYPE:
        # Crop with a 15% padding around the face to emulate how the model was
        # trained.
        pad = 15
        crop = (Crop.from_detection(detection_node)
                .pad_percent(top=pad, bottom=pad, left=pad, right=pad)
                .apply(frame))

        prediction = self.send_to_batch(crop).result()

        detection_node.encoding = prediction.vector


class Capsule(BaseCapsule):
    name = "recognizer_face"
    description = "✨ Recognize faces. Works best close-up."
    version = 2
    input_type = NodeDescription(
        size=NodeDescription.Size.SINGLE,
        detections=["face"])
    output_type = NodeDescription(
        size=NodeDescription.Size.SINGLE,
        detections=["face"],
        encoded=True)
    backend_loader = lambda capsule_files, device: Backend(
        device=device,
        model_bytes=capsule_files["encoder_face_center_loss.pb"],
        model_name="vggface2_center_loss")
    options = {
        "recognition_threshold": FloatOption(
            default=0.8,
            min_val=0.0,
            max_val=None)
    }
