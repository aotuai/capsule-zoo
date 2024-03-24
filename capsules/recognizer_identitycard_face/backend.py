from typing import Dict

import numpy as np

from vcap import (
    Crop,
    BaseBackend,
    DETECTION_NODE_TYPE,
    BaseStreamState,
    DetectionNode,
    OPTION_TYPE)

from vcap_utils import OpenFaceEncoder


class Backend(OpenFaceEncoder):
    def process_frame(self, frame: np.ndarray,
                      detection_node: DETECTION_NODE_TYPE,
                      options: Dict[str, OPTION_TYPE],
                      state: BaseStreamState) -> DETECTION_NODE_TYPE:
        detections = []
        face_num = len(detection_node)
        if face_num != 2:
            return detections

        # Crop with a 15% padding around the face to emulate how the model was
        # trained.
        prediction = []
        for det in detection_node:
            #crop = Resize(frame).crop_bbox(det.bbox).frame
            pad = 15
            crop = (Crop.from_detection(det)
                    .pad_percent(top=pad, bottom=pad, left=pad, right=pad)
                    .apply(frame))

            prediction.append(self.send_to_batch(crop).result())

        # detection_node.encoding = prediction.vector
        value, confident = self.vector_compare(prediction)

        coords = detection_node[0].coords #.extend(detection_node[1].coords)
        face_node = DetectionNode(
            name="face_compare",
            coords=coords,
            extra_data={"face_compare": value, "confidence": confident})
        detections.append(face_node)

        return detections

    @staticmethod
    def vector_compare(predictions):
        # compare prediction.vertor at here

        return True, 0.5

