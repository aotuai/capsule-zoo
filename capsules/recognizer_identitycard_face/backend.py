from typing import Dict

import numpy as np

from vcap import (
    Crop,
    BaseBackend,
    DETECTION_NODE_TYPE,
    BaseStreamState,
    DetectionNode,
    OPTION_TYPE)

from vcap_utils import OpenFaceEncoder, cosine_distance


class Backend(OpenFaceEncoder):
    def process_frame(self, frame: np.ndarray,
                      detection_node: DETECTION_NODE_TYPE,
                      options: Dict[str, OPTION_TYPE],
                      state: BaseStreamState) -> DETECTION_NODE_TYPE:
        detections = []
        #face_num = len(detection_node)
        #if face_num != 2:
        #    return detections

        # Crop with a 15% padding around the face to emulate how the model was
        # trained.
        prediction = []
        for det in detection_node:
            # crop = Resize(frame).crop_bbox(det.bbox).frame
            pad = 15
            crop = (Crop.from_detection(det)
                    .pad_percent(top=pad, bottom=pad, left=pad, right=pad)
                    .apply(frame))

            prediction.append(self.send_to_batch(crop).result())

        # detection_node.encoding = prediction.vector
        value, confident = self.vector_compare(prediction, options["recognition_threshold"])

        coords = detection_node[0].coords  # .extend(detection_node[1].coords)
        face_node = DetectionNode(
            name="face_compare",
            coords=coords,
            extra_data={"face_compare": value, "confidence": confident})
        detections.append(face_node)

        return detections

    @staticmethod
    def vector_compare(predictions, recognition_threshold=0.5):
        # compare prediction.vertor at here
        if len(predictions) != 2:
            return False, 0.0

        identity_vec = np.array([predictions[0].vector])
        candidate_vec = np.array([predictions[1].vector])

        '''
        # cosine similarity [0~10], the greater the distance value, the more similar it is.
        distances = cosine_distance(candidate_vec, identity_vec)
        print(distances)
        distances = abs(distances)
        lowest_index = np.argmin(distances)
        lowest_distance = distances[lowest_index][0]
        print(f"cosine_distance: {lowest_distance}")
        return lowest_distance < recognition_threshold, lowest_distance
        '''
        # Euclidean distance [0~2], the smaller the distance value, the more similar it is.
        distance = np.linalg.norm( candidate_vec - identity_vec)
        #print(f"norm: {distance}")
        return distance < recognition_threshold, distance


