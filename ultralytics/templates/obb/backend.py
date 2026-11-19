from typing import Dict

import numpy as np
from vcap import DETECTION_NODE_TYPE, OPTION_TYPE, BaseStreamState, DetectionNode
from vcap_utils import BaseOpenVINOBackend

from .decode import NEEDS_NMS, decode
from .labels import labels
from .obbs import corners, rotated_nms, to_frame
from .settings import model_includes_preprocessing
from .preprocess import letterbox, to_model_input


class Backend(BaseOpenVINOBackend):
    def process_frame(
        self,
        frame: np.ndarray,
        detection_node: DETECTION_NODE_TYPE,
        options: Dict[str, OPTION_TYPE],
        state: BaseStreamState,
    ) -> DETECTION_NODE_TYPE:
        if frame is None:
            return None

        input_name = self.input_blob_names[0]
        _, _, input_h, input_w = self.compiled_model.input(input_name).shape
        frame_h, frame_w = frame.shape[:2]

        image, scale, pad_left, pad_top = letterbox(frame, input_w, input_h)
        outputs = self.send_to_batch({input_name: to_model_input(image, model_includes_preprocessing)}).result()
        raw = next(np.asarray(o) for o in outputs.values() if np.ndim(o) == 3)

        rboxes, scores, class_ids = decode(raw, options["threshold"], len(labels))
        if NEEDS_NMS:
            keep = rotated_nms(rboxes, scores, class_ids, options["iou_threshold"])
        else:  # the model removed the duplicate boxes itself
            keep = scores.argsort()[::-1]
        rboxes, scores, class_ids = rboxes[keep], scores[keep], class_ids[keep]

        # the four corners of each box in frame pixels: the polygon of the detection
        polygons = to_frame(corners(rboxes), scale, pad_left, pad_top, frame_w, frame_h)
        return [
            DetectionNode(
                name=labels[int(class_id)],
                coords=np.rint(polygon).astype(int).tolist(),
                extra_data={"confidence": float(score), "angle": round(float(np.degrees(rbox[4])), 1)},
            )
            for polygon, rbox, score, class_id in zip(polygons, rboxes, scores, class_ids)
        ]

    def close(self):
        super().close()
