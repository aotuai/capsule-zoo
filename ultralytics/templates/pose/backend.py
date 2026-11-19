from typing import Dict

import numpy as np
from vcap import DETECTION_NODE_TYPE, OPTION_TYPE, BaseStreamState, DetectionNode, rect_to_coords
from vcap_utils import BaseOpenVINOBackend

from .decode import NEEDS_NMS, decode
from .keypoints import keypoint_dim, keypoint_names
from .labels import labels
from .nms import class_aware_nms
from .poses import keypoint_data, to_frame
from .settings import model_includes_preprocessing
from .preprocess import letterbox, to_model_input, unletterbox


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

        boxes, scores, class_ids, keypoints = decode(
            raw, options["threshold"], len(labels), len(keypoint_names), keypoint_dim)
        if NEEDS_NMS:
            keep = class_aware_nms(boxes, scores, class_ids, options["iou_threshold"], 300)
        else:  # the model removed the duplicate boxes itself
            keep = scores.argsort()[::-1]
        boxes = unletterbox(boxes[keep], scale, pad_left, pad_top, frame_w, frame_h)

        return [
            DetectionNode(
                name=labels[int(class_id)],
                coords=rect_to_coords(box.tolist()),
                extra_data={
                    "confidence": float(score),
                    "keypoints": keypoint_data(
                        to_frame(points, scale, pad_left, pad_top, frame_w, frame_h),
                        keypoint_names, options["keypoint_threshold"]),
                },
            )
            for box, score, class_id, points in zip(boxes, scores[keep], class_ids[keep], keypoints[keep])
        ]

    def close(self):
        super().close()
