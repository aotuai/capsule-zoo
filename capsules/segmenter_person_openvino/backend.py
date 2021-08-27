from typing import Dict, List

import numpy as np
import cv2

from vcap import (
    DetectionNode,
    Resize,
    DETECTION_NODE_TYPE,
    OPTION_TYPE,
    BaseStreamState,
    rect_to_coords)
from vcap_utils import BaseOpenVINOBackend


class Backend(BaseOpenVINOBackend):
    label_map = {
        1: "person"
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        input_blob_name = self.input_blob_names[0]
        input_blob = self.net.inputs[input_blob_name]
        _, _, h, w = input_blob.shape

        self.blob_h = h
        self.blob_w = w

    def process_frame(self, frame: np.ndarray,
                      detection_node: DETECTION_NODE_TYPE,
                      options: Dict[str, OPTION_TYPE],
                      state: BaseStreamState) -> DETECTION_NODE_TYPE:
        frame_h, frame_w, _ = frame.shape

        if options["keep_aspect_ratio"]:
            # Resize the image to keep the same aspect ratio and to
            # fit it to a window of a target size.
            scale_x = scale_y = min(self.blob_h / frame_h,
                                    self.blob_w / frame_w)
            input_image = cv2.resize(frame, None, fx=scale_x, fy=scale_y)
        else:
            # Resize the image to a target size.
            scale_x = self.blob_w / frame_w
            scale_y = self.blob_h / frame_h
            input_image = cv2.resize(frame, (self.blob_w, self.blob_h))
        input_image_size = input_image.shape[:2]
        input_image = np.pad(input_image,
                             ((0, self.blob_h - input_image_size[0]),
                              (0, self.blob_w - input_image_size[1]),
                              (0, 0)),
                             mode='constant', constant_values=0)
        # Change data layout from HWC to CHW.
        input_image = input_image.transpose((2, 0, 1))
        input_image = input_image.reshape(
            (1, 3, self.blob_h, self.blob_w)).astype(np.float32)
        input_image_info = np.asarray(
            [[input_image_size[0], input_image_size[1], 1]], dtype=np.float32)

        # Run inference
        input_dict = {
            "im_data": input_image,
            "im_info": input_image_info
        }

        prediction = self.send_to_batch(input_dict).get()

        detections = self.parse_results(
            prediction=prediction,
            scale_x=scale_x,
            scale_y=scale_y,
            threshold=0.85,
            input_h=frame_h,
            input_w=frame_w,
            as_bbox=options["convert_to_bounding_box"])

        return list(detections)

    def parse_results(self,
                      prediction,
                      scale_x: float,
                      scale_y: float,
                      input_w: int,
                      input_h: int,
                      threshold: float,
                      as_bbox: bool) \
            -> List[DetectionNode]:
        boxes = prediction['boxes']
        boxes[:, 0::2] /= scale_x
        boxes[:, 1::2] /= scale_y
        scores = prediction['scores']
        classes = prediction['classes'].astype(np.uint32)

        if as_bbox:
            detections_filter = scores > threshold
            scores = scores[detections_filter]
            classes = classes[detections_filter]
            boxes = boxes[detections_filter]

            for box, class_id, score in zip(boxes, classes, scores):
                if class_id not in self.label_map:
                    continue
                yield DetectionNode(
                    name=self.label_map[class_id],
                    coords=rect_to_coords(box.tolist()),
                    extra_data={"detection_confidence": float(score)}
                )
        else:
            zipped = zip(boxes, classes, prediction['raw_masks'], scores)
            for box, cls, raw_mask, score in zipped:
                if cls not in self.label_map or score < threshold:
                    continue

                raw_cls_mask = raw_mask[cls, ...]
                mask = Backend.segm_postprocess(
                    box, raw_cls_mask, input_h, input_w)
                contours, _ = cv2.findContours(
                    mask.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                largest_contour = max(contours, key=cv2.contourArea).tolist()
                largest_contour = [c[0] for c in largest_contour]

                yield DetectionNode(
                    name=self.label_map[cls],
                    coords=largest_contour,
                    extra_data={"detection_confidence": float(score)}
                )

    @staticmethod
    def segm_postprocess(box, raw_cls_mask, im_h, im_w):
        # Add zero border to prevent upsampling artifacts on segment borders.
        raw_cls_mask = np.pad(raw_cls_mask, ((1, 1), (1, 1)), 'constant',
                              constant_values=0)
        extended_box = Backend.expand_box(box, raw_cls_mask.shape[0] / (
                raw_cls_mask.shape[0] - 2.0)).astype(int)
        w, h = np.maximum(extended_box[2:] - extended_box[:2] + 1, 1)
        x0, y0 = np.clip(extended_box[:2], a_min=0, a_max=[im_w, im_h])
        x1, y1 = np.clip(extended_box[2:] + 1, a_min=0, a_max=[im_w, im_h])

        raw_cls_mask = cv2.resize(raw_cls_mask, (w, h)) > 0.5
        mask = raw_cls_mask.astype(np.uint8)
        # Put an object mask in an image mask.
        im_mask = np.zeros((im_h, im_w), dtype=np.uint8)
        im_mask[y0:y1, x0:x1] = mask[
                                (y0 - extended_box[1]):(y1 - extended_box[1]),
                                (x0 - extended_box[0]):(x1 - extended_box[0])]
        return im_mask

    @staticmethod
    def expand_box(box, scale):
        w_half = (box[2] - box[0]) * .5
        h_half = (box[3] - box[1]) * .5
        x_c = (box[2] + box[0]) * .5
        y_c = (box[3] + box[1]) * .5
        w_half *= scale
        h_half *= scale
        box_exp = np.zeros(box.shape)
        box_exp[0] = x_c - w_half
        box_exp[2] = x_c + w_half
        box_exp[1] = y_c - h_half
        box_exp[3] = y_c + h_half
        return box_exp
