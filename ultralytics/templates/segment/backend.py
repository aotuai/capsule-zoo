import threading
from typing import Dict

import numpy as np
from vcap import DETECTION_NODE_TYPE, OPTION_TYPE, BaseStreamState, DetectionNode, rect_to_coords
from vcap_utils import BaseOpenVINOBackend

from .class_head import ClassHeadCutter
from .decode import NEEDS_NMS, decode
from .labels import labels
from .masks import mask_polygons, unletterbox_points
from .nms import class_aware_nms
from . import settings
from .settings import model_includes_preprocessing
from .preprocess import letterbox, to_model_input, unletterbox
from .selection import selected_classes

MAX_OBJECTS = 100
"""The most objects that get a mask in one frame (the best ones)."""


class Backend(BaseOpenVINOBackend):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # A model with thousands of classes (prompt-free) is cut down to the selected classes: see class_head.py
        self._class_head = None
        self._frame_lock = threading.Lock()
        if getattr(settings, "class_head_cut", False):  # older settings.py files do not have it
            self._class_head = ClassHeadCutter(self, len(labels))

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
        selected = selected_classes(options.get("classes"), labels)  # only models with the classes option
        model_input = {input_name: to_model_input(image, model_includes_preprocessing)}
        id_map = None  # the class numbers of the classes the model gives, when its class head was cut
        if self._class_head is not None:
            with self._frame_lock:  # the model must not change between choosing it and its result
                id_map = self._class_head.use(selected)
                outputs = self.send_to_batch(model_input).result()
        else:
            outputs = self.send_to_batch(model_input).result()
        arrays = [np.asarray(o) for o in outputs.values()]
        raw = next(o for o in arrays if o.ndim == 3)  # the detections with the mask coefficients
        protos = next(o for o in arrays if o.ndim == 4)[0]  # the mask prototypes, (masks, H, W)

        if id_map is not None:  # the cut model gives only the selected classes
            boxes, scores, class_ids, coefficients = decode(raw, options["threshold"], len(id_map), protos.shape[0])
            class_ids = id_map[class_ids]
        else:
            extra = {} if selected is None else {"selected": selected}
            boxes, scores, class_ids, coefficients = decode(raw, options["threshold"], len(labels), protos.shape[0], **extra)
        if NEEDS_NMS:
            keep = class_aware_nms(boxes, scores, class_ids, options["iou_threshold"], MAX_OBJECTS)
        else:  # the model removed the duplicate boxes itself
            keep = scores.argsort()[::-1][:MAX_OBJECTS]
        boxes, scores, class_ids, coefficients = boxes[keep], scores[keep], class_ids[keep], coefficients[keep]

        polygons = mask_polygons(protos, coefficients, boxes, (input_h, input_w), options["mask_threshold"])
        frame_boxes = unletterbox(boxes, scale, pad_left, pad_top, frame_w, frame_h)

        nodes = []
        for box, polygon, score, class_id in zip(frame_boxes, polygons, scores, class_ids):
            if polygon is None:  # no usable mask: report the box
                coords = rect_to_coords(box.tolist())
            else:
                points = unletterbox_points(polygon, scale, pad_left, pad_top, frame_w, frame_h)
                coords = np.rint(points).astype(int).tolist()
            nodes.append(DetectionNode(
                name=labels[int(class_id)],
                coords=coords,
                extra_data={"confidence": float(score)},
            ))
        return nodes

    def close(self):
        super().close()
