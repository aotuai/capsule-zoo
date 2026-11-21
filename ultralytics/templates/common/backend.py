import threading
from typing import Dict

import numpy as np
from vcap import DETECTION_NODE_TYPE, OPTION_TYPE, BaseStreamState, DetectionNode, rect_to_coords
from vcap_utils import BaseOpenVINOBackend

from .class_head import ClassHeadCutter
from .decode import NEEDS_NMS, decode
from .labels import labels
from .nms import class_aware_nms
from . import settings
from .settings import model_includes_preprocessing
from .preprocess import letterbox, to_model_input, unletterbox
from .selection import selected_classes


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
        # Some exports also list the raw (1, 3, H, W, 5 + classes) head tensors: use the 3-D detection output
        raw = next(np.asarray(o) for o in outputs.values() if np.ndim(o) == 3)

        if id_map is not None:  # the cut model gives only the selected classes
            boxes, scores, class_ids = decode(raw, options["threshold"], len(id_map), (input_h, input_w))
            class_ids = id_map[class_ids]
        else:
            extra = {} if selected is None else {"selected": selected}
            boxes, scores, class_ids = decode(raw, options["threshold"], len(labels), (input_h, input_w), **extra)
        if NEEDS_NMS:
            keep = class_aware_nms(boxes, scores, class_ids, options["iou_threshold"], 1000)
        else:  # the model removed the duplicate boxes itself
            keep = scores.argsort()[::-1]
        boxes = unletterbox(boxes[keep], scale, pad_left, pad_top, frame_w, frame_h)

        return [
            DetectionNode(
                name=labels[int(class_id)],
                coords=rect_to_coords(box.tolist()),
                extra_data={"confidence": float(score)},
            )
            for box, score, class_id in zip(boxes, scores[keep], class_ids[keep])
        ]

    def close(self):
        super().close()
