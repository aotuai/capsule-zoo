import logging
from typing import Dict, Tuple
import numpy as np
from .detector_qrcode import QrBarDecoder, CodeResult

from vcap import (
    Resize,
    BaseBackend,
    DetectionNode,
    rect_to_coords,
    BaseStreamState,
    DETECTION_NODE_TYPE,
    OPTION_TYPE, BoundingBox)


class Backend(BaseBackend):
    def process_frame(self, frame: np.ndarray,
                      detection_nodes: DETECTION_NODE_TYPE,
                      options: Dict[str, OPTION_TYPE],
                      state: BaseStreamState) -> DETECTION_NODE_TYPE:
        detections = []

        try:
            mode = options["mode"]
            decoder = QrBarDecoder(timeout_ms=500, annotate_path=None, debug_dir=None)
            with decoder:
                results = decoder.decode_image(frame, mode=mode)
                for ret in results:
                    x, y, w, h = ret.rect
                    block_node = DetectionNode(
                        name="barcode",
                        coords=[[x, y], [x+w, y], [x+w, y+h], [x, y+h]],
                        extra_data={"code": ret.data, "codetype": ret.barcode_type},
                        attributes={})
                    detections.append(block_node)

        except Exception as e:
            logging.error(f"detector_barcode process_frame: {e}")

        return detections



