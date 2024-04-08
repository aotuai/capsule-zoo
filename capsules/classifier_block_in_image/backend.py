import logging
from typing import Dict, Tuple
import numpy as np
from .segment import detect_lines

from vcap import (
    Resize,
    BaseBackend,
    DetectionNode,
    rect_to_coords,
    BaseStreamState,
    DETECTION_NODE_TYPE,
    OPTION_TYPE, BoundingBox)

detection_confidence = "confidence"

class Backend(BaseBackend):
    def process_frame(self, frame: np.ndarray,
                      detection_nodes: DETECTION_NODE_TYPE,
                      options: Dict[str, OPTION_TYPE],
                      state: BaseStreamState) -> DETECTION_NODE_TYPE:
        # detections = []
        # logging.warning(len(detection_nodes))
        # start_time = time.time()
        try:
            det_lines = detect_lines(frame, options)
            hlines, vlines = det_lines.detect_lines()
            # logging.warning(hlines)
            # logging.warning(vlines)
            block_rects = det_lines.get_rectangles(hlines, vlines)
            # for block in block_rects:
            #    logging.warning(block)

            blocks = self.divide_into_blocks(detection_nodes, block_rects, options)
        except Exception as e:
            logging.error(f"classifier_block_in_image: {e}")
        '''
        i = 1
        for rect in block_rects:
            block_node = DetectionNode(
                name="block",
                coords=rect_to_coords(rect),
                extra_data={},
                attributes={"block_name": blocks[i]["block_name"]})
            detections.append(block_node)
            i += 1
        '''
        #for det in detection_nodes:
        #    logging.warning(det)
        #for det in detections:
        #    logging.warning(det)

        return


    def divide_into_blocks(self, detection_nodes, block_rects, options):
        block = [{"detections":[]} for _ in range(len(block_rects)+1)]

        for det in detection_nodes:
            block_no = self.bbox_in_blocks(det.bbox, block_rects)
            #if block_no != -1:
            block[block_no]["block_no"] = block_no
            if not "block_name" in block[block_no]:
                if block_no == 0:
                    block[block_no]["block_name"] = "noname"
                elif block_no == options["ic_block"]:
                    block[block_no]["block_name"] = "identity_card"
                elif block_no == options["signature_block"]:
                    block[block_no]["block_name"] = "signature"
                elif block_no == options["reception_block"]:
                    block[block_no]["block_name"] = "reception"
                elif block_no == options["portrait_block"]:
                    block[block_no]["block_name"] = "portrait"
                else:
                    block[block_no]["block_name"] = f"block_{block_no}"
            # (block[block_no]["detections"]).append(det)
            det.attributes["block_name"] = block[block_no]["block_name"]

        return block

    @staticmethod
    def bbox_in_blocks( bbox, rects):
        i = 0
        for rect in rects:
            i += 1
            if bbox.x1 >= rect[0] and bbox.x1 <= rect[2] and bbox.x2 >= rect[0] and bbox.x2 <= rect[2] and \
                bbox.y1 >= rect[1] and bbox.y1 <= rect[3] and bbox.y2 >= rect[1] and bbox.y2 <= rect[3]:
                return i
        return 0

