import logging
from typing import Dict, Tuple
import numpy as np
import tempfile
from pathlib import Path
import time
import cv2

from vcap import (
    Resize,
    BaseBackend,
    DetectionNode,
    rect_to_coords,
    DETECTION_NODE_TYPE,
    OPTION_TYPE, BoundingBox)

from .ocr_det_rec import OcrDetRec, get_onet_sessions, get_character_dict
from .ocr_state import StreamState

detection_confidence = "confidence"


class Backend(BaseBackend):
    def __init__(self, det_bytes, rec_bytes, dict_bytes,
                 device: str = None):
        super().__init__()

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir_path = Path(temp_dir)

            det_onnx_file = str(temp_dir_path) + "/det.onnx"
            det_onnx = open(det_onnx_file, 'wb')
            det_onnx.write(det_bytes)
            det_onnx.close()

            rec_onnx_file = str(temp_dir_path) + "/rec.onnx"
            rec_onnx = open(rec_onnx_file, 'wb')
            rec_onnx.write(rec_bytes)
            rec_onnx.close()

            dict_character_file = str(temp_dir_path) + "/dict_characters.txt"
            dict_character = open(dict_character_file, 'wb')
            dict_character.write(dict_bytes)
            dict_character.close()

            self.onet_det_session, self.onet_rec_session = get_onet_sessions(det_onnx_file, rec_onnx_file)
            self.dict_character = get_character_dict(dict_character_file)

    def process_frame(self, frame: np.ndarray,
                      detection_nodes: DETECTION_NODE_TYPE,
                      options: Dict[str, OPTION_TYPE],
                      state: StreamState) -> DETECTION_NODE_TYPE:
        detections = []

        try:
            start_time = time.time()
            is_cell = options["cell"]
            to_gray = options["to_gray"]
            if to_gray:
                frame1 = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
                frame = cv2.cvtColor(frame1, cv2.COLOR_GRAY2BGR)
            cell_x, cell_y = 0, 0
            if is_cell:
                cell_x = options["cell_x"]
                cell_y = options["cell_y"]
                cell_width = options["cell_width"]
                cell_height = options["cell_height"]
                cell_bbox = BoundingBox(cell_x, cell_y, cell_x + cell_width, cell_y + cell_height)
                frame = Resize(frame).crop_bbox(cell_bbox).frame

                # if state.is_similar_frame(frame):
                #     state.update_last_frame(frame)
                #     detections = []
                #     detections.extend(state.get_last_detections())
                #     return detections

            # Setup OCR-检测-识别 System
            ocr_sys = OcrDetRec(self.onet_det_session, self.onet_rec_session, self.dict_character)

            # OCR-检测-识别
            ocr_sys.ocr_det_rec_img(frame)

            # 得到检测框
            dt_boxes = ocr_sys.get_boxes()

            # 识别 results: 单纯的识别结果，results_info: 识别结果+置信度
            results, results_info = ocr_sys.recognition_img(dt_boxes)

            raw_detections = self.create_ocr_text_detections(dt_boxes, results, is_cell, (cell_x, cell_y))
            detections = [d for d in raw_detections if d is not None]

            end_time = time.time()
            end_time = end_time - start_time

            '''
            if end_time <= 3:
                time.sleep(3)
            elif end_time <= 6:
                time.sleep(round(9-end_time))
            elif end_time <= 9:
                time.sleep(round(12-end_time))
            '''

            if is_cell:
                state.update_last_frame(frame, detections)
        except Exception as e:
            logging.warning(f"Failed to process ocr frame, {e}")

        return detections

    def create_ocr_text_detections(self, boxes, txts, is_cell, offset: Tuple[int, int]):
        detections = []

        if txts is None or len(txts) != len(boxes):
            txts = [None] * len(boxes)

        for idx, (box, txt) in enumerate(zip(boxes, txts)):
            extra_data = {"ocr": txt[0],
                          detection_confidence: float(txt[1]),
                          "serial_no": idx}
            if is_cell:
                offset_x, offset_y = offset
            else:
                offset_x, offset_y = 0, 0
            x1, y1 = int(box[0][0]) + offset_x, int(box[0][1]) + offset_y
            x2, y2 = int(box[2][0]) + offset_x, int(box[2][1]) + offset_y

            box_rect = [x1, y1, x2, y2]
            detection = DetectionNode(name='text',
                                      coords=rect_to_coords(box_rect),
                                      extra_data=extra_data)

            detections.append(detection)

        return detections
