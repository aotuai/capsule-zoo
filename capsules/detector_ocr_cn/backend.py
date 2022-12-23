import logging
from typing import Dict, List, Any
from time import time

import numpy as np
import tempfile
from pathlib import Path

from vcap import (
    Resize,
    BaseBackend,
    DetectionNode,
    rect_to_coords,
    BaseStreamState,
    DETECTION_NODE_TYPE,
    OPTION_TYPE)

from .ocr_det_rec import OcrDetRec, get_onet_sessions, get_character_dict

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
                      state: BaseStreamState) -> DETECTION_NODE_TYPE:

        screen_enable = options["screen"]
        cell_enable   = options["cell"]
        object_enable = options["object"]
        noise_enable  = options["noise"]

        logging.info(f"OCR INPUT: {detection_nodes}")
        textzones = []
        if  (screen_enable is True and detection_nodes.class_name == 'screen') or \
            (cell_enable   is True and detection_nodes.class_name in ['home_cell', 'roaming_cell']) or \
            (object_enable is True and detection_nodes.class_name in ['home_cell object', 'roaming_cell object']) or \
            (noise_enable  is True and detection_nodes.class_name in ['home_cell object noise', 'roaming_cell object noise']):

            textzones.append(detection_nodes)

        logging.info(f"OCR textzones: {textzones}")
        detections = []
        for textzone in textzones:
            crop = Resize(frame).crop_bbox(textzone.bbox).frame

            time1 = time()
    
            # Setup OCR-检测-识别 System
            ocr_sys = OcrDetRec(self.onet_det_session, self.onet_rec_session, self.dict_character)
    
            # OCR-检测-识别
            ocr_sys.ocr_det_rec_img(crop)
    
            # 得到检测框
            dt_boxes = ocr_sys.get_boxes()
    
            # 识别 results: 单纯的识别结果，results_info: 识别结果+置信度
            results, results_info = ocr_sys.recognition_img(dt_boxes)
    
            time2 = time()
    
            logging.info(f"OCR OUTPUT: {textzone}")
            detections += self.create_ocr_text_detections(textzone.bbox, dt_boxes, results)

        return [d for d in detections if d is not None]

    def create_ocr_text_detections(self, textzone_box, boxes, txts) -> DetectionNode:

        detections = []

        if txts is None or len(txts) != len(boxes):
            txts = [None] * len(boxes)

        x_offset = textzone_box.x1
        y_offset = textzone_box.y1

        for idx, (box, txt) in enumerate(zip(boxes, txts)):

            extra_data = {"ocr": txt[0],
                          detection_confidence: float(txt[1])}

            box_rect = [int(box[0][0]) + x_offset, int(box[0][1]) + y_offset,
                        int(box[2][0]) + x_offset, int(box[2][1]) + y_offset]
            detection = DetectionNode( name='text',
                    coords=rect_to_coords(box_rect),
                    extra_data=extra_data)

            detections.append(detection)

        return detections

