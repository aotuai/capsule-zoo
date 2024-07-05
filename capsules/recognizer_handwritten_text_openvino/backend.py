from typing import Dict, List, Tuple

import numpy as np
import cv2
import tempfile
from pathlib import Path
import logging

from vcap import (
    DetectionNode,
    Resize,
    DETECTION_NODE_TYPE,
    rect_to_coords,
    BoundingBox,
    OPTION_TYPE,
    BaseStreamState)
from vcap_utils import BaseOpenVINOBackend
from .utils.codec import CTCCodec
from .bold_strokes import bold_strokes, get_handwriten_text_area_image

OV_INPUT_TYPE = Dict[str, np.ndarray]

class OpenVINOModel(BaseOpenVINOBackend):
    def __init__(self, model_xml: bytes, weights_bin: bytes, char_list: bytes, device_name: str, ie_core=None):
        super().__init__(model_xml, weights_bin, device_name, ie_core)

        self.top_k = 20
        self.designated_characters = None
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir_path = Path(temp_dir)
            dict_character_file = str(temp_dir_path) + "/scut_ept.txt"
            dict_character = open(dict_character_file, 'wb')
            dict_character.write(char_list)
            dict_character.close()
            self.char_list = get_characters(dict_character_file)

class Backend(OpenVINOModel):
    label_map: Dict[int, str] = {1: "text"}

    def process_frame(self, frame: np.ndarray,
                      detection_node: DETECTION_NODE_TYPE,
                      options: Dict[str, OPTION_TYPE],
                      state: BaseStreamState) -> DETECTION_NODE_TYPE:
        detections = []

        logging.info(f"{frame.shape}")
        try:
            is_cell = options["cell"]

            cell_x, cell_y = 0, 0
            if is_cell:
                cell_x = options["cell_x"]
                cell_y = options["cell_y"]
                cell_width = options["cell_width"]
                cell_height = options["cell_height"]
                cell_bbox = BoundingBox(cell_x, cell_y, cell_x + cell_width, cell_y + cell_height)
                frame = Resize(frame).crop_bbox(cell_bbox).frame

            input_dict, box = self.hw_prepare_inputs(frame, options)
            logging.info(f"{box}")
            prediction = self.send_to_batch(input_dict).result()
            result = self.hw_parse_detection_results(prediction)
            box = [i+j for i, j in zip(box, [cell_x, cell_y, cell_x, cell_y])]
            res = DetectionNode(
                name=self.label_map[1],
                coords=rect_to_coords(box),
                extra_data={"ocr": result[0], "confidence": 0.0})
            detections.append(res)

        except Exception as e:
            logging.warning(f"Failed to process ocr frame, {e}")

        return detections

    def hw_prepare_inputs(self, frame: np.ndarray, options: Dict[str, OPTION_TYPE], frame_input_name: str = None) \
            -> Tuple[OV_INPUT_TYPE, List]:
        # remove watermark
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #logging.warning(f"gray h={gray.shape[0]} w={gray.shape[1]} pad={options['strokes_pad']}")
        src, box = get_handwriten_text_area_image(gray, options["strokes_pad"])
        #logging.warning(f"{box}")
        input_blob_name = frame_input_name or self.input_blob_names[0]
        _, _, input_height, input_width = self.net.input_info[input_blob_name].input_data.shape

        #if options["to_bold_strokes"]:
        #    src = bold_strokes(src)

        ratio = float(src.shape[1]) / float(src.shape[0])
        tw = int(input_height * ratio)
        rsz = cv2.resize(src, (tw, input_height), interpolation=cv2.INTER_AREA).astype(np.float32)

        if options["to_bold_strokes"]:
            rsz = bold_strokes(rsz)

        # [h,w] -> [c,h,w]
        img = rsz[None, :, :]
        _, h, w = img.shape
        # right edge padding
        pad_img = np.pad(img, ((0, 0), (0, input_height - h), (0, input_width - w)), mode='edge')

        return {input_blob_name: pad_img}, box

    def hw_parse_detection_results(
            self, results: np.ndarray,
            boxes_output_name: str = None ) -> List[str]:

        codec = CTCCodec(self.char_list, self.designated_characters, self.top_k)
        nodes: List[DetectionNode] = []

        output_blob_name = boxes_output_name or self.output_blob_names[0]
        inference_results = results[output_blob_name]
        result = codec.decode(inference_results)

        return result

def get_characters(char_file):
    '''Get characters'''
    with open(char_file, 'r', encoding='utf-8') as f:
        return ''.join(line.strip('\n') for line in f)
