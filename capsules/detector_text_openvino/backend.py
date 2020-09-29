from typing import Dict, List, Any

import numpy as np
import cv2

from vcap import (
    DetectionNode,
    DETECTION_NODE_TYPE,
    OPTION_TYPE,
    BaseStreamState,
    BaseBackend,
    rect_to_coords)
from vcap_utils import (
    BaseOpenVINOBackend,
)

SOS_INDEX = 0
EOS_INDEX = 1
MAX_SEQ_LEN = 28
ALPHABET = '  0123456789abcdefghijklmnopqrstuvwxyz'


# We have to do this because we need there to be a process_frame to use it
class OpenVINOModel(BaseOpenVINOBackend):
    def process_frame(self,
                      frame: np.ndarray,
                      detection_node: DETECTION_NODE_TYPE,
                      options: Dict[str, OPTION_TYPE],
                      state: BaseStreamState) -> DETECTION_NODE_TYPE:
        raise NotImplemented('This backend is not for processing frames.'
                             'It is only used for storing a model.')


class Backend(BaseBackend):
    label_map: Dict[int, str] = {1: "text"}

    def __init__(self, detector: OpenVINOModel,
                 recognizer_encoder: OpenVINOModel,
                 recognizer_decoder: OpenVINOModel):
        super().__init__()
        self.detector = detector
        self.recognizer_encoder = recognizer_encoder
        self.recognizer_decoder = recognizer_decoder

    @property
    def workload(self) -> float:
        return (self.detector.workload +
                self.recognizer_encoder.workload +
                self.recognizer_decoder.workload)

    def batch_predict(self, input_data_list: List[Any]) -> List[Any]:
        raise NotImplemented('dont use this lel')

    def process_frame(self, frame: np.ndarray,
                      detection_node: DETECTION_NODE_TYPE,
                      options: Dict[str, OPTION_TYPE],
                      state: BaseStreamState) -> DETECTION_NODE_TYPE:
        n, c, h, w = self.detector.net.inputs['im_data'].shape
        hidden_shape = self.recognizer_decoder.net.inputs['prev_hidden'].shape

        input_dict, resize = self.detector.prepare_inputs(
            frame,
            frame_input_name="im_data"
        )
        input_dict["im_data"] = (input_dict["im_data"]
                                 .reshape((n, c, h, w)).astype(np.float32))

        input_image_size = self.detector.net.inputs['im_data'].shape[-2:]
        input_image_info = np.asarray(
            [[input_image_size[0], input_image_size[1], 1]], dtype=np.float32)
        input_dict["im_info"] = input_image_info
        prediction = self.detector.send_to_batch(input_dict).get()

        scores = prediction["scores"]
        detections_filter = scores > options["threshold"]
        scores = scores[detections_filter]
        rects = prediction["boxes"][detections_filter]
        text_features = prediction["text_features"][detections_filter]

        feature_queues = []
        for text_feature in text_features:
            feature_queues.append(
                self.recognizer_encoder.send_to_batch({'input': text_feature}))

        detections = []
        for score, rect, feature_queue in zip(scores, rects, feature_queues):
            feature = feature_queue.get()['output']
            feature.reshape(feature, (feature.shape[0], feature.shape[1], -1))
            feature.transpose(feature, (0, 2, 1))

            hidden = np.zeros(hidden_shape)
            prev_symbol_index = np.ones((1,)) * SOS_INDEX

            text = ''
            for _ in range(MAX_SEQ_LEN):
                decoder_output = self.recognizer_decoder.send_to_batch({
                    'prev_symbol': prev_symbol_index,
                    'prev_hidden': hidden,
                    'encoder_outputs': feature
                }).get()
                symbols_distr = decoder_output['output']
                prev_symbol_index = int(np.argmax(symbols_distr, axis=1))
                if prev_symbol_index == EOS_INDEX:
                    break
                text += ALPHABET[prev_symbol_index]
                hidden = decoder_output['hidden']

            detections.append(DetectionNode(
                name="text",
                coords=rect_to_coords(rect.tolist()),
                extra_data={
                    "detection_confidence": float(score),
                    "text": text
                },
            ))
        return resize.scale_and_offset_detection_nodes(detections)
