from typing import Dict, List, Any, Tuple

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
        raise NotImplemented('This backend is not for processing frames. '
                             'It is only used for storing a model.')

    def get_input_shape(self, input_name: str) -> Tuple[int, int, int, int]:
        return self.model.input(input_name).shape

    def get_input_tensor(self, input_name: str):
        return self.model.input(input_name)


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

    def process_frame(self, frame: np.ndarray,
                      detection_node: DETECTION_NODE_TYPE,
                      options: Dict[str, OPTION_TYPE],
                      state: BaseStreamState) -> DETECTION_NODE_TYPE:

        n, c, h, w = self.detector.get_input_shape('im_data')
        hidden_shape = self.recognizer_decoder.get_input_shape('prev_hidden')

        input_dict, resize = self.detector.prepare_inputs(
            frame,
            frame_input_name="im_data"
        )
        input_dict["im_data"] = (input_dict["im_data"]
                                 .reshape((n, c, h, w)).astype(np.float32))

        input_image_size = self.detector.get_input_shape('im_data')[-2:]
        input_image_info = np.asarray(
            [[input_image_size[0], input_image_size[1], 1]], dtype=np.float32)
        input_dict["im_info"] = input_image_info

        prediction = self.detector.send_to_batch(input_dict).result()

        scores_key = next((key for key in prediction.keys() if 'scores' in key.names), None)
        boxes_key = next((key for key in prediction.keys() if 'boxes' in key.names), None)
        features_key = next((key for key in prediction.keys() if 'text_features' in key.names), None)

        if not all([scores_key, boxes_key, features_key]):
            raise KeyError(f"Missing required keys in prediction. Available keys: {prediction.keys()}")

        scores = prediction[scores_key]
        detections_filter = scores > options["threshold"]
        scores = scores[detections_filter]
        rects = prediction[boxes_key][detections_filter]
        text_features = prediction[features_key][detections_filter]

        feature_queues = []
        for text_feature in text_features:
            text_feature = np.expand_dims(text_feature, axis=0)
            feature_queues.append(
                self.recognizer_encoder.send_to_batch({'input': text_feature}))

        detections = []
        for score, rect, feature_queue in zip(scores, rects, feature_queues):
            feature_result = feature_queue.result()
            output_key = next(key for key in feature_result.keys() if 'output' in key.names)
            feature = feature_result[output_key]
            feature = np.reshape(feature,
                                 (feature.shape[0], feature.shape[1], -1))
            feature = np.transpose(feature, (0, 2, 1))

            hidden = np.zeros(hidden_shape)
            prev_symbol_index = np.ones((1,)) * SOS_INDEX

            text = ''
            for _ in range(MAX_SEQ_LEN):
                decoder_output = self.recognizer_decoder.send_to_batch({
                    'prev_symbol': prev_symbol_index,
                    'prev_hidden': hidden,
                    'encoder_outputs': feature
                }).result()
                output_key = next(key for key in decoder_output.keys() if 'output' in key.names)

                symbols_distr = decoder_output[output_key]
                prev_symbol_index = int(np.argmax(symbols_distr, axis=1))
                if prev_symbol_index == EOS_INDEX:
                    break
                text += ALPHABET[prev_symbol_index]

                hidden_key = next(key for key in decoder_output.keys() if 'hidden' in key.names)
                hidden = decoder_output[hidden_key]

            detections.append(DetectionNode(
                name="text",
                coords=rect_to_coords(rect.tolist()),
                extra_data={
                    "detection_confidence": float(score),
                    "text": text
                },
            ))
        return resize.scale_and_offset_detection_nodes(detections)
