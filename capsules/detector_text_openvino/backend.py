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

from scipy.special import softmax
from typing import Tuple

ALPHABET = '  abcdefghijklmnopqrstuvwxyz0123456789'
SOS_INDEX = 0
EOS_INDEX = 1
MAX_SEQ_LEN = 28

# We have to do this because we need there to be a process_frame to use it
class OpenVINOModel(BaseOpenVINOBackend):
    def process_frame(self,
                      frame: np.ndarray,
                      detection_node: DETECTION_NODE_TYPE,
                      options: Dict[str, OPTION_TYPE],
                      state: BaseStreamState) -> DETECTION_NODE_TYPE:
        raise NotImplemented('This backend is not for processing frames. '
                             'It is only used for storing a model.')

    def close(self):
        super().close()

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

    def close(self):
        self.detector.close()
        self.recognizer_encoder.close()
        self.recognizer_decoder.close()
        super().close()

    @property
    def workload(self) -> float:
        return (self.detector.workload +
                self.recognizer_encoder.workload +
                self.recognizer_decoder.workload)

    def process_frame(self, frame: np.ndarray,
                      detection_node: DETECTION_NODE_TYPE,
                      options: Dict[str, OPTION_TYPE],
                      state: BaseStreamState) -> DETECTION_NODE_TYPE:
        n, c, h, w = self.detector.get_input_shape('image')
        hidden_shape = self.recognizer_decoder.get_input_shape('prev_hidden')

        input_dict, resize = self.detector.prepare_inputs(
            frame,
            frame_input_name="image"
        )
        input_dict["image"] = (input_dict["image"]
                                 .reshape((n, c, h, w)).astype(np.float32))

        prediction = self.detector.send_to_batch(input_dict).result()

        boxes_key = next((key for key in prediction.keys() if 'boxes' in key.names), None)
        scores_key = next((key for key in prediction.keys() if 'scores' in key.names), None)
        features_key = next((key for key in prediction.keys() if 'text_features' in key.names), None)
        labels_key = next((key for key in prediction.keys() if 'labels' in key.names), None)

        if not all([boxes_key, features_key]):
            raise KeyError(f"Missing required keys in prediction. Available keys: {prediction.keys()}")

        rects = prediction[boxes_key]
        text_features = prediction[features_key]

        if scores_key:
            scores = prediction[scores_key]
            detections_filter = scores > options["threshold"]
            scores = scores[detections_filter]
            rects = prediction[detections_filter]
            text_features = prediction[detections_filter]
        else:
            scores = rects[:, -1]
            detections_filter = scores > options["threshold"]
            scores = scores[detections_filter]
            rects = rects[detections_filter]
            text_features = text_features[detections_filter]

        if rects.shape[-1] > 4:
            rects = rects[:, :4]
        elif rects.shape[-1] < 4:
            raise ValueError(f"Unexpected shape for rects: {rects.shape}")

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
            prev_symbol_index = np.array([SOS_INDEX], dtype=np.int32)

            text = ''
            text_confidence = 1.0
            for _ in range(MAX_SEQ_LEN):
                decoder_output = self.recognizer_decoder.send_to_batch({
                    'prev_symbol': prev_symbol_index,
                    'prev_hidden': hidden,
                    'encoder_outputs': feature
                }).result()
                symbols_distr_key = next(key for key in decoder_output.keys() if 'output' in key.names)
                hidden_key = next(key for key in decoder_output.keys() if 'hidden' in key.names)

                symbols_distr = decoder_output[symbols_distr_key]
                hidden = decoder_output[hidden_key]

                symbols_distr_softmaxed = softmax(symbols_distr, axis=1)[0]
                prev_symbol_index = int(np.argmax(symbols_distr, axis=1))

                text_confidence *= symbols_distr_softmaxed[prev_symbol_index]

                if prev_symbol_index == EOS_INDEX:
                    break

                text += ALPHABET[prev_symbol_index]
                prev_symbol_index = np.array([prev_symbol_index], dtype=np.int32)

            detections.append(DetectionNode(
                name="text",
                coords=rect_to_coords(rect.tolist()),
                extra_data={
                    "detection_confidence": float(score),
                    "confidence": float(text_confidence),
                    "text": text
                },
            ))
        return resize.scale_and_offset_detection_nodes(detections)
