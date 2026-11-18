from vcap import BaseBackend, DETECTION_NODE_TYPE, OPTION_TYPE, DetectionNode, rect_to_coords
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor
import threading
import numpy as np
import cv2
import base64
import requests
import time
import logging

from .stream_state import StreamState

URL = "https://api.openai.com/v1/chat/completions"
REQUEST_TIMEOUT_SECONDS = 60  # a worker thread (and the shutdown of the capsule) must never wait for the API forever
RETRY_DELAY_SECONDS = 10  # after a failed call, wait at least this long before the next one
MAX_WORKERS = 4
# With image detail "low" the API looks at a 512 x 512 picture anyway; a larger crop only costs upload time
MAX_CROP_EDGE = {"low": 512, "high": 1024, "auto": 1024}

# The options are shared by all streams (the core hands out one cached dict), so the call quota is counted under a lock
_quota_lock = threading.Lock()


def is_reasoning_model(model: str) -> bool:
    """The GPT-5 family and the o-series think before they answer: they take max_completion_tokens and
    reasoning_effort, and reject temperature, top_p and the penalties."""
    model = model.strip().lower()
    return model.startswith(("gpt-5", "o1", "o3", "o4"))


def build_request(jpg_as_base64: str, options: Dict[str, OPTION_TYPE]) -> dict:
    model = options["model"]
    detail = str(options["image_detail"]).strip() or "low"
    data = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": options["prompt"]},
                    {"type": "image_url",
                     "image_url": {"url": f"data:image/jpeg;base64,{jpg_as_base64}", "detail": detail}},
                ],
            }
        ],
    }
    if is_reasoning_model(model):
        # the budget also covers the thinking, so keep it above a few hundred
        data["max_completion_tokens"] = options["max_tokens"]
        effort = str(options["reasoning_effort"]).strip()
        if effort:
            data["reasoning_effort"] = effort
    else:
        data["max_tokens"] = options["max_tokens"]
        data["temperature"] = options["temperature"]
    return data


def reply_text(body: dict) -> str:
    choices = body.get("choices") or []
    if not choices:
        return ""
    return str((choices[0].get("message") or {}).get("content") or "").strip()


def person_boxes(detection_node, min_pixels: int, frame_shape) -> List[List[int]]:
    """The boxes [x1, y1, x2, y2] of the people that are big enough, the biggest first, clipped to the frame."""
    frame_h, frame_w = frame_shape[:2]
    found: List[Tuple[int, List[int]]] = []
    for det in detection_node:
        coords = det["coords"] if isinstance(det, dict) else det.coords
        x1, y1 = max(0, int(coords[0][0])), max(0, int(coords[0][1]))
        x2, y2 = min(frame_w, int(coords[2][0])), min(frame_h, int(coords[2][1]))
        area = (x2 - x1) * (y2 - y1)
        if x2 > x1 and y2 > y1 and area >= min_pixels:
            found.append((area, [x1, y1, x2, y2]))
    found.sort(key=lambda item: item[0], reverse=True)  # a list, so two people with the same area both stay
    return [box for _, box in found]


class Backend(BaseBackend):
    def __init__(self):
        super().__init__()
        self.executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)
        self.submit_num = 0
        logging.debug("chatgpt_classifier_agent: thread pool created")

    def close(self):
        super().close()
        # Waits for the running requests, which end within REQUEST_TIMEOUT_SECONDS
        self.executor.shutdown(wait=True)
        logging.debug("chatgpt_classifier_agent: thread pool shut down")

    def chatgpt_classifier_agent(self, jpg_as_base64: str, options: Dict[str, OPTION_TYPE], coords, state: StreamState, idx):
        """Runs in a worker thread: one request for one person."""
        logging.debug(f"chatgpt_classifier_agent thread {idx} running")
        try:
            headers = {"Content-Type": "application/json", "Authorization": f"Bearer {options['api_key']}"}
            http_proxy = options["http_proxy"]
            proxies = {"http": http_proxy, "https": http_proxy} if http_proxy != "" else None
            response = requests.post(URL, headers=headers, json=build_request(jpg_as_base64, options),
                                     proxies=proxies, timeout=REQUEST_TIMEOUT_SECONDS)
            if response.status_code == 200:
                text = reply_text(response.json())
                if text:
                    state.set_detection_response(DetectionNode(name="chatgpt", coords=coords, extra_data={"chatgpt": text}))
                else:
                    logging.error(f"chatgpt_classifier_agent thread {idx}: the reply has no text "
                                  "(raise max_tokens, or lower reasoning_effort)")
                    self._wait_after_failure(state, options)
            else:
                logging.error(f"chatgpt_classifier_agent thread {idx}: HTTP {response.status_code}: {response.text[:300]}")
                self._wait_after_failure(state, options)
        except Exception as e:
            logging.error(f"chatgpt_classifier_agent thread {idx}: {e}")
            self._wait_after_failure(state, options)
        finally:
            state.request_finished()
        logging.debug(f"chatgpt_classifier_agent thread {idx} end")

    @staticmethod
    def _wait_after_failure(state: StreamState, options: Dict[str, OPTION_TYPE]) -> None:
        """A failed call counts like a made one, but the retry comes sooner than a long detection interval would."""
        interval = options["detection_interval"]
        state.set_last_detection_timestamp(time.time() - interval + max(interval, RETRY_DELAY_SECONDS))

    @staticmethod
    def _take_quota(options: Dict[str, OPTION_TYPE]) -> bool:
        """One call of the quota ("current_count"); False when it is used up."""
        with _quota_lock:
            if options["current_count"] <= 0:
                return False
            options["current_count"] -= 1
            return True

    def process_frame(self, frame: np.ndarray,
                      detection_node: DETECTION_NODE_TYPE,
                      options: Dict[str, OPTION_TYPE],
                      state: StreamState) -> DETECTION_NODE_TYPE:
        if (time.time() - state.get_last_detection_timestamp()) < options["detection_interval"]:
            return state.get_detection_response()
        # No new requests while the last ones have no answer yet: a slow API must not pile up calls
        if state.get_pending() > 0 or not detection_node or options["current_count"] <= 0:
            return state.get_detection_response()
        if not str(options["api_key"]).strip():
            logging.warning("chatgpt_classifier_agent: fill in the api_key option to use this capsule")
            self._wait_after_failure(state, options)
            return state.get_detection_response()

        try:
            detail = str(options["image_detail"]).strip() or "low"
            max_edge = MAX_CROP_EDGE.get(detail, 1024)
            boxes = person_boxes(detection_node, options["min_person_pixels"], frame.shape)
            for x1, y1, x2, y2 in boxes[:options["max_detections_per_frame"]]:
                if not self._take_quota(options):
                    logging.warning("chatgpt_classifier_agent: the call quota (current_count) is used up")
                    break
                crop = frame[y1:y2, x1:x2]
                if max(crop.shape[:2]) > max_edge:
                    scale = max_edge / max(crop.shape[:2])
                    crop = cv2.resize(crop, (max(1, round(crop.shape[1] * scale)), max(1, round(crop.shape[0] * scale))),
                                      interpolation=cv2.INTER_AREA)
                _, buffer = cv2.imencode('.jpg', crop)
                jpg_as_base64 = base64.b64encode(buffer).decode('utf-8')
                state.set_last_detection_timestamp(time.time())
                state.request_started()
                try:
                    self.executor.submit(self.chatgpt_classifier_agent, jpg_as_base64, options,
                                         rect_to_coords([x1, y1, x2, y2]), state, self.submit_num)
                except Exception:
                    state.request_finished()
                    raise
                self.submit_num += 1
        except Exception as e:
            logging.error(f"chatgpt_classifier_agent: {e}")

        return state.get_detection_response()
