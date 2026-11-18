from vcap import BaseBackend, DETECTION_NODE_TYPE, OPTION_TYPE, DetectionNode
from typing import Dict
import numpy as np
import cv2
import base64
import requests
import os
import time
import logging

from .stream_state import StreamState

# The cheapest current model is the default (see capsule.py); the options let you choose another one
REQUEST_TIMEOUT_SECONDS = 60  # a stream thread must never wait for the API forever
RETRY_DELAY_SECONDS = 10  # after a failed call, wait at least this long (a bad key must not cost a call per frame)
MAX_IMAGE_EDGE = 1568  # Claude shrinks larger images itself; sending less is faster and costs the same


def reply_text(body: dict) -> str:
    """The text of an Anthropic Messages API reply: its ``text`` content blocks (a reply can also start with
    ``thinking`` blocks, so the blocks are read by type, not by position)."""
    return "".join(
        block.get("text", "") for block in body.get("content", []) if block.get("type") == "text"
    ).strip()


class Backend(BaseBackend):

    def claude(self, jpg_as_base64: str, options: Dict[str, OPTION_TYPE],) -> requests.Response:
        api_key = options["api_key"]
        http_proxy = options["http_proxy"]
        prompt = options["prompt"]
        model = options["model"]
        max_tokens = options["max_tokens"]
        effort = str(options["effort"]).strip()

        url = "https://api.anthropic.com/v1/messages"
        headers = {
            "Content-Type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
        }
        data = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source":
                            {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": jpg_as_base64
                            }
                        },
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ]
                }
            ],
            "max_tokens": max_tokens
        }
        # How much the model thinks before it answers: "low" is right for a True/False question (and cheapest).
        # Leave the option empty for a model that has no effort setting (Claude Haiku 4.5, Sonnet 4.5, ...).
        # No temperature: the current models only accept their default.
        if effort:
            data["output_config"] = {"effort": effort}

        proxies = {"http": http_proxy, "https": http_proxy} if http_proxy != "" else None
        return requests.post(url, headers=headers, json=data, proxies=proxies, timeout=REQUEST_TIMEOUT_SECONDS)

    @staticmethod
    def _wait_after_failure(state: StreamState, options: Dict[str, OPTION_TYPE]) -> None:
        """Count a failed call like a made one, but retry sooner than a long detection interval would."""
        interval = options["detection_interval"]
        state.set_last_detection_timestamp(time.time() - interval + max(interval, RETRY_DELAY_SECONDS))

    def process_frame(self, frame: np.ndarray,
                      detection_node: DETECTION_NODE_TYPE,
                      options: Dict[str, OPTION_TYPE],
                      state: StreamState) -> DETECTION_NODE_TYPE:
        detections = []
        if((time.time() - state.get_last_detection_timestamp()) < options["detection_interval"]):
            return detections
        if not str(options["api_key"]).strip():
            logging.warning("claude: fill in the api_key option to use this capsule")
            self._wait_after_failure(state, options)
            return detections
        height, width = frame.shape[:2]
        sent = frame
        if max(height, width) > MAX_IMAGE_EDGE:
            scale = MAX_IMAGE_EDGE / max(height, width)
            sent = cv2.resize(frame, (round(width * scale), round(height * scale)), interpolation=cv2.INTER_AREA)
        _, buffer = cv2.imencode('.jpg', sent)
        jpg_as_base64 = base64.b64encode(buffer).decode('utf-8')

        try:
            response = self.claude(jpg_as_base64, options)
            if response.status_code == 200:
                body = response.json()
                extra_data = reply_text(body)
                if extra_data:
                    detections.append(DetectionNode( name="claude", coords=[[0,0], [width,0], [width,height], [0,height]], extra_data={"claude": extra_data}))
                    state.set_last_detection_timestamp(time.time())
                else:  # for example stop_reason "max_tokens" before any text: raise the max_tokens option
                    logging.error(f"claude: the reply has no text (stop_reason: {body.get('stop_reason')})")
                    self._wait_after_failure(state, options)
            else:
                logging.error(f"claude: HTTP {response.status_code}: {response.text[:300]}")
                self._wait_after_failure(state, options)
        except Exception as e:
            logging.error(f"claude: {e}")
            self._wait_after_failure(state, options)

        return detections
