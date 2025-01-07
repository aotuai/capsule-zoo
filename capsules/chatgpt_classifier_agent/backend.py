from vcap import BaseBackend, DETECTION_NODE_TYPE, OPTION_TYPE, DetectionNode
from typing import Dict
import numpy as np
import cv2
import base64
import requests
import os
import time

from .stream_state import StreamState

class Backend(BaseBackend):

    def chatgpt(self, jpg_as_base64: str, options: Dict[str, OPTION_TYPE],) -> str:
        api_key = options["api_key"]
        http_proxy = options["http_proxy"]
        prompt = options["prompt"]
        model = options["model"]
        temperature = options["temperature"]
        max_tokens = options["max_tokens"]

        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        data = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{jpg_as_base64}"
                            }
                        }
                    ]
                }
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": 1,
            "frequency_penalty": 0,
            "presence_penalty": 0
        }

        if http_proxy != "":
            proxies = {
                "http": http_proxy,
                "https": http_proxy
            }
            response = requests.post(url, headers=headers, json=data, proxies=proxies)
        else:
            response = requests.post(url, headers=headers, json=data)
        return response

    def process_frame(self, frame: np.ndarray,
                      detection_node: DETECTION_NODE_TYPE,
                      options: Dict[str, OPTION_TYPE],
                      state: StreamState) -> DETECTION_NODE_TYPE:
        detections = []
        if((time.time() - state.get_last_detection_timestamp()) < options["detection_interval"]):
            return detections
        height, width = frame.shape[:2]
        _, buffer = cv2.imencode('.jpg', frame)
        jpg_as_base64 = base64.b64encode(buffer).decode('utf-8')

        response = self.chatgpt(jpg_as_base64, options)
        if response.status_code == 200:
            extra_data = response['choices'][0]['message']['content']
            detections.append(DetectionNode( name="chatgpt", coords=[[0,0], [width,0], [width,height], [0,height]], extra_data={"chatgpt": extra_data}))
            state.set_last_detection_timestamp(time.time())
        return detections
