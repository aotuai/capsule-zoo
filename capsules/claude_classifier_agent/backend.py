from vcap import BaseBackend, DETECTION_NODE_TYPE, OPTION_TYPE, BaseStreamState, DetectionNode
from typing import Dict
import numpy as np
import cv2
import base64
import requests
import os

class Backend(BaseBackend):

    def claude(self, jpg_as_base64: str, options: Dict[str, OPTION_TYPE],) -> str:
        api_key = options["api_key"]
        http_proxy = options["http_proxy"]
        prompt = options["prompt"]
        model = options["model"]
        temperature = options["temperature"]
        max_tokens = options["max_tokens"]

        url = "https://api.anthropic.com/v1/messages"
        headers = {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json"
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
            "temperature": temperature,
            "max_tokens": max_tokens
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
                      state: BaseStreamState) -> DETECTION_NODE_TYPE:
        detections = []
        height, width = frame.shape[:2]
        _, buffer = cv2.imencode('.jpg', frame)
        jpg_as_base64 = base64.b64encode(buffer).decode('utf-8')

        response = self.claude(jpg_as_base64, options)
        if response.status_code == 200:
            extra_data = response['choices'][0]['message']['content']
            detections.append(DetectionNode( name="claude", coords=[[0,0], [width,0], [width,height], [0,height]], extra_data={"claude": extra_data}))
        return detections
