from vcap import BaseBackend, DETECTION_NODE_TYPE, OPTION_TYPE, BaseStreamState, DetectionNode
from typing import Dict
import numpy as np
import cv2
import base64
import requests
import os

class Backend(BaseBackend):

    def claude(self, jpg_as_base64: str, options: Dict[str, OPTION_TYPE],) -> str:
        api_key = os.environ.get('CLAUDE_APIKEY','')
        http_proxy = os.environ.get('CLAUDE_HTTPPROXY','')
        prompt = os.environ.get('CLAUDE_PROMPT','')

        proxy = {
            "http": http_proxy,
            "https": http_proxy
        }
        url = "https://api.anthropic.com/v1/messages"
        headers = {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json"
        }
        data = {
            "model": "claude-3-5-sonnet-20240620",
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
            "temperature": options["temperature"],
            "max_tokens": options["max_tokens"]
        }
        response = requests.post(url, headers=headers, json=data, proxies=proxy)
        return response.json()

    def process_frame(self, frame: np.ndarray,
                      detection_node: DETECTION_NODE_TYPE,
                      options: Dict[str, OPTION_TYPE],
                      state: BaseStreamState) -> DETECTION_NODE_TYPE:
        detections = []
        height, width = frame.shape[:2]
        _, buffer = cv2.imencode('.jpg', frame)
        jpg_as_base64 = base64.b64encode(buffer).decode('utf-8')
        extra_data = self.claude(jpg_as_base64, options)
        detections.append(DetectionNode( name="claude", coords=[[0,0], [width,0], [width,height], [0,height]], extra_data={"claude": extra_data}))
        return detections
    

