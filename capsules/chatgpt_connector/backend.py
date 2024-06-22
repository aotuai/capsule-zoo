from vcap import BaseBackend, DETECTION_NODE_TYPE, OPTION_TYPE, BaseStreamState, DetectionNode
from typing import Dict
import numpy as np
import cv2
import base64
import requests
import os

class Backend(BaseBackend):

    def chatgpt(self, jpg_as_base64: str, options: Dict[str, OPTION_TYPE],) -> str:
        api_key = os.environ.get('CHATGPT_APIKEY','')
        http_proxy = os.environ.get('CHATGPT_HTTPPROXY','')
        prompt = os.environ.get('CHATGPT_PROMPT','')

        proxy = {
            "http": http_proxy,
            "https": http_proxy
        }
        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        data = {
            "model": "gpt-4o",
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
            "temperature": options["temperature"],
            "max_tokens": options["max_tokens"],
            "top_p": 1,
            "frequency_penalty": 0,
            "presence_penalty": 0
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
        extra_data = self.chatgpt(jpg_as_base64, options)
        detections.append(DetectionNode( name="chatgpt", coords=[[0,0], [width,0], [width,height], [0,height]], extra_data={"chatgpt": extra_data}))
        return detections
    

