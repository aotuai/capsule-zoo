from vcap import BaseBackend, DETECTION_NODE_TYPE, OPTION_TYPE, DetectionNode, BoundingBox, Resize
from typing import Dict
import numpy as np
import cv2
import base64
import requests
import os
import time
import threading
import logging

from .stream_state import StreamState


class Backend(BaseBackend):

    def chatgpt(self, jpg_as_base64: str, options: Dict[str, OPTION_TYPE], x, y, width, height, state: StreamState, idx):
        state.increase_thread_num(1)
        logging.info(f"chatgpt thread {idx} running")
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

        # '''
        try:
            if http_proxy != "":
                proxies = {
                    "http": http_proxy,
                    "https": http_proxy
                }
                response = requests.post(url, headers=headers, json=data, proxies=proxies)
            else:
                response = requests.post(url, headers=headers, json=data)

            if response.status_code == 200:
                # detections = []
                extra_data = response.json()['choices'][0]['message']['content']
                node = DetectionNode(name="chatgpt", coords=[[x, y], [x+width, y], [x+width, y+height], [x, y+height]],
                                     extra_data={"chatgpt": extra_data})
                state.set_detection_response(node)
        except Exception as e:
            logging.error(f"chatgpt: {e}")

        '''
        # for local debug
        time.sleep(2)
        if True:
            # detections = []
            extra_data = f"{idx}"
            node = DetectionNode(name="chatgpt", coords=[[x, y], [x+width, y], [x+width, y+height], [x, y+height]],
                                 extra_data={"chatgpt": extra_data})
            state.set_detection_response(node)
        '''

        logging.info(f"chatgpt thread {idx} end")
        state.reduced_thread_num(1)

    def create_chatgpt_thread(self, image_base64, options, x, y, width, height, state):
        idx = state.get_thread_idx()
        t = threading.Thread(target=self.chatgpt,
                             args=(image_base64, options, x, y, width, height, state, idx),
                             name=f"chatgpt-thread-{idx}")
        state.set_thread_idx(idx+1)
        t.start()

    def process_frame(self, frame: np.ndarray,
                      detection_node: DETECTION_NODE_TYPE,
                      options: Dict[str, OPTION_TYPE],
                      state: StreamState) -> DETECTION_NODE_TYPE:

        logging.info(f"options['current_count'] = {options['current_count']}")
        if (time.time() - state.get_last_detection_timestamp()) < options["detection_interval"]:
            return state.get_detection_response()

        if state.check_thread_full() is False and options["current_count"] > 0:
                #state.check_search_count_full(options["search_count_limit"], options["max_search_count"]) is False:
            if detection_node:
                min_pixels = 900
                person_pixels = {}
                for det in detection_node:
                    if type(det) == dict:
                        coor = det["coords"]
                    else:
                        coor = det.__dict__["coords"]
                    width = coor[1][0] - coor[0][0]
                    height = coor[2][1] - coor[0][1]
                    cur_pixels = width * height
                    if cur_pixels >= min_pixels:
                        person_pixels[cur_pixels] = [coor[0][0], coor[0][1], coor[2][0], coor[2][1]]
                person_list = [(k, person_pixels[k]) for k in sorted(person_pixels.keys(), reverse=True)]

                num = 0
                for (_, cell) in person_list:
                    cell_bbox = BoundingBox(cell[0], cell[1], cell[2], cell[3])
                    frame_crop = Resize(frame).crop_bbox(cell_bbox).frame

                    height, width = frame_crop.shape[:2]
                    # logging.info(f"crop frame height={height}, width={width}")
                    _, buffer = cv2.imencode('.jpg', frame_crop)
                    jpg_as_base64 = base64.b64encode(buffer).decode('utf-8')
                    state.set_last_detection_timestamp(time.time())
                    self.create_chatgpt_thread(jpg_as_base64, options, cell[0], cell[1], width, height, state)
                    num += 1
                    if num >= options["max_detections_per_frame"]:
                        break
                options["current_count"] -= 1

        #response = self.chatgpt(jpg_as_base64, options)
        #if response.status_code == 200:
        #    extra_data = response.json()['choices'][0]['message']['content']
        #    detections.append(DetectionNode( name="chatgpt", coords=[[0,0], [width,0], [width,height], [0,height]], extra_data={"chatgpt": extra_data}))
        #    state.set_last_detection_timestamp(time.time())

        detections = state.get_detection_response()

        return detections
