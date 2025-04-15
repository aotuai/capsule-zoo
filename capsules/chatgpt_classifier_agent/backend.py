from vcap import BaseBackend, DETECTION_NODE_TYPE, OPTION_TYPE, DetectionNode, BoundingBox, Resize, rect_to_coords
from typing import Dict
import numpy as np
import cv2
import base64
import requests
import time
import logging

from .stream_state import StreamState
from concurrent.futures import ThreadPoolExecutor
from queue import Queue, Full


class BoundedThreadPoolExecutor(ThreadPoolExecutor):
    def __init__(self, max_workers, max_queue_size):
        super().__init__(max_workers)
        self._work_queue = Queue(maxsize=max_queue_size)

    def submit(self, fn, *args, **kwargs):
        # 先检查队列是否已满（提前拒绝任务）
        if self._work_queue.full():
            raise Full(f"Task queue is full (max size: {self._work_queue.maxsize})")

        # 调用父类 submit 方法，由父类封装为 _WorkItem
        return super().submit(fn, *args, **kwargs)


class Backend(BaseBackend):
    def __init__(self):
        super().__init__()
        self.executor = BoundedThreadPoolExecutor(max_workers=10, max_queue_size=100)
        self.submit_num = 0
        logging.info(f"chatgpt_classifier_agent create thread pool")

    def close(self):
        """关闭线程池,等待所有线程任务完成"""
        self.executor.shutdown(wait=True)
        logging.info(f"chatgpt_classifier_agent shutdown thread pool end.")

    def chatgpt(self, jpg_as_base64: str, options: Dict[str, OPTION_TYPE], coords, state: StreamState, idx):
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
                extra_data = response.json()['choices'][0]['message']['content']
                node = DetectionNode(name="chatgpt", coords=coords,
                                     extra_data={"chatgpt": extra_data})
                state.set_detection_response(node)
        except Exception as e:
            logging.error(f"chatgpt thread {idx}: {e}")

        logging.info(f"chatgpt thread {idx} end")


    def process_frame(self, frame: np.ndarray,
                      detection_node: DETECTION_NODE_TYPE,
                      options: Dict[str, OPTION_TYPE],
                      state: StreamState) -> DETECTION_NODE_TYPE:

        logging.info(f"options['current_count'] = {options['current_count']}")
        if (time.time() - state.get_last_detection_timestamp()) < options["detection_interval"]:
            return state.get_detection_response()

        if options["current_count"] > 0:
            if detection_node:
                try:
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
                        _, buffer = cv2.imencode('.jpg', frame_crop)
                        jpg_as_base64 = base64.b64encode(buffer).decode('utf-8')
                        state.set_last_detection_timestamp(time.time())
                        coords = rect_to_coords([cell[0], cell[1], cell[2], cell[3]])
                        self.executor.submit(self.chatgpt, jpg_as_base64, options, coords, state, self.submit_num)
                        self.submit_num += 1
                        num += 1
                        if num >= options["max_detections_per_frame"]:
                            break
                    options["current_count"] -= 1
                except Exception as e:
                    logging.error(f"{e}")

        detections = state.get_detection_response()

        return detections
