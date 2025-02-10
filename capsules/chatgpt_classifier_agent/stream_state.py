from vcap import BaseStreamState
import threading

class StreamState(BaseStreamState):
    def __init__(self):
        self.lock = threading.RLock()
        self.last_detection_timestamp = 0
        self.detections = []
        self.thread_max = 20
        self.thread_num = 0

    def set_last_detection_timestamp(self, current_timestamp):
        self.last_detection_timestamp = current_timestamp
        return self.last_detection_timestamp

    def get_last_detection_timestamp(self):
        return self.last_detection_timestamp

    def set_detection_response(self, detections):
        self.lock.acquire()
        self.detections.append(detections)
        self.lock.release()

    def get_detection_response(self):
        self.lock.acquire()
        if len(self.detections) > 0:
            detections = self.detections.pop(0)
        else:
            detections = []
        self.lock.release()
        return detections

    def increase_thread_num(self, num):
        self.lock.acquire()
        self.thread_num += num
        self.lock.release()

    def reduced_thread_num(self, num):
        self.lock.acquire()
        if self.thread_num >= num:
            self.thread_num -= num
        self.lock.release()

    def check_thread_full(self):
        self.lock.acquire()
        ret = True if self.thread_num >= self.thread_max else False
        self.lock.release()
        return ret
