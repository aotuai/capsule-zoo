from vcap import BaseStreamState
import threading

class StreamState(BaseStreamState):
    def __init__(self):
        self.lock = threading.RLock()
        # the variables need to lock
        self.detections = []

        # the variables don't need to lock because only accessed by main thread
        self.last_detection_timestamp = 0


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
        detections = []
        for k in range(len(self.detections)):
            detections.append(self.detections.pop(0))
        self.lock.release()
        return detections

