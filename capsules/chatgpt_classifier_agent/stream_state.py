from vcap import BaseStreamState
import threading


class StreamState(BaseStreamState):
    def __init__(self):
        self.lock = threading.RLock()
        # the variables that are shared with the worker threads (guarded by the lock)
        self.detections = []
        self.pending = 0  # the number of requests that were sent and have no answer yet
        self.last_detection_timestamp = 0

    def set_last_detection_timestamp(self, current_timestamp):
        with self.lock:
            self.last_detection_timestamp = current_timestamp
            return self.last_detection_timestamp

    def get_last_detection_timestamp(self):
        with self.lock:
            return self.last_detection_timestamp

    def set_detection_response(self, detections):
        with self.lock:
            self.detections.append(detections)

    def get_detection_response(self):
        """The answers that arrived since the last call; each answer is returned once."""
        with self.lock:
            detections, self.detections = self.detections, []
            return detections

    def request_started(self):
        with self.lock:
            self.pending += 1

    def request_finished(self):
        with self.lock:
            self.pending = max(0, self.pending - 1)

    def get_pending(self):
        with self.lock:
            return self.pending
