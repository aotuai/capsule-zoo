from vcap import BaseStreamState
import threading

class StreamState(BaseStreamState):
    def __init__(self):
        self.lock = threading.RLock()
        # the variables need to lock
        self.detections = []
        self.thread_max = 20
        self.thread_num = 0

        # the variables don't need to lock because only accessed by main thread
        self.last_detection_timestamp = 0
        self.thread_idx = 0
        # self.limit_start_thread_idx = 0
        # self.last_search_count_limit = True

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

    #def set_last_search_count_limit(self, current_limit):
    #    self.last_search_count_limit = current_limit
    #    return self.last_search_count_limit

    #def get_last_search_count_limit(self):
    #    return self.last_search_count_limit

    def set_thread_idx(self, idx):
        self.thread_idx = idx
        return self.thread_idx

    def get_thread_idx(self):
        return self.thread_idx

    '''
    def set_limit_start_thread_idx(self):
        self.limit_start_thread_idx = self.thread_idx

    def check_search_count_full(self, current_search_count_limit, max_search_count):
        if current_search_count_limit is True and self.thread_idx - self.limit_start_thread_idx + 1 > max_search_count:
            return True
        return False
    '''