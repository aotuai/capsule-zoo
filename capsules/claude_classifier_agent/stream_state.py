from vcap import BaseStreamState

class StreamState(BaseStreamState):
    def __init__(self):
        self.last_detection_timestamp = 0

    def set_last_detection_timestamp(self, current_timestamp):
        self.last_detection_timestamp = current_timestamp
        return self.last_detection_timestamp
    def get_last_detection_timestamp(self):
        return self.last_detection_timestamp
