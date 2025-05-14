from vcap import BaseStreamState
import time
import logging


class StreamState(BaseStreamState):
    def __init__(self):
        self.window_start = time.time()
        self.duration = 0
        self.counter = 0
        self.category = None
        self.values = None
        self.true_counter = {}

    def set_window(self, duration, counter, category, values):
        if not (duration == self.duration and counter == self.counter and
                category == self.category and values == self.values):
            self.duration = duration
            self.counter = counter
            self.category = category
            self.values = values
            self.true_counter = {}
            for value in self.values:
                self.true_counter[value] = 0

    def reset_window(self):
        self.window_start = time.time()
        for value in self.values:
            self.true_counter[value] = 0

    def add_true_counter(self, value):
        cur_time = time.time()
        if cur_time - self.window_start > self.duration:
            self.reset_window()
        self.true_counter[value] += 1
        logging.info(f"add_true_counter: {self.duration},{self.counter},{self.true_counter}")
        if self.true_counter[value] >= self.counter:
            self.true_counter[value] = 0
            return True
        return False


