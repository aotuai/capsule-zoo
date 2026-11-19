from vcap import BaseStreamState
import threading
import time

MIN_IOU = 0.3  # the least overlap for an answer to belong to a person in a later frame


def box_of(detection):
    """[x1, y1, x2, y2] of a detection."""
    xs = [p[0] for p in detection.coords]
    ys = [p[1] for p in detection.coords]
    return [min(xs), min(ys), max(xs), max(ys)]


def iou(a, b):
    """Intersection over union of two boxes [x1, y1, x2, y2]."""
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    if w <= 0 or h <= 0:
        return 0.0
    inter = w * h
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


class StreamState(BaseStreamState):
    def __init__(self):
        self.lock = threading.RLock()
        # the variables that are shared with the worker threads (guarded by the lock)
        self.pending = 0  # the number of requests that were sent and have no answer yet
        self.last_detection_timestamp = 0
        self.answers = []  # [{"box", "text", "at"}]: the answers that are still shown

    def set_last_detection_timestamp(self, current_timestamp):
        with self.lock:
            self.last_detection_timestamp = current_timestamp
            return self.last_detection_timestamp

    def get_last_detection_timestamp(self):
        with self.lock:
            return self.last_detection_timestamp

    def add_answer(self, box, text, key):
        """Keep the answer of the model for the person that was at box [x1, y1, x2, y2] when it was asked.
        It is shown under the key (the name of the model that gave it)."""
        with self.lock:
            # a newer answer replaces the older one about the same person
            self.answers = [a for a in self.answers if iou(a["box"], box) < 0.5]
            self.answers.append({"box": list(box), "text": text, "key": key, "at": time.time()})

    def annotate(self, people, hold_seconds):
        """Write the answers into the extra data of the people that are in this frame. The answer is matched to the
        person whose box overlaps its box most (the person may have moved since the question), and the box follows
        the person. An answer older than hold_seconds is dropped. Returns the number of people annotated."""
        with self.lock:
            now = time.time()
            self.answers = [a for a in self.answers if now - a["at"] <= hold_seconds]
            free = list(people)
            annotated = 0
            for answer in sorted(self.answers, key=lambda a: a["at"], reverse=True):
                best, best_iou = None, MIN_IOU
                for person in free:
                    overlap = iou(answer["box"], box_of(person))
                    if overlap >= best_iou:
                        best, best_iou = person, overlap
                if best is not None:
                    free.remove(best)
                    best.extra_data[answer["key"]] = answer["text"]
                    answer["box"] = box_of(best)
                    annotated += 1
            return annotated

    def request_started(self):
        with self.lock:
            self.pending += 1

    def request_finished(self):
        with self.lock:
            self.pending = max(0, self.pending - 1)

    def get_pending(self):
        with self.lock:
            return self.pending
