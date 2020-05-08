from typing import Deque
from uuid import uuid4, UUID
from enum import auto, Enum
from collections import deque

from vcap import DetectionNode


class TrackState(Enum):
    confirmed = auto()
    deleted = auto()
    tentative = auto()


class Track:
    """This is an internal datatype used by the tracker"""

    def __init__(self, first_det, max_misses, n_hits_to_init):
        self.track_id: UUID = uuid4()
        self.detections: Deque[DetectionNode] = deque(
            maxlen=int(n_hits_to_init + 10))
        self.state: TrackState = TrackState.tentative
        self._misses = 0

        # Capsule options
        self.max_misses = max_misses
        self.n_hits_to_init = n_hits_to_init

        # Initialize the track
        self.update(first_det)

    def update(self, det: DetectionNode):
        # Assign a Track ID to the detection (this communicates to brainframe)
        det.track_id = self.track_id
        self.detections.append(det)
        self._misses = 0
        if (self.state is TrackState.tentative and
                len(self.detections) > self.n_hits_to_init):
            self.state = TrackState.confirmed

    def mark_missed(self):
        self._misses += 1

        if self.state is TrackState.tentative:
            self.state = TrackState.deleted

        if self._misses > self.max_misses:
            self.state = TrackState.deleted

    @property
    def latest_det(self):
        return self.detections[-1]

    @property
    def is_confirmed(self):
        return self.state is TrackState.confirmed

    @property
    def is_tentative(self):
        return self.state is TrackState.tentative

    @property
    def is_deleted(self):
        return self.state is TrackState.deleted
