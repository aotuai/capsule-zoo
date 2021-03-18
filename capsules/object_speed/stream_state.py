from collections import OrderedDict
from typing import Optional, Tuple
from uuid import UUID
from dataclasses import dataclass

from vcap import BaseStreamState


@dataclass
class DetectionTimestamp:
    tstamp: float
    """The timestamp when the detection was seen"""

    coordinate: Tuple[int, int]
    """The coordinate where the object was seen at that timestamp. This 
    represents the centroid of the bounding box of the detection. 
    """


class StreamState(BaseStreamState):
    """Basically an LRU cache that stores the last-seen timestamp and
    coordinate keyed to detection.track_id
    """

    def __init__(self):
        self.cache = OrderedDict()
        """How many detection's information it can keep in the cache"""

    def get(self, key: UUID) -> Optional[DetectionTimestamp]:
        """
        :param key: The detection.track_id
        :return: (timestamp, (x, y)) or (None, None) if there was no detection
            information for this track ID
        """
        if key not in self.cache:
            return None
        else:
            self.cache.move_to_end(key)
            return self.cache[key]

    def put(self, key: UUID,
            detection_tstamp: DetectionTimestamp,
            capacity: float) -> None:
        """Records the new value and knocks out the least-recently-used value

        :param key: The detection.track_id
        :param detection_tstamp: The new DetectionTimestamp
        :param capacity: How many values the OrderedDict should be allowed to
        hold
        """
        self.cache[key] = detection_tstamp
        self.cache.move_to_end(key)
        if len(self.cache) > capacity:
            self.cache.popitem(last=False)
