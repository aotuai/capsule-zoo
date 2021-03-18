from collections import OrderedDict
from typing import Union, Tuple
from uuid import UUID

from vcap import BaseStreamState


class StreamState(BaseStreamState):
    """Basically an LRU cache that stores the last-seen timestamp and
    coordinate keyed to detection.track_id
    """

    def __init__(self):
        self.cache = OrderedDict()
        """How many detection's information it can keep in the cache"""

    def get(self, key: UUID) -> Union[Tuple[float, Tuple[int]],
                                      Tuple[None, None]]:
        """
        :param key: The detection.track_id
        :return: (timestamp, (x, y)) or (None, None) if there was no detection
            information for this track ID
        """
        if key not in self.cache:
            return None, None
        else:
            self.cache.move_to_end(key)
            return self.cache[key]

    def put(self, key: UUID,
            value: Tuple[float, Tuple[int]],
            capacity) -> None:
        """Records the new value and knocks out the least-recently-used value

        :param key: The detection.track_id
        :param value: (timestamp, (x, y)) of the detection
        :param capacity: How many values the OrderedDict should be allowed to
        hold
        """
        self.cache[key] = value
        self.cache.move_to_end(key)
        if len(self.cache) > capacity:
            self.cache.popitem(last=False)
