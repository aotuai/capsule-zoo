from typing import Optional

from vcap import BaseStreamState

from .tracker import Tracker


class StreamState(BaseStreamState):
    def __init__(self):
        self.tracker: Optional[Tracker] = None
        self.tracker_options = None

    def get_tracker(self, **options) -> Tracker:
        """Get a tracker with the given capsule options, or create a new one
        if the capsule options have changed."""
        if self.tracker is None or options != self.tracker_options:
            self.tracker = Tracker(**options)
            self.tracker_options = options
        return self.tracker
