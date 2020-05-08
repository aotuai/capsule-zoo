from typing import Dict
from uuid import uuid4, uuid5
import logging

import numpy as np

from vcap import (
    BaseBackend,
    DETECTION_NODE_TYPE,
    OPTION_TYPE)
from . import config
from .stream_state import StreamState


class Backend(BaseBackend):
    namespace = uuid4()

    def process_frame(self, frame: np.ndarray,
                      detection_nodes: DETECTION_NODE_TYPE,
                      options: Dict[str, OPTION_TYPE],
                      state: StreamState) -> DETECTION_NODE_TYPE:
        tracker = state.get_tracker(
            min_iou=options[config.min_iou_for_iou_match],
            max_misses=options[config.max_misses],
            n_hits_to_init=config.min_track_length)

        # Separate the license_plates from vehicles, to use different
        # tracking methods on them.
        license_plates = [det for det in detection_nodes
                          if det.class_name == "license_plate"]
        vehicles = [det for det in detection_nodes
                    if det.class_name != "license_plate"]

        tracker.predict()
        tracker.update(vehicles)

        # License plate track_ids are based on the "license_plate_string"
        for license_plate in license_plates:
            if "license_plate_string" in license_plate.extra_data:
                plate_text = license_plate.extra_data["license_plate_string"]
                license_plate.track_id = uuid5(self.namespace, plate_text)
            else:
                logging.warning("Tracker received a license plate without a "
                                "'license_plate_string' extra data field. Has "
                                "the license plate reading capsule loaded yet?")
                license_plate.track_id = uuid4()

        return detection_nodes

    def close(self):
        """There are no models to unload for this backend."""
        pass
