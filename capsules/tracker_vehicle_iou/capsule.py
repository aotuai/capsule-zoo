from vcap import (
    BaseCapsule,
    NodeDescription,
    DeviceMapper
)

from .backend import Backend
from .stream_state import StreamState
from . import config


class Capsule(BaseCapsule):
    name = "tracker_vehicle"
    description = "✨ Efficient vehicle tracker using IOU."
    version = 1
    device_mapper = DeviceMapper.map_to_single_cpu()
    stream_state = StreamState
    input_type = NodeDescription(
        size=NodeDescription.Size.ALL,
        detections=config.tracks_classes,
        extra_data=["detection_confidence"])
    output_type = NodeDescription(
        size=NodeDescription.Size.ALL,
        tracked=True,
        detections=config.tracks_classes,
        extra_data=["detection_confidence"])
    backend_loader = lambda capsule_files, device: Backend()
    options = config.capsule_options
