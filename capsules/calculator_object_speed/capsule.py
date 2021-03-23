from vcap import (
    BaseCapsule,
    NodeDescription,
    DeviceMapper,
    IntOption
)

from .backend import Backend
from .stream_state import StreamState


class Capsule(BaseCapsule):
    name = "calculator_object_speed"
    description = "✨ Measure pixel-per-second speed on tracked detections, " \
                  "and puts the information in the extra_data field."
    version = 1
    device_mapper = DeviceMapper.map_to_single_cpu()
    stream_state = StreamState
    input_type = NodeDescription(
        size=NodeDescription.Size.SINGLE,
        tracked=True)
    output_type = NodeDescription(
        size=NodeDescription.Size.SINGLE,
        tracked=True,
        extra_data=["pixels_per_second_speed", "pixel_travel", "time_elapsed"])
    backend_loader = lambda capsule_files, device: Backend()
    options = {
        "cache_capacity": IntOption(
            description="The size of the cache that holds information "
                        "regarding previously tracked detections",
            min_val=1, max_val=None, default=1000)
    }
