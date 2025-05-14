from vcap import BaseCapsule, NodeDescription, FloatOption, IntOption, TextOption, DeviceMapper, BoolOption
from .backend import Backend, out_class_name
from .stream_state import StreamState


class Capsule(BaseCapsule):
    name = "classification_detections_filter"
    description = "v0.1 The capsule takes detections from other capsules, filter the detections based on "\
                  "the options setting and present the filtered results."
    version = 1
    stream_state = StreamState
    device_mapper = DeviceMapper.map_to_single_cpu()
    input_type = NodeDescription(
        size=NodeDescription.Size.ALL,
    )
    output_type = NodeDescription(
        size=NodeDescription.Size.ALL,
        detections=[out_class_name],
        extra_data=[out_class_name]
    )
    backend_loader = lambda capsule_files, device: Backend()
    options = {
        "class_name": TextOption(
            default="person",
            description="filtration class name, e.g.: person",
        ),
        "attributes_category": TextOption(
            default="behavior",
            description="filtration attributes category, e.g.: behavior",
        ),
        "attributes_values": TextOption(
            default='["drinking", "phoning", "smoking"]',
            description='filtration attributes values',
            # it's json format string of list, e.g.: '["drinking", "phoning", "smoking"]'
        ),
        "time_window_duration": IntOption(
            default=20,
            min_val=1,
            max_val=3600,
            description="the time window duration, unit: Seconds",
        ),
        "true_counter": IntOption(
            default=6,
            min_val=1,
            max_val=86400,
            description="The number of counts required to confirm the status within the given time window",
        )
    }
