from vcap import BaseCapsule, DeviceMapper, FloatOption, NodeDescription, TextOption

from .backend import Backend
from .labels import labels


class Capsule(BaseCapsule):
    name = __CAPSULE_NAME__
    description = __CAPSULE_DESCRIPTION__
    version = __CAPSULE_VERSION__
    device_mapper = DeviceMapper.map_to_openvino_devices()
    input_type = NodeDescription(size=NodeDescription.Size.NONE)
    output_type = NodeDescription(size=NodeDescription.Size.ALL, detections=labels__OUTPUT_EXTRA__)
    backend_loader = lambda capsule_files, device: Backend(
        model_xml=capsule_files["model.xml"],
        weights_bin=capsule_files["model.bin"],
        device_name=device,
    )
    options = {
        "threshold": FloatOption(
            default=0.25,
            min_val=0.01,
            max_val=1.0,
            description="The confidence threshold for filtering detections.",
        ),
__IOU_OPTION__    }
