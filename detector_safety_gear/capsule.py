from vcap import (
    BaseCapsule,
    NodeDescription,
    DeviceMapper,
    FloatOption
)
from vcap_utils import BackendRpcProcess
from .backend import Backend


class Capsule(BaseCapsule):
    name = "detector_safety_gear_openvino"
    version = 1
    device_mapper = DeviceMapper.map_to_openvino_devices()
    input_type = NodeDescription(size=NodeDescription.Size.NONE)
    output_type = NodeDescription(
        size=NodeDescription.Size.ALL,
        detections=["safety vest", "safety hat"]
    )
    backend_loader = lambda capsule_files, device: Backend(
        model_xml=capsule_files["worker_safety_mobilenet_FP16.xml"],
        weights_bin=capsule_files["worker_safety_mobilenet_FP16.bin"],
        device_name=device
    )
    options = {
        "threshold": FloatOption(
            default=0.5,
            min_val=0.0,
            max_val=None
        )
    }
