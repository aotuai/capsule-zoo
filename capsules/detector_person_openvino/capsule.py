from vcap import (
    BaseCapsule,
    NodeDescription,
    DeviceMapper,
    FloatOption,
    common_detector_options
)
from vcap_utils import BackendRpcProcess
from .backend import Backend


class Capsule(BaseCapsule):
    name = "detector_person_openvino"
    description = "OpenVINO generic person detector."
    version = 1
    device_mapper = DeviceMapper.map_to_openvino_devices()
    input_type = NodeDescription(size=NodeDescription.Size.NONE)
    output_type = NodeDescription(
        size=NodeDescription.Size.ALL,
        detections=["person"])
    backend_loader = lambda capsule_files, device: BackendRpcProcess(
        Backend,
        model_xml=capsule_files["person-detection-0202-INT8.xml"],
        weights_bin=capsule_files["person-detection-0202-INT8.bin"],
        device_name=device
    )
    options = {
        **common_detector_options,
        "max_detection_overlap": FloatOption(
            default=1.0,
            min_val=0.0,
            max_val=1.0)
    }
