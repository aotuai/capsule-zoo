from vcap import (
    BaseCapsule,
    NodeDescription,
    DeviceMapper,
    common_detector_options
)
from vcap_utils import BackendRpcProcess
from .backend import Backend


class Capsule(BaseCapsule):
    name = "detector_face_openvino"
    description = "✨ OpenVINO fast face detector."
    version = 1
    device_mapper = DeviceMapper.map_to_openvino_devices()
    input_type = NodeDescription(size=NodeDescription.Size.NONE)
    output_type = NodeDescription(
        size=NodeDescription.Size.ALL,
        detections=["face"])
    backend_loader = lambda capsule_files, device: BackendRpcProcess(
        Backend,
        model_xml=capsule_files["face-detection-adas-0001-fp16.xml"],
        weights_bin=capsule_files["face-detection-adas-0001-fp16.bin"],
        device_name=device
    )
    options = common_detector_options
