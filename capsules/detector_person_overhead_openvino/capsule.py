from vcap import (
    BaseCapsule,
    NodeDescription,
    DeviceMapper,
    common_detector_options
)
from vcap_utils import BackendRpcProcess
from .backend import Backend


class Capsule(BaseCapsule):
    name = "detector_person_overhead_openvino"
    description = "OpenVINO fast person detector. Works best in " \
                  "surveillance perspectives from a downwards facing point " \
                  "of view."
    version = 1
    device_mapper = DeviceMapper.map_to_openvino_devices()
    input_type = NodeDescription(size=NodeDescription.Size.NONE)
    output_type = NodeDescription(
        size=NodeDescription.Size.ALL,
        detections=["person"])
    backend_loader = lambda capsule_files, device: BackendRpcProcess(
        Backend,
        model_xml=capsule_files["person-detection-retail-0013-fp16.xml"],
        weights_bin=capsule_files["person-detection-retail-0013-fp16.bin"],
        device_name=device
    )
    options = common_detector_options
