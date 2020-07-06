from vcap import (
    BaseCapsule,
    NodeDescription,
    DeviceMapper,
    common_detector_options
)
from .backend import Backend


class Capsule(BaseCapsule):
    name = "detector_person_overhead_openvino"
    description = "OpenVINO fast person detector. Works best in " \
                  "surveillance perspectives from a downwards facing point " \
                  "of view."
    version = 1
    device_mapper = DeviceMapper.map_to_all_myriad()
    input_type = NodeDescription(size=NodeDescription.Size.NONE)
    output_type = NodeDescription(
        size=NodeDescription.Size.ALL,
        detections=["person"])
    backend_loader = lambda capsule_files, device: Backend(
        model_xml=capsule_files["person-detection-retail-0013.xml"],
        weights_bin=capsule_files["person-detection-retail-0013.bin"],
        device_name=device
    )
    options = common_detector_options
