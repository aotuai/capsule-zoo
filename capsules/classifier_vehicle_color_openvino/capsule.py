from vcap import BaseCapsule, NodeDescription, DeviceMapper
from .backend import Backend
from . import config


class Capsule(BaseCapsule):
    name = "classifier_vehicle_color_openvino"
    description = "CPU-only vehicle color classifier."
    version = 1
    device_mapper = DeviceMapper.map_to_single_cpu()
    input_type = NodeDescription(
        size=NodeDescription.Size.SINGLE,
        detections=config.vehicle_types)
    output_type = NodeDescription(
        size=NodeDescription.Size.SINGLE,
        detections=config.vehicle_types,
        attributes={"color": config.colors})
    backend_loader = lambda capsule_files, device: Backend(
        model_xml=capsule_files[
            "vehicle-attributes-recognition-barrier-0039.xml"],
        weights_bin=capsule_files[
            "vehicle-attributes-recognition-barrier-0039.bin"],
        device_name=device
    )
