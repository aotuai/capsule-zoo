from vcap import BaseCapsule, NodeDescription, DeviceMapper
from vcap_utils import BackendRpcProcess
from .backend import Backend
from . import config


class Capsule(BaseCapsule):
    name = "classifier_vehicle_color_openvino"
    description = "OpenVINO vehicle color classifier."
    version = 1
    device_mapper = DeviceMapper.map_to_openvino_devices()
    input_type = NodeDescription(
        size=NodeDescription.Size.SINGLE,
        detections=config.vehicle_classifications + ["vehicle"])
    output_type = NodeDescription(
        size=NodeDescription.Size.SINGLE,
        detections=config.vehicle_classifications + ["vehicle"],
        attributes={"color": config.colors,
                    "vehicle_type": config.vehicle_classifications})
    backend_loader = lambda capsule_files, device: BackendRpcProcess(
        Backend,
        model_xml=capsule_files[
            "vehicle-attributes-recognition-barrier-0039.xml"],
        weights_bin=capsule_files[
            "vehicle-attributes-recognition-barrier-0039.bin"],
        device_name=device
    )
