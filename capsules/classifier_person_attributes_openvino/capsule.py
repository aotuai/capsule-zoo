from vcap import BaseCapsule, NodeDescription, DeviceMapper
from vcap_utils import BackendRpcProcess

from .backend import Backend, ATTRIBUTES, options


class Capsule(BaseCapsule):
    name = "classifier_face_age_gender_openvino"
    description = "OpenVINO face age/gender classifier."
    version = 1
    device_mapper = DeviceMapper.map_to_openvino_devices()
    input_type = NodeDescription(
        size=NodeDescription.Size.SINGLE,
        detections=["person"])
    output_type = NodeDescription(
        size=NodeDescription.Size.SINGLE,
        detections=["person"],
        attributes=ATTRIBUTES,
    )
    backend_loader = lambda capsule_files, device: BackendRpcProcess(
        Backend,
        model_xml=capsule_files[
            "person-attributes-recognition-crossroad-0230.xml"],
        weights_bin=capsule_files[
            "person-attributes-recognition-crossroad-0230.bin"],
        device_name=device
    )
    options = options
