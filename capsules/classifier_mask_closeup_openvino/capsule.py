from vcap import BaseCapsule, NodeDescription, DeviceMapper, FloatOption
from vcap_utils import BackendRpcProcess
from .backend import Backend


class Capsule(BaseCapsule):
    name = "classifier_mask_closeup_openvino"
    description = "OpenVINO face mask classifier."
    version = 1
    device_mapper = DeviceMapper.map_to_openvino_devices()
    input_type = NodeDescription(
        size=NodeDescription.Size.SINGLE,
        detections=["face"])
    output_type = NodeDescription(
        size=NodeDescription.Size.SINGLE,
        detections=["face"],
        attributes={"mask": Backend.LABELS},
        extra_data=["mask_confidence"])
    backend_loader = lambda capsule_files, device: BackendRpcProcess(
        Backend,
        model_xml=capsule_files["face_mask.xml"],
        weights_bin=capsule_files["face_mask.bin"],
        device_name=device
    )
    options = {
        "threshold": FloatOption(
            description="Scores under this value are deemed to not be not "
                        "wearing a mask.",
            default=0.3,
            min_val=None,
            max_val=None
        )
    }
