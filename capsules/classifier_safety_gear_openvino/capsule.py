from vcap import BaseCapsule, NodeDescription, DeviceMapper

from .backend import Backend

from .config import (
    safety_hat,
    safety_vest,
    with_safety_hat,
    with_safety_vest,
    capsule_options,
)


class Capsule(BaseCapsule):
    name = "classifier_safety_gear_openvino"
    description = "Roughly identify if person is wearing safety hat " \
                  "and safety vest."
    version = 1
    device_mapper = DeviceMapper.map_to_openvino_devices()
    input_type = NodeDescription(
        size=NodeDescription.Size.ALL,
        detections=["person"])
    output_type = NodeDescription(
        size=NodeDescription.Size.ALL,
        detections=["person"],
        attributes={safety_hat: with_safety_hat,
                    safety_vest: with_safety_vest},
        extra_data=["safety_hat_iou", "safety_hat_confidence",
                    "safety_vest_iou", "safety_vest_confidence"],
    )
    backend_loader = lambda capsule_files, device: Backend(
        model_xml=capsule_files["worker_safety_mobilenet_FP16.xml"],
        weights_bin=capsule_files["worker_safety_mobilenet_FP16.bin"],
        device_name=device
    )
    options = capsule_options
