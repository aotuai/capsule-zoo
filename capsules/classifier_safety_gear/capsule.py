from vcap import (
    BaseCapsule,
    NodeDescription,
    DeviceMapper,
)
from .backend import Backend
from .config import (
    safety_hat,
    safety_vest,
    with_safety_hat,
    with_safety_vest,
    capsule_options,
    attributes,
)


class Capsule(BaseCapsule):
    name = "classifier_safety_gear_openvino"
    description = "Roughly identify if person wearing safety hat " \
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
        extra_data=[attributes[safety_hat]["iou"],
                    attributes[safety_hat]["confidence"],
                    attributes[safety_vest]["iou"],
                    attributes[safety_vest]["confidence"],
                    ],
    )
    backend_loader = lambda capsule_files, device: Backend(
        model_xml=capsule_files["worker_safety_mobilenet_FP16.xml"],
        weights_bin=capsule_files["worker_safety_mobilenet_FP16.bin"],
        device_name=device
    )
    options = capsule_options
