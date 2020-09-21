from vcap import (
    BaseCapsule,
    NodeDescription,
    DeviceMapper,
    common_detector_options,
    BoolOption
)
from vcap_utils import BackendRpcProcess
from .backend import Backend


class Capsule(BaseCapsule):
    name = "segmenter_person_openvino"
    description = "✨ OpenVINO person segmenter."
    version = 1
    device_mapper = DeviceMapper.map_to_openvino_devices()
    input_type = NodeDescription(size=NodeDescription.Size.NONE)
    output_type = NodeDescription(
        size=NodeDescription.Size.ALL,
        detections=["person"])
    backend_loader = lambda capsule_files, device: BackendRpcProcess(
        Backend,
        model_xml=capsule_files["instance-segmentation-security-0050.xml"],
        weights_bin=capsule_files["instance-segmentation-security-0050.bin"],
        device_name=device
    )
    options = {
        **common_detector_options,
        "keep_aspect_ratio": BoolOption(
            default=False,
            description="If True, it will keep the aspect ratio of "
                        "the image, then pad the sides before "
                        "feeding the image into the network."
        ),
        "convert_to_bounding_box": BoolOption(
            default=False,
            description="If True, it will convert segmentations to bounding "
                        "boxes."
        )
    }
