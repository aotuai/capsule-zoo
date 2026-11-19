from vcap import BaseCapsule, DeviceMapper, EnumOption, FloatOption, NodeDescription

from .backend import Backend
from .labels import labels

input_classes = __INPUT_CLASSES__
attribute_name = __ATTRIBUTE_NAME__


class Capsule(BaseCapsule):
    name = __CAPSULE_NAME__
    description = __CAPSULE_DESCRIPTION__
    version = __CAPSULE_VERSION__
    device_mapper = DeviceMapper.map_to_openvino_devices()
    # A classifier follows a detector: it takes the detections of these classes and adds an attribute to each
    input_type = NodeDescription(size=NodeDescription.Size.SINGLE, detections=input_classes)
    output_type = NodeDescription(
        size=NodeDescription.Size.SINGLE,
        detections=input_classes,
        attributes={attribute_name: labels},
        extra_data=[attribute_name + "_confidence"],
    )
    backend_loader = lambda capsule_files, device: Backend(
        model_xml=capsule_files["model.xml"],
        weights_bin=capsule_files["model.bin"],
        device_name=device,
        attribute_name=attribute_name,
    )
    options = {
        "min_confidence": FloatOption(
            default=0.0,
            min_val=0.0,
            max_val=1.0,
            description="The attribute is only set when the best class is at least this sure (0: always set it).",
        ),
        "fit": EnumOption(
            default="center_crop",
            choices=["center_crop", "pad", "stretch"],
            description=(
                "How the detection's image is fitted to the model's square input. center_crop: the way "
                "Ultralytics classifiers are tested (scale to the short side, keep the middle). pad: keep the "
                "whole image, fill the rest. stretch: squeeze the whole image."
            ),
        ),
    }
