from vcap import BaseCapsule, DeviceMapper, NodeDescription

from .backend import ATTRIBUTES, Backend, options


class Capsule(BaseCapsule):
    name = "classifier_person_attributes_openvino"
    description = "v1.1 OpenVINO powered person classifier, " \
                  "for general person appearance attributes."
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
    backend_loader = lambda capsule_files, device: Backend(
        model_xml=capsule_files[
            "person-attributes-recognition-crossroad-0230-fp16.xml"],
        weights_bin=capsule_files[
            "person-attributes-recognition-crossroad-0230-fp16.bin"],
        device_name=device
    )
    options = options
