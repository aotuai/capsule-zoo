from vcap import BaseCapsule, NodeDescription, DeviceMapper
from .backend import Backend, EMOTION_TYPES


class Capsule(BaseCapsule):
    name = "classifier_face_emotion_openvino"
    description = "OpenVINO face emotion classifier."
    version = 1
    device_mapper = DeviceMapper.map_to_openvino_devices()
    input_type = NodeDescription(
        size=NodeDescription.Size.SINGLE,
        detections=["face"])
    output_type = NodeDescription(
        size=NodeDescription.Size.SINGLE,
        detections=["face"],
        attributes={"emotion": EMOTION_TYPES},
        extra_data=["emotion_confidence"]
    )
    backend_loader = lambda capsule_files, device: Backend(
        model_xml=capsule_files[
            "emotions-recognition-retail-0003-fp16.xml"],
        weights_bin=capsule_files[
            "emotions-recognition-retail-0003-fp16.bin"],
        device_name=device
    )
