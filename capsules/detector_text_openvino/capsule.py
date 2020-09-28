from vcap import (
    BaseCapsule,
    NodeDescription,
    DeviceMapper,
    common_detector_options,
)

from .backend import Backend, OpenVINOModel


class Capsule(BaseCapsule):
    name = "detector_text_openvino"
    description = "✨ OpenVINO text detector."
    version = 1
    device_mapper = DeviceMapper.map_to_openvino_devices()
    input_type = NodeDescription(size=NodeDescription.Size.NONE)
    output_type = NodeDescription(
        size=NodeDescription.Size.ALL,
        detections=["text"],
        extra_data=["detection_confidence"])

    backend_loader = lambda capsule_files, device: Backend(
        detector=OpenVINOModel(
            model_xml=capsule_files[
                str("models/text-spotting-0001-detector.xml")],
            weights_bin=capsule_files[
                str("models/text-spotting-0001-detector.bin")],
            device_name=device
        ),
        recognizer_encoder=OpenVINOModel(
            model_xml=capsule_files[
                str("models/text-spotting-0001-recognizer-encoder.xml")],
            weights_bin=capsule_files[
                str("models/text-spotting-0001-recognizer-encoder.bin")],
            device_name=device
        ),
        recognizer_decoder=OpenVINOModel(
            model_xml=capsule_files[
                str("models/text-spotting-0001-recognizer-decoder.xml")],
            weights_bin=capsule_files[
                str("models/text-spotting-0001-recognizer-decoder.bin")],
            device_name=device
        )
    )
    options = common_detector_options
