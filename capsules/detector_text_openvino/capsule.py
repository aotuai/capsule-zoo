from vcap import (
    BaseCapsule,
    NodeDescription,
    DeviceMapper,
    common_detector_options,
)

from .backend import Backend, OpenVINOModel


class Capsule(BaseCapsule):
    name = "detector_text_openvino"
    description = "✨ OpenVINO text detector and reader."
    version = 1
    device_mapper = DeviceMapper.map_to_openvino_devices()
    input_type = NodeDescription(size=NodeDescription.Size.NONE)
    output_type = NodeDescription(
        size=NodeDescription.Size.ALL,
        detections=["text"],
        extra_data=["detection_confidence", "text"])

    backend_loader = lambda capsule_files, device: Backend(
        detector=OpenVINOModel(
            model_xml=capsule_files[
                "models/text-spotting-0002-detector-fp16.xml"],
            weights_bin=capsule_files[
                "models/text-spotting-0002-detector-fp16.bin"],
            device_name=device
        ),
        recognizer_encoder=OpenVINOModel(
            model_xml=capsule_files[
                "models/text-spotting-0002-recognizer-encoder-fp16.xml"],
            weights_bin=capsule_files[
                "models/text-spotting-0002-recognizer-encoder-fp16.bin"],
            device_name=device
        ),
        recognizer_decoder=OpenVINOModel(
            model_xml=capsule_files[
                "models/text-spotting-0002-recognizer-decoder-fp16.xml"],
            weights_bin=capsule_files[
                "models/text-spotting-0002-recognizer-decoder-fp16.bin"],
            device_name=device
        )
    )
    options = common_detector_options
