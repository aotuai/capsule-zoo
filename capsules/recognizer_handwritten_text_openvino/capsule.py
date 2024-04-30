from vcap import (
    BaseCapsule,
    NodeDescription,
    DeviceMapper,
    common_detector_options,
    BoolOption,
    IntOption
)
from .backend import Backend

detection_confidence = "confidence"

class Capsule(BaseCapsule):
    name = "recognizer_handwritten_text_openvino"
    description = "✨ v1.0 OpenVINO handwritten simplified chinese text (only one line) recognition."
    version = 1
    device_mapper = DeviceMapper.map_to_single_cpu()
    input_type = NodeDescription(size=NodeDescription.Size.NONE)
    output_type = NodeDescription(
        size=NodeDescription.Size.ALL,
        detections=["text"],
        extra_data=[detection_confidence, "ocr"])
    backend_loader = lambda capsule_files, device: Backend(
        model_xml=capsule_files["handwritten-simplified-chinese-recognition-0001-fp16.xml"],
        weights_bin=capsule_files["handwritten-simplified-chinese-recognition-0001-fp16.bin"],
        char_list=capsule_files["scut_ept.txt"],
        device_name=device
    )
    options = {
        **common_detector_options,
        "multi_line": BoolOption(default=False),
        "to_bold_strokes": BoolOption(default=True),
        "strokes_pad": IntOption(
            default=20, min_val=0, max_val=None,
            description="The pad value for up/down/left/right of strokes."),
        "cell": BoolOption(default=True),
        "cell_x": IntOption(
            default=37, min_val=0, max_val=None,
            description="X coordinate at the upper left corner of the cell."),
        "cell_y": IntOption(
            default=870, min_val=0, max_val=None,
            description="Y coordinate at the upper left corner of the cell."),
        "cell_width": IntOption(
            default=530, min_val=10, max_val=None,
            description="The width of the cell."),
        "cell_height": IntOption(
            default=346, min_val=10, max_val=None,
            description="The height of the cell."),
    }
