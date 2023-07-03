from vcap import (
    BaseCapsule,
    NodeDescription,
    DeviceMapper,
    BoolOption,
    IntOption,
    common_detector_options,
)

from .backend import Backend
from .ocr_state import StreamState

detection_confidence = "confidence"


class Capsule(BaseCapsule):
    name = "detector_ocr_cn"
    description = "✨ v1.2 OCR text detector and recognition: support over 6000 Chinese charactors."
    version = 1
    stream_state = StreamState
    device_mapper = DeviceMapper.map_to_single_cpu()
    input_type = NodeDescription(size=NodeDescription.Size.NONE)
    output_type = NodeDescription(
        size=NodeDescription.Size.ALL,
        detections=["text"],
        extra_data=[detection_confidence, "ocr"])
    backend_loader = lambda capsule_files, device: Backend(
        device=device,
        det_bytes=capsule_files["models/-1_3_640_640_det.onnx"],
        rec_bytes=capsule_files["models/-1_3_640_640_rec.onnx"],
        dict_bytes=capsule_files["ppocr_keys_v1.txt"]
    )
    options = {
        **common_detector_options,
        "cell": BoolOption(default=True),
        "cell_x": IntOption(
            default=115, min_val=0, max_val=None,
            description="X coordinate at the upper left corner of the cell."),
        "cell_y": IntOption(
            default=0, min_val=0, max_val=None,
            description="Y coordinate at the upper left corner of the cell."),
        "cell_width": IntOption(
            default=410, min_val=10, max_val=None,
            description="The width of the cell."),
        "cell_height": IntOption(
            default=720, min_val=10, max_val=None,
            description="The height of the cell."),
    }
