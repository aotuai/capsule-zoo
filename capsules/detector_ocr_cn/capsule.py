from vcap import (
    BaseCapsule,
    NodeDescription,
    DeviceMapper,
    BoolOption,
    IntOption,
    BaseStreamState,
    common_detector_options,
)

from .backend import Backend
detection_confidence = "confidence"

class Capsule(BaseCapsule):
    name = "detector_ocr_cn"
    description = "✨ v1.1 OCR text detector and recognition: support over 6000 Chinese charactors."
    version = 1
    stream_state = BaseStreamState
    device_mapper = DeviceMapper.map_to_single_cpu()
    input_type = NodeDescription(
        size=NodeDescription.Size.SINGLE,
        detections=["home_cell", "home_cell object", "home_cell object noise",
                    "roaming_cell", "roaming_cell object", "roaming_cell object noise",
                    "screen"])
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
        "screen": BoolOption(default=False),
        "cell": BoolOption(default=False),
        "object": BoolOption(default=True),
        "noise": BoolOption(default=True),
        "min_detection_area": IntOption(
            default=0,
            min_val=0,
            max_val=None),
        "max_detection_area": IntOption(
            default=99999999,
            min_val=0,
            max_val=None),
    }


