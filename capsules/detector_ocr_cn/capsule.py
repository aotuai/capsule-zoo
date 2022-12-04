from vcap import (
    BaseCapsule,
    NodeDescription,
    DeviceMapper,
    BaseStreamState,
    common_detector_options,
)

from .backend import Backend
detection_confidence = "confidence"

class Capsule(BaseCapsule):
    name = "detector_ocr_cn"
    description = "✨ v1.0 Chinese OCR text detector and reader."
    version = 1
    stream_state = BaseStreamState
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
    options = common_detector_options
