from vcap import (
    BaseCapsule,
    NodeDescription,
    DeviceMapper,
    BoolOption,
    common_detector_options,
    IntOption,
    FloatOption
)
from vcap_utils import BackendRpcProcess
from .backend import Backend


class Capsule(BaseCapsule):
    name = "detector_person_vehicle_bike_openvino"
    description = ("OpenVINO person, vehicle, and bike detector. Optimized "
                   "for outdoor street crosswalk scenarios.")
    version = 1
    device_mapper = DeviceMapper.map_to_openvino_devices()
    input_type = NodeDescription(size=NodeDescription.Size.NONE)
    output_type = NodeDescription(
        size=NodeDescription.Size.ALL,
        detections=["vehicle", "person", "bike"])
    backend_loader = lambda capsule_files, device: BackendRpcProcess(
        Backend,
        model_xml=capsule_files[
            "person-vehicle-bike-detection-crossroad-1016-fp16.xml"],
        weights_bin=capsule_files[
            "person-vehicle-bike-detection-crossroad-1016-fp16.bin"],
        device_name=device
    )
    options = {
        **common_detector_options,
        "include_person_detections": BoolOption(default=True),
        "include_vehicle_detections": BoolOption(default=True),
        "include_bike_detections": BoolOption(default=True),
        "min_detection_area": IntOption(
            default=0,
            min_val=0,
            max_val=None),
        "max_detection_area": IntOption(
            default=99999999,
            min_val=0,
            max_val=None),
        "max_detection_overlap": FloatOption(
            default=1.0,
            min_val=0.0,
            max_val=1.0)
    }
