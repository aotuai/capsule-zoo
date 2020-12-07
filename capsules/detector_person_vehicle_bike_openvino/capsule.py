from vcap import (
    BaseCapsule,
    NodeDescription,
    DeviceMapper,
    BoolOption,
    common_detector_options
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
            "person-vehicle-bike-detection-crossroad-1016-fp32.xml"],
        weights_bin=capsule_files[
            "person-vehicle-bike-detection-crossroad-1016-fp32.bin"],
        device_name=device
    )
    options = {
        **common_detector_options,
        "only_person_detections": BoolOption(
            default=False,
            description="Filter out anything that's not a person detection")
    }
