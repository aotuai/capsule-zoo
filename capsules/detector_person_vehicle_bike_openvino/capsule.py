from vcap import (
    BaseCapsule,
    NodeDescription,
    DeviceMapper,
    common_detector_options
)
from .backend import Backend


class Capsule(BaseCapsule):
    name = "detector_person_vehicle_bike_openvino"
    description = ("OpenVINO person, vehicle, and bike detector. Optimized "
                   "for surveillance camera scenarios.")
    version = 1
    device_mapper = DeviceMapper.map_to_openvino_devices()
    input_type = NodeDescription(size=NodeDescription.Size.NONE)
    output_type = NodeDescription(
       size=NodeDescription.Size.ALL,
       detections=["vehicle", "person", "bike"])
    backend_loader = lambda capsule_files, device: Backend(
       model_xml=capsule_files["person-vehicle-bike-detection-crossroad-1016-fp32.xml"],
       weights_bin=capsule_files["person-vehicle-bike-detection-crossroad-1016-fp32.bin"],
       device_name=device
    )
    options = common_detector_options
