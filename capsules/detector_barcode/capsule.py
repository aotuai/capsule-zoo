from vcap import (
    BaseCapsule,
    NodeDescription,
    DeviceMapper,
    EnumOption,
    common_detector_options,
)

from .backend import Backend


class Capsule(BaseCapsule):
    name = "detector_barcode"
    description = ("✨ v0.1 it performs detection and recognition (i.e., decoding the data) and fully supports Code 128, "
                   "Code 39, GS1-128, and QR Codes.")
    version = 1

    input_type = NodeDescription(size=NodeDescription.Size.NONE)

    output_type = NodeDescription(
        size=NodeDescription.Size.ALL,
        detections=["barcode"],
        extra_data = ["codetype", "code"])

    backend_loader = lambda capsule_files, device: Backend()

    options = {
        **common_detector_options,
        "mode": EnumOption(default="auto",
                           choices=["auto", "sequential", "parallel"],
                           description="the method of detecting strategy"),
    }
