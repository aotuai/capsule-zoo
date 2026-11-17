#
# Copyright (c) 2026 AOTU, Inc.  All rights reserved. Contains AOTU's Proprietary Information. RESTRICTED
# COMPUTER SOFTWARE. LIMITED RIGHTS DATA.
#

from vcap import BaseCapsule, DeviceMapper, FloatOption, NodeDescription

from .backend import Backend
from .empty_backend import EmptyBackend
from .labels import labels

MODEL_XML = "model.xml"
MODEL_BIN = "model.bin"


def load_backend(capsule_files, device):
    """The real detector when the capsule contains a model, else a backend that detects nothing."""
    if MODEL_XML in capsule_files and MODEL_BIN in capsule_files:
        return Backend(
            model_xml=capsule_files[MODEL_XML],
            weights_bin=capsule_files[MODEL_BIN],
            device_name=device,
        )
    return EmptyBackend()


class Capsule(BaseCapsule):
    name = "detector_yolov5_mo_torch"
    description = (
        "Template for YOLOv5-style object detectors (OpenVINO). It contains NO model, so it detects nothing until "
        "you add one: package your own OpenVINO model (model.xml + model.bin) with the BrainFrame Marketplace "
        "(Package Model, or Download & Package for Ultralytics models), or add the two files to this capsule and "
        "repackage it. Input: the video frame. Output: a list of detections, each with a class name, a box and "
        "a confidence (80 COCO classes by default, see labels.py). YOLO models from Ultralytics are licensed "
        "under AGPL-3.0: see README.md."
    )
    version = 1
    device_mapper = DeviceMapper.map_to_openvino_devices()
    input_type = NodeDescription(size=NodeDescription.Size.NONE)
    output_type = NodeDescription(size=NodeDescription.Size.ALL, detections=labels)
    backend_loader = load_backend
    options = {
        "threshold": FloatOption(
            default=0.25,
            min_val=0.01,
            max_val=1.0,
            description="The confidence threshold for filtering detections.",
        ),
        "iou_threshold": FloatOption(
            default=0.45,
            min_val=0.0,
            max_val=1.0,
            description="Non-max-suppression IOU threshold.",
        ),
    }
