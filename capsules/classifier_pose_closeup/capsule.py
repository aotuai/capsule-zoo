from vcap import (
    BaseCapsule,
    NodeDescription
)

from . import config
from .backend import Backend


class Capsule(BaseCapsule):
    name = "classifier_pose_closeup"
    description = "✨ Roughly identify the current pose of a person."
    version = 1
    input_type = NodeDescription(
        size=NodeDescription.Size.ALL,
        detections=["person"])
    output_type = NodeDescription(
        size=NodeDescription.Size.ALL,
        detections=["person"],
        attributes={config.pose: config.all_poses},
        extra_data=[config.pose_confidence, config.pose_iou])
    backend_loader = lambda capsule_files, device: Backend(
        device=device,
        model_bytes=capsule_files["detector.pb"],
        metadata_bytes=capsule_files["dataset_metadata.json"])
    options = config.capsule_options
