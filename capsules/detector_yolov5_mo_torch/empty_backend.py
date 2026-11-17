"""Used while this capsule has no model: it loads, reports how to add one, and detects nothing."""
import logging
from typing import Dict

from vcap import BaseBackend, DETECTION_NODE_TYPE, OPTION_TYPE, BaseStreamState

logger = logging.getLogger(__name__)

NO_MODEL_MESSAGE = (
    "detector_yolov5_mo_torch has no model, so it detects nothing. Add model.xml and model.bin "
    "(an OpenVINO YOLO model) to this capsule and repackage it, or package your model with the "
    "BrainFrame Marketplace (Package Model, or Download & Package). See README.md."
)


class EmptyBackend(BaseBackend):
    def __init__(self):
        super().__init__()
        logger.warning(NO_MODEL_MESSAGE)

    def process_frame(self, frame, detection_node: DETECTION_NODE_TYPE, options: Dict[str, OPTION_TYPE],
                      state: BaseStreamState) -> DETECTION_NODE_TYPE:
        return []

    def close(self) -> None:
        super().close()  # stops the batch thread that BaseBackend starts
