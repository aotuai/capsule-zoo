from vcap import (
    BaseCapsule,
    NodeDescription,
    DeviceMapper,
    BoolOption,
    IntOption,
    common_detector_options,
)

from .backend import Backend

detection_confidence = "confidence"
category = "block_name"
values = ["identity_card", "signature", "reception", "portrait", "noname",
          "block_1", "block_2", "block_3", "block_4", "block_5", "block_6", "block_7", "block_8"]

class Capsule(BaseCapsule):
    name = "classifier_block_in_image"
    description = "✨ v1.0 segment blocks in image which have horizontal and vertical lines, and give a name to block according to ocr text detections."
    version = 1
    #stream_state = StreamState
    #device_mapper = DeviceMapper.map_to_single_cpu()

    input_type = NodeDescription(size=NodeDescription.Size.ALL,
        detections = ["text","face","person"])

    output_type = NodeDescription(
        size=NodeDescription.Size.ALL,
        detections=["text"],
        attributes = {category: values})
        #extra_data=[detection_confidence, "ocr"])

    backend_loader = lambda capsule_files, device: Backend()

    options = {
        "split_mode": IntOption(
            default=0, min_val=None, max_val=None,
            description="the segment mode of horizontal and vertical lines."),
        "min_line_spacing": IntOption(
            default=12, min_val=0, max_val=None,
            description="the minimum line spacing."),
        "min_vertical_spacing": IntOption(
            default=12, min_val=0, max_val=None,
            description="the minimum vertical spacing."),
        "lines_pad": IntOption(
            default=10, min_val=0, max_val=None,
            description="Determine the pad value of similar horizontal lines"),
        "vertical_pad": IntOption(
            default=10, min_val=0, max_val=None,
            description="Determine the pad value of similar vertical lines"),
        "min_rect_width": IntOption(
            default=537, min_val=0, max_val=None,
            description="the minimum width of rectangular block."),
        "min_rect_height": IntOption(
            default=300, min_val=0, max_val=None,
            description="the minimum height of rectangular block."),

        "ic_block": IntOption(
            default=1, min_val=0, max_val=None,
            description="The block number where the identity_card block resides"),

        "portrait_block": IntOption(
            default=2, min_val=0, max_val=None,
            description="The block number where the face_photo block resides"),

        "signature_block": IntOption(
            default=3, min_val=0, max_val=None,
            description="The block number where the signature block resides"),

        "ic_back_block": IntOption(
            default=4, min_val=0, max_val=None,
            description="The block number where the face_photo block resides"),

        "reception_block": IntOption(
            default=6, min_val=0, max_val=None,
            description="The block number where the face_photo block resides"),
    }
