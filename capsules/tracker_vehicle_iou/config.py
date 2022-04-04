from vcap import FloatOption, IntOption

min_iou_for_iou_match = "min_iou_for_iou_match"
high_detection_threshold = "high_detection_threshold"
min_track_length = "min_track_length"
max_misses = "max_misses"
tracks_classes = [
    "car",
    "motorcycle",
    "bus",
    "train",
    "truck",
    "boat",
    "vehicle",
    "license_plate",
    "bike",
    "special vehicle",
    "person"
]

capsule_options = {
    min_iou_for_iou_match: FloatOption(
        min_val=0.0,
        max_val=1.0,
        default=0.5),
    min_track_length: IntOption(
        min_val=0,
        max_val=None,
        default=2),
    max_misses: IntOption(
        default=10,
        min_val=0,
        max_val=None)
}
