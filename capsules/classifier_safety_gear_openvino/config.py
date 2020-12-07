from vcap import FloatOption

safety_hat = "safety_hat"
safety_vest = "safety_vest"

safety_gears = [safety_hat, safety_vest]

with_safety_hat = ["without_safe_hat", "with_safe_hat"]
with_safety_vest = ["without_safe_vest", "with_safe_vest"]

confidence_threshold = "confidence_threshold"
safety_hat_iou_threshold = "safety_hat_iou_threshold"
safety_vest_iou_threshold = "safety_vest_iou_threshold"

attributes = {
    safety_hat: {
        "attribute_name": "safety_hat",
        "iou": "safety_hat_iou",
        "confidence": "safety_hat_confidence",
        "iou_threshold": safety_hat_iou_threshold,
        "possible_values": with_safety_hat
    },
    safety_vest: {
        "attribute_name": "safety_vest",
        "iou": "safety_vest_iou",
        "confidence": "safety_vest_confidence",
        "iou_threshold": safety_vest_iou_threshold,
        "possible_values": with_safety_vest

    }
}

label_map = {1: "unknown", 2: "safety vest", 3: "unknown", 4: "safety hat"}

capsule_options = {
    confidence_threshold: FloatOption(
        default=0.5,
        min_val=0.0,
        max_val=1.0
    ),
    safety_hat_iou_threshold: FloatOption(
        default=0.1,
        min_val=0.0,
        max_val=1.0
    ),
    safety_vest_iou_threshold: FloatOption(
        default=0.5,
        min_val=0.0,
        max_val=1.0
    )
}
