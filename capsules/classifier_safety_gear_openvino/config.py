from vcap import FloatOption

safety_hat = "safety_hat"
safety_vest = "safety_vest"

gear_types = [safety_hat, safety_vest]

with_safety_hat = ["without_safety_hat", "with_safety_hat"]
with_safety_vest = ["without_safety_vest", "with_safety_vest"]

confidence_threshold = "confidence_threshold"
safety_hat_iou_threshold = "safety_hat_iou_threshold"
safety_vest_iou_threshold = "safety_vest_iou_threshold"

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
