
from vcap import (
    BaseCapsule,
    NodeDescription,
    FloatOption,
)

from .backend import Backend

class Capsule(BaseCapsule):
    name = "recognizer_identitycard_face"
    description = "✨ v1.1 Recognize and compare identity card face and the other face. Works best close-up."
    version = 1
    input_type = NodeDescription(
        size=NodeDescription.Size.ALL,
        detections=["face"])
    output_type = NodeDescription(
        size=NodeDescription.Size.ALL,
        detections=["face_compare"],
        extra_data=["face_compare_confidence"]
        )
    backend_loader = lambda capsule_files, device: Backend(
        device=device,
        model_bytes=capsule_files["encoder_face_center_loss.pb"],
        model_name="vggface2_center_loss")
    options = {
        "threshold": FloatOption(
            default=0.34,
            min_val=0.0,
            max_val=1.0)
    }
