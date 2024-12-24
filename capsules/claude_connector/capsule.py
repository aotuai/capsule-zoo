from vcap import BaseCapsule, NodeDescription, FloatOption, IntOption
from .backend import Backend

class Capsule(BaseCapsule):
    name = "claude_connector"
    version = 1
    input_type = NodeDescription(size=NodeDescription.Size.NONE)
    output_type = NodeDescription(
        size=NodeDescription.Size.SINGLE,
        detections=["claude"],
        extra_data=["claude"]
        )
    backend_loader = lambda capsule_files, device: Backend()
    options = {
        "temperature": FloatOption(
            default = 1.0,
            min_val = 0.0,
            max_val = 1.0,
            description ="Amount of randomness injected into the response. Use temperature closer to 0.0 for analytical / multiple choice, and closer to 1.0 for creative and generative tasks."
        ),
        "max_tokens": IntOption(
            default = 256,
            min_val = 0,
            max_val = 8192,
            description ="The maximum number of tokens to generate before stopping."
        )
        }