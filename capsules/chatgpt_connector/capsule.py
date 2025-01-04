from vcap import BaseCapsule, NodeDescription, FloatOption, IntOption
from .backend import Backend
from .stream_state import StreamState

class Capsule(BaseCapsule):
    name = "chatgpt_connector"
    version = 1
    stream_state = StreamState
    input_type = NodeDescription(size=NodeDescription.Size.NONE)
    output_type = NodeDescription(
        size=NodeDescription.Size.SINGLE,
        detections=["chatgpt"],
        extra_data=["chatgpt"]
        )
    backend_loader = lambda capsule_files, device: Backend()
    options = {
        "temperature": FloatOption(
            default = 1.0,
            min_val = 0.0,
            max_val = 2.0,
            description ="What sampling temperature to use, between 0 and 2. Higher values like 0.8 will make the output more random, while lower values like 0.2 will make it more focused and deterministic."
        ),
        "max_tokens": IntOption(
            default = 256,
            min_val = 0,
            max_val = 8192,
            description ="The maximum number of tokens that can be generated in the chat completion."
        ),
        "detection_interval": IntOption(
            default = 0,
            min_val = 0,
            max_val = 85400,
            description ="The time interval for detection, measured in seconds."
        )
        }