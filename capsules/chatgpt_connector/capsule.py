from vcap import BaseCapsule, NodeDescription, FloatOption, IntOption, TextOption
from .backend import Backend

class Capsule(BaseCapsule):
    name = "chatgpt_connector"
    version = 1
    input_type = NodeDescription(size=NodeDescription.Size.NONE)
    output_type = NodeDescription(
        size=NodeDescription.Size.SINGLE,
        detections=["chatgpt"],
        extra_data=["chatgpt"]
        )
    backend_loader = lambda capsule_files, device: Backend()
    options = {
        "api_key": TextOption(
            default = "",
            description ="Fill in the API Key",
        ),
        "http_proxy": TextOption(
            default = "",
            description ="If an http_proxy is required to access the service",
        ),
        "prompt": TextOption(
            default = "How many people wearing safety hat in the picture? Please reply with the number, for example: 0. Do not include anything else other than the number",
            description ="The prompt sent to the service",
        ),
        "model": TextOption(
            default = "gpt-4o",
            description ="The model name provided by the API",
        ),
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
        )
        }
