from vcap import BaseCapsule, NodeDescription, IntOption, TextOption, DeviceMapper
from .backend import Backend
from .stream_state import StreamState

class Capsule(BaseCapsule):
    name = "claude_classifier_agent"
    description = "v0.2. The capsule takes detections from other capsules, and send a prompt request " \
                  "to Anthropic API for classification and publish the answers to BrainFrame API or for other "\
                  "capsules to consume and generate fused results."
    version = 1
    stream_state = StreamState
    device_mapper = DeviceMapper.map_to_single_cpu()
    input_type = NodeDescription(
        size=NodeDescription.Size.ALL,
        detections=["person"]
    )
    output_type = NodeDescription(
        size=NodeDescription.Size.ALL,
        detections=["claude"],
        extra_data=["claude"]
    )
    backend_loader = lambda capsule_files, device: Backend()
    options = {
        "model": TextOption(
            default = "claude-haiku-5-5",
            description ="The model name provided by the API. The default, Claude Haiku 5.5, is the cheapest current model "
                         "(about US$0.10 per million input tokens); use claude-sonnet-5-5 or claude-opus-5-5 for harder questions.",
        ),
        "input_detections": TextOption(
            default = "person",
            description ="The input detection name",
        ),
        "output_attributes": TextOption(
            default = "helmet",
            description ="The classification results will be included in output attributes as True or False",
        ),
        "prompt": TextOption(
            default = "Does this person in the picture wear a safety hat? Please reply with True or False. Please do not include anything else other than the word True or False",
            description ="The prompt sent to the service",
        ),
        "detection_interval": IntOption(
            default = 60,
            min_val = 0,
            max_val = 86400,
            description ="The time interval for detection, measured in seconds."
        ),
        "api_key": TextOption(
            default = "",
            description ="Fill in the API Key",
        ),
        "http_proxy": TextOption(
            default = "",
            description ="If an http_proxy is required to access the service",
        ),
        "effort": TextOption(
            default = "low",
            description ="How much the model thinks before it answers: low, medium, high, xhigh or max. 'low' is cheapest and "
                         "enough for a True/False question. Leave it empty for a model that has no effort setting (Claude Haiku 4.5, Sonnet 4.5 or older).",
        ),
        "max_tokens": IntOption(
            default = 1024,
            min_val = 1,
            max_val = 8192,
            description ="The maximum number of tokens to generate before stopping. It also covers the model's thinking, so keep it above a few hundred."
        )
    }
