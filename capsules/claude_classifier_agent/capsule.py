from vcap import BaseCapsule, NodeDescription, FloatOption, IntOption, TextOption
from .backend import Backend

class Capsule(BaseCapsule):
    name = "claude_classifier_agent"
    description = "✨ v1.0. The capsule takes detections from other capsules, and send a prompt request " \
                  "to Anthropic API for classification and publish the answers to BrainFrame API or for other "\
                  "capsules to consume and generate fused results."
    version = 1
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
        "api_key": TextOption(
            default = "",
            description ="Fill in the API Key",
        ),
        "http_proxy": TextOption(
            default = "",
            description ="If an http_proxy is required to access the service",
        ),
        "prompt": TextOption(
            default = "Does this person in the picture wear a safety hat? Please reply with True or False. Please do not include anything else other than the number",
            description ="The prompt sent to the service",
        ),
        "model": TextOption(
            default = "claude-3-5-sonnet-20240620",
            description ="The model name provided by the API",
        ),
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
