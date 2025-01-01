from vcap import BaseCapsule, NodeDescription, FloatOption, IntOption, TextOption
from .backend import Backend

class Capsule(BaseCapsule):
    name = "chatgpt_classifier_agent"
    description = "✨ v1.0. The capsule takes detections from other capsules, and send a prompt request " \
                  "to OpenAI API for classification and publish the answers to BrainFrame API or for other "\
                  "capsules to consume and generate fused results."
    version = 1
    input_type = NodeDescription(
        size=NodeDescription.Size.ALL,
        detections=["person"]
    )
    output_type = NodeDescription(
        size=NodeDescription.Size.ALL,
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
            default = "Does this person in the picture wear a safety hat? Please reply with True or False. Please do not include anything else other than the number",
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
