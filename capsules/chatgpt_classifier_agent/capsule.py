from vcap import BaseCapsule, NodeDescription, FloatOption, IntOption, TextOption, DeviceMapper
from .backend import Backend

class Capsule(BaseCapsule):
    name = "chatgpt_classifier_agent"
    description = "✨ v1.0. The capsule takes detections from other capsules, and send a prompt request " \
                  "to OpenAI API for classification and publish the answers to BrainFrame API or for other "\
                  "capsules to consume and generate fused results."
    version = 1
    device_mapper = DeviceMapper.map_to_single_cpu()
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
        "model": TextOption(
            default = "gpt-4o",
            description ="The model name provided by the API",
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
        "max_calls_in_minute": IntOption(
            default = -1,
            min_val = -1,
            max_val = 60,
            description ="Limit the maximum number of calls in a minute. -1 is unlimited"
        ),
        "max_calls_in_hour": IntOption(
            default = -1,
            min_val = -1,
            max_val = 3600,
            description ="Limit the maximum number of calls in an hour. -1 is unlimited"
        ),
        "max_calls_in_day": IntOption(
            default = -1,
            min_val = -1,
            max_val = 86400,
            description ="Limit the maximum number of calls in a day. -1 is unlimited"
        ),
        "max_calls_in_week": IntOption(
            default = -1,
            min_val = -1,
            max_val = 604800,
            description ="Limit the maximum number of calls in a week. -1 is unlimited"
        ),
        "max_calls_in_month": IntOption(
            default = -1,
            min_val = -1,
            max_val = 18748800,
            description ="Limit the maximum number of calls in a month. -1 is unlimited"
        ),
        "api_key": TextOption(
            default = "",
            description ="Fill in the API Key",
        ),
        "http_proxy": TextOption(
            default = "",
            description ="If an http_proxy is required to access the service",
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
