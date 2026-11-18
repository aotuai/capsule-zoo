from vcap import BaseCapsule, NodeDescription, IntOption, TextOption, DeviceMapper
from .backend import Backend
from .stream_state import StreamState


class Capsule(BaseCapsule):
    name = "classifier_model_router"
    description = "v0.1. The capsule takes the people detected by other capsules, cuts each one out of the frame and sends " \
                  "it with a prompt to a cloud AI model for classification, and publishes the answers to BrainFrame API " \
                  "or for other capsules to consume and generate fused results. The provider option selects the model: " \
                  "Claude (Anthropic API) or ChatGPT (OpenAI API)."
    version = 1
    stream_state = StreamState
    device_mapper = DeviceMapper.map_to_single_cpu()
    input_type = NodeDescription(
        size=NodeDescription.Size.ALL,
        detections=["person"]
    )
    output_type = NodeDescription(
        size=NodeDescription.Size.ALL,
        detections=["classification"],
        extra_data=["classification"]
    )
    backend_loader = lambda capsule_files, device: Backend()
    options = {
        "provider": TextOption(
            default = "claude",
            description ="Which model to ask: 'claude' (Anthropic API) or 'chatgpt' (OpenAI API). The api_key must be a key of that provider.",
        ),
        "model": TextOption(
            default = "",
            description ="The model name. Leave it empty to use the cheapest current model of the provider: claude-haiku-5-5 for claude "
                         "(about US$0.10 per million input tokens), gpt-5-nano for chatgpt (about US$0.05).",
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
            description ="Fill in the API Key of the selected provider",
        ),
        "http_proxy": TextOption(
            default = "",
            description ="If an http_proxy is required to access the service",
        ),
        "effort": TextOption(
            default = "low",
            description ="How much the model thinks before it answers. claude: low, medium, high, xhigh or max. chatgpt (GPT-5 and o-series models): "
                         "minimal, low, medium or high; 'minimal' is the cheapest. 'low' is enough for a True/False question. "
                         "Leave it empty for a model that has no such setting (for example Claude Haiku 4.5 or gpt-4o).",
        ),
        "image_detail": TextOption(
            default = "low",
            description ="chatgpt only: how closely the model looks at the picture: low (fixed small cost, enough for a yes/no question "
                         "about a person), high or auto.",
        ),
        "min_person_pixels": IntOption(
            default = 900,
            min_val = 1,
            max_val = 10000000,
            description ="People whose box has fewer pixels than this (width x height) are not sent to the model."
        ),
        "max_tokens": IntOption(
            default = 1024,
            min_val = 1,
            max_val = 8192,
            description ="The maximum number of tokens the model may generate. It also covers the model's thinking, so keep it above a few hundred."
        ),
        "max_detections_per_frame": IntOption(
            default=2,
            min_val=0,
            max_val=10,
            description="The maximum number of people that can be sent to the model per frame."
        ),
        "current_count": IntOption(
            default=10,
            min_val=0,
            max_val=8192,
            description="The number of calls that are still allowed (one per person sent to the model). Each call lowers it; at 0 nothing is sent. "
                        "It is kept in memory only: it is back to the configured value after the options are changed or BrainFrame restarts."
        )
    }
