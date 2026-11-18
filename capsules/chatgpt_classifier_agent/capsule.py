from vcap import BaseCapsule, NodeDescription, FloatOption, IntOption, TextOption, DeviceMapper
from .backend import Backend
from .stream_state import StreamState

class Capsule(BaseCapsule):
    name = "chatgpt_classifier_agent"
    description = "v0.4 The capsule takes detections from other capsules, and send a prompt request " \
                  "to OpenAI API for classification and publish the answers to BrainFrame API or for other "\
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
        detections=["chatgpt"],
        extra_data=["chatgpt"]
    )
    backend_loader = lambda capsule_files, device: Backend()
    options = {
        "model": TextOption(
            default = "gpt-5-nano",
            description ="The model name provided by the API. The default, GPT-5 nano, is the cheapest OpenAI model "
                         "(about US$0.05 per million input tokens); use a larger model for harder questions.",
        ),
        "prompt": TextOption(
            default = "How many people in the picture are holding a phone? Reply with an Arabic numeral only, for example: 0. Do not add anything else.",
            description = "The prompt sent to the model together with the picture of one person (a crop of the frame). Ask for a short answer in a fixed format. Examples (copy one): 1) How many people in the picture are holding a phone? Reply with an Arabic numeral only, for example: 0. Do not add anything else. 2) This is a cropped photo of one person. Is the person wearing a hard hat (safety helmet)? A cap or hood does not count. Answer with exactly one word: Yes, No, or Unknown (use Unknown if the head is not visible or the picture is too unclear to tell). Do not write anything else. 3) This is a cropped photo of one person. Is the person wearing a high-visibility vest? Answer with exactly one word: Yes, No, or Unknown. Do not write anything else. 4) This is a cropped photo of one person. Which best describes the posture: standing, sitting, walking, running, lying down, or unknown? Answer with exactly one of these words. Do not write anything else.",
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
        "temperature": FloatOption(
            default = 1.0,
            min_val = 0.0,
            max_val = 2.0,
            description ="What sampling temperature to use, between 0 and 2. Only used for models that do not think before they answer "
                         "(for example gpt-4o); the GPT-5 family and the o-series do not take it."
        ),
        "reasoning_effort": TextOption(
            default = "minimal",
            description ="How much a GPT-5 or o-series model thinks before it answers: minimal, low, medium or high. 'minimal' is cheapest and "
                         "enough for a True/False question. Leave it empty to use the model's own default. Not used for other models."
        ),
        "image_detail": TextOption(
            default = "low",
            description ="How closely the model looks at the picture: low (fixed small cost, enough for a yes/no question about a person), "
                         "high or auto."
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
            description ="The maximum number of tokens that can be generated in the chat completion. For a model that thinks it also covers the thinking, so keep it above a few hundred."
        ),
        "max_detections_per_frame": IntOption(
            default=2,
            min_val=0,
            max_val=10,
            description="The maximum number of people that can be detected by chatgpt_classifier_agent per frame."
        ),
        "current_count": IntOption(
            default=10,
            min_val=0,
            max_val=8192,
            description="The number of calls that are still allowed (one per person sent to the model). Each call lowers it; at 0 nothing is sent. It is kept in memory only: it is back to the configured value after the options are changed or BrainFrame restarts."
        )
    }
