from vcap import BaseCapsule, NodeDescription, IntOption, TextOption, DeviceMapper
from .backend import Backend
from .stream_state import StreamState


class Capsule(BaseCapsule):
    name = "classifier_model_router"
    description = "v0.7. Routes to Claude, ChatGPT, Gemini, Qwen, GLM, DeepSeek, or any OpenAI-compatible model (also a local one). " \
                  "The capsule takes the people detected by other capsules, cuts each person out of the frame and asks the model " \
                  "selected by the provider option (claude is the default) about the picture with a prompt (for example: is the person " \
                  "wearing a hard hat?). The answer is added to the extra data of that person under the key model_router, for the " \
                  "BrainFrame API and for other capsules to use. Fill in the API key of the selected provider. To control the cost, at most " \
                  "max_detections_per_frame people are sent every detection_interval seconds, and current_count limits the number of calls."
    version = 1
    stream_state = StreamState
    device_mapper = DeviceMapper.map_to_single_cpu()
    input_type = NodeDescription(
        size=NodeDescription.Size.ALL,
        detections=["person"]
    )
    output_type = NodeDescription(
        size=NodeDescription.Size.ALL,
        detections=["person"],
        extra_data=["model_router"]
    )
    backend_loader = lambda capsule_files, device: Backend()
    options = {
        "provider": TextOption(
            default = "claude",
            description ="Which model to ask: claude (Anthropic API), chatgpt (OpenAI API), gemini (Google API), qwen (Alibaba), glm (Zhipu / Z.ai), "
                         "deepseek, or custom (any OpenAI-compatible server: fill in base_url and custom_model). The model and the api_key of that provider are used.",
        ),
        "claude_model": TextOption(
            default = "claude-haiku-5-5",
            description ="The model used when the provider is claude. The default is the cheapest current Claude model (about US$0.10 per million input tokens); "
                         "use claude-sonnet-5-5 or claude-opus-5-5 for harder questions.",
        ),
        "chatgpt_model": TextOption(
            default = "gpt-5-nano",
            description ="The model used when the provider is chatgpt. The default is the cheapest current OpenAI model (about US$0.05 per million input tokens).",
        ),
        "gemini_model": TextOption(
            default = "gemini-3.1-flash-lite",
            description ="The model used when the provider is gemini. The default is the cheapest current Gemini 3 model; the Google AI Studio free tier (with a daily limit) may cover it, which makes it good for testing.",
        ),
        "qwen_model": TextOption(
            default = "qwen3-vl-flash",
            description ="The model used when the provider is qwen (Alibaba Model Studio). It must be a vision model (qwen3-vl-...).",
        ),
        "glm_model": TextOption(
            default = "glm-4.6v-flash",
            description ="The model used when the provider is glm (Zhipu / Z.ai). The default is a vision model that is reported to be free, which makes it good for testing.",
        ),
        "deepseek_model": TextOption(
            default = "deepseek-v4-flash-vision-exp",
            description ="The model used when the provider is deepseek. It must be a model that accepts images; check the current name in the DeepSeek documentation.",
        ),
        "custom_model": TextOption(
            default = "",
            description ="The model used when the provider is custom (an OpenAI-compatible server): for example a model name of vLLM, Ollama, Moonshot Kimi or Doubao. There is no default.",
        ),
        "base_url": TextOption(
            default = "",
            description ="The endpoint of the chat completions API, for the providers with an OpenAI-compatible API (chatgpt, qwen, glm, deepseek, custom). "
                         "Empty = the standard endpoint of the provider (required for custom), for example https://api.openai.com/v1/chat/completions. "
                         "Use it for the China mainland endpoint of qwen or glm, a gateway, or a local server (http://host:11434/v1/chat/completions).",
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
        "claude_api_key": TextOption(
            default = "",
            description ="Your Anthropic API key (used when the provider is claude). There is no default: it is your own key.",
        ),
        "chatgpt_api_key": TextOption(
            default = "",
            description ="Your OpenAI API key (used when the provider is chatgpt). There is no default: it is your own key.",
        ),
        "gemini_api_key": TextOption(
            default = "",
            description ="Your Google AI Studio API key (used when the provider is gemini). There is no default: it is your own key.",
        ),
        "qwen_api_key": TextOption(
            default = "",
            description ="Your Alibaba Model Studio (DashScope) API key (used when the provider is qwen). There is no default: it is your own key.",
        ),
        "glm_api_key": TextOption(
            default = "",
            description ="Your Z.ai / Zhipu API key (used when the provider is glm). There is no default: it is your own key.",
        ),
        "deepseek_api_key": TextOption(
            default = "",
            description ="Your DeepSeek API key (used when the provider is deepseek). There is no default: it is your own key.",
        ),
        "custom_api_key": TextOption(
            default = "",
            description ="The API key of the custom server (used when the provider is custom). Leave it empty if the server needs none (for example a local one).",
        ),
        "http_proxy": TextOption(
            default = "",
            description ="If an http_proxy is required to access the service",
        ),
        "effort": TextOption(
            default = "low",
            description ="How much the model thinks before it answers. claude: low, medium, high, xhigh or max. chatgpt (GPT-5 and o-series models): "
                         "minimal, low, medium or high; 'minimal' is the cheapest. gemini (Gemini 3 models): minimal, low, medium or high. The other providers ignore it "
                         "(qwen and glm are asked not to think). 'low' is enough for a True/False question. "
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
