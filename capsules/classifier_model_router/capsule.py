from vcap import BaseCapsule, NodeDescription, IntOption, TextOption, EnumOption, DeviceMapper
from .backend import Backend
from .stream_state import StreamState


class Capsule(BaseCapsule):
    name = "classifier_model_router"
    description = "v0.8. Routes to Claude, ChatGPT, Gemini, Qwen, GLM, DeepSeek, or any OpenAI-compatible model (also a local one). " \
                  "The capsule takes the detections of other capsules (the classes named in the input_classes option: person by default), cuts each one out of the frame and asks the model " \
                  "selected by the provider option (a drop-down list; claude is the default) about the picture with a prompt (for example: is the person " \
                  "wearing a hard hat?). The answer is added to the extra data of that person under the key model_router, for the " \
                  "BrainFrame API and for other capsules to use. Fill in the API key of the selected provider. To control the cost, at most " \
                  "max_detections detections are sent every api_call_interval seconds, and max_api_calls limits the number of calls."
    version = 1
    stream_state = StreamState
    device_mapper = DeviceMapper.map_to_single_cpu()
    # Detections of any class reach the capsule (an empty list of classes accepts all); it picks the ones named in the
    # input_classes option (person by default) itself
    input_type = NodeDescription(
        size=NodeDescription.Size.ALL
    )
    output_type = NodeDescription(
        size=NodeDescription.Size.ALL,
        extra_data=["model_router"]
    )
    backend_loader = lambda capsule_files, device: Backend()
    options = {
        "max_tokens": IntOption(
            default = 1024,
            min_val = 1,
            max_val = 8192,
            description ="The maximum number of tokens the model may generate. It also covers the model's thinking, so keep it above a few hundred."
        ),
        "max_api_calls": IntOption(
            default=10,
            min_val=0,
            max_val=8192,
            description="The number of calls that are still allowed (one per detection sent to the model). Each call lowers it; at 0 nothing is sent. "
                        "It is kept in memory only: it is back to the configured value after the options are changed or BrainFrame restarts."
        ),
        "api_call_interval": IntOption(
            default = 60,
            min_val = 0,
            max_val = 86400,
            description ="The time between two rounds of calls to the model, in seconds. Nothing new is sent while an earlier request has no answer yet."
        ),
        "provider": EnumOption(
            default = "claude",
            choices = ["claude", "chatgpt", "gemini", "qwen", "glm", "deepseek", "custom"],
            description ="Which model to ask: claude (Anthropic API), chatgpt (OpenAI API), gemini (Google API), qwen (Alibaba), glm (Zhipu / Z.ai), "
                         "deepseek, or custom (any OpenAI-compatible server: fill in base_url and custom_model). The model and the api_key of that provider are used.",
        ),
        "claude_model": EnumOption(
            default = "claude-haiku-5-5",
            choices = ["claude-haiku-5-5", "claude-sonnet-5-5", "claude-opus-5-5", "claude-fable-5-2"],
            description ="[show_when provider=claude] The model used when the provider is claude. The default is the cheapest current Claude model (about US$0.10 per million input tokens); "
                         "choose claude-sonnet-5-5 or claude-opus-5-5 for harder questions. A model that is not in the list: type it in model_override.",
        ),
        "chatgpt_model": EnumOption(
            default = "gpt-5-nano",
            choices = ["gpt-5-nano", "gpt-5-mini", "gpt-5"],
            description ="[show_when provider=chatgpt] The model used when the provider is chatgpt. The default is the cheapest current OpenAI model (about US$0.05 per million input tokens). "
                         "A model that is not in the list: type it in model_override.",
        ),
        "gemini_model": EnumOption(
            default = "gemini-3.1-flash-lite",
            choices = ["gemini-3.1-flash-lite", "gemini-2.5-flash", "gemini-2.5-pro"],
            description ="[show_when provider=gemini] The model used when the provider is gemini. The default is the cheapest current Gemini 3 model; the Google AI Studio free tier (with a daily limit) may cover it, which makes it good for testing. "
                         "A model that is not in the list: type it in model_override.",
        ),
        "qwen_model": EnumOption(
            default = "qwen3-vl-flash",
            choices = ["qwen3-vl-flash", "qwen3-vl-plus"],
            description ="[show_when provider=qwen] The model used when the provider is qwen (Alibaba Model Studio). It must be a vision model (qwen3-vl-...). A model that is not in the list: type it in model_override.",
        ),
        "glm_model": EnumOption(
            default = "glm-4.6v-flash",
            choices = ["glm-4.6v-flash", "glm-4.6v"],
            description ="[show_when provider=glm] The model used when the provider is glm (Zhipu / Z.ai). The default is a vision model that is reported to be free, which makes it good for testing. "
                         "A model that is not in the list: type it in model_override.",
        ),
        "deepseek_model": EnumOption(
            default = "deepseek-v4-flash-vision-exp",
            choices = ["deepseek-v4-flash-vision-exp"],
            description ="[show_when provider=deepseek] The model used when the provider is deepseek. It must be a model that accepts images; check the current name in the DeepSeek documentation and type it in model_override.",
        ),
        "custom_model": TextOption(
            default = "",
            description ="[show_when provider=custom] [single_line] The model used when the provider is custom (an OpenAI-compatible server): for example a model name of vLLM, Ollama, Moonshot Kimi or Doubao. There is no default.",
        ),
        "model_override": TextOption(
            default = "",
            description ="[show_when provider=custom] [single_line] A model name to use instead of the one chosen in the drop-down list of the selected provider (for example a newer model, or a model of a custom server). "
                         "Empty = use the drop-down list. It is sent to the selected provider as it is.",
        ),
        "input_classes": TextOption(
            default = "person",
            description ="[single_line] The class names of the detections to send to the model, separated by commas, for example: person or person, car. "
                         "Only detections of these classes are cut out and asked about, and get the answer in their extra data. Write the prompt for them too (the default prompt is about a person). "
                         "Empty = detections of every class (the number sent is still limited by max_detections and max_api_calls).",
        ),
        "max_detections": IntOption(
            default=2,
            min_val=0,
            max_val=10,
            description="The maximum number of detections that can be sent to the model per frame, the biggest first."
        ),
        "min_detection_pixels": IntOption(
            default = 900,
            min_val = 1,
            max_val = 10000000,
            description ="Detections whose box has fewer pixels than this (width x height) are not sent to the model."
        ),
        "claude_api_key": TextOption(
            default = "",
            description ="[show_when provider=claude] [single_line] Your Anthropic API key (used when the provider is claude). There is no default: it is your own key.",
        ),
        "chatgpt_api_key": TextOption(
            default = "",
            description ="[show_when provider=chatgpt] [single_line] Your OpenAI API key (used when the provider is chatgpt). There is no default: it is your own key.",
        ),
        "gemini_api_key": TextOption(
            default = "",
            description ="[show_when provider=gemini] [single_line] Your Google AI Studio API key (used when the provider is gemini). There is no default: it is your own key.",
        ),
        "qwen_api_key": TextOption(
            default = "",
            description ="[show_when provider=qwen] [single_line] Your Alibaba Model Studio (DashScope) API key (used when the provider is qwen). There is no default: it is your own key.",
        ),
        "glm_api_key": TextOption(
            default = "",
            description ="[show_when provider=glm] [single_line] Your Z.ai / Zhipu API key (used when the provider is glm). There is no default: it is your own key.",
        ),
        "deepseek_api_key": TextOption(
            default = "",
            description ="[show_when provider=deepseek] [single_line] Your DeepSeek API key (used when the provider is deepseek). There is no default: it is your own key.",
        ),
        "custom_api_key": TextOption(
            default = "",
            description ="[show_when provider=custom] [single_line] The API key of the custom server (used when the provider is custom). Leave it empty if the server needs none (for example a local one).",
        ),
        "base_url": TextOption(
            default = "",
            description ="[show_when provider=chatgpt|qwen|glm|deepseek|custom] [single_line] The endpoint of the chat completions API, for the providers with an OpenAI-compatible API (chatgpt, qwen, glm, deepseek, custom). "
                         "Empty = the standard endpoint of the provider (required for custom), for example https://api.openai.com/v1/chat/completions. "
                         "Use it for the China mainland endpoint of qwen or glm, a gateway, or a local server (http://host:11434/v1/chat/completions).",
        ),
        "http_proxy": TextOption(
            default = "",
            description ="[single_line] If an http_proxy is required to access the service",
        ),
        "claude_effort": EnumOption(
            default = "low",
            choices = ["off", "low", "medium", "high", "xhigh", "max"],
            description ="[show_when provider=claude] How much the model thinks before it answers. 'low' is enough for a True/False question; more thinking costs more. "
                         "off = send no setting (for a model that has none, for example Claude Haiku 4.5).",
        ),
        "chatgpt_effort": EnumOption(
            default = "low",
            choices = ["off", "minimal", "low", "medium", "high"],
            description ="[show_when provider=chatgpt] How much the model thinks before it answers (GPT-5 and o-series models). 'minimal' is the cheapest; 'low' is enough for a True/False question. "
                         "off = send no setting (for a model that has none, for example gpt-4o).",
        ),
        "gemini_effort": EnumOption(
            default = "low",
            choices = ["off", "minimal", "low", "medium", "high"],
            description ="[show_when provider=gemini] How much the model thinks before it answers (the thinking level of Gemini 3 models; older Gemini models ignore it). "
                         "'minimal' is the cheapest; 'low' is enough for a True/False question. off = send no setting.",
        ),
        "image_detail": TextOption(
            default = "low",
            description ="[show_when provider=chatgpt] chatgpt only: how closely the model looks at the picture: low (fixed small cost, enough for a yes/no question "
                         "about a person), high or auto.",
        ),
        "prompt": TextOption(
            default = "How many people in the picture are holding a phone? Reply with an Arabic numeral only, for example: 0. Do not add anything else.",
            description = "The prompt sent to the model together with the picture of one person (a crop of the frame). Ask for a short answer in a fixed format. Examples (copy one): 1) How many people in the picture are holding a phone? Reply with an Arabic numeral only, for example: 0. Do not add anything else. 2) This is a cropped photo of one person. Is the person wearing a hard hat (safety helmet)? A cap or hood does not count. Answer with exactly one word: Yes, No, or Unknown (use Unknown if the head is not visible or the picture is too unclear to tell). Do not write anything else. 3) This is a cropped photo of one person. Is the person wearing a high-visibility vest? Answer with exactly one word: Yes, No, or Unknown. Do not write anything else. 4) This is a cropped photo of one person. Which best describes the posture: standing, sitting, walking, running, lying down, or unknown? Answer with exactly one of these words. Do not write anything else.",
        )
    }
