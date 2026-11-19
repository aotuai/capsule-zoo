# classifier_model_router capsule

## Overview

The `classifier_model_router` capsule takes the detections of another capsule (people by default, see `input_classes`), cuts each one out of the frame and sends the crop with a prompt to a cloud AI model. The text the model returns (for example `True` or `1`) is stored in the `classification` field of the `extra_data` output in `BrainFrame`, on a box around that person.

The `provider` option selects which model is asked:

| `provider` | API | Model option (default) | API key option |
| --- | --- | --- | --- |
| `claude` (default) | Anthropic | `claude_model` = `claude-haiku-5-5`, the cheapest current Claude model | `claude_api_key` |
| `chatgpt` | OpenAI | `chatgpt_model` = `gpt-5-nano`, the cheapest current OpenAI model | `chatgpt_api_key` |
| `gemini` | Google (`generateContent`) | `gemini_model` = `gemini-3.1-flash-lite`; the Google AI Studio free tier (with daily limits) may cover it | `gemini_api_key` |
| `qwen` | Alibaba Model Studio (OpenAI-compatible) | `qwen_model` = `qwen3-vl-flash` | `qwen_api_key` |
| `glm` | Zhipu / Z.ai (OpenAI-compatible) | `glm_model` = `glm-4.6v-flash`, reported to be free: good for testing | `glm_api_key` |
| `deepseek` | DeepSeek (OpenAI-compatible) | `deepseek_model` = `deepseek-v4-flash-vision-exp` (check the current vision model name) | `deepseek_api_key` |
| `custom` | any OpenAI-compatible server | `custom_model` (no default) and `base_url` (required) | `custom_api_key` (may stay empty, for example for a local server) |

**Status:** `claude` and `chatgpt` were tried with live keys. `gemini`, `qwen`, `glm`, `deepseek` and `custom` are templates built from the providers' documentation: the request and reply formats are unit-tested, but they have not been tried with a live key yet. Prices and model names change often; check the provider's page.
Use `custom` for Moonshot Kimi, Doubao, Perplexity, or a local server (vLLM, Ollama): fill in `base_url` (the chat completions endpoint), `custom_model` and, if needed, `custom_api_key`. The `base_url` option also replaces the endpoint of `chatgpt`, `qwen`, `glm` and `deepseek`, for example for the China mainland endpoint (`https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions` for qwen, `https://open.bigmodel.cn/api/paas/v4/chat/completions` for glm).
The pictures of the people are sent to the servers of the selected provider: check that it fits your privacy rules.

**Drop-down lists need a BrainFrame web client that supports them** (the capsule options of type enum); an older client does not show these options. It is the same capsule for all of them, and each provider has its own model and key option, so you can switch the `provider` option without typing anything again or loading another capsule. The capsule asks one provider at a time; it does not choose between them by itself. If you only use one provider, `claude_classifier_agent` and `chatgpt_classifier_agent` do the same for one provider each.

## Deployment steps

In the options, choose the provider, fill in the API key of that provider (`claude_api_key`, `chatgpt_api_key`, `gemini_api_key`...; they are empty because they are your own keys), and, if you need one, the proxy server used for the request (only HTTP proxies are supported at the moment, not SOCKS5), and the prompt used to analyze the picture.

    ```text
    PROVIDER: "claude"
    CLAUDE API KEY: "sk-an**************AA"
    HTTP PROXY: "http://192.168.31.19:1081"
    PROMPT: "How many people in the picture are holding a phone? Reply with an Arabic numeral only, for example: 0. Do not add anything else."
    ```

## Viewing the analysis output

The answer is added to the `extra_data` of the person it describes, as `model_router`: the capsule has no detections of its own, and another capsule can read the key `model_router` from the person.
In the BrainFrame web client, switch on **Extra Data** in the stream's display options, and the person's label shows `model_router: <answer>`.
It stays on the person (also when the person moves a little) until the next answer arrives or `api_call_interval` seconds have passed.
A person without a label was not asked (yet), or the answer has expired: that is *unknown*, not 0. Only people who are big enough (`min_detection_pixels`), among the biggest `max_detections`, are sent, and only once per `api_call_interval`; a `0` on a person is the model's own answer.
The answer is also in the `extra_data_json` of the person in the detection table and in the BrainFrame API, for example:

    ```txt
    class_name | extra_data_json
    -----------+----------------------------------------
    person     | {"confidence": 0.86, "model_router": "1"}
    ```

## Options

- `provider`: a drop-down list: `claude` (default), `chatgpt`, `gemini`, `qwen`, `glm`, `deepseek` or `custom`.
- `<provider>_model`: a drop-down list of the models of each provider (the default is the cheapest one, see the table; `custom_model` is a text box without a default). Only the list of the selected provider is used. For harder questions, choose a larger model, for example `claude-sonnet-5-5` or `gpt-5-mini`.
- `model_override`: a model name that is not in the lists (a newer model, a model of a custom server). When it is filled in, it is used instead of the drop-down list of the selected provider. Empty = use the list. It is only shown (and used) when the provider is `custom`; for the other providers choose the model in the list.
- `<provider>_api_key`: your own keys, so they have no default. Only the key of the selected provider is used. `http_proxy` is empty unless you need a proxy.
- `claude_effort`, `chatgpt_effort`, `gemini_effort`: how much the model thinks before it answers, a drop-down list per provider (shown only for that provider). `claude`: `low`, `medium`, `high`, `xhigh` or `max`. `chatgpt` (GPT-5 and o-series models) and `gemini` (Gemini 3 models): `minimal` (cheapest), `low`, `medium` or `high`. `off` sends no setting, for a model that has none (for example Claude Haiku 4.5 or gpt-4o). The other providers ignore it (`qwen` and `glm` are asked not to think). The default `low` is enough for a True/False answer.
- `max_tokens`: defaults to 1024. It also covers the model's thinking, so a value that is too small can leave no text in the answer.
- `base_url`: the endpoint of the OpenAI-compatible providers; empty = the provider's standard one (required for `custom`).
- `image_detail`: `chatgpt` only: `low` (the default: a small fixed cost, enough for a yes/no question about a person), `high` or `auto`. A crop larger than 512 pixels is shrunk before it is sent (1024 for `high` and `auto`).
- `input_classes`: the class names of the detections to send, separated by commas (default `person`; for example `person, car`). Detections of other classes are left alone. Write the prompt for the classes you choose: the default prompt is about a person. Empty = every class (the number sent is still limited by `max_detections` and `max_api_calls`).
- `min_detection_pixels`: detections whose box has fewer pixels than this (width x height, default 900) are not sent.
- `max_detections`: how many detections (the biggest first) are sent for one frame, default 2.
- `max_api_calls`: the number of calls that are still allowed, one per detection sent. Each call lowers it, and at 0 nothing is sent. It is kept in memory only, so it is back to the configured value after the options are changed or BrainFrame restarts.
- The current models only accept their default sampling parameters, so there is no `temperature` option.

Calling a large model is expensive, so you can set the interval between calls with `api_call_interval`, in seconds (the default is 60). The capsule sends nothing new while an earlier request has no answer yet. When a call fails (wrong key, network error, no text in the reply) it waits at least 10 seconds before it tries again; with an empty key for the selected provider it makes no call. A request times out after 60 seconds. Because the capsule only analyzes one frame per interval, you may not see a detection box in the BrainFrame client for long, but the output is in the database. You can also see the model's results with the command line tool `brainframe-apps get_zone_statuses`.

## Example prompts

The default is the first one. They are also listed in the description of the `prompt` option, so you can copy them from the options dialog. The picture is a crop of one person (the answer for a count is then 0 or 1, so the one-word examples fit better).

1. `How many people in the picture are holding a phone? Reply with an Arabic numeral only, for example: 0. Do not add anything else.`
2. `This is a cropped photo of one person. Is the person wearing a hard hat (safety helmet)? A cap or hood does not count. Answer with exactly one word: Yes, No, or Unknown (use Unknown if the head is not visible or the picture is too unclear to tell). Do not write anything else.`
3. `This is a cropped photo of one person. Is the person wearing a high-visibility vest? Answer with exactly one word: Yes, No, or Unknown. Do not write anything else.`
4. `This is a cropped photo of one person. Which best describes the posture: standing, sitting, walking, running, lying down, or unknown? Answer with exactly one of these words. Do not write anything else.`

Ask for a short answer in a fixed format: it is cheaper, and easy to use in alarms and rules. Allow "Unknown" when the answer cannot be seen (for example a head that is cut off), so a guess does not become an alarm.

## Open issues

1. Need support to setup a limit for cost control: Max number of calls in a minute/hour/day/week/month. 0 is for no limitation
1. Need support to save number of calls in a database, so the cost control continuous after a brainframe restart
1. Ideas for real routing: fall back to the other provider when a call fails, or choose the provider by prompt or by person
