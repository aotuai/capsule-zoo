# claude_classifier_agent capsule

## Overview

The `claude_classifier_agent` capsule connects to a Claude multimodal model (by default `claude-haiku-5-5`, the cheapest current model, about US$0.10 per million input tokens). It analyzes the picture directly from a prompt and stores the text the model returns (for example `True` or `1`) in the `claude` field of the `extra_data` output in `BrainFrame`. The answer is currently limited to 512 characters, so write the prompt carefully to keep the answer from getting too long.

## Deployment steps

In the options, fill in the API key, the proxy server used for the request (only HTTP proxies are supported at the moment, not SOCKS5), and the prompt used to analyze the picture.

    ```text
    API KEY: "sk-an**************AA"
    HTTP PROXY: "http://192.168.31.19:1081"
    PROMPT: "How many people in the picture are holding a phone? Reply with an Arabic numeral only, for example: 0. Do not add anything else."
    ```

## Viewing the analysis output

1. Run this command in the database container to query the latest rows of the detection table: `docker exec -it brainframe-database-1 psql -U user -d brainframe -c "select * from detection where class_name = 'claude' order by id desc;"` For example:

    ```txt
    id    | parent_id | class_name | identity_id |  extra_data_json  |              coords_json              | track_id
    ----------+-----------+------------+-------------+-------------------+---------------------------------------+----------
    44782514 |           | claude     |             | {"claude":"1"}    | [[0,0],[1920,0],[1920,1080],[0,1080]] |
    44782513 |           | claude     |             | {"claude":"0"}    | [[0,0],[1920,0],[1920,1080],[0,1080]] |
    ```

## Options

The capsule currently lets you configure `model`, `effort` and `max_tokens`. For details, see the Claude documentation: <https://docs.anthropic.com/en/api/messages>

- `model`: defaults to `claude-haiku-5-5` (the cheapest). For harder questions, use `claude-sonnet-5-5` or `claude-opus-5-5`.
- `effort`: how much the model thinks before it answers. The default is `low` (cheapest, and enough for a True/False answer); the other choices are `medium`, `high`, `xhigh` and `max`. Leave it empty when you use a model that has no such setting (for example Claude Haiku 4.5).
- `max_tokens`: defaults to 1024. It also covers the model's thinking, so a value that is too small can leave no text in the answer.
- The current models only accept their default sampling parameters, so there is no `temperature` option any more.

When a call fails (wrong key, network error, no text in the reply), the capsule waits at least 10 seconds before it tries again, so it does not call the API on every frame. With an empty `api_key` it makes no call. A request times out after 60 seconds, and a picture larger than 1568 pixels is shrunk before it is sent.

Calling a large model is expensive, so you can set the interval between calls with `detection_interval`, in seconds (the default is 60). Because the capsule only analyzes one frame per interval, you may not see a detection box in the BrainFrame client, but the output is in the database. You can also see the model's results with the command line tool `brainframe-apps get_zone_statuses`.

## Example prompts

The default is the first one. They are also listed in the description of the `prompt` option, so you can copy them from the options dialog. The picture is the whole frame, so these count people.

1. `How many people in the picture are holding a phone? Reply with an Arabic numeral only, for example: 0. Do not add anything else.`
2. `How many people in the picture are wearing a hard hat (safety helmet)? A cap or hood does not count. Reply with an Arabic numeral only, for example: 0. Do not add anything else.`
3. `How many people in the picture are wearing a high-visibility vest? Reply with an Arabic numeral only, for example: 0. Do not add anything else.`
4. `Is anyone in the picture lying on the floor? Answer with exactly one word: Yes, No, or Unknown. Do not write anything else.`

Ask for a short answer in a fixed format: it is cheaper, and easy to use in alarms and rules. Allow "Unknown" when the answer cannot be seen (for example a head that is cut off), so a guess does not become an alarm.

## Open issues

1. Need support to take detection name as an option
1. Need support to setup a limit for cost control: Max number of calls in a minute/hour/day/week/month. 0 is for no limitation
1. Need support to save number of calls in a database, so the cost control continuous after a brainframe restart
