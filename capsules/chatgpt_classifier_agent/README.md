# chatgpt_classifier_agent capsule

## Overview

The `chatgpt_classifier_agent` capsule takes the people detected by another capsule, cuts each person out of the frame and sends the crop with a prompt to an OpenAI multimodal model (by default `gpt-5-nano`, the cheapest OpenAI model, about US$0.05 per million input tokens). The text the model returns (for example `True` or `1`) is stored in the `chatgpt` field of the `extra_data` output in `BrainFrame`, on a box around that person. The answer is currently limited to 512 characters, so write the prompt carefully to keep the answer from getting too long.

## Deployment steps

In the options, fill in the API key, the proxy server used for the request (only HTTP proxies are supported at the moment, not SOCKS5), and the prompt used to analyze the picture.

    ```text
    API KEY: "sk-dK54******************1XG"
    HTTP PROXY: "http://192.168.31.19:1081"
    PROMPT: "How many people in this picture wear a yellow safety helmet? Reply with an Arabic numeral, for example: 0. Do not add punctuation or anything else to the reply."
    ```

## Viewing the analysis output

1. Enter the database container: `docker exec -it brainframe-database-1 /bin/bash`
1. Log in to the database command line: `psql -U user -d brainframe`
1. Query the latest rows of the detection table: `select * from detection where class_name = 'chatgpt' order by id desc;` For example:

    ```txt
    brainframe=# select * from detection where class_name = 'chatgpt' order by id desc;
        id    | parent_id | class_name | identity_id | extra_data_json |       coords_json        | track_id
    ----------+-----------+------------+-------------+-----------------+--------------------------+----------
    44782433 |           | chatgpt    |             | {"chatgpt":"1"}  | [[100,100],[300,100],[300,500],[100,500]] |
    44782432 |           | chatgpt    |             | {"chatgpt":"2"}  | [[620,90],[800,90],[800,480],[620,480]]   |
    ```

## Options

For the model call parameters, see the OpenAI documentation: <https://platform.openai.com/docs/api-reference/chat/create>

- `model`: defaults to `gpt-5-nano` (the cheapest). For harder questions, use a larger model.
- `reasoning_effort`: how much a GPT-5 or o-series model thinks before it answers: `minimal` (the default, cheapest, and enough for a True/False answer), `low`, `medium` or `high`. Leave it empty to use the model's own default.
- `max_tokens`: defaults to 1024. For a model that thinks it also covers the thinking, so a value that is too small can leave no text in the answer sent as `max_completion_tokens`.
- `temperature`: only used with models that do not think before they answer (for example `gpt-4o`). The GPT-5 family and the o-series reject it, so it is not sent to them.
- `image_detail`: `low` (the default: a small fixed cost, enough for a yes/no question about a person), `high` or `auto`. With `low`, a crop larger than 512 pixels is shrunk before it is sent.
- `min_person_pixels`: people whose box has fewer pixels than this (width x height, default 900) are not sent.
- `max_detections_per_frame`: how many people (the biggest first) are sent for one frame, default 2.
- `current_count`: the number of calls that are still allowed, one per person sent. Each call lowers it, and at 0 nothing is sent. It is kept in memory only, so it is back to the configured value after the options are changed or BrainFrame restarts.

Calling a large model is expensive, so you can set the interval between calls with `detection_interval`, in seconds (the default is 60). The capsule sends nothing new while an earlier request has no answer yet. When a call fails (wrong key, network error, no text in the reply) it waits at least 10 seconds before it tries again; with an empty `api_key` it makes no call. A request times out after 60 seconds. Because the capsule only analyzes one frame per interval, you may not see a detection box in the BrainFrame client for long, but the output is in the database. You can also see the model's results with the command line tool `brainframe-apps get_zone_statuses`.

## Open issues

1. Need support to take detection name as an option
1. Need support to setup a limit for cost control: Max number of calls in a minute/hour/day/week/month. 0 is for no limitation
1. Need support to save number of calls in a database, so the cost control continuous after a brainframe restart
