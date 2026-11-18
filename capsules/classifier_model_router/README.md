# classifier_model_router capsule

## Overview

The `classifier_model_router` capsule takes the people detected by another capsule, cuts each person out of the frame and sends the crop with a prompt to a cloud AI model. The text the model returns (for example `True` or `1`) is stored in the `classification` field of the `extra_data` output in `BrainFrame`, on a box around that person.

The `provider` option selects which model is asked:

| `provider` | API | Default model (cheapest current) | API key |
| --- | --- | --- | --- |
| `claude` (default) | Anthropic | `claude-haiku-5-5`, about US$0.10 per million input tokens | an Anthropic key |
| `chatgpt` | OpenAI | `gpt-5-nano`, about US$0.05 per million input tokens | an OpenAI key |

It is the same capsule for both, so you can switch the provider (and the key) in the options without loading another capsule. The capsule asks one provider at a time; it does not choose between them by itself. If you only use one provider, `claude_classifier_agent` and `chatgpt_classifier_agent` do the same for one provider each.

## Deployment steps

In the options, fill in the provider, the API key of that provider, the proxy server used for the request (only HTTP proxies are supported at the moment, not SOCKS5), and the prompt used to analyze the picture.

    ```text
    PROVIDER: "claude"
    API KEY: "sk-an**************AA"
    HTTP PROXY: "http://192.168.31.19:1081"
    PROMPT: "How many people in this picture wear a yellow safety helmet? Reply with an Arabic numeral, for example: 0. Do not add punctuation or anything else to the reply."
    ```

## Viewing the analysis output

1. Run this command in the database container to query the latest rows of the detection table: `docker exec -it brainframe-database-1 psql -U user -d brainframe -c "select * from detection where class_name = 'classification' order by id desc;"` For example:

    ```txt
    id    | parent_id | class_name     | identity_id |       extra_data_json        |                coords_json                | track_id
    ----------+-----------+----------------+-------------+------------------------------+-------------------------------------------+----------
    44782433 |           | classification |             | {"classification":"1"}       | [[100,100],[300,100],[300,500],[100,500]] |
    44782432 |           | classification |             | {"classification":"2"}       | [[620,90],[800,90],[800,480],[620,480]]   |
    ```

## Options

- `provider`: `claude` (default) or `chatgpt` (`anthropic` and `openai` are accepted as well). With any other value the capsule makes no call and logs why.
- `model`: leave it empty to use the cheapest current model of the provider (table above). For harder questions, use a larger model, for example `claude-sonnet-5-5` or a larger GPT model.
- `effort`: how much the model thinks before it answers. `claude`: `low`, `medium`, `high`, `xhigh` or `max`. `chatgpt` (GPT-5 and o-series models): `minimal` (cheapest), `low`, `medium` or `high`. The default `low` is enough for a True/False answer. Leave it empty for a model that has no such setting (for example Claude Haiku 4.5 or gpt-4o).
- `max_tokens`: defaults to 1024. It also covers the model's thinking, so a value that is too small can leave no text in the answer.
- `image_detail`: `chatgpt` only: `low` (the default: a small fixed cost, enough for a yes/no question about a person), `high` or `auto`. A crop larger than 512 pixels is shrunk before it is sent (1024 for `high` and `auto`).
- `min_person_pixels`: people whose box has fewer pixels than this (width x height, default 900) are not sent.
- `max_detections_per_frame`: how many people (the biggest first) are sent for one frame, default 2.
- `current_count`: the number of calls that are still allowed, one per person sent. Each call lowers it, and at 0 nothing is sent. It is kept in memory only, so it is back to the configured value after the options are changed or BrainFrame restarts.
- The current models only accept their default sampling parameters, so there is no `temperature` option.

Calling a large model is expensive, so you can set the interval between calls with `detection_interval`, in seconds (the default is 60). The capsule sends nothing new while an earlier request has no answer yet. When a call fails (wrong key, network error, no text in the reply) it waits at least 10 seconds before it tries again; with an empty `api_key` it makes no call. A request times out after 60 seconds. Because the capsule only analyzes one frame per interval, you may not see a detection box in the BrainFrame client for long, but the output is in the database. You can also see the model's results with the command line tool `brainframe-apps get_zone_statuses`.

## Open issues

1. Need support to setup a limit for cost control: Max number of calls in a minute/hour/day/week/month. 0 is for no limitation
1. Need support to save number of calls in a database, so the cost control continuous after a brainframe restart
1. Ideas for real routing: fall back to the other provider when a call fails, or choose the provider by prompt or by person
