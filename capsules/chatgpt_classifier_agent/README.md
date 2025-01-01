# chatgpt_classifier_agent 胶囊简介

## 概述

`chatgpt_classifier_agent`胶囊连接到`gpt-4o`多模态大模型，直接基于提示词对画面内容进行分析，将ChatGPT返回的内容存放在`BrainFrame`的`extra_data`输出中，目前最长不超过512个字符，注意精心构造提示词，防止内容超长。

## 部署步骤参考

在Option中填入APIKey；发起请求时使用的代理服务器（目前仅支持HTTP代理，不支持SOCK5代理）；针对图像内容进行分析的提示词。

    ```text
    API KEY："sk-dK54******************1XG"
    HTTP PROXY："http://192.168.31.19:1081"
    PROMPT："这张图中戴黄色安全帽的人有几个？请使用阿拉伯数字回复，例如：0。回复内容中请勿添加标点符号等任何其他内容。"
    ```

## 查看分析输出结果示例

1. 进入数据库所在容器：`docker exec -it brainframe_database_1 /bin/bash`
1. 登录数据库命令行：`psql -U user -d brainframe`
1. 查询detection表内最新数据:`select * from detection order by id desc;`例如：

    ```txt
    brainframe=# select * from detection order by id desc;
        id    | parent_id | class_name | identity_id |                      extra_data_json                       |                    coords_json                    | track_id
    ----------+-----------+------------+-------------+------------------------------------------------------------+---------------------------------------------------+----------
    44782433 |           | chatgpt    |             | {"chatgpt":{"id":"chatcmpl-9abTjX2TkMMhwv17NQ0FvGrTKFwwv","object":"chat.completion","created":1718510903,"model":"gpt-4o-2024-05-13","choices":[{"index":0,"message":{"role":"assistant","content":"1"},"logprobs":null,"finish_reason":"stop"}],"usage":{"prompt_tokens":1150,"completion_tokens":1,"total_tokens":1151},"system_fingerprint":"fp_aa87380ac5"}}                                                                                                               | [[0,0],[1920,0],[1920,1080],[0,1080]]             |
    44782432 |           | chatgpt    |             | {"chatgpt":{"id":"chatcmpl-9abTfnufyYTRyUt4m7rZGncERQIPg","object":"chat.completion","created":1718510899,"model":"gpt-4o-2024-05-13","choices":[{"index":0,"message":{"role":"assistant","content":"2"},"logprobs":null,"finish_reason":"stop"}],"usage":{"prompt_tokens":1150,"completion_tokens":1,"total_tokens":1151},"system_fingerprint":"fp_aa87380ac5"}}                                                                                                               | [[0,0],[1920,0],[1920,1080],[0,1080]]             |
    44782431 |           | chatgpt    |             | {"chatgpt":{"id":"chatcmpl-9abTaWEGqbtY4TMUcaBNeFeUAu1gz","object":"chat.completion","created":1718510894,"model":"gpt-4o-2024-05-13","choices":[{"index":0,"message":{"role":"assistant","content":"2"},"logprobs":null,"finish_reason":"stop"}],"usage":{"prompt_tokens":1150,"completion_tokens":1,"total_tokens":1151},"system_fingerprint":"fp_aa87380ac5"}}                                                                                                               | [[0,0],[1920,0],[1920,1080],[0,1080]]             |
    
    ```

## 胶囊支持的参数配置

目前胶囊支持针对`model`, `temperature`和`max_tokens`进行配置，具体说明参考OpenAI官方说明：<https://platform.openai.com/docs/api-reference/chat/create>

## 遗留问题

1. Need support to take detection name as an option
1. Need support to setup a limit for cost control: Max number of calls in a minute/hour/day/week/month. 0 is for no limitation
1. Need support to save number of calls in a database, so the cost control continuous after a brainframe restart
