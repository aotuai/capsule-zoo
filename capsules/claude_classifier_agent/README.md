# claude_classifier_agent胶囊简介

## 概述

`claude_classifier_agent`胶囊连接到`claude-3-5-sonnet`多模态大模型，直接基于提示词对画面内容进行分析，将模型返回的内容存放在`BrainFrame`的`extra_data`输出中，目前最长不超过512个字符，注意精心构造提示词，防止内容超长。

## 部署步骤参考

在Option中填入APIKey；发起请求时使用的代理服务器（目前仅支持HTTP代理，不支持SOCK5代理）；针对图像内容进行分析的提示词。

    ```text
    API KEY: "sk-an**************AA"
    HTTP PROXY: "http://192.168.31.19:1081"
    PROMPT: "这张图中戴黄色安全帽的人有几个？请使用阿拉伯数字回复，例如：0。回复内容中请勿添加标点符号等任何其他内容。"
    ```

## 查看分析输出结果示例

1. 执行数据库容器中的命令查询detection表内最新数据：`docker exec -it brainframe_database_1 psql -U user -d brainframe -c "select * from detection order by id desc;"`例如：

    ```txt
    id    | parent_id | class_name | identity_id |                                                                                                                        extra_data_json                                                                                                                         |              coords_json              | track_id
    ----------+-----------+------------+-------------+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+---------------------------------------+----------
    44782514 |           | claude     |             | {"claude":{"id":"msg_01QFWnXNyxoMGR2u4aMNgXxn","type":"message","role":"assistant","model":"claude-3-5-sonnet-20240620","content":[{"type":"text","text":"1"}],"stop_reason":"end_turn","stop_sequence":null,"usage":{"input_tokens":1631,"output_tokens":5}}} | [[0,0],[1920,0],[1920,1080],[0,1080]] |
    44782513 |           | claude     |             | {"claude":{"id":"msg_01GpoPsNpQ988h4ThWDpfQkp","type":"message","role":"assistant","model":"claude-3-5-sonnet-20240620","content":[{"type":"text","text":"0"}],"stop_reason":"end_turn","stop_sequence":null,"usage":{"input_tokens":1631,"output_tokens":5}}} | [[0,0],[1920,0],[1920,1080],[0,1080]] |
    ```

## 胶囊支持的参数配置

目前胶囊支持针对'model', `temperature`和`max_tokens`进行配置，具体说明参考Claude官方说明：<https://docs.anthropic.com/en/api/messages>

## 遗留问题

1. Need support to take detection name as an option
1. Need support to setup a limit for cost control: Max number of calls in a minute/hour/day/week/month. 0 is for no limitation
1. Need support to save number of calls in a database, so the cost control continuous after a brainframe restart
