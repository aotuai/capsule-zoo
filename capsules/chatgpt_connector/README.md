# chatgpt_connector胶囊简介

## 概述

`chatgpt_connector`胶囊连接到`gpt-4o`多模态大模型，直接基于提示词对画面内容进行分析，将ChatGPT返回的内容存放在`BrainFrame`的`extra_data`输出中，目前最长不超过512个字符，注意精心构造提示词，防止内容超长。

## 部署步骤参考

1. 新建或修改`/usr/local/share/brainframe/.env`。加入下面的三个环境变量用来指定访问`ChatGPT`的APIKey；发起请求时使用的代理服务器（目前仅支持HTTP代理，不支持SOCK5代理）；针对图像内容进行分析的提示词。

    ```text
    CHATGPT_APIKEY="sk-dK54******************1XG"
    CHATGPT_HTTPPROXY="http://192.168.31.19:1081"
    CHATGPT_PROMPT="这张图中戴黄色安全帽的人有几个？请使用阿拉伯数字回复，例如：0。回复内容中请勿添加标点符号等任何其他内容。"
    ```

1. 修改`/usr/local/share/brainframe/docker-compose.yml`，在`service.core.environment`中加入三个环境变量

    ```text
    CHATGPT_APIKEY: ${CHATGPT_APIKEY}
    CHATGPT_HTTPPROXY: ${CHATGPT_HTTPPROXY}
    CHATGPT_PROMPT: ${CHATGPT_PROMPT}
    ```

1. 重新创建服务容器

    ```bash
    brainframe compose down
    brainframe compose up -d
    ```

1. 复制胶囊到Brainframe中

    ```bash
    mv chatgpt_connector.cap $(brainframe info data_path)/capsules/
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

目前胶囊支持针对`temperature`和`max_tokens`两个大模型调用参数进行配置，具体说明参考OpenAI官方说明：<https://platform.openai.com/docs/api-reference/chat/create>

大模型调用的开销比较大，可以通过`detection_interval`已秒为单位设置调用的时间间隔。该参数默认为0，设置为非0值时，因为只是间隔一定的时间抽一帧检测，在BrainFrame客户端上可能看不到检测框，但数据库中可以看到检测输出。使用命令行工具`brainframe-apps get_zone_statuses`也可以观察到大模型检测输出的检测结果。

## 遗留问题

目前BrainFrame的Option暂不支持文本类型，针对APIKey、代理服务器和提示词的配置暂时放在环境变量中完成，这样操作起来比较麻烦，而且如果需要针对不同的视频配置不同的提示词，后续还需要考虑一个扩展方案。
