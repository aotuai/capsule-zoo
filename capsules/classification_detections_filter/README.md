# classification_detections_filter 胶囊简介

## 概述

`classification_detections_filter`胶囊对分类胶囊检测到的结果进行过滤，满足过滤条件的detection 将输出filter_confirmed。

## 胶囊支持的参数配置

1. class_name: 需要过滤的class_name， string type, e.g.: "person"
2. attributes_category: 需要过滤的attributes属性里的key， string type, e.g.: "behavior"
3. attributes_values: 需要过滤的attributes属性里的values， list json format string, e.g.: '["drinking", "phoning", "smoking"]'
4. time_window_duration: 过滤器的时间窗，单位: 秒
5. true_counter: 在时间窗口内，需要累计检测出 value 为 true 的最大值

## 胶囊过滤方式
1. 时间窗口起点设置为当前时间，每个value的counter 清零
2. 在给定的时间窗内，指定value 为true 的检测数达到设定的 true_counter 时，输出filter_confirmed 检测结果，返回1；
3. 当前时间窗结束时，若所有values为true的检测数都未达到 true_counter，返回1；

## 胶囊输出结果
1. 时间窗口内未达到设定检测数时，输出[]
2. 时间窗口内达到设定检测数时，输出[detections]
   "class_name":"filter_confirmed"
   "extra_data":{"filter_confirmed":{attributes_category:attributes_value}}

3. [detections] 举例:
   [{"class_name":"filter_confirmed",
    "coords":[[371,135],[2252,135],[2252,1383],[371,1383]],
    "children":[],"attributes":{},
    "extra_data":{"filter_confirmed":{"behavior":"drinking"}},
    "track_id":null,"with_identity":null}]
