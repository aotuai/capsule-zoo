# classification_detections_filter capsule

## Overview

The `classification_detections_filter` capsule filters the results detected by a classifier capsule. A detection that meets the filter conditions is output as filter_confirmed.

## Options

1. class_name: the class_name to filter, string type, e.g.: "person"
2. attributes_category: the key in the attributes to filter, string type, e.g.: "behavior"
3. attributes_values: the values in the attributes to filter, list json format string, e.g.: '["drinking", "phoning", "smoking"]'
4. time_window_duration: the time window of the filter, in seconds
5. true_counter: the number of detections whose value is true that must add up within the time window

## How the capsule filters
1. The start of the time window is set to the current time, and the counter of every value is reset to zero
2. Within the given time window, when the number of detections whose value is true reaches the set true_counter, the filter_confirmed detection is output and 1 is returned;
3. When the current time window ends, if the number of detections whose value is true has not reached true_counter for any of the values, 1 is returned;

## Output
1. When the set number of detections is not reached within the time window, the output is []
2. When the set number of detections is reached within the time window, the output is [detections]
   "class_name":"filter_confirmed"
   "extra_data":{"filter_confirmed":{attributes_category:attributes_value}}

3. Example of [detections]:
   [{"class_name":"filter_confirmed",
    "coords":[[371,135],[2252,135],[2252,1383],[371,1383]],
    "children":[],"attributes":{},
    "extra_data":{"filter_confirmed":{"behavior":"drinking"}},
    "track_id":null,"with_identity":null}]
