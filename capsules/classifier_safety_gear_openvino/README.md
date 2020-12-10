# About the Capsule
## Usage
This capsule is for classifying if a person is wearing safety equipment 
(safety vest and safety hat). It should be used in conjunction with a person 
detector. 

This capsule will assign two attributes to a people detections, and the possible
value are as following:
```
"safety_hat": ["with_safety_hat", "without_safety_hat"],
"safety_hat_vest": ["with_safety_vest", "without_safety_vest"]
```

## Model
###  Model File Origin
The pretrained model was part of [Intel IoT Developer Program](https://software.intel.com/content/www/us/en/develop/topics/iot/reference-implementations/safety-gear-detector.html),
and downloaded from [safety-gear-detector-python GitHub repository](https://github.com/intel-iot-devkit/safety-gear-detector-python),
then converted to an OpenVino model using the following command:

```shell
docker run -it \
    --volume $(pwd):/workding_dir \
    --workdir /workding_dir \
    openvino/ubuntu18_dev:2020.3 \
    python3 /opt/intel/openvino/deployment_tools/model_optimizer/mo_caffe.py \
        --input_model /workding_dir/resources/worker-safety-mobilenet/worker_safety_mobilenet.caffemodel \
        -o /workding_dir/resources/worker-safety-mobilenet/FP32/worker_safety_mobilenet.caffemodel \
        --data_type FP32
docker run -it \
    --volume $(pwd):/workding_dir \
    --workdir /workding_dir \
    openvino/ubuntu18_dev:2020.3 \
    python3 /opt/intel/openvino/deployment_tools/model_optimizer/mo_caffe.py \
        --input_model /workding_dir/resources/worker-safety-mobilenet/worker_safety_mobilenet.caffemodel \
        -o /workding_dir/resources/worker-safety-mobilenet/FP16/worker_safety_mobilenet.caffemodel \
        --data_type FP16
```

