# About the Capsule
## Usage
This capsule is for detecting safety equipment (safety vest and safety hat).

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
