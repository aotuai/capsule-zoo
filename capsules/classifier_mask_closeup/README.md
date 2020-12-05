# About the Capsule
## Usage
This capsule is for detecting if a person has a mask on their face. 

## Model
###  Model File Origin
The pretrained model was downloaded from the 
[didi maskdetection respository](https://github.com/didi/maskdetection), and was
then converted to an OpenVINO model using this command:

```shell
docker run -it \
    --volume $(pwd):/model \
    --workdir /model \
    openvino/ubuntu18_dev:2020.3 \
    python3 /opt/intel/openvino/deployment_tools/model_optimizer/mo.py \
        --framework caffe \
        --input_model /model/face_mask.caffemodel \
        --input_proto /model/deploy.prototxt
```
