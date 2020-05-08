# About the Capsule
## Usage
This capsule is for detecting poses of persons in video. It should be used
in conjunction with a person detection capsule. It is faster than most classifiers, 
because it is backed by a detection algorithm, and uses IOU to match pose detections
with human detections- to which it then adds labels to.

## Model
### Architecture & Performance
This model is a faster_rcnn_resnet_101 architecture.

### Sources
Trained using the [Tensorflow Object Detection API](
https://github.com/tensorflow/models/blob/master/research/object_detection)

###  Model File Origin
The pretrained model was downloaded from the
[Tensorflow Object Detection Model Zoo](
https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md)

### Dataset
This model was trained on the [AVA Dataset](https://research.google.com/ava/).