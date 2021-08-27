# About the Capsule
## Usage
This capsule is for detecting people and vehicles, but only when segmentation
information is necessary. That is, when a tight polygon approximation of the 
objects is necessary. 

## Model
### Architecture & Performance
This model is is a Mask R-CNN with ResNet50 backbone.

### Sources
Trained using the [Tensorflow Object Detection API](
https://github.com/tensorflow/models/blob/master/research/object_detection)

Some of the image preprocessing and mask postprocessing code was pulled from
this [python demo][python demo]. 

###  Model File Origin
The pretrained model was downloaded from the Open
[OpenCV Open Model Zoo][open model zoo]. 
For more information on this specific model, see its [model page][model page].

[open model zoo]: https://github.com/opencv/open_model_zoo
[model page]:
https://docs.openvinotoolkit.org/2020.3/_models_intel_instance_segmentation_security_0050_description_instance_segmentation_security_0050.html
[python demo]: 
https://github.com/openvinotoolkit/open_model_zoo/blob/master/demos/python_demos/instance_segmentation_demo/instance_segmentation_demo.py