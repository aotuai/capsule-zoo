# About the Capsule
## Usage
This capsule is for detecting and reading text in video. 

## Model
This capsule backend uses three different openVINO models. one for recognition, one for encoding the shapes, and one for decoding the text.
###  Model File Origin
The pretrained models were downloaded from the [OpenCV Open Model Zoo][open model zoo]. 
The documentation for the composite model can be found on the 
[OpenVINO model page][composite model documentation].

[open model zoo]: https://github.com/opencv/open_model_zoo
[composite model documentation]: https://docs.openvino.ai/2022.3/omz_models_model_text_spotting_0005.html
