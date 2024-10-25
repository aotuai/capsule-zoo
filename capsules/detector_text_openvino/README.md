# About the Capsule
## Usage
This capsule is for detecting and reading text in video. 

## Model
This capsule backend uses three different openVINO models. one for recognition, one for encoding the shapes, and one for decoding the text.
###  Model File Origin
The pretrained models were downloaded from the [OpenCV Open Model Zoo][open model zoo]. 
The documentation for the composite model can be found on the 
[OpenVINO model page text spotting 0004][composite model 0004 documentation] and
[OpenVINO model page text spotting 0005][composite model 0005 documentation].

[open model zoo]: https://github.com/opencv/open_model_zoo
[composite model 0004 documentation]: https://docs.openvinotoolkit.org/2021.3/omz_models_model_text_spotting_0004.html
[composite model 0005 documentation]: https://docs.openvino.ai/2022.3/omz_models_model_text_spotting_0005.html
