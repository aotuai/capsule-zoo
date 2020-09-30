# About the Capsule
## Usage
This capsule is for detecting and reading text in video. 

## Model
This capsule backend uses three different openVINO models. one for recognition, one for encoding the shapes, and one for decoding the text.
###  Model File Origin
The pretrained models were downloaded from the [OpenCV Open Model Zoo](https://github.com/opencv/open_model_zoo). 
For more information on these specific models, see the model pages below.

[text detector](https://docs.openvinotoolkit.org/2020.1/_models_intel_text_spotting_0001_detector_description_text_spotting_0001_detector.html)

[text recognizer encoder](https://docs.openvinotoolkit.org/2020.1/_models_intel_text_spotting_0001_recognizer_decoder_description_text_spotting_0001_recognizer_decoder.html)

[text recognizer decoder](https://docs.openvinotoolkit.org/2020.1/_models_intel_text_spotting_0001_recognizer_encoder_description_text_spotting_0001_recognizer_encoder.html)
