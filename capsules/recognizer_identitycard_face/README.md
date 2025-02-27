# About the Capsule
## Usage
This capsule creates encodings for face detections, which BrainFrame can use
to perform face recognition / tracking tasks. It must be used in conjunction 
with a face detection capsule. 

## Model
### Architecture & Performance
This model uses the FaceNet architecture.

Recognition accuracy on LFW dataset: 99%+

Loss Type: center_loss

### Sources
Trained using davidsandberg's [ Tensorflow FaceNet implementation](
https://github.com/davidsandberg/facenet).

### Dataset
This model was trained on the 
[VGGFace2](http://www.robots.ox.ac.uk/~vgg/data/vgg_face2/) 
dataset.