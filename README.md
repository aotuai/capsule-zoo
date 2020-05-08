# Capsule Zoo
[![CircleCI](https://circleci.com/gh/aotuai/capsules.svg?style=svg)](https://circleci.com/gh/aotuai/capsules)

# Introduction

This repository is used to store open source Vision Capsules created by Aotu. These
capsules are built using the [Open Vision Capsule format][open vision capsules].

All of these models are compatible with and best run by BrainFrame. More information
about Brainframe can be found [at our website](http://aotu.ai)

# Repository Structure
All capsules are within the capsules/ directory. An unpackaged capsule is simply 
a directory with the required capsule files. For documentation on the format and 
how to write your own capsules, please refer to the [Open Vision Capsules docs][ovc docs].


Here is an example of what you might see in the capsules/ directory: 
```commandline
.
└── capsules
    ├── gender_classifier
    │   ├── capsule.py
    │   ├── meta.conf
    │   ├── model.LICENSE
    │   ├── model.weird_format
    │   └── README.md
    ├── person_detector
    │   ├── capsule.py
    │   ├── meta.conf
    │   ├── model.LICENSE
    │   ├── model.pb
    │   ├── README.md
    │   └── supporting_code.py
    └── vehicle_detector
        ├── capsule.py
        ├── meta.conf
        ├── model.LICENSE
        ├── README.md
        ├── openvino_model.bin
        └── openvino_model.xml
```
Each subdirectory such as gender_classifier/ or person_detector/ is an unpackaged
capsule that can be loaded (and automatically packaged) by BrainFrame. If you 
have an existing BrainFrame installation, simply copy the desired capsule directory
into the BrainFrame servers `capsules/` directory, and restart the server. BrainFrame
will automatically package and load the capsule. 

For most or all of the capsules, the code was created by Aotu while the model 
files within may have been pulled from license friendly 3rd party sources. 
Each capsule that has a model will contain a `model.LICENSE` file which is
the license associated with the source of that model. 
Furthermore, if the model is from a 3rd party, links to the 3rd party creator 
of the model will be given in the README.md of the capsule directory.

# Git LFS

This repository uses Git LFS to store model files and other large resources.
Please see [the installation guide][install git lfs] for details.

# Running Tests

This repository contains a test suite that packages up and runs every capsule
in the `capsules` directory.

Before running tests, install the necessary dependencies.

```commandline
pip install -r requirements.txt
```

Then, run the tests with `pytest`.

```commandline
pytest .
```

[install git lfs]: https://github.com/git-lfs/git-lfs/wiki/Installation
[open vision capsules]: https://github.com/opencv/open_vision_capsules
[ovc docs]: https://openvisioncapsules.readthedocs.io/en/latest/
