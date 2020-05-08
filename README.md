# BrainFrame VisionCapsule Zoo
| Tests | [![CircleCI](https://circleci.com/gh/aotuai/capsule_zoo/tree/master.svg?style=svg&circle-token=afb518744b2fce9932f645d081390eb4222d0b1d)](https://circleci.com/gh/aotuai/capsule_zoo/tree/master) |
|-------|---------------------------------------------------------------------------------------------------------------|

# Introduction

This repository is used to store open source VisionCapsules created by Aotu. These
capsules are built using the [OpenVisionCapsule format][open vision capsules].

All of these models are compatible with and best run by BrainFrame. More information
about Brainframe can be found [at our website](http://aotu.ai).

# Repository Structure
All capsules are within the capsules/ directory. An unpackaged capsule is simply 
a directory with the required capsule files. For documentation on the format and 
how to write your own capsules, please refer to the [OpenVisionCapsules docs][ovc docs].


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
Each subdirectory, such as `gender_classifier/` or `person_detector/` is an unpackaged
capsule that can be loaded (and automatically packaged) by BrainFrame. If you 
have an existing BrainFrame installation, simply copy the desired capsule directory
into the BrainFrame server's `capsules/` directory, and restart the server. BrainFrame
will automatically package and load the capsule. 

Currently all capsules in the repository were developed by Aotu, but pull 
requests are encouraged if you would like to contribute.
The model files the capsules use are either developed by Aotu or by 3rd parties,
but permissively licensed in either case. Each capsule that has a model will 
contain a `model.LICENSE` file associated with the source for that model. In 
addition, if the model is from a 3rd party, links to the upstream source of the 
model will be provided in the `README.md` of the capsule directory.

# Git LFS

This repository uses Git LFS to store model files and other large resources.
Please see [the installation guide][install git lfs] for details.

# Running Tests

This repository contains a test suite that packages up and runs every capsule
in the `capsules` directory. The tests use the `vcap` libraries existing test
utility that fuzzes many inputs and assesses if the capsule is conforming to the
 standard.

Before running tests, install the necessary dependencies.

```commandline
pip install -r tests/requirements.txt
```

Then, run the tests with `pytest`, at the root of the repository.

```commandline
pytest .
```

[install git lfs]: https://github.com/git-lfs/git-lfs/wiki/Installation
[open vision capsules]: https://github.com/opencv/open_vision_capsules
[ovc docs]: https://openvisioncapsules.readthedocs.io/en/latest/
