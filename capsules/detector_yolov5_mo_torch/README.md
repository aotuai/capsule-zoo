# detector_yolov5_mo_torch (template, no model)

A YOLO object detector capsule for BrainFrame, built on OpenVINO. **This capsule contains no model**: loaded as it is,
it runs but detects nothing (the BrainFrame log says so). You add the model.

- **Input:** the full video frame (resized with padding to the model's input size).
- **Output:** a list of detections, each with a class name, a bounding box and a confidence.
  The class names are in `labels.py` (the 80 COCO classes by default): edit them to match your model.
- **Options:** `threshold` (confidence, default 0.25) and `iou_threshold` (non-max suppression, default 0.45).

## Use it

1. **Easiest: the BrainFrame Marketplace.** Open the Marketplace, then in My VisionCapsules use **Package Model**
   (your own OpenVINO `model.xml` + `model.bin`), or in the Ultralytics list use **Download & Package** to build a
   capsule from Ultralytics weights. The Marketplace fills this template for you.
2. **By hand.** Put your OpenVINO model next to `capsule.py` as `model.xml` and `model.bin`, set `labels.py` to your class
   names (in the model's output order), then package:
   `openvisioncapsule-tools capsule_packaging --capsule-dir <folder holding this capsule>`.
   If your model does not already divide the image by 255 and swap BGR to RGB, set
   `model_includes_preprocessing = False` in `settings.py`.

Supported model outputs: `(1, N, 5 + classes)` (original YOLOv5, `decode.py`). For anchor-free models
(YOLOv5u, YOLOv8, YOLO11) use the Marketplace, which picks the matching decoder.

## Licence

The capsule code is AOTU's. A model you add has its own licence: **Ultralytics YOLO models are AGPL-3.0**
(https://www.ultralytics.com/license). See `licenses/NOTICE`. Add `licenses/model.LICENSE` when you add a model.
