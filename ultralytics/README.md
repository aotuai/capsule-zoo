# Ultralytics YOLO capsule templates

These are the capsule templates that the BrainFrame **VisionCapsules AI Marketplace** uses to package
[Ultralytics](https://docs.ultralytics.com/models) YOLO models into the open vision capsule format. The marketplace
downloads the model from ultralytics.com, converts it to OpenVINO on the user's own machine, fills in a template and
packages the result with `openvisioncapsule-tools`.

A template is **not** a finished capsule: it has placeholders (`__CAPSULE_NAME__`, `__CAPSULE_DESCRIPTION__`,
`__CAPSULE_VERSION__`, ...) for the name, the description, the class names (`labels.py`) and the model files
(`model.xml`, `model.bin`, `meta.json`). You can use them without the marketplace: copy `common/` and one template
directory into a new capsule directory (files of the template replace files of `common/`), fill the placeholders and
add the OpenVINO model.

## Layout

```
templates/
  common/            files every capsule gets: capsule.py, backend.py, preprocess.py, nms.py, meta.conf
  <template>/        files that are specific to one model output (decode.py, ...); they replace the ones in common/
```

A few templates build on another one and only hold what is different. They are copied after their parent:

| Template | Parent |
|---|---|
| `segment_end2end` | `segment` |
| `pose_end2end` | `pose` |
| `obb_end2end` | `obb` |

The `classify` template does not use `nms.py`. Templates for models that already remove duplicate boxes (`yolo_end2end`,
`rtdetr`, `segment_end2end`, `pose_end2end`, `obb_end2end`) have no `iou_threshold` option.

## Which template for which model

The template follows the **output of the exported model**, so the same template serves every model family that exports
that way.

| Models | Task | Template |
|---|---|---|
| YOLOv3u, YOLOv5u (and the 1280 px "6u"), YOLOv8, YOLOv9, YOLO11, YOLO12, YOLO-World, YOLOE (all, also YOLOE-26), YOLOv8 variants (Open Images V7, x6, v8loader) | detect | `yolov5_anchor_free` |
| Original YOLOv5 (ONNX or OpenVINO export) | detect | `yolov5_anchor` |
| YOLO26, YOLOv10, YOLO26 variants (Objects365, distilled) | detect | `yolo_end2end` |
| RT-DETR | detect | `rtdetr` |
| YOLOv8, YOLOv9, YOLO11, YOLOE (all), FastSAM | segment | `segment` |
| YOLO26 | segment | `segment_end2end` |
| YOLOv8, YOLO11 (also the P6 model) | pose | `pose` |
| YOLO26 | pose | `pose_end2end` |
| YOLOv8, YOLO11 | obb | `obb` |
| YOLO26 (also its text detector) | obb | `obb_end2end` |
| YOLOv8, YOLO11, YOLO26 (`-cls`) | classify | `classify` |

What each template outputs:

| Template | Model output | What the capsule reports |
|---|---|---|
| `yolov5_anchor` | `(1, N, 5 + classes)` | boxes (NMS in the capsule) |
| `yolov5_anchor_free` | `(1, 4 + classes, N)` | boxes (NMS in the capsule) |
| `yolo_end2end` | `(1, 300, 6)` | boxes, already without duplicates |
| `rtdetr` | `(1, 300, 6)` with centre boxes as fractions | boxes |
| `segment` | `(1, 4 + classes + 32, N)` and the mask prototypes | a polygon outlining each object |
| `segment_end2end` | `(1, 300, 6 + 32)` and the mask prototypes | a polygon outlining each object |
| `pose` | `(1, 4 + classes + keypoints * 3, N)` | boxes, keypoints in extra data |
| `pose_end2end` | `(1, 300, 6 + keypoints * 3)` | boxes, keypoints in extra data |
| `obb` | `(1, 4 + classes + 1, N)` | rotated rectangles (four corners) and the angle in extra data |
| `obb_end2end` | `(1, 300, 7)` | rotated rectangles |
| `classify` | `(1, classes)` | an attribute on the detections of another capsule |

## The `classes` option

A capsule made from an open-vocabulary model (YOLO-World, YOLOE) has a `classes` option (`yolov5_anchor_free`,
`yolo_end2end`, `segment` and `segment_end2end` templates): the class names to report, separated by commas; empty means all.
`common/selection.py` turns the text into class numbers and `decode()` takes the best class among those only
(`selected` argument). The model's **vocabulary** is what was baked in when it was packaged (a text-prompt model: the names
typed in the marketplace's vocabulary box, which starts with the 80 COCO classes; a prompt-free YOLOE model: its own 4,585
names). The option starts with the **default class names** typed at packaging, and can be changed to any name of the
vocabulary. To use it in a template of your own, add the `classes` `TextOption` to `capsule.py` (see `CLASSES_OPTION` in the
marketplace's `packager/packager/render.py`).

## Models that are not packaged yet

These Ultralytics weights are not packaged yet. This is what has not been built, not what cannot be done:

| Models | What is missing |
|---|---|
| `yolo11n-grayscale` | A template for a one-channel input (the converter reads it, but the capsule fails to run it with three color channels; the template would convert the frame to grayscale first). |
| YOLOv8 `-human`, YOLO26 `-reid` and `-s3d` | A converter that can read them: the released Ultralytics that it installs lacks the model classes these files need (`HumanModel`, `ReidModel`, the `s3d` module). They may then fit an existing template (not checked). |
| YOLO26 `-depth` and `-sem` | A decision on how a capsule reports them (a depth map or a class per pixel has no BrainFrame output type; semantic segmentation could become polygons per class). The converter refuses these tasks today. |
| YOLO26 `-stereo3ddet` | A way to give the capsule two images (not tried). |
| SAM, SAM 2 and MobileSAM | A capsule that takes prompts, for example the boxes of another capsule, and returns masks (not tried). |
| YOLO-NAS | A converter for its library and a template for its output (not tried). |

## License

The templates are original AOTU code. The model weights and the Ultralytics library are licensed by Ultralytics
(AGPL-3.0); a capsule packaged from them is for the person who packaged it. See the marketplace documentation.

## Note

The marketplace repository
(`visioncapsule-marketplace/packager/templates/`) holds the copy that the packager runs. Keep both in step when a
template changes.
