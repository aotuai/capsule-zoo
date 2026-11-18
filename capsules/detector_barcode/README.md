# detector_barcode

## Documentation

### Overview

`detector_barcode` is a high-performance barcode and QR code decoding algorithm capsule built on an OpenCV image-processing pipeline. It supports multiple engines, multi-strategy progressive decoding, parallel execution, and automatic correction of skewed or perspective-distorted images before decoding.

---

### Features

| Feature | Description |
|---------|-------------|
| **Multi-engine support** | pyrxing (recommended, Python ≥ 3.11) or pyzbar (Python 3.8+); the available engine is selected automatically |
| **Progressive strategies** | Quick preprocess → Full preprocess → Perspective correction → Rotation correction → Multi-angle rotation |
| **Parallel decoding** | Thread pool runs all strategies concurrently; once any strategy succeeds, others are signalled to stop (speed-first) |
| **Sequential decoding** | Strategies execute one by one; stops at the first success (accuracy-first) |
| **Auto mode** | Automatically picks parallel or sequential based on pixel count (≥ 500×500 uses parallel) |
| **Perspective correction** | Detects rectangular / trapezoidal regions and applies perspective warp to flatten distortion before decoding (enabled only when distortion is detected) |
| **Rotation correction** | Detects the dominant line angle via HoughLines and auto-rotates to align |
| **Multi-code decoding** | Returns all codes found in the image (content, type, coordinates, confidence) in a single call |
| **Coordinate mapping** | Decoded coordinates are mapped back to the original image coordinate system via inverse transform chains |

---

### Requirements

- Python **≥ 3.8**
- At least one decoding engine installed (pyrxing or pyzbar)

---

### Dependencies

opencv-python

numpy

pillow

**Decoding engine (install at least one)**

```bash
# Recommended: pyrxing (high-performance, zxing-cpp via Rust bindings, requires Python ≥ 3.11)
pip install pyrxing

# Alternative: pyzbar (compatible with Python 3.8+, requires zbar system library)
pip install pyzbar
```

> **pyzbar system dependency**
>
> - **Ubuntu / Debian**: `sudo apt-get install libzbar0`
> - **macOS**: `brew install zbar`
> - **Windows**: `pip install pyzbar` — no extra steps needed (pre-compiled DLL included in the wheel)

---

### Engine decoding capability (pyzbar vs pyrxing)

| #    | Code Format    | pyzbar        | pyrxing   |
| ---- | -------------- | ------------- | --------- |
| 01   | Code128        | OK            | OK        |
| 02   | GS1-128        | OK            | OK        |
| 03   | ITF            | OK            | OK        |
| 04   | Code39         | OK            | OK        |
| 05   | DataMatrix     | Not supported | OK        |
| 06   | QRCode         | OK            | OK        |
| 07   | Code128 (Rack) | OK            | OK        |
| 08   | Code39 (Rack)  | OK            | OK        |
| 09   | PDF417         | Not supported | OK        |
| 10   | MaxiCode       | Not supported | OK        |
| 11   | UPC-A          | OK            | OK        |
| 12   | UPC-E          | Not supported | OK        |
| 13   | EAN-13         | OK            | OK        |
|      | **Total**      | **9/13**      | **13/13** |

------

### Return Value

Each decoded result is a `DetectionNode` object with the following fields:

| Field | Sub-field | Value | Description |
|-------|-----------|-------|-------------|
| `class_name` | - | barcode | |
| `extra_data` | `codetype` | `str` | Code format, e.g. `QR_CODE`, `EAN_13`, `CODE_128` |
| | `code` | `str` | Decoded content |
| `coords` | - | [(x0,y0), (x1,y1), (x2,y2), (x3,y3)] | Bounding rectangle vertices in the original image (pixels) Notes1 |
| `confidence` | - | None | Confidence score (always `None` — see note below) Notes2 |

###### Notes1:

When decoded with **pyrxing**:

   - `coords` for a barcode is returned as a straight line — this is by pyrxing design, not a bug

###### Notes2:

When decoded with **pyrxing**:

   - No confidence value is provided

When decoded with **pyzbar**:

   - pyzbar returns a `quality` value representing the decode quality of the symbol, which is unrelated to confidence

Therefore `confidence` always returns `None` in this capsule.

---

### Decoding Strategy Pipeline

The tool builds and executes up to 10 strategies in the following order:

| Strategy | Description |
|----------|-------------|
| Quick decode | Histogram equalization, ~5–10 ms |
| Full decode | Bilateral filter + CLAHE enhancement, ~20–30 ms |
| Perspective correction | Detects rectangular / trapezoidal regions and warps perspective (only when distortion is present) |
| Dominant-angle rotation | Auto-rotates to align based on HoughLines dominant angle |
| Perspective + rotation | Perspective correction followed by rotation alignment (combined strategy) |
| Rotation 45 / 90 / 180 / 270 / 315° | Fixed-angle rotations to cover all code orientations |

---

