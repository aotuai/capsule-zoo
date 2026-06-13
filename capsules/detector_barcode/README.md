# detector_barcode

[中文](#中文说明) | [English](#english-documentation)

---

## 中文说明

### 简介

`detector_barcode` 是一个高性能的条码 / 二维码解码算法胶囊，基于 OpenCV 图像处理流水线构建，支持多引擎、多策略、并行解码，并可自动对倾斜或透视畸变的图像进行校正后再解码。

---

### 功能特性

| 特性 | 说明 |
|------|------|
| **多引擎支持** | pyrxing（推荐，Python ≥ 3.11）或 pyzbar（兼容 Python 3.8+），自动检测并选择可用引擎 |
| **渐进式策略** | 快速预处理 → 完整预处理 → 透视校正 → 旋转校正 → 多角度旋转，逐级提升识别率 |
| **并行解码** | 多策略线程池并行执行，任一策略成功后主动通知其他线程停止，速度优先 |
| **顺序解码** | 逐级执行，首个成功即停，精度优先 |
| **自动模式** | 根据图片像素数自动选择并行/顺序模式（≥ 500×500 用并行） |
| **透视校正** | 自动检测矩形/梯形目标区域，应用透视变换展平畸变，校正后再解码 |
| **旋转校正** | 基于 HoughLines 检测主导角度，自动旋转对齐 |
| **多码解码** | 单次调用可返回图像中所有码的结果（内容、类型、坐标、置信度） |
| **坐标映射** | 解码坐标自动通过逆变换链映射回原图坐标系 |

---

### 环境要求

- Python **≥ 3.8**
- 解码引擎至少安装一个（pyrxing 或 pyzbar）

---

### 依赖

opencv-python 

numpy

pillow

**解码引擎（二选一，至少安装一个）**

```bash
# 推荐：pyrxing（高性能，基于 zxing-cpp + Rust 绑定，需要 Python ≥ 3.11）
pip install pyrxing

# 备选：pyzbar（兼容 Python 3.8+，需系统预先安装 zbar 库）
pip install pyzbar
```

> **pyzbar 系统依赖**
>
> - **Ubuntu / Debian**：`sudo apt-get install libzbar0`
> - **macOS**：`brew install zbar`
> - **Windows**：直接 `pip install pyzbar`，无需额外操作（wheel 包含预编译 DLL）

------

### 引擎解码能力(pyzbar vs pyrxing)

| #    | 码制           | pyzbar   | pyrxing   |
| ---- | -------------- | -------- | --------- |
| 01   | Code128        | OK       | OK        |
| 02   | GS1-128        | OK       | OK        |
| 03   | ITF            | OK       | OK        |
| 04   | Code39         | OK       | OK        |
| 05   | DataMatrix     | 不支持   | OK        |
| 06   | QRCode         | OK       | OK        |
| 07   | Code128 (Rack) | OK       | OK        |
| 08   | Code39 (Rack)  | OK       | OK        |
| 09   | PDF417         | 不支持   | OK        |
| 10   | MaxiCode       | 不支持   | OK        |
| 11   | UPC-A          | OK       | OK        |
| 12   | UPC-E          | 不支持   | OK        |
| 13   | EAN-13         | OK       | OK        |
|      | **合计**       | **9/13** | **13/13** |

---

### 返回值说明

每条解码结果为 `DetectionNode`对象，包含以下字段：

| 字段 | 子字段 | 值 | 说明 |
|------|------|------|------|
| `class_name` | - | barcode |  |
| `extra_data` | `codetype` | `str` | 码类型，如 `QR_CODE`、`EAN_13`、`CODE_128` 等 |
|  | `code` | `str` | 解码数据 |
| `coords` | - | [(x0,y0), (x1,y1), (x2,y2), (x3,y3)] | 在原图中的外接矩形坐标（像素）注1 |
| `confidence` | - | None | 置信度（引擎不支持时为 `None`）注2 |

###### 注1：

当使用pyrxing 解码时：

   - 解码bar code 返回coords 是一直线，pyrxing 设计如此，不是bug;

###### 注2：

当使用pyrxing 解码时：

   - 无 confidence

当使用pyzbar 解码时：

   - 返回的是quality, 代表解码图形的质量，与confidence 无关

因此本算法胶囊confidence 均返回None

---

### 解码策略说明

工具内部按以下顺序构建并执行解码策略（共最多 10 个）：

| 策略 | 说明 |
|------|------|
| 快速解码 | 直方图均衡化，约 5–10 ms |
| 完整解码 | 双边滤波 + CLAHE 增强，约 20–30 ms |
| 透视校正 | 检测矩形/梯形区域，透视展平后解码（仅在图像含畸变区域时启用） |
| 检测角旋转 | HoughLines 检测主导角度，自动旋转对齐 |
| 透视 + 旋转 | 先透视校正，再旋转对齐（组合策略） |
| 旋转 45 / 90 / 180 / 270 / 315° | 固定角度旋转，覆盖各方向条码 |

---

## English Documentation

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

