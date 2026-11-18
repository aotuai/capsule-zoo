"""
Barcode / QR code decoding tool - full optimized version

Python version required: >= 3.8

Dependencies:
    opencv-python     image processing and transforms
    numpy             matrix operations
    pyrxing (recommended)  main decoding engine (high-performance QR/barcode); BrainFrame OS uses Python 3.9, so pyrxing (needs Python >= 3.11) cannot be used at the moment
    pyzbar (fallback)       fallback decoding engine
    Pillow             image conversion for pyzbar (required on the pyzbar path)

Features:
1. Progressive processing pipeline (fast path first)
2. Parallel decoding + active stop mechanism
3. Smart perspective correction
4. Complete resource management
5. Multi-code decoding, outputs type and coordinates
6. Annotated image output
"""

import sys
import cv2
import numpy as np
import threading
import concurrent.futures
import time
import logging
import os
from typing import Optional, Tuple, List, Callable, Any, Dict
from dataclasses import dataclass, field
from enum import Enum

#logging.basicConfig(level=logging.INFO)
# Try to import the decoding libraries
# PIL/Pillow is a dependency of both engines:
#   - pyrxing: read_barcodes only accepts a str or an ImageProtocol, a numpy.ndarray must be converted to a PIL Image first
#   - pyzbar: takes a PIL Image as input directly
try:
    from PIL import Image as PILImage
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    logging.warning("PIL/Pillow is not installed, the decoding engines will be degraded or unavailable")

try:
    import pyrxing as rx
    PYRXING_AVAILABLE = True
    PYZBAR_AVAILABLE = False
    if not PIL_AVAILABLE:
        logging.warning("PIL/Pillow is not installed, pyrxing cannot take numpy array input")
    logging.info("Using the pyrxing decoding engine")
except ImportError:
    PYRXING_AVAILABLE = False
    try:
        from pyzbar.pyzbar import decode as zbar_decode
        PYZBAR_AVAILABLE = True
        if not PIL_AVAILABLE:
            logging.warning("PIL/Pillow is not installed, the pyzbar path will be unavailable")
        logging.info("Using the pyzbar decoding engine")
    except ImportError as e:
        PYZBAR_AVAILABLE = False
        err_msg = f"Please install pyrxing or pyzbar: {e}"
        raise ImportError(err_msg)


class DecodeStatus(Enum):
    """Decoding state"""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    STOPPED = "stopped"


@dataclass
class CodeResult:
    """The decoding result of a single code"""
    data: str                           # decoded content
    barcode_type: str                   # code type (QR_CODE, EAN_13, etc.)
    rect: Tuple[int, int, int, int]     # bounding rectangle (x, y, w, h)
    points: List[Tuple[int, int]] = field(default_factory=list)  # polygon vertices [(x,y), ...]
    confidence: Optional[float] = None  # confidence (0.0~1.0), None when the decoding engine does not provide it

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "data": self.data,
            "type": self.barcode_type,
            "rect": {"x": self.rect[0], "y": self.rect[1], "w": self.rect[2], "h": self.rect[3]},
            "points": self.points,
        }
        if self.confidence is not None:
            result["confidence"] = self.confidence
        return result


class QrBarDecoder:
    """Stoppable decoder - can actively stop the other threads"""

    def __init__(self, timeout_ms: int = 500, check_interval_ms: int = 10,
                 annotate_path: Optional[str] = None, debug_dir: Optional[str] = None):
        """
        Args:
            timeout_ms: overall timeout (milliseconds)
            check_interval_ms: stop check interval (milliseconds)
            annotate_path: output directory of the decoded images, nothing is saved when None
            debug_dir: directory for the debug images, nothing is saved when None
        """
        self.timeout_ms = timeout_ms
        self.check_interval_ms = check_interval_ms
        self.stop_event = threading.Event()
        self.executor = None
        self.annotate_path = annotate_path
        self.debug_dir = debug_dir
        self._debug_counter = 0

    def __enter__(self) -> "QrBarDecoder":
        """Context manager entry, supports the with QrBarDecoder() as decoder: syntax"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit: makes sure the resources (threads, events, etc.) are cleaned up"""
        self.stop_event.set()
        # Do not suppress any exception, the caller decides how to handle it
        return False

    def _save_debug_image(self, image: np.ndarray, method_name: str):
        """Save a debug image, supports Chinese paths (cv2.imwrite has encoding problems with non-ASCII paths on Windows, so imencode + native Python writing is used)"""
        self._debug_counter += 1
        safe_name = method_name.replace(" ", "_").replace("/", "_").replace("\\", "_")
        filename = f"{self._debug_counter:02d}_{safe_name}.png"
        debug_path = os.path.join(self.debug_dir, filename)
        try:
            success, buf = cv2.imencode('.png', image)
            if success:
                with open(debug_path, 'wb') as f:
                    f.write(buf.tobytes())
                logging.debug(f"Debug image saved: {debug_path}")
            else:
                logging.debug(f"Failed to encode the debug image: {method_name}")
        except Exception as e:
            logging.debug(f"Failed to save the debug image: {e}")

    # ========== Image preprocessing ==========

    def _quick_preprocess(self, image: np.ndarray) -> np.ndarray:
        """Fast preprocessing (~5-10ms)"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # Fast histogram equalization
        gray = cv2.equalizeHist(gray)
        return gray

    def _full_preprocess(self, image: np.ndarray) -> np.ndarray:
        """Full preprocessing (~20-30ms)"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # Denoise
        gray = cv2.bilateralFilter(gray, 9, 75, 75)

        # CLAHE contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        gray = clahe.apply(gray)

        return gray

    @staticmethod
    def _identity_preprocess(image: np.ndarray) -> np.ndarray:
        """Identity preprocessing (the image is already preprocessed, return it as is)"""
        return image

    # ========== Core decoding function ==========

    def _decode_core(self, image: np.ndarray) -> List[CodeResult]:
        """
        Core decoding logic, returns all decoding results (with type and coordinates)
        The engine is chosen by the PYRXING_AVAILABLE / PYZBAR_AVAILABLE constants,
        no external parameter is needed - pyrxing comes first and pyzbar is the automatic fallback.
        """
        # Convert to RGB (both pyrxing and pyzbar need RGB input)
        if len(image.shape) == 3:
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        results = []

        # Use pyrxing first
        # pyrxing.read_barcodes only accepts a str (path) or an object that conforms to ImageProtocol (such as a PIL Image)
        # numpy.ndarray does not satisfy ImageProtocol (it has no .width/.height/.convert/.load),
        # so it must be converted to a PIL Image first, otherwise ValueError('value must be either str or conform to ImageProtocol') is raised
        if PYRXING_AVAILABLE and PIL_AVAILABLE:
            logging.info(f"Decoding with pyrxing")
            try:
                pil_for_rx = PILImage.fromarray(rgb)
                barcodes = rx.read_barcodes(pil_for_rx)
                for bc in barcodes:
                    # Handle the attribute names of different pyrxing versions
                    if hasattr(bc, 'text'):
                        data = bc.text
                    elif hasattr(bc, 'data'):
                        raw = bc.data
                        data = raw.decode('utf-8', errors='ignore') if isinstance(raw, bytes) else str(raw)
                    else:
                        continue

                    if hasattr(bc, 'format'):
                        btype = bc.format
                    elif hasattr(bc, 'symbol_type'):
                        btype = bc.symbol_type.name
                    else:
                        btype = "UNKNOWN"

                    # Extract the coordinate points
                    pts = []
                    if hasattr(bc, 'points') and bc.points:
                        pts = [(int(p.x), int(p.y)) for p in bc.points]
                    elif hasattr(bc, 'position') and bc.position:
                        pts = [(int(p.x), int(p.y)) for p in bc.position]

                    # Compute the rect
                    if pts:
                        xs = [p[0] for p in pts]
                        ys = [p[1] for p in pts]
                        x_min, x_max = min(xs), max(xs)
                        y_min, y_max = min(ys), max(ys)
                        rect = (x_min, y_min, x_max - x_min, y_max - y_min)
                    else:
                        rect = (0, 0, 0, 0)

                    # Extract the confidence (supported by some pyrxing versions)
                    conf = None
                    if hasattr(bc, 'confidence'):
                        conf = float(bc.confidence)

                    results.append(CodeResult(data=data, barcode_type=btype,
                                              rect=rect, points=pts, confidence=conf))
            except Exception as e:
                logging.debug(f"pyrxing decoding failed: {e}")

        # When pyrxing has no result or is unavailable, try pyzbar
        if not results and PYZBAR_AVAILABLE and PIL_AVAILABLE:
            logging.info(f"Decoding with pyzbar")
            try:
                pil_img = PILImage.fromarray(rgb)
                decoded = zbar_decode(pil_img)
                for d in decoded:
                    data = d.data.decode('utf-8', errors='ignore')
                    btype = d.type

                    # pyzbar provides rect and polygon
                    r = d.rect
                    rect = (r.left, r.top, r.width, r.height)

                    pts = [(p.x, p.y) for p in d.polygon]

                    # The quality field of pyzbar is the decoding quality (an integer, depends on the code type)
                    conf = float(d.quality) if hasattr(d, 'quality') else None

                    results.append(CodeResult(data=data, barcode_type=btype,
                                              rect=rect, points=pts, confidence=conf))
            except Exception as e:
                logging.debug(f"pyzbar decoding failed: {e}")

        return results

    def decode_with_stop_check(self,
                               image: np.ndarray,
                               method_name: str,
                               preprocess_func: Callable[[np.ndarray], np.ndarray],
                               stop_event: Optional[threading.Event] = None) -> List[CodeResult]:
        """
        Decoding task with stop checks, returns all decoding results

        The preprocessing function is given by preprocess_func (fast/full preprocessing),
        the decoding engine is chosen automatically by PYRXING_AVAILABLE / PYZBAR_AVAILABLE, no external parameter is needed.

        Args:
            image: input image
            method_name: name of the method (used in the logs)
            preprocess_func: preprocessing function, with the signature (np.ndarray) -> np.ndarray
            stop_event: external stop signal (the instance-level stop_event is used when None)
        """
        # Prefer the local stop_event that is passed in, to avoid the race condition of a shared instance
        _stop = stop_event if stop_event is not None else self.stop_event
        start_time = time.time()

        # Check the stop flag
        if _stop.is_set():
            logging.debug(f"{method_name} the stop signal was already received when the task started")
            return []

        # Preprocessing
        try:
            processed = preprocess_func(image)
        except Exception as e:
            logging.debug(f"{method_name} preprocessing failed: {e}")
            return []

        # Debug: save the image of each strategy after processing (placed before the stop check, so that the image of every strategy is saved in parallel mode)
        if self.debug_dir:
            self._save_debug_image(processed, method_name)

        # Check the stop flag again
        if _stop.is_set():
            logging.debug(f"{method_name} the stop signal was received after preprocessing")
            return []

        # Decoding: the engine is chosen inside _decode_core from the libraries that are available
        results = self._decode_core(processed)

        elapsed_ms = (time.time() - start_time) * 1000

        if results:
            logging.info(f"✅ {method_name} succeeded, found {len(results)} code(s), took {elapsed_ms:.0f}ms")
            return results

        logging.debug(f"{method_name} failed, took {elapsed_ms:.0f}ms")
        return []

    # ========== Perspective correction ==========

    def _needs_perspective_correction(self, image: np.ndarray) -> bool:
        """
        Quickly decide whether perspective correction is needed (~2ms)

        Detect whether the image shows a clear "trapezoid distortion" - that is, two groups of
        approximately parallel line segments with clearly different directions (such as the four edges of a tilted document/card).
        Scattered segments from curved edges (such as glasses frames) do not match the "two groups of parallel lines" pattern and are filtered out.

        True is returned when (all must hold):
        1. At least 5 valid line segments are detected
        2. The standard deviation of the angles is > 25° (stricter than the original 15°, to exclude scattered curved edges)
        3. There are at least two clearly different dominant angle clusters (at least 2 lines in each)
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # Downsample for speed
        if gray.shape[0] > 400:
            scale = 400 / gray.shape[0]
            gray = cv2.resize(gray, None, fx=scale, fy=scale)

        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=50)

        if lines is None or len(lines) < 5:
            return False

        # Collect all angles and map them to the range [0, 90) (a line has no direction)
        # The theta of HoughLines is in the range [0, π); after conversion:
        # - theta ∈ [0, π/2) → angle ∈ [0°, 90°)
        # - theta ∈ [π/2, π) → angle ∈ [90°, 180°) → equivalent to [0°, 90°)
        angles = []
        for line in lines[:20]:
            theta = line[0][1]
            angle = theta * 180 / np.pi
            # Map to [0, 90): a line has no direction, 0° and 180° are equivalent, and so are -87° and 87°
            if angle >= 90:
                angle = 180 - angle
            angles.append(angle)

        if len(angles) < 5:
            return False

        # Condition 1: the spread of the angles must be large enough
        # Note: the standard deviation is only a rough filter; the real protection is provided by the
        # rectangularity, vertex count and area filters inside _perspective_correct, so the threshold should not be too high here
        if np.std(angles) <= 15:
            return False

        # Condition 2: there must be at least two clearly different angle clusters (a trapezoid has two pairs of opposite sides)
        # Groups: "nearly horizontal" (< 45°) and "nearly vertical" (>= 45°)
        horizontal = [a for a in angles if a < 45]
        vertical = [a for a in angles if a >= 45]

        if len(horizontal) < 2 or len(vertical) < 2:
            return False

        # The difference between the median angles of the two groups must be large enough
        med_h = float(np.median(horizontal))
        med_v = float(np.median(vertical))
        angle_diff = abs(med_v - med_h)

        if angle_diff <= 30:
            return False

        # Condition 3: exclude an exact front-view rectangle (horizontal lines ≈ 0° and vertical lines ≈ 90°)
        # If both groups of lines are exactly aligned with the axes, it is a front view and no perspective correction is needed
        h_aligned = all(a < 5 for a in horizontal)
        v_aligned = all(a > 85 for a in vertical)
        if h_aligned and v_aligned:
            return False

        return True

    def _perspective_correct(self, image: np.ndarray,
                             return_matrix: bool = False) -> Any:
        """
        Perspective correction (~30-50ms)

        Find a rectangular area with clear perspective distortion in the image (such as a tilted document/card)
        and correct it to a front view.
        If no valid target area is found, None is returned (instead of distorting the original image).

        Validity constraints (to avoid mistaking the edges of the whole image, background color blocks, or elliptical/round objects for the target):
        1. The contour area is < 85% of the image area (excludes a "whole image contour")
        2. The contour area is > 8% of the image area (excludes noise and small barcode areas; the target to correct should be a large area such as a document/card)
        3. The aspect ratio is within [0.15, 6.5] (excludes extremely flat/thin false detections)
        4. Rectangularity (contour area / minAreaRect area): >= 0.60 with 4 vertices, >= 0.82 with 5-6 vertices
        5. The approximated polygon has <= 6 vertices (excludes ellipses/circles/irregular shapes)

        Args:
            return_matrix: if True, returns (corrected image, perspective matrix); otherwise only the corrected image
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        img_h, img_w = image.shape[:2]
        img_area = img_h * img_w

        # Downsample for speed
        scale = 1.0
        if gray.shape[0] > 600 or gray.shape[1] > 600:
            scale = min(600 / max(gray.shape), 1.0)
            if scale < 1.0:
                gray_small = cv2.resize(gray, None, fx=scale, fy=scale)
            else:
                gray_small = gray
        else:
            gray_small = gray
            scale = 1.0

        # Adaptive threshold
        binary = cv2.adaptiveThreshold(gray_small, 255,
                                       cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, 11, 5)

        # Morphological operations
        kernel = np.ones((5, 5), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        # Find the contours (take the 5 candidates with the largest area)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

        for contour in contours:
            area = cv2.contourArea(contour)
            # Area converted back to the original scale
            real_area = area / (scale * scale)

            # Constraint 1: exclude areas that are too small (noise/small barcode areas) or too large (edges of the whole image)
            # Perspective correction of a very small rectangle is pointless (the target should be a large area such as a document/card)
            if real_area < img_area * 0.08 or real_area > img_area * 0.85:
                logging.debug(f"Perspective correction: skipping contour (area ratio {real_area/img_area:.2%}）")
                continue

            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

            if len(approx) < 4:
                continue

            # Constraint 0: check the number of vertices of the approximated polygon
            # A rectangle has 4 vertices; too many vertices (>6) means an ellipse/circle/irregular shape
            if len(approx) > 6:
                logging.debug(f"Perspective correction: skipping contour ({len(approx)} vertices, probably an ellipse/circle)")
                continue

            # Get the minimum-area rotated rectangle
            rect = cv2.minAreaRect(contour)
            (cx, cy), (rw, rh), angle = rect

            # Constraint 3: aspect ratio filter (excludes extremely thin/flat false detections)
            if rw == 0 or rh == 0:
                continue
            aspect = max(rw, rh) / min(rw, rh)
            if aspect > 6.5 or aspect < 0.15:
                logging.debug(f"Perspective correction: skipping contour (aspect ratio {aspect:.2f}, out of the reasonable range)")
                continue

            # Constraint 4: rectangularity check (contour area / minimum bounding rectangle area)
            # The ratio is close to 1.0 for a rectangle, about 0.6~0.95 for a trapezoid, ≈0.785 for an ellipse and ≈0.785 for a circle
            # Combined judgment of rectangularity and vertex count:
            #   - 4 vertices + rectangularity >= 0.60 → rectangle/trapezoid (keep)
            #   - 5-6 vertices + rectangularity >= 0.82 → rounded rectangle (keep)
            #   - otherwise → ellipse/irregular (exclude)
            # Note: the areas must be compared at the same scale (both computed at the downsampled scale)
            min_rect_area = rw * rh
            if min_rect_area <= 0:
                continue
            rect_extent = area / min_rect_area  # area and rw*rh are both downsampled values
            if len(approx) == 4 and rect_extent < 0.60:
                logging.debug(f"Perspective correction: skipping contour (rectangularity {rect_extent:.2f}, 4 vertices but rectangularity too low)")
                continue
            elif len(approx) > 4 and rect_extent < 0.82:
                logging.debug(f"Perspective correction: skipping contour (rectangularity {rect_extent:.2f}, {len(approx)} vertices, probably an ellipse/circle)")
                continue

            box = cv2.boxPoints(rect)

            if scale < 1.0:
                box = box / scale

            logging.debug(f"Perspective correction: found a valid area, area ratio {real_area/img_area:.2%}, angle {angle:.1f}°, rectangularity {rect_extent:.2f}")
            corrected, matrix = self._apply_perspective_transform(image, box.astype(np.float32))
            if return_matrix:
                return corrected, matrix
            return corrected

        logging.debug("Perspective correction: no target area that meets the conditions, skipping")
        if return_matrix:
            return None, None
        return None

    def _apply_perspective_transform(self, image: np.ndarray, points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply the perspective transform, returns (corrected image, perspective matrix)"""
        rect = self._order_points(points)
        (tl, tr, br, bl) = rect

        width = max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl))
        height = max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr))

        width, height = int(max(width, 100)), int(max(height, 50))

        dst = np.array([[0, 0], [width-1, 0], [width-1, height-1], [0, height-1]],
                       dtype=np.float32)

        matrix = cv2.getPerspectiveTransform(rect, dst)
        corrected = cv2.warpPerspective(image, matrix, (width, height))

        return corrected, matrix

    def _order_points(self, pts: np.ndarray) -> np.ndarray:
        """Order the four corner points"""
        rect = np.zeros((4, 2), dtype=np.float32)

        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]

        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]

        return rect

    def _detect_dominant_angle(self, image: np.ndarray, max_lines: int = 30) -> Optional[float]:
        """
        Detect the dominant angle of the lines in the image
        Uses histogram clustering to find the densest angle range and takes the median
        Returns the angle (in degrees), or None
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # Downsample
        if gray.shape[0] > 500:
            scale = 500 / gray.shape[0]
            small = cv2.resize(gray, None, fx=scale, fy=scale)
        else:
            small = gray

        edges = cv2.Canny(small, 50, 150)
        lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=80)

        if lines is None:
            return None

        # Collect all angles and map them to the range [-90, 90)
        angles = []
        for line in lines[:max_lines]:
            theta = line[0][1]
            angle = theta * 180 / np.pi
            if angle >= 90:
                angle -= 180
            angles.append(angle)

        if len(angles) < 2:
            return None

        # Sort by angle and take the largest gap as the cluster boundary
        angles.sort()
        n = len(angles)
        max_gap = -1
        max_gap_idx = 0
        for i in range(n):
            next_i = (i + 1) % n
            gap = angles[next_i] - angles[i]
            if next_i == 0:
                gap += 180  # wrap around
            if gap > max_gap:
                max_gap = gap
                max_gap_idx = next_i

        # Starting from the largest gap, take the consecutive majority of the angles as the main cluster
        sorted_from_gap = angles[max_gap_idx:] + angles[:max_gap_idx]
        half = max(1, n // 2)
        main_cluster = sorted_from_gap[:half]

        return float(np.median(main_cluster))

    def _rotate_correct(self, image: np.ndarray) -> np.ndarray:
        """In-plane rotation correction, any angle is supported"""
        angle = self._detect_dominant_angle(image)

        if angle is None or abs(angle) <= 1.0:
            return image

        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, matrix, (w, h),
                                 flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_REPLICATE)
        return rotated

    # ========== Building the strategies and the inverse transforms ==========

    def _build_strategies(self, image: np.ndarray,
                          stop_event: Optional[threading.Event] = None) -> List[tuple]:
        """
        Build the unified list of decoding strategies, shared by the sequential and parallel modes

        Returns:
            list of strategies, each: (image, method_name, preprocess_func, stop_event, inv_transforms)
            inv_transforms: inverse transform chain [(M_inv, is_perspective), ...], applied in reverse order
        """
        _stop = stop_event if stop_event is not None else self.stop_event
        strategies = []

        # Strategy 1: fast preprocessing (original image, no coordinate conversion needed)
        strategies.append((
            image, "fast decoding", self._quick_preprocess, _stop, []
        ))

        # Strategy 2: full preprocessing (original image, no coordinate conversion needed)
        strategies.append((
            image, "full decoding", self._full_preprocess, _stop, []
        ))

        # Precompute the perspective correction (shared by strategies 3 and 5)
        # Try the perspective correction directly; the filter conditions inside _perspective_correct
        # (area, rectangularity, vertex count, etc.) decide whether a correction is needed, no pre-check is required
        persp_img = None
        M_persp_inv = None
        persp_img, M_persp = self._perspective_correct(image, return_matrix=True)
        if persp_img is not None and M_persp is not None:
            M_persp_inv = np.linalg.inv(M_persp)

            # Strategy 3: perspective correction + full preprocessing
            strategies.append((
                persp_img, "perspective correction", self._full_preprocess, _stop,
                [(M_persp_inv, True)]
            ))

        # Precompute the rotation by the detected angle (shared by strategies 4 and 5)
        detected_angle = self._detect_dominant_angle(image)
        rot_img = None
        M_rot_inv = None
        if detected_angle is not None and abs(detected_angle) > 1.0:
            h, w = image.shape[:2]
            center = (w // 2, h // 2)
            M_fwd = cv2.getRotationMatrix2D(center, detected_angle, 1.0)
            rot_img = cv2.warpAffine(image, M_fwd, (w, h),
                                     flags=cv2.INTER_LINEAR,
                                     borderMode=cv2.BORDER_REPLICATE)
            M_rot_inv = cv2.invertAffineTransform(M_fwd)

            # Strategy 4: rotation by the detected angle + full preprocessing
            strategies.append((
                rot_img, f"rotate by detected angle {detected_angle:.1f} degrees", self._full_preprocess, _stop,
                [(M_rot_inv, False)]
            ))

        # Strategy 5: perspective + rotation combined
        if persp_img is not None and M_persp_inv is not None:
            persp_angle = self._detect_dominant_angle(persp_img)
            if persp_angle is not None and abs(persp_angle) > 1.0:
                ph, pw = persp_img.shape[:2]
                pcenter = (pw // 2, ph // 2)
                M_protf = cv2.getRotationMatrix2D(pcenter, persp_angle, 1.0)
                persp_rot_img = cv2.warpAffine(persp_img, M_protf, (pw, ph),
                                               flags=cv2.INTER_LINEAR,
                                               borderMode=cv2.BORDER_REPLICATE)
                M_protf_inv = cv2.invertAffineTransform(M_protf)
                # Inverse transform chain: first the inverse rotation, then the inverse perspective
                strategies.append((
                    persp_rot_img, "perspective + rotation", self._full_preprocess, _stop,
                    [(M_protf_inv, False), (M_persp_inv, True)]
                ))

        # Strategies 6-10: fixed-angle rotation (rotates the grayscale image after fast preprocessing)
        quick_gray = self._quick_preprocess(image)
        for angle in [45, 90, 180, 270, 315]:
            h, w = quick_gray.shape[:2]
            center = (w // 2, h // 2)
            matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(quick_gray, matrix, (w, h))
            M_rot_inv = cv2.invertAffineTransform(matrix)
            strategies.append((
                rotated, f"rotate {angle} degrees", self._identity_preprocess, _stop,
                [(M_rot_inv, False)]
            ))

        return strategies

    @staticmethod
    def _apply_inv_transforms(results: List[CodeResult],
                               inv_transforms: List[Tuple[np.ndarray, bool]]) -> List[CodeResult]:
        """
        Apply the coordinate mapping in the reverse order of the inverse transform chain

        inv_transforms: [(M_inv, is_perspective), ...]
        Order: the last item of the chain is applied first (reverse order), because the transforms are recorded in forward order
        For example: original → perspective → rotation → decode; inv = [(M_rot_inv, F), (M_persp_inv, T)]
        Inverse mapping: first the inverse rotation, then the inverse perspective
        """
        if not inv_transforms or not results:
            return results
        for M_inv, is_perspective in reversed(inv_transforms):
            results = [QrBarDecoder._inverse_map_result(r, M_inv, is_perspective)
                       for r in results]
        return results

    # ========== Drawing the annotations ==========

    @staticmethod
    def _inverse_map_result(result: CodeResult, M_inv: np.ndarray,
                            is_perspective: bool = False) -> CodeResult:
        """Map the result coordinates back to the coordinate system of the original image through the inverse transform"""
        if M_inv is None:
            return result

        # Inverse-map the polygon vertices
        new_points = []
        if result.points:
            pts = np.array([result.points], dtype=np.float32)
            if is_perspective:
                mapped = cv2.perspectiveTransform(pts, M_inv)
            else:
                mapped = cv2.transform(pts, M_inv)
            new_points = [(int(round(p[0])), int(round(p[1]))) for p in mapped[0]]

        # Inverse-map the four corners of the rect and recompute the bounding rectangle
        x, y, w, h = result.rect
        corners = np.array([[[x, y], [x + w, y], [x + w, y + h], [x, y + h]]],
                           dtype=np.float32)
        if is_perspective:
            mapped_corners = cv2.perspectiveTransform(corners, M_inv)
        else:
            mapped_corners = cv2.transform(corners, M_inv)
        mc = [(int(round(p[0])), int(round(p[1]))) for p in mapped_corners[0]]
        xs = [p[0] for p in mc]
        ys = [p[1] for p in mc]
        new_rect = (min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys))

        return CodeResult(data=result.data, barcode_type=result.barcode_type,
                          rect=new_rect, points=new_points,
                          confidence=result.confidence)

    @staticmethod
    def _draw_annotations(image: np.ndarray, results: List[CodeResult]) -> np.ndarray:
        """Draw the annotation box, type and content of the codes on the image"""
        annotated = image.copy()
        colors = [
            (0, 0, 255),    # red
            (255, 0, 0),    # blue
            (0, 165, 255),  # orange
            (255, 0, 255),  # purple
            (0, 255, 255),  # yellow
        ]

        for i, r in enumerate(results):
            color = colors[i % len(colors)]
            x, y, w, h = r.rect

            # Draw the bounding rectangle
            cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)

            # Label text: type + content (truncated)
            label_text = f"[{r.barcode_type}] {r.data}"
            if len(label_text) > 60:
                label_text = label_text[:57] + "..."

            # Compute the position and size of the text
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            thickness = 1
            (tw, th), baseline = cv2.getTextSize(label_text, font, font_scale, thickness)

            # Label background
            label_y = max(y - 6, th + 6)
            cv2.rectangle(annotated,
                          (x, label_y - th - 6),
                          (x + tw + 4, label_y + baseline),
                          color, -1)
            cv2.putText(annotated, label_text,
                        (x + 2, label_y - 4),
                        font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

        return annotated

    # ========== Sequential decoding (progressive) ==========

    def decode_sequential(self, image: np.ndarray) -> List[CodeResult]:
        """
        Sequential decoding (accuracy first)
        Runs the strategies one by one and stops at the first success
        Returns the list of all decoded code results

        Args:
            image: image
        """
        start_time = time.time()
        logging.info("Decoding in sequential mode")

        strategies = self._build_strategies(image)
        results = []

        for i, (img, method_name, preprocess, _stop, inv_transforms) in enumerate(strategies, 1):
            level_name = f"level {i}-{method_name}"
            logging.info(f"Level {i}: {method_name}...")

            level_results = self.decode_with_stop_check(img, level_name, preprocess)
            if level_results:
                # Apply the inverse transform chain
                level_results = self._apply_inv_transforms(level_results, inv_transforms)
                results = level_results
                elapsed = (time.time() - start_time) * 1000
                logging.info(f"✅ Level {i} succeeded, found {len(results)} code(s), total time {elapsed:.0f}ms")
                break

        elapsed = (time.time() - start_time) * 1000
        if not results:
            logging.error(f"All levels failed, total time {elapsed:.0f}ms")

        logging.info(f"Total time of sequential decoding: {elapsed:.0f}ms")

        # Save the annotated image
        if self.annotate_path and results:
            self._save_annotation(image, results)

        return results

    # ========== Parallel decoding (speed first) ==========

    def decode_parallel(self, image: np.ndarray) -> List[CodeResult]:
        """
        Parallel decoding (speed first)
        Runs several strategies at the same time and collects all successful results (deduplicated)

        Args:
            image: image
        """
        start_time = time.time()
        logging.info("Decoding in parallel mode")

        # Create an independent local stop flag for every call, to avoid the race condition caused by sharing the instance flag
        local_stop = threading.Event()

        # Build the unified strategy list
        strategies = self._build_strategies(image, stop_event=local_stop)

        # Run in parallel with a thread pool
        all_raw = self._run_parallel_strategies(strategies, start_time, local_stop)

        # Deduplicate (by the data content)
        seen = set()
        unique = []
        for r in all_raw:
            if r.data not in seen:
                seen.add(r.data)
                unique.append(r)

        elapsed = (time.time() - start_time) * 1000
        logging.info(f"Total time of parallel decoding: {elapsed:.0f}ms")

        # Save the annotated image
        if self.annotate_path and unique:
            self._save_annotation(image, unique)

        return unique

    def _save_annotation(self, image: np.ndarray, results: List[CodeResult]):
        """Save the annotated image"""
        try:
            annotated = self._draw_annotations(image, results)
            cv2.imwrite(self.annotate_path, annotated)
            logging.info(f"Annotated image saved: {self.annotate_path}")
        except Exception as e:
            logging.error(f"Failed to save the annotated image: {e}")

    def _run_parallel_strategies(self, strategies: List[tuple], start_time: float,
                                  local_stop: Optional[threading.Event] = None) -> List[CodeResult]:
        """
        Run several decoding strategies in parallel
        Collects all successful results (different strategies may find different codes)
        The coordinates are mapped back to the coordinate system of the original image through the inverse transform chain automatically

        Args:
            strategies: the list of strategies returned by _build_strategies
                        each: (image, method_name, preprocess_func, stop_event, inv_transforms)
            start_time: start of the timing
            local_stop: local stop event (passed in by decode_parallel, to avoid race conditions)
        """
        # Use the local stop that is passed in, and fall back to the instance-level one if there is none (when called in sequential mode)
        _stop = local_stop if local_stop is not None else self.stop_event
        all_results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(strategies)) as executor:
            futures = {}

            # Submit all tasks
            for img, method_name, preprocess, stop_evt, inv_transforms in strategies:
                future = executor.submit(self.decode_with_stop_check,
                                         img, f"parallel-{method_name}", preprocess, stop_evt)
                futures[future] = (method_name, inv_transforms)

            def _collect_future(future):
                """Collect the result of a single future (including the inverse transform chain mapping)"""
                method_name, inv_transforms = futures[future]
                try:
                    results = future.result()
                except Exception as e:
                    logging.debug(f"{method_name} error: {e}")
                    return
                if results:
                    results = self._apply_inv_transforms(results, inv_transforms)
                    elapsed = (time.time() - start_time) * 1000
                    logging.info(f"✅ {method_name} found {len(results)} code(s), elapsed {elapsed:.0f}ms")
                    all_results.extend(results)
                else:
                    elapsed = (time.time() - start_time) * 1000
                    logging.info(f"❌ {method_name} failed, elapsed {elapsed:.0f}ms")

            # Wait for all the results
            try:
                for future in concurrent.futures.as_completed(futures, timeout=self.timeout_ms / 1000):
                    _collect_future(future)

            except concurrent.futures.TimeoutError:
                # as_completed timed out, but some strategies are still running; wait for them to finish and collect the results
                pending_names = [name for f, (name, _) in futures.items() if not f.done()]
                logging.warning(f"Parallel decoding exceeded {self.timeout_ms}ms, waiting for the unfinished strategies: {pending_names}")
                remaining = {f for f in futures if not f.done()}
                if remaining:
                    # Wait for the remaining strategies to finish (at most 5 more seconds)
                    done_set, not_done = concurrent.futures.wait(remaining, timeout=5.0)
                    for future in done_set:
                        _collect_future(future)
                    if not_done:
                        still_pending = [futures[f][0] for f in not_done]
                        logging.error(f"These strategies are still unfinished: {still_pending}")

            except Exception as e:
                logging.error(f"Parallel decoding error: {e}")

            finally:
                # Log a summary of all the strategies
                total = len(futures)
                done = sum(1 for f in futures if f.done())
                logging.info(f"Strategy summary: {total} in total, {done} done, "
                            f"{len(all_results)} code(s) found")
                _stop.set()
                # Compatible with Python 3.8 (the cancel_futures parameter is only supported in 3.9+)
                if sys.version_info >= (3, 9):
                    executor.shutdown(wait=False, cancel_futures=True)
                else:
                    executor.shutdown(wait=False)

        return all_results

    # ========== Hybrid mode (recommended) ==========

    def decode(self, image_path: str, mode: str = "auto") -> List[CodeResult]:
        """
        Unified decoding interface

        Args:
            image_path: image path
            mode: "sequential" (accuracy first), "parallel" (speed first), "auto" (chosen automatically)

        Returns:
            List of CodeResult, an empty list on failure
        """
        # Read the image to get its basic information
        image = cv2.imread(image_path)
        logging.info(f"{image_path}")
        if image is None:
            return []
        return self.decode_image(image, mode)

    def decode_image(self, image: np.ndarray, mode: str = "auto") -> List[CodeResult]:
        # Auto mode: choose by the size and complexity of the image
        if mode == "auto":
            # Large image (many pixels, lots of detail): the parallel strategies can try several transforms at once, which is faster overall
            # Small image (few pixels, simple scene): the sequential mode has less overhead and avoids the extra cost of threads
            if image.shape[0] * image.shape[1] >= 500 * 500:
                mode = "parallel"
            else:
                mode = "sequential"
            logging.info(f"Mode chosen automatically: {mode}")

        if mode == "sequential":
            return self.decode_sequential(image)
        else:
            return self.decode_parallel(image)


# ========== Convenience function ==========

def decode_barcode(image_path: str, mode: str = "auto", timeout_ms: int = 500,
                   annotate_path: Optional[str] = None,
                   debug_dir: Optional[str] = None) -> List[CodeResult]:
    """
    Convenience decoding function

    Args:
        image_path: image path
        mode: "sequential", "parallel", "auto"
        timeout_ms: timeout (milliseconds)
        annotate_path: path to save the annotated image, nothing is saved when None
        debug_dir: directory for the debug images, nothing is saved when None

    Returns:
        List of CodeResult, an empty list on failure
    """
    decoder = QrBarDecoder(timeout_ms=timeout_ms, annotate_path=annotate_path, debug_dir=debug_dir)
    with decoder:
        return decoder.decode(image_path, mode=mode)


# ========== Formatted output ==========

def format_result(r: CodeResult, index: int = 0) -> str:
    """Format the output of a single code"""
    x, y, w, h = r.rect
    lines = []
    prefix = f"  Code {index}" if index else "  Code"
    lines.append(f"{prefix}: [{r.barcode_type}] {r.data}")
    lines.append(f"    Coordinates: ({x}, {y}, {w}, {h})")
    lines.append(f"    Confidence: {r.confidence}")
    return "\n".join(lines)


# ========== Tests and examples ==========

def test_parallel_stop_mechanism():
    """Test the parallel stop mechanism"""
    print("\n" + "="*60)
    print("Test of parallel decoding and the active stop mechanism")
    print("="*60)

    # Create a simple test image (generated by code here)
    def create_test_barcode():
        """Create a simple test barcode image"""
        import numpy as np
        # Create a simulated barcode (a real image should be used in practice)
        img = np.ones((200, 400, 3), dtype=np.uint8) * 255

        # Draw simple barcode lines
        for i in range(10, 390, 20):
            cv2.rectangle(img, (i, 50), (i+10, 150), (0, 0, 0), -1)

        return img

    # Save the test image
    test_img = create_test_barcode()
    temp_path = "test_barcode_temp.png"
    cv2.imwrite(temp_path, test_img)

    try:
        # Test the parallel mode
        print("\nTest 1: parallel mode (speed first)")
        start = time.time()
        results = decode_barcode(temp_path, mode="parallel", timeout_ms=500)
        elapsed = (time.time() - start) * 1000
        for j, r in enumerate(results, 1):
            print(format_result(r, j))
        print(f"Time: {elapsed:.0f}ms")

        # Test the sequential mode
        print("\nTest 2: sequential mode (accuracy first)")
        start = time.time()
        results = decode_barcode(temp_path, mode="sequential", timeout_ms=500)
        elapsed = (time.time() - start) * 1000
        for j, r in enumerate(results, 1):
            print(format_result(r, j))
        print(f"Time: {elapsed:.0f}ms")

        # Test the auto mode
        print("\nTest 3: auto mode")
        start = time.time()
        results = decode_barcode(temp_path, mode="auto", timeout_ms=500)
        elapsed = (time.time() - start) * 1000
        for j, r in enumerate(results, 1):
            print(format_result(r, j))
        print(f"Time: {elapsed:.0f}ms")

    finally:
        # Clean up the temporary files
        import os
        if os.path.exists(temp_path):
            os.remove(temp_path)


def test_stop_mechanism_demo():
    """Demonstrate how the stop mechanism works"""
    print("\n" + "="*60)
    print("Stop mechanism demo")
    print("="*60)

    from concurrent.futures import ThreadPoolExecutor
    import threading

    stop_flag = threading.Event()
    results = []

    def slow_task(name: str, delay: float, success_step: int):
        """Simulate a slow task"""
        for i in range(10):
            if stop_flag.is_set():
                print(f"  {name} received the stop signal and exited at step {i}")
                return None
            time.sleep(delay)
            if i == success_step:
                print(f"  ✅ {name} succeeded at step {i}")
                return f"{name} result"
        return None

    print("\nSimulate 3 parallel tasks; the others stop immediately after the first one succeeds:")

    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {
            executor.submit(slow_task, "Task A (fast)", 0.01, 2): "A",
            executor.submit(slow_task, "Task B (slow)", 0.05, 8): "B",
            executor.submit(slow_task, "Task C (medium)", 0.03, 5): "C",
        }

        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            if result:
                print(f"\n🎉 Got the result: {result}")
                stop_flag.set()  # Stop the other tasks

                # Cancel the unfinished tasks
                for f in futures:
                    if not f.done():
                        f.cancel()
                break

    print("\nStop mechanism demo finished")


if __name__ == "__main__":
    import sys

    IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif', '.webp', '.gif'}

    def decode_path(path: str, mode: str, annotate_dir: Optional[str] = None,
                    debug_dir: Optional[str] = None):
        """Decode a single path (a file or a directory)"""
        if os.path.isdir(path):
            files = sorted([
                os.path.join(path, f) for f in os.listdir(path)
                if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS
            ])
            if not files:
                print(f"No image files found in the directory: {path}")
                return
            print(f"Found {len(files)} image file(s) in the directory {path}\n")
            success_count = 0
            fail_count = 0
            tt = time.time()
            for i, fp in enumerate(files, 1):
                print(f"[{i}/{len(files)}] {os.path.basename(fp)}")
                ann_path = _build_annotate_path(fp, annotate_dir) if annotate_dir else None
                t0 = time.time()
                results = decode_barcode(fp, mode=mode, timeout_ms=500, annotate_path=ann_path,
                                         debug_dir=debug_dir)
                elapsed_ms = (time.time() - t0) * 1000
                if results:
                    for j, r in enumerate(results, 1):
                        print(format_result(r, j))
                    success_count += 1
                else:
                    print(f"  ❌ Decoding failed")
                    fail_count += 1
                print(f"  Time: {elapsed_ms:.0f}ms")
            print(f"\n{'='*60}")
            print(f"Done: {success_count} succeeded, {fail_count} failed, {len(files)} file(s) in total")
            elapsed_ms = (time.time() - tt) * 1000
            print(f"Total time: {elapsed_ms:.0f}ms")

            if annotate_dir:
                print(f"Annotated images saved in: {annotate_dir}")
            print(f"{'='*60}")
        elif os.path.isfile(path):
            ann_path = _build_annotate_path(path, annotate_dir) if annotate_dir else None
            t0 = time.time()
            results = decode_barcode(path, mode=mode, timeout_ms=500, annotate_path=ann_path,
                                     debug_dir=debug_dir)
            elapsed_ms = (time.time() - t0) * 1000
            print(f"\n{'='*60}")
            if results:
                for j, r in enumerate(results, 1):
                    print(format_result(r, j))
                if ann_path:
                    print(f"\nAnnotated image saved: {ann_path}")
            else:
                print(f"❌ Decoding failed")
            print(f"Time: {elapsed_ms:.0f}ms")
            print(f"{'='*60}")
        else:
            print(f"Path does not exist: {path}")


    def _build_annotate_path(image_path: str, annotate_dir: str) -> str:
        """Build the path to save the annotated image from the path of the original image"""
        basename = os.path.splitext(os.path.basename(image_path))[0]
        return os.path.join(annotate_dir, f"{basename}_annotated.png")

    if len(sys.argv) > 1:
        path = sys.argv[1]
        mode = "auto"
        annotate_dir = None
        debug_dir = None

        # Parse the arguments
        args = sys.argv[2:]
        for arg in args:
            if arg in ("sequential", "parallel", "auto"):
                mode = arg
            elif arg.startswith("--annotate="):
                annotate_dir = arg[len("--annotate="):]
            elif arg == "--annotate":
                annotate_dir = os.path.join(os.path.dirname(path) or ".", "annotated")
            elif arg.startswith("--debug="):
                debug_dir = arg[len("--debug="):]
            elif arg == "--debug":
                debug_dir = os.path.join(os.path.dirname(path) or ".", "debug")

        if path == "--test":
            test_parallel_stop_mechanism()
            test_stop_mechanism_demo()
        else:
            # If an annotation directory is given, create it automatically
            if annotate_dir and not os.path.exists(annotate_dir):
                os.makedirs(annotate_dir, exist_ok=True)
            # If a debug directory is given, create it automatically
            if debug_dir and not os.path.exists(debug_dir):
                os.makedirs(debug_dir, exist_ok=True)
            decode_path(path, mode, annotate_dir, debug_dir)
    else:
        print("Usage: python detector_qrcode.py <image path|directory path> [sequential|parallel|auto] [--annotate[=directory]] [--debug[=directory]]")
        print("\nModes:")
        print("  sequential - run one by one (accuracy first, suits complex scenes)")
        print("  parallel   - run in parallel (speed first, suits real-time scenes)")
        print("  auto       - chosen automatically (by the image size)")
        print("\nAnnotation:")
        print("  --annotate          save the annotated images in the annotated subdirectory next to the original image")
        print("  --annotate=directory  save the annotated images in the given directory")
        print("\nDebugging:")
        print("  --debug             save the image of each strategy after processing in the debug subdirectory next to the original image")
        print("  --debug=directory    save the image of each strategy after processing in the given directory")
        print("\nExamples:")
        print("  python detector_qrcode.py test.png")
        print("  python detector_qrcode.py ./images/ parallel --annotate")
        print("  python detector_qrcode.py D:/qrcodes/ --annotate=D:/output/")
        print("  python detector_qrcode.py test.png --debug")
        print("  python detector_qrcode.py ./images/ --debug=./debug_output/")
        print("\nRun the tests:")
        print("  python detector_qrcode.py --test")
