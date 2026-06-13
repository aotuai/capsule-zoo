"""
条码/二维码解码工具 - 完整优化版

Python 版本要求：≥ 3.8

依赖库：
    opencv-python     图像处理和变换
    numpy             矩阵运算
    pyrxing (推荐)     主解码引擎（高性能 QR/条码）, 因 brainframe os 使用python 3.9, 目前无法使用pyrxing (需要 python>=3.11)
    pyzbar (备选)      备选解码引擎
    Pillow             pyzbar 图像转换（pyzbar 路径必需）

特性：
1. 渐进式处理流程（快速路径优先）
2. 并行解码 + 主动停止机制
3. 智能透视校正
4. 完整的资源管理
5. 多码解码，输出类型和坐标
6. 图形标注输出
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
# 尝试导入解码库
# PIL/Pillow 是两个引擎都需要的依赖：
#   - pyrxing：read_barcodes 只接受 str 或 ImageProtocol，numpy.ndarray 需先转为 PIL Image
#   - pyzbar：直接使用 PIL Image 输入
try:
    from PIL import Image as PILImage
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    logging.warning("PIL/Pillow 未安装，解码引擎将降级或不可用")

try:
    import pyrxing as rx
    PYRXING_AVAILABLE = True
    PYZBAR_AVAILABLE = False
    if not PIL_AVAILABLE:
        logging.warning("PIL/Pillow 未安装，pyrxing 无法接收 numpy 数组输入")
    logging.info("使用 pyrxing 解码引擎")
except ImportError:
    PYRXING_AVAILABLE = False
    try:
        from pyzbar.pyzbar import decode as zbar_decode
        PYZBAR_AVAILABLE = True
        if not PIL_AVAILABLE:
            logging.warning("PIL/Pillow 未安装，pyzbar 路径将不可用")
        logging.info("使用 pyzbar 解码引擎")
    except ImportError as e:
        PYZBAR_AVAILABLE = False
        err_msg = f"Please install pyrxing or pyzbar: {e}"
        raise ImportError(err_msg)


class DecodeStatus(Enum):
    """解码状态"""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    STOPPED = "stopped"


@dataclass
class CodeResult:
    """单个码的解码结果"""
    data: str                           # 解码内容
    barcode_type: str                   # 码类型（QR_CODE, EAN_13 等）
    rect: Tuple[int, int, int, int]     # 外接矩形 (x, y, w, h)
    points: List[Tuple[int, int]] = field(default_factory=list)  # 多边形顶点 [(x,y), ...]
    confidence: Optional[float] = None  # 置信度 (0.0~1.0)，解码引擎未提供时为 None

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
    """可停止的解码器 - 支持主动停止其他线程"""

    def __init__(self, timeout_ms: int = 500, check_interval_ms: int = 10,
                 annotate_path: Optional[str] = None, debug_dir: Optional[str] = None):
        """
        Args:
            timeout_ms: 整体超时时间（毫秒）
            check_interval_ms: 停止检查间隔（毫秒）
            annotate_path: 解码图像输出目录，为 None 时不保存
            debug_dir: 调试图像保存目录，为 None 时不保存
        """
        self.timeout_ms = timeout_ms
        self.check_interval_ms = check_interval_ms
        self.stop_event = threading.Event()
        self.executor = None
        self.annotate_path = annotate_path
        self.debug_dir = debug_dir
        self._debug_counter = 0

    def __enter__(self) -> "QrBarDecoder":
        """上下文管理器入口，支持 with QrBarDecoder() as decoder: 语法"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """上下文管理器出口：确保资源清理（线程、事件等）"""
        self.stop_event.set()
        # 抑制所有异常，交由调用方决定如何处理
        return False

    def _save_debug_image(self, image: np.ndarray, method_name: str):
        """保存调试图像，支持中文路径（cv2.imwrite 在 Windows 下对非 ASCII 路径有编码问题，改用 imencode + Python 原生写入）"""
        self._debug_counter += 1
        safe_name = method_name.replace(" ", "_").replace("/", "_").replace("\\", "_")
        filename = f"{self._debug_counter:02d}_{safe_name}.png"
        debug_path = os.path.join(self.debug_dir, filename)
        try:
            success, buf = cv2.imencode('.png', image)
            if success:
                with open(debug_path, 'wb') as f:
                    f.write(buf.tobytes())
                logging.debug(f"调试图已保存: {debug_path}")
            else:
                logging.debug(f"调试图编码失败: {method_name}")
        except Exception as e:
            logging.debug(f"保存调试图失败: {e}")

    # ========== 图像预处理 ==========

    def _quick_preprocess(self, image: np.ndarray) -> np.ndarray:
        """快速预处理（~5-10ms）"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # 快速直方图均衡化
        gray = cv2.equalizeHist(gray)
        return gray

    def _full_preprocess(self, image: np.ndarray) -> np.ndarray:
        """完整预处理（~20-30ms）"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # 去噪
        gray = cv2.bilateralFilter(gray, 9, 75, 75)

        # CLAHE对比度增强
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        gray = clahe.apply(gray)

        return gray

    @staticmethod
    def _identity_preprocess(image: np.ndarray) -> np.ndarray:
        """恒等预处理（图像已预处理过，直接返回）"""
        return image

    # ========== 核心解码函数 ==========

    def _decode_core(self, image: np.ndarray) -> List[CodeResult]:
        """
        核心解码逻辑，返回所有解码结果（含类型和坐标）
        引擎选择由 PYRXING_AVAILABLE / PYZBAR_AVAILABLE 常量控制，
        无需外部参数指定——pyrxing 优先，失败自动 fallback 到 pyzbar。
        """
        # 转换为RGB（pyrxing / pyzbar 均需要 RGB 输入）
        if len(image.shape) == 3:
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        results = []

        # 优先使用 pyrxing
        # pyrxing.read_barcodes 只接受 str（路径）或符合 ImageProtocol 的对象（如 PIL Image）
        # numpy.ndarray 不满足 ImageProtocol（缺少 .width/.height/.convert/.load），
        # 必须先转为 PIL Image，否则抛出 ValueError('value must be either str or conform to ImageProtocol')
        if PYRXING_AVAILABLE and PIL_AVAILABLE:
            logging.info(f"解码使用pyrxing")
            try:
                pil_for_rx = PILImage.fromarray(rgb)
                barcodes = rx.read_barcodes(pil_for_rx)
                for bc in barcodes:
                    # 兼容不同 pyrxing 版本的属性名
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

                    # 提取坐标点
                    pts = []
                    if hasattr(bc, 'points') and bc.points:
                        pts = [(int(p.x), int(p.y)) for p in bc.points]
                    elif hasattr(bc, 'position') and bc.position:
                        pts = [(int(p.x), int(p.y)) for p in bc.position]

                    # 计算 rect
                    if pts:
                        xs = [p[0] for p in pts]
                        ys = [p[1] for p in pts]
                        x_min, x_max = min(xs), max(xs)
                        y_min, y_max = min(ys), max(ys)
                        rect = (x_min, y_min, x_max - x_min, y_max - y_min)
                    else:
                        rect = (0, 0, 0, 0)

                    # 提取置信度（部分 pyrxing 版本支持）
                    conf = None
                    if hasattr(bc, 'confidence'):
                        conf = float(bc.confidence)

                    results.append(CodeResult(data=data, barcode_type=btype,
                                              rect=rect, points=pts, confidence=conf))
            except Exception as e:
                logging.debug(f"pyrxing解码失败: {e}")

        # pyrxing 无结果 或 不可用时，尝试 pyzbar
        if not results and PYZBAR_AVAILABLE and PIL_AVAILABLE:
            logging.info(f"解码使用pyzbar")
            try:
                pil_img = PILImage.fromarray(rgb)
                decoded = zbar_decode(pil_img)
                for d in decoded:
                    data = d.data.decode('utf-8', errors='ignore')
                    btype = d.type

                    # pyzbar 提供 rect 和 polygon
                    r = d.rect
                    rect = (r.left, r.top, r.width, r.height)

                    pts = [(p.x, p.y) for p in d.polygon]

                    # pyzbar 的 quality 字段表示解码质量（整数，条码类型相关）
                    conf = float(d.quality) if hasattr(d, 'quality') else None

                    results.append(CodeResult(data=data, barcode_type=btype,
                                              rect=rect, points=pts, confidence=conf))
            except Exception as e:
                logging.debug(f"pyzbar解码失败: {e}")

        return results

    def decode_with_stop_check(self,
                               image: np.ndarray,
                               method_name: str,
                               preprocess_func: Callable[[np.ndarray], np.ndarray],
                               stop_event: Optional[threading.Event] = None) -> List[CodeResult]:
        """
        带停止检查的解码任务，返回所有解码结果

        预处理函数由 preprocess_func 指定（快速/完整预处理），
        解码引擎由 PYRXING_AVAILABLE / PYZBAR_AVAILABLE 自动选择，无需外部参数控制。

        Args:
            image: 输入图像
            method_name: 方法名称（用于日志）
            preprocess_func: 预处理函数，签名为 (np.ndarray) -> np.ndarray
            stop_event: 外部停止信号（为 None 时使用实例级 stop_event）
        """
        # 优先使用传入的局部 stop_event，避免实例共享的竞态条件
        _stop = stop_event if stop_event is not None else self.stop_event
        start_time = time.time()

        # 检查停止标志
        if _stop.is_set():
            logging.debug(f"{method_name} 任务启动时已收到停止信号")
            return []

        # 预处理
        try:
            processed = preprocess_func(image)
        except Exception as e:
            logging.debug(f"{method_name} 预处理失败: {e}")
            return []

        # 调试：保存各策略处理后的图像（放在 stop check 之前，确保并行模式下每个策略的图像都能保存）
        if self.debug_dir:
            self._save_debug_image(processed, method_name)

        # 再次检查停止标志
        if _stop.is_set():
            logging.debug(f"{method_name} 预处理后收到停止信号")
            return []

        # 解码：引擎由 _decode_core 内部根据可用库自动选择
        results = self._decode_core(processed)

        elapsed_ms = (time.time() - start_time) * 1000

        if results:
            logging.info(f"✅ {method_name} 成功，发现 {len(results)} 个码，耗时 {elapsed_ms:.0f}ms")
            return results

        logging.debug(f"{method_name} 失败，耗时 {elapsed_ms:.0f}ms")
        return []

    # ========== 透视校正 ==========

    def _needs_perspective_correction(self, image: np.ndarray) -> bool:
        """
        快速判断是否需要透视校正（~2ms）

        检测图像中是否存在明显的"梯形畸变"特征——即存在方向差异显著的两组
        近似平行线段（如倾斜放置的文档/卡片的四边）。
        曲线边缘（如眼镜框）产生的零散线段不满足"两组平行线"特征，会被过滤。

        返回 True 的条件（需同时满足）：
        1. 检测到至少 5 条有效线段
        2. 角度标准差 > 25°（比原来 15° 更严格，排除零散曲线边缘）
        3. 存在至少两个明显不同的主导角度簇（每组至少 2 条线）
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # 下采样加速
        if gray.shape[0] > 400:
            scale = 400 / gray.shape[0]
            gray = cv2.resize(gray, None, fx=scale, fy=scale)

        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=50)

        if lines is None or len(lines) < 5:
            return False

        # 收集所有角度，映射到 [0, 90) 范围（线条方向无正反）
        # HoughLines 的 theta 在 [0, π) 范围，转换后：
        # - theta ∈ [0, π/2) → angle ∈ [0°, 90°)
        # - theta ∈ [π/2, π) → angle ∈ [90°, 180°) → 等价于 [0°, 90°)
        angles = []
        for line in lines[:20]:
            theta = line[0][1]
            angle = theta * 180 / np.pi
            # 映射到 [0, 90)：线条方向无正反，0°和180°等价，-87°和87°等价
            if angle >= 90:
                angle = 180 - angle
            angles.append(angle)

        if len(angles) < 5:
            return False

        # 条件1：角度分散度必须足够大
        # 注意：标准差只是粗筛，真正的保护由 _perspective_correct 内部的
        # 矩形度、顶点数、面积等过滤条件实现，所以这里阈值不宜过高
        if np.std(angles) <= 15:
            return False

        # 条件2：必须存在至少两个明显不同的角度簇（梯形有两组对边）
        # 分组："接近水平"（< 45°）和"接近垂直"（>= 45°）
        horizontal = [a for a in angles if a < 45]
        vertical = [a for a in angles if a >= 45]

        if len(horizontal) < 2 or len(vertical) < 2:
            return False

        # 两组的角度中位数差异必须足够大
        med_h = float(np.median(horizontal))
        med_v = float(np.median(vertical))
        angle_diff = abs(med_v - med_h)

        if angle_diff <= 30:
            return False

        # 条件3：排除精确的正视图矩形（水平线≈0° 且 垂直线≈90°）
        # 如果两组线都精确对齐坐标轴，说明是正视图，不需要透视校正
        h_aligned = all(a < 5 for a in horizontal)
        v_aligned = all(a > 85 for a in vertical)
        if h_aligned and v_aligned:
            return False

        return True

    def _perspective_correct(self, image: np.ndarray,
                             return_matrix: bool = False) -> Any:
        """
        透视校正（~30-50ms）

        寻找图像中具有明显透视畸变的矩形区域（如倾斜放置的文件/卡片），
        并将其校正为正视图。
        若未找到有效的目标区域，返回 None（而非扭曲原图）。

        有效性约束（防止把整图边缘、背景色块、椭圆/圆形物体误判为目标）：
        1. 轮廓面积 < 图像面积的 85%（排除"整图轮廓"）
        2. 轮廓面积 > 图像面积的 8%（排除噪点和小条码区域，校正目标应为文档/卡片等大区域）
        3. 宽高比在 [0.15, 6.5] 范围内（排除极度扁平/细长的误检）
        4. 矩形度（轮廓面积 / minAreaRect 面积）：4顶点时≥0.60，5-6顶点时≥0.82
        5. 近似多边形顶点数 ≤ 6（排除椭圆/圆形/不规则形状）

        Args:
            return_matrix: 若为 True，返回 (校正图像, 透视矩阵)；否则只返回校正图像
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        img_h, img_w = image.shape[:2]
        img_area = img_h * img_w

        # 下采样加速
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

        # 自适应阈值
        binary = cv2.adaptiveThreshold(gray_small, 255,
                                       cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, 11, 5)

        # 形态学操作
        kernel = np.ones((5, 5), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        # 查找轮廓（取面积最大的前5个候选）
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

        for contour in contours:
            area = cv2.contourArea(contour)
            # 换算回原始尺度的面积
            real_area = area / (scale * scale)

            # 约束1：面积过小（噪点/小条码区域）或过大（整图边缘）均排除
            # 面积过小的矩形做透视校正没有意义（校正目标应该是文档/卡片等大区域）
            if real_area < img_area * 0.08 or real_area > img_area * 0.85:
                logging.debug(f"透视校正：跳过轮廓（面积比 {real_area/img_area:.2%}）")
                continue

            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

            if len(approx) < 4:
                continue

            # 约束0：近似多边形顶点数检查
            # 矩形应为 4 个顶点；顶点数过多（>6）说明是椭圆/圆形/不规则形状
            if len(approx) > 6:
                logging.debug(f"透视校正：跳过轮廓（顶点数 {len(approx)}，疑似椭圆/圆形）")
                continue

            # 获取最小外接旋转矩形
            rect = cv2.minAreaRect(contour)
            (cx, cy), (rw, rh), angle = rect

            # 约束3：宽高比过滤（排除极度细长/扁平的误检）
            if rw == 0 or rh == 0:
                continue
            aspect = max(rw, rh) / min(rw, rh)
            if aspect > 6.5 or aspect < 0.15:
                logging.debug(f"透视校正：跳过轮廓（宽高比 {aspect:.2f}，超出合理范围）")
                continue

            # 约束4：矩形度检查（轮廓面积 / 最小外接矩形面积）
            # 矩形物体比值接近 1.0；梯形约 0.6~0.95；椭圆≈0.785，圆形≈0.785
            # 矩形度+顶点数联合判断：
            #   - 4顶点 + 矩形度>=0.60 → 矩形/梯形（保留）
            #   - 5-6顶点 + 矩形度>=0.82 → 圆角矩形（保留）
            #   - 其他 → 椭圆/不规则（排除）
            # 注意：必须用同尺度面积比（都以下采样后尺度计算）
            min_rect_area = rw * rh
            if min_rect_area <= 0:
                continue
            rect_extent = area / min_rect_area  # area 和 rw*rh 都是下采样后的值
            if len(approx) == 4 and rect_extent < 0.60:
                logging.debug(f"透视校正：跳过轮廓（矩形度 {rect_extent:.2f}，4顶点但矩形度过低）")
                continue
            elif len(approx) > 4 and rect_extent < 0.82:
                logging.debug(f"透视校正：跳过轮廓（矩形度 {rect_extent:.2f}，{len(approx)}顶点，疑似椭圆/圆形）")
                continue

            box = cv2.boxPoints(rect)

            if scale < 1.0:
                box = box / scale

            logging.debug(f"透视校正：找到有效区域，面积比 {real_area/img_area:.2%}，角度 {angle:.1f}°，矩形度 {rect_extent:.2f}")
            corrected, matrix = self._apply_perspective_transform(image, box.astype(np.float32))
            if return_matrix:
                return corrected, matrix
            return corrected

        logging.debug("透视校正：未找到满足条件的目标区域，跳过")
        if return_matrix:
            return None, None
        return None

    def _apply_perspective_transform(self, image: np.ndarray, points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """应用透视变换，返回 (校正后图像, 透视矩阵)"""
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
        """排序四个角点"""
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
        检测图像中线条的主导角度
        使用直方图聚类，找到最密集的角度区间，取中位数
        返回角度（度），或 None
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # 下采样
        if gray.shape[0] > 500:
            scale = 500 / gray.shape[0]
            small = cv2.resize(gray, None, fx=scale, fy=scale)
        else:
            small = gray

        edges = cv2.Canny(small, 50, 150)
        lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=80)

        if lines is None:
            return None

        # 收集所有角度，映射到 [-90, 90) 范围
        angles = []
        for line in lines[:max_lines]:
            theta = line[0][1]
            angle = theta * 180 / np.pi
            if angle >= 90:
                angle -= 180
            angles.append(angle)

        if len(angles) < 2:
            return None

        # 按角度排序，找最大间隔作为聚类边界
        angles.sort()
        n = len(angles)
        max_gap = -1
        max_gap_idx = 0
        for i in range(n):
            next_i = (i + 1) % n
            gap = angles[next_i] - angles[i]
            if next_i == 0:
                gap += 180  # 环绕
            if gap > max_gap:
                max_gap = gap
                max_gap_idx = next_i

        # 从最大间隔处开始，取连续的多数角度作为主聚类
        sorted_from_gap = angles[max_gap_idx:] + angles[:max_gap_idx]
        half = max(1, n // 2)
        main_cluster = sorted_from_gap[:half]

        return float(np.median(main_cluster))

    def _rotate_correct(self, image: np.ndarray) -> np.ndarray:
        """平面旋转校正，支持任意角度"""
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

    # ========== 策略构建与逆变换 ==========

    def _build_strategies(self, image: np.ndarray,
                          stop_event: Optional[threading.Event] = None) -> List[tuple]:
        """
        构建统一解码策略列表，供顺序/并行模式共用

        Returns:
            策略列表，每项: (image, method_name, preprocess_func, stop_event, inv_transforms)
            inv_transforms: 逆变换链 [(M_inv, is_perspective), ...]，逆序应用
        """
        _stop = stop_event if stop_event is not None else self.stop_event
        strategies = []

        # 策略1：快速预处理（原图，无需坐标转换）
        strategies.append((
            image, "快速解码", self._quick_preprocess, _stop, []
        ))

        # 策略2：完整预处理（原图，无需坐标转换）
        strategies.append((
            image, "完整解码", self._full_preprocess, _stop, []
        ))

        # 预计算透视校正（策略3/5共用）
        # 直接尝试透视校正，由 _perspective_correct 内部的过滤条件
        # （面积、矩形度、顶点数等）决定是否需要校正，无需前置判断
        persp_img = None
        M_persp_inv = None
        persp_img, M_persp = self._perspective_correct(image, return_matrix=True)
        if persp_img is not None and M_persp is not None:
            M_persp_inv = np.linalg.inv(M_persp)

            # 策略3：透视校正 + 完整预处理
            strategies.append((
                persp_img, "透视校正", self._full_preprocess, _stop,
                [(M_persp_inv, True)]
            ))

        # 预计算检测角旋转（策略4/5共用）
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

            # 策略4：检测角旋转 + 完整预处理
            strategies.append((
                rot_img, f"检测角旋转{detected_angle:.1f}度", self._full_preprocess, _stop,
                [(M_rot_inv, False)]
            ))

        # 策略5：透视+旋转 组合
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
                # 逆变换链: 先逆旋转，再逆透视
                strategies.append((
                    persp_rot_img, "透视+旋转", self._full_preprocess, _stop,
                    [(M_protf_inv, False), (M_persp_inv, True)]
                ))

        # 策略6-10：固定角度旋转（对快速预处理后的灰度图旋转）
        quick_gray = self._quick_preprocess(image)
        for angle in [45, 90, 180, 270, 315]:
            h, w = quick_gray.shape[:2]
            center = (w // 2, h // 2)
            matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(quick_gray, matrix, (w, h))
            M_rot_inv = cv2.invertAffineTransform(matrix)
            strategies.append((
                rotated, f"旋转{angle}度", self._identity_preprocess, _stop,
                [(M_rot_inv, False)]
            ))

        return strategies

    @staticmethod
    def _apply_inv_transforms(results: List[CodeResult],
                               inv_transforms: List[Tuple[np.ndarray, bool]]) -> List[CodeResult]:
        """
        按逆变换链逆序应用坐标映射

        inv_transforms: [(M_inv, is_perspective), ...]
        应用顺序: 链的最后一项先应用（逆序），因为变换是正序记录的
        例如: 原图 → 透视 → 旋转 → 解码; inv = [(M_rot_inv, F), (M_persp_inv, T)]
        逆映射: 先逆旋转，再逆透视
        """
        if not inv_transforms or not results:
            return results
        for M_inv, is_perspective in reversed(inv_transforms):
            results = [QrBarDecoder._inverse_map_result(r, M_inv, is_perspective)
                       for r in results]
        return results

    # ========== 标注绘制 ==========

    @staticmethod
    def _inverse_map_result(result: CodeResult, M_inv: np.ndarray,
                            is_perspective: bool = False) -> CodeResult:
        """将结果坐标通过逆变换映射回原图坐标系"""
        if M_inv is None:
            return result

        # 逆映射多边形顶点
        new_points = []
        if result.points:
            pts = np.array([result.points], dtype=np.float32)
            if is_perspective:
                mapped = cv2.perspectiveTransform(pts, M_inv)
            else:
                mapped = cv2.transform(pts, M_inv)
            new_points = [(int(round(p[0])), int(round(p[1]))) for p in mapped[0]]

        # 逆映射 rect 四角，重新计算外接矩形
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
        """在图像上绘制码的标注框、类型和内容"""
        annotated = image.copy()
        colors = [
            (0, 0, 255),    # 红
            (255, 0, 0),    # 蓝
            (0, 165, 255),  # 橙
            (255, 0, 255),  # 紫
            (0, 255, 255),  # 黄
        ]

        for i, r in enumerate(results):
            color = colors[i % len(colors)]
            x, y, w, h = r.rect

            # 绘制外接矩形
            cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)

            # 标签文字：类型 + 内容（截断）
            label_text = f"[{r.barcode_type}] {r.data}"
            if len(label_text) > 60:
                label_text = label_text[:57] + "..."

            # 计算文字位置和大小
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            thickness = 1
            (tw, th), baseline = cv2.getTextSize(label_text, font, font_scale, thickness)

            # 标签背景
            label_y = max(y - 6, th + 6)
            cv2.rectangle(annotated,
                          (x, label_y - th - 6),
                          (x + tw + 4, label_y + baseline),
                          color, -1)
            cv2.putText(annotated, label_text,
                        (x + 2, label_y - 4),
                        font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

        return annotated

    # ========== 顺序解码（渐进式） ==========

    def decode_sequential(self, image: np.ndarray) -> List[CodeResult]:
        """
        顺序解码（精度优先）
        逐个执行策略，首个成功即停
        返回所有解码到的码结果列表

        Args:
            image: 图片
        """
        start_time = time.time()
        logging.info("顺序模式解码")

        strategies = self._build_strategies(image)
        results = []

        for i, (img, method_name, preprocess, _stop, inv_transforms) in enumerate(strategies, 1):
            level_name = f"级别{i}-{method_name}"
            logging.info(f"级别{i}: {method_name}...")

            level_results = self.decode_with_stop_check(img, level_name, preprocess)
            if level_results:
                # 应用逆变换链
                level_results = self._apply_inv_transforms(level_results, inv_transforms)
                results = level_results
                elapsed = (time.time() - start_time) * 1000
                logging.info(f"✅ 级别{i}成功，发现 {len(results)} 个码，总耗时 {elapsed:.0f}ms")
                break

        elapsed = (time.time() - start_time) * 1000
        if not results:
            logging.error(f"所有级别均失败, 总耗时 {elapsed:.0f}ms")

        logging.info(f"顺序解码总耗时: {elapsed:.0f}ms")

        # 保存标注图
        if self.annotate_path and results:
            self._save_annotation(image, results)

        return results

    # ========== 并行解码（速度优先） ==========

    def decode_parallel(self, image: np.ndarray) -> List[CodeResult]:
        """
        并行解码（速度优先）
        多个策略同时进行，收集所有成功结果（去重）

        Args:
            image: 图片
        """
        start_time = time.time()
        logging.info("并行模式解码")

        # 每次调用创建独立的局部停止标志，避免实例共享导致的竞态条件
        local_stop = threading.Event()

        # 构建统一策略列表
        strategies = self._build_strategies(image, stop_event=local_stop)

        # 使用线程池并行执行
        all_raw = self._run_parallel_strategies(strategies, start_time, local_stop)

        # 去重（按 data 内容）
        seen = set()
        unique = []
        for r in all_raw:
            if r.data not in seen:
                seen.add(r.data)
                unique.append(r)

        elapsed = (time.time() - start_time) * 1000
        logging.info(f"并行解码总耗时: {elapsed:.0f}ms")

        # 保存标注图
        if self.annotate_path and unique:
            self._save_annotation(image, unique)

        return unique

    def _save_annotation(self, image: np.ndarray, results: List[CodeResult]):
        """保存标注图"""
        try:
            annotated = self._draw_annotations(image, results)
            cv2.imwrite(self.annotate_path, annotated)
            logging.info(f"标注图已保存: {self.annotate_path}")
        except Exception as e:
            logging.error(f"保存标注图失败: {e}")

    def _run_parallel_strategies(self, strategies: List[tuple], start_time: float,
                                  local_stop: Optional[threading.Event] = None) -> List[CodeResult]:
        """
        并行执行多个解码策略
        收集所有成功结果（不同策略可能发现不同的码）
        自动通过逆变换链将坐标映射回原图坐标系

        Args:
            strategies: _build_strategies 返回的策略列表
                        每项: (image, method_name, preprocess_func, stop_event, inv_transforms)
            start_time: 计时起点
            local_stop: 局部停止事件（由 decode_parallel 传入，避免竞态条件）
        """
        # 使用传入的局部 stop，若无则回退到实例级（顺序模式调用时）
        _stop = local_stop if local_stop is not None else self.stop_event
        all_results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(strategies)) as executor:
            futures = {}

            # 提交所有任务
            for img, method_name, preprocess, stop_evt, inv_transforms in strategies:
                future = executor.submit(self.decode_with_stop_check,
                                         img, f"并行-{method_name}", preprocess, stop_evt)
                futures[future] = (method_name, inv_transforms)

            def _collect_future(future):
                """收集单个 future 的结果（含逆变换链映射）"""
                method_name, inv_transforms = futures[future]
                try:
                    results = future.result()
                except Exception as e:
                    logging.debug(f"{method_name} 异常: {e}")
                    return
                if results:
                    results = self._apply_inv_transforms(results, inv_transforms)
                    elapsed = (time.time() - start_time) * 1000
                    logging.info(f"✅ {method_name} 发现 {len(results)} 个码，已耗时 {elapsed:.0f}ms")
                    all_results.extend(results)
                else:
                    elapsed = (time.time() - start_time) * 1000
                    logging.info(f"❌ {method_name} 失败，已耗时 {elapsed:.0f}ms")

            # 等待所有结果
            try:
                for future in concurrent.futures.as_completed(futures, timeout=self.timeout_ms / 1000):
                    _collect_future(future)

            except concurrent.futures.TimeoutError:
                # as_completed 超时，但仍有策略在运行；等待其完成并收集结果
                pending_names = [name for f, (name, _) in futures.items() if not f.done()]
                logging.warning(f"并行解码超过 {self.timeout_ms}ms，等待未完成策略: {pending_names}")
                remaining = {f for f in futures if not f.done()}
                if remaining:
                    # 等待剩余策略完成（最多额外等 5 秒）
                    done_set, not_done = concurrent.futures.wait(remaining, timeout=5.0)
                    for future in done_set:
                        _collect_future(future)
                    if not_done:
                        still_pending = [futures[f][0] for f in not_done]
                        logging.error(f"以下策略仍未完成: {still_pending}")

            except Exception as e:
                logging.error(f"并行解码异常: {e}")

            finally:
                # 列出所有策略执行摘要
                total = len(futures)
                done = sum(1 for f in futures if f.done())
                logging.info(f"策略执行汇总: 共 {total} 个, 完成 {done} 个, "
                            f"成功 {len(all_results)} 个码")
                _stop.set()
                # 兼容 Python 3.8（cancel_futures 参数在 3.9+ 才支持）
                if sys.version_info >= (3, 9):
                    executor.shutdown(wait=False, cancel_futures=True)
                else:
                    executor.shutdown(wait=False)

        return all_results

    # ========== 混合模式（推荐） ==========

    def decode(self, image_path: str, mode: str = "auto") -> List[CodeResult]:
        """
        统一解码接口

        Args:
            image_path: 图片路径
            mode: "sequential"（精度优先）, "parallel"（速度优先）, "auto"（自动选择）

        Returns:
            CodeResult 列表，失败返回空列表
        """
        # 读取图片获取基本信息
        image = cv2.imread(image_path)
        logging.info(f"{image_path}")
        if image is None:
            return []
        return self.decode_image(image, mode)

    def decode_image(self, image: np.ndarray, mode: str = "auto") -> List[CodeResult]:
        # 自动模式：根据图片大小和复杂度选择
        if mode == "auto":
            # 大图片（像素多、细节丰富）：并行策略可同时尝试多种变换，整体更快
            # 小图片（像素少、简单场景）：顺序模式开销更低，避免多线程额外损耗
            if image.shape[0] * image.shape[1] >= 500 * 500:
                mode = "parallel"
            else:
                mode = "sequential"
            logging.info(f"自动选择模式: {mode}")

        if mode == "sequential":
            return self.decode_sequential(image)
        else:
            return self.decode_parallel(image)


# ========== 便捷函数 ==========

def decode_barcode(image_path: str, mode: str = "auto", timeout_ms: int = 500,
                   annotate_path: Optional[str] = None,
                   debug_dir: Optional[str] = None) -> List[CodeResult]:
    """
    便捷解码函数

    Args:
        image_path: 图片路径
        mode: "sequential", "parallel", "auto"
        timeout_ms: 超时时间（毫秒）
        annotate_path: 标注图保存路径，为 None 则不保存
        debug_dir: 调试图像保存目录，为 None 则不保存

    Returns:
        CodeResult 列表，失败返回空列表
    """
    decoder = QrBarDecoder(timeout_ms=timeout_ms, annotate_path=annotate_path, debug_dir=debug_dir)
    with decoder:
        return decoder.decode(image_path, mode=mode)


# ========== 格式化输出 ==========

def format_result(r: CodeResult, index: int = 0) -> str:
    """格式化单个码的输出"""
    x, y, w, h = r.rect
    lines = []
    prefix = f"  码{index}" if index else "  码"
    lines.append(f"{prefix}: [{r.barcode_type}] {r.data}")
    lines.append(f"    坐标: ({x}, {y}, {w}, {h})")
    lines.append(f"    置信度: {r.confidence}")
    return "\n".join(lines)


# ========== 测试和示例 ==========

def test_parallel_stop_mechanism():
    """测试并行停止机制"""
    print("\n" + "="*60)
    print("测试并行解码和主动停止机制")
    print("="*60)

    # 创建一个简单的测试图片（这里用代码生成）
    def create_test_barcode():
        """创建一个简单的测试条码图片"""
        import numpy as np
        # 创建一个模拟条码（实际应该用真实图片）
        img = np.ones((200, 400, 3), dtype=np.uint8) * 255

        # 画简单的条码线条
        for i in range(10, 390, 20):
            cv2.rectangle(img, (i, 50), (i+10, 150), (0, 0, 0), -1)

        return img

    # 保存测试图片
    test_img = create_test_barcode()
    temp_path = "test_barcode_temp.png"
    cv2.imwrite(temp_path, test_img)

    try:
        # 测试并行模式
        print("\n测试1: 并行模式（速度优先）")
        start = time.time()
        results = decode_barcode(temp_path, mode="parallel", timeout_ms=500)
        elapsed = (time.time() - start) * 1000
        for j, r in enumerate(results, 1):
            print(format_result(r, j))
        print(f"耗时: {elapsed:.0f}ms")

        # 测试顺序模式
        print("\n测试2: 顺序模式（精度优先）")
        start = time.time()
        results = decode_barcode(temp_path, mode="sequential", timeout_ms=500)
        elapsed = (time.time() - start) * 1000
        for j, r in enumerate(results, 1):
            print(format_result(r, j))
        print(f"耗时: {elapsed:.0f}ms")

        # 测试自动模式
        print("\n测试3: 自动模式")
        start = time.time()
        results = decode_barcode(temp_path, mode="auto", timeout_ms=500)
        elapsed = (time.time() - start) * 1000
        for j, r in enumerate(results, 1):
            print(format_result(r, j))
        print(f"耗时: {elapsed:.0f}ms")

    finally:
        # 清理临时文件
        import os
        if os.path.exists(temp_path):
            os.remove(temp_path)


def test_stop_mechanism_demo():
    """演示停止机制的工作原理"""
    print("\n" + "="*60)
    print("停止机制演示")
    print("="*60)

    from concurrent.futures import ThreadPoolExecutor
    import threading

    stop_flag = threading.Event()
    results = []

    def slow_task(name: str, delay: float, success_step: int):
        """模拟慢速任务"""
        for i in range(10):
            if stop_flag.is_set():
                print(f"  {name} 收到停止信号，在第{i}步退出")
                return None
            time.sleep(delay)
            if i == success_step:
                print(f"  ✅ {name} 成功于第{i}步")
                return f"{name} result"
        return None

    print("\n模拟3个并行任务，第一个成功后其他立即停止：")

    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {
            executor.submit(slow_task, "任务A(快)", 0.01, 2): "A",
            executor.submit(slow_task, "任务B(慢)", 0.05, 8): "B",
            executor.submit(slow_task, "任务C(中)", 0.03, 5): "C",
        }

        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            if result:
                print(f"\n🎉 获得结果: {result}")
                stop_flag.set()  # 停止其他任务

                # 取消未完成的任务
                for f in futures:
                    if not f.done():
                        f.cancel()
                break

    print("\n停止机制演示完成")


if __name__ == "__main__":
    import sys

    IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif', '.webp', '.gif'}

    def decode_path(path: str, mode: str, annotate_dir: Optional[str] = None,
                    debug_dir: Optional[str] = None):
        """解码单个路径（文件或目录）"""
        if os.path.isdir(path):
            files = sorted([
                os.path.join(path, f) for f in os.listdir(path)
                if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS
            ])
            if not files:
                print(f"目录中没有找到图片文件: {path}")
                return
            print(f"在目录 {path} 中找到 {len(files)} 个图片文件\n")
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
                    print(f"  ❌ 解码失败")
                    fail_count += 1
                print(f"  耗时: {elapsed_ms:.0f}ms")
            print(f"\n{'='*60}")
            print(f"完成: 成功 {success_count}, 失败 {fail_count}, 共 {len(files)} 个文件")
            elapsed_ms = (time.time() - tt) * 1000
            print(f"总耗时: {elapsed_ms:.0f}ms")

            if annotate_dir:
                print(f"标注图保存在: {annotate_dir}")
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
                    print(f"\n标注图已保存: {ann_path}")
            else:
                print(f"❌ 解码失败")
            print(f"耗时: {elapsed_ms:.0f}ms")
            print(f"{'='*60}")
        else:
            print(f"路径不存在: {path}")


    def _build_annotate_path(image_path: str, annotate_dir: str) -> str:
        """根据原图路径生成标注图保存路径"""
        basename = os.path.splitext(os.path.basename(image_path))[0]
        return os.path.join(annotate_dir, f"{basename}_annotated.png")

    if len(sys.argv) > 1:
        path = sys.argv[1]
        mode = "auto"
        annotate_dir = None
        debug_dir = None

        # 解析参数
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
            # 如果指定了标注目录，自动创建
            if annotate_dir and not os.path.exists(annotate_dir):
                os.makedirs(annotate_dir, exist_ok=True)
            # 如果指定了调试目录，自动创建
            if debug_dir and not os.path.exists(debug_dir):
                os.makedirs(debug_dir, exist_ok=True)
            decode_path(path, mode, annotate_dir, debug_dir)
    else:
        print("用法: python detector_qrcode.py <图片路径|目录路径> [sequential|parallel|auto] [--annotate[=目录]] [--debug[=目录]]")
        print("\n模式说明:")
        print("  sequential - 顺序执行（精度优先，适合复杂场景）")
        print("  parallel   - 并行执行（速度优先，适合实时场景）")
        print("  auto       - 自动选择（根据图片大小）")
        print("\n标注说明:")
        print("  --annotate          在原图同目录的 annotated 子目录保存标注图")
        print("  --annotate=目录     在指定目录保存标注图")
        print("\n调试说明:")
        print("  --debug             在原图同目录的 debug 子目录保存各策略处理后的图像")
        print("  --debug=目录        在指定目录保存各策略处理后的图像")
        print("\n示例:")
        print("  python detector_qrcode.py test.png")
        print("  python detector_qrcode.py ./images/ parallel --annotate")
        print("  python detector_qrcode.py D:/qrcodes/ --annotate=D:/output/")
        print("  python detector_qrcode.py test.png --debug")
        print("  python detector_qrcode.py ./images/ --debug=./debug_output/")
        print("\n运行测试:")
        print("  python detector_qrcode.py --test")
