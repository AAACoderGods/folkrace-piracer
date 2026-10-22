#!/usr/bin/env python3
"""
DECXIN 双目深度相机标定辅助工具（纯命令行 CLI 版）。

用途：准确算出 DepthCameraConfig.stereo_scale_mm_px（单位 mm*px）。
原理：depth_mm = scale_mm_px / disparity_px
     故 scale_mm_px = depth_mm * disparity_px
把目标放在已知距离 D（毫米），读出中心区域视差 d（像素），scale = D * d。

用法：
1. 观察读数模式：python test_stereo_calib.py
2. 标定采集模式：python test_stereo_calib.py --distance 1000
"""

import argparse
import sys
import time

try:
    import cv2
except ImportError:
    print("缺少 OpenCV：pip install opencv-python")
    sys.exit(1)

try:
    import numpy as np
except ImportError:
    print("缺少 numpy：pip install numpy")
    sys.exit(1)

# ==================== 1. 尝试读取生产环境配置 ====================
try:
    from config import DepthCameraConfig
    _CFG = DepthCameraConfig()
    print("成功加载 config.py 生产配置。")
except Exception:
    print("⚠️ 未找到 config.py 或加载失败，将使用默认参数运行。")
    _CFG = None


def _cfg(name, default):
    """从 config.py 读取字段，若不存在则返回默认值"""
    if _CFG is not None and hasattr(_CFG, name):
        return getattr(_CFG, name)
    return default


# SGBM 及预处理核心参数
DEFAULT_NUM_DISPARITIES = int(_cfg("stereo_num_disparities", 192))
DEFAULT_BLOCK_SIZE = int(_cfg("stereo_block_size", 15))
DEFAULT_MIN_DISPARITY = int(_cfg("stereo_min_disparity", 0))
DEFAULT_UNIQUENESS_RATIO = int(_cfg("stereo_uniqueness_ratio", 8))
DEFAULT_SPECKLE_WINDOW = int(_cfg("stereo_speckle_window_size", 200))
DEFAULT_SPECKLE_RANGE = int(_cfg("stereo_speckle_range", 2))
DEFAULT_DISP12_MAX_DIFF = int(_cfg("stereo_disp12_max_diff", 2))
DEFAULT_MEDIAN_BLUR = int(_cfg("stereo_median_blur", 5))
DEFAULT_PRE_FILTER_CAP = int(_cfg("stereo_pre_filter_cap", 31))
DEFAULT_SGBM_MODE = str(_cfg("stereo_sgbm_mode", "3way"))
DEFAULT_CLAHE_CLIP = float(_cfg("stereo_clahe_clip_limit", 2.0))
DEFAULT_SWAP_EYES = bool(_cfg("stereo_swap_eyes", False))

# 立体极线校正矩阵
MAP1_X = _cfg("map1_x", None)
MAP1_Y = _cfg("map1_y", None)
MAP2_X = _cfg("map2_x", None)
MAP2_Y = _cfg("map2_y", None)


# ==================== 2. 工具函数与核心算法 ====================
def parse_args():
    p = argparse.ArgumentParser(description="标定 DECXIN 双目相机的 stereo_scale_mm_px（命令行版）")
    p.add_argument("--distance", type=float, default=None,
                   help="目标到相机的实际距离，单位毫米（如 1000）。给了就进入标定模式，不给进入观察读数模式。")
    p.add_argument("--device", type=str, default="",
                   help="V4L2 节点，默认自动寻找。")
    p.add_argument("--width", type=int, default=1280, help="总帧宽（双眼），默认 1280")
    p.add_argument("--height", type=int, default=480, help="帧高（单眼高），默认 480")
    p.add_argument("--samples", type=int, default=30, help="标定采样帧数，默认 30")
    p.add_argument("--swap-eyes", action="store_true", default=DEFAULT_SWAP_EYES,
                   help=f"翻转左右眼（默认随 config: {DEFAULT_SWAP_EYES}）")
    p.add_argument("--clahe-clip", type=float, default=DEFAULT_CLAHE_CLIP,
                   help=f"CLAHE 强度，0 关闭，默认 {DEFAULT_CLAHE_CLIP}")
    return p.parse_args()


def candidate_devices(preferred):
    if preferred:
        return [preferred]
    import glob
    import platform
    if platform.system() == "Linux":
        return sorted(glob.glob("/dev/video*"))
    return [0, 1, 2]


def test_device(device, width, height):
    cap = cv2.VideoCapture(device)
    if not cap.isOpened():
        return False
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    ok, frame = cap.read()
    cap.release()
    return ok and frame is not None


def pick_device(preferred, width, height):
    for device in candidate_devices(preferred):
        if test_device(device, width, height):
            return device
    return None


def _sgbm_mode_flag(name):
    key = (name or "3way").strip().lower()
    table = {
        "sgbm": getattr(cv2, "STEREO_SGBM_MODE_SGBM", 0),
        "hh":   getattr(cv2, "STEREO_SGBM_MODE_HH", 1),
        "3way": getattr(cv2, "STEREO_SGBM_MODE_SGBM_3WAY", 2),
        "hh4":  getattr(cv2, "STEREO_SGBM_MODE_HH4", 3),
    }
    return table.get(key, table["3way"])


def _build_matcher(num_disparities, block_size):
    nd = max(16, num_disparities - num_disparities % 16)
    bs = block_size if block_size % 2 == 1 and block_size >= 3 else 15
    return cv2.StereoSGBM_create(
        minDisparity=DEFAULT_MIN_DISPARITY,
        numDisparities=nd,
        blockSize=bs,
        P1=8 * bs * bs,
        P2=32 * bs * bs,
        disp12MaxDiff=DEFAULT_DISP12_MAX_DIFF,
        uniquenessRatio=DEFAULT_UNIQUENESS_RATIO,
        speckleWindowSize=DEFAULT_SPECKLE_WINDOW,
        speckleRange=DEFAULT_SPECKLE_RANGE,
        preFilterCap=DEFAULT_PRE_FILTER_CAP,
        mode=_sgbm_mode_flag(DEFAULT_SGBM_MODE),
    )


def process_stereo_geometry(gray_left, gray_right, num_disparities, block_size, clahe_clip=0.0):
    # 1. Remap 极线校正
    if MAP1_X is not None and MAP1_Y is not None and MAP2_X is not None and MAP2_Y is not None:
        gray_left = cv2.remap(gray_left, MAP1_X, MAP1_Y, cv2.INTER_LINEAR)
        gray_right = cv2.remap(gray_right, MAP2_X, MAP2_Y, cv2.INTER_LINEAR)

    # 2. CLAHE 对比度增强
    if clahe_clip > 0.0:
        try:
            clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(8, 8))
            gray_left = clahe.apply(gray_left)
            gray_right = clahe.apply(gray_right)
        except Exception:
            pass

    # 3. SGBM 视差计算
    matcher = _build_matcher(num_disparities, block_size)
    raw16 = matcher.compute(gray_left, gray_right)
    if raw16 is None or raw16.dtype != np.int16:
        return None

    # 4. 斑点过滤 (填为无效标记)
    invalid_val = (DEFAULT_MIN_DISPARITY - 1) * 16
    try:
        cv2.filterSpeckles(raw16, invalid_val, DEFAULT_SPECKLE_WINDOW, DEFAULT_SPECKLE_RANGE)
    except Exception:
        pass

    # 5. 转 float 像素视差
    raw = raw16.astype(np.float32) / 16.0

    # 6. 中值滤波
    if DEFAULT_MEDIAN_BLUR >= 3 and DEFAULT_MEDIAN_BLUR % 2 == 1:
        try:
            raw = cv2.medianBlur(raw, DEFAULT_MEDIAN_BLUR)
        except Exception:
            pass

    # 7. 提取中心高可信 ROI (自动规避死区)
    h, w = raw.shape[:2]
    y0, y1 = int(h * 0.35), int(h * 0.65)
    
    dead_zone_x = DEFAULT_MIN_DISPARITY + num_disparities
    x_start = max(int(w * 0.3
