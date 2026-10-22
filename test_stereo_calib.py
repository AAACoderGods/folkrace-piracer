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
    x_start = max(int(w * 0.35), dead_zone_x)
    x_end = max(int(w * 0.65), x_start + 20)

    if x_start >= w - 10:
        return None

    patch = raw[y0:y1, x_start:x_end]
    
    # 8. 过滤有效点
    mask = patch > (DEFAULT_MIN_DISPARITY + 0.1)
    if not np.any(mask):
        return None

    valid_pixels = patch[mask]
    return float(np.median(valid_pixels)), float(np.std(valid_pixels))


def get_single_frame_disparity(frame, args):
    h, w = frame.shape[:2]
    if w % 2 != 0:
        w -= 1
        frame = frame[:, :w]
    half = w // 2

    if args.swap_eyes:
        left, right = frame[:, half:], frame[:, :half]
    else:
        left, right = frame[:, :half], frame[:, half:]

    gray_left = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)
    
    return process_stereo_geometry(gray_left, gray_right, DEFAULT_NUM_DISPARITIES, 
                                   DEFAULT_BLOCK_SIZE, args.clahe_clip)


# ==================== 3. 主逻辑控制 ====================
def main():
    args = parse_args()

    device = pick_device(args.device, args.width, args.height)
    if device is None:
        print("❌ 未找到可正常输出的双目视频节点。请检查连线。")
        sys.exit(2)
        
    print(f"打开视频节点: {device}")
    cap = cv2.VideoCapture(device)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, 30)

    print("-" * 50)
    print(f"当前运行参数: numDisp={DEFAULT_NUM_DISPARITIES}, blockSize={DEFAULT_BLOCK_SIZE}, minDisp={DEFAULT_MIN_DISPARITY}")
    print(f"滤波配置: uniqRatio={DEFAULT_UNIQUENESS_RATIO}, medianBlur={DEFAULT_MEDIAN_BLUR}, clahe={args.clahe_clip}")
    print("-" * 50)

    try:
        # 1. 自动采样标定模式
        if args.distance is not None:
            target_dist = args.distance
            print(f"🎯 进入【标定计算模式】: 目标距离设为 {target_dist} mm")
            print(f"正在采集 {args.samples} 帧有效数据，请保持目标静止...")
            
            valid_disparities = []
            consecutive_failures = 0
            
            while len(valid_disparities) < args.samples:
                ok, frame = cap.read()
                if not ok or frame is None:
                    continue
                    
                res = get_single_frame_disparity(frame, args)
                if res is not None:
                    median_disp, std_disp = res
                    if std_disp < 1.5:
                        valid_disparities.append(median_disp)
                        print(f"  [采样进度 {len(valid_disparities):2d}/{args.samples}] 视差: {median_disp:6.2f} px | Std: {std_disp:4.2f}")
                        consecutive_failures = 0
                    else:
                        print(f"  ⚠️ 视差离散度过高 (Std: {std_disp:4.2f})，抛弃此帧。请平正对齐目标！")
                else:
                    consecutive_failures += 1
                    if consecutive_failures > 50:
                        print("\n❌ 错误: 持续无法找到有效视差匹配。请检查目标纹理与左右眼设置。")
                        sys.exit(3)
                time.sleep(0.03)

            final_disp = float(np.median(valid_disparities))
            suggested_scale = target_dist * final_disp
            print("=" * 60)
            print("🎉 标定采集完成！")
            print(f"▶ 采样平均视差: {final_disp:.3f} px")
            print(f"▶ 推荐填入 config.py 的配置:")
            print(f"  stereo_scale_mm_px = {suggested_scale:.2f}")
            print("=" * 60)

        # 2. 纯命令行实时观察模式
        else:
            print("👀 进入【命令行实时观察模式】（按 Ctrl+C 退出）\n")
            while True:
                ok, frame = cap.read()
                if not ok or frame is None:
                    continue

                res = get_single_frame_disparity(frame, args)
                if res is not None:
                    median_disp, std_disp = res
                    # ASCII 可视化视差条（最大长度 40 字符）
                    bar_len = int(min(median_disp, 200) / 5)
                    bar_str = "█" * bar_len
                    
                    # 稳定性指示灯：Std < 1.5 显示 OK，否则显示 ⚠️
                    status = "🟢 OK" if std_disp < 1.5 else "🔴 波动"

                    # 行内实时刷新
                    sys.stdout.write(
                        f"\r[实时视差] {median_disp:6.2f} px | Std: {std_disp:4.2f} | 状态: {status} | [{bar_str:<40s}]"
                    )
                    sys.stdout.flush()
                else:
                    sys.stdout.write("\r[实时视差] 正在寻找有效视差匹配...                                 ")
                    sys.stdout.flush()

                time.sleep(0.03)

    except KeyboardInterrupt:
        print("\n\n用户手动退出程序。")
    finally:
        cap.release()


if __name__ == "__main__":
    main()
