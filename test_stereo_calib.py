#!/usr/bin/env python3
"""
DECXIN 双目深度相机标定辅助。

用途：算出 DepthCameraConfig.stereo_scale_mm_px（单位 mm*px）。
原理：depth_mm = scale_mm_px / disparity_px
      故      scale_mm_px = depth_mm * disparity_px
把目标放在已知距离 D（毫米），读出中心区域视差 d（像素），scale = D * d。

本脚本用的匹配器和参数与运行时 depth_camera.py 完全一致（CLAHE + StereoSGBM
+ filterSpeckles + medianBlur），这样标定出的 scale 才会和实际运行时的
深度计算一致。参数从 config.py 里读，所以无论你怎么调 config 都不用改这个脚本。

用法：
    # 把高对比目标（棋盘格/十字靶）放在正前方 1000 mm 处，正对镜头，然后：
    python3 test_stereo_calib.py --distance 1000

不给 --distance 时进入"读数模式"：持续打印中心视差，方便你对着目标观察。
"""

import argparse
import statistics
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

# 从 config.py 读取运行时的参数。这样"标定工具"和"生产管线"永远是同一套
# 参数，避免出现"标定值对不上运行时"这种极难排查的问题。
try:
    from config import DepthCameraConfig
    _CFG = DepthCameraConfig()
except Exception:
    _CFG = None


def _cfg(name, default):
    """Read a field from config.py's DepthCameraConfig, falling back to default."""
    if _CFG is not None and hasattr(_CFG, name):
        return getattr(_CFG, name)
    return default


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


def parse_args():
    p = argparse.ArgumentParser(description="标定 DECXIN 双目相机的 stereo_scale_mm_px")
    p.add_argument("--distance", type=float, default=None,
                   help="目标到相机的实际距离，单位毫米（如 1000）。给了就打印建议 scale。")
    p.add_argument("--device", type=str, default="",
                   help="V4L2 节点，默认自动找第一个能出帧的 USB 视频节点。")
    p.add_argument("--width", type=int, default=1280, help="总帧宽（双眼），默认 1280")
    p.add_argument("--height", type=int, default=480, help="帧高（单眼高），默认 480")
    p.add_argument("--samples", type=int, default=15, help="采样帧数，默认 15")
    p.add_argument("--swap-eyes", action="store_true",
                   default=DEFAULT_SWAP_EYES,
                   help=f"翻转左右眼（默认随 config.depth.stereo_swap_eyes={DEFAULT_SWAP_EYES}）")
    p.add_argument("--clahe-clip", type=float, default=DEFAULT_CLAHE_CLIP,
                   help=f"CLAHE 对比度增强强度，0 关闭，默认 {DEFAULT_CLAHE_CLIP}")
    return p.parse_args()


def candidate_devices(preferred):
    if preferred:
        return [preferred]
    import glob
    return sorted(glob.glob("/dev/video*"))


def open_capture(device, width, height, fps=30):
    cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        return None
    return True


def pick_device(preferred):
    for device in candidate_devices(preferred):
        if open_capture(device, 1280, 480):
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
    """与 depth_camera._stereo_frame_to_depth 里构造的 SGBM 完全一致。"""
    nd = max(16, num_disparities - num_disparities % 16)
    bs = block_size if block_size % 2 == 1 and block_size >= 5 else 15
    p1 = 8 * bs * bs
    p2 = 32 * bs * bs
    return cv2.StereoSGBM_create(
        minDisparity=DEFAULT_MIN_DISPARITY,
        numDisparities=nd,
        blockSize=bs,
        P1=p1,
        P2=p2,
        disp12MaxDiff=DEFAULT_DISP12_MAX_DIFF,
        uniquenessRatio=DEFAULT_UNIQUENESS_RATIO,
        speckleWindowSize=DEFAULT_SPECKLE_WINDOW,
        speckleRange=DEFAULT_SPECKLE_RANGE,
        preFilterCap=DEFAULT_PRE_FILTER_CAP,
        mode=_sgbm_mode_flag(DEFAULT_SGBM_MODE),
    )


def central_disparity(gray_left, gray_right, num_disparities, block_size,
                      clahe_clip=0.0):
    """
    在中心区域计算有效视差的中位数，与运行时的处理链路一致：
    CLAHE -> SGBM -> filterSpeckles -> medianBlur -> 取正整数视差。
    """
    if clahe_clip > 0.0:
        try:
            clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(8, 8))
            gray_left = clahe.apply(gray_left)
            gray_right = clahe.apply(gray_right)
        except Exception:
            pass

    matcher = _build_matcher(num_disparities, block_size)
    raw16 = matcher.compute(gray_left, gray_right)
    if raw16 is None:
        return None
    if raw16.dtype != np.int16:
        raw16 = raw16.astype(np.int16)
    raw16[raw16 < 0] = 0
    try:
        cv2.filterSpeckles(raw16, 0, DEFAULT_SPECKLE_WINDOW, DEFAULT_SPECKLE_RANGE)
    except Exception:
        pass

    raw = raw16.astype(np.float32) / 16.0
    if DEFAULT_MEDIAN_BLUR >= 3 and DEFAULT_MEDIAN_BLUR % 2 == 1:
        try:
            raw = cv2.medianBlur(raw, DEFAULT_MEDIAN_BLUR)
        except Exception:
            pass

    if not np.any(raw > 0.0):
        return None
    h, w = raw.shape[:2]
    y0, y1 = int(h * 0.4), int(h * 0.6)
    patch = raw[y0:y1, :]
    mask = patch > 0.0
    if not np.any(mask):
        return None
    return float(np.median(patch[mask]))


def _one_sample(frame, args):
    h, w = frame.shape[:2]
    if w % 2 != 0:
        w -= 1
        frame = frame[:, :w]
    half = w // 2
    if args.swap_eyes:
        left = frame[:, half:]
        right = frame[:, :half]
    else:
        left = frame[:, :half]
        right = frame[:, half:]
    gray_left = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)
    return central_disparity(gray_left, gray_right,
                             DEFAULT_NUM_DISPARITIES, DEFAULT_BLOCK_SIZE,
                             args.clahe_clip)


def main():
    args = parse_args()

    device = pick_device(args.device)
    if device is None:
        print("没有找到能出帧的 USB 视频节点。先确认：")
        print("  v4l2-ctl --list-devices")
        print("  ls -l /dev/video*")
        sys.exit(2)
    print("使用节点：%s" % device)
    print("采样参数：numDisp=%d, blockSize=%d, uniq=%d, disp12=%d, mode=%s, CLAHE=%.1f, swap_eyes=%s"
          % (DEFAULT_NUM_DISPARITIES, DEFAULT_BLOCK_SIZE, DEFAULT_UNIQUENESS_RATIO,
             DEFAULT_DISP12_MAX_DIFF, DEFAULT_SGBM_MODE, args.clahe_clip,
             args.swap_eyes))

    cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
    if not cap.isOpened():
        print("打不开 %s" % device)
        sys.exit(2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, 30)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    actual_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    raw_fourcc = int(cap.get(cv2.CAP_PROP_FOURCC) or 0)
    fourcc = "".join(chr((raw_fourcc >> (8 * i)) & 0xFF) for i in range(4)).strip()
    print("协商结果：%dx%d @ %.1ffps, fourcc '%s'（请求 %dx%d, MJPG）"
          % (actual_w, actual_h, actual_fps, fourcc or "(未报告)",
             args.width, args.height))
    if actual_w and actual_w != args.width:
        print("⚠️  设备没有给出请求的宽度 %d，实际 %d。标定 scale 时要用实际宽度，"
              % (args.width, actual_w))
        print("    否则运行时 depth_camera 会自动按比例换算，看到的距离会有偏差。")

    for _ in range(10):
        cap.read()
        time.sleep(0.05)

    samples = []
    try:
        if args.distance is None:
            print("读数模式：持续打印中心视差，Ctrl+C 退出。")
            while True:
                ok, frame = cap.read()
                if not ok or frame is None:
                    continue
                d = _one_sample(frame, args)
                if d is not None:
                    print("中心视差 ≈ %.1f px（有效像素可能很少，注意观察）" % d)
                else:
                    print("中心视差：无有效匹配")
                time.sleep(0.3)
        else:
            print("采集 %d 帧，目标距离 %d mm ..." % (args.samples, int(args.distance)))
            for _ in range(args.samples):
                ok, frame = cap.read()
                if not ok or frame is None:
                    continue
                d = _one_sample(frame, args)
                if d is not None:
                    samples.append(d)
                time.sleep(0.05)
    finally:
        cap.release()

    if args.distance is None:
        return

    if not samples:
        print("没有测到有效视差。可能：目标对比度不够、距离超出匹配范围、或镜头被挡。")
        print("SGBM 在亮光下更容易匹配；对白墙、天花板这类无纹理场景本来就会失败。")
        print("提示：跑一下 test_stereo_debug.py，看 disparity.png 是不是一片黑。")
        print("     如果是，先用 --swap-eyes 试一下，或提高 --clahe-clip 到 3.0-4.0。")
        sys.exit(3)

    median_d = statistics.median(samples)
    scale = args.distance * median_d
    print()
    print("=" * 60)
    print(" 采样帧数      : %d" % len(samples))
    print(" 视差中位数    : %.2f px" % median_d)
    print(" 目标距离      : %d mm" % int(args.distance))
    print(" 建议的 scale  : %.1f mm*px" % scale)
    print("=" * 60)
    print()
    print("把下面放进 config.py 的 DepthCameraConfig：")
    print("    stereo_scale_mm_px: float = %.1f" % scale)
    print("    stereo_width: int = %d" % (actual_w // 2 if actual_w else args.width // 2))
    print("    stereo_height: int = %d" % (actual_h if actual_h else args.height))
    print()
    print("再重新运行主程序，stereo 后端就不会回退 mock 了。")
    print("如果多次测量的 scale 差异很大，说明目标不够平或对比度不够，重测。")


if __name__ == "__main__":
    main()
