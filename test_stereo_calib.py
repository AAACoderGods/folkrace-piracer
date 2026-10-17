#!/usr/bin/env python3
"""
DECXIN 双目深度相机标定辅助。

用途：算出 DepthCameraConfig.stereo_scale_mm_px（单位 mm*px）。
它等于 focal_px * baseline_mm，也就是"深度 × 视差"的常数。
stereo 后端缺它不会启动（会回退 mock），所以必须先测出来。

原理：depth_mm = scale_mm_px / disparity_px
      故      scale_mm_px = depth_mm * disparity_px
把目标放在已知距离 D（毫米），读出中心区域视差 d（像素），scale = D * d。

用法：
    # 把高对比目标（棋盘格/十字靶）放在正前方 1000 mm 处，正对镜头，然后：
    python3 test_stereo_calib.py --distance 1000

脚本会采集若干帧，取中心区域有效视差的中位数 d，打印建议填入
config.py 的 stereo_scale_mm_px。多测几个距离（500 / 1000 / 2000 mm）
取平均更稳。

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


def parse_args():
    p = argparse.ArgumentParser(description="标定 DECXIN 双目相机的 stereo_scale_mm_px")
    p.add_argument("--distance", type=float, default=None,
                   help="目标到相机的实际距离，单位毫米（如 1000）。给了就打印建议 scale。")
    p.add_argument("--device", type=str, default="",
                   help="V4L2 节点，默认自动找第一个能出帧的 USB 视频节点。")
    p.add_argument("--width", type=int, default=1280, help="总帧宽（双眼），默认 1280")
    p.add_argument("--height", type=int, default=480, help="帧高（单眼高），默认 480")
    p.add_argument("--samples", type=int, default=15, help="采样帧数，默认 15")
    p.add_argument("--eye", choices=["left", "right"], default="left",
                   help="用哪只眼做参考，默认 left")
    p.add_argument("--num-disparities", type=int, default=128,
                   help="必须与 config.depth.stereo_num_disparities 一致，默认 128")
    p.add_argument("--block-size", type=int, default=15,
                   help="必须与 config.depth.stereo_block_size 一致，默认 15")
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


def central_disparity(gray_left, gray_right, num_disparities, block_size):
    """
    在中心区域计算有效视差的中位数。
    视差 0 表示 StereoBM 认为没有匹配，不计入。
    """
    nd = max(16, num_disparities - num_disparities % 16)
    bs = block_size if block_size % 2 == 1 and block_size >= 5 else 15
    bm = cv2.StereoBM_create(numDisparities=nd, blockSize=bs)
    raw = bm.compute(gray_left, gray_right).astype(np.float32) / 16.0
    valid = raw > 0.0
    if not np.any(valid):
        return None
    h, w = raw.shape[:2]
    # 中心矩形：纵向 40%-60%，横向整体。目标应放在画面正中。
    y0, y1 = int(h * 0.4), int(h * 0.6)
    patch = raw[y0:y1, :]
    mask = patch > 0.0
    if not np.any(mask):
        return None
    return float(np.median(patch[mask]))


def main():
    args = parse_args()

    device = pick_device(args.device)
    if device is None:
        print("没有找到能出帧的 USB 视频节点。先确认：")
        print("  v4l2-ctl --list-devices")
        print("  ls -l /dev/video*")
        sys.exit(2)
    print("使用节点：%s" % device)

    cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
    if not cap.isOpened():
        print("打不开 %s" % device)
        sys.exit(2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, 30)

    # 丢掉前几帧，让自动曝光稳定。
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
                    print("中心视差 ≈ %.1f px" % d)
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
    print()
    print("再重新运行主程序，stereo 后端就不会回退 mock 了。")
    print("如果多次测量的 scale 差异很大，说明目标不够平或对比度不够，重测。")


def _one_sample(frame, args):
    h, w = frame.shape[:2]
    if w % 2 != 0:
        w -= 1
        frame = frame[:, :w]
    half = w // 2
    left = frame[:, :half]
    right = frame[:, half:]
    gray_left = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)
    return central_disparity(gray_left, gray_right,
                             args.num_disparities, args.block_size)


if __name__ == "__main__":
    main()
