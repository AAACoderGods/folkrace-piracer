#!/usr/bin/env python3
"""
把 DECXIN 的一帧彩色拆成左右两半，验证它到底是不是双目相机。

背景：test_depth_camera.py 已确认这台相机没有任何 UVC 深度格式，
      只输出 MJPG / YUYV，而且每个分辨率都是双倍宽度
      （1600x600 = 2 x 800x600，3840x1080 = 2 x 1920x1080，……）。
      这符合「左右两个镜头并排成一张彩色图」的双目特征，
      但也可能是单镜头画面被拉伸/复制成双倍宽，肉眼在缩略图上分不清。

本脚本给出可判定的数字：
    1. 抓一帧彩色，按中线切成左半、右半
    2. 分别存成 PNG，再存一张差异图（放大 4 倍，否则肉眼看不见）
    3. 打印两半的逐像素差异，以及相位相关估计出的整体错位

判定：
    平均差小、响应接近 1、水平位移 dx 明显非 0   -> 双目（有视差）
    平均差接近 0、响应接近 1、dx 也接近 0        -> 单镜头画面被复制/拉伸
    平均差很大、响应很低                          -> 两半是两幅独立画面

用法（车头对准有远近层次的场景：墙角、桌沿、门口都行）：
    python3 test_stereo_split.py
    # 会在当前目录生成 left_*.png / right_*.png / diff_*.png，都不覆盖旧文件
    # 把三张图，以及文字输出，一起贴回给开发者。
"""

import argparse
import sys
import time

import numpy as np

try:
    import cv2
except ImportError:
    print("需要 OpenCV：pip install opencv-python")
    sys.exit(1)


def parse_args():
    p = argparse.ArgumentParser(description="把相机的一帧拆成左右两半，判定是否双目")
    p.add_argument("--device", default="/dev/video1",
                   help="V4L2 节点（默认 /dev/video1，即 DECXIN 的流节点）")
    p.add_argument("--width", type=int, default=1280,
                   help="整帧宽度，会被对半切开（默认 1280 -> 左右各 640）")
    p.add_argument("--height", type=int, default=480,
                   help="整帧高度（默认 480）")
    return p.parse_args()


def open_capture(device, width, height):
    """
    开流并读一帧。返回 (cap, frame) 或 (None, None)。

    两次尝试：显式 CAP_V4L2，再退到 CAP_ANY。有些节点只认其中一种。
    MJPG 优先，因为它比未压缩的 YUYV 省 USB 带宽，同样的线更容易跑到目标尺寸。
    """
    attempts = [getattr(cv2, "CAP_V4L2", cv2.CAP_ANY), cv2.CAP_ANY]
    for backend in attempts:
        cap = cv2.VideoCapture(device, backend)
        if not cap.isOpened():
            continue

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        try:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        except Exception:
            pass

        # 让相机按新尺寸稳定下来，再抓真正要用的那一帧：刚改完属性那几帧
        # 常常还是旧尺寸或者没曝光完整。
        time.sleep(0.3)
        for _ in range(3):
            ok, frame = cap.read()
            if ok and frame is not None:
                return cap, frame
        cap.release()
    return None, None


def estimate_shift(left_gray, right_gray):
    """
    相位相关估计两幅图之间的整体位移（像素）。

    返回 (dx, dy, response)；出错时返回 (None, None, None)。
    dx 就是左右眼视差的直接估计，response 越接近 1，
    越说明两幅图是同一场景的平移版本——这正是双目的特征。
    """
    try:
        lf = left_gray.astype(np.float32)
        rf = right_gray.astype(np.float32)
        (dx, dy), response = cv2.phaseCorrelate(lf, rf)
        return float(dx), float(dy), float(response)
    except Exception as e:
        print("  (相位相关失败：%s)" % e)
        return None, None, None


def main():
    args = parse_args()

    print("=" * 68)
    print(" DECXIN 双目拆分检查")
    print("=" * 68)
    print("设备: %s   请求尺寸: %dx%d" % (args.device, args.width, args.height))
    print()

    cap, frame = open_capture(args.device, args.width, args.height)
    if frame is None:
        print("❌ 读不到帧。")
        print("   先跑 test_depth_camera.py，确认 %s 能打开、能出帧，再回来。" % args.device)
        sys.exit(1)

    try:
        h, w = frame.shape[:2]
        mid = w // 2
        # 优先按中线切；奇数宽时右半少一列，直接用 mid:mid+mid 保证两半等宽。
        left = frame[:, :mid].copy()
        right = frame[:, mid:mid + mid].copy()

        print("实际尺寸: %dx%d   左半 %dx%d   右半 %dx%d" % (
            w, h, left.shape[1], left.shape[0], right.shape[1], right.shape[0]))
        print()

        stamp = time.strftime("%Y%m%d_%H%M%S")
        left_path = "left_%s.png" % stamp
        right_path = "right_%s.png" % stamp
        diff_path = "diff_%s.png" % stamp

        # 差异图放大若干倍再存：真实的视差差常常很小，原样存下来几乎全黑，
        # 放大后结构（边缘、近处物体的错位）才看得出来。
        diff = cv2.absdiff(left, right)
        diff_vis = cv2.convertScaleAbs(diff, alpha=4.0)

        cv2.imwrite(left_path, left)
        cv2.imwrite(right_path, right)
        cv2.imwrite(diff_path, diff_vis)
        print("已保存: %s" % left_path)
        print("已保存: %s" % right_path)
        print("已保存: %s   (差异 x4 增强)" % diff_path)
        print()

        mean_abs = float(diff.mean())
        # 按元素计数（H*W*3），所以占比以通道为单位，够用了。
        nonzero_ratio = float(np.count_nonzero(diff)) / float(diff.size)

        print("--- 逐像素差异 ---")
        print("  平均绝对差   : %.2f   (0 = 两半逐像素完全相同)" % mean_abs)
        print("  非零像素占比 : %.1f%%" % (nonzero_ratio * 100.0))
        print()

        left_gray = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
        right_gray = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)
        dx, dy, response = estimate_shift(left_gray, right_gray)

        print("--- 相位相关（两半的整体错位）---")
        if dx is None:
            print("  无法估计。")
        else:
            print("  水平位移 dx  : %+.2f px" % dx)
            print("  垂直位移 dy  : %+.2f px" % dy)
            print("  相关响应     : %.3f   (越接近 1 越像同一场景的平移)" % response)
        print()

        print("=" * 68)
        print(" 怎么读结果")
        print("=" * 68)
        print("  平均差小、响应接近 1、dx 明显非 0")
        print("      -> 双目。同一个点在左右两半里横向错开了 dx 像素。")
        print("         把三张图和这段输出发回来，下一步谈标定与视差。")
        print()
        print("  平均差接近 0、响应接近 1、dx 也接近 0")
        print("      -> 不是双目。两半是同一画面，被设备复制或拉伸成了双倍宽。")
        print("         这台相机没有任何深度能力，收工。")
        print()
        print("  平均差很大、响应很低")
        print("      -> 两半是两幅独立画面，不是一个场景的左右眼。")
        print("         看看 left_*.png / right_*.png 是不是两个方向或两个场景。")
        print()
        print("  直接看 PNG 也能佐证：")
        print("     双目 -> 两图角度略有不同，近处物体横向错开更多。")
        print("     单目 -> 两图完全一样。")
    finally:
        if cap is not None:
            cap.release()


if __name__ == "__main__":
    main()
