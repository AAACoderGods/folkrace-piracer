#!/usr/bin/env python3
"""
双目深度相机诊断工具。

当"测距不准"或"经常读不出"时，跑一下这个脚本。它会完整走一遍运行时的管线
（CLAHE -> SGBM -> filterSpeckles -> medianBlur -> 深度换算），并把每一步的
中间结果保存成 PNG，同时打印统计。

生成的图（默认保存在 ./stereo_debug/ 目录下）：
    left.png      原始左眼图（灰度化前）
    right.png     原始右眼图（灰度化前）
    disparity.png 视差图（伪彩色，越亮表示视差越大即越近）
    depth.png     深度图（伪彩色，红色越深越近）
    overlay.png   左右图并排，并画 5 条水平参考线

如果 disparity.png 一片黑 → SGBM 匹配失败，问题在图像或参数
如果 overlay.png 里相同特征不在同一水平线上 → 没有立体校正，问题在硬件
如果 disparity.png 有内容但 depth.png 颜色不对 → 标定值不准
如果 left/right 有内容但 overlay 里左右眼"看起来一模一样" → 相机是单目的

用法：
    python3 test_stereo_debug.py
    python3 test_stereo_debug.py --distance 800
    python3 test_stereo_debug.py --swap-eyes
    python3 test_stereo_debug.py --out-dir /tmp/stereo_dbg --samples 10
"""

import argparse
import os
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

# 从 config.py 读取参数，保证诊断与运行时完全一致。
try:
    from config import DepthCameraConfig
    _CFG = DepthCameraConfig()
except Exception:
    _CFG = None


def _cfg(name, default):
    if _CFG is not None and hasattr(_CFG, name):
        return getattr(_CFG, name)
    return default


def parse_args():
    p = argparse.ArgumentParser(description="双目相机诊断")
    p.add_argument("--device", type=str, default="",
                   help="V4L2 节点，默认扫描第一个能出帧的 USB 视频节点")
    p.add_argument("--width", type=int, default=1280, help="总帧宽，默认 1280")
    p.add_argument("--height", type=int, default=480, help="帧高，默认 480")
    p.add_argument("--distance", type=float, default=None,
                   help="已知目标距离（mm），用来核对 scale")
    p.add_argument("--samples", type=int, default=5,
                   help="采样几帧，取有效像素最多的一帧保存，默认 5")
    p.add_argument("--out-dir", type=str, default="stereo_debug")
    p.add_argument("--swap-eyes", action="store_true",
                   default=bool(_cfg("stereo_swap_eyes", False)),
                   help="翻转左右眼（默认随 config.depth.stereo_swap_eyes）")
    p.add_argument("--clahe-clip", type=float, default=None,
                   help="CLAHE 强度，默认取 config.depth.stereo_clahe_clip_limit")
    return p.parse_args()


def _sgbm_mode_flag(name):
    key = (name or "3way").strip().lower()
    table = {
        "sgbm": getattr(cv2, "STEREO_SGBM_MODE_SGBM", 0),
        "hh":   getattr(cv2, "STEREO_SGBM_MODE_HH", 1),
        "3way": getattr(cv2, "STEREO_SGBM_MODE_SGBM_3WAY", 2),
        "hh4":  getattr(cv2, "STEREO_SGBM_MODE_HH4", 3),
    }
    return table.get(key, table["3way"])


def _build_matcher():
    nd = int(_cfg("stereo_num_disparities", 192))
    if nd < 16 or nd % 16 != 0:
        nd = 192
    bs = int(_cfg("stereo_block_size", 15))
    if bs < 5 or bs % 2 == 0:
        bs = 15
    p1 = 8 * bs * bs
    p2 = 32 * bs * bs
    return cv2.StereoSGBM_create(
        minDisparity=int(_cfg("stereo_min_disparity", 0)),
        numDisparities=nd,
        blockSize=bs,
        P1=p1,
        P2=p2,
        disp12MaxDiff=max(0, int(_cfg("stereo_disp12_max_diff", 2))),
        uniquenessRatio=max(0, int(_cfg("stereo_uniqueness_ratio", 8))),
        speckleWindowSize=max(0, int(_cfg("stereo_speckle_window_size", 200))),
        speckleRange=max(0, int(_cfg("stereo_speckle_range", 2))),
        preFilterCap=int(_cfg("stereo_pre_filter_cap", 31)),
        mode=_sgbm_mode_flag(_cfg("stereo_sgbm_mode", "3way")),
    )


def _compute(frame, swap_eyes, clahe_clip, scale):
    h, w = frame.shape[:2]
    if w % 2 != 0:
        w -= 1
        frame = frame[:, :w]
    half = w // 2
    if swap_eyes:
        left = frame[:, half:]
        right = frame[:, :half]
    else:
        left = frame[:, :half]
        right = frame[:, half:]

    gray_left = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)

    proc_left = gray_left.copy()
    proc_right = gray_right.copy()
    if clahe_clip > 0.0:
        clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(8, 8))
        proc_left = clahe.apply(proc_left)
        proc_right = clahe.apply(proc_right)

    matcher = _build_matcher()
    raw16 = matcher.compute(proc_left, proc_right)
    if raw16 is None:
        return left, right, None, None, None
    if raw16.dtype != np.int16:
        raw16 = raw16.astype(np.int16)
    raw16[raw16 < 0] = 0
    try:
        cv2.filterSpeckles(raw16, 0,
                           max(0, int(_cfg("stereo_speckle_window_size", 200))),
                           max(0, int(_cfg("stereo_speckle_range", 2))))
    except Exception:
        pass
    disparity = raw16.astype(np.float32) / 16.0

    median_k = int(_cfg("stereo_median_blur", 5))
    if median_k >= 3 and median_k % 2 == 1:
        try:
            disparity = cv2.medianBlur(disparity, median_k)
        except Exception:
            pass
    disparity = np.where(disparity > 0.0, disparity, 0.0)

    depth = np.zeros_like(disparity)
    valid = disparity > 0.0
    if np.any(valid):
        depth[valid] = np.clip(scale / disparity[valid], 0.0, 65535.0)

    return left, right, proc_left, proc_right, (disparity, depth)


def _colorize_depth(depth, min_mm, max_mm):
    """深度图伪彩色：红=近，蓝=远，白=无读数。"""
    if depth is None or depth.size == 0:
        return None
    clipped = np.clip(depth, min_mm, max_mm)
    norm = ((clipped - min_mm) / max(1.0, (max_mm - min_mm)) * 255.0).astype(np.uint8)
    norm = np.where(depth > 0.0, norm, 255)
    colored = cv2.applyColorMap(norm, cv2.COLORMAP_TURBO)
    colored = np.where(depth[..., None] > 0.0, colored, 255)
    return colored.astype(np.uint8)


def _colorize_disparity(disp, max_disp):
    if disp is None or disp.size == 0:
        return None
    norm = np.clip(disp / max(1.0, max_disp) * 255.0, 0, 255).astype(np.uint8)
    colored = cv2.applyColorMap(norm, cv2.COLORMAP_TURBO)
    colored[disp <= 0] = 0
    return colored


def _print_stats(disp, depth, scale):
    if disp is None:
        print("  ⚠️  视差图为 None —— SGBM 或灰度化失败")
        return
    total = disp.size
    nz = int(np.count_nonzero(disp > 0.0))
    ratio = nz / max(1, total)
    print("  有效像素比  : %.1f%% (%d / %d)" % (ratio * 100.0, nz, total))
    if nz == 0:
        print("  ⚠️  整幅视差图没有任何有效匹配 —— 这就是'读不出'的直接原因")
        return
    valid_disp = disp[disp > 0.0]
    print("  视差分布    : 中位数 %.1f, P10 %.1f, P90 %.1f (px)"
          % (float(np.median(valid_disp)),
             float(np.percentile(valid_disp, 10)),
             float(np.percentile(valid_disp, 90))))
    print("  视差近端上界: %.1f px（SGBM 搜索范围上限）"
          % float(_cfg("stereo_num_disparities", 192) - 1))

    if depth is not None:
        valid_depth = depth[depth > 0.0]
        if valid_depth.size:
            print("  深度分布    : 中位数 %.0f, P10 %.0f, P90 %.0f (mm)"
                  % (float(np.median(valid_depth)),
                     float(np.percentile(valid_depth, 10)),
                     float(np.percentile(valid_depth, 90))))

    print("  配置 scale  : %.1f mm*px" % scale)


def _print_obstacle_reading(depth):
    """用运行时的 measure_obstacle_mm 逻辑读一次，确认整条链路通。"""
    if depth is None:
        print("  障碍测量    : None（没有深度图）")
        return
    h, w = depth.shape
    cy = int(np.clip(_cfg("obstacle_scan_y_ratio", 0.70), 0.0, 1.0) * (h - 1))
    band_half = max(1, int(round(h * float(_cfg("obstacle_scan_band_height_ratio", 0.15)))))
    y0 = max(0, cy - band_half)
    y1 = min(h, cy + band_half + 1)
    hw = int(np.clip(_cfg("obstacle_scan_width_ratio", 0.50), 0.0, 1.0) * w / 2.0)
    x0 = max(0, w // 2 - hw)
    x1 = min(w, w // 2 + hw)
    band = depth[y0:y1, x0:x1]
    vmin = float(_cfg("min_depth_mm", 150.0))
    vmax = float(_cfg("max_depth_mm", 5000.0))
    valid = band[(band >= vmin) & (band <= vmax)]
    min_px = int(_cfg("obstacle_min_valid_pixels", 20))
    pct = float(_cfg("obstacle_percentile", 10.0))
    if valid.size < min_px:
        print("  障碍测量    : None（带内有效像素 %d < %d）" % (valid.size, min_px))
        return
    near = float(np.percentile(valid, max(1.0, min(99.0, pct))))
    print("  障碍测量    : %.0f mm（%d 个有效像素，取第 %.0f 百分位）"
          % (near, valid.size, pct))


def main():
    args = parse_args()

    clahe_clip = args.clahe_clip
    if clahe_clip is None:
        clahe_clip = float(_cfg("stereo_clahe_clip_limit", 2.0))

    scale = float(_cfg("stereo_scale_mm_px", 0.0))
    if scale <= 0.0:
        focal = float(_cfg("stereo_focal_px", 0.0))
        baseline = float(_cfg("stereo_baseline_mm", 0.0))
        if focal > 0.0 and baseline > 0.0:
            scale = focal * baseline
    print("=" * 64)
    print("双目相机诊断")
    print("  scale          : %.1f mm*px（0 表示未标定）" % scale)
    print("  numDisparities : %d，blockSize %d，uniqueness %d"
          % (int(_cfg("stereo_num_disparities", 192)),
             int(_cfg("stereo_block_size", 15)),
             int(_cfg("stereo_uniqueness_ratio", 8))))
    print("  disp12MaxDiff  : %d，preFilterCap %d，SGBM mode %s"
          % (int(_cfg("stereo_disp12_max_diff", 2)),
             int(_cfg("stereo_pre_filter_cap", 31)),
             _cfg("stereo_sgbm_mode", "3way")))
    print("  CLAHE clip     : %.1f（0 = 关闭）" % clahe_clip)
    print("  swap_eyes      : %s（命令行 --swap-eyes 覆盖配置）"
          % args.swap_eyes)
    print("=" * 64)

    if scale <= 0.0:
        print("⚠️  scale 为 0，深度会算不出来。先跑 test_stereo_calib.py。")
        print("   （诊断图仍会生成，但 depth.png 和障碍测量会是无效的）")

    # 找节点
    devices = sorted(__import__("glob").glob("/dev/video*")) if not args.device else [args.device]
    chosen = None
    for dev in devices:
        cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)
        if not cap.isOpened():
            continue
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
        cap.set(cv2.CAP_PROP_FPS, 30)
        ok, frame = cap.read()
        cap.release()
        if ok and frame is not None:
            chosen = dev
            break
    if chosen is None:
        print("没有找到能出帧的 USB 视频节点。先确认：")
        print("  v4l2-ctl --list-devices")
        print("  ls -l /dev/video*")
        sys.exit(2)
    print("使用节点：%s" % chosen)

    cap = cv2.VideoCapture(chosen, cv2.CAP_V4L2)
    if not cap.isOpened():
        print("打不开 %s" % chosen)
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
    print("协商：%dx%d @ %.1ffps, fourcc '%s'（请求 %dx%d）"
          % (actual_w, actual_h, actual_fps, fourcc or "(未报告)",
             args.width, args.height))

    # 丢弃前几帧，等自动曝光稳定
    for _ in range(10):
        cap.read()
        time.sleep(0.05)

    # 采样若干帧，保存有效像素最多的那一帧
    best = None
    best_ratio = -1.0
    try:
        for _ in range(max(1, args.samples)):
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            result = _compute(frame, args.swap_eyes, clahe_clip, scale)
            disp = result[4][0] if result[4] is not None else None
            ratio = 0.0 if disp is None else float(np.count_nonzero(disp > 0.0)) / float(disp.size)
            if ratio > best_ratio:
                best = result
                best_ratio = ratio
            time.sleep(0.05)
    finally:
        cap.release()

    if best is None:
        print("没有任何一帧能出片，检查一下相机和线缆。")
        sys.exit(2)

    left, right, proc_left, proc_right, payload = best
    if payload is None:
        disparity, depth = None, None
    else:
        disparity, depth = payload

    print()
    print("统计：")
    _print_stats(disparity, depth, scale)
    if depth is not None:
        _print_obstacle_reading(depth)

    if args.distance is not None and disparity is not None:
        nz = disparity[disparity > 0.0]
        if nz.size:
            med = float(np.median(nz))
            expected = scale / max(1.0, float(args.distance)) if scale > 0 else 0.0
            print()
            print("对于已知距离 %d mm：" % int(args.distance))
            if scale > 0:
                print("  预期视差    : %.1f px（scale / distance）" % expected)
            print("  实测中位视差: %.1f px" % med)
            if scale > 0 and abs(expected) > 0.1:
                err = (med - expected) / expected * 100.0
                print("  偏差        : %+.1f%%" % err)
                if abs(err) > 15.0:
                    print("  ⚠️  偏差过大。可尝试：")
                    print("     - test_stereo_calib.py 重新测量 scale")
                    print("     - 如果目标距离 <0.3m 或 >3m，SGBM 精度都会下降")
                    print("     - 检查目标是否与镜头正对，是否平整")

    # 保存诊断图
    out_dir = args.out_dir
    try:
        os.makedirs(out_dir, exist_ok=True)
    except OSError as e:
        print("无法创建 %s: %s" % (out_dir, e))
        sys.exit(2)

    cv2.imwrite(os.path.join(out_dir, "left.png"), left)
    cv2.imwrite(os.path.join(out_dir, "right.png"), right)

    if disparity is not None:
        max_disp = float(_cfg("stereo_num_disparities", 192))
        colored_disp = _colorize_disparity(disparity, max_disp)
        if colored_disp is not None:
            cv2.imwrite(os.path.join(out_dir, "disparity.png"), colored_disp)

    if depth is not None:
        vmin = float(_cfg("min_depth_mm", 150.0))
        vmax = float(_cfg("max_depth_mm", 5000.0))
        colored_depth = _colorize_depth(depth, vmin, vmax)
        if colored_depth is not None:
            cv2.imwrite(os.path.join(out_dir, "depth.png"), colored_depth)

    # overlay：左右图并排 + 5 条水平参考线，用于检查极线对齐
    try:
        hh = min(left.shape[0], right.shape[0])
        overlay = np.hstack([left[:hh], right[:hh]])
        oh, ow = overlay.shape[:2]
        for frac in (0.1, 0.3, 0.5, 0.7, 0.9):
            y = int(oh * frac)
            cv2.line(overlay, (0, y), (ow, y), (0, 0, 255), 1)
        cv2.imwrite(os.path.join(out_dir, "overlay.png"), overlay)
    except Exception as e:
        print("生成 overlay 失败：%s" % e)

    print()
    print("诊断图已保存到：%s/" % os.path.abspath(out_dir))
    print("  left.png      左眼原图")
    print("  right.png     右眼原图")
    print("  disparity.png 视差图（黑 = 匹配失败，亮 = 视差大即近）")
    print("  depth.png     深度图（红近蓝远，白 = 无读数）")
    print("  overlay.png   左右并排 + 水平参考线（检查极线对齐）")
    print()
    print("判读建议：")
    print("  - disparity.png 一片黑 → 匹配失败。先试 --swap-eyes；无效则检查光照/纹理。")
    print("  - overlay.png 中同一物体在左右两图不在同一条红线上 → 硬件未做极线校正，")
    print("    SGBM 只能匹配到极少数点，需要换成支持校正的相机或自己做校正。")
    print("  - disparity.png 有内容但 depth.png 颜色明显不对 → scale 不准，重跑")
    print("    test_stereo_calib.py。")


if __name__ == "__main__":
    main()
