#!/usr/bin/env python3
"""Inspect HITNet ranging or fit calibration from recorded stereo frames.

Record frames on the Pi with depth.stereo_record_dir set, then copy that folder
to a machine with ONNX Runtime and the configured model. For multi-range
calibration, create a CSV with columns ``frame,distance_mm``; paths may be
relative to the CSV file, e.g.::

    frame,distance_mm
    00012.jpg,500
    00048.jpg,1000
    00105.jpg,1500

The target should fill most of the centre ROI at each distance.
"""
import argparse
import csv
import os
import sys

import cv2
import numpy as np

from config import DepthCameraConfig
from depth_camera import DepthCamera


def _center_disparity(disparity):
    h, w = disparity.shape
    roi = disparity[h // 3: 2 * h // 3, w // 3: 2 * w // 3]
    valid = roi[np.isfinite(roi) & (roi > 0.0)]
    if valid.size < 20:
        raise ValueError(f"中心区域有效视差点太少：{valid.size}")
    low, high = np.percentile(valid, [10, 90])
    stable = valid[(valid >= low) & (valid <= high)]
    median = float(np.median(stable))
    mad = float(np.median(np.abs(stable - median)))
    return median, mad, float(valid.size / roi.size)


def _fit_calibration(samples):
    """Fit model disparity = scale / distance + offset."""
    distances = np.asarray([sample[0] for sample in samples], dtype=np.float64)
    disparities = np.asarray([sample[1] for sample in samples], dtype=np.float64)
    if len(np.unique(distances)) < 3:
        raise ValueError("至少需要 3 个不同的已知距离，建议覆盖实际使用范围")
    if (not np.all(np.isfinite(distances)) or np.any(distances <= 0.0)
            or not np.all(np.isfinite(disparities))):
        raise ValueError("距离必须为正数，视差必须为有限值")

    design = np.column_stack((1.0 / distances, np.ones_like(distances)))
    scale_model, offset = np.linalg.lstsq(design, disparities, rcond=None)[0]
    if scale_model <= 0.0:
        raise ValueError("拟合得到的视差尺度非正，请检查左右眼顺序和样本")
    predicted = design @ np.array([scale_model, offset])
    residual = disparities - predicted
    total = float(np.sum((disparities - np.mean(disparities)) ** 2))
    r_squared = 1.0 - float(np.sum(residual ** 2)) / total if total > 0.0 else 0.0
    return float(scale_model), float(offset), r_squared


def _load_frame(path):
    frame = cv2.imread(path)
    if frame is None:
        raise ValueError(f"无法读取双目帧：{path}")
    if frame.ndim != 3 or frame.shape[1] < 2 or frame.shape[1] % 2:
        raise ValueError(f"帧不是有效的左右拼接图：{path} {frame.shape}")
    half = frame.shape[1] // 2
    return frame[:, :half], frame[:, half:]


def _infer_disparity(cam, left, right, swap_eyes):
    if swap_eyes:
        left, right = right, left
    disparity = cam._disparity_from_fast_stereonet(left, right)
    if disparity is None:
        raise ValueError("HITNet 推理失败")
    return disparity


def _calibration_samples(csv_path, cam, swap_eyes):
    samples = []
    eye_widths = set()
    with open(csv_path, newline="", encoding="utf-8-sig") as file:
        rows = csv.DictReader(file)
        if not rows.fieldnames or not {"frame", "distance_mm"}.issubset(rows.fieldnames):
            raise ValueError("标定 CSV 必须包含 frame,distance_mm 两列")
        for line, row in enumerate(rows, start=2):
            frame_path = row["frame"].strip()
            distance = float(row["distance_mm"])
            if not frame_path:
                raise ValueError(f"第 {line} 行缺少 frame")
            if not os.path.isabs(frame_path):
                frame_path = os.path.join(os.path.dirname(os.path.abspath(csv_path)), frame_path)
            left, right = _load_frame(frame_path)
            if left.shape != right.shape:
                raise ValueError(f"第 {line} 行左右图尺寸不一致：{left.shape} / {right.shape}")
            eye_widths.add(left.shape[1])
            disparity = _infer_disparity(cam, left, right, swap_eyes)
            median, mad, valid_ratio = _center_disparity(disparity)
            samples.append((distance, median))
            print(f"{os.path.basename(frame_path)}: {distance:.0f} mm, "
                  f"视差 {median:.2f} px, MAD {mad:.2f}, 有效 {valid_ratio:.0%}")
    if not samples:
        raise ValueError("标定 CSV 中没有样本行")
    if len(eye_widths) != 1:
        raise ValueError("所有标定帧必须使用相同的单眼图像宽度")
    return samples, eye_widths.pop()


def main():
    parser = argparse.ArgumentParser(description="HITNet 单帧测距或多距离标定")
    parser.add_argument("--dir", default="stereo_debug",
                        help="旧格式 left.png/right.png 所在目录")
    parser.add_argument("--distance", type=float, help="单帧靶子真实距离（mm）")
    parser.add_argument("--frame", help="已录制的左右拼接双目帧 JPG")
    parser.add_argument("--calibration-csv",
                        help="标定 CSV，列为 frame,distance_mm；frame 可相对 CSV")
    parser.add_argument("--swap-eyes", action="store_true")
    args = parser.parse_args()

    if args.calibration_csv and (args.frame or args.distance is not None):
        parser.error("--calibration-csv 不能与 --frame 或 --distance 同时使用")
    if args.frame and args.distance is None:
        parser.error("--frame 模式必须提供 --distance")
    if not args.calibration_csv and not args.frame and args.distance is None:
        parser.error("请提供 --frame/--distance、--calibration-csv，或左右图及 --distance")
    if args.distance is not None and args.distance <= 0.0:
        parser.error("--distance 必须大于 0")
    if args.calibration_csv and not os.path.isfile(args.calibration_csv):
        sys.exit(
            f"找不到标定文件：{args.calibration_csv}\n"
            "请先手动创建 CSV，记录每张双目帧对应的实测距离，例如：\n"
            "  frame,distance_mm\n"
            "  00000.jpg,500\n"
            "  00030.jpg,800\n"
            "  00060.jpg,1200\n"
            "frame 路径可相对于 CSV 所在目录；每个距离至少一帧，"
            "至少需要 3 个不同距离。index.csv 不包含实测距离，不能直接用作标定表。"
        )

    cfg = DepthCameraConfig()
    cam = DepthCamera(cfg)
    if not cam._init_fast_stereonet():
        sys.exit("模型没加载成功（检查模型路径和 onnxruntime）")

    if args.calibration_csv:
        try:
            samples, eye_w = _calibration_samples(args.calibration_csv, cam, args.swap_eyes)
        except (OSError, ValueError, csv.Error) as exc:
            sys.exit(f"标定 CSV/帧读取失败：{exc}")
        if len(samples) < 3:
            sys.exit("标定至少需要 3 帧，且距离须覆盖至少 3 个不同值")
        scale_model, offset, r_squared = _fit_calibration(samples)
        model_w = cam._fast_stereonet_input_width or eye_w
        print(f"\nHITNet 标定拟合 R²={r_squared:.4f}")
        print("将以下两项写入 config.py 的 DepthCameraConfig：")
        print(f"  stereo_scale_mm_px = {scale_model * cfg.stereo_width / model_w:.2f}")
        print(f"  stereo_fast_stereonet_offset_px = {offset:.4f}")
        print("标定点预测误差：")
        for distance, measured in samples:
            estimate = scale_model / (measured - offset) if measured > offset else float("inf")
            error = (estimate - distance) / distance * 100.0
            print(f"  实测 {distance:.0f} mm -> 预测 {estimate:.0f} mm ({error:+.1f}%)")
        return

    if args.frame:
        left, right = _load_frame(args.frame)
    else:
        left = cv2.imread(os.path.join(args.dir, "left.png"))
        right = cv2.imread(os.path.join(args.dir, "right.png"))
        if left is None or right is None:
            sys.exit("找不到 left.png / right.png；可用 --frame 直接读取 stereo_record 中的 JPG")

    eye_w = left.shape[1]
    model_w = cam._fast_stereonet_input_width or eye_w
    disparity = _infer_disparity(cam, left, right, args.swap_eyes)
    median, mad, valid_ratio = _center_disparity(disparity)
    scale = cam._scale_for_eye_width(cam._stereo_effective_scale(), eye_w)
    scale_model = scale * model_w / eye_w
    offset = float(cfg.stereo_fast_stereonet_offset_px)
    estimate = scale_model / (median - offset) if median > offset else float("inf")
    expected_disp = scale_model / args.distance
    recommended_scale = (median - offset) * args.distance * cfg.stereo_width / model_w

    print(f"当前测距 {estimate:.0f} mm，已知距离 {args.distance:.0f} mm")
    print(f"中心视差 {median:.2f}px, MAD {mad:.2f}, 有效比例 {valid_ratio:.0%}")
    print(f"模型分辨率 scale={scale_model:.0f}, offset={offset:.2f}px, "
          f"按当前 scale 预期视差={expected_disp:.2f}px")
    print(f"若只做单点 scale 校正，建议 stereo_scale_mm_px ≈ {recommended_scale:.0f}")

    norm = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    out = os.path.join(args.dir, "hitnet_disparity.png")
    os.makedirs(args.dir, exist_ok=True)
    if not cv2.imwrite(out, cv2.applyColorMap(norm, cv2.COLORMAP_MAGMA)):
        sys.exit(f"无法保存诊断视差图：{out}")
    print("视差图已保存：", out)


if __name__ == "__main__":
    main()
