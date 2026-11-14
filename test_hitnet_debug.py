"""
HITNet 诊断：用 test_stereo_debug.py 存下的 left.png / right.png 跑一遍模型，
看它在已知距离的靶子上给出多大的视差，并反推出 scale。

用法（在 Pi 上，靶子放在 1.5 m 处、画面中央要有纹理）：
    python3 test_stereo_debug.py --distance 1500        # 先存 stereo_debug/left.png 等
    python3 test_hitnet_debug.py --distance 1500

输出：
  - 模型分辨率下中心区域的视差分位数
  - 按当前 config scale 算出的距离
  - 若真实距离已知，反推 "应有的视差" 和 "实测视差" 的比值
    比值 ~1  → 模型没问题，问题在别处（分位数/过滤/scale）
    比值 >>1 → 模型视差偏大（图像未校正、左右眼反了、画面纹理差）
"""
import argparse
import os
import sys

import cv2
import numpy as np

from config import DepthCameraConfig
from depth_camera import DepthCamera


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dir", default="stereo_debug")
    p.add_argument("--distance", type=float, required=True, help="靶子真实距离 mm")
    p.add_argument("--swap-eyes", action="store_true")
    args = p.parse_args()

    left = cv2.imread(os.path.join(args.dir, "left.png"))
    right = cv2.imread(os.path.join(args.dir, "right.png"))
    if left is None or right is None:
        sys.exit("找不到 left.png / right.png，先跑 test_stereo_debug.py")
    if args.swap_eyes:
        left, right = right, left

    cfg = DepthCameraConfig()
    cam = DepthCamera(cfg)
    if not cam._init_fast_stereonet():
        sys.exit("模型没加载成功（路径/onnxruntime？）")

    eye_w = left.shape[1]
    model_w = cam._fast_stereonet_input_width or eye_w
    disp = cam._disparity_from_fast_stereonet(left, right)
    if disp is None:
        sys.exit("推理失败")

    scale = cam._stereo_effective_scale()
    scale_model = scale * model_w / eye_w
    expect = scale_model / args.distance

    h, w = disp.shape
    roi = disp[h // 3: 2 * h // 3, w // 3: 2 * w // 3]
    q = np.percentile(roi, [5, 10, 25, 50, 75, 90, 95])
    print(f"眼宽 {eye_w}px, 模型宽 {model_w}px, config scale {scale:.0f} -> 模型分辨率 scale {scale_model:.0f}")
    print(f"在 {args.distance:.0f} mm 处，按 scale 期望视差 ≈ {expect:.1f} px（模型分辨率）")
    print("中心区域视差分位数 5/10/25/50/75/90/95: " + " ".join(f"{v:.1f}" for v in q))
    print("按 scale 换算的距离 mm（同分位数）:      " +
          " ".join(f"{scale_model / max(v, 1e-3):.0f}" for v in q))
    print(f"中位数视差 / 期望视差 = {q[3] / expect:.2f}")
    print(f"若中位数视差对应真实距离，则 scale_model 应为 {q[3] * args.distance:.0f}"
          f"（config 里 stereo_scale_mm_px ≈ {q[3] * args.distance * eye_w / model_w:.0f}）")

    norm = cv2.normalize(disp, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    out = os.path.join(args.dir, "hitnet_disparity.png")
    cv2.imwrite(out, cv2.applyColorMap(norm, cv2.COLORMAP_MAGMA))
    print("已保存", out)


if __name__ == "__main__":
    main()
