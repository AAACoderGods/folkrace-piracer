#!/usr/bin/env python3
"""
DECXIN-SM-2930V1 深度相机本地诊断。

目的：不动项目任何代码，只回答一个问题 ——
树莓派在哪一层认到了这块相机，它以什么格式提供数据。

四层逐层递进：
    1. USB 层    内核有没有枚举到这个 USB 设备
    2. 驱动层    有没有生成 /dev/video* 节点
    3. 格式层    节点暴露了哪些格式（关键：有没有 16 位深度格式）
    4. 应用层    OpenCV 实际能不能读到深度数据

用法：
    python3 test_depth_camera.py > depth_report.txt 2>&1
    把 depth_report.txt 的内容整段贴回即可。
也可以直接跑，复制屏幕输出。
"""

import glob
import os
import shutil
import subprocess
import sys

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    np = None
    NUMPY_AVAILABLE = False

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    cv2 = None
    CV2_AVAILABLE = False


def section(title):
    print()
    print("=" * 68)
    print(" " + title)
    print("=" * 68)


def run(cmd):
    """执行 shell 命令，返回 stdout+stderr 文本。绝不抛异常。"""
    try:
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=20
        )
        out = (result.stdout or "") + (result.stderr or "")
        return out.strip() or "(无输出)"
    except Exception as e:
        return "(命令执行失败: %s)" % e


def step_usb():
    section("1/4  USB 设备枚举")
    print(run("lsusb"))
    print()
    print("看什么：上面每一行是一个 USB 设备。")
    print("  - 出现新的一行（尤其带 Camera / Webcam / 厂商名），说明 USB 层认到了。")
    print("  - 完全没有新增行，就是硬件层问题：")
    print("    换数据线、换 USB 口、确认没和 Pi 供电口搞混、看相机指示灯。")


def step_nodes():
    section("2/4  /dev/video* 设备节点")
    nodes = sorted(glob.glob("/dev/video*"))
    if not nodes:
        print("找不到任何 /dev/video* 节点。")
        print()
        print("含义：USB 层可能认到了设备，但没有生成视频节点。")
        print("可能原因：内核 UVC 驱动没匹配上，或需要厂商内核模块。")
        print("尝试：拔插一次后看 dmesg | tail -30，以及 lsmod | grep -i uvc")
        return []

    for node in nodes:
        try:
            target = os.readlink(node)
            print("  %-16s -> %s" % (node, target))
        except OSError:
            print("  %s" % node)

    print()
    print("看什么：同时提供 RGB 和深度的相机通常生成两个节点，")
    print("比如 video0 是 RGB、video2 是深度。")
    return nodes


def step_v4l2(nodes):
    section("3/4  每个节点支持的像素格式")

    if shutil.which("v4l2-ctl") is None:
        print("没装 v4l-utils，无法枚举格式。先装：")
        print("  sudo apt install -y v4l-utils")
        print("装好后重跑本脚本。")
        return

    print("--- v4l2-ctl --list-devices ---")
    print(run("v4l2-ctl --list-devices"))
    print()

    for node in nodes:
        print("--- %s 的格式列表 ---" % node)
        print(run("v4l2-ctl -d %s --list-formats-ext" % node))
        print()

    print("这是最关键的一步。重点找 16 位深度格式：")
    print("  'Z16 ' / 'Y16 ' / 'Y12 ' / 'Z12 '  —— 带空格的 4 字符码")
    print("  或描述里带 Depth / 16-bit / TOF 的条目")
    print()
    print("若所有节点只有 YUYV / MJPG / H264 这类 8 位格式，")
    print("说明这块相机没有通过 UVC 暴露深度流 —— 那就要走厂商 SDK。")


def classify(frame):
    """把帧的形状翻译成「这是不是深度数据」。"""
    if not NUMPY_AVAILABLE:
        return ""
    if frame.dtype == np.uint16:
        return "uint16 单通道 ==> 这就是深度数据。"
    if frame.dtype == np.uint8 and frame.ndim == 3 and frame.shape[2] == 2:
        return "uint8 双通道 ==> 16 位原始数据，可 view 成 uint16，是深度。"
    if frame.dtype == np.uint8 and frame.ndim == 3 and frame.shape[2] == 3:
        return "8 位三通道 RGB ==> 深度已丢失，CONVERT_RGB=0 未生效。"
    if frame.dtype == np.uint8 and frame.ndim == 2:
        return "8 位单通道灰度 ==> 不是 16 位深度。"
    return ""


def sample_pixel(frame):
    try:
        px = frame[frame.shape[0] // 2, frame.shape[1] // 2]
        return str(px)
    except Exception:
        return "(取样失败)"


def try_read(node, backend_name, backend):
    cap = None
    try:
        cap = cv2.VideoCapture(node, backend)
        if not cap.isOpened():
            return "%s: 打不开" % backend_name

        prop = getattr(cv2, "CAP_PROP_CONVERT_RGB", None)
        if prop is not None:
            cap.set(prop, 0)

        ok, frame = cap.read()
        if not ok or frame is None:
            return "%s: 打开了但读不出帧" % backend_name

        text = "%s: shape=%s dtype=%s 中心像素=%s" % (
            backend_name, frame.shape, frame.dtype, sample_pixel(frame)
        )
        note = classify(frame)
        if note:
            text += "\n         " + note
        return text
    except Exception as e:
        return "%s: 异常 %s" % (backend_name, e)
    finally:
        if cap is not None:
            cap.release()


def try_fourcc(node, fourcc):
    cap = None
    try:
        backend = getattr(cv2, "CAP_V4L2", cv2.CAP_ANY)
        cap = cv2.VideoCapture(node, backend)
        if not cap.isOpened():
            return "节点打不开"

        prop = getattr(cv2, "CAP_PROP_CONVERT_RGB", None)
        if prop is not None:
            cap.set(prop, 0)
        if len(fourcc) == 4:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        ok, frame = cap.read()
        if ok and frame is not None:
            note = classify(frame)
            text = "OK  shape=%s dtype=%s 中心像素=%s" % (
                frame.shape, frame.dtype, sample_pixel(frame)
            )
            if note:
                text += "\n         " + note
            return text
        return "读不到帧"
    except Exception as e:
        return "异常 %s" % e
    finally:
        if cap is not None:
            cap.release()


def step_cv2(nodes):
    section("4/4  OpenCV 实际读取")

    if not CV2_AVAILABLE:
        print("没装 OpenCV (cv2)，跳过这一步。")
        print("  pip install opencv-python")
        return

    print("OpenCV 版本: %s" % cv2.__version__)
    print()

    print("--- 4a) 默认方式逐节点读取 ---")
    for node in nodes:
        print("  节点 %s" % node)
        print("    " + try_read(node, "CAP_V4L2", getattr(cv2, "CAP_V4L2", cv2.CAP_ANY)))
        print("    " + try_read(node, "CAP_ANY", cv2.CAP_ANY))
    print()

    print("--- 4b) 指定 fourcc 逐个尝试 ---")
    candidates = ["Z16 ", "Y16 ", "Y12 ", "Z12 ", "GREY", "YUYV"]
    for node in nodes:
        print("  节点 %s" % node)
        for fourcc in candidates:
            print("    fourcc='%s' -> %s" % (fourcc, try_fourcc(node, fourcc)))
    print()


def main():
    print("=" * 68)
    print(" PiRacer Pro 深度相机诊断 (DECXIN-SM-2930V1)")
    print("=" * 68)
    print("Python %s | numpy: %s | cv2: %s" % (
        sys.version.split()[0],
        "有" if NUMPY_AVAILABLE else "无",
        "有" if CV2_AVAILABLE else "无",
    ))

    step_usb()
    nodes = step_nodes()
    step_v4l2(nodes)

    if nodes:
        step_cv2(nodes)
    else:
        section("4/4  OpenCV 实际读取")
        print("没有 /dev/video* 节点，跳过。先解决第 2 层的问题。")

    section("结论怎么读")
    print("  第 3 层有 'Z16 ' / 'Y16 '")
    print("      -> 相机确实有 UVC 深度流。把那个节点名和 fourcc 填进")
    print("         config.py 的 depth.uvc_device / depth.uvc_fourcc 即可。")
    print()
    print("  第 4a 读到 uint16 或 (h,w,2)")
    print("      -> 深度管线是通的，只是默认节点选错了，改 uvc_device。")
    print()
    print("  第 4a 读到 8 位三通道")
    print("      -> CONVERT_RGB=0 没生效；看第 4b 哪个 fourcc 能读回 uint16。")
    print()
    print("  第 3 层全是 8 位格式")
    print("      -> 这块相机不走 UVC 暴露深度，需要厂商 SDK；")
    print("         请把 SDK 包名和读取一帧的示例代码发给我。")
    print()
    print("  没有 /dev/video* 节点")
    print("      -> 纯硬件/驱动问题，与代码无关。")


if __name__ == "__main__":
    main()
