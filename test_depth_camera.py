#!/usr/bin/env python3
"""
DECXIN-SM-2930V1 深度相机本地诊断。

目的：不动项目任何代码，只回答一个问题 ——
树莓派在哪一层认到了这块相机，它以什么格式提供数据。

五层逐层递进：
    1. USB 层        内核有没有枚举到这个 USB 设备
    2. 驱动层        有没有生成 /dev/video* 节点
    3. 格式层        节点暴露了哪些格式（关键：有没有 16 位深度格式）
    4. 应用层        OpenCV 实际读到什么，以及设备协商成了哪个格式
    5. 扩展单元层    有没有 UVC XU 控制能把深度流切出来

第 4 步的「协商格式回读」是关键：YUYV 彩色和 16 位深度都是
2 字节/像素，CONVERT_RGB=0 之后 OpenCV 交出的形状一模一样 (h, w, 2)。
只看形状会把彩色误判成深度，所以本脚本一律回读 CAP_PROP_FOURCC，
以设备实际协商的格式为准。

用法：
    python3 test_depth_camera.py > depth_report.txt 2>&1
    把 depth_report.txt 的内容整段贴回即可。
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

# 4:2:2 彩色格式：每像素 2 字节，和 16 位深度形状相同，但不是深度。
COLOR_FOURCCS = {"YUYV", "UYVY", "YVYU", "VYUY", "MJPG", "JPEG"}
# 16 位深度/灰度格式。比较时统一去空格、转大写。
DEPTH_FOURCCS = {"Z16", "Y16", "Y12", "Z12", "Y14", "Y10", "GREY"}


def section(title):
    print()
    print("=" * 68)
    print(" " + title)
    print("=" * 68)


def run(cmd):
    """执行 shell 命令，返回 stdout+stderr 文本。绝不抛异常。"""
    try:
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=25
        )
        out = (result.stdout or "") + (result.stderr or "")
        return out.strip() or "(无输出)"
    except Exception as e:
        return "(命令执行失败: %s)" % e


def fourcc_str(value):
    """把 OpenCV 的 CAP_PROP_FOURCC 数值还原成 4 字符码。"""
    try:
        value = int(value)
    except Exception:
        return "?"
    if value <= 0:
        return "-"
    return "".join(chr((value >> (8 * i)) & 0xFF) for i in range(4))


def norm_fourcc(text):
    return (text or "").upper().strip()


def classify(frame, negotiated=""):
    """结合帧形状和协商格式，判断这是不是深度数据。"""
    if not NUMPY_AVAILABLE:
        return ""
    neg = norm_fourcc(negotiated)

    if frame.dtype == np.uint16:
        return "uint16 单通道 —— 这是深度数据（形状已无歧义）。"

    if frame.dtype == np.uint8 and frame.ndim == 3 and frame.shape[2] == 2:
        if neg in COLOR_FOURCCS:
            return ("uint8 双通道，但协商格式是 '%s'（4:2:2 彩色）—— "
                    "这是彩色流被 CONVERT_RGB=0 拆成两字节/像素，不是深度。" % negotiated)
        if neg in DEPTH_FOURCCS:
            return ("uint8 双通道，协商格式 '%s' —— 16 位深度，可 view 成 uint16。" % negotiated)
        return ("uint8 双通道，协商格式 '%s' 无法断定 —— 以回读的格式为准，"
                "形状本身说明不了问题。" % negotiated)

    if frame.dtype == np.uint8 and frame.ndim == 3 and frame.shape[2] == 3:
        return "8 位三通道 RGB —— 深度已丢失，CONVERT_RGB=0 未生效。"
    if frame.dtype == np.uint8 and frame.ndim == 2:
        return "8 位单通道灰度 —— 不是 16 位深度（除非协商格式是 GREY，且你确实要灰度）。"
    return ""


def sample_pixel(frame):
    try:
        px = frame[frame.shape[0] // 2, frame.shape[1] // 2]
        return str(px)
    except Exception:
        return "(取样失败)"


def step_usb():
    section("1/5  USB 设备枚举")
    print(run("lsusb"))
    print()
    print("看什么：上面每一行是一个 USB 设备。")
    print("  - 出现新的一行（尤其带 Camera / Webcam / 厂商名），说明 USB 层认到了。")
    print("  - 完全没有新增行，就是硬件层问题：")
    print("    换数据线、换 USB 口、确认没和 Pi 供电口搞混、看相机指示灯。")


def step_nodes():
    section("2/5  /dev/video* 设备节点")
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
    section("3/5  每个节点支持的像素格式")

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
    print("说明这块相机没有通过 UVC 暴露深度流 —— 那就要走厂商 SDK，")
    print("或把双目画面切成左右两幅自己在主机端算视差。")


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

        negotiated = fourcc_str(cap.get(cv2.CAP_PROP_FOURCC))
        text = "%s: shape=%s dtype=%s 协商格式='%s' 中心像素=%s" % (
            backend_name, frame.shape, frame.dtype, negotiated, sample_pixel(frame)
        )
        note = classify(frame, negotiated)
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
        if not ok or frame is None:
            return "读不到帧"

        negotiated = fourcc_str(cap.get(cv2.CAP_PROP_FOURCC))
        accepted = "接受" if norm_fourcc(negotiated) == norm_fourcc(fourcc) else "忽略"
        note = classify(frame, negotiated)
        text = "请求='%s' 协商='%s' (%s) shape=%s dtype=%s 中心像素=%s" % (
            fourcc, negotiated, accepted, frame.shape, frame.dtype, sample_pixel(frame)
        )
        if note:
            text += "\n         " + note
        return text
    except Exception as e:
        return "异常 %s" % e
    finally:
        if cap is not None:
            cap.release()


def step_cv2(nodes):
    section("4/5  OpenCV 实际读取")

    if not CV2_AVAILABLE:
        print("没装 OpenCV (cv2)，跳过这一步。")
        print("  pip install opencv-python")
        return

    print("OpenCV 版本: %s" % cv2.__version__)
    print()
    print("注意看 '协商格式' —— 这才是判据。shape 是 (h, w, 2) 既有可能是")
    print("16 位深度，也有可能是 YUYV 彩色，光看 shape 分不出来。")
    print()

    print("--- 4a) 默认方式逐节点读取 ---")
    for node in nodes:
        print("  节点 %s" % node)
        print("    " + try_read(node, "CAP_V4L2", getattr(cv2, "CAP_V4L2", cv2.CAP_ANY)))
        print("    " + try_read(node, "CAP_ANY", cv2.CAP_ANY))
    print()

    print("--- 4b) 指定 fourcc 逐个尝试 ---")
    print("     '接受' 表示设备真的切到了该格式；'忽略' 表示它照旧吐原来的流。")
    candidates = ["Z16 ", "Y16 ", "Y12 ", "Z12 ", "GREY", "YUYV"]
    for node in nodes:
        print("  节点 %s" % node)
        for fourcc in candidates:
            print("    %s" % try_fourcc(node, fourcc))
    print()


def step_xu(nodes):
    section("5/5  UVC 控制与扩展单元 (XU)")

    print("有些相机把深度藏在 UVC 扩展单元后面，要发一条控制才会切出来。")
    print("下面把控制列表和 USB 描述符都打出来，找 depth / disparity / IR / ToF 字样。")
    print()

    # DECXIN 的节点是 video1(流) + video2(metadata)；找不到就退回全部节点。
    targets = [n for n in nodes if n in ("/dev/video1", "/dev/video2")] or nodes

    for node in targets:
        print("--- %s ---" % node)

        if shutil.which("v4l2-ctl"):
            print("[标准控制列表 v4l2-ctl --list-ctrls]")
            print(run("v4l2-ctl -d %s --list-ctrls" % node))
            print()
            print("[全部 kcontrols（找 XU / processing unit / depth）]")
            print(run("v4l2-ctl -d %s --all" % node))
            print()
        else:
            print("(缺 v4l2-ctl，跳过控制列表)")

        if shutil.which("uvcdynctrl"):
            print("[uvcdynctrl 扩展单元]")
            print(run("uvcdynctrl -d %s -l" % node))
            print()
        else:
            print("(缺 uvcdynctrl，跳过扩展单元；可选：sudo apt install uvcdynctrl)")
        print()

    print("[USB 描述符：找 Extension Unit / Processing Unit / Depth / Disparity]")
    print("完整描述符需要 root；下面这条不需要，能显示接口类与单元名。")
    print(run("lsusb -v -d 1bcf:2d4f 2>/dev/null | "
              "grep -iE 'unit|depth|disparity|interface|bInterfaceClass|iInterface'"))
    print()

    print("=" * 68)
    print(" 这一层怎么读")
    print("=" * 68)
    print("  出现 depth / disparity / ToF / IR 字样，或某个 Extension Unit 的")
    print("  名字看起来像在控制深度")
    print("      -> 深度被 XU 关着，还没研究透。把这几行原文发回来，")
    print("         我去查对应的 UVC 控制位，写一段切换代码。")
    print()
    print("  全是常规的控制（亮度、曝光、白平衡那类），没有任何单元名像深度")
    print("      -> 相机不通过 UVC 暴露深度。剩下的路只有：")
    print("         1) 厂商 SDK（如果有的话）")
    print("         2) 主机端双目匹配（跑 test_stereo_split.py 先确认是双目）")


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
        section("4/5  OpenCV 实际读取")
        print("没有 /dev/video* 节点，跳过。先解决第 2 层的问题。")

    step_xu(nodes)

    section("结论怎么读")
    print("  第 3 层出现 'Z16 ' / 'Y16 '")
    print("      -> 相机有 UVC 深度流。把节点名和 fourcc 填进")
    print("         config.py 的 depth.uvc_device / depth.uvc_fourcc 即可。")
    print()
    print("  第 4a 协商格式是 Z16/Y16/Y12/Z12")
    print("      -> 深度管线通，按上面填配置即可。")
    print()
    print("  第 4a 协商格式是 YUYV/MJPG")
    print("      -> 这是彩色流，不是深度。设备忽略了深度格式请求。")
    print("         结合第 3 层的格式表判断，多半没有 UVC 深度流。")
    print()
    print("  第 5 层没有任何 XU / depth 字样")
    print("      -> 相机不通过 UVC 暴露深度，需要厂商 SDK，")
    print("         或把双目画面切成左右两幅在主机端算视差。")
    print()
    print("  没有 /dev/video* 节点")
    print("      -> 纯硬件/驱动问题，与代码无关。")


if __name__ == "__main__":
    main()
