"""
Camera abstraction module supporting Picamera2 for Raspberry Pi 4/5
and fallback to OpenCV VideoCapture or synthetic frames for local testing.
"""

import glob
import logging
import os
import re
import threading
import time

import numpy
import numpy as np

try:
    from picamera2 import Picamera2
    PICAMERA2_AVAILABLE = True
except ImportError:
    PICAMERA2_AVAILABLE = False

try:
    import cv2
except ImportError:
    cv2 = None

# Aspect ratio above which a frame is assumed to be two eyes side by side. The
# DECXIN-SM-2930V1 lists 1280x480 / 1600x600 / 2560x960 (2.67), 2560x1080 (2.37)
# and 3840x1080 / 2560x720 (3.56). A single-lens camera is 4:3 (1.33) or 16:9
# (1.78), so 2.2 separates them without touching any normal footage.
_STEREO_ASPECT_THRESHOLD = 2.2

# Where the kernel ties a /dev/videoN node to the bus it hangs off. Used to keep
# CSI / codec nodes out of the V4L2 candidate list: on a Pi with no CSI module
# /dev/video0 still exists, opens cleanly, and never produces a frame.
_SYSFS_V4L = "/sys/class/video4linux"


def _is_usb_video_node(device) -> bool:
    """True when the node hangs off a USB bus, rather than CSI / codec / ISP."""
    if not isinstance(device, str):
        return True
    if not device.startswith("/dev/"):
        return True
    if not os.path.isdir(_SYSFS_V4L):
        return True
    try:
        real = os.path.realpath(os.path.join(_SYSFS_V4L, os.path.basename(device), "device"))
    except OSError:
        return True
    return "/usb" in real


def _device_sort_key(path: str):
    """Natural order for /dev/video2 < /dev/video10."""
    match = re.search(r"(\d+)$", path)
    return (int(match.group(1)) if match else 1 << 30, path)


def _fourcc_str(value) -> str:
    """把 CAP_PROP_FOURCC 的整数还原成四字符码，用于日志诊断。"""
    try:
        value = int(value)
    except (TypeError, ValueError):
        return ""
    if value <= 0:
        return ""
    return "".join(chr((value >> (8 * i)) & 0xFF) for i in range(4)).strip()


class Camera:
    """Wrapper around Picamera2 with fallback support."""

    def __init__(self, width: int = 640, height: int = 480, framerate: int = 30,
                 format: str = "RGB888", split_stereo_frame: str = "auto",
                 stereo_eye: str = "left",
                 stereo_capture_width: int = 1280,
                 stereo_capture_height: int = 480,
                 prefer_v4l2: bool = False,
                 v4l2_device: str = "",
                 v4l2_fourcc: str = "MJPG",
                 v4l2_auto_exposure_mode: float = -1.0,
                 v4l2_exposure_absolute: float = -1.0,
                 v4l2_gain: float = -1.0):
        self.width = width
        self.height = height
        self.framerate = framerate
        self.format = format
        # "auto" | "always" | "never" -- see CameraConfig.split_stereo_frame.
        self.split_stereo_frame = split_stereo_frame
        # "left" | "right" -- which half of a side-by-side pair to keep.
        self.stereo_eye = stereo_eye
        # 拆分前的原始尺寸（左右两眼并排的总宽 x 单眼高）。只有
        # split_stereo_frame="always"（或在 "auto" 下 width/height 恰好是一只眼
        # 的尺寸）时生效，见 _compute_capture_size()。
        self.stereo_capture_width = int(stereo_capture_width)
        self.stereo_capture_height = int(stereo_capture_height)
        # 双目模式下走 cv2.VideoCapture(V4L2) 直连而不是 Picamera2。
        # Picamera2 会经过 libcamera 的软处理（tone mapping / 缩放），对一块
        # 只有 MJPG 的双目 UVC 相机来说这是画质的主要损失来源。只有在双目模式
        # 下才启用，CSI 相机不受影响。
        self.prefer_v4l2 = bool(prefer_v4l2)
        # V4L2 节点。空串 = 自动扫描 USB 视频节点。
        #
        # 之前这里硬编码 cv2.VideoCapture(0)。在这台 Pi 上 /dev/video0 是
        # CSI（unicam）节点：能打开，但永远读不出帧。Camera.start() 因此
        # 报告成功，capture_frame() 每帧落到合成假图 —— 车实际在"看"一条
        # 画出来的路。真正的双目相机是 /dev/video1。填 "/dev/video1" 或
        # "1" 可强制指定；留空则按 USB 节点扫描，跳过 CSI / codec。
        self.v4l2_device = (v4l2_device or "").strip()
        # V4L2 采集时强制使用的像素格式。
        #
        # 这是之前只有 25fps 的根因：不设置 FOURCC 时驱动会选 YUYV（未压缩），
        # 1280x480@30fps ≈ 37 MB/s，正好卡在 USB 2.0 的实测上限附近，驱动自动
        # 降级到 25fps 甚至更低。MJPG 只要 1/10 的带宽，能跑满 30fps。
        # 恢复成 "auto" 或空串表示"不干预，让驱动自己选"。
        self.v4l2_fourcc = (v4l2_fourcc or "").strip()
        # 三个可选的 V4L2 控制量，默认 -1 表示"不要碰设备"。
        #
        # 曝光/增益的必要性：低光下驱动会拉长曝光时间，帧率被压到 25/20/15fps，
        # 同时噪点增加导致 SGBM 匹配失败 → 深度读数为 0。手动固定曝光能同时
        # 解决这两个问题。但不同 OpenCV 后端对 CAP_PROP_AUTO_EXPOSURE 的枚举
        # 理解不一致（0.25/0.75 与 1/3 都见过），所以默认关闭、由用户实验。
        self.v4l2_auto_exposure_mode = float(v4l2_auto_exposure_mode)
        self.v4l2_exposure_absolute = float(v4l2_exposure_absolute)
        self.v4l2_gain = float(v4l2_gain)
        self.picam2 = None
        self.cap = None
        self.is_running = False
        # 采集决定只记录一次；30 Hz 下每帧打日志会把真正有用的行冲掉。
        self._split_logged = False
        # 实际采集尺寸，start() 里决定。
        self._capture_size = (int(width), int(height))
        # 最近一帧原始（未拆分）图像。深度相机靠这块共享帧工作，避免把同一块
        # UVC 双目设备打开两次——两个句柄互相抢帧会让左右眼来自不同时刻，
        # 视差解出来的距离全错。
        self._raw_frame = None
        self._raw_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self):
        """Initialize and start the camera stream."""
        self._capture_size = self._compute_capture_size()
        logging.info("Camera: capture size %dx%d (split_stereo_frame=%s, "
                     "prefer_v4l2=%s, v4l2_device=%s, target %dx%d)",
                     self._capture_size[0], self._capture_size[1],
                     self.split_stereo_frame, self.prefer_v4l2,
                     self.v4l2_device or "(auto-scan USB)", self.width, self.height)

        # 双目模式下优先走 V4L2 直连。失败再落回 Picamera2，这样即使
        # video1 被占用或不是这台相机，程序也能起来。
        if self.prefer_v4l2 and (self.split_stereo_frame or "").lower() != "never":
            if self._open_v4l2_capture("preferred"):
                return

        if not PICAMERA2_AVAILABLE:
            logging.warning("⚠️  'picamera2' module not imported. If using venv, ensure --system-site-packages was used.")

        if PICAMERA2_AVAILABLE:
            logging.info("Initializing Picamera2 (OV5647 compatible)...")
            try:
                self.picam2 = Picamera2()
                try:
                    cam_config = self.picam2.create_preview_configuration(
                        main={"size": self._capture_size}
                    )
                except Exception:
                    cam_config = self.picam2.create_video_configuration(
                        main={"size": self._capture_size, "format": "RGB888"}
                    )
                self.picam2.configure(cam_config)
                self.picam2.start()
                # Warm up and capture first frame for depth camera
                time.sleep(0.5)
                try:
                    warmup_frame = self.picam2.capture_array()
                    warmup_frame = numpy.rot90(warmup_frame, 2, (0, 1))
                    if warmup_frame is not None and len(warmup_frame.shape) == 3:
                        if warmup_frame.shape[2] == 4 and cv2 is not None:
                            warmup_frame = cv2.cvtColor(warmup_frame, cv2.COLOR_RGBA2BGR)
                        elif warmup_frame.shape[2] == 3 and cv2 is not None and self.format.startswith("RGB"):
                            warmup_frame = cv2.cvtColor(warmup_frame, cv2.COLOR_RGB2BGR)
                    self._remember_raw(warmup_frame)
                    logging.debug("Camera: warm-up frame captured for depth camera.")
                except Exception as e:
                    logging.debug(f"Camera: warm-up frame capture failed (non-critical): {e}")
                self.is_running = True
                logging.info(f"✅ Picamera2 started successfully "
                             f"({self._capture_size[0]}x{self._capture_size[1]} @ {self.framerate}fps)")
                return
            except Exception as e:
                logging.error(f"❌ Picamera2 failed to start: {e}", exc_info=True)
                if self.picam2 is not None:
                    try:
                        self.picam2.close()
                    except Exception:
                        pass
                    self.picam2 = None

        if self._open_v4l2_capture("fallback"):
            return

        logging.warning("⚠️  No physical camera could be opened. Running in synthetic mock frame mode.")
        self.is_running = True

    def _compute_capture_size(self):
        """
        How big to ask the device for.

        The two eyes arrive side by side in one frame, so the caller's
        width/height are the *per-eye* size and the device has to be asked for
        the double-wide pair, or the crop afterwards keeps half of a scaled
        frame and the depth pipeline gets a squashed image.

        "always" always requests the pair. "auto" requests it when the caller's
        width and height line up with exactly one eye of stereo_capture_width /
        stereo_capture_height -- that is the statement "these dimensions
        describe one eye of a stereo pair". Any other geometry keeps the
        single-lens behaviour, so a normal webcam is not forced into a
        resolution it does not offer. "never" always keeps the target size.
        """
        mode = (self.split_stereo_frame or "auto").lower()
        if mode == "never":
            return (int(self.width), int(self.height))
        looks_like_eye = (int(self.width) * 2 == int(self.stereo_capture_width)
                          and int(self.height) == int(self.stereo_capture_height))
        if mode == "auto" and not looks_like_eye:
            return (int(self.width), int(self.height))
        return (max(int(self.width), self.stereo_capture_width),
                max(int(self.height), self.stereo_capture_height))

    def _apply_v4l2_controls(self):
        """
        应用用户显式设置的 V4L2 控制量（曝光 / 增益）。

        全部为 -1（默认）时什么都不做：自动曝光对"有时候能读、有时候读不出"
        的场景其实更友好，只有确认是低光导致 SGBM 崩掉时才需要手动固定。
        每个 set 都单独 try：一个控制量不被驱动支持，不应该影响其余控制量的
        应用，更不应该影响相机启动。
        """
        if cv2 is None or self.cap is None:
            return
        if self.v4l2_auto_exposure_mode >= 0:
            try:
                self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, self.v4l2_auto_exposure_mode)
                logging.info("Camera: V4L2 auto-exposure mode -> %.3f "
                             "(value meaning depends on the OpenCV/V4L2 build)",
                             self.v4l2_auto_exposure_mode)
            except Exception as e:
                logging.warning("Camera: setting CAP_PROP_AUTO_EXPOSURE failed: %s", e)
        if self.v4l2_exposure_absolute >= 0:
            try:
                self.cap.set(cv2.CAP_PROP_EXPOSURE, self.v4l2_exposure_absolute)
                logging.info("Camera: V4L2 exposure -> %.3f", self.v4l2_exposure_absolute)
            except Exception as e:
                logging.warning("Camera: setting CAP_PROP_EXPOSURE failed: %s", e)
        if self.v4l2_gain >= 0:
            try:
                self.cap.set(cv2.CAP_PROP_GAIN, self.v4l2_gain)
                logging.info("Camera: V4L2 gain -> %.3f", self.v4l2_gain)
            except Exception as e:
                logging.warning("Camera: setting CAP_PROP_GAIN failed: %s", e)

    def _candidate_capture_devices(self):
        """
        V4L2 nodes worth trying, most likely first.

        Only USB nodes are returned by the auto-scan. On this Pi /dev/video0 is
        a CSI/unicam node that opens cleanly and never produces a frame, and the
        codec / ISP nodes below it behave the same way. Reading one yields no
        error, just a synthetic frame, so an open that "succeeds" there is more
        dangerous than useless.
        """
        candidates = []
        if self.v4l2_device:
            try:
                candidates.append(int(self.v4l2_device))
            except ValueError:
                candidates.append(self.v4l2_device)

        usb_nodes = [p for p in glob.glob("/dev/video*") if _is_usb_video_node(p)]
        for node in sorted(usb_nodes, key=_device_sort_key):
            if node not in candidates:
                candidates.append(node)
        return candidates

    def _open_v4l2_capture(self, why: str) -> bool:
        """
        Try each candidate node until one actually produces a frame.

        Reading one frame is the point. `isOpened()` alone is not enough: a node
        that opens but never delivers a frame used to be treated as success, and
        every later capture then fell through to the synthetic road, so the car
        followed a drawn line instead of the track.
        """
        if cv2 is None:
            return False
        for device in self._candidate_capture_devices():
            if self._try_open_v4l2(device, why):
                return True
        logging.warning("OpenCV VideoCapture (%s) found no usable USB video node.", why)
        return False

    def _try_open_v4l2(self, device, why: str) -> bool:
        try:
            backend = getattr(cv2, "CAP_V4L2", cv2.CAP_ANY)
            self.cap = cv2.VideoCapture(device, backend)
            if not self.cap.isOpened():
                logging.debug("V4L2: %s could not be opened.", device)
                self._release_cap()
                return False

            # 先设 FOURCC，再设分辨率。顺序不能反：部分 V4L2 驱动在设定格式时
            # 会顺带重置分辨率，先设尺寸后设格式会把尺寸请求丢掉。
            if len(self.v4l2_fourcc) == 4:
                self.cap.set(cv2.CAP_PROP_FOURCC,
                             cv2.VideoWriter_fourcc(*self.v4l2_fourcc))
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._capture_size[0])
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._capture_size[1])
            self.cap.set(cv2.CAP_PROP_FPS, self.framerate)
            self._apply_v4l2_controls()

            # 读回协商结果并打印。这是之前最缺的诊断信息：驱动拒绝 FOURCC / 只
            # 支持固定分辨率列表时，会静默降级到低分辨率或低帧率，而旧代码没有
            # 任何一个地方能让人看见"配置写的 30fps/1280x480，实际跑的是
            # 25fps/864x320"。这段日志就是那道唯一的光。
            actual_fourcc = _fourcc_str(self.cap.get(cv2.CAP_PROP_FOURCC))
            actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
            actual_fps = float(self.cap.get(cv2.CAP_PROP_FPS) or 0.0)
            logging.info(
                "V4L2 negotiated (%s, %s): %dx%d @ %.1ffps, fourcc '%s' "
                "(requested %dx%d @ %dfps, fourcc '%s')",
                device, why, actual_w, actual_h, actual_fps,
                actual_fourcc or "(unreported)",
                self._capture_size[0], self._capture_size[1], self.framerate,
                self.v4l2_fourcc or "(none)")

            if len(self.v4l2_fourcc) == 4 and actual_fourcc and actual_fourcc != self.v4l2_fourcc:
                logging.warning(
                    "V4L2: device ignored the requested pixel format '%s' and is "
                    "streaming '%s'. On a full-width stereo pair that format is the "
                    "usual reason for a 25fps cap -- run "
                    "`v4l2-ctl -d %s --list-formats-ext` and use one of the "
                    "listed formats.", self.v4l2_fourcc, actual_fourcc, device)
            if (actual_w and actual_h
                    and (actual_w != self._capture_size[0]
                         or actual_h != self._capture_size[1])):
                logging.warning(
                    "V4L2: device uses %dx%d, not the requested %dx%d. Depth scale "
                    "is adjusted automatically to the real eye width, but the vision "
                    "crop and the target width/height no longer line up.",
                    actual_w, actual_h,
                    self._capture_size[0], self._capture_size[1])

            # 关键验证：真正读一帧。isOpened() 为 True 还不够 —— 有些节点（比如
            # 无 CSI 模组时的 /dev/video0）能打开但永远不出帧，旧代码会把它当成功，
            # 之后每帧都落到合成假图，车实际上在"看"一条画出来的路。
            ok, test_frame = self.cap.read()
            if not ok or test_frame is None:
                logging.warning(
                    "V4L2: %s opened but produced no frame. Trying the next node.",
                    device)
                self._release_cap()
                return False
            self._remember_raw(test_frame)

            self.is_running = True
            logging.info("✅ OpenCV VideoCapture (%s) started at %dx%d @ %dfps using %s.",
                         why, self._capture_size[0], self._capture_size[1],
                         self.framerate, device)
            return True
        except Exception as e:
            logging.warning(f"OpenCV VideoCapture ({why}, {device}) raised: {e}")
            self._release_cap()
            return False

    def _release_cap(self):
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None

    def capture_frame(self) -> np.ndarray:
        """
        Capture a single frame as a NumPy array (BGR format for OpenCV processing).
        Returns:
            np.ndarray: BGR image frame (H, W, 3)
        """
        if not self.is_running:
            raise RuntimeError("Camera is not started. Call camera.start() first.")

        frame = None
        if PICAMERA2_AVAILABLE and self.picam2 is not None:
            try:
                frame = self.picam2.capture_array()
                # 180-degree rotation to undo the sensor's mounting orientation.
                # axes=(0, 1) is the conventional spelling; (1, 0) happens to give
                # the same result for k=2 but reads as if the axes were swapped.
                frame = numpy.rot90(frame, 2, (0, 1))
                if frame is not None and len(frame.shape) == 3:
                    if frame.shape[2] == 4 and cv2 is not None:
                        frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
                    elif frame.shape[2] == 3 and cv2 is not None and self.format.startswith("RGB"):
                        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                # 先缓存原始帧，再拆分。深度相机从这里拿到的就是真正未切割的
                # 双目帧，一眼左边一眼右边。
                self._remember_raw(frame)
                # Split before the vision pipeline sees anything. On a Pi with no
                # CSI module Picamera2 can enumerate a dual-lens UVC camera, and a
                # side-by-side pair would send the line follower into the seam
                # between the two eyes.
                frame = self._apply_stereo_split(frame)
            except Exception as e:
                logging.warning(f"Error capturing from Picamera2: {e}")

        elif self.cap is not None and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret and frame is not None:
                # 同上：深度相机要的是拆分前的完整双目帧。
                self._remember_raw(frame)
                # Split *before* resizing: the kept eye should be resized on its
                # own. Resizing first would squash the pair and then halve it,
                # leaving a half-width, half-height eye.
                frame = self._apply_stereo_split(frame)
                if (frame.shape[1], frame.shape[0]) != (self.width, self.height):
                    frame = cv2.resize(frame, (self.width, self.height))

        # Fallback synthetic frame with simulated road for testing without hardware
        if frame is None:
            frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            # Draw mock road (gray trapezoid) and white line in the middle
            if cv2 is not None:
                pts = np.array([
                    [self.width * 0.2, self.height],
                    [self.width * 0.4, self.height * 0.4],
                    [self.width * 0.6, self.height * 0.4],
                    [self.width * 0.8, self.height]
                ], np.int32)
                cv2.fillPoly(frame, [pts], (80, 80, 80))
                cv2.line(frame, (int(self.width * 0.5), int(self.height * 0.4)),
                         (int(self.width * 0.5), self.height), (255, 255, 255), 4)

        # Crop bottom 1/5 of the frame
        if frame is not None and len(frame.shape) == 3:
            crop_height = int(frame.shape[0] * 4 / 5)
            frame = frame[:crop_height, :, :]

        return frame

    def latest_stereo_frame(self):
        """
        Return the most recent *un-split* frame, or None if no frame yet.

        This is the shared feed for depth_camera.py's stereo backend. The same
        UVC dual-lens device must not be opened twice: libcamera (through
        Picamera2) or the V4L2 handle above already owns it here, so a second
        cv2.VideoCapture on the same node either fails outright -- which used to
        show up as "stereo fell back to mock" -- or, worse, succeeds and hands
        out frames interleaved with vision's, so the "left" and "right" halves
        that StereoBM receives come from different moments and the disparity map
        is pure noise.

        A reference is returned rather than a copy. capture_frame() rebinds
        self._raw_frame to a fresh array each frame, so a caller holding the old
        reference keeps a consistent (older) frame rather than seeing fields
        mutate under it.
        """
        with self._raw_lock:
            return self._raw_frame

    def _remember_raw(self, frame):
        """Cache the raw frame for latest_stereo_frame()."""
        if frame is None:
            return
        with self._raw_lock:
            self._raw_frame = frame

    def stop(self):
        """Stop and release camera resources."""
        if self.picam2 is not None:
            try:
                self.picam2.stop()
                self.picam2.close()
            except Exception as e:
                logging.error(f"Error stopping Picamera2: {e}")
            self.picam2 = None

        self._release_cap()

        with self._raw_lock:
            self._raw_frame = None

        self.is_running = False
        logging.info("Camera stopped.")

    # ------------------------------------------------------------------
    # Side-by-side stereo split
    # ------------------------------------------------------------------

    def _looks_like_stereo_pair(self, frame: np.ndarray) -> bool:
        """Decide whether a frame is two eyes squeezed into one image."""
        if frame is None or frame.ndim < 2:
            return False
        h, w = frame.shape[:2]
        if h <= 0 or w <= 0:
            return False
        return (w / float(h)) >= _STEREO_ASPECT_THRESHOLD

    def _split_stereo(self, frame: np.ndarray) -> np.ndarray:
        """
        Return the left (default) or right half of a side-by-side stereo frame.

        The pair is assumed to be already rectified -- the DECXIN outputs one
        image straight from the module with the eyes aligned on the same rows,
        so this is a straight vertical cut, not a warp.
        """
        h, w = frame.shape[:2]
        half = w // 2
        if half <= 0:
            return frame
        if (self.stereo_eye or "left").lower() == "right":
            eye = frame[:, half:2 * half]
        else:
            eye = frame[:, :half]
        return np.ascontiguousarray(eye)

    def _apply_stereo_split(self, frame: np.ndarray) -> np.ndarray:
        """
        Honour split_stereo_frame and return the frame the vision pipeline should
        see. "never" and a pass on the auto check both return the frame untouched.

        The first decision is always logged. When auto declines to split, the log
        prints the frame size and its aspect ratio, because the usual cause is a
        capture path that already rescaled the double-wide frame down to a normal
        ratio (e.g. Picamera2 preview fitting 1280x480 into 640x460). In that case
        the only fix is camera.split_stereo_frame="always".
        """
        if frame is None or self.split_stereo_frame == "never":
            return frame

        if self.split_stereo_frame != "always" and not self._looks_like_stereo_pair(frame):
            if not self._split_logged:
                self._split_logged = True
                h, w = frame.shape[:2]
                logging.info(
                    "Camera: not splitting frame %s (aspect %.2f < %.1f). If this "
                    "is actually a dual-lens pair, set camera.split_stereo_frame="
                    "'always'.", frame.shape, w / max(1, h), _STEREO_ASPECT_THRESHOLD)
            return frame

        out = self._split_stereo(frame)
        if not self._split_logged:
            self._split_logged = True
            logging.info(
                "Camera: side-by-side stereo frame %s -> %s (using the %s eye) for "
                "vision.", frame.shape, out.shape, self.stereo_eye)
        return out
