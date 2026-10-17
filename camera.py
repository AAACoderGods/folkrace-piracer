"""
Camera abstraction module supporting Picamera2 for Raspberry Pi 4/5
and fallback to OpenCV VideoCapture or synthetic frames for local testing.
"""

import logging
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


class Camera:
    """Wrapper around Picamera2 with fallback support."""

    def __init__(self, width: int = 640, height: int = 480, framerate: int = 30,
                 format: str = "RGB888", split_stereo_frame: str = "auto",
                 stereo_eye: str = "left",
                 stereo_capture_width: int = 1280,
                 stereo_capture_height: int = 480,
                 prefer_v4l2: bool = False):
        self.width = width
        self.height = height
        self.framerate = framerate
        self.format = format
        # "auto" | "always" | "never" -- see CameraConfig.split_stereo_frame.
        self.split_stereo_frame = split_stereo_frame
        # "left" | "right" -- which half of a side-by-side pair to keep.
        self.stereo_eye = stereo_eye
        # 拆分前的原始尺寸（左右两眼并排的总宽 x 单眼高）。只有
        # split_stereo_frame="always" 时生效，见 _compute_capture_size()。
        self.stereo_capture_width = int(stereo_capture_width)
        self.stereo_capture_height = int(stereo_capture_height)
        # 双目模式下走 cv2.VideoCapture(V4L2) 直连而不是 Picamera2。
        # Picamera2 会经过 libcamera 的软处理（tone mapping / 缩放），对一块
        # 只有 MJPG 的双目 UVC 相机来说这是画质的主要损失来源。只有在双目模式
        # 下才启用，CSI 相机不受影响。
        self.prefer_v4l2 = bool(prefer_v4l2)
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
                     "prefer_v4l2=%s, target %dx%d)",
                     self._capture_size[0], self._capture_size[1],
                     self.split_stereo_frame, self.prefer_v4l2,
                     self.width, self.height)

        # 双目 + 需要 V4L2 直连时，优先这里。失败再落回 Picamera2，
        # 这样即使 video0 被占用或不是这台相机，程序也能起来。
        if self.prefer_v4l2 and (self.split_stereo_frame or "").lower() == "always":
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
                # Warm up
                time.sleep(0.5)
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

        "always" means the caller has told us the frame is a dual-lens pair. In
        that case we need the device's native side-by-side geometry, or the
        per-eye crop afterwards is half of a scaled pair and the depth pipeline
        gets a squashed image. "auto"/"never" stay at the target size so a normal
        single-lens camera is not forced into a resolution it does not offer.
        """
        if (self.split_stereo_frame or "auto").lower() != "always":
            return (int(self.width), int(self.height))
        return (max(int(self.width), self.stereo_capture_width),
                max(int(self.height), self.stereo_capture_height))

    def _open_v4l2_capture(self, why: str) -> bool:
        """
        Try `cv2.VideoCapture(0)` at the configured capture size.

        Called from start() as either the preferred path (prefer_v4l2 + stereo
        mode) or the fallback after Picamera2 failed. Returns True when a live
        handle is open; on any failure the handle is released and Picamera2 is
        still available as a last resort, so this never turns a working run into
        a broken one.
        """
        if cv2 is None:
            return False
        try:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                logging.warning("OpenCV VideoCapture (%s) could not open index 0.", why)
                self._release_cap()
                return False
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._capture_size[0])
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._capture_size[1])
            self.cap.set(cv2.CAP_PROP_FPS, self.framerate)
            self.is_running = True
            logging.info("✅ OpenCV VideoCapture (%s) started at %dx%d @ %dfps.",
                         why, self._capture_size[0], self._capture_size[1], self.framerate)
            return True
        except Exception as e:
            logging.warning(f"OpenCV VideoCapture ({why}) raised: {e}")
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
