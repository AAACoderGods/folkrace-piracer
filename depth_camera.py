"""
Depth camera adapter for PiRacer Pro.

Backends:
    stereo    - DECXIN-SM-2930V1 and other dual-lens UVC modules. The device
                exposes only MJPG/YUYV colour and puts the two lenses side by
                side in one double-wide frame; depth is reconstructed on the
                host by stereo matching. This is the backend for the camera on
                the car, because it has no UVC depth format at all.
    decxin    - a UVC camera that really does expose a 16-bit depth format
                (Z16/Y16/...). Not this hardware; kept for other cameras.
    realsense - Intel RealSense via pyrealsense2.
    mock      - synthesises depth frames so the calling path can be exercised on
                a development machine with no hardware attached.

Design rules, in order of importance:

1. The depth camera is *optional*. Every failure mode -- library not installed,
   no device on the bus, backend raising during start, frames timing out -- ends
   in a fallback rather than an exception. A depth camera that is missing must
   leave the car following the line exactly as it did before, which is why
   nothing here is allowed to raise out of start().

2. Units are millimetres, everywhere. RealSense returns raw uint16 counts scaled
   by a per-device depth_scale, and the UVC device hands back a bare uint16 buffer
   with no scaling metadata at all; both are converted once, in the backend, so
   the rest of the codebase never has to think about device-specific units.

3. The depth camera only reports distance. It does not drive, does not steer,
   and does not know what mode the car is in -- main.py owns that decision.

4. The stream format is read back from the device, never inferred from shape.

5. The UVC device is opened by at most one component.

6. The UVC-based backends capture on a background thread.

7. Calibration is per eye *width*.
"""

import glob
import logging
import os
import re
import threading
import time
from typing import List, Optional, Tuple

import numpy as np

from config import DepthCameraConfig

# Each backend is probed once at import.
try:
    import pyrealsense2 as rs
    _REALSENSE_AVAILABLE = True
except ImportError:
    rs = None
    _REALSENSE_AVAILABLE = False

try:
    import cv2
    _CV2_AVAILABLE = True
except ImportError:
    cv2 = None
    _CV2_AVAILABLE = False


# Pixel formats that actually carry 16-bit depth over UVC.
_DEPTH_FOURCCS = {"Z16", "Y16", "Y12", "Z12", "Y14", "Y10"}

# 4:2:2 colour formats.
_COLOUR_FOURCCS = {"YUYV", "UYVY", "YVYU", "VYUY", "MJPG", "JPEG", "RGB3", "BGR3"}

# Where the kernel ties a /dev/videoN node to the bus it hangs off.
_SYSFS_V4L = "/sys/class/video4linux"


def _fourcc_str(value) -> str:
    """Turn a CAP_PROP_FOURCC integer back into its four-character code."""
    try:
        value = int(value)
    except (TypeError, ValueError):
        return ""
    if value <= 0:
        return ""
    return "".join(chr((value >> (8 * i)) & 0xFF) for i in range(4)).strip()


def _device_sort_key(path: str):
    """Natural order for /dev/video2 < /dev/video10."""
    match = re.search(r"(\d+)$", path)
    return (int(match.group(1)) if match else 1 << 30, path)


def _is_usb_video_node(device: str) -> bool:
    """True when the node hangs off a USB bus, rather than the CSI pipeline."""
    if not device.startswith("/dev/"):
        return True
    if not os.path.isdir(_SYSFS_V4L):
        return True
    try:
        real = os.path.realpath(os.path.join(_SYSFS_V4L, os.path.basename(device), "device"))
    except OSError:
        return True
    return "/usb" in real


def _sgbm_mode_flag(name: str) -> int:
    """
    Resolve a config string ("sgbm"/"3way"/"hh"/"hh4") into the OpenCV constant.

    Falls back to SGBM_3WAY on any unknown value, because that is the safest
    default on a Pi: the quality difference against "hh" is small at 640x480,
    but the runtime difference is large enough to decide whether the sampler
    hits its target rate.
    """
    if cv2 is None:
        return 0
    key = (name or "3way").strip().lower()
    table = {
        "sgbm": getattr(cv2, "STEREO_SGBM_MODE_SGBM", 0),
        "hh":   getattr(cv2, "STEREO_SGBM_MODE_HH", 1),
        "3way": getattr(cv2, "STEREO_SGBM_MODE_SGBM_3WAY", 2),
        "hh4":  getattr(cv2, "STEREO_SGBM_MODE_HH4", 3),
    }
    return table.get(key, table["3way"])


class DepthCamera:
    """
    Facade over the available depth backends, with a uniform frame interface.

    Usage mirrors Camera:
        cam = DepthCamera(config.depth, frame_source=camera)
        cam.start()
        colour, depth = cam.capture_frame()   # depth is mm, 0 means "no reading"
        cam.stop()
    """

    def __init__(self, config: DepthCameraConfig, frame_source=None):
        self.config = config
        self._frame_source = frame_source
        self.backend: str = "mock"
        self.is_running = False
        self._pipeline = None
        self._uvc = None
        self._uvc_lock = threading.Lock()
        self._uvc_fourcc = ""
        self._uvc_device_in_use = ""
        self._stereo_lock = threading.Lock()
        self._stereo_stop_event: Optional[threading.Event] = None
        self._stereo_thread: Optional[threading.Thread] = None
        self._stereo_latest: Optional[np.ndarray] = None
        self._stereo_external = False
        self._stereo_first_logged = False
        self._stereo_scale_logged = False
        # 连续无有效读数的帧数：达到 5 时打印一次完整的诊断提示，
        # 每个启动只打一次，避免 10Hz 下刷屏。
        self._stereo_empty_streak = 0
        self._stereo_empty_warned = False
        self._depth_scale_m = 0.001

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self):
        """Bring up the configured backend, falling back to mock on any failure."""
        requested = (self.config.backend or "mock").lower()

        if requested == "stereo":
            if not _CV2_AVAILABLE:
                logging.warning("Depth camera: 'stereo' requested but OpenCV is not "
                                "installed. pip install opencv-python")
            elif not (hasattr(cv2, "StereoSGBM_create") or hasattr(cv2, "StereoBM_create")):
                logging.warning("Depth camera: 'stereo' requested but this OpenCV "
                                "build has neither StereoSGBM_create nor "
                                "StereoBM_create.")
            elif self._start_stereo():
                self.backend = "stereo"
                self.is_running = True
                logging.info("✅ Depth camera started: stereo (UVC on %s, "
                             "eye %dx%d, scale=%.1f mm*px, %d disparities, %s mode)",
                             self._uvc_device_in_use,
                             self.config.stereo_width, self.config.stereo_height,
                             self._stereo_effective_scale(),
                             self.config.stereo_num_disparities,
                             self.config.stereo_sgbm_mode if self.config.stereo_use_sgbm else "BM")
                return
            else:
                logging.warning("Depth camera: stereo failed to start; "
                                "falling back to mock.")

        elif requested == "decxin":
            if not _CV2_AVAILABLE:
                logging.warning("Depth camera: 'decxin' requested but OpenCV is not "
                                "installed. pip install opencv-python")
            elif self._start_decxin():
                self.backend = "decxin"
                self.is_running = True
                logging.info("✅ Depth camera started: DECXIN (UVC on %s, fourcc '%s') "
                             "(%dx%d @ %dfps, %.3f mm/count)",
                             self._uvc_device_in_use,
                             self._uvc_fourcc or "(unreported)",
                             self.config.width, self.config.height,
                             self.config.fps, self.config.uvc_depth_unit_mm)
                return
            else:
                logging.warning("Depth camera: DECXIN (UVC) failed to start; "
                                "falling back to mock.")

        elif requested == "realsense":
            if not _REALSENSE_AVAILABLE:
                logging.warning("Depth camera: 'realsense' requested but pyrealsense2 "
                                "is not installed. pip install pyrealsense2")
            elif self._start_realsense():
                self.backend = "realsense"
                self.is_running = True
                logging.info("✅ Depth camera started: Intel RealSense "
                             "(%dx%d @ %dfps, scale %.4f m/count)",
                             self.config.width, self.config.height,
                             self.config.fps, self._depth_scale_m)
                return
            else:
                logging.warning("Depth camera: RealSense failed to start; "
                                "falling back to mock.")
        else:
            logging.info("Depth camera: backend '%s' is unknown; using mock.", requested)

        self.backend = "mock"
        self.is_running = True
        logging.warning("Depth camera: running the MOCK backend -- the forward "
                        "obstacle guard is INACTIVE (mock reports a clear 3 m wall). "
                        "Line following is unaffected.")

    def capture_frame(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Return one (colour BGR, depth mm) pair.

        The colour frame is None unless DepthCameraConfig.enable_color_stream is
        set. The depth frame is float32 in millimetres with 0 marking "no reading".
        """
        if not self.is_running:
            raise RuntimeError("DepthCamera is not started. Call start() first.")

        if self.backend == "stereo":
            return self._capture_stereo()
        if self.backend == "decxin":
            return self._capture_decxin()
        if self.backend == "realsense":
            return self._capture_realsense()
        return self._capture_mock()

    def stop(self):
        """Stop the stream and release the device. Safe to call repeatedly."""
        if self._stereo_stop_event is not None:
            self._stereo_stop_event.set()
        if self._stereo_thread is not None:
            self._stereo_thread.join(timeout=5.0)
            self._stereo_thread = None
        self._stereo_stop_event = None
        self._stereo_external = False
        with self._stereo_lock:
            self._stereo_latest = None

        if self._pipeline is not None:
            try:
                self._pipeline.stop()
            except Exception as e:
                logging.debug(f"Error stopping RealSense pipeline: {e}")
            self._pipeline = None

        with self._uvc_lock:
            if self._uvc is not None:
                try:
                    self._uvc.release()
                except Exception as e:
                    logging.debug(f"Error releasing UVC depth capture: {e}")
                self._uvc = None

        self._uvc_fourcc = ""
        self._uvc_device_in_use = ""
        self.is_running = False
        logging.info("Depth camera stopped.")

    # ------------------------------------------------------------------
    # Obstacle ranging
    # ------------------------------------------------------------------

    def measure_obstacle_mm(self, depth_frame: Optional[np.ndarray]) -> Optional[float]:
        """
        Nearest valid distance on the forward scan band, in millimetres.

        Three things this used to get wrong:

        1. It scanned a single ROW, so one dead scanline -- a white wall, a dark
           tyre, a specular reflection, a shadow under the bumper -- made the
           entire call return None even though the rest of the frame was fine.
           Fixed by scanning a vertical band, obstacle_scan_band_height_ratio
           wide, centred on obstacle_scan_y_ratio.

        2. It took min() BEFORE filtering by [min_depth_mm, max_depth_mm], so a
           single spurious close-range match poisoned the whole band: the code
           saw 120 mm, rejected it as out of range, and returned None -- even
           though 99% of the band was a clean 800 mm. Fixed by filtering first,
           then taking the statistic.

        3. It took a plain min() even on the surviving pixels, which makes a
           single erroneous match indistinguishable from a real wall and is a
           reliable source of phantom emergency stops. Fixed by taking the
           obstacle_percentile-th percentile: 10 by default, which is "how close
           does the nearest 10% of the band get". That is still conservative --
           a real obstacle covers many pixels, so it survives the percentile --
           but a lone match does not.

        Returns None when the band holds too few usable readings, which is the
        correct answer for a camera pointed at the sky or one that has just lost
        the device. Returning a large number there would quietly turn "no data"
        into "no obstacle", which is the same as switching the guard off.
        """
        if depth_frame is None or depth_frame.size == 0:
            return None

        h, w = depth_frame.shape[:2]
        if h == 0 or w == 0:
            return None

        # 纵向：以 obstacle_scan_y_ratio 为中心，向上下各延伸 ±band_half_h。
        center_y = int(np.clip(self.config.obstacle_scan_y_ratio, 0.0, 1.0) * (h - 1))
        band_half_h = max(1, int(round(h * float(self.config.obstacle_scan_band_height_ratio))))
        y0 = max(0, center_y - band_half_h)
        y1 = min(h, center_y + band_half_h + 1)

        # 横向：画面中心的一条竖带，宽度为 obstacle_scan_width_ratio × w。
        half_w = int(np.clip(self.config.obstacle_scan_width_ratio, 0.0, 1.0) * w / 2.0)
        x0 = max(0, w // 2 - half_w)
        x1 = min(w, w // 2 + half_w)

        if x1 <= x0 or y1 <= y0:
            return None

        band = depth_frame[y0:y1, x0:x1]

        # 先按 [min_depth_mm, max_depth_mm] 过滤，再统计。
        valid_mask = (band >= self.config.min_depth_mm) & (band <= self.config.max_depth_mm)
        valid = band[valid_mask]
        if valid.size < int(self.config.obstacle_min_valid_pixels):
            return None

        # 分位数抗单个错误点；语义见上方注释。
        percentile = float(self.config.obstacle_percentile)
        percentile = max(1.0, min(99.0, percentile))
        try:
            near = float(np.percentile(valid, percentile))
        except Exception:
            return None
        if not (self.config.min_depth_mm <= near <= self.config.max_depth_mm):
            return None
        return near

    def depth_grid(self, depth_frame: Optional[np.ndarray]) -> Optional[dict]:
        """
        Coarse depth map for the web console's cursor readout.
        """
        if depth_frame is None or depth_frame.size == 0:
            return None

        h, w = depth_frame.shape[:2]
        if h == 0 or w == 0:
            return None

        gw = max(1, min(int(self.config.ui_grid_width), w))
        gh = max(1, min(int(self.config.ui_grid_height), h))
        ys = np.clip(np.arange(gh) * h // gh, 0, h - 1)
        xs = np.clip(np.arange(gw) * w // gw, 0, w - 1)
        cells = np.asarray(depth_frame[np.ix_(ys, xs)], dtype=np.float32)

        valid = (cells >= self.config.min_depth_mm) & (cells <= self.config.max_depth_mm)
        cells = np.where(valid, cells, 0.0)
        return {
            "w": int(gw),
            "h": int(gh),
            "mm": [int(round(float(v))) for v in cells.reshape(-1)],
        }

    # ------------------------------------------------------------------
    # Stereo backend
    # ------------------------------------------------------------------

    def _stereo_effective_scale(self) -> float:
        """The mm*px constant used to turn disparity into depth."""
        scale = float(self.config.stereo_scale_mm_px)
        if scale > 0.0:
            return scale
        focal = float(self.config.stereo_focal_px)
        baseline = float(self.config.stereo_baseline_mm)
        if focal > 0.0 and baseline > 0.0:
            return focal * baseline
        return 0.0

    def _scale_for_eye_width(self, baseline_scale: float, eye_w: int) -> float:
        """
        把"在 config.stereo_width 宽度下标定的 scale"换算到实际的单眼宽度上。
        """
        ref_w = int(self.config.stereo_width or 0)
        if ref_w <= 0 or eye_w <= 0 or eye_w == ref_w:
            return baseline_scale
        scaled = baseline_scale * (float(eye_w) / float(ref_w))
        if not self._stereo_scale_logged:
            self._stereo_scale_logged = True
            logging.warning(
                "Depth camera: eye width is %dpx but stereo_scale_mm_px was "
                "measured at stereo_width=%dpx. Scaling %.1f -> %.1f mm*px to "
                "keep distances meaningful. This is a safety net, not a "
                "substitute for calibrating at the real width: run "
                "test_stereo_calib.py and set stereo_width/scale to match.",
                eye_w, ref_w, baseline_scale, scaled)
        return scaled

    def _start_stereo(self) -> bool:
        """Bring up stereo capture, preferring to share camera.py's raw frame."""
        scale = self._stereo_effective_scale()
        logging.info("Depth camera: stereo starting (scale=%.1f mm*px, "
                     "source=%s, target eye %dx%d, %d disparities, %s mode, "
                     "swap_eyes=%s, CLAHE clip=%.1f)",
                     scale,
                     "frame_source" if self._frame_source is not None else "(none)",
                     self.config.stereo_width, self.config.stereo_height,
                     self.config.stereo_num_disparities,
                     self.config.stereo_sgbm_mode if self.config.stereo_use_sgbm else "BM",
                     self.config.stereo_swap_eyes,
                     self.config.stereo_clahe_clip_limit)
        if scale <= 0.0:
            logging.warning(
                "Depth camera: stereo backend is NOT calibrated. Set "
                "DepthCameraConfig.stereo_scale_mm_px to (distance_mm * disparity_px) "
                "measured from a target at a known distance, or set stereo_focal_px "
                "and stereo_baseline_mm from the datasheet. Measure it with:\n"
                "    python3 test_stereo_calib.py --distance 1000\n"
                "Refusing to run uncalibrated: a wrong scale would fake an emergency "
                "stop or miss a real obstacle. Falling back to mock for now.")
            return False

        num_disp = int(self.config.stereo_num_disparities)
        if num_disp < 16 or num_disp % 16 != 0:
            num_disp = 128
        closest_mm = scale / max(1, num_disp - 1)
        if closest_mm > self.config.emergency_stop_mm:
            logging.warning(
                "Depth camera: stereo search range only resolves down to %.0f mm, "
                "which is farther than the emergency stop (%.0f mm). Obstacles "
                "inside that blind zone will not register. Raise "
                "stereo_num_disparities (currently %d) or check the scale.",
                closest_mm, self.config.emergency_stop_mm, num_disp)

        # 1) Share camera.py's raw stereo frame.
        if self._try_start_stereo_from_source(scale):
            return True

        # 2) Fallback: open device ourselves.
        candidates = self._candidate_uvc_devices()
        for device in candidates:
            if self._open_stereo_device(device):
                self._stereo_external = False
                self._uvc_device_in_use = device
                self._launch_stereo_sampler()
                logging.info("UVC stereo: opened %s directly.", device)
                return True

        logging.warning(
            "Depth camera: no USB video node produced a usable stereo pair, and no "
            "shared frame source was available. Tried: %s. Confirm the device with "
            "`v4l2-ctl --list-devices` and its formats with "
            "`v4l2-ctl -d <node> --list-formats-ext`.",
            ", ".join(candidates) or "(none)")
        return False

    def _try_start_stereo_from_source(self, scale: float) -> bool:
        """Try to use camera.py's raw frame as the stereo feed."""
        source = self._frame_source
        if source is None or not hasattr(source, "latest_stereo_frame"):
            logging.info(
                "Depth camera: no shared frame source (frame_source missing or does not "
                "implement latest_stereo_frame), will try to open UVC device directly.")
            return False

        deadline = time.monotonic() + 5.0
        frame = None
        while time.monotonic() < deadline:
            try:
                frame = source.latest_stereo_frame()
            except Exception as e:
                logging.debug("stereo: external frame source read failed: %s", e)
                return False
            if frame is not None:
                break
            time.sleep(0.05)

        if frame is None:
            logging.warning(
                "Depth camera: did not receive frame from camera.py within 5 seconds "
                "(latest_stereo_frame returned None). V4L2 capture may not have produced "
                "the first frame yet, or it's a static frame. Switching to direct device open.")
            return False

        h, w = frame.shape[:2]
        if w < 320:
            logging.warning(
                "Depth camera: frame from camera.py is %dx%d, too narrow to be a "
                "stereo pair (need at least 320 wide). Check camera.stereo_capture_width "
                "and that split_stereo_frame='always' is in effect.", w, h)
            return False

        eye_w = w // 2
        real_scale = self._scale_for_eye_width(scale, eye_w)
        depth = self._stereo_frame_to_depth(frame, real_scale)
        if depth is None:
            logging.warning("Depth camera: frame from external source could not compute disparity, switching to direct device open.")
            return False

        self._stereo_external = True
        self._uvc_device_in_use = f"camera.py shared frame ({w}x{h})"
        self._launch_stereo_sampler()
        logging.info(
            "UVC stereo: reusing camera.py's stereo frame (%dx%d, eye %d wide), "
            "not opening the same device again. scale now %.1f mm*px.",
            w, h, eye_w, real_scale)
        return True

    def _launch_stereo_sampler(self):
        """Create the stop event and start the background depth thread."""
        self._stereo_stop_event = threading.Event()
        with self._stereo_lock:
            self._stereo_latest = None
        self._stereo_first_logged = False
        self._stereo_empty_streak = 0
        self._stereo_empty_warned = False
        self._stereo_thread = threading.Thread(
            target=self._stereo_sampler_loop,
            name="stereo-sampler", daemon=True)
        self._stereo_thread.start()

    def _open_stereo_device(self, device: str) -> bool:
        """Try one node as a stereo source. Leaves the handle open on success."""
        cap = None
        try:
            device_arg = device
            try:
                device_arg = int(device)
            except (TypeError, ValueError):
                pass

            cap = cv2.VideoCapture(device_arg, cv2.CAP_V4L2)
            if not cap.isOpened():
                logging.debug("stereo: %s could not be opened.", device)
                return False

            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.stereo_width * 2)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.stereo_height)
            cap.set(cv2.CAP_PROP_FPS, self.config.fps)

            asked_w = self.config.stereo_width * 2
            asked_h = self.config.stereo_height
            actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
            actual_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
            actual_fourcc = _fourcc_str(cap.get(cv2.CAP_PROP_FOURCC))
            logging.info("UVC stereo %s negotiated %dx%d @ %.1ffps, fourcc '%s' "
                         "(requested %dx%d, MJPG)",
                         device, actual_w or 0, actual_h or 0, actual_fps,
                         actual_fourcc or "(unreported)",
                         asked_w, asked_h)

            ok, frame = cap.read()
            if not ok or frame is None:
                logging.debug("stereo: %s opened but produced no frame.", device)
                return False

            if frame.ndim != 3 or frame.shape[1] < 2 or frame.shape[1] % 2 != 0:
                logging.debug("stereo: %s frame shape %s is not a stereo pair.",
                              device, getattr(frame, "shape", None))
                return False

            real_scale = self._scale_for_eye_width(
                self._stereo_effective_scale(), frame.shape[1] // 2)
            depth = self._stereo_frame_to_depth(frame, real_scale)
            if depth is None:
                logging.debug("stereo: %s failed to produce a depth frame.", device)
                return False

            self._uvc_fourcc = actual_fourcc
            self._uvc = cap
            cap = None
            return True
        except Exception as e:
            logging.debug("stereo: opening %s raised %s", device, e)
            return False
        finally:
            if cap is not None:
                try:
                    cap.release()
                except Exception:
                    pass

    def _stereo_frame_to_depth(self, frame: Optional[np.ndarray],
                               scale: float) -> Optional[np.ndarray]:
        """
        Turn one double-wide frame into a float32 millimetre depth map, or None.

        The left half is the reference and the right half the match image. Three
        changes since the previous version, each targeting one of the symptoms
        behind "the reading is wrong / missing half the time":

        1. CLAHE on each eye before matching. SGBM needs edges; a plain indoor
           wall or floor gives it almost none, and the valid-pixel ratio falls to
           single digits. Local contrast equalisation pulls weak texture out of
           the noise floor and routinely lifts that ratio by 5-10x. It changes
           contrast only, not geometry, so the calibration constant is unaffected.

        2. stereo_swap_eyes: some dual-lens UVC modules put the right eye on the
           left of the frame, and SGBM assumes left-on-left. With the eyes
           reversed the disparity map is close to empty -- an easy thing to blame
           on calibration. One flag fixes it once test_stereo_debug.py has shown
           that is what is happening.

        3. Reads pre_filter_cap and the SGBM mode from config rather than the
           hardcoded 31 / SGBM defaults, so the tuning that finally works on the
           car can be saved rather than re-derived.

        `scale` must already be normalised to this frame's eye width; see
        _scale_for_eye_width.
        """
        if frame is None or not isinstance(frame, np.ndarray) or frame.ndim != 3:
            return None
        h, w = frame.shape[:2]
        if w % 2 != 0:
            w -= 1
            frame = frame[:, :w]
        if w < 2 or h < 8:
            return None

        half = w // 2
        # 左右眼是否翻转由 config 决定；默认保持"左前右后"。
        if bool(getattr(self.config, "stereo_swap_eyes", False)):
            left = frame[:, half:]
            right = frame[:, :half]
        else:
            left = frame[:, :half]
            right = frame[:, half:]

        try:
            if left.shape[2] == 3:
                left_g = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
                right_g = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)
            elif left.shape[2] == 1:
                left_g = left[:, :, 0]
                right_g = right[:, :, 0]
            else:
                return None
        except Exception as e:
            logging.debug("stereo: grayscale conversion failed: %s", e)
            return None

        # CLAHE：局部对比度增强，低光/弱纹理场景的匹配率关键提升项。
        clip = float(getattr(self.config, "stereo_clahe_clip_limit", 0.0))
        if clip > 0.0:
            try:
                clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8))
                left_g = clahe.apply(left_g)
                right_g = clahe.apply(right_g)
            except Exception as e:
                logging.debug("stereo: CLAHE failed (non-critical): %s", e)

        num_disp = int(self.config.stereo_num_disparities)
        if num_disp < 16 or num_disp % 16 != 0:
            num_disp = 128
        block = int(self.config.stereo_block_size)
        if block < 5 or block % 2 == 0:
            block = 15

        pre_filter = int(getattr(self.config, "stereo_pre_filter_cap", 31))
        if pre_filter < 1:
            pre_filter = 31

        try:
            if self.config.stereo_use_sgbm:
                p1 = 8 * block * block
                p2 = 32 * block * block
                matcher = cv2.StereoSGBM_create(
                    minDisparity=int(self.config.stereo_min_disparity),
                    numDisparities=num_disp,
                    blockSize=block,
                    P1=p1,
                    P2=p2,
                    disp12MaxDiff=max(0, int(self.config.stereo_disp12_max_diff)),
                    uniquenessRatio=max(0, int(self.config.stereo_uniqueness_ratio)),
                    speckleWindowSize=max(0, int(self.config.stereo_speckle_window_size)),
                    speckleRange=max(0, int(self.config.stereo_speckle_range)),
                    preFilterCap=pre_filter,
                    mode=_sgbm_mode_flag(getattr(self.config, "stereo_sgbm_mode", "3way")),
                )
            else:
                matcher = cv2.StereoBM_create(numDisparities=num_disp, blockSize=block)
            raw16 = matcher.compute(left_g, right_g)
        except Exception as e:
            logging.debug("stereo: disparity matcher failed: %s", e)
            return None
        if raw16 is None:
            return None

        if raw16.dtype != np.int16:
            raw16 = raw16.astype(np.int16)
        raw16[raw16 < 0] = 0
        try:
            cv2.filterSpeckles(raw16, 0,
                               max(0, int(self.config.stereo_speckle_window_size)),
                               max(0, int(self.config.stereo_speckle_range)))
        except Exception as e:
            logging.debug("stereo: filterSpeckles failed (non-critical): %s", e)

        disparity = raw16.astype(np.float32) / 16.0

        median_k = int(self.config.stereo_median_blur)
        if median_k >= 3 and median_k % 2 == 1:
            try:
                disparity = cv2.medianBlur(disparity, median_k)
            except Exception as e:
                logging.debug("stereo: medianBlur failed (non-critical): %s", e)

        disparity = np.where(disparity > 0.0, disparity, 0.0)
        depth = np.zeros(disparity.shape, dtype=np.float32)
        valid = disparity > 0.0
        if np.any(valid):
            depth[valid] = np.clip(scale / disparity[valid], 0.0, 65535.0)
        return depth

    def _stereo_sampler_loop(self):
        """Background sampler: keep the cached depth frame fresh."""
        scale = self._stereo_effective_scale()
        fps = int(getattr(self.config, "stereo_target_fps", 10) or 10)
        period = 1.0 / max(1.0, float(fps))
        event = self._stereo_stop_event
        if event is None:
            return

        last_frame = None

        while not event.is_set():
            started = time.monotonic()
            frame = None
            try:
                if self._stereo_external:
                    frame = self._frame_source.latest_stereo_frame()
                else:
                    with self._uvc_lock:
                        cap = self._uvc
                        if cap is None:
                            break
                        ok, raw = cap.read()
                    if ok and raw is not None:
                        frame = raw
            except Exception as e:
                logging.debug("stereo: sampler read failed: %s", e)

            if frame is not None and frame is not last_frame:
                last_frame = frame
                eye_w = frame.shape[1] // 2 if frame.ndim >= 2 else 0
                real_scale = self._scale_for_eye_width(scale, eye_w)
                depth = self._stereo_frame_to_depth(frame, real_scale)
                with self._stereo_lock:
                    self._stereo_latest = depth

                valid_count = 0 if depth is None else int(np.count_nonzero(depth))

                # 连续多帧无有效读数时打印一次完整诊断，帮助定位问题。
                if valid_count > 0:
                    self._stereo_empty_streak = 0
                else:
                    self._stereo_empty_streak += 1
                    if (self._stereo_empty_streak >= 5
                            and not self._stereo_empty_warned):
                        self._stereo_empty_warned = True
                        logging.warning(
                            "stereo: 5 consecutive frames with NO valid depth pixels. "
                            "The disparity matcher is not producing any matches at all. "
                            "Common causes, in the order worth checking:\n"
                            "  1. Lens caps / focus / a covered sensor.\n"
                            "  2. stereo_swap_eyes - some modules put the right eye on "
                            "the left of the frame. Flip it and try again.\n"
                            "  3. Lighting - SGBM needs edges; a plain wall in a dark "
                            "room has none. Raise stereo_clahe_clip_limit to 3.0-4.0.\n"
                            "  4. stereo_scale_mm_px - if the calibration constant is "
                            "wrong, every match is out of range and looks like 'no "
                            "reading'. Run test_stereo_calib.py.\n"
                            "Run test_stereo_debug.py to dump the raw left/right and "
                            "disparity frames. It shows exactly which step fails.")

                if valid_count > 0 and not self._stereo_first_logged:
                    self._stereo_first_logged = True
                    valid_ratio = float(valid_count) / float(depth.size)
                    valid_px = depth[depth > 0.0]
                    median_mm = float(np.median(valid_px)) if valid_px.size else 0.0
                    logging.info(
                        "stereo: first depth frame ready (eye %dpx, scale %.1f): "
                        "valid pixels %.1f%%, median %.0f mm. If the ratio is low, "
                        "SGBM cannot find matches -- the scene is too smooth / too "
                        "far / too dark. Try raising stereo_block_size, or "
                        "stereo_num_disparities, or fixing the exposure in "
                        "CameraConfig.v4l2_*.",
                        eye_w, real_scale, valid_ratio * 100.0, median_mm)

            elapsed = time.monotonic() - started
            remaining = period - elapsed
            if remaining > 0:
                event.wait(remaining)

    def _capture_stereo(self):
        """Return the cached depth frame from the sampler thread."""
        with self._stereo_lock:
            depth = self._stereo_latest
        return None, depth

    # ------------------------------------------------------------------
    # DECXIN / generic UVC single-stream backend
    # ------------------------------------------------------------------

    def _candidate_uvc_devices(self) -> List[str]:
        """Nodes to try, most likely first."""
        configured = str(self.config.uvc_device or "").strip()
        candidates: List[str] = []

        if configured:
            if configured.startswith("/dev/") and not _is_usb_video_node(configured):
                logging.warning(
                    "UVC depth: %s is not a USB video node, so it is not the depth "
                    "camera. Not opening it -- that would take the camera away from "
                    "picamera2. Scanning USB video nodes instead.", configured)
            else:
                candidates.append(configured)

        for path in sorted(glob.glob("/dev/video*"), key=_device_sort_key):
            if path in candidates:
                continue
            if _is_usb_video_node(path):
                candidates.append(path)

        if not candidates and configured:
            candidates.append(configured)
        return candidates

    def _start_decxin(self) -> bool:
        """Open the UVC depth stream, scanning the USB video nodes until one accepts."""
        candidates = self._candidate_uvc_devices()
        if not candidates:
            logging.warning(
                "UVC depth: no USB video nodes found at all. Check that the camera "
                "is plugged in and enumerated: `ls -l /dev/video*` and "
                "`v4l2-ctl --list-devices`.")
            return False

        notes: List[str] = []
        for device in candidates:
            ok, note = self._open_decxin_device(device)
            notes.append(f"{device}: {note}")
            if ok:
                logging.info("UVC depth: using %s ('%s'). %s", device,
                             self._uvc_fourcc or "format unreported", note)
                return True

        logging.warning(
            "UVC depth: no usable 16-bit depth stream found. Tried:")
        for line in notes:
            logging.warning("  %s", line)
        logging.warning(
            "If every node reports a colour format (YUYV/UYVY/MJPG), this camera "
            "does not expose depth over UVC. Use backend='stereo' for a dual-lens "
            "module, and check `v4l2-ctl -d <node> --list-formats-ext`.")
        return False

    def _open_decxin_device(self, device: str) -> Tuple[bool, str]:
        """Try one node. Returns (success, human-readable outcome)."""
        cap = None
        try:
            device_arg = device
            try:
                device_arg = int(device)
            except (TypeError, ValueError):
                pass

            cap = cv2.VideoCapture(device_arg, cv2.CAP_V4L2)
            if not cap.isOpened():
                return False, "could not be opened"

            convert_prop = getattr(cv2, "CAP_PROP_CONVERT_RGB", None)
            if convert_prop is not None:
                cap.set(convert_prop, 0)

            if self.config.uvc_fourcc and len(self.config.uvc_fourcc) == 4:
                cap.set(cv2.CAP_PROP_FOURCC,
                        cv2.VideoWriter_fourcc(*self.config.uvc_fourcc))

            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.height)
            cap.set(cv2.CAP_PROP_FPS, self.config.fps)

            ok, raw = cap.read()
            if not ok or raw is None:
                return False, "opened but produced no frame (metadata node?)"

            negotiated = _fourcc_str(cap.get(cv2.CAP_PROP_FOURCC))
            code = negotiated.upper()
            if code in _DEPTH_FOURCCS:
                pass
            elif code in _COLOUR_FOURCCS:
                return False, (f"colour stream '{negotiated}', not depth -- the "
                               f"device ignored the fourcc request")
            elif not code:
                logging.warning(
                    "UVC depth: %s did not report its pixel format. Assuming 16-bit "
                    "depth from the frame shape (%s). If distances look wrong, run "
                    "`v4l2-ctl -d %s --list-formats-ext`.", device, raw.shape, device)
            else:
                return False, f"negotiated '{negotiated}', which is not a depth format"

            depth = self._decode_raw_depth(raw, negotiated)
            if depth is None:
                return False, (f"frame is not a 16-bit depth buffer "
                               f"(dtype={raw.dtype} shape={raw.shape} "
                               f"negotiated='{negotiated or 'unreported'}')")

            self._uvc = cap
            cap = None
            self._uvc_device_in_use = device
            self._uvc_fourcc = negotiated
            return True, f"depth stream, {raw.shape[1]}x{raw.shape[0]}"
        except Exception as e:
            return False, f"raised {e}"
        finally:
            if cap is not None:
                try:
                    cap.release()
                except Exception:
                    pass

    def _capture_decxin(self):
        """One frame from the UVC device. Colour is never opened on this backend."""
        try:
            ok, raw = self._uvc.read()
        except Exception as e:
            logging.debug(f"UVC depth capture failed: {e}")
            return None, None

        if not ok or raw is None:
            return None, None

        depth = self._decode_raw_depth(raw, self._uvc_fourcc)
        if depth is None:
            return None, None
        return None, depth

    def _decode_raw_depth(self, raw: Optional[np.ndarray],
                          negotiated: str = "") -> Optional[np.ndarray]:
        """Normalise one raw UVC buffer into float32 millimetres, or None."""
        if raw is None or raw.size == 0:
            return None

        code = (negotiated or "").strip().upper()

        if code in _COLOUR_FOURCCS:
            logging.debug("UVC depth: dropping frame, device negotiated colour "
                          "format '%s'.", negotiated)
            return None

        if raw.dtype == np.uint16:
            counts = raw
        elif raw.dtype == np.uint8 and raw.ndim == 3 and raw.shape[2] == 2:
            counts = np.ascontiguousarray(raw).view(np.uint16).reshape(raw.shape[0], raw.shape[1])
        else:
            return None

        return counts.astype(np.float32) * float(self.config.uvc_depth_unit_mm)

    # ------------------------------------------------------------------
    # RealSense backend
    # ------------------------------------------------------------------

    def _start_realsense(self) -> bool:
        try:
            pipeline = rs.pipeline()
            cfg = rs.config()
            if self.config.device_serial:
                cfg.enable_device(self.config.device_serial)
            cfg.enable_stream(rs.stream.depth, self.config.width,
                              self.config.height, rs.format.z16, self.config.fps)
            if self.config.enable_color_stream:
                cfg.enable_stream(rs.stream.color, self.config.width,
                                  self.config.height, rs.format.bgr8, self.config.fps)

            profile = pipeline.start(cfg)
            self._pipeline = pipeline

            try:
                sensor = profile.get_device().first_depth_sensor()
                self._depth_scale_m = float(sensor.get_depth_scale())
            except Exception as e:
                logging.debug(f"Could not read RealSense depth scale: {e}")
            return True
        except Exception as e:
            logging.warning(f"RealSense start failed: {e}")
            return False

    def _capture_realsense(self):
        try:
            frames = self._pipeline.wait_for_frames(self.config.capture_timeout_ms)
        except RuntimeError:
            return None, None
        except Exception as e:
            logging.debug(f"RealSense capture failed: {e}")
            return None, None

        try:
            color = None
            if self.config.enable_color_stream:
                color_frame = frames.get_color_frame()
                if color_frame:
                    color = np.asanyarray(color_frame.get_data())

            depth = None
            depth_frame = frames.get_depth_frame()
            if depth_frame:
                raw = np.asanyarray(depth_frame.get_data()).astype(np.float32)
                depth = raw * (self._depth_scale_m * 1000.0)

            return color, depth
        except Exception as e:
            logging.debug(f"RealSense frame decode failed: {e}")
            return None, None

    # ------------------------------------------------------------------
    # Mock backend
    # ------------------------------------------------------------------

    def _capture_mock(self):
        """Synthesise a flat wall at 3 m with a little sensor noise."""
        h, w = self.config.height, self.config.width
        depth = np.full((h, w), 3000.0, dtype=np.float32)
        depth += np.random.randint(-50, 51, size=(h, w)).astype(np.float32)
        np.clip(depth, 0.0, 65535.0, out=depth)

        color = None
        if self.config.enable_color_stream:
            color = np.zeros((h, w, 3), dtype=np.uint8)
        return color, depth
