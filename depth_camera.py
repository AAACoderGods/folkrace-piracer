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
   and does not know what mode the car is in -- main.py owns that decision. This
   keeps the same separation webcontrol.py already uses: no HTTP here, no
   hardware policy either.

4. The stream format is read back from the device, never inferred from shape.
   A 4:2:2 colour format (YUYV and friends) is two bytes per pixel, so with
   CONVERT_RGB=0 it arrives as a uint8 (h, w, 2) array -- the exact shape of a
   Z16 depth buffer. Treating one as the other yields an array of plausible
   noise that fails every range check and looks like "the camera has no
   reading". CAP_PROP_FOURCC is the only thing that distinguishes them, so it
   is checked before any UVC buffer is interpreted as depth.

5. The device node is discovered, not assumed. `v4l2-ctl --list-devices` numbers
   nodes in USB enumeration order, which is not stable. Candidates are filtered
   by their sysfs bus (USB only) and then verified by actually reading a frame.

6. The UVC-based backends capture on a background thread. A live stereo frame
   costs tens of milliseconds to decode and match, and blocking the 30 Hz
   control loop on that would slow steering and throttling just as much. The
   sampler thread caches the latest depth frame; the main loop only ever reads
   the cache.

To add a new backend (OAK-D, Orbbec, a vendor SDK, ...):
    - add a probe block for the library at the top
    - add a `_start_<name>` that returns True on success
    - add a `_capture_<name>` returning (colour BGR or None, depth mm or None)
    - wire both into start() and capture_frame() next to the existing branches
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

# Each backend is probed once at import. A missing library is the normal case on
# a dev machine, so it is a warning at start() time, not an error here.
try:
    import pyrealsense2 as rs
    _REALSENSE_AVAILABLE = True
except ImportError:
    rs = None
    _REALSENSE_AVAILABLE = False

# The UVC backends need only OpenCV, which the rest of the project already uses
# for vision -- so on a normal install this is always present.
try:
    import cv2
    _CV2_AVAILABLE = True
except ImportError:
    cv2 = None
    _CV2_AVAILABLE = False


# Pixel formats that actually carry 16-bit depth over UVC. Compared with the
# trailing space stripped, because the fourth character of a FOURCC is padding.
_DEPTH_FOURCCS = {"Z16", "Y16", "Y12", "Z12", "Y14", "Y10"}

# 4:2:2 colour formats. These are two bytes per pixel as well, so CONVERT_RGB=0
# leaves them in exactly the same uint8 (h, w, 2) shape that a Z16 buffer has.
# The distinction cannot be made from the array alone -- see rule 4 above.
_COLOUR_FOURCCS = {"YUYV", "UYVY", "YVYU", "VYUY", "MJPG", "JPEG", "RGB3", "BGR3"}

# Where the kernel ties a /dev/videoN node to the bus it hangs off. Resolving this
# link is the only reliable way to tell a USB camera from the CSI pipeline: the
# names and numbers both move between boots, the bus does not.
_SYSFS_V4L = "/sys/class/video4linux"


def _fourcc_str(value) -> str:
    """
    Turn a CAP_PROP_FOURCC integer back into its four-character code.

    Empty string means the device (or the OpenCV build) did not report a format.
    That is treated as "unknown", not as a failure: on several Pi builds the V4L2
    backend returns 0 for anything outside its small table of known codes, and
    refusing to start in that case would reject working depth cameras.
    """
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
    """
    True when the node hangs off a USB bus, rather than the CSI pipeline.

    picamera2 drives the CSI camera through libcamera, but its V4L2 nodes (and
    the codec/ISP nodes that accompany them) live on a platform bus. Opening one
    of those here would take the main camera away from the vision pipeline, which
    is a far worse failure than having no depth guard -- so the filter is a
    safety rule, not an optimisation.

    A missing sysfs (odd container, or a kernel without CONFIG_VIDEO_DEV) means
    the question cannot be answered, and the node is accepted: refusing to try
    anything would trade a rare conflict for a guaranteed "no depth camera".
    """
    if not device.startswith("/dev/"):
        # Not a path at all -- an OpenCV index. Nothing to inspect.
        return True
    if not os.path.isdir(_SYSFS_V4L):
        return True
    try:
        real = os.path.realpath(os.path.join(_SYSFS_V4L, os.path.basename(device), "device"))
    except OSError:
        return True
    return "/usb" in real


class DepthCamera:
    """
    Facade over the available depth backends, with a uniform frame interface.

    Usage mirrors Camera:
        cam = DepthCamera(config.depth)
        cam.start()
        colour, depth = cam.capture_frame()   # depth is mm, 0 means "no reading"
        cam.stop()
    """

    def __init__(self, config: DepthCameraConfig):
        self.config = config
        self.backend: str = "mock"
        self.is_running = False
        # Only one of these is ever live; stop() releases whichever is set.
        self._pipeline = None
        self._uvc = None
        # Guard for self._uvc: the stereo sampler reads it on a background
        # thread while stop() may be releasing it from the main thread.
        self._uvc_lock = threading.Lock()
        # What the UVC device actually negotiated, as opposed to what config
        # asked for. Kept so every frame can be re-checked against it: a USB
        # link that renegotiates mid-run would otherwise silently swap the
        # colour and depth interpretations.
        self._uvc_fourcc = ""
        self._uvc_device_in_use = ""
        # Stereo sampler state. The thread reads frames and writes the latest
        # depth map; the main loop reads the cache without ever blocking.
        self._stereo_lock = threading.Lock()
        self._stereo_stop_event: Optional[threading.Event] = None
        self._stereo_thread: Optional[threading.Thread] = None
        self._stereo_latest: Optional[np.ndarray] = None
        # Per-device scale factor from raw counts to metres. RealSense sets it
        # per model; 0.001 (i.e. 1 count = 1 mm) is the common default and the
        # value the mock backend reports, so it is a safe fallback if start()
        # never got far enough to read the real one.
        self._depth_scale_m = 0.001

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self):
        """
        Bring up the configured backend, falling back to mock on any failure.

        Never raises. main.py constructs this object unconditionally whenever
        --allow-depth-camera is set, and the run must not depend on whether a
        depth camera actually answered.
        """
        requested = (self.config.backend or "mock").lower()

        if requested == "stereo":
            if not _CV2_AVAILABLE:
                logging.warning("Depth camera: 'stereo' requested but OpenCV is not "
                                "installed. pip install opencv-python")
            elif not hasattr(cv2, "StereoBM_create"):
                logging.warning("Depth camera: 'stereo' requested but this OpenCV "
                                "build has no cv2.StereoBM_create.")
            elif self._start_stereo():
                self.backend = "stereo"
                self.is_running = True
                logging.info("✅ Depth camera started: stereo (UVC on %s, %dx%d, "
                             "scale=%.1f mm*px, %d disparities)",
                             self._uvc_device_in_use,
                             self.config.stereo_width, self.config.stereo_height,
                             self._stereo_effective_scale(),
                             self.config.stereo_num_disparities)
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
        set; line following uses the main camera, not this one. The depth frame
        is float32 in millimetres with 0 marking "no reading". Either element may
        be None on a dropped frame -- callers must handle that.
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
        # Stop the sampler before touching the device handle: it reads the cap on
        # a background thread, and releasing while it is mid-read is a crash.
        if self._stereo_stop_event is not None:
            self._stereo_stop_event.set()
        if self._stereo_thread is not None:
            # The sampler's read() can block for up to a frame period; give it
            # room to notice the stop event and leave the loop.
            self._stereo_thread.join(timeout=5.0)
            self._stereo_thread = None
        self._stereo_stop_event = None
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

        Returns None when the band holds no usable reading at all -- which is the
        correct answer for a camera that is pointed at the sky or has just lost
        the device. Returning a large number here instead would quietly turn
        "no data" into "no obstacle", which is the same as switching the guard off.
        """
        if depth_frame is None or depth_frame.size == 0:
            return None

        h, w = depth_frame.shape[:2]
        row = int(np.clip(self.config.obstacle_scan_y_ratio, 0.0, 1.0) * (h - 1))
        half = int(np.clip(self.config.obstacle_scan_width_ratio, 0.0, 1.0) * w / 2.0)
        x0 = max(0, w // 2 - half)
        x1 = min(w, w // 2 + half)
        band = depth_frame[row, x0:x1]

        # 0 is the "no return" marker every depth device here uses; drop it before
        # taking the minimum, otherwise every frame with a single dropout reads as
        # an obstacle at 0 mm.
        valid = band[band > 0]
        if valid.size == 0:
            return None

        near = float(np.min(valid))
        if not (self.config.min_depth_mm <= near <= self.config.max_depth_mm):
            return None
        return near

    def depth_grid(self, depth_frame: Optional[np.ndarray]) -> Optional[dict]:
        """
        Coarse depth map for the web console's cursor readout.

        Shaped as {"w", "h", "mm"} with the cells row-major, because that is what
        the browser needs to turn a pointer position into a lookup -- no image
        decoding, no canvas. Cells outside the valid range are written as 0, the
        same "no reading" marker the rest of this file uses.

        This is nearest-neighbour sampling, not an average: the grid is a readout
        for a human hovering over the picture, and a mean over a cell would blur
        a thin obstacle (a post, a leg) out of existence exactly where the
        operator is pointing at it.
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
        """
        The mm*px constant used to turn disparity into depth.

        Prefers the directly measured stereo_scale_mm_px. Falls back to the
        physical focal*基线 product for users who have a datasheet instead of a
        measuring tape. Returns 0.0 when neither is set, which start() treats as
        "refuse to run uncalibrated".
        """
        scale = float(self.config.stereo_scale_mm_px)
        if scale > 0.0:
            return scale
        focal = float(self.config.stereo_focal_px)
        baseline = float(self.config.stereo_baseline_mm)
        if focal > 0.0 and baseline > 0.0:
            return focal * baseline
        return 0.0

    def _start_stereo(self) -> bool:
        """
        Bring up MVPG stereo capture on the first USB node that yields a frame.

        Calibration is mandatory: without a scale, identical left/right images
        produce a disparity of zero everywhere, which the range check reads as
        "no obstacle". That is indistinguishable from a clear road, and it is the
        one failure mode a forward guard must never have.
        """
        scale = self._stereo_effective_scale()
        if scale <= 0.0:
            logging.warning(
                "Depth camera: stereo backend is NOT calibrated. Set "
                "DepthCameraConfig.stereo_scale_mm_px to (distance_mm * disparity_px) "
                "measured from a target at a known distance, or set stereo_focal_px "
                "and stereo_baseline_mm from the datasheet. Refusing to run "
                "uncalibrated: a wrong scale would fake an emergency stop or miss a "
                "real obstacle. Falling back to mock for now.")
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

        candidates = self._candidate_uvc_devices()
        for device in candidates:
            if self._open_stereo_device(device):
                self._stereo_stop_event = threading.Event()
                with self._stereo_lock:
                    self._stereo_latest = None
                self._stereo_thread = threading.Thread(
                    target=self._stereo_sampler_loop,
                    name="stereo-sampler", daemon=True)
                self._stereo_thread.start()
                logging.info("UVC stereo: using %s.", device)
                return True

        logging.warning(
            "Depth camera: no USB video node produced a usable stereo pair. "
            "Tried: %s. Confirm the device with `v4l2-ctl --list-devices` and its "
            "formats with `v4l2-ctl -d <node> --list-formats-ext`.",
            ", ".join(candidates) or "(none)")
        return False

    def _open_stereo_device(self, device: str) -> bool:
        """
        Try one node as a stereo source. Leaves the handle open on success.
        """
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

            # MJPG is what gives 30 fps at 1280x480 with tolerable USB bandwidth.
            # YUYV is also listed, but only at 10 fps -- too slow for the guard.
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.stereo_width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.stereo_height)
            cap.set(cv2.CAP_PROP_FPS, self.config.fps)

            ok, frame = cap.read()
            if not ok or frame is None:
                logging.debug("stereo: %s opened but produced no frame.", device)
                return False

            if frame.ndim != 3 or frame.shape[1] < 2 or frame.shape[1] % 2 != 0:
                logging.debug("stereo: %s frame shape %s is not a stereo pair.",
                              device, getattr(frame, "shape", None))
                return False

            # One sample is weak evidence, but it is enough to reject a node whose
            # disparity map is completely flat (metadata node, or a lens cap).
            scale = self._stereo_effective_scale()
            depth = self._stereo_frame_to_depth(frame, scale)
            if depth is None:
                logging.debug("stereo: %s failed to produce a depth frame.", device)
                return False

            # Hand the handle over so finally does not release it.
            self._uvc_fourcc = _fourcc_str(cap.get(cv2.CAP_PROP_FOURCC))
            self._uvc = cap
            cap = None
            self._uvc_device_in_use = device
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

        The left half is the reference and the right half the match image; image
        rows from a real stereo pair are already rectified, so no rectification
        is attempted here. Anything that is not a real match is left at 0.0 --
        the "no reading" marker -- rather than turned into a plausible distance.
        """
        if frame is None or not isinstance(frame, np.ndarray) or frame.ndim != 3:
            return None
        h, w = frame.shape[:2]
        if w < 2 or w % 2 != 0 or h < 8:
            return None

        half = w // 2
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

        num_disp = int(self.config.stereo_num_disparities)
        if num_disp < 16 or num_disp % 16 != 0:
            num_disp = 128
        block = int(self.config.stereo_block_size)
        if block < 5 or block % 2 == 0:
            block = 15

        try:
            bm = cv2.StereoBM_create(numDisparities=num_disp, blockSize=block)
            # BM output is fixed-point with four fractional bits; divide by 16
            # for pixels. Invalid locations come out as a multiple of -16.
            raw = bm.compute(left_g, right_g).astype(np.float32) / 16.0
        except Exception as e:
            logging.debug("stereo: StereoBM failed: %s", e)
            return None

        disparity = np.where(raw > 0.0, raw, 0.0)
        depth = np.zeros(disparity.shape, dtype=np.float32)
        valid = disparity > 0.0
        if np.any(valid):
            depth[valid] = np.clip(scale / disparity[valid], 0.0, 65535.0)
        return depth

    def _stereo_sampler_loop(self):
        """Background sampler: keep the cached depth frame fresh."""
        scale = self._stereo_effective_scale()
        period = 1.0 / max(1.0, float(self.config.fps))
        event = self._stereo_stop_event
        if event is None:
            return

        while not event.is_set():
            started = time.monotonic()
            depth = None
            try:
                with self._uvc_lock:
                    cap = self._uvc
                    if cap is None:
                        break
                    ok, frame = cap.read()
                if ok and frame is not None:
                    depth = self._stereo_frame_to_depth(frame, scale)
            except Exception as e:
                logging.debug("stereo: sampler read failed: %s", e)
            with self._stereo_lock:
                self._stereo_latest = depth

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
        """
        Nodes to try, most likely first.

        The configured value is a hint, not a rule: it is tried first when it is
        a USB node, and every other USB video node follows. USB numbering is not
        stable across boots, so a run that worked yesterday can point the hint at
        the metadata node (which never produces frames) or at a node that no
        longer exists. Scanning costs a handful of opens on a failure path nobody
        is timing, and turns "the camera moved" from a broken run into a log line.

        Non-USB nodes are excluded outright -- see _is_usb_video_node.
        """
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
            # Nothing passed the USB filter (or sysfs is unavailable). Fall back to
            # the configured node so the behaviour degrades to a single attempt
            # rather than silently doing nothing.
            candidates.append(configured)
        return candidates

    def _start_decxin(self) -> bool:
        """
        Open the UVC depth stream, scanning the USB video nodes until one accepts.

        Returns False on any failure so start() can fall back. The first frame is
        decoded as part of the attempt on purpose: if conversion settings were not
        honoured, every later frame would fail the same way, and failing here
        produces one loud warning with the per-node reasons instead of a depth
        guard that is silently blind for the whole run.
        """
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
                # Give the winning line its own INFO record: the per-node notes
                # below are only worth printing when nothing worked.
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
        """
        Try one node.

        Returns (success, human-readable outcome). On success the handle is left
        open and owned by self._uvc; on any failure it is released before return
        so the next candidate starts clean.
        """
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

            # Without this OpenCV converts every frame to 8-bit BGR, which throws
            # the depth data away -- the not-a-depth-buffer check below catches it.
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
                # Typical for the UVC metadata node that pairs with the capture
                # node: it opens fine and never produces a frame.
                return False, "opened but produced no frame (metadata node?)"

            negotiated = _fourcc_str(cap.get(cv2.CAP_PROP_FOURCC))
            code = negotiated.upper()
            if code in _DEPTH_FOURCCS:
                pass
            elif code in _COLOUR_FOURCCS:
                return False, (f"colour stream '{negotiated}', not depth -- the "
                               f"device ignored the fourcc request")
            elif not code:
                # Shape-only acceptance, kept for OpenCV builds that cannot report
                # a non-standard FOURCC. Flag it loudly: if the device is actually
                # streaming YUYV this branch manufactures garbage distances.
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

            # Success: hand the handle over so the finally block leaves it open.
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
            # A missed frame is not a fault; the guard treats None as "no data".
            return None, None

        depth = self._decode_raw_depth(raw, self._uvc_fourcc)
        if depth is None:
            return None, None
        return None, depth

    def _decode_raw_depth(self, raw: Optional[np.ndarray],
                          negotiated: str = "") -> Optional[np.ndarray]:
        """
        Normalise one raw UVC buffer into float32 millimetres, or None.

        `negotiated` is the FOURCC the device reported for this stream. It is the
        only way to tell these apart, because several of them share a shape:

            uint16 (h, w)      - the camera already hands back 16-bit counts
            uint8  (h, w, 2)   - the same data as raw little-endian bytes, which is
                                 what OpenCV gives when it cannot widen the element
            uint8  (h, w, 2)   - OR a 4:2:2 colour frame seen through
                                 CONVERT_RGB=0, which is *not* depth and would
                                 decode into nonsense if the shape were trusted
            uint8  (h, w, 3)   - OpenCV converted to BGR after all; the depth is
                                 gone and no amount of reinterpretation brings it
                                 back, so this is reported as "no reading" rather
                                 than as a plausible-looking 8-bit height map.
        """
        if raw is None or raw.size == 0:
            return None

        code = (negotiated or "").strip().upper()

        # Reject a negotiated 4:2:2 colour format outright. Checked here as well
        # as in _open_decxin_device because a USB link that renegotiates mid-run
        # would otherwise start feeding colour frames into the distance band.
        if code in _COLOUR_FOURCCS:
            logging.debug("UVC depth: dropping frame, device negotiated colour "
                          "format '%s'.", negotiated)
            return None

        if raw.dtype == np.uint16:
            counts = raw
        elif raw.dtype == np.uint8 and raw.ndim == 3 and raw.shape[2] == 2:
            # Two bytes per pixel, little-endian on every platform this runs on.
            # Reached only for a depth FOURCC or when the format was unreported.
            counts = np.ascontiguousarray(raw).view(np.uint16).reshape(raw.shape[0], raw.shape[1])
        else:
            return None

        # Float32 in mm rather than raw counts: the rest of the code compares
        # against thresholds in millimetres, and a units mix-up there would read
        # as a 1000x error in either direction.
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

            # Read the device's own scale factor rather than assuming 1 mm/count.
            # D455 and friends differ from D435 here, and hardcoding it would put
            # every distance off by a constant factor.
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
            # Positional timeout is the form that exists across pyrealsense2
            # releases; the keyword spelling is newer and not always present.
            frames = self._pipeline.wait_for_frames(self.config.capture_timeout_ms)
        except RuntimeError:
            # wait_for_frames raises on timeout. That is a dropped frame, not a
            # fault: return empty and let the control loop carry on.
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
        """
        Synthesise a flat wall at 3 m with a little sensor noise.

        Deliberately inert: 3000 mm is inside the valid range but well beyond
        warning_mm, so the guard never fires from mock data. A mock that triggered
        emergency stops would make the fallback path worse than having no depth
        camera at all.
        """
        h, w = self.config.height, self.config.width
        depth = np.full((h, w), 3000.0, dtype=np.float32)
        depth += np.random.randint(-50, 51, size=(h, w)).astype(np.float32)
        np.clip(depth, 0.0, 65535.0, out=depth)

        color = None
        if self.config.enable_color_stream:
            color = np.zeros((h, w, 3), dtype=np.uint8)
        return color, depth
