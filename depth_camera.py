"""
Depth camera adapter for PiRacer Pro.

Backends:
    decxin    - DECXIN-SM-2930V1 over plain UVC/V4L2, no vendor library needed.
                This is the default because that is the camera on the car.
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

To add a new backend (OAK-D, Orbbec, a vendor SDK, ...):
    - add a probe block for the library at the top
    - add a `_start_<name>` that returns True on success
    - add a `_capture_<name>` returning (colour BGR or None, depth mm or None)
    - wire both into start() and capture_frame() next to the existing branches
"""

import logging
from typing import Optional, Tuple

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

# The UVC backend needs only OpenCV, which the rest of the project already uses
# for vision -- so on a normal install this is always present.
try:
    import cv2
    _CV2_AVAILABLE = True
except ImportError:
    cv2 = None
    _CV2_AVAILABLE = False


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

        if requested == "decxin":
            if not _CV2_AVAILABLE:
                logging.warning("Depth camera: 'decxin' requested but OpenCV is not "
                                "installed. pip install opencv-python")
            elif self._start_decxin():
                self.backend = "decxin"
                self.is_running = True
                logging.info("✅ Depth camera started: DECXIN (UVC on %s, fourcc '%s') "
                             "(%dx%d @ %dfps, %.3f mm/count)",
                             self.config.uvc_device, self.config.uvc_fourcc,
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
        logging.info("Depth camera: mock backend, synthesising depth frames.")

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

        if self.backend == "decxin":
            return self._capture_decxin()
        if self.backend == "realsense":
            return self._capture_realsense()
        return self._capture_mock()

    def stop(self):
        """Stop the stream and release the device. Safe to call repeatedly."""
        if self._pipeline is not None:
            try:
                self._pipeline.stop()
            except Exception as e:
                logging.debug(f"Error stopping RealSense pipeline: {e}")
            self._pipeline = None

        if self._uvc is not None:
            try:
                self._uvc.release()
            except Exception as e:
                logging.debug(f"Error releasing UVC depth capture: {e}")
            self._uvc = None

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
    # DECXIN / UVC backend
    # ------------------------------------------------------------------

    def _start_decxin(self) -> bool:
        """
        Open the UVC depth stream.

        Returns False on any failure so start() can fall back. The first frame is
        decoded as part of start() on purpose: if conversion settings were not
        honoured, every later frame would fail the same way, and failing here
        produces one loud warning instead of a depth guard that is silently blind
        for the whole run.
        """
        device = self.config.uvc_device
        try:
            device = int(device)
        except (TypeError, ValueError):
            pass

        cap = None
        try:
            cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
            if not cap.isOpened():
                logging.warning("UVC depth: could not open %s. Check the node with "
                                "`v4l2-ctl --list-devices`.", self.config.uvc_device)
                return False

            # Without this OpenCV converts every frame to 8-bit BGR, which throws
            # the depth data away -- the 3-channel guard below is what catches it.
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
                logging.warning("UVC depth: %s opened but produced no frame.",
                                self.config.uvc_device)
                return False

            if self._decode_raw_depth(raw) is None:
                logging.warning(
                    "UVC depth: %s returned a %s frame of shape %s, which is not a "
                    "16-bit depth buffer. The device is probably not in a depth "
                    "format -- set uvc_fourcc to one of the formats listed by "
                    "`v4l2-ctl -d %s --list-formats-ext`.",
                    self.config.uvc_device, raw.dtype, raw.shape, self.config.uvc_device)
                return False

            self._uvc = cap
            return True
        except Exception as e:
            logging.warning(f"UVC depth start failed: {e}")
            if cap is not None:
                try:
                    cap.release()
                except Exception:
                    pass
            return False

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

        depth = self._decode_raw_depth(raw)
        if depth is None:
            return None, None
        return None, depth

    def _decode_raw_depth(self, raw: Optional[np.ndarray]) -> Optional[np.ndarray]:
        """
        Normalise one raw UVC buffer into float32 millimetres, or None.

        Three shapes turn up in practice, and only the first two are depth:
            uint16 (h, w)      - the camera already hands back 16-bit counts
            uint8  (h, w, 2)   - the same data as raw little-endian bytes, which is
                                 what OpenCV gives when it cannot widen the element
            uint8  (h, w, 3)   - OpenCV converted to BGR after all; the depth is
                                 gone and no amount of reinterpretation brings it
                                 back, so this is reported as "no reading" rather
                                 than as a plausible-looking 8-bit height map.
        """
        if raw is None or raw.size == 0:
            return None

        if raw.dtype == np.uint16:
            counts = raw
        elif raw.dtype == np.uint8 and raw.ndim == 3 and raw.shape[2] == 2:
            # Two bytes per pixel, little-endian on every platform this runs on.
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
