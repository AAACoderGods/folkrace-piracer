"""
Camera abstraction module supporting Picamera2 for Raspberry Pi 4/5
and fallback to OpenCV VideoCapture or synthetic frames for local testing.
"""

import logging
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

    def __init__(self, width: int = 1920, height: int = 1080, framerate: int = 30,
                 format: str = "RGB888", split_stereo_frame: str = "auto",
                 stereo_eye: str = "left"):
        self.width = width
        self.height = height
        self.framerate = framerate
        self.format = format
        # "auto" | "always" | "never" -- see CameraConfig.split_stereo_frame.
        self.split_stereo_frame = split_stereo_frame
        # "left" | "right" -- which half of a side-by-side pair to keep.
        self.stereo_eye = stereo_eye
        self.picam2 = None
        self.cap = None
        self.is_running = False
        # The split decision is logged once, not every frame: at 30 Hz it would
        # drown the log, and the answer never changes during a run.
        self._split_logged = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self):
        """Initialize and start the camera stream."""
        if not PICAMERA2_AVAILABLE:
            logging.warning("⚠️  'picamera2' module not imported. If using venv, ensure --system-site-packages was used.")

        if PICAMERA2_AVAILABLE:
            logging.info("Initializing Picamera2 (OV5647 compatible)...")
            try:
                self.picam2 = Picamera2()
                try:
                    cam_config = self.picam2.create_preview_configuration(
                        main={"size": (self.width, self.height)}
                    )
                except Exception:
                    cam_config = self.picam2.create_video_configuration(
                        main={"size": (self.width, self.height), "format": "RGB888"}
                    )
                self.picam2.configure(cam_config)
                self.picam2.start()
                # Warm up
                time.sleep(0.5)
                self.is_running = True
                logging.info(f"✅ Picamera2 started successfully ({self.width}x{self.height} @ {self.framerate}fps)")
                return
            except Exception as e:
                logging.error(f"❌ Picamera2 failed to start: {e}", exc_info=True)
                if self.picam2 is not None:
                    try:
                        self.picam2.close()
                    except Exception:
                        pass
                    self.picam2 = None

        if cv2 is not None:
            logging.warning("Trying OpenCV VideoCapture fallback (index 0)...")
            try:
                self.cap = cv2.VideoCapture(0)
                if self.cap.isOpened():
                    self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                    self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                    self.cap.set(cv2.CAP_PROP_FPS, self.framerate)
                    self.is_running = True
                    logging.info("✅ OpenCV VideoCapture fallback started.")
                    return
            except Exception as e:
                logging.warning(f"OpenCV VideoCapture failed: {e}")

        logging.warning("⚠️  No physical camera could be opened. Running in synthetic mock frame mode.")
        self.is_running = True

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

    def stop(self):
        """Stop and release camera resources."""
        if self.picam2 is not None:
            try:
                self.picam2.stop()
                self.picam2.close()
            except Exception as e:
                logging.error(f"Error stopping Picamera2: {e}")
            self.picam2 = None

        if self.cap is not None:
            try:
                self.cap.release()
            except Exception as e:
                logging.error(f"Error releasing VideoCapture: {e}")
            self.cap = None

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
