"""
Configuration file for PiRacer Pro Folkrace Autonomous Vehicle.
Contains all hardware settings, PID parameters, vision thresholds, and sensor configs.
"""

from dataclasses import dataclass, field


@dataclass
class CameraConfig:
    # Camera resolution and framerate
    # OV5647 native fast binned mode is 640x480 @ ~60fps
    width: int = 640
    height: int = 460
    framerate: int = 30
    format: str = "RGB888"  # or 'BGR888'


@dataclass
class DepthCameraConfig:
    """
    Optional depth camera settings.

    The depth camera runs *alongside* the main camera rather than replacing it:
    Camera keeps feeding the line-following pipeline unchanged, and depth only
    adds a forward-obstacle guard on top. That split is deliberate -- the depth
    camera can drop off the USB bus, run out of frames, or be missing entirely,
    and none of those should stop the car from following the line.

    Front is where the depth camera earns its place: sensors.py covers the sides
    with ultrasonic/IR, and nothing there looks at what is directly ahead.
    """
    # Master switch. --allow-depth-camera is what normally flips this.
    enable_depth: bool = False

    # Backend to use: "decxin" (UVC), "realsense", or "mock".
    # An unavailable backend always falls back to mock rather than raising, so a
    # missing library never takes the car off the track -- it just loses depth.
    backend: str = "decxin"

    # Device serial to open; empty means "the first one that answers".
    # Only the RealSense backend uses this.
    device_serial: str = ""

    # Depth stream resolution and rate.
    width: int = 640
    height: int = 480
    fps: int = 30

    # How long to wait for a frame before giving up on this iteration, in ms.
    # Deliberately short: the depth guard runs inside the 30 Hz control loop, so
    # a stalled depth camera must not be allowed to hold the whole loop hostage.
    # Only the RealSense backend uses it; OpenCV's blocking read is left as-is
    # because a UVC device that stops answering is already returning empty frames.
    capture_timeout_ms: int = 200

    # Whether to also open the colour stream. Off by default -- the main camera
    # already provides colour for line following, and an extra colour stream is
    # pure USB bandwidth that the operator's drive commands have to share.
    enable_color_stream: bool = False

    # ------------------------------------------------------------------
    # DECXIN-SM-2930V1 / generic UVC depth backend
    #
    # The camera hangs off the Pi's USB-C port as a plain UVC video device, so it
    # needs no vendor library -- which is also why this backend is the default.
    # Enumerate what the Pi actually sees with:
    #     v4l2-ctl --list-devices
    #     v4l2-ctl -d /dev/video1 --list-formats-ext
    # and put the *capture* node of the depth camera (not its metadata node, and
    # never the CSI camera) in uvc_device.
    #
    # The DECXIN enumerates as a pair of nodes plus a media node:
    #     /dev/video1   -> the UVC capture interface, this is the one that streams
    #     /dev/video2   -> the metadata node for video1, never returns a frame
    #     /dev/media5   -> media controller, not opened directly
    # Note that /dev/video0 on this car is the CSI camera (unicam, the picamera2
    # one); pointing this setting at video0 would fight the main camera instead.
    # ------------------------------------------------------------------

    # V4L2 device node, or a bare OpenCV index like "0".
    uvc_device: str = "/dev/video1"

    # Pixel format requested from the device. "Z16 " / "Y16 " are the 16-bit
    # depth formats a UVC depth camera normally exposes; the trailing space is
    # part of the four-character code, not a typo.
    uvc_fourcc: str = "Z16 "

    # Millimetres represented by one raw count. 1.0 is the common case (the device
    # already reports millimetres); a device reporting in 0.1 mm units wants 0.1.
    # If every measured distance looks off by a constant factor, this is the knob.
    uvc_depth_unit_mm: float = 1.0

    # ------------------------------------------------------------------
    # Ranging
    # ------------------------------------------------------------------

    # Readings outside this range (in mm) are treated as invalid. Guards against
    # both the 0 that depth cameras report for "no return" and the occasional
    # overflow value, either of which would otherwise read as a wall right in
    # front of the car and trigger a phantom emergency stop.
    min_depth_mm: float = 150.0
    max_depth_mm: float = 5000.0

    # The forward scan band, as fractions of the frame.
    # The row sits low in the image on purpose: that is the ground the front
    # wheels are about to reach, not the horizon, which is always far away and
    # would make the guard blind.
    obstacle_scan_y_ratio: float = 0.70
    obstacle_scan_width_ratio: float = 0.50

    # Distance thresholds in mm. Below emergency_stop_mm the car is commanded to
    # (0, 0) immediately; below warning_mm it is only speed-limited.
    emergency_stop_mm: float = 300.0
    warning_mm: float = 800.0

    # The throttle ceiling inside the warning band.
    warning_throttle: float = 0.15

    # ------------------------------------------------------------------
    # Web console readout
    # ------------------------------------------------------------------

    # Resolution of the coarse depth grid handed to the browser for the cursor
    # readout. Deliberately tiny: the page polls it once a second and only needs
    # enough detail to answer "roughly how far is the wall under this spot", so
    # 32x24 keeps the whole payload to a few kB instead of shipping a full frame.
    ui_grid_width: int = 32
    ui_grid_height: int = 24


@dataclass
class VisionConfig:
    # Region of Interest (ROI) - vertical cropping from bottom (0.0: top, 1.0: bottom)
    roi_top_ratio: float = 0.45  # Focus on the track in front, ignore ceiling/horizon
    roi_bottom_ratio: float = 0.95

    # Track detection mode: 'lane_line' (white/yellow line), 'color_mask', or 'edge_contours'
    detection_mode: str = "edge_contours"

    # Color thresholding parameters (HSV) for track / line following
    # Default white line threshold in HSV
    hsv_white_lower: tuple = (0, 0, 180)
    hsv_white_upper: tuple = (180, 50, 255)

    # Default dark road / track threshold (if tracking dark track on light surface)
    hsv_dark_lower: tuple = (0, 0, 0)
    hsv_dark_upper: tuple = (180, 255, 80)

    # Canny edge detection thresholds
    canny_threshold1: int = 50
    canny_threshold2: int = 150

    # Number of scanlines / slices for center extraction
    num_scan_slices: int = 5

    # Car physical dimensions (in centimeters)
    car_width_cm: float = 15.0
    # Physical track width in centimetres. Used only to turn car_width_cm into a
    # dimensionless safety-margin ratio; it is not a pixel measurement.
    track_width_cm: float = 60.0

    # Debug image visualization flag
    show_debug_window: bool = False


@dataclass
class ControlConfig:
    # control
    kp: float = 0.65
    ki: float = 0
    kd: float = 0.12
    base_throttle: float = 0.3
    max_throttle: float = 0.5
    min_throttle: float = 0.22
    throttle_deadband: float = 0.18
    turn_slowdown_factor: float = 0.06
    reverse_throttle: float = -0.25
    steering_trim: float = 0
    esc_arm_time: float = 1.0
    invert_steering: bool = False
    invert_throttle: bool = False
    steering_sensitivity: float = 1.0
    min_steering: float = -1.0
    max_steering: float = 1.0

    # vision
    roi_top_ratio: float = 0.45
    roi_bottom_ratio: float = 0.95
    detection_mode: str = "edge_contours"
    num_scan_slices: int = 5
    canny_threshold1: int = 50
    canny_threshold2: int = 150


@dataclass
class SensorConfig:
    # Enable external distance sensors (IR / Ultrasonic)
    enable_sensors: bool = False

    # Left & Right Ultrasonic sensor pin definitions (BCM numbering)
    left_trig_pin: int = 23
    left_echo_pin: int = 24
    right_trig_pin: int = 27
    right_echo_pin: int = 22

    # Left & Right Infrared digital/analog sensor pins
    left_ir_pin: int = 17
    right_ir_pin: int = 18

    # Obstacle / Wall detection thresholds (in centimeters)
    emergency_stop_dist_cm: float = 15.0
    side_warning_dist_cm: float = 25.0

    # Interval (seconds) between ultrasonic sweeps in the background sampling thread.
    # Must exceed the worst-case blocking time of one sweep, otherwise readings go stale.
    ultrasonic_sample_interval: float = 0.05


@dataclass
class StreamConfig:
    # Web MJPEG Video Streamer settings (for viewing live camera feed in browser on PC)
    enable_stream: bool = False
    host: str = "0.0.0.0"
    port: int = 8080
    jpeg_quality: int = 70
    # Frames per second pushed to the browsers.
    #
    # The control loop produces ~30 fps and hands every frame over. At quality 70
    # that is roughly 10-15 Mbit/s of JPEG, which is the video stream's share of a
    # link the operator is also driving the car over -- and a drive command that
    # arrives later than RemoteConfig.deadman_timeout_s stops the car. Halving the
    # rate roughly halves that share and 15 fps is still a perfectly usable view.
    # Raise it only if the car is on a wired or otherwise quiet network.
    fps: int = 15


@dataclass
class RemoteConfig:
    """
    Web remote control settings for debugging (live tuning + manual driving).

    Both gates default to False. The web page is served to anyone on the LAN, so
    the terminal command that starts the program is what decides whether a control
    tab exists at all -- opening the page is never enough to take control of the
    car. A competition run must stay autonomous, so neither gate belongs in the
    race-day command line.
    """
    # Terminal-side gates (--allow-tuning / --allow-manual). Without them the web
    # page renders the video feed and telemetry only.
    allow_tuning: bool = False   # Live parameter tuning; the car stays autonomous.
    allow_manual: bool = False   # Manual drive tab; implies allow_tuning.

    # Deadman: if no drive command arrives within this window while armed, the car
    # is commanded to a full stop. Mandatory safety net -- do not disable.
    # Keep it above webui.SEND_TIMEOUT_MS (600 ms): the page abandons a drive request
    # after that long and sends a fresh one, and a window shorter than one retry would
    # stop the car on a link that is merely slow rather than actually gone.
    deadman_timeout_s: float = 0.8

    # Hard limits applied to operator input (server side, defence in depth).
    # max_manual_throttle deliberately sits below ControlConfig.max_throttle (0.50):
    # manual driving must never be faster than autonomous driving.
    max_manual_throttle: float = 0.40
    max_manual_reverse: float = -0.25
    max_manual_steering: float = 1.0


@dataclass
class AppConfig:
    camera: CameraConfig = field(default_factory=CameraConfig)
    depth: DepthCameraConfig = field(default_factory=DepthCameraConfig)
    vision: VisionConfig = field(default_factory=VisionConfig)
    control: ControlConfig = field(default_factory=ControlConfig)
    sensor: SensorConfig = field(default_factory=SensorConfig)
    stream: StreamConfig = field(default_factory=StreamConfig)
    remote: RemoteConfig = field(default_factory=RemoteConfig)

    # Log level and loop target rate
    target_loop_hz: int = 30

    # Robustness: abort the main loop (and stop the vehicle) after this many
    # consecutive frame-processing errors. A single bad frame is skipped instead.
    max_consecutive_frame_errors: int = 10
