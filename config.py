"""
Configuration file for PiRacer Pro Folkrace Autonomous Vehicle.
Contains all hardware settings, PID parameters, vision thresholds, and sensor configs.
"""

from dataclasses import dataclass, field


@dataclass
class CameraConfig:
    # 输出给 vision.py / webui 的分辨率（单眼）。
    #
    # 这个值要与 stereo_capture_width / 2 和 stereo_capture_height 对齐，
    # 否则 camera.py 会把单眼硬 resize 到这个尺寸——之前 width=432/height=240
    # 配 stereo_capture 864x320，就是把 432x320 的眼睛压成 432x240 的元凶，
    # VisionConfig 的 ROI 也跟着一起错位。
    #
    # 640x480 是 1280x480 双目拆出来的单眼原生尺寸，跟摄像头对齐后不会有任何
    # resize 拉伸。
    width: int = 640
    height: int = 480
    framerate: int = 30
    format: str = "RGB888"  # or 'BGR888'

    # Side-by-side stereo split for the *track* camera.
    #
    # The DECXIN-SM-2930V1 and similar dual-lens UVC modules emit one double-wide
    # frame with the left eye on the left half and the right eye on the right
    # half. That is what depth_camera.py's "stereo" backend wants, and it splits
    # the pair itself -- but the same frame must NEVER reach vision.py, because
    # the line follower would see two tracks side by side and steer straight at
    # the seam between the eyes.
    #
    # It reaches the track camera when the CSI module is missing and camera.py
    # falls back to cv2.VideoCapture(0), which on this Pi now enumerates the
    # DECXIN as /dev/video0. The same rule applies if a Picamera2 build ever
    # captures the same UVC device.
    #
    #   "auto"    - split only when the frame's width/height >= 2.2, which is
    #               true of every DECXIN native mode and of nothing a single-lens
    #               webcam produces.
    #   "always"  - split every frame in half, and ask the device for the full
    #               stereo_capture_width x stereo_capture_height (below) instead
    #               of the target width x height. main.py forces this whenever the
    #               depth backend is "stereo", so the depth camera gets the
    #               native, un-squashed pair through Camera.latest_stereo_frame().
    #   "never"   - leave the frame alone.
    split_stereo_frame: str = "auto"
    # Which half survives: "left" (the reference eye used by depth_camera.py) or
    # "right". The choice is cosmetic for line following -- both eyes see the
    # same track -- but keeping it consistent avoids surprises when comparing a
    # frame against the depth pipeline.
    stereo_eye: str = "left"

    # 拆分前的原始尺寸：左右两眼并排的总宽 / 单眼的高。
    # 只有 split_stereo_frame="always" 时生效——此时 Camera 会以这个尺寸向设备
    # 请求采集，并把未拆分帧共享给 depth_camera.py。
    #
    # 1280x480 是 DECXIN-SM-2930V1 的一个原生模式（30fps@MJPG），单眼 640x480。
    # 之前的 864x320 不在模组的支持列表里（native 只有 1280x480 / 1600x600 /
    # 2560x960 / 2560x1080 / 3840x1080 / 2560x720），驱动会回退到最近的一个模式，
    # 导致 depth_camera 按错误宽度计算 scale。请按 `v4l2-ctl -d <node>
    # --list-formats-ext` 的真实列表选。
    #
    # 改了这两项后，stereo_scale_mm_px 必须重新校准：视差里 focal 是按像素算的，
    # 分辨率改变后同一块靶子在同一距离上给出的视差也会改变，旧的 scale 会把距离
    # 算错。depth_camera 会按眼宽自动线性换算，但那只是防止彻底失效的兜底，
    # 精确值仍然要实测。
    stereo_capture_width: int = 1280
    stereo_capture_height: int = 480

    # 双目模式下走 V4L2 直连，绕过 Picamera2。
    #
    # Picamera2 会对 UVC 的 MJPG 流再做一层 ISP 处理（tone mapping / 缩放），
    # 对一块只有 MJPG 的双目相机来说这是画质的主要损失来源；cv2.VideoCapture
    # 走 V4L2 直读解码后的帧，明显更锐。只有在 split_stereo_frame="always"
    # （即 main.py 已经确认这是一台双目相机）时才会启用，CSI 相机不受影响。
    # 若直连失败，start() 会自动回退到 Picamera2。
    prefer_v4l2: bool = True

    # 强制使用的 V4L2 像素格式。
    #
    # 这是之前只有 25fps 的根因：不设 FOURCC 时驱动会选 YUYV（未压缩），
    # 1280x480@30fps ≈ 37 MB/s 就卡在 USB 2.0 的实测上限附近，驱动自动降级到
    # 25fps（甚至更低）。MJPG 只要约 1/10 的带宽，能稳定跑满 30fps。
    # 设成 "" 表示不干预、让驱动自己选（调试时用）。
    v4l2_fourcc: str = "MJPG"

    # 可选的 V4L2 曝光 / 增益控制。默认 -1 表示"不要碰设备"。
    #
    # 什么时候需要动它们：低光下驱动会拉长曝光时间，帧率被压到 25/20/15fps，
    # 同时噪点增加，SGBM 在噪声大的图上大量匹配失败 → 深度读数为 0。
    # 固定住曝光和增益能让深度读数稳定，代价是亮处可能过曝，跑赛道前要调。
    #
    # v4l2_auto_exposure_mode 的取值依赖驱动和 OpenCV 后端：常见的是
    # 1 = 手动、3 = 自动（V4L2_EXPOSURE_* 枚举），也有 0.25/0.75 的旧形式。
    # 建议先用 `v4l2-ctl -d /dev/video0 --list-ctrls` 看设备支持什么，再试。
    v4l2_auto_exposure_mode: float = -1.0
    # 曝光绝对值（单位是 100µs，具体取决于驱动）。仅在 auto_exposure_mode 手动时有效。
    v4l2_exposure_absolute: float = -1.0
    # 模拟增益。低光下适当抬高能减少噪点对匹配的破坏，太高又会引入新的噪点。
    v4l2_gain: float = -1.0


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
    with ultrasonic/IR, and nothing there looks directly ahead.
    """
    # Master switch. --allow-depth-camera is what normally flips this.
    enable_depth: bool = False

    # Backend to use: "stereo", "decxin", "realsense", or "mock".
    #
    #   stereo    - DECXIN-SM-2930V1 and similar dual-lens UVC modules. The
    #               device exposes only MJPG/YUYV colour (confirmed with
    #               `v4l2-ctl -d /dev/video0 --list-formats-ext`: no Z16/Y16
    #               anywhere), and encodes the two lenses as one double-wide
    #               side-by-side image. Depth is reconstructed on the host by
    #               stereo matching. REQUIRES CALIBRATION -- see below.
    #               Main.py forces camera.split_stereo_frame="always" so that
    #               Camera captures the native pair; depth_camera.py then reuses
    #               Camera.latest_stereo_frame() instead of opening the device a
    #               second time.
    #   decxin    - a UVC camera that really does list a 16-bit depth FOURCC
    #               (Z16/Y16/...). Not this hardware; kept for other cameras.
    #   realsense - Intel RealSense via pyrealsense2.
    #   mock      - synthesises a flat 3 m wall. The guard never fires, which
    #               is the intended "no depth camera" fallback.
    #
    # An unavailable backend always falls back to mock rather than raising, so a
    # missing library never takes the car off the track -- it just loses depth.
    backend: str = "stereo"

    # Device serial to open; empty means "the first one that answers".
    # Only the RealSense backend uses this.
    device_serial: str = ""

    # Depth stream resolution and rate. For the stereo backend these size the
    # *mock* fallback; the live stereo frame is sized by stereo_width/_height.
    width: int = 640
    height: int = 480
    fps: int = 30

    # How long to wait for a frame before giving up on this iteration, in ms.
    # Deliberately short: the depth guard runs inside the 30 Hz control loop, so
    # a stalled depth camera must not be allowed to hold the whole loop hostage.
    # Only the RealSense backend uses it. The UVC-based backends (stereo, decxin)
    # read on a background thread instead, so a slow frame cannot stall the loop.
    capture_timeout_ms: int = 200

    # Whether to also open the colour stream. Off by default -- the main camera
    # already provides colour for line following, and an extra colour stream is
    # pure USB bandwidth that the operator's drive commands have to share.
    enable_color_stream: bool = False

    # ------------------------------------------------------------------
    # Stereo backend (DECXIN-SM-2930V1 and similar dual-lens UVC modules)
    #
    # The two lenses are interleaved in a single UVC frame: left eye = left
    # half, right eye = right half. The frame is shared with camera.py, so
    # SGBM works on the native, un-squashed pair.
    #
    # Disparity comes back as horizontal pixel shift; depth is then
    #     depth_mm = scale_mm_px / disparity_px
    # so stereo *must* be calibrated before it can report millimetres. A wrong
    # scale is not just "off by a bit": it either triggers a phantom emergency
    # stop, or -- worse -- reports a clear road where there is a wall. The
    # backend refuses to start without a scale and falls back to mock.
    #
    # test_stereo_calib.py measures the number for you: put a high-contrast
    # target at a known distance, run
    #     python3 test_stereo_calib.py --distance 1000
    # and copy the printed stereo_scale_mm_px into this file.
    # ------------------------------------------------------------------

    # 单眼分辨率（深度计算用）。
    #
    # 必须与 camera.stereo_capture_width / 2、camera.stereo_capture_height 相等。
    # 若与实际帧宽不一致，depth_camera 会按比例自动换算 scale 兜底，但那只能
    # 防止彻底失效，精度仍需要按实际宽度重新标定。
    stereo_width: int = 640
    stereo_height: int = 480

    # 视差计算的频率上限。SGBM 比原来的 StereoBM 慢，但在这个尺寸上 640x480
    # 单次约 20-30ms（Pi 4 单核），跑满 30 Hz 会吃掉一个核并拖慢 PID，
    # 10 Hz 对前方障碍检测足够。与 camera 的采集率解耦：camera 每帧更新原始帧，
    # sampler 按这个频率取最新一帧。
    stereo_target_fps: int = 10

    # Calibration, in mm * px. This is the ONE number to measure, and it is
    # exactly focal_px * baseline_mm. To get it:
    #   1. Print a high-contrast target (chequerboard, printed crosshair).
    #   2. Stand it at a known distance D in millimetres, square to the camera.
    #   3. Read the disparity d at that target from the centre of a disparity
    #      frame (or run test_stereo_calib.py, which reads it for you).
    #   4. Set stereo_scale_mm_px = D * d.
    #
    # IMPORTANT: this value is tied to the eye resolution in stereo_width. If
    # stereo_width / stereo_height (or camera.stereo_capture_width / height)
    # changes, the same target at the same distance produces a different
    # disparity and the scale MUST be re-measured. depth_camera.py scales this
    # number linearly with the real eye width as a safety net, but a wrong
    # baseline would still be wrong.
    #
    # 39074 是从旧值 26375（432x320 单眼）按宽度线性推算的 640 宽单眼等效值，
    # 用于把旧标定迁移过来。它不是实测值——正式比赛前请用
    # test_stereo_calib.py 重新测量并覆盖这里。
    #
    # A scale of 0 means "not calibrated": the stereo backend refuses to start
    # and falls back to mock. That is deliberate, not a bug.
    stereo_scale_mm_px: float = 39074.0

    # Physical alternative to the measured scale: if the module's datasheet
    # gives the lens focal length (pixels, at the eye width set above) and the
    # baseline (distance between the lenses, mm), the product is the same
    # number. Only used when stereo_scale_mm_px is 0.
    stereo_focal_px: float = 0.0
    stereo_baseline_mm: float = 0.0

    # 用 StereoSGBM 而不是 StereoBM。
    #
    # BM 对噪声极敏感：低光或者纹理少的场景（白墙、天花板）下，有效像素比会
    # 掉到个位数，表现为"深度经常读不出"。SGBM 用平滑代价聚合，在这种场景下
    # 能保留大量有效点，代价是慢 3-5 倍。10Hz 采样下这个代价可以接受。
    # 设 False 可以退回 BM（用于对比排查）。
    stereo_use_sgbm: bool = True

    # Stereo matching parameters.
    #
    # num_disparities must be divisible by 16; it is the widest pixel shift the
    # matcher searches, so it sets the CLOSEST distance the guard can resolve:
    #     closest_mm = scale_mm_px / (num_disparities - 1)
    # 在 640x480 单眼 + scale=39074 下，160 对应最近约 246mm，比
    # emergency_stop_mm=300 略近，够用。大尺寸需要按比例增大。
    # The backend warns at start if that is farther than emergency_stop_mm,
    # because obstacles inside the blind zone would never register.
    stereo_num_disparities: int = 160

    # block_size must be odd. Larger smooths noise but blurs thin obstacles.
    stereo_block_size: int = 15

    # SGBM 的最小视差（pixel）。通常 0 即可，只有在用了极窄基线或者需要裁掉
    # 视野边缘的噪声时才需要抬高。
    stereo_min_disparity: int = 0

    # 唯一性比率：匹配代价最优值必须比次优值明显好，才认为这个像素有效。
    # 越小越容易产生噪声点，越大越容易漏掉真障碍。5..15 是常见范围。
    stereo_uniqueness_ratio: int = 10

    # Speckle filter：把"孤立的小块匹配噪声"抹成 0。
    # speckle_window_size 是判定为孤立块的最大面积，speckle_range 是允许的
    # 视差差异。调大能把低光下的零星噪点压掉，代价是会抹掉真实的细小障碍。
    stereo_speckle_window_size: int = 200
    stereo_speckle_range: int = 2

    # 左右一致性校验的容差（SGBM LR check 的阈值），单位是 pixel。
    # 0 表示关闭。1-2 能有效剔除半遮挡区域的错误匹配。
    stereo_disp12_max_diff: int = 1

    # 视差图上做一次中值滤波，核大小（奇数）。1 = 关闭。
    # 这一步同时平滑掉斑点，让 measure_obstacle_mm 的 min() 不轻易被单点触发。
    stereo_median_blur: int = 5

    # ------------------------------------------------------------------
    # DECXIN / generic UVC single-stream backend ("decxin")
    #
    # Only relevant for a camera that really does expose a 16-bit depth format
    # over UVC; the DECXIN-SM-2930V1 on this car does not (it needs "stereo").
    # Kept so the same code drives a Z16/Y16 device without changes.
    #
    # The node is discovered at runtime; the value below is only a hint.
    # `v4l2-ctl --list-devices` numbers nodes in USB enumeration order, which is
    # not stable. Non-USB nodes are excluded, because the CSI camera that
    # picamera2 drives sits on a platform bus and must not be taken away.
    # ------------------------------------------------------------------

    # Preferred V4L2 node, or a bare OpenCV index like "0".
    # Empty means "scan every USB video node" -- the usual, robust choice.
    # A path that is not a USB node is refused rather than opened.
    uvc_device: str = ""

    # Pixel format requested from the device. "Z16 " / "Y16 " are the 16-bit
    # depth formats a UVC depth camera normally exposes; the trailing space is
    # part of the four-character code, not a typo. If the device ignores it and
    # stays on YUYV, the backend says so in the log rather than decoding colour
    # as depth. Confirm with:
    #     v4l2-ctl -d <node> --list-formats-ext
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
class OLEDConfig:
    """
    OLED display settings for PiRacer Pro onboard screen.
    Shows CPU, memory usage, and car speed.
    """
    enable_oled: bool = True
    width: int = 128
    height: int = 64
    i2c_address: int = 0x3C  # Default I2C address for SSD1306
    update_interval: float = 0.5  # Update display every 0.5 seconds


@dataclass
class AppConfig:
    camera: CameraConfig = field(default_factory=CameraConfig)
    depth: DepthCameraConfig = field(default_factory=DepthCameraConfig)
    vision: VisionConfig = field(default_factory=VisionConfig)
    control: ControlConfig = field(default_factory=ControlConfig)
    sensor: SensorConfig = field(default_factory=SensorConfig)
    stream: StreamConfig = field(default_factory=StreamConfig)
    remote: RemoteConfig = field(default_factory=RemoteConfig)
    oled: OLEDConfig = field(default_factory=OLEDConfig)

    # Log level and loop target rate
    target_loop_hz: int = 30

    # Robustness: abort the main loop (and stop the vehicle) after this many
    # consecutive frame-processing errors. A single bad frame is skipped instead.
    max_consecutive_frame_errors: int = 10
