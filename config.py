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
    # falls back to cv2.VideoCapture, which on this Pi enumerates the DECXIN as
    # /dev/video1. The same rule applies if a Picamera2 build ever captures the
    # same UVC device.
    #
    #   "auto"    - split when the frame's width/height >= 2.2, which is true of
    #               every DECXIN native mode and of nothing a single-lens webcam
    #               produces. If width and height describe exactly one eye of the
    #               configured stereo pair (2 * width == stereo_capture_width and
    #               height == stereo_capture_height), the device is asked for the
    #               full pair; that keeps the auto path from silently shrinking
    #               the eyes to a non-native resolution.
    #   "always"  - split every frame in half, and always ask the device for the
    #               full stereo_capture_width x stereo_capture_height (below).
    #               main.py forces this whenever the depth backend is "stereo",
    #               so the depth camera gets the native, un-squashed pair through
    #               Camera.latest_stereo_frame().
    #   "never"   - leave the frame alone.
    split_stereo_frame: str = "auto"
    # Which half survives: "left" (the reference eye used by depth_camera.py) or
    # "right". The choice is cosmetic for line following -- both eyes see the
    # same track -- but keeping it consistent avoids surprises when comparing a
    # frame against the depth pipeline.
    stereo_eye: str = "left"

    # 拆分前的原始尺寸：左右两眼并排的总宽 / 单眼的高。
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
    # 走 V4L2 直读解码后的帧，明显更锐。只有在 split_stereo_frame 不是 "never"
    # 时才会尝试，失败会自动回退到 Picamera2。
    prefer_v4l2: bool = True

    # 强制使用的 V4L2 节点。
    #
    # 空串 = 自动扫描 USB 视频节点（跳过 CSI / codec）。在这台 Pi 上留空即可：
    # 唯一能出帧的彩色节点是 /dev/video1，camera.py 会按 USB 总线自动挑中它，
    # 并读一帧验证后才宣告成功。
    #
    # 什么时候要填：如果树莓派上同时挂了多块 USB 相机，或者自动扫描选错了。
    # 可填 "/dev/video1" 或 "1"。
    #
    # 之前 camera.py 硬编码 cv2.VideoCapture(0)，而这里 /dev/video0 是 CSI
    # （unicam）节点：能打开，但永远读不出帧。Camera.start() 因此报告成功，
    # capture_frame() 每帧落到合成假图 —— 车实际在"看"一条画出来的路。
    v4l2_device: str = ""

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
    # Only the RealSense backend uses it.
    capture_timeout_ms: int = 200

    # Whether to also open the colour stream. Off by default.
    enable_color_stream: bool = False

    # ------------------------------------------------------------------
    # Stereo backend
    #
    # The two lenses are interleaved in a single UVC frame: left eye = left
    # half, right eye = right half. The frame is shared with camera.py, so
    # SGBM works on the native, un-squashed pair.
    #
    # Disparity comes back as horizontal pixel shift; depth is then
    #     depth_mm = scale_mm_px / disparity_px
    # so stereo *must* be calibrated before it can report millimetres.
    #
    # test_stereo_calib.py measures the number for you: put a high-contrast
    # target at a known distance, run
    #     python3 test_stereo_calib.py --distance 1000
    # and copy the printed stereo_scale_mm_px into this file.
    # ------------------------------------------------------------------

    # 单眼分辨率（深度计算用）。
    #
    # 必须与 camera.stereo_capture_width / 2、camera.stereo_capture_height 相等。
    stereo_width: int = 640
    stereo_height: int = 480

    # 视差计算的频率上限。SGBM 比原来的 StereoBM 慢，但在这个尺寸上 640x480
    # 单次约 20-30ms（Pi 4 单核），跑满 30 Hz 会吃掉一个核并拖慢 PID，
    # 10 Hz 对前方障碍检测足够。与 camera 的采集率解耦：camera 每帧更新原始帧，
    # sampler 按这个频率取最新一帧。
    stereo_target_fps: int = 10

    # Calibration, in mm * px. This is the ONE number to measure, and it is
    # exactly focal_px * baseline_mm.
    #
    # IMPORTANT: this value is tied to the eye resolution in stereo_width. If
    # the resolution changes, the same target at the same distance produces a
    # different disparity and the scale MUST be re-measured. depth_camera.py
    # scales this number linearly with the real eye width as a safety net, but
    # a wrong baseline would still be wrong.
    #
    # 39074 是从旧值 26375（432x320 单眼）按宽度线性推算的 640 宽单眼等效值，
    # 用于把旧标定迁移过来。它不是实测值——正式比赛前请用
    # test_stereo_calib.py 重新测量并覆盖这里。
    #
    # A scale of 0 means "not calibrated": the stereo backend refuses to start
    # and falls back to mock. That is deliberate, not a bug.
    stereo_scale_mm_px: float = 39074.0

    # Physical alternative to the measured scale: if the module's datasheet
    # gives the lens focal length (pixels) and the baseline (mm), the product
    # is the same number. Only used when stereo_scale_mm_px is 0.
    stereo_focal_px: float = 0.0
    stereo_baseline_mm: float = 0.0

    # 用 StereoSGBM 而不是 StereoBM。
    #
    # BM 对噪声极敏感：低光或者纹理少的场景（白墙、天花板）下，有效像素比会
    # 掉到个位数，表现为"深度经常读不出"。SGBM 用平滑代价聚合，在这种场景下
    # 能保留大量有效点，代价是慢 3-5 倍。
    stereo_use_sgbm: bool = True

    # SGBM 的算法变体，取值 "sgbm" / "3way" / "hh" / "hh4"。
    #
    # 含义：
    #   "sgbm" - OpenCV 默认 SGBM，质量中等，速度中等
    #   "3way" - 3 路扫描，比 sgbm 慢一点但匹配更稳（推荐用于低光/弱纹理）
    #   "hh"   - 高质量模式，慢 3 倍以上
    #   "hh4"  - 4.4+ 才有的高质量快变体
    #
    # Pi 4 上 640x480 + 192 disparity 的组合，用 "3way" 大约 80-120ms/帧，
    # stereo_target_fps=10 基本能跑满。如果 CPU 吃紧，改回 "sgbm"。
    stereo_sgbm_mode: str = "3way"

    # Stereo matching parameters.
    #
    # num_disparities must be divisible by 16; it is the widest pixel shift the
    # matcher searches, so it sets the CLOSEST distance the guard can resolve:
    #     closest_mm = scale_mm_px / (num_disparities - 1)
    # 在 640x480 单眼 + scale=39074 下，192 对应最近约 204mm，比
    # emergency_stop_mm=300 留出 100mm 缓冲。
    # 如果实测 scale 更大（比如 50000），直接改大这个值——否则近处会读不出。
    stereo_num_disparities: int = 192

    # block_size must be odd. Larger smooths noise but blurs thin obstacles.
    stereo_block_size: int = 15

    # SGBM 的最小视差（pixel）。通常 0 即可。
    stereo_min_disparity: int = 0

    # 唯一性比率：匹配代价最优值必须比次优值明显好，才认为这个像素有效。
    # 越小越容易产生噪声点，越大越容易漏掉真障碍。实测 8 在低纹理场景下
    # 明显比默认 10 多保留有效点，代价是零星噪声。如果深度图开始跳，
    # 调回 10。
    stereo_uniqueness_ratio: int = 8

    # Speckle filter：把"孤立的小块匹配噪声"抹成 0。
    stereo_speckle_window_size: int = 200
    stereo_speckle_range: int = 2

    # 左右一致性校验的容差（SGBM LR check 的阈值），单位 pixel。
    # 0 表示关闭。1-2 能有效剔除半遮挡区域的错误匹配。
    # 实测在低纹理场景下 1 太严（远处匹配全被拒），2 是恢复有效读数的关键。
    stereo_disp12_max_diff: int = 2

    # 视差图上做一次中值滤波，核大小（奇数）。1 = 关闭。
    stereo_median_blur: int = 5

    # 预处理：CLAHE 增强对比度。设 0.0 表示关闭。
    #
    # 干什么：在灰度图上做局部直方图均衡，把弱纹理（灰墙、地砖）从噪声里
    # 拉出来。SGBM 需要的是"有对比度的边缘"，低光或均匀光照下原图没有多少
    # 可匹配的细节，有效像素比会掉到个位数。
    #
    # 实测效果：室内 640x480 场景，关闭时有效像素比 ~5%，clipLimit=2.0 时
    # 能到 30-50%。同时因为 CLAHE 只改变对比度不改变几何，已匹配点的视差值
    # 不变，不需要重新标定。
    #
    # 副作用：clipLimit 过高（>4）会在强光区产生伪影，反而降低匹配率。
    stereo_clahe_clip_limit: float = 2.0

    # SGBM 的 preFilterCap：预处理阶段 Sobel 梯度的截断值，影响边缘响应强度。
    # 默认 31 是 OpenCV 推荐值，低光下有时调到 63 能给弱边缘更多权重。
    stereo_pre_filter_cap: int = 31

    # DECXIN 的左右顺序是否要翻转。
    #
    # SGBM 假设 right image 相对 left image 向右平移（视差为正）。有些廉价
    # 双目模组把右眼排在双帧的左半边，硬编码的"左前右后"就反了——表现是
    # 视差图几乎全黑，或有效像素比极低。
    #
    # 判断方法：跑 test_stereo_debug.py，如果 disparity.png 一片黑而 left/right
    # 两张图本身有内容，就把它设成 True 再试。
    stereo_swap_eyes: bool = False

    # ------------------------------------------------------------------
    # Obstacle ranging
    # ------------------------------------------------------------------

    # Readings outside this range (in mm) are treated as invalid. Guards against
    # both the 0 that depth cameras report for "no return" and the occasional
    # overflow value, either of which would otherwise read as a wall right in
    # front of the car and trigger a phantom emergency stop.
    min_depth_mm: float = 150.0
    max_depth_mm: float = 5000.0

    # The forward scan band, as fractions of the frame.
    # 纵向中心：0.70 大致是车头前 1 米处的地面。
    obstacle_scan_y_ratio: float = 0.70
    # 纵向厚度：以 obstacle_scan_y_ratio 为中心，向上/下各延伸这个比例。
    # 0.15 意味着实际扫描的是 55%-85% 高度的一条带，而不是一行——
    # 单行一旦落在白墙、反光区、车头阴影上就整次失效，这就是旧代码
    # "很多时候没读数"的主要原因之一。
    obstacle_scan_band_height_ratio: float = 0.15
    # 横向宽度：画面中心这条比例的宽度用于测距。
    obstacle_scan_width_ratio: float = 0.50

    # 一次有效测量至少需要多少像素。
    # 太少时（比如 SGBM 只匹配到 3 个像素）测出来的"最近距离"是噪声，
    # 会误触发刹车。20 是在 640x480 + 640 宽扫描带上测到的典型下限。
    obstacle_min_valid_pixels: int = 20

    # 取分位数而不是 min()。
    #
    # 语义：把所有在 [min, max] 范围内的像素排序，取第 P 百分位的值。
    # 10 意味着"前方有 10% 的像素比这个距离更近"——比 min() 更抗单个错误
    # 匹配点，比中位数更保守（不会因为一两个远处点而漏掉真障碍）。
    #
    # 如果误刹车频发，降低到 5；如果真障碍检测不够灵敏，提高到 20-30。
    obstacle_percentile: float = 10.0

    # Distance thresholds in mm. Below emergency_stop_mm the car is commanded to
    # (0, 0) immediately; below warning_mm it is only speed-limited.
    emergency_stop_mm: float = 300.0
    warning_mm: float = 800.0

    # The throttle ceiling inside the warning band.
    warning_throttle: float = 0.15

    # ------------------------------------------------------------------
    # DECXIN / generic UVC single-stream backend ("decxin")
    # ------------------------------------------------------------------

    # Preferred V4L2 node, or a bare OpenCV index like "0".
    uvc_device: str = ""

    # Pixel format requested from the device.
    uvc_fourcc: str = "Z16 "

    # Millimetres represented by one raw count.
    uvc_depth_unit_mm: float = 1.0

    # ------------------------------------------------------------------
    # Web console readout
    # ------------------------------------------------------------------

    # Resolution of the coarse depth grid handed to the browser for the cursor
    # readout.
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
    hsv_white_lower: tuple = (0, 0, 180)
    hsv_white_upper: tuple = (180, 50, 255)

    hsv_dark_lower: tuple = (0, 0, 0)
    hsv_dark_upper: tuple = (180, 255, 80)

    # Canny edge detection thresholds
    canny_threshold1: int = 50
    canny_threshold2: int = 150

    # Number of scanlines / slices for center extraction
    num_scan_slices: int = 5

    # Car physical dimensions (in centimeters)
    car_width_cm: float = 15.0
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
    ultrasonic_sample_interval: float = 0.05


@dataclass
class StreamConfig:
    # Web MJPEG Video Streamer settings (for viewing live camera feed in browser on PC)
    enable_stream: bool = False
    host: str = "0.0.0.0"
    port: int = 8080
    jpeg_quality: int = 70
    fps: int = 15


@dataclass
class RemoteConfig:
    """
    Web remote control settings for debugging (live tuning + manual driving).
    """
    allow_tuning: bool = False   # Live parameter tuning; the car stays autonomous.
    allow_manual: bool = False   # Manual drive tab; implies allow_tuning.

    deadman_timeout_s: float = 0.8

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
