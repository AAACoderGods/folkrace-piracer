#!/usr/bin/env python3
"""
Main entry point for PiRacer Pro Folkrace Autonomous Vehicle.
Integrates OpenCV vision, Picamera2 capture, PID motion control, and sensor avoidance.
"""

import argparse
import logging
import signal
import sys
import time

from config import AppConfig
from camera import Camera
from depth_camera import DepthCamera
from vision import RoadPerception
from controller import VehicleController
from sensors import SensorManager
from streamer import WebStreamer
from webcontrol import RemoteState, SOURCE_AUTO, SOURCE_MANUAL, SOURCE_PAUSED
from oled import OLEDDisplay

try:
    import cv2
except ImportError:
    cv2 = None

# Global running state flag
running = True


def signal_handler(signum, frame):
    """Handle termination signals safely."""
    global running
    logging.info(f"Termination signal {signum} received. Stopping vehicle...")
    running = False


def resolve_drive(remote: RemoteState, vehicle: VehicleController,
                  auto_steering: float, auto_throttle: float):
    """
    Pick the drive command for this frame.

    Priority: autonomous PID -> operator command -> deadman (0,0) -> paused (0,0).
    The last two are the safety net around an operator who stopped sending.
    """
    command = remote.resolve(time.monotonic())

    # A transition between driving modes must not inherit the integrator and
    # derivative history of the previous one.
    if remote.consume_pid_reset():
        vehicle.reset_pid()

    if command is None:
        return auto_steering, auto_throttle, SOURCE_AUTO
    return command.steering, command.throttle, command.source


def draw_mode_banner(frame, source: str):
    """
    Stamp the driving mode across the top of the video frame.

    The stream is where people actually look during a run, so an overlay is the
    most reliable way to make "this car is not driving itself" impossible to miss.
    Autonomous driving draws nothing, keeping a race-day feed clean.
    """
    if source == SOURCE_AUTO or cv2 is None:
        return

    if source == SOURCE_MANUAL:
        text, colour = "MANUAL CONTROL - NOT AUTONOMOUS", (0, 0, 220)
    elif source == SOURCE_PAUSED:
        text, colour = "PAUSED - CAR HELD", (0, 140, 255)
    else:
        text, colour = "DEADMAN - COMMANDS LOST", (0, 0, 220)

    width = frame.shape[1]
    cv2.rectangle(frame, (0, 0), (width, 46), colour, -1)
    (text_width, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    cv2.putText(frame, text, (max(0, (width - text_width) // 2), 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)


class DepthGuard:
    """
    Forward-obstacle guard built on the depth camera.

    Complements sensors.py rather than duplicating it: the ultrasonic and IR
    sensors watch the sides, this watches what is directly ahead. It is written
    as a class only so the "last logged at" timestamp has somewhere to live --
    a car stopped in front of a wall would otherwise log a warning on every
    single frame of the 30 Hz loop.
    """

    def __init__(self, camera: DepthCamera, config):
        self.camera = camera
        self.config = config
        self._last_log_ts = 0.0
        # Latest reading and coarse map, kept for the web console. Both are
        # refreshed on every call so the page shows this frame's numbers rather
        # than a stale one; None means "no usable depth right now".
        self.last_distance_mm = None
        self.grid = None

    def apply(self, steering: float, throttle: float):
        """
        Return (steering, throttle, is_emergency) after the depth guard has run.

        Only the emergency band changes steering, and it changes it to zero --
        steering *around* an obstacle the depth camera can see is not something
        this layer is allowed to attempt, because it cannot see the walls the
        car would be steered into. The warning band only caps throttle.
        """
        try:
            _, depth_frame = self.camera.capture_frame()
        except Exception as e:
            logging.debug(f"Depth capture failed: {e}")
            return steering, throttle, False

        self.grid = self.camera.depth_grid(depth_frame)
        distance_mm = self.camera.measure_obstacle_mm(depth_frame)
        self.last_distance_mm = distance_mm
        if distance_mm is None:
            return steering, throttle, False

        if distance_mm < self.config.emergency_stop_mm:
            # Edge-triggered on a 1 s interval. vehicle.stop() logs its own
            # transition, but "why did it stop" is a different question when the
            # depth camera is what asked, so a separate line is worth the noise.
            now = time.monotonic()
            if now - self._last_log_ts > 1.0:
                logging.warning("Depth guard: emergency stop, obstacle at %.0f mm.",
                                distance_mm)
                self._last_log_ts = now
            return 0.0, 0.0, True

        if distance_mm < self.config.warning_mm:
            return steering, min(throttle, self.config.warning_throttle), False

        return steering, throttle, False


def resolve_split_mode(config: AppConfig) -> str:
    """
    Decide how Camera should treat a side-by-side stereo frame.

    When the depth backend is "stereo", the module is known to be a dual-lens
    device, and the same device feeds the track camera. The auto heuristic can be
    defeated by a capture path that rescales the double-wide pair down to a normal
    aspect ratio before Camera sees it, so the split is forced in that case. The
    operator can still opt out with "never".
    """
    mode = (config.camera.split_stereo_frame or "auto").lower()
    if mode == "never":
        return mode
    if config.depth.enable_depth and (config.depth.backend or "").lower() == "stereo":
        if mode != "always":
            logging.info("Camera: forcing split_stereo_frame='always' because the "
                         "depth backend is 'stereo' -- the same dual-lens device "
                         "feeds the track camera.")
        return "always"
    return mode


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Folkrace Autonomous Driving on PiRacer Pro")
    parser.add_argument("--display", action="store_true", help="Show live OpenCV visual debug window (GUI)")
    parser.add_argument("--stream", action="store_true", help="Enable Web MJPEG live camera stream to browser on PC")
    parser.add_argument("--port", type=int, default=8080, help="Web stream HTTP port (default: 8080)")
    parser.add_argument("--throttle", type=float, default=None, help="Override base throttle (e.g. 0.3)")
    parser.add_argument("--mode", type=str, choices=["edge_contours", "lane_line", "color_mask"],
                        default=None, help="Road perception detection mode")
    parser.add_argument("--enable-sensors", action="store_true", help="Enable ultrasonic/IR side sensors")
    parser.add_argument("--fps", type=int, default=30, help="Target loop frequency in Hz")
    # Debug tooling. Neither flag belongs in a competition command line: with them
    # absent the web page is a plain video feed and the car behaves exactly as before.
    parser.add_argument("--allow-tuning", action="store_true",
                        help="Show the live parameter tuning tab on the web page (debug only)")
    parser.add_argument("--allow-manual", action="store_true",
                        help="Show the manual drive tab on the web page (test only; implies --allow-tuning)")
    # Optional hardware. With this absent the depth camera is never constructed,
    # so a machine without a working depth device behaves identically to today.
    parser.add_argument("--allow-depth-camera", action="store_true",
                        help="Enable the depth camera (DECXIN / UVC) for forward-obstacle ranging")
    return parser.parse_args()


def main():
    global running

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S"
    )

    # Register system signals for graceful exit
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    args = parse_args()

    # Load configuration
    config = AppConfig()
    if args.display:
        config.vision.show_debug_window = True
    if args.stream:
        config.stream.enable_stream = True
        config.vision.show_debug_window = True  # Generate vision overlay for web stream
    if args.port:
        config.stream.port = args.port
    if args.throttle is not None:
        config.control.base_throttle = args.throttle
    if args.mode is not None:
        config.vision.detection_mode = args.mode
    if args.enable_sensors:
        config.sensor.enable_sensors = True
    if args.fps is not None:
        config.target_loop_hz = args.fps
    if args.allow_manual:
        config.remote.allow_manual = True
        config.remote.allow_tuning = True   # driving by hand and tuning go together
    elif args.allow_tuning:
        config.remote.allow_tuning = True
    if args.allow_depth_camera:
        config.depth.enable_depth = True

    logging.info("=" * 50)
    logging.info(" Starting Folkrace PiRacer Pro Autonomous System")
    logging.info(f" Mode: {config.vision.detection_mode}")
    logging.info(f" Base Throttle: {config.control.base_throttle}")
    logging.info(f" Local GUI Display: {args.display}")
    logging.info(f" Web Browser Stream: {config.stream.enable_stream} (Port: {config.stream.port})")
    logging.info(f" Sensors Enabled: {config.sensor.enable_sensors}")
    logging.info(f" Depth Camera: {config.depth.enable_depth} "
                 f"(backend: {config.depth.backend}, device: {config.depth.uvc_device})")
    logging.info(f" Web Live Tuning: {config.remote.allow_tuning} | Web Manual Control: {config.remote.allow_manual}")
    logging.info(f" OLED Display: {config.oled.enable_oled}")
    logging.info("=" * 50)

    if config.remote.allow_manual:
        logging.warning("=" * 50)
        logging.warning(" MANUAL CONTROL ENABLED - TESTING ONLY")
        logging.warning(" The car starts PAUSED and waits for someone to hold ARM.")
        logging.warning(" A competition run must use neither --allow-manual nor --allow-tuning.")
        logging.warning("=" * 50)

    # Initialize subsystems
    camera = Camera(
        width=config.camera.width,
        height=config.camera.height,
        framerate=config.camera.framerate,
        format=config.camera.format,
        split_stereo_frame=resolve_split_mode(config),
        stereo_eye=config.camera.stereo_eye,
        stereo_capture_width=config.camera.stereo_capture_width,
        stereo_capture_height=config.camera.stereo_capture_height,
        prefer_v4l2=config.camera.prefer_v4l2,
        # V4L2 采集格式与曝光控制。
        #
        # 这 4 个参数不传进去，config.py 里新增的字段就是死代码——Camera 用的是
        # 显式关键字传参，不是 **vars(config.camera)，任何漏传的字段都会静默
        # 使用 Camera.__init__ 的默认值。之前 v4l2_fourcc 一直生效是因为默认值
        # 就是 "MJPG"，但曝光/增益这几个默认是 -1（不干预），必须在 config 里
        # 改动才会起作用——所以这里必须显式转发。
        v4l2_fourcc=config.camera.v4l2_fourcc,
        v4l2_auto_exposure_mode=config.camera.v4l2_auto_exposure_mode,
        v4l2_exposure_absolute=config.camera.v4l2_exposure_absolute,
        v4l2_gain=config.camera.v4l2_gain,
    )
    # Constructed only when the flag was passed. Without it the depth camera is
    # simply absent, which is the same shape as "constructed but fell back to mock
    # with no obstacle ever in range".
    #
    # frame_source=camera：stereo 后端从 camera.py 取未切割的双目原始帧，而不是
    # 再打开一次同一块 UVC 设备。webui 显示的是 camera.py 切出来的单眼图，
    # 距离计算用的是完整双目帧——同一时刻的同一帧，两边不冲突。
    depth_camera = DepthCamera(config.depth, frame_source=camera) if config.depth.enable_depth else None
    depth_guard = DepthGuard(depth_camera, config.depth) if depth_camera is not None else None
    perception = RoadPerception(config.vision)
    vehicle = VehicleController(config.control)
    sensors = SensorManager(config.sensor)
    # Always constructed: without a streamer it is inert (resolve() returns None on
    # every frame), so the main loop never needs an "if remote is not None" branch.
    remote = RemoteState(config, config.remote)
    streamer = WebStreamer(
        host=config.stream.host,
        port=config.stream.port,
        jpeg_quality=config.stream.jpeg_quality,
        fps=config.stream.fps,
        remote=remote,
    ) if config.stream.enable_stream else None
    oled = OLEDDisplay(config.oled)

    try:
        camera.start()
        if depth_camera is not None:
            depth_camera.start()
        vehicle.start()
        sensors.start()
        if streamer is not None:
            streamer.start()
        oled.start()
        logging.info("All subsystems initialized. Starting main control loop...")

        loop_period = 1.0 / config.target_loop_hz
        frame_count = 0
        fps_start_time = time.time()
        current_fps = 0.0

        max_frame_errors = config.max_consecutive_frame_errors
        consecutive_errors = 0

        while running:
            loop_start = time.time()

            try:
                # 1. Capture camera frame
                frame = camera.capture_frame()

                # 2. Vision perception: process road condition & calculate steering offset
                error, detected, debug_frame = perception.process_frame(frame)

                # 3. Motion control calculation (PID + dynamic speed scaling).
                #    Computed on every frame even when a remote mode is in charge:
                #    a stale dt would fire a derivative spike on the switch back,
                #    and the tuning page wants the numbers. Nothing is actuated
                #    from it while a remote mode owns the car.
                auto_steering, auto_throttle = vehicle.compute_control(error, detected)

                # 4. Remote override (operator command, deadman hold, or paused)
                steering, throttle, drive_source = resolve_drive(
                    remote, vehicle, auto_steering, auto_throttle
                )

                # 5. Sensor override (side proximity & obstacle avoidance). The
                #    side-wall steering bias applies only while the car steers
                #    itself -- under manual control the operator is the steering
                #    authority. The emergency brake keeps the last word either way.
                steering, throttle, is_emergency = sensors.get_obstacle_override(
                    steering, throttle, apply_steering_bias=(drive_source == SOURCE_AUTO)
                )

                # 5b. Depth guard (front proximity). Skipped once an emergency is
                #     already latched: the depth camera takes time to fetch a frame
                #     and there is nothing left to add to a full stop.
                if depth_guard is not None and not is_emergency:
                    steering, throttle, is_emergency = depth_guard.apply(steering, throttle)

                # 6. Apply actuator commands
                if is_emergency:
                    vehicle.stop()
                else:
                    vehicle.set_drive(steering, throttle)

                # 6b. Update OLED display with current stats
                oled.update(throttle)

                # 7. Publish telemetry to the web console.
                #    The coarse depth grid rides along so the page can answer "how
                #    far is the wall under the cursor" without a second endpoint or
                #    a full-depth frame on the wire. It is a few kB and the page
                #    only polls it once a second, so this costs the drive link
                #    nothing measurable -- which matters, because that link is
                #    what the deadman timeout is counting.
                telemetry = {
                    "fps": round(current_fps, 1),
                    "steering": round(steering, 3),
                    "throttle": round(throttle, 3),
                    "detected": bool(detected),
                    "source": drive_source,
                }
                if depth_guard is not None:
                    nearest = depth_guard.last_distance_mm
                    telemetry["depth_mm"] = round(nearest, 1) if nearest is not None else None
                    telemetry["depth_grid"] = depth_guard.grid
                remote.publish_telemetry(telemetry)

                # 8. Compose telemetry visual frame for GUI and/or Web Stream
                visual_output_frame = debug_frame if debug_frame is not None else frame
                if visual_output_frame is not None and cv2 is not None:
                    overlay_text = f"FPS: {current_fps:.1f} | Steer: {steering:+.2f} | Thr: {throttle:.2f}"
                    cv2.putText(visual_output_frame, overlay_text, (20, visual_output_frame.shape[0] - 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    draw_mode_banner(visual_output_frame, drive_source)

                # Push to Web Streamer (browser view on PC/phone)
                if streamer is not None and visual_output_frame is not None:
                    streamer.update_frame(visual_output_frame)

                # Local OpenCV GUI window display (if enabled)
                if args.display and visual_output_frame is not None and cv2 is not None:
                    cv2.imshow("PiRacer Folkrace Vision", visual_output_frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q') or key == 27:  # 'q' or ESC
                        logging.info("Exit requested via GUI key press.")
                        break

                # FPS calculation (only counts fully processed frames)
                frame_count += 1
                elapsed_fps = time.time() - fps_start_time
                if elapsed_fps >= 1.0:
                    current_fps = frame_count / elapsed_fps
                    frame_count = 0
                    fps_start_time = time.time()
                    logging.info(f"Running @ {current_fps:.1f} FPS | Steer: {steering:+.2f} | Throttle: {throttle:.2f}")

            except Exception as e:
                # A single dropped frame must not end the run — only sustained failure should.
                consecutive_errors += 1
                logging.error(f"Frame processing error ({consecutive_errors}/{max_frame_errors}): {e}", exc_info=True)
                if consecutive_errors >= max_frame_errors:
                    logging.error("Persistent frame errors detected. Stopping vehicle for safety.")
                    break
            else:
                consecutive_errors = 0

            # Sleep to maintain target loop frequency (also throttles the error path)
            process_duration = time.time() - loop_start
            sleep_time = loop_period - process_duration
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        logging.info("KeyboardInterrupt received.")
    except Exception as e:
        logging.error(f"Unexpected error in main loop: {e}", exc_info=True)
    finally:
        logging.info("Cleaning up and stopping vehicle...")
        vehicle.stop()
        camera.stop()
        if depth_camera is not None:
            depth_camera.stop()
        sensors.cleanup()
        if streamer is not None:
            streamer.stop()
        oled.stop()
        if cv2 is not None:
            cv2.destroyAllWindows()
        logging.info("System terminated safely.")


if __name__ == "__main__":
    main()
