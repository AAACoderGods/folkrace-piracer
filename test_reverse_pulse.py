#!/usr/bin/env python3
"""
倒车阶段 0 —— 把「油门百分比」换算成「PWM 脉宽」，并顺带量电池电压。

为什么需要这一步
----------------
ESC 的判据是 PCA9685 输出的 50 Hz PWM 脉宽（µs），不是 throttle 百分比。
百分比 → 脉宽的换算取决于驱动后端：

    ServoKit 回退路径   显式设了 set_pulse_width_range(1000, 2000)
    piracer-py 主力路径  set_throttle_percent 内部映射不透明

如果两者不一致，之前台架上「−1.0 能入档」的结论就无法直接搬进生产代码。
所以这个脚本绕过 piracer-py，直接经 PCA9685 CH1 输出精确脉宽。

这个脚本做四件事
----------------
1. 直接经 PCA9685 CH1 输出精确脉宽（1500 µs = 中位）；
2. 分两组扫出倒车门槛：
     A 组：每档前带「前进 1650 µs → 中位」前奏（复现台架上成功过的条件）
     B 组：不带前进前奏，纯中位 → 脉冲（回答 T4 那个没测准的问题）
3. 每档读 INA219（0x42）的电池电压/电流，排除「电压低 → ESC 锁倒车」；
4. 打印 piracer-py 的 set_throttle_percent 源码，看它的真实映射。

已知边界（来自 test_reverse_engage.py 的 T1–T5）：
    1375 µs（throttle −0.25）不动
    1250 µs（throttle −0.50）能入档
所以先在边界附近用 25 µs 步长，再往深处用 50 µs。

用法（车轮必须悬空！）：
    python3 test_reverse_pulse.py

如果 INA219 读不到，先装：
    pip3 install adafruit-circuitpython-ina219

全程约 2 分钟。每一档会问一次「轮子倒转了吗」，结束后打印汇总表。
"""

import sys
import time
import logging

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s")

NEUTRAL_US = 1500
FULL_FORWARD_US = 2000
FULL_REVERSE_US = 1000
THROTTLE_SPAN_US = 500.0        # throttle ±1.0 对应 ±500 µs

SWEEP_US = [
    1400, 1375, 1350, 1325, 1300, 1275, 1250,
    1200, 1150, 1100, 1050, 1000,
]

# 前奏用的油门和时长（贴近台架上成功过的条件）。
PRELUDE_US = 1650
PRELUDE_S = 1.0
PRELUDE_NEUTRAL_S = 2.0
PULSE_HOLD_S = 1.2
SETTLE_S = 1.0


def us_to_throttle(us):
    """把脉宽换算成 ServoKit 的 throttle；±500 µs 对应 ±1.0。"""
    return (us - NEUTRAL_US) / THROTTLE_SPAN_US


def load_servokit():
    try:
        from adafruit_servokit import ServoKit
    except ImportError as e:
        logging.error(f"❌ adafruit-circuitpython-servokit 不可用：{e}")
        logging.error("   在树莓派上：pip3 install adafruit-circuitpython-servokit")
        return None

    try:
        kit = ServoKit(channels=16)
        # CH1 = 油门 ESC。这里显式设定脉宽范围，保证 us → throttle 的换算是确定的。
        kit.continuous_servo[1].set_pulse_width_range(FULL_REVERSE_US, FULL_FORWARD_US)
        logging.info("✅ PCA9685 已就绪（CH1 = 油门 ESC，50 Hz，脉冲范围 1000–2000 µs）")
        return kit
    except Exception as e:
        logging.error(f"❌ 无法初始化 ServoKit / PCA9685：{e}")
        logging.error("   检查 I2C 是否启用（i2cdetect -y 1 应能看到 0x40）。")
        return None


def load_ina219():
    try:
        import board
        from adafruit_ina219 import INA219
        ina = INA219(board.I2C())
        logging.info("✅ INA219 已就绪（0x42），将记录电池电压 / 电流")
        return ina
    except Exception as e:
        logging.warning(f"⚠️  INA219 不可用（{e}）—— 跳过电压记录。")
        logging.warning("    需要的话：pip3 install adafruit-circuitpython-ina219")
        return None


def read_power(ina):
    if ina is None:
        return None
    try:
        return ina.bus_voltage, ina.current
    except Exception:
        return None


def dump_piracer_mapping():
    """把 piracer-py 的真实映射打出来，供和 ServoKit 的 1000–2000 对照。"""
    try:
        import inspect
        from piracer.vehicles import PiRacerPro
    except Exception as e:
        logging.info(f"piracer-py 不可用，跳过源码检查：{e}")
        return

    for name in ("__init__", "set_throttle_percent"):
        fn = getattr(PiRacerPro, name, None)
        if fn is None:
            continue
        try:
            src = inspect.getsource(fn)
        except Exception as e:
            logging.info(f"无法读取 PiRacerPro.{name} 源码：{e}")
            continue
        logging.info("=" * 58)
        logging.info(f" piracer-py PiRacerPro.{name} 源码")
        logging.info("=" * 58)
        for line in src.splitlines():
            logging.info("  " + line)
        logging.info("")


def main():
    logging.info("=" * 58)
    logging.info(" 倒车阶段 0 —— 脉宽扫描 + 电池电压 ")
    logging.info("=" * 58)

    dump_piracer_mapping()

    kit = load_servokit()
    if kit is None:
        return
    ina = load_ina219()
    servo = kit.continuous_servo[1]

    def set_us(us):
        servo.throttle = us_to_throttle(us)

    def hold(us, seconds, label=""):
        p = read_power(ina)
        power = f"   电池 {p[0]:.2f} V / {p[1]:.1f} mA" if p else ""
        logging.info(f"  -> {label}{us} µs  (throttle {us_to_throttle(us):+.2f}){power}")
        set_us(us)
        time.sleep(seconds)

    results = []

    def ask(us, phase):
        """交互记录结果；非 TTY（如重定向）时安静跳过。"""
        if not sys.stdin.isatty():
            return
        try:
            ans = input(f"     [{phase}] {us} µs 轮子倒转了吗？[y/N/q] ").strip().lower()
        except EOFError:
            return
        if ans == "q":
            raise KeyboardInterrupt
        results.append((phase, us, ans.startswith("y")))

    try:
        logging.info("注意：车轮必须悬空！3 秒后开始。")
        time.sleep(3.0)

        logging.info("")
        logging.info("中位解锁 2 秒（注意听 ESC 的提示音）……")
        hold(NEUTRAL_US, 2.0)

        # ---- A 组：带前进前奏 ----
        logging.info("")
        logging.info("=" * 58)
        logging.info(" A 组：每档前有「前进 → 中位」前奏（复现台架成功条件）")
        logging.info("=" * 58)
        for us in SWEEP_US:
            logging.info("")
            logging.info(f"--- A: {us} µs (throttle {us_to_throttle(us):+.2f}) ---")
            hold(PRELUDE_US, PRELUDE_S, "前进 ")
            hold(NEUTRAL_US, PRELUDE_NEUTRAL_S, "中位 ")
            hold(us, PULSE_HOLD_S, "脉冲 ")
            hold(NEUTRAL_US, SETTLE_S, "中位 ")
            ask(us, "A")

        # ---- B 组：无前进前奏 ----
        logging.info("")
        logging.info("=" * 58)
        logging.info(" B 组：不带前进前奏，纯中位 → 脉冲（回答 T4）")
        logging.info("=" * 58)
        for us in SWEEP_US:
            logging.info("")
            logging.info(f"--- B: {us} µs (throttle {us_to_throttle(us):+.2f}) ---")
            hold(NEUTRAL_US, PRELUDE_NEUTRAL_S, "中位 ")
            hold(us, PULSE_HOLD_S, "脉冲 ")
            hold(NEUTRAL_US, SETTLE_S, "中位 ")
            ask(us, "B")

    finally:
        set_us(NEUTRAL_US)
        logging.info("")
        logging.info("油门已回中位（1500 µs）。")

    logging.info("")
    logging.info("=" * 58)
    logging.info(" 汇总 ")
    logging.info("=" * 58)
    if not results:
        logging.info(" 没有交互记录（非 TTY 运行）。请对照上面的 µs 自己记下门槛。")
    else:
        for phase in ("A", "B"):
            rows = [(us, ok) for (p, us, ok) in results if p == phase]
            if not rows:
                continue
            # 扫描顺序是 µs 从大到小，所以第一个成功项就是最靠近中位的门槛。
            first = next((us for us, ok in rows if ok), None)
            if first is None:
                logging.info(f" {phase} 组：全无反应")
            else:
                logging.info(f" {phase} 组：从 {first} µs (throttle {us_to_throttle(first):+.2f}) 开始倒转")

    logging.info("")
    logging.info(" 怎么读：")
    logging.info("   A 有、B 无 → 倒车前必须有前进前奏，修法要加这一步")
    logging.info("   A 无、B 有 → 前奏反而有害（不太可能，但照数据办）")
    logging.info("   A、B 相同 → 前奏无关，只需要幅度够大")
    logging.info("   两组的门槛值就是生产代码里入档脉冲的下限；建议再加 10–20% 余量。")
    logging.info("=" * 58)


if __name__ == "__main__":
    main()
