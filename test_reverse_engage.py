#!/usr/bin/env python3
"""
倒车接线复核 —— 把「B 段第一次 -1.0 能倒转」拆成四个能一次判定的问题。

已知的两个端点：

    前进 1s → 中位 2s → 直接跳到 -1.0    → 倒转 ✔  （test_reverse.py B 段第一次）
    中位 → 缓降 / 直跳到 -0.25           → 不动 ✘  （静止按 S、test_hardware.py）

中间机制还不知道，而它决定怎么修，所以下面五段各只回答一个问题：

    T1  基线：B 段能不能重复出来？

    T2  A 段重现：递增加深、中途不回中位，到底能不能倒？
            能   → 纯幅度阈值，改 config.py 一个数就完事
            不能 → ESC 在负行程"开始那一刻"就定调了，之后加深无效，必须打脉冲

    T3  强脉冲 → 回中位 → 再用 -0.25 保持：还倒吗？
            倒   → 脉冲式修法可行，max_manual_reverse 保持 -0.25
            不倒 → 操作值本身就必须够大，阈值由 T5 给出

    T4  没有前进段，静止直接拉 -1.0：行不行？
            不行 → 倒车前必须先给一脚前进，修法会复杂不少

    T5  直接跳的幅度阈值：-0.5 / -0.7 / -0.9 哪一档开始倒？

用法（车轮必须悬空！）：
    python3 test_reverse_engage.py
    # 或 chmod +x 之后：./test_reverse_engage.py

全程约 45 秒。每段开始前会打印「看什么」，盯着轮子记结果即可。
"""

import time
import logging

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s")

SWEEP = [-0.2, -0.4, -0.6, -0.8, -1.0]   # T2：递增加深，中途不回中位
THRESHOLD = [-0.5, -0.7, -0.9]           # T5：每一档都是从中位直接跳


def load_driver():
    """返回 (driver, driver_type)，回退顺序和 test_hardware.py / test_reverse.py 一致。"""
    try:
        from piracer.vehicles import PiRacerPro
        logging.info("✅ 已加载 piracer-py (PiRacerPro)")
        return PiRacerPro(), "piracer-py"
    except Exception as e:
        logging.warning(f"piracer-py 不可用：{e}")

    try:
        from adafruit_servokit import ServoKit
        driver = ServoKit(channels=16)
        driver.servo[0].set_pulse_width_range(1000, 2000)
        driver.continuous_servo[1].set_pulse_width_range(1000, 2000)
        logging.info("✅ 已加载 adafruit-servokit (PCA9685)")
        return driver, "servokit"
    except Exception as e:
        logging.warning(f"ServoKit 不可用：{e}")

    return None, None


def main():
    logging.info("=" * 58)
    logging.info(" 倒车接线复核 —— 五段，每段一个问题 ")
    logging.info("=" * 58)

    driver, driver_type = load_driver()
    if driver is None:
        logging.error("❌ 没有可用的硬件驱动！请检查 I2C 和 piracer-py 是否装好。")
        return

    def throttle(value):
        if driver_type == "piracer-py":
            driver.set_throttle_percent(value)
        else:
            driver.continuous_servo[1].throttle = value

    def hold(value, seconds):
        logging.info(f"  -> 油门 {value:+.2f}  持续 {seconds:.2f}s")
        throttle(value)
        time.sleep(seconds)

    def watch(what):
        logging.info("")
        logging.info(f"【看什么】{what}")

    def settle():
        hold(0.0, 1.2)   # 段与段之间回中位，让 ESC 状态归零
        time.sleep(0.4)

    logging.info("注意：车轮必须悬空！3 秒后开始运行。")
    time.sleep(3.0)

    try:
        logging.info("")
        logging.info("中位解锁 2 秒（注意听 ESC 的提示音）……")
        hold(0.0, 2.0)

        # ---- T1：基线，复现 B ----
        watch("轮子应该倒转。不倒就说明前面的结论不可重复，后面几段先别急着信。")
        logging.info("--- T1) 基线：前进 1s → 中位 2s → 直接跳 -1.0 ---")
        hold(0.3, 1.0)
        hold(0.0, 2.0)
        hold(-1.0, 1.5)
        settle()

        # ---- T2：复现 A ----
        watch("轮子会不会倒转？以及从哪一档开始？全程不回中位，一路加深到 -1.0。")
        logging.info("--- T2) 递增不回中位：-0.2 → -0.4 → -0.6 → -0.8 → -1.0 ---")
        hold(0.3, 1.0)
        hold(0.0, 2.0)
        for value in SWEEP:
            hold(value, 0.8)
        hold(-1.0, 1.5)     # 最后一档多停一会儿，别和中位停留混淆
        settle()

        # ---- T3：脉冲后回落 ----
        watch("打一记 -1.0 短脉冲、回中位、再切到 -0.25 —— 轮子还倒吗？"
              "这直接决定修法能不能用脉冲，以及 max_manual_reverse 要不要改。")
        logging.info("--- T3) 强脉冲 → 回中位 → 低值保持 ---")
        hold(0.3, 1.0)
        hold(0.0, 2.0)
        hold(-1.0, 0.35)    # 脉冲
        hold(0.0, 0.25)     # 松一下
        hold(-0.25, 2.0)    # 回落后的操作值
        settle()

        # ---- T4：无前进段 ----
        # 注意：T3 刚让 ESC 进过倒车，有些 ESC 的倒车状态会一直保持到
        # 下一次前进为止。所以 T4 即使倒了也偏乐观 —— 要下结论就重启一次
        # 脚本，单独看 T4 在「干净状态」下的表现。
        watch("这次前面没有前进段，直接从静止拉 -1.0。倒不倒也决定了"
              "「车停着按 S 能不能退」这个最常用的场景。"
              "（注意：T3 刚进过倒车，这里倒了要重启脚本再看一次才算数）")
        logging.info("--- T4) 无前进段：中位 2s → 直接拉 -1.0 ---")
        hold(0.0, 2.0)
        hold(-1.0, 1.5)
        settle()

        # ---- T5：幅度阈值 ----
        watch("三档都是从中位直接跳、每档之间回中位隔开。哪一档开始倒，"
              "那个值就是脉冲该用的强度。")
        logging.info("--- T5) 直接跳的幅度阈值：-0.5 / -0.7 / -0.9 ---")
        hold(0.3, 1.0)
        hold(0.0, 1.5)
        for value in THRESHOLD:
            hold(value, 1.2)
            hold(0.0, 1.5)

    finally:
        # 退出时绝不能让 ESC 保持某个值，Ctrl-C 也算。
        throttle(0.0)
        logging.info("油门已回中位。")

    logging.info("")
    logging.info("=" * 58)
    logging.info(" 怎么读结果 ")
    logging.info("=" * 58)
    logging.info(" T1 不倒        → 先别信后面的数据，回头查接线/电池/解锁")
    logging.info(" T2 倒了        → 纯幅度阈值：把 config.py 的 max_manual_reverse")
    logging.info("                  调到 T2 里开始倒的那一档（或 T5 的值）即可")
    logging.info(" T2 没倒        → 起点定调：进入倒车时必须先打一记强脉冲")
    logging.info(" T3 还倒        → 脉冲式修法可行，max_manual_reverse 保持 -0.25")
    logging.info(" T3 不倒了      → 操作值本身必须够强，用 T5 的阈值当下限")
    logging.info(" T4 倒了        → 静止起步可以直接倒，不需要前进预热")
    logging.info(" T4 没倒        → 倒车前必须先给一小段前进，修法要加这一步")
    logging.info("=" * 58)


if __name__ == "__main__":
    main()
