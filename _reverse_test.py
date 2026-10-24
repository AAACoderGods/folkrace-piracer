"""Throwaway check for the reverse-engage pulse. Run, then delete."""

import time
import logging

logging.basicConfig(level=logging.WARNING)

from config import AppConfig, RemoteConfig
from webcontrol import RemoteState, SOURCE_MANUAL


def fresh_state():
    remote = RemoteConfig(allow_manual=True)
    state = RemoteState(AppConfig(remote=remote), remote)
    state.set_mode("arm")
    return state, remote


def main():
    # 1. Press into reverse from neutral -> pulse.
    state, remote = fresh_state()
    state.set_drive(0.0, -0.25)
    t0 = time.monotonic()
    cmd = state.resolve(t0)
    assert cmd.source == SOURCE_MANUAL
    assert cmd.throttle == remote.reverse_engage_pulse, cmd
    print("1 ok: press into reverse ->", cmd.throttle)

    # 2. Pulse holds through its window.
    cmd = state.resolve(t0 + remote.reverse_engage_s * 0.5)
    assert cmd.throttle == remote.reverse_engage_pulse, cmd
    print("2 ok: pulse held through the window")

    # 3. Window over -> operator's own value.
    cmd = state.resolve(t0 + remote.reverse_engage_s + 0.01)
    assert cmd.throttle == -0.25, cmd
    print("3 ok: falls back to operator value", cmd.throttle)

    # 4. Release -> zero, and no pulse returns.
    state.set_drive(0.0, 0.0)
    cmd = state.resolve(t0 + remote.reverse_engage_s + 0.02)
    assert cmd.throttle == 0.0, cmd
    print("4 ok: release stops the car")

    # 5. Press again -> a fresh pulse.
    state.set_drive(0.0, -0.25)
    cmd = state.resolve(time.monotonic())
    assert cmd.throttle == remote.reverse_engage_pulse, cmd
    print("5 ok: second press pulses again")

    # 6. Smooth ramp, exactly how the page sends it (small steps, 10 Hz).
    state, remote = fresh_state()
    for value in [-0.03, -0.07, -0.12, -0.18, -0.25]:
        state.set_drive(0.0, value)
        cmd = state.resolve(time.monotonic())
        assert cmd.throttle == remote.reverse_engage_pulse, (value, cmd)
    cmd = state.resolve(time.monotonic() + remote.reverse_engage_s + 0.5)
    assert cmd.throttle == -0.25, cmd
    print("6 ok: smooth ramp triggers on the first negative step, then holds -0.25")

    # 7. Forward is untouched.
    state, remote = fresh_state()
    state.set_drive(0.0, 0.4)
    cmd = state.resolve(time.monotonic())
    assert cmd.throttle == 0.4, cmd
    print("7 ok: forward unchanged", cmd.throttle)

    # 8. STOP cancels a pulse that is still running.
    state, remote = fresh_state()
    state.set_drive(0.0, -0.25)
    assert state.resolve(time.monotonic()).throttle == remote.reverse_engage_pulse
    state.set_mode("stop")
    cmd = state.resolve(time.monotonic())
    assert cmd.throttle == 0.0, cmd
    print("8 ok: stop cancels a running pulse")

    print("\nall checks passed")


if __name__ == "__main__":
    main()
