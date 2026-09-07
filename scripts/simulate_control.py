#!/usr/bin/env python3
"""Simulate the reel minigame and compare control strategies.

The reel bar is modelled as a damped double integrator — hold accelerates it
right, release accelerates it left — which is the behaviour the real minigame
exhibits. The fish is a random walk with occasional darts.

Run:  python3 scripts/simulate_control.py
"""

import pathlib
import random
import sys

import numpy as np
from dataclasses import dataclass

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from reel_controller import ControlParams, ReelController  # noqa: E402

DT = 0.02              # 20 ms tick, matching scan_interval_ms

# A controller sees the world late and imperfectly, and a simulation without
# those two effects cannot reproduce oscillation at all — a noiseless,
# zero-latency bang-bang loop tracks essentially perfectly, which is not what
# the real macro does.
#
# LATENCY_TICKS covers screen capture, OpenCV work and the input round trip.
# POSITION_NOISE is the per-frame spread of the detector's own estimate.
LATENCY_TICKS = 3          # ~60 ms
# The detector's error is not Gaussian. Measured across the recorded clips,
# the bar centre's median frame-to-frame step is under 0.005 track widths while
# the 90th percentile reaches 0.10 — a heavy tail of occasional bad edges. That
# tail is what a squared braking term turns into a lurch, so a simulation with
# only small Gaussian noise cannot show the failure at all.
POSITION_NOISE = 0.006     # track widths, 1 sigma, the bulk of the distribution
OUTLIER_RATE = 0.10        # fraction of readings that land in the tail
OUTLIER_SCALE = 0.07       # typical size of a tail error
DROPOUT_RATE = 0.20        # fraction of ticks with no usable reading
# Measured by scripts/measure_physics.py from tests/clips/physics-*.mov.
# The asymmetry matters: it sets the hold fraction that keeps the bar still.
ACCEL_HOLD = 0.90      # track widths / s^2, rightward while held
ACCEL_FREE = 1.05      # leftward while released
DAMPING = 0.8          # velocity decay
BAR_WIDTH = 0.13


@dataclass
class Bar:
    pos: float = 0.5   # centre
    vel: float = 0.0

    def step(self, hold: bool, dt: float = DT) -> None:
        self.vel += (ACCEL_HOLD if hold else -ACCEL_FREE) * dt
        self.vel -= self.vel * DAMPING * dt
        self.pos += self.vel * dt
        lo, hi = BAR_WIDTH / 2, 1.0 - BAR_WIDTH / 2
        if self.pos <= lo:
            self.pos, self.vel = lo, max(0.0, self.vel)
        elif self.pos >= hi:
            self.pos, self.vel = hi, min(0.0, self.vel)


class Fish:
    """Random walk with darts, clamped to the track."""

    def __init__(self, seed: int):
        self.rng = random.Random(seed)
        self.pos = 0.5
        self.vel = 0.0

    def step(self, dt: float = DT) -> None:
        if self.rng.random() < 0.012:
            self.vel = self.rng.uniform(-0.9, 0.9)
        self.vel += self.rng.gauss(0, 0.9) * dt
        self.vel *= 0.97
        self.pos += self.vel * dt
        if self.pos < 0.02:
            self.pos, self.vel = 0.02, abs(self.vel)
        elif self.pos > 0.98:
            self.pos, self.vel = 0.98, -abs(self.vel)


def _perceive(history, rng, tick):
    """What the controller can see this tick: a delayed, noisy sample."""
    idx = tick - LATENCY_TICKS
    if idx < 0:
        return None
    fish, bar = history[idx]
    if rng.random() < DROPOUT_RATE:
        return None

    def noisy(value):
        err = rng.gauss(0, POSITION_NOISE)
        if rng.random() < OUTLIER_RATE:
            err += rng.choice((-1.0, 1.0)) * abs(rng.gauss(OUTLIER_SCALE, OUTLIER_SCALE / 2))
        return value + err

    return (noisy(fish), noisy(bar))


def run_new(seed: int, ticks: int, **overrides):
    params = ControlParams()
    for key, value in overrides.items():
        setattr(params, key, value)

    bar, fish = Bar(), Fish(seed)
    ctrl = ReelController(params)
    rng = random.Random(seed * 31 + 7)
    on = 0
    t = 0.0
    errors = []
    positions = []
    hold = False
    history = []
    last_seen = -99

    for tick in range(ticks):
        history.append((fish.pos, bar.pos))
        seen = _perceive(history, rng, tick)
        if seen is not None:
            hold = ctrl.decide(seen[0], seen[1], now=t).hold
            last_seen = tick
        elif tick - last_seen > 6:
            hold = False        # sustained blindness: release, as the macro does

        bar.step(hold)
        fish.step()
        if abs(fish.pos - bar.pos) <= BAR_WIDTH / 2:
            on += 1
        positions.append(bar.pos)
        errors.append(abs(fish.pos - bar.pos))
        t += DT

    reversals = count_oscillations(positions)
    return on / ticks, bar.pos, reversals, float(np.mean(errors))


def count_oscillations(positions, min_amplitude: float = 0.02) -> int:
    """Direction changes in the bar's travel with a real excursion behind them.

    Counting sign changes of velocity does not work: velocity is near zero
    exactly when it flips, so any magnitude threshold applied at the flip tick
    rejects every genuine reversal. This instead tracks the running extremum and
    counts a reversal once the trace retraces from it by `min_amplitude`, which
    is what a person actually sees as the bar wobbling.
    """
    if not positions:
        return 0

    count = 0
    direction = 0            # 0 unknown, +1 rising, -1 falling
    extremum = positions[0]

    for value in positions:
        if direction > 0:
            if value > extremum:
                extremum = value
            elif extremum - value >= min_amplitude:
                count += 1
                direction = -1
                extremum = value
        elif direction < 0:
            if value < extremum:
                extremum = value
            elif value - extremum >= min_amplitude:
                count += 1
                direction = 1
                extremum = value
        else:
            if value - extremum >= min_amplitude:
                direction = 1
                extremum = value
            elif extremum - value >= min_amplitude:
                direction = -1
                extremum = value
    return count


def run_old(seed: int, ticks: int, dwell: float = 0.08):
    """Reconstruction of the current macro's rule: position-only bang-bang,
    with the dwell timer that *skips the call* when a change is blocked."""
    bar, fish = Bar(), Fish(seed)
    on = 0
    t = 0.0
    physical_hold = False
    last_action = None
    last_time = -99.0
    for _ in range(ticks):
        want = "hold" if fish.pos >= bar.pos else "release"
        blocked = (t - last_time) < dwell and want != last_action
        if not blocked:
            physical_hold = want == "hold"
            last_action = want
            last_time = t
        # When blocked, nothing is applied: the button keeps its previous
        # physical state. That is the latch.
        bar.step(physical_hold)
        fish.step()
        if abs(fish.pos - bar.pos) <= BAR_WIDTH / 2:
            on += 1
        t += DT
    return on / ticks, bar.pos


def main() -> int:
    ticks = 1500  # 30 seconds
    seeds = range(12)

    new = [run_new(s, ticks) for s in seeds]
    old = [run_old(s, ticks) for s in seeds]

    def summarise(name, rows):
        on = sum(r[0] for r in rows) / len(rows)
        pinned = sum(1 for r in rows if r[1] > 0.85)
        extra = ""
        if len(rows[0]) > 2:
            rev = sum(r[2] for r in rows) / len(rows)
            err = sum(r[3] for r in rows) / len(rows)
            extra = f"   reversals/30s {rev:5.0f}   mean |error| {err:.3f}"
        print(f"{name:28} time-on-target {on:6.1%}   "
              f"pinned {pinned}/{len(rows)}{extra}")

    print(f"{len(list(seeds))} runs x {ticks * DT:.0f}s, bar width {BAR_WIDTH}\n")
    summarise("current logic (position)", old)
    summarise("switching law (velocity)", new)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
