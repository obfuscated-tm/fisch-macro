"""
reel_controller.py — Switching law for the Fisch reel bar.

The reel bar is a *double integrator*: holding the mouse accelerates it right,
releasing accelerates it left. There is no input that holds it still — staying
put is something you achieve by alternating, not by asking for it.

Two consequences drive this design:

1.  **Switching on position alone cannot work.** If you hold whenever the fish
    is to the right, you arrive at the fish travelling at maximum speed and sail
    straight past it, then repeat in the other direction. The decision has to be
    made on where the bar *will be* once it has shed its current velocity, not
    on where it is now.

2.  **The command must be reasserted every tick.** Holding is a latching
    physical state. Any logic that can decline to act — a dwell timer, a
    cooldown, a deadband that returns early — leaves the button *down* and the
    bar accelerating right. That asymmetry is a one-way ratchet toward the right
    wall. So `decide()` returns a definite hold/release every call and the
    caller unconditionally applies it.

Because the only input is binary, proportional control has to be expressed as a
*duty cycle*: the fraction of ticks spent holding. Pure bang-bang — hold while
the fish is right, release while it is left — can only ever command full
acceleration one way or the other, so around the target it necessarily
overshoots, reverses, and overshoots again. That is the oscillation.

The duty cycle is realised with a first-order sigma-delta modulator rather than
a fixed-period PWM block. A block period would add its own oscillation at the
block frequency; the modulator instead spreads the holds across ticks so the
running average tracks the requested duty at tick resolution.

The neutral duty — the fraction that holds the bar still — follows from the
measured asymmetry between the two accelerations, and the integral term trims
whatever that estimate gets wrong for a particular rod.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from typing import Deque, Optional, Tuple


@dataclass
class ControlParams:
    """All units are normalised track widths, and seconds."""

    # Measured from tests/clips/physics-*.mov via scripts/measure_physics.py:
    # holding accelerated the bar right at 0.90 track widths/s^2 and releasing
    # drove it left at 1.05, i.e. very nearly symmetric. That symmetry is what
    # makes the plain double-integrator model appropriate.
    bar_accel: float = 1.0

    # Input and capture latency, applied as a linear lead on the bar. Separate
    # from braking because it is a property of the macro, not of the game.
    latency_seconds: float = 0.06

    # How far ahead to project the fish's motion. Small — the fish changes
    # direction unpredictably, so long lookaheads amplify noise into overshoot.
    lead_seconds: float = 0.07
    max_lead: float = 0.12

    # Duty-cycle control. neutral_duty is the hold fraction that cancels
    # gravity: with the measured accelerations (0.90 right, 1.05 left) a bar
    # held for a fraction d accelerates by d*0.90 - (1-d)*1.05, which is zero at
    # d = 1.05 / (0.90 + 1.05) = 0.538.
    neutral_duty: float = 0.538
    duty_kp: float = 10.0       # duty per track width of error
    duty_ki: float = 0.6        # per second; trims a wrong neutral estimate
    integral_clamp: float = 0.22    # hard bound on the integral's duty contribution
    integral_leak: float = 0.4      # per second decay, so it cannot wind up

    # Velocity estimation.
    velocity_window: int = 5
    velocity_max: float = 4.0        # sanity clamp, track widths per second

    # Hard bound on the braking projection. Because braking goes as v^2, a
    # single mis-detected bar edge would otherwise project the bar several track
    # widths away and slam the duty cycle to a limit — the bar lurches, the next
    # frame corrects it, and the result is a visible oscillation.
    max_brake_distance: float = 0.35

    # Treat the fish as stationary below this speed, so noise in a parked fish
    # does not get projected into a phantom lead.
    stationary_speed: float = 0.05


class _Differentiator:
    """Velocity from timestamped samples, robust to loop jitter and dropouts."""

    def __init__(self, window: int, clamp: float):
        self._samples: Deque[Tuple[float, float]] = deque(maxlen=window)
        self._clamp = clamp

    def reset(self) -> None:
        self._samples.clear()

    def add(self, value: float, now: Optional[float] = None) -> None:
        self._samples.append((time.time() if now is None else now, value))

    @property
    def value(self) -> Optional[float]:
        return self._samples[-1][1] if self._samples else None

    def velocity(self) -> float:
        """Track widths per second — median of pairwise slopes (Theil-Sen).

        Differencing the first and last sample, as this did originally, hands
        the entire estimate to two points. The detector's error distribution has
        a heavy tail — measured on recorded clips, the bar centre's median
        frame-to-frame step is under 0.005 track widths but the 90th percentile
        reaches 0.10 — so one bad endpoint produces a wildly wrong velocity.
        That matters more than usual here because the braking term is quadratic
        in velocity, which squares the error.

        Taking the median slope over all sample pairs ignores a minority of bad
        points entirely, at negligible cost for a window this small.
        """
        if len(self._samples) < 2:
            return 0.0

        slopes = []
        samples = list(self._samples)
        for i in range(len(samples)):
            ti, xi = samples[i]
            for j in range(i + 1, len(samples)):
                tj, xj = samples[j]
                dt = tj - ti
                if dt > 1e-4:
                    slopes.append((xj - xi) / dt)
        if not slopes:
            return 0.0

        slopes.sort()
        mid = len(slopes) // 2
        v = (
            slopes[mid]
            if len(slopes) % 2
            else 0.5 * (slopes[mid - 1] + slopes[mid])
        )
        return max(-self._clamp, min(self._clamp, v))


@dataclass
class ControlDecision:
    hold: bool
    fish_projected: float
    bar_projected: float
    error: float
    fish_velocity: float
    bar_velocity: float
    reason: str
    duty: float = 0.0
    integral: float = 0.0


class ReelController:
    """Decides hold-vs-release for one reeling session."""

    def __init__(self, params: Optional[ControlParams] = None):
        self.p = params or ControlParams()
        self.reset()

    def reset(self) -> None:
        self._fish = _Differentiator(self.p.velocity_window, self.p.velocity_max)
        self._bar = _Differentiator(self.p.velocity_window, self.p.velocity_max)
        self._holding = False
        self._accumulator = 0.0
        self._integral = 0.0
        self._last_tick: Optional[float] = None

    @property
    def holding(self) -> bool:
        return self._holding

    def decide(
        self,
        fish_x: float,
        bar_center: float,
        now: Optional[float] = None,
    ) -> ControlDecision:
        """Return the command for this tick. Always returns a definite state."""
        now = time.time() if now is None else now
        self._fish.add(fish_x, now)
        self._bar.add(bar_center, now)

        fish_v = self._fish.velocity()
        bar_v = self._bar.velocity()

        # Project the fish forward, but only if it is genuinely moving.
        if abs(fish_v) < self.p.stationary_speed:
            fish_projected = fish_x
        else:
            lead = fish_v * self.p.lead_seconds
            lead = max(-self.p.max_lead, min(self.p.max_lead, lead))
            fish_projected = fish_x + lead

        # Project the bar forward by the distance it needs to stop. For a
        # double integrator that distance goes as v^2, not v, so the linear
        # approximation used before under-braked at speed and over-braked when
        # crawling. Signed by v|v| so the term always points the way the bar is
        # already travelling.
        #
        # Switching on this quantity is the time-optimal law for this system:
        # hold while the fish is beyond where the bar could still stop.
        braking = bar_v * abs(bar_v) / (2.0 * max(self.p.bar_accel, 1e-3))
        braking = max(-self.p.max_brake_distance, min(self.p.max_brake_distance, braking))
        bar_projected = bar_center + bar_v * self.p.latency_seconds + braking

        error = fish_projected - bar_projected

        dt = 0.02 if self._last_tick is None else max(1e-3, min(0.5, now - self._last_tick))
        self._last_tick = now

        # Integral term. It exists to trim a neutral_duty that is slightly wrong
        # for this rod, so it leaks continuously and is hard-clamped: an
        # unbounded integrator on a saturating actuator winds up during the long
        # full-throttle chases and then overshoots badly on arrival.
        self._integral += error * dt
        self._integral -= self._integral * self.p.integral_leak * dt
        limit = self.p.integral_clamp / max(self.p.duty_ki, 1e-6)
        self._integral = max(-limit, min(limit, self._integral))

        duty = (
            self.p.neutral_duty
            + self.p.duty_kp * error
            + self.p.duty_ki * self._integral
        )
        duty = max(0.0, min(1.0, duty))

        # Sigma-delta: accumulate the requested duty and hold on overflow. Over
        # any window the hold fraction equals `duty`, but the holds are spread
        # across ticks instead of arriving in one block, so the bar hovers
        # rather than swinging at the block frequency.
        self._accumulator += duty
        if self._accumulator >= 1.0:
            hold = True
            self._accumulator -= 1.0
        else:
            hold = False

        self._holding = hold
        return ControlDecision(
            hold=hold,
            fish_projected=fish_projected,
            bar_projected=bar_projected,
            error=error,
            fish_velocity=fish_v,
            bar_velocity=bar_v,
            reason="hold" if hold else "release",
            duty=duty,
            integral=self._integral,
        )
