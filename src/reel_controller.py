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

    # The same plant, split out because the estimator below has to reproduce it
    # step for step rather than approximate it. bar_accel stays the symmetric
    # figure used for the braking distance, where only the magnitude matters.
    accel_hold: float = 0.90
    accel_free: float = 1.05
    damping: float = 0.8

    # Input and capture latency. Every reading describes the world this long
    # ago, so the bar has already moved under commands this controller issued
    # and has not yet seen the result of.
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

    # Track the bar with a model-driven estimator rather than by differentiating
    # its observed positions.
    #
    # Handing the controller a perfect bar reading is worth ~18 points of win
    # rate in the hard cells, and a perfect fish reading about the same
    # (scripts/simulate_control.py). The bar's share of that is recoverable
    # without better vision: its dynamics are known and every command it has
    # received was issued from here, so the estimator can run the plant forward
    # and use the picture only to correct drift. Differentiating noisy positions
    # throws that away and then squares the error in the braking term.
    #
    # Standard alpha-beta gains. Beta is scaled by the update interval, not by
    # the latency: the correction is applied on every tick, so dividing a
    # measurement-sized innovation by the much smaller latency turns detector
    # noise into velocity swings an order of magnitude larger than the bar's
    # real motion. Beta also has to stay well under alpha for the filter to
    # settle rather than ring.
    observer: bool = True
    observer_alpha: float = 0.5      # position correction per measurement
    observer_beta: float = 0.02      # velocity correction, per second of interval

    # The fish is deliberately NOT filtered this way, though its glides look
    # like an obvious fit for it. A constant-velocity estimator is only right
    # *within* a movement, and the fish picks a new destination every 1-2.5
    # seconds; at each of those turns the model is exactly wrong, and the filter
    # spends its settling time lagging the one event the controller most needs
    # to react to. Measured over the hard cells it cost 3 points of win rate
    # against plain short-window differentiation, which carries no model and so
    # has nothing to be wrong about. The bar is different: its model is not a
    # guess, it is the plant this class is driving.


class _BarEstimator:
    """Where the bar is now, from the plant plus a delayed, noisy picture.

    Every reading describes the bar as it was ``latency_seconds`` ago, so it
    cannot be applied to the current estimate: the two are not contemporaneous.
    Applying a stale innovation to the present state anyway is what makes a
    naive filter ring -- measured here, anything above a very small velocity
    gain destabilised it, because each correction was still arriving a frame
    late and fighting the one before.

    So the correction is applied *where it belongs*. The filter keeps a short
    history of full states and the command that drove each step; a measurement
    corrects the state at its own timestamp, and the filter then replays the
    intervening commands through the known plant to bring that correction
    forward. The bar's dynamics are known exactly and every command came from
    this class, so the replay is not an approximation.
    """

    def __init__(self, params: "ControlParams"):
        self.p = params
        self.reset()

    def reset(self) -> None:
        self.x: Optional[float] = None
        self.v = 0.0
        # (time, x, v, hold, dt) for each step, newest last.
        self._hist: Deque[Tuple[float, float, float, bool, float]] = deque(maxlen=128)

    def _advance(self, x: float, v: float, hold: bool, dt: float):
        a = self.p.accel_hold if hold else -self.p.accel_free
        v += a * dt
        v -= v * self.p.damping * dt
        return x + v * dt, v

    def predict(self, hold: bool, dt: float, now: float) -> None:
        if self.x is None:
            return
        self.x, self.v = self._advance(self.x, self.v, hold, dt)
        self._hist.append((now, self.x, self.v, hold, dt))

    def correct(self, measured: float, now: float) -> None:
        if self.x is None:
            self.x, self.v = measured, 0.0
            self._hist.clear()
            return

        target = now - max(self.p.latency_seconds, 1e-3)
        idx = None
        for k in range(len(self._hist) - 1, -1, -1):
            if self._hist[k][0] <= target:
                idx = k
                break
        if idx is None:
            return

        # Correct the state as it stood when the picture was taken.
        _, x, v, _, dt = self._hist[idx]
        innovation = measured - x
        x += self.p.observer_alpha * innovation
        v += self.p.observer_beta * innovation / max(dt, 1e-3)

        # Replay the commands issued since, so the correction arrives in the
        # present having been through the same dynamics the bar went through.
        rebuilt = []
        for k in range(idx + 1, len(self._hist)):
            t_k, _, _, hold_k, dt_k = self._hist[k]
            x, v = self._advance(x, v, hold_k, dt_k)
            rebuilt.append((t_k, x, v, hold_k, dt_k))

        v = max(-self.p.velocity_max, min(self.p.velocity_max, v))
        self.x, self.v = x, v
        for k, entry in enumerate(rebuilt, start=idx + 1):
            self._hist[k] = entry


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
        self._estimator = _BarEstimator(self.p)
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

        dt_pred = 0.02 if self._last_tick is None else max(1e-3, min(0.5, now - self._last_tick))
        self._estimator.predict(self._holding, dt_pred, now)
        self._estimator.correct(bar_center, now)
        fish_v = self._fish.velocity()
        bar_v = self._bar.velocity()

        # Project the fish forward, faded in with speed rather than switched
        # on at a threshold. Switching it meant the aim point jumped between
        # fish_x and fish_x + lead whenever the measured speed crossed
        # stationary_speed — measured in the closed loop, 223 times over 7200
        # ticks, by as much as 0.0128 of the track in a single tick. The error
        # this feeds is multiplied by duty_kp, which is 10, so that is a swing
        # of more than a tenth of the duty cycle caused by nothing at all.
        #
        # The taper is exactly 1 at stationary_speed, so anything moving faster
        # than that is projected precisely as it was before.
        lead = fish_v * self.p.lead_seconds
        lead *= min(1.0, abs(fish_v) / max(self.p.stationary_speed, 1e-6))
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
        if self.p.observer:
            bar_now, bar_v_now = self._estimator.x, self._estimator.v
        else:
            bar_now = bar_center + bar_v * self.p.latency_seconds
            bar_v_now = bar_v

        braking = bar_v_now * abs(bar_v_now) / (2.0 * max(self.p.bar_accel, 1e-3))
        braking = max(-self.p.max_brake_distance, min(self.p.max_brake_distance, braking))
        bar_projected = bar_now + braking

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
