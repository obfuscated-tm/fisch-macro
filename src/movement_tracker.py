"""
Velocity tracking for minigame position prediction.

Inspired by digmacro's movement tracker + kinematic lookahead — maintains a
short history of positions to estimate velocity and acceleration without
overreacting to single-frame noise.
"""

from __future__ import annotations

import time
from collections import deque
from typing import Deque, Optional, Tuple


class MovementTracker:
    """Track normalized position (0–1) and estimate velocity / acceleration."""

    # Movement smaller than this over the sampled window is taken to be
    # detector noise rather than the fish: a position is read off a track a few
    # hundred pixels wide, so a single pixel of jitter is worth ~0.002.
    NOISE_SPAN = 0.008

    def __init__(self, history_size: int = 12):
        self._history: Deque[Tuple[float, float]] = deque(maxlen=history_size)

    def reset(self) -> None:
        self._history.clear()

    def add_sample(self, position: float) -> None:
        self._history.append((time.time(), position))

    @property
    def sample_count(self) -> int:
        return len(self._history)

    def velocity(self) -> float:
        """Units: normalized position per second (full history window)."""
        if len(self._history) < 2:
            return 0.0
        t0, x0 = self._history[0]
        t1, x1 = self._history[-1]
        dt = t1 - t0
        if dt <= 1e-6:
            return 0.0
        return (x1 - x0) / dt

    def recent_velocity(self, window_seconds: float = 0.2) -> float:
        """Velocity over the latest samples only — ignores stale movement when fish stops.

        Movement below the noise span is faded out rather than cut off. It used
        to return exactly zero as soon as the window spanned less than 0.008,
        which put a cliff under everything built on top: the prediction blend is
        scaled by this speed, so a fish crossing that line switched the whole
        lookahead off and on between one frame and the next. Replaying the fish
        of tests/clips/hallucinate.mov through it, the prediction turned fully
        off and fully on 22 times in a single fight while the fish was moving
        steadily, and each switch moved the aim point by up to 0.02 of the track
        for no physical reason. That is the chatter; the taper is the fix.
        """
        if len(self._history) < 2:
            return 0.0

        now = self._history[-1][0]
        cutoff = now - max(window_seconds, 0.05)
        points = [(t, x) for t, x in self._history if t >= cutoff]
        if len(points) < 2:
            return 0.0

        t0, x0 = points[0]
        t1, x1 = points[-1]
        dt = t1 - t0
        if dt <= 1e-6:
            return 0.0

        # Two spans are measured: the window this velocity is taken over, and
        # the handful of most recent samples, so that a fish which has just
        # stopped is not still credited with the movement that got it here.
        spans = [max(x for _, x in points) - min(x for _, x in points)]
        tail = [x for _, x in list(self._history)[-5:]]
        if len(tail) >= 3:
            spans.append(max(tail) - min(tail))
        return (x1 - x0) / dt * self._confidence(spans)

    def _confidence(self, spans) -> float:
        """How much of the measured movement to believe, in [0, 1].

        Faded rather than switched: anything that scales an estimate by this
        has to stay continuous as the fish slows, or the thing it feeds jumps
        between two values one frame apart.
        """
        return min(1.0, min(spans) / self.NOISE_SPAN)

    def _recent_spans(self):
        """Spans of the movement window and of the last few samples."""
        history = list(self._history)
        spans = [max(x for _, x in history) - min(x for _, x in history)]
        tail = [x for _, x in history[-5:]]
        if len(tail) >= 3:
            spans.append(max(tail) - min(tail))
        return spans

    def speed(self, window_seconds: float = 0.2) -> float:
        """Absolute recent speed (normalized position per second)."""
        return abs(self.recent_velocity(window_seconds))

    def acceleration(self) -> float:
        """Smoothed rate of velocity change (normalized / s²).

        A least-squares slope of the per-interval velocities against time,
        rather than the difference between the first and last of them. That
        difference was neither smoothed, despite the docstring, nor an
        acceleration: it never divided by the time the change took, so it
        carried the units of a velocity and its size depended on how fast the
        loop happened to be ticking. It also read whatever the two noisiest
        estimates in the window disagreed about — replayed over a stationary
        fish it returned ±0.17 while the true answer was zero.
        """
        history = list(self._history)
        if len(history) < 3:
            return 0.0

        times, velocities = [], []
        prev = history[0]
        for current in history[1:]:
            dt = current[0] - prev[0]
            if dt > 1e-6:
                velocities.append((current[1] - prev[1]) / dt)
                times.append((current[0] + prev[0]) / 2.0)
            prev = current
        if len(velocities) < 2:
            return 0.0

        span = times[-1] - times[0]
        if span <= 1e-6:
            return 0.0

        mean_t = sum(times) / len(times)
        mean_v = sum(velocities) / len(velocities)
        denominator = sum((t - mean_t) ** 2 for t in times)
        if denominator <= 1e-9:
            return 0.0
        slope = sum(
            (t - mean_t) * (v - mean_v) for t, v in zip(times, velocities)
        ) / denominator

        # A fish that has not really moved has no acceleration to report. The
        # slope of a jittering signal is not small — one pixel of noise over a
        # 20ms tick is 0.08 of velocity — so it has to be faded out by the same
        # measure the velocity uses, or the lookahead is driven by quantisation.
        return slope * self._confidence(self._recent_spans())

    def predict(
        self,
        current: float,
        lookahead_seconds: float,
        use_acceleration: bool = True,
        window_seconds: float = 0.2,
    ) -> float:
        """Kinematic lookahead: x + v*t + 0.5*a*t² using recent velocity only."""
        v = self.recent_velocity(window_seconds)
        if abs(v) < 1e-4:
            return current
        a = self.acceleration() if use_acceleration else 0.0
        t = max(0.0, lookahead_seconds)
        predicted = current + (v * t) + (0.5 * a * t * t)
        return max(0.0, min(1.0, predicted))

    def arrival_seconds(self, target: float, current: float) -> Optional[float]:
        """Seconds until *current* reaches *target* at current velocity, or None."""
        v = self.recent_velocity()
        if abs(v) < 1e-4:
            return None
        if (v > 0 and target < current) or (v < 0 and target > current):
            return None
        return (target - current) / v
