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
        """Velocity over the latest samples only — ignores stale movement when fish stops."""
        if len(self._history) < 2:
            return 0.0
        tail = list(self._history)[-5:]
        if len(tail) >= 3:
            tail_xs = [x for _, x in tail]
            if max(tail_xs) - min(tail_xs) < 0.008:
                return 0.0

        now = self._history[-1][0]
        cutoff = now - max(window_seconds, 0.05)
        points = [(t, x) for t, x in self._history if t >= cutoff]
        if len(points) < 2:
            return 0.0
        xs = [x for _, x in points]
        if max(xs) - min(xs) < 0.008:
            return 0.0
        t0, x0 = points[0]
        t1, x1 = points[-1]
        dt = t1 - t0
        if dt <= 1e-6:
            return 0.0
        return (x1 - x0) / dt

    def speed(self, window_seconds: float = 0.2) -> float:
        """Absolute recent speed (normalized position per second)."""
        return abs(self.recent_velocity(window_seconds))

    def acceleration(self) -> float:
        """Smoothed rate of velocity change (normalized / s²)."""
        if len(self._history) < 3:
            return 0.0
        velocities = []
        prev = self._history[0]
        for current in list(self._history)[1:]:
            dt = current[0] - prev[0]
            if dt > 1e-6:
                velocities.append((current[1] - prev[1]) / dt)
            prev = current
        if len(velocities) < 2:
            return 0.0
        return velocities[-1] - velocities[0]

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
