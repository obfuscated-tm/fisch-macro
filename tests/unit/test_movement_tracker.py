"""What the tracker reports has to be physics, not the shape of the noise.

Two defects fed the prediction chatter that was reported as the aim point
"oscillating". Both are about continuity and units rather than tuning.

1. ``recent_velocity`` returned exactly zero as soon as the sampled window
   spanned less than 0.008, so a fish drifting across that line switched the
   entire lookahead off and on between one frame and the next.
2. ``acceleration`` returned the difference between the first and last
   per-interval velocities. That is not an acceleration — it never divided by
   the time the change took — and it is not smoothed either, despite saying so:
   it read whatever the two noisiest estimates in the window disagreed about.
"""

import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src import movement_tracker as mt  # noqa: E402

TICK = 0.02
QUANT = 1 / 613.0   # one pixel of a track that wide


@pytest.fixture
def clock(monkeypatch):
    """A fake clock, so a trajectory can be fed at an exact tick rate."""
    state = {"t": 0.0}

    class _Time:
        @staticmethod
        def time():
            return state["t"]

    monkeypatch.setattr(mt, "time", _Time)
    return state


def feed(clock, positions):
    tracker = mt.MovementTracker()
    for x in positions:
        clock["t"] += TICK
        tracker.add_sample(x)
    return tracker


def test_acceleration_recovers_a_known_acceleration(clock):
    true_a = 2.0
    tracker = feed(clock, [
        0.3 + 0.5 * (i * TICK) + 0.5 * true_a * (i * TICK) ** 2 for i in range(12)
    ])

    assert tracker.acceleration() == pytest.approx(true_a, rel=0.05)


def test_a_fish_at_constant_speed_is_not_accelerating(clock):
    tracker = feed(clock, [0.3 + 0.5 * i * TICK for i in range(12)])

    assert tracker.acceleration() == pytest.approx(0.0, abs=0.1)


def test_a_still_fish_is_not_accelerating(clock):
    """One pixel of detector jitter is 0.08 of velocity at a 20ms tick.

    Differenced, that is a large number to hand to a lookahead, and it is
    entirely noise.
    """
    jitter = [0.5, 0.5 + QUANT, 0.5, 0.5 + QUANT] * 3
    tracker = feed(clock, jitter)

    assert abs(tracker.acceleration()) < 0.3


def test_velocity_has_no_cliff_at_the_noise_floor(clock):
    """Velocity must fade in with movement, not switch on.

    Anything scaled by this — the whole prediction blend is — jumps by whatever
    the cliff is worth, one frame to the next, if it steps.
    """
    speeds = []
    for step_px in range(1, 9):
        step = step_px * QUANT
        tracker = feed(clock, [0.4 + step * i for i in range(12)])
        speeds.append(tracker.speed(0.2))

    assert speeds == sorted(speeds), "velocity must rise with movement"
    for slower, faster in zip(speeds, speeds[1:]):
        assert faster - slower < 0.12, "velocity stepped rather than climbed"
    assert speeds[0] > 0.0, "the slowest movement still reads as movement"


def test_a_stopped_fish_reports_no_velocity(clock):
    """The fade must still reach zero — a stationary reading is not movement."""
    tracker = feed(clock, [0.5] * 12)

    assert tracker.speed(0.2) == 0.0
    assert tracker.acceleration() == 0.0
