"""The aim point must not step as the fish crosses "stationary".

ReelController is what production predicts with: the macro hands it the raw
fish position and steers on decision.fish_projected. It used to switch the
projection off entirely below stationary_speed, so a fish hovering around that
speed moved the aim point between fish_x and fish_x + lead from one tick to the
next. The error that feeds is multiplied by duty_kp, which is 10 by default, so
a step there is a step in the duty cycle caused by nothing physical.

Replaying the fight in tests/clips/hallucinate.mov, the projection was switched
off on 53 of 79 ticks, which is how often the boundary is actually in play.
"""

import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.reel_controller import ControlParams, ReelController  # noqa: E402

DT = 0.02


def lead_at(speed, params=None):
    """The projection offset the controller settles on for a steady speed."""
    params = params or ControlParams()
    ctrl = ReelController(params)
    x, t = 0.5, 0.0
    decision = None
    for _ in range(12):
        decision = ctrl.decide(x, 0.5, now=t)
        x += speed * DT
        t += DT
    return decision.fish_projected - x + speed * DT


def test_the_projection_fades_in_rather_than_switching_on():
    params = ControlParams()
    step = params.stationary_speed / 8.0
    speeds = [step * i for i in range(1, 17)]   # spans the threshold

    leads = [lead_at(v, params) for v in speeds]

    for slower, faster in zip(leads, leads[1:]):
        assert faster - slower < step * params.lead_seconds * 3, (
            "the aim point stepped as the fish crossed stationary_speed"
        )
    assert leads == sorted(leads), "a faster fish must never be led less"


def test_a_fish_above_the_threshold_is_projected_exactly_as_before():
    """The taper is 1 at stationary_speed, so faster fish are untouched."""
    params = ControlParams()
    speed = params.stationary_speed * 3

    assert lead_at(speed, params) == pytest.approx(
        speed * params.lead_seconds, rel=0.15
    )


def test_a_still_fish_is_not_led():
    assert lead_at(0.0) == pytest.approx(0.0, abs=1e-6)
