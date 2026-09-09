"""Repeated actions must not repeat exactly — and the reel loop must not wobble.

The macro's timings all come from config values, so left alone it charges every
cast for the same microsecond count forever. These tests hold the scatter that
fixes that, and the boundary it must not cross: the switching law in
src/reel_controller.py realises a duty cycle at tick resolution, and noise added
there is not disguise, it is tracking error.
"""

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.humanize import jitter_point, jitter_seconds  # noqa: E402


def test_scatter_is_actually_random():
    draws = {jitter_seconds(0.68, 0.12) for _ in range(200)}
    assert len(draws) > 190, "duration repeated itself far too often"


def test_scatter_stays_within_two_sigma():
    value, frac = 0.68, 0.12
    bound = 2.0 * value * frac
    for _ in range(2000):
        assert abs(jitter_seconds(value, frac) - value) <= bound + 1e-9


def test_floor_is_never_breached():
    # A large fraction on a small duration is the case that would otherwise
    # produce a cast held for a negative number of seconds.
    for _ in range(2000):
        assert jitter_seconds(0.05, 0.9, floor=0.02) >= 0.02


def test_zero_fraction_leaves_the_value_alone():
    assert jitter_seconds(2.5, 0.0) == 2.5


def test_click_scatter_stays_inside_the_radius():
    for _ in range(2000):
        x, y = jitter_point(500, 400, 3.0)
        # Rounding to whole pixels can push a point at the rim out by half a
        # pixel either way, which is why this is not a bare <= 3.
        assert (x - 500) ** 2 + (y - 400) ** 2 <= (3.0 + 1.0) ** 2


def test_zero_radius_leaves_the_target_alone():
    assert jitter_point(500, 400, 0.0) == (500, 400)


def test_the_reel_loop_is_left_deterministic():
    source = (ROOT / "src" / "reel_controller.py").read_text()
    assert "humanize" not in source and "random" not in source, (
        "the switching law must stay deterministic: jitter there costs win "
        "rate and disguises nothing the fish's own motion does not already"
    )
