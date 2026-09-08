"""The bar must not swallow the far end of the track.

The track is translucent, so its two ends lie over different scenery and read
as different shades of the same element. On tests/frames/other-rod-fail they
sat 18.7 Lab apart against a tolerance of 18.0 — a miss of less than a unit —
and the end that missed was counted as bar and merged into it, giving readings
like 0.05-1.00 for a bar that ends at 0.65.

That error is the one behind an "ON TARGET" that is not: a bar reported as
reaching the end of the track contains the fish by definition, so the macro is
told it is winning while the fish is nowhere near the real bar.
"""

import pathlib
import sys

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402

from src.reel_vision import ReelVision  # noqa: E402


def _lab(bgr):
    """One BGR colour in Lab, the space the reader compares in."""
    return cv2.cvtColor(np.uint8([[bgr]]), cv2.COLOR_BGR2LAB).astype(np.float32)[0, 0]

WIDTH = 600
BAND = 24
PAD = 24

BAR = (200, 220, 120)
MARKER = (120, 62, 100)


def build(track_left=(80, 40, 70), track_right=(80, 40, 70), bar=(0.05, 0.65)):
    height = BAND + 2 * PAD
    frame = np.zeros((height, WIDTH, 3), np.uint8)
    frame[:] = (170, 180, 190)
    band = slice(PAD, PAD + BAND)

    b0, b1 = int(bar[0] * WIDTH), int(bar[1] * WIDTH)
    frame[band, :b0] = track_left
    frame[band, b0:b1] = BAR
    frame[band, b1:] = track_right

    mx = (b0 + b1) // 2
    frame[:, mx - 7:mx + 7] = MARKER
    return frame


# Two shades of the same track, 21 Lab apart — just past the 18.0 tolerance,
# as the real pair on other-rod-fail was at 18.7.
TRACK = (80, 40, 70)
TRACK_LIT = (100, 58, 90)


def test_a_shaded_track_end_is_not_swallowed_by_the_bar():
    """The far end differs in shade but is still track, and memory knows it."""
    vision = ReelVision()

    # The remembered background sits between the two shades. That is where a
    # fight puts it: which end is taken as background depends on where the bar
    # is parked, so over a few seconds the memory has seen both.
    vision.read(build(), pre_located=True, now=0.0, pad_rows=PAD)
    vision._bg_lab = (vision._bg_lab + _lab(TRACK_LIT)) / 2.0

    shaded = build(track_right=TRACK_LIT)
    reading = vision.read(shaded, pre_located=True, now=1 / 15.0, pad_rows=PAD)

    assert reading.bar_right == pytest.approx(0.65, abs=0.03), (
        "the bar was merged with the far end of the track"
    )
    assert reading.bar_left == pytest.approx(0.05, abs=0.03)


def test_a_stale_background_memory_is_not_trusted():
    """A remembered background that no longer resembles the track is ignored.

    Otherwise a biome change would leave the reader marking the bar itself as
    background and splitting it in two.
    """
    vision = ReelVision()
    vision.read(build(), pre_located=True, now=0.0, pad_rows=PAD)

    # Everything changes at once, as it does when the fight moves biome.
    moved = build(track_left=(30, 120, 200), track_right=(30, 120, 200))
    reading = vision.read(moved, pre_located=True, now=1 / 15.0, pad_rows=PAD)

    assert reading.bar_left is not None
    assert reading.bar_left == pytest.approx(0.05, abs=0.05)
    assert reading.bar_right == pytest.approx(0.65, abs=0.05)
