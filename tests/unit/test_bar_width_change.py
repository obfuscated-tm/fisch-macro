"""A rod's bar is not a fixed size, and the reader has to follow it.

Some rods change the bar's size at random during a fight; Castbound's shrinks
once the catch stops being perfect — measured on tests/clips/hallucinate.mov it
halves, from 0.67 of the track down to 0.35.

The width used to be settled from the first dozen readings and never revisited,
which fights the game in both directions: a bar that grows has the excess cut
back off, and one that shrinks is refused for being too narrow, costing the
reading altogether. What must *not* happen is the opposite mistake — accepting
a merged reading as a new size — so a change is adopted only once a window of
consistent readings agrees on it.
"""

import pathlib
import sys

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.reel_vision import ReelVision  # noqa: E402

WIDTH = 600
BAND = 24
PAD = 24

TRACK = (80, 40, 70)
BAR = (200, 220, 120)
MARKER = (120, 62, 100)


def frame(bar):
    """One track with the bar spanning `bar` (a pair of fractions)."""
    img = np.zeros((BAND + 2 * PAD, WIDTH, 3), np.uint8)
    img[:] = (170, 180, 190)
    band = slice(PAD, PAD + BAND)
    b0, b1 = int(bar[0] * WIDTH), int(bar[1] * WIDTH)
    img[band, :b0] = TRACK
    img[band, b0:b1] = BAR
    img[band, b1:] = TRACK
    mx = (b0 + b1) // 2
    img[:, mx - 7:mx + 7] = MARKER
    return img


def play(vision, spans, start=0.0):
    """Feed a sequence of bar spans; return the width read from each."""
    widths = []
    for i, span in enumerate(spans):
        reading = vision.read(
            frame(span), pre_located=True, now=start + i / 15.0, pad_rows=PAD
        )
        widths.append(reading.bar_width)
    return widths


def centred(width):
    return (0.5 - width / 2, 0.5 + width / 2)


def test_a_bar_that_shrinks_is_still_read():
    """Castbound's case: the bar halves, ending far below the original floor."""
    vision = ReelVision()

    # Long enough at full size to settle the width.
    play(vision, [centred(0.60)] * 20)
    settled = vision._established_width
    assert settled == pytest.approx(0.60, abs=0.05)

    # Then it shrinks past 0.45 x settled, which used to be refused outright
    # and never read again. At the rate hallucinate.mov actually shrinks —
    # about 0.004 of the track per frame — so the window the re-settling looks
    # at holds one size rather than a ramp.
    shrunk = [centred(w) for w in np.linspace(0.58, 0.20, 110)]
    widths = play(vision, shrunk, start=2.0)

    # Crossing the floor costs a short blind spot while the refusals pile up
    # into a window that agrees. It has to be short, and it has to end.
    missed = sum(1 for w in widths if w is None)
    assert missed <= 15, f"the bar was lost for {missed} frames"

    tail = [w for w in widths[-8:] if w is not None]
    assert len(tail) == 8, "the shrunken bar never came back"
    assert tail[-1] == pytest.approx(0.20, abs=0.05)


def test_a_bar_that_grows_is_not_cut_back_down():
    """The other reported shape: a rod that changes size outright, mid-fight."""
    vision = ReelVision()
    play(vision, [centred(0.25)] * 20)

    grown = [centred(0.60)] * 20
    widths = play(vision, grown, start=2.0)

    tail = [w for w in widths[-8:] if w is not None]
    assert len(tail) == 8, "the grown bar stopped being read"
    assert tail[-1] == pytest.approx(0.60, abs=0.06)


def test_an_erratic_reading_does_not_resettle_the_width():
    """A merge is wide too, but it disagrees with itself frame to frame."""
    vision = ReelVision()
    play(vision, [centred(0.30)] * 20)
    settled = vision._established_width

    # Alternating wildly — nothing here is a size the bar actually holds.
    erratic = [centred(w) for w in (0.9, 0.32, 0.85, 0.31, 0.95, 0.30) * 3]
    play(vision, erratic, start=2.0)

    assert vision._established_width == pytest.approx(settled, rel=0.2), (
        "an inconsistent run was mistaken for a genuine change of size"
    )
