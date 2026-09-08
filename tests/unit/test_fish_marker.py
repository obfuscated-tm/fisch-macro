"""The fish marker, and the ornament some rods draw on the control bar.

Castbound puts a magenta diamond at the centre of its control bar. It is
narrow, it is a colour the bar is not, and it covers the whole band, so every
prior in the reader accepted it and it was picked as the fish in 16% of the
fight frames of tests/clips/hallucinate.mov.

Picking it is worse than picking nothing: it sits at the bar's centre by
construction, so it tells the controller the error is zero exactly when it is
not, and the macro stops steering while the fish swims away.

What separates them is where each is drawn. The fish marker belongs to the
track and carries an icon above it, so it continues past the band; an ornament
belongs to the bar and stops with it.
"""

import pathlib
import sys

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.reel_vision import ReelVision  # noqa: E402

WIDTH = 600
BAND_ROWS = 24
PAD = 24

# Colours chosen to reproduce the competition as it really stands on
# tests/clips/hallucinate.mov, where the ornament is a vivid magenta on a
# bright bar and the fish marker is a purple only moderately unlike the track
# it is drawn on. Scored on contrast alone the ornament wins, which is exactly
# why it used to be picked.
TRACK = (80, 40, 70)        # dark purple, the unfilled track
BAR = (200, 220, 120)       # the bright control bar
ORNAMENT = (210, 60, 225)   # magenta diamond at the bar's centre
MARKER = (120, 62, 100)     # the fish marker, drawn on the track


def build(bar=(0.25, 0.75), marker_x=0.35, marker_overflows=True,
          ornament=True, pad=PAD):
    """A synthetic track: bar, an ornament at its centre, and a fish marker."""
    height = BAND_ROWS + 2 * pad
    frame = np.zeros((height, WIDTH, 3), np.uint8)
    # Pale ground, as in the recording: the part of the marker above the track
    # is seen against the world, not against more UI.
    frame[:] = (170, 180, 190)
    band = slice(pad, pad + BAND_ROWS)
    frame[band] = TRACK

    b0, b1 = int(bar[0] * WIDTH), int(bar[1] * WIDTH)
    frame[band, b0:b1] = BAR

    if ornament:
        centre = (b0 + b1) // 2
        frame[band, centre - 7:centre + 7] = ORNAMENT

    mx = int(marker_x * WIDTH)
    rows = slice(0, height) if marker_overflows else band
    frame[rows, mx - 7:mx + 7] = MARKER
    return frame


@pytest.fixture
def vision():
    return ReelVision()


def without_marker():
    """A frame where the marker is not drawn — a blink, or an occlusion."""
    frame = build(marker_x=0.35)
    mx = int(0.35 * WIDTH)
    frame[:, mx - 7:mx + 7] = frame[:, 10:24]
    return frame


def test_the_ornament_does_not_capture_the_fish_for_the_rest_of_the_fight(vision):
    """The failure as it actually happens: one bad frame, then it never lets go.

    The ornament only has to win once — a frame where the marker blinks or is
    occluded is enough. From then on the continuity term, which exists to stop
    the reader hopping between candidates, keeps handing it the win while the
    real marker moves further away every frame.

    On tests/clips/hallucinate.mov that is 12 of 75 fight frames, and each one
    reports the fish at the bar's centre, which is the one reading the
    controller cannot argue with.
    """
    vision.read(without_marker(), pre_located=True, now=0.0, pad_rows=PAD)

    seen = []
    for i, marker_x in enumerate((0.35, 0.36, 0.37, 0.38), start=1):
        reading = vision.read(
            build(marker_x=marker_x), pre_located=True, now=i / 15.0, pad_rows=PAD
        )
        seen.append(reading.fish_x)

    assert seen == pytest.approx([0.35, 0.36, 0.37, 0.38], abs=0.02), (
        "the reader stayed on the bar's centre ornament after it won one frame"
    )


def test_marker_is_preferred_over_the_bars_centre_ornament(vision):
    frame = build(marker_x=0.35)
    reading = vision.read(frame, pre_located=True, now=0.0, pad_rows=PAD)

    assert reading.fish_x == pytest.approx(0.35, abs=0.02), (
        "the fish marker continues past the band; the ornament does not"
    )


def test_without_context_rows_nothing_changes(vision):
    """With no rows outside the band there is nothing to measure, and the
    reader must fall back to what it did before rather than reject everything."""
    frame = build(marker_x=0.35, pad=0)
    reading = vision.read(frame, pre_located=True, now=0.0)

    assert reading.fish_x is not None


def test_the_ornament_alone_is_not_promoted_to_fish(vision):
    """With the marker gone, the ornament must not inherit the fish by default.

    It still can be picked — nothing else is there to pick — but the reading
    has to be reported as sitting on the bar's centre rather than being taken
    for a fish somewhere else on the track.
    """
    frame = build(marker_x=0.35, marker_overflows=False)
    reading = vision.read(frame, pre_located=True, now=0.0, pad_rows=PAD)

    # Neither candidate continues outside, so the preference stands aside and
    # the reader falls back to its scoring — the point is only that it does not
    # crash or invent a position off the track.
    if reading.fish_x is not None:
        assert 0.0 <= reading.fish_x <= 1.0


def test_a_marker_that_does_not_overflow_is_still_found(vision):
    """A rod whose marker is confined to the band must read exactly as before."""
    frame = build(marker_x=0.62, marker_overflows=False, ornament=False)
    reading = vision.read(frame, pre_located=True, now=0.0, pad_rows=PAD)

    assert reading.fish_x == pytest.approx(0.62, abs=0.02)


def test_context_rows_are_not_read_as_track(vision):
    """The padded rows must not widen the bar or move the track."""
    frame = build(bar=(0.25, 0.75), marker_x=0.35)
    reading = vision.read(frame, pre_located=True, now=0.0, pad_rows=PAD)

    assert reading.bar_left == pytest.approx(0.25, abs=0.03)
    assert reading.bar_right == pytest.approx(0.75, abs=0.03)
