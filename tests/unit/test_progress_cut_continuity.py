"""The fill boundary does not teleport mid-fight either.

_progress_step returns the single sharpest colour step along the band's
profile, by argmax. On a rainbow-gradient rod the unfilled remainder of the
bar is the same gradient washed out, so the true fill boundary is a weak step
-- weak enough that a colour transition *inside* the gradient competes with it
on even terms. Whichever is fractionally sharper on a given frame wins, and the
reading snaps between the two.

Traced through production's reader on tests/clips/STRUGGLE-ROD.mov, frames
272-290, every one of them read from a band sitting correctly on the bar:

    0.34 0.52 0.51 0.51 0.37 0.37 0.52 0.52 0.39 0.52 0.39 0.40 0.52 0.52 0.40

A bar being yanked around by a fish sweeps through the values between; it does
not teleport between two fixed numbers and back six times in nineteen frames.

So the cut carries the same continuity the row band was given: a step near
where the boundary was last frame is preferred, and given up only to one that
is decisively sharper.
"""

import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.config import ConfigManager  # noqa: E402
from src.detector import Detector  # noqa: E402


def _profile(width, edges):
    """A Lab-ish column profile that steps up at each (column, size)."""
    profile = np.zeros((width, 3), dtype=np.float32)
    for column, size in edges:
        profile[column:] += size
    return profile


def _reader():
    return Detector(ConfigManager(str(ROOT)))


def test_the_sharpest_step_still_wins_with_no_history():
    reader = _reader()
    found = reader._progress_step(_profile(200, [(60, 12.0), (140, 30.0)]))
    assert found is not None
    assert abs(found[0] - 140) <= 2


def test_a_step_near_last_frame_s_cut_is_preferred():
    """The rival is sharper, but the boundary was over there a frame ago."""
    reader = _reader()
    profile = _profile(200, [(60, 25.0), (140, 30.0)])
    found = reader._progress_step(profile, prefer=60)
    assert found is not None
    assert abs(found[0] - 60) <= 2


def test_a_decisively_sharper_step_still_takes_it():
    """A slash really does move the boundary a long way in one frame."""
    reader = _reader()
    profile = _profile(200, [(60, 6.0), (140, 30.0)])
    found = reader._progress_step(profile, prefer=60)
    assert found is not None
    assert abs(found[0] - 140) <= 2


def test_the_preference_is_local():
    """A far-away step gets no help from having been the cut once."""
    reader = _reader()
    profile = _profile(200, [(30, 25.0), (140, 30.0)])
    # 30 is far outside the window around the previous cut at 150.
    found = reader._progress_step(profile, prefer=150)
    assert found is not None
    assert abs(found[0] - 140) <= 2


def test_a_new_fight_forgets_where_the_boundary_was():
    detector = _reader()
    detector._progress_cut = 0.42
    detector.reset_session()
    assert detector._progress_cut is None
