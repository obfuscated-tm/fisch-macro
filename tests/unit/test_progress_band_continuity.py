"""The progress bar does not move mid-fight, so the reader should not either.

detect_progress used to re-decide which row band held the bar from scratch on
every frame, on confidence alone. The two or three bands that pass the presence
tests are often separated by very little, so the winner could alternate between
the bar and a patch of scenery from one frame to the next -- and the loser's
reading is not a near miss, it is a fill fraction measured off something that
is not the bar, arriving downstream as a confident wrong number.

Measured over the project's own recordings (tests/clips, 12746 frames): 288
frame-to-frame progress changes were larger than the game can physically
produce, and 219 of them -- 76% -- landed on exactly the frame the band
selection flipped to a non-overlapping row range, against a 4% base rate for
switches. Every one of the fifteen worst, all of them a near-empty bar reading
as near-full or the reverse, was switch-correlated. The five clips that never
switched band produced no impossible readings at all.

These exercise the selection rule directly, with synthetic readings, so they
need no clip fixtures and no display.
"""

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.detector import Detector  # noqa: E402


class _Reader(Detector):
    """Just the band-selection rule, without a config or a screen."""

    def __init__(self):
        self._progress_band = None


def test_the_most_convincing_band_wins_when_there_is_no_history():
    reader = _Reader()
    reads = [((0, 8), 0.20, 1.0), ((9, 17), 0.80, 2.0)]
    assert reader._pick_progress_band(reads) == ((9, 17), 0.80)


def test_a_marginally_better_rival_does_not_steal_the_band():
    """The bar has not moved; a rival a shade more convincing is noise."""
    reader = _Reader()
    reader._progress_band = (0, 8)
    reads = [((0, 8), 0.42, 2.0), ((9, 17), 0.97, 2.6)]
    band, fill = reader._pick_progress_band(reads)
    assert band == (0, 8)
    assert fill == 0.42


def test_a_decisively_better_rival_does_take_the_band():
    """A window resize or a new fight really does move the bar."""
    reader = _Reader()
    reader._progress_band = (0, 8)
    reads = [((0, 8), 0.42, 1.0), ((9, 17), 0.97, 9.0)]
    band, fill = reader._pick_progress_band(reads)
    assert band == (9, 17)
    assert fill == 0.97


def test_the_band_is_released_when_nothing_is_there_any_more():
    reader = _Reader()
    reader._progress_band = (0, 8)
    reads = [((20, 28), 0.55, 1.0)]
    assert reader._pick_progress_band(reads) == ((20, 28), 0.55)


def test_overlap_is_what_counts_as_the_same_band():
    """Bands shift by a row or two as the ROI jitters; that is still the bar."""
    reader = _Reader()
    reader._progress_band = (4, 12)
    reads = [((5, 13), 0.50, 1.0), ((30, 38), 0.90, 1.4)]
    band, _ = reader._pick_progress_band(reads)
    assert band == (5, 13)


def test_a_new_fight_starts_without_an_incumbent():
    """Otherwise the last fight's rows outrank the new fight's actual bar."""
    from src.config import ConfigManager

    detector = Detector(ConfigManager(str(ROOT)))
    assert detector._progress_band is None
    detector._progress_band = (0, 8)
    detector.reset_session()
    assert detector._progress_band is None
