"""The whole-strip fallback is only for a strip that is actually bar-sized.

detect_progress offers the entire strip as a last-resort candidate, for a
caller that has already cropped tight to the fill. No caller does: both pass
the *padded* strip, which padded_progress_roi grows by the ROI's own height at
each end and so is three times a bar tall by construction.

So on the frames where the outline search found nothing, the fallback read the
whole padded strip as though it were the bar, and the padding is where the
game draws everything else. On tests/clips it returned 95% against a rainbow
gradient, a splash of impact VFX and the rod itself, one frame after reading
11% off the real bar and one frame before reading 12% again -- a confident
wrong number, which is the one kind of answer the reader is meant never to
give.
"""

import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.config import ConfigManager  # noqa: E402
from src.detector import Detector  # noqa: E402


def _reader(monkeypatch_bands_to_nothing=True):
    """A detector whose outline search finds nothing, as on the bad frames."""
    detector = Detector(ConfigManager(str(ROOT)))
    detector._progress_bands = lambda _frame: []
    # Any band handed to the reader reads as a full, confident bar.
    detector._read_progress_band = lambda _strip, prefer=None: (0.95, 1.5)
    return detector


def _strip(rows):
    return np.zeros((rows, 200, 3), dtype=np.uint8)


def test_the_padded_strip_is_not_offered_as_a_band():
    """Three times a bar's height is not a bar, however convincingly it reads."""
    reader = _reader()
    assert reader.detect_progress(_strip(42)) is None


def test_a_tight_crop_still_gets_the_fallback():
    """Which is what the fallback was for."""
    reader = _reader()
    assert reader.detect_progress(_strip(12)) == 0.95


def test_the_gate_matches_the_band_size_limits():
    """One row outside the range either way, and it is not a bar."""
    reader = _reader()
    assert reader.detect_progress(_strip(Detector.PROGRESS_MIN_BAND)) == 0.95
    assert reader.detect_progress(_strip(Detector.PROGRESS_MAX_BAND)) == 0.95
    assert reader.detect_progress(_strip(Detector.PROGRESS_MIN_BAND - 1)) is None
    assert reader.detect_progress(_strip(Detector.PROGRESS_MAX_BAND + 1)) is None


def test_a_real_band_is_still_read_from_a_padded_strip():
    """The gate only removes the whole-strip candidate, not the found ones."""
    detector = Detector(ConfigManager(str(ROOT)))
    detector._progress_bands = lambda _frame: [(14, 25)]
    detector._read_progress_band = lambda _strip, prefer=None: (0.42, 2.0)
    assert detector.detect_progress(_strip(42)) == 0.42
