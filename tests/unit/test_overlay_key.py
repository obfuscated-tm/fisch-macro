"""The live-view key has to describe what the overlay actually draws.

Two things had gone wrong. The key was a hand-written sentence that had drifted
from the drawing code — it called the off-target bracket blue when it is amber,
and never mentioned the aim marker or the progress strip. And two of the things
it did name were never drawn at all: ``detect_all`` rendered the frame before
the control telemetry existed, so the predicted-fish line and the aim marker
had nothing to draw from and the macro used that frame as-is.
"""

import pathlib
import sys

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.config import ConfigManager  # noqa: E402
from src.detector import (  # noqa: E402
    OVERLAY_COLORS,
    OVERLAY_LEGEND,
    DetectionResult,
    Detector,
    legend_hex,
)

# Drawn as alternatives to one another, so a single frame can only show one.
MUTUALLY_EXCLUSIVE = {"bar_on", "bar_off"}


@pytest.fixture
def detector(tmp_path):
    return Detector(ConfigManager(str(tmp_path)))


@pytest.fixture
def frame():
    return np.full((26, 260, 3), 70, np.uint8)


def _telemetry():
    return {
        "predicted_fish_x": 0.71,
        "effective_bar": 0.35,
        "progress_smooth": 0.53,
        "macro_state": "Reeling",
    }


def _pixels_of(vis, name):
    want = np.array(OVERLAY_COLORS[name], dtype=int)
    flat = vis.reshape(-1, 3).astype(int)
    return int((np.abs(flat - want).sum(axis=1) == 0).sum())


def test_every_key_entry_names_a_real_colour():
    for name, label in OVERLAY_LEGEND:
        assert name in OVERLAY_COLORS, f"{label!r} points at no colour"
        assert legend_hex(name).startswith("#")
        assert len(legend_hex(name)) == 7


def test_key_covers_everything_the_overlay_draws():
    assert set(OVERLAY_COLORS) == {name for name, _label in OVERLAY_LEGEND}


@pytest.mark.parametrize("on_target", [False, True])
def test_overlay_draws_every_key_entry(detector, frame, on_target):
    result = DetectionResult(
        bar_active=True, fish_x=0.62, bar_left=0.22, bar_right=0.42,
        on_target=on_target, progress=0.46,
    )
    vis = detector.get_debug_frame(frame, result, extras=_telemetry())

    expected_bracket = "bar_on" if on_target else "bar_off"
    for name, label in OVERLAY_LEGEND:
        if name in MUTUALLY_EXCLUSIVE and name != expected_bracket:
            continue
        assert _pixels_of(vis, name) > 0, f"{label} is in the key but was not drawn"


def test_predicted_and_aim_markers_need_the_control_telemetry(detector, frame):
    """The regression itself: without extras these two cannot be drawn.

    Which is why the macro now re-renders from ``debug_source`` instead of
    using the frame ``detect_all`` produced before the controller had decided
    anything.
    """
    result = DetectionResult(
        bar_active=True, fish_x=0.62, bar_left=0.22, bar_right=0.42, progress=0.46,
    )

    bare = detector.get_debug_frame(frame, result)
    assert _pixels_of(bare, "fish_predicted") == 0
    assert _pixels_of(bare, "bar_aim") == 0

    full = detector.get_debug_frame(frame, result, extras=_telemetry())
    assert _pixels_of(full, "fish_predicted") > 0
    assert _pixels_of(full, "bar_aim") > 0


def test_detect_all_keeps_the_analysed_crop_for_re_rendering():
    """``debug_source`` is what makes the re-render possible."""
    assert "debug_source" in DetectionResult.__dataclass_fields__


def test_overlay_marks_positions_where_the_numbers_say(detector, frame):
    """A marker at x=0.25 lands a quarter of the way across, not elsewhere."""
    result = DetectionResult(
        bar_active=True, fish_x=0.25, bar_left=0.60, bar_right=0.80, progress=0.0,
    )
    vis = detector.get_debug_frame(frame, result)

    want = np.array(OVERLAY_COLORS["fish"], dtype=int)
    hits = np.argwhere((np.abs(vis.astype(int) - want).sum(axis=2) == 0))
    assert hits.size, "fish marker not drawn"

    track_width = vis.shape[1]
    centre = hits[:, 1].mean() / track_width
    assert centre == pytest.approx(0.25, abs=0.02)
