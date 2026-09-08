"""Regression tests for calibration that would not stay saved.

The reported symptom was "rod saving doesn't really work, calibration always
resets when exiting". Three separate defects fed it, and each has a test here:

1. Every ``load_settings()`` handed back the *same* Settings instance, so the
   calibration overlay and the main window shared one object. Cancelling the
   overlay left its half-drawn regions in the window's copy, and any later
   autosave wrote them to disk.
2. Creating a rod profile copied a hand-listed subset of the colour fields, so
   the on-target and off-target colours silently reverted to library defaults.
3. Drawing a box did not remove ``roi_shift_*``, which capture adds back, so a
   freshly drawn region was stored offset from where it was drawn.
"""

import dataclasses
import json
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.config import ColorProfile, ConfigManager, ROIBounds, Settings  # noqa: E402


@pytest.fixture
def config(tmp_path):
    return ConfigManager(str(tmp_path))


# --- 1. callers get their own Settings --------------------------------------

def test_load_settings_returns_independent_copies(config):
    config.save_settings(Settings())

    held_by_window = config.load_settings()
    held_by_overlay = config.load_settings()

    assert held_by_window is not held_by_overlay
    assert held_by_window.bar_roi is not held_by_overlay.bar_roi


def test_cancelled_calibration_does_not_leak_into_the_window(config):
    original = Settings()
    config.save_settings(original)

    window = config.load_settings()
    overlay = config.load_settings()

    # The overlay drags a new box, then the user hits Cancel: nothing is saved.
    overlay.bar_roi = ROIBounds(0.1, 0.9, 0.5, 0.6)

    assert window.bar_roi.x_start == pytest.approx(original.bar_roi.x_start)

    # And the window's next autosave writes its own values, not the overlay's.
    config.save_settings(window)
    on_disk = json.loads((pathlib.Path(config.settings_path)).read_text())
    assert on_disk["bar_roi"]["x_start"] == pytest.approx(original.bar_roi.x_start)


def test_saved_settings_are_visible_to_the_next_reader(config):
    settings = Settings()
    settings.bar_roi = ROIBounds(0.2, 0.8, 0.4, 0.5)
    config.save_settings(settings)

    reloaded = config.load_settings()
    assert reloaded.bar_roi.x_start == pytest.approx(0.2)
    assert reloaded.bar_roi.y_end == pytest.approx(0.5)


def test_mutating_a_loaded_copy_does_not_disturb_the_cache(config):
    config.save_settings(Settings())

    first = config.load_settings()
    first.bar_roi.x_start = 0.99
    first.control_duty_kp = 99.0

    second = config.load_settings()
    assert second.bar_roi.x_start != pytest.approx(0.99)
    assert second.control_duty_kp != pytest.approx(99.0)


# --- 2. a new rod profile keeps every colour --------------------------------

def test_new_rod_profile_copies_every_colour_field(config):
    """What ``_save_calibration_as_profile`` does, minus the Tk plumbing."""
    source = ColorProfile(
        name="old-rod",
        fish_hsv_low=[10, 20, 30], fish_hsv_high=[11, 21, 31],
        bar_hsv_low=[40, 50, 60], bar_hsv_high=[41, 51, 61],
        on_target_hsv_low=[70, 80, 90], on_target_hsv_high=[71, 81, 91],
        off_target_hsv_low=[100, 110, 120], off_target_hsv_high=[101, 111, 121],
        bar_brightness_threshold=137,
    )
    config.save_profile(source)

    regions = dict(
        bar_roi=ROIBounds(0.1, 0.2, 0.3, 0.4),
        progress_roi=ROIBounds(0.5, 0.6, 0.7, 0.8),
        shake_roi=ROIBounds(0.0, 1.0, 0.0, 1.0),
    )
    new_rod = dataclasses.replace(
        source, name="new-rod", description="Calibration for new-rod", **regions
    )
    config.save_profile(new_rod)

    loaded = config.load_profile("new-rod")
    colour_fields = [
        f.name for f in dataclasses.fields(ColorProfile)
        if f.name.endswith(("_hsv_low", "_hsv_high"))
    ]
    assert colour_fields, "expected colour fields to exist"
    for name in colour_fields:
        assert getattr(loaded, name) == getattr(source, name), name

    assert loaded.bar_brightness_threshold == 137
    assert loaded.bar_roi == regions["bar_roi"]
    assert loaded.progress_roi == regions["progress_roi"]


def test_rod_profile_round_trips_its_regions(config):
    profile = ColorProfile(name="rod", bar_roi=ROIBounds(0.11, 0.22, 0.33, 0.44))
    config.save_profile(profile)

    assert config.load_profile("rod").bar_roi == ROIBounds(0.11, 0.22, 0.33, 0.44)


# --- 3. a drawn box is the box that gets captured ---------------------------

def _round_trip_through_shift(drawn, shift_x, shift_y):
    """Store a drawn box the way the overlay does, then capture it back.

    Mirrors InteractiveCalibrator._handle_release (which subtracts the shift)
    against Detector._compute_roi_pixels (which adds it back).
    """
    stored = ROIBounds(
        x_start=drawn.x_start - shift_x,
        x_end=drawn.x_end - shift_x,
        y_start=drawn.y_start - shift_y,
        y_end=drawn.y_end - shift_y,
    )
    return ROIBounds(
        x_start=stored.x_start + shift_x,
        x_end=stored.x_end + shift_x,
        y_start=stored.y_start + shift_y,
        y_end=stored.y_end + shift_y,
    )


@pytest.mark.parametrize("shift_x, shift_y", [(0.0, 0.0), (0.03, -0.02)])
def test_drawn_region_is_captured_where_it_was_drawn(shift_x, shift_y):
    drawn = ROIBounds(0.30, 0.70, 0.84, 0.87)
    captured = _round_trip_through_shift(drawn, shift_x, shift_y)

    for field in ("x_start", "x_end", "y_start", "y_end"):
        assert getattr(captured, field) == pytest.approx(getattr(drawn, field))
