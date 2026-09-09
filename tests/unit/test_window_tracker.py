"""The window tracker has to work off macOS, not just fall back to None.

Every ROI in this project is a fraction of the game window, so a platform
without a working backend cannot compute a single capture rect — the macro is
dead on arrival there. These cover the backend selection, the name matching
that has to recognise Sober as well as Roblox, and the caching contract the
detector relies on to avoid querying the window server every frame.
"""

import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.window_tracker import (  # noqa: E402
    WindowBounds,
    WindowTracker,
    create_backend,
    report_backend,
)
from src.window_tracker.base import (  # noqa: E402
    DEFAULT_APP_NAMES,
    WindowBackend,
    matches_app_name,
)


class FakeBackend(WindowBackend):
    """A backend that returns whatever the test hands it, and counts calls."""

    name = "fake"
    default_scale = 1.0

    def __init__(self, bounds=None, scale=None, hint=None):
        self.bounds = bounds
        self.scale = scale
        self.hint = hint
        self.find_calls = 0
        self.scale_calls = 0

    def find_window(self, app_names):
        self.find_calls += 1
        return self.bounds

    def detect_scale_factor(self):
        self.scale_calls += 1
        return self.scale

    def diagnostics(self):
        return self.hint


# -- backend selection -------------------------------------------------


@pytest.mark.parametrize(
    "platform, expected",
    [
        ("darwin", "macOS/Quartz"),
        ("win32", "Windows/Win32"),
        ("linux", "Linux/X11"),
        ("linux2", "Linux/X11"),
    ],
)
def test_each_supported_platform_gets_its_backend(platform, expected):
    backend = create_backend(platform)
    assert backend is not None, f"{platform} should have a backend"
    assert backend.name == expected


def test_an_unknown_platform_degrades_instead_of_raising():
    assert create_backend("sunos5") is None

    tracker = WindowTracker(backend=None)
    tracker.backend = None
    assert tracker.get_roblox_bounds() is None
    assert tracker.get_scale_factor() == 1.0


def test_every_backend_imports_off_its_own_platform():
    """A syntax or import error in the Win32 path must not wait for Windows."""
    for platform in ("darwin", "win32", "linux"):
        assert create_backend(platform) is not None


# -- name matching -----------------------------------------------------


@pytest.mark.parametrize(
    "candidate",
    ["Roblox", "roblox", "Roblox Player", "Sober", "sober", "org.vinegarhq.Sober"],
)
def test_the_game_is_recognised_under_each_client_name(candidate):
    assert matches_app_name(candidate, DEFAULT_APP_NAMES)


@pytest.mark.parametrize("candidate", ["", None, "Finder", "Terminal", "Discord"])
def test_unrelated_windows_are_not_matched(candidate):
    assert not matches_app_name(candidate, DEFAULT_APP_NAMES)


# -- caching -----------------------------------------------------------


def test_bounds_are_cached_between_frames():
    backend = FakeBackend(bounds=WindowBounds(0, 33, 1512, 882))
    tracker = WindowTracker(backend=backend)

    first = tracker.get_roblox_bounds()
    second = tracker.get_roblox_bounds()

    assert first == second
    assert backend.find_calls == 1, "the second frame should reuse the cache"


def test_invalidating_the_cache_forces_a_fresh_lookup():
    backend = FakeBackend(bounds=WindowBounds(0, 33, 1512, 882))
    tracker = WindowTracker(backend=backend)

    tracker.get_roblox_bounds()
    tracker.invalidate_cache()
    tracker.get_roblox_bounds()

    assert backend.find_calls == 2


def test_a_missing_window_is_not_cached_as_a_hit():
    backend = FakeBackend(bounds=None)
    tracker = WindowTracker(backend=backend)

    assert tracker.get_roblox_bounds() is None
    assert tracker.get_roblox_bounds() is None
    assert backend.find_calls == 2, "a miss must keep retrying, not stick"


def test_a_backend_that_raises_does_not_take_the_macro_down():
    class Exploding(FakeBackend):
        def find_window(self, app_names):
            raise RuntimeError("window server went away")

    tracker = WindowTracker(backend=Exploding())
    assert tracker.get_roblox_bounds() is None


# -- scale factor ------------------------------------------------------


def test_a_detected_scale_is_used_and_then_cached():
    backend = FakeBackend(scale=2.0)
    tracker = WindowTracker(backend=backend)

    assert tracker.get_scale_factor() == 2.0
    assert tracker.get_scale_factor() == 2.0
    assert backend.scale_calls == 1


def test_undetectable_scale_falls_back_to_the_backend_default():
    backend = FakeBackend(scale=None)
    backend.default_scale = 2.0
    tracker = WindowTracker(backend=backend)

    assert tracker.get_scale_factor() == 2.0


def test_macos_still_assumes_retina_but_the_others_assume_1x():
    """Guessing 1x on Retina silently captures a quarter of the ROI."""
    assert create_backend("darwin").default_scale == 2.0
    assert create_backend("win32").default_scale == 1.0
    assert create_backend("linux").default_scale == 1.0


# -- diagnostics -------------------------------------------------------


def test_wayland_is_called_out_by_name(monkeypatch):
    monkeypatch.setenv("WAYLAND_DISPLAY", "wayland-0")
    monkeypatch.setenv("DISPLAY", ":0")

    hint = create_backend("linux").diagnostics()
    assert hint is not None and "Wayland" in hint


def test_a_headless_shell_is_called_out_too(monkeypatch):
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    monkeypatch.delenv("XDG_SESSION_TYPE", raising=False)
    monkeypatch.delenv("DISPLAY", raising=False)

    hint = create_backend("linux").diagnostics()
    assert hint is not None and "DISPLAY" in hint


def test_a_healthy_x11_session_reports_nothing(monkeypatch):
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    monkeypatch.setenv("XDG_SESSION_TYPE", "x11")
    monkeypatch.setenv("DISPLAY", ":0")

    assert create_backend("linux").diagnostics() is None


def test_the_hint_is_logged_once_not_every_frame(caplog):
    backend = FakeBackend(bounds=None, hint="switch to X11")
    tracker = WindowTracker(backend=backend)

    with caplog.at_level("WARNING"):
        for _ in range(5):
            tracker.invalidate_cache()
            tracker.get_roblox_bounds()

    assert sum("switch to X11" in r.message for r in caplog.records) == 1


# -- the CI diagnostic -------------------------------------------------
#
# CI used to smoke-test the packaged binary with --list-profiles, which
# returns before a WindowTracker is ever built. A green Windows or Linux
# build therefore proved the bundle imported and packaged, and said nothing
# about whether window lookup worked. --check-backend is what closes that.


def test_check_backend_reports_a_working_backend(capsys):
    tracker = WindowTracker(backend=FakeBackend(
        bounds=WindowBounds(0, 33, 800, 480), scale=1.0
    ))

    assert report_backend(tracker) == 0
    out = capsys.readouterr().out
    assert "fake" in out and "800x480" in out and "1.0x" in out


def test_check_backend_actually_calls_the_platform_lookup(capsys):
    """The point of the flag: exercise find_window, not just import it."""
    backend = FakeBackend(bounds=None, scale=1.0)
    report_backend(WindowTracker(backend=backend))

    assert backend.find_calls == 1
    assert backend.scale_calls == 1


def test_check_backend_fails_when_the_platform_has_no_backend(capsys):
    tracker = WindowTracker(backend=FakeBackend())
    tracker.backend = None

    assert report_backend(tracker) == 1


def test_a_missing_game_window_is_not_a_failure(capsys):
    """Nothing is running in CI; that must not be read as a broken backend."""
    assert report_backend(WindowTracker(backend=FakeBackend(scale=1.0))) == 0
    assert "not found" in capsys.readouterr().out


def test_an_environment_hint_is_surfaced(capsys):
    tracker = WindowTracker(backend=FakeBackend(scale=1.0, hint="looks like Wayland"))
    report_backend(tracker)

    assert "looks like Wayland" in capsys.readouterr().out


def test_an_implausible_scale_factor_is_a_failure(capsys):
    tracker = WindowTracker(backend=FakeBackend(scale=0.0))
    tracker._scale_factor = 0.0

    assert report_backend(tracker) == 1
