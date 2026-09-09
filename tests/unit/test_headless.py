"""The headless runner is the only way to use the macro on a small display.

The GUI panel is 372x572 with a 340x500 minimum, so anything smaller cannot
show it at all. That makes this path load-bearing rather than a convenience,
and the parts that bit in the GUI — stale window bounds, hotkey double-fire on
key release — have to be handled here too.
"""

import pathlib
import sys
import threading

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.headless import HeadlessRunner  # noqa: E402
from src.macro import MacroStats  # noqa: E402
from src.window_tracker import WindowBounds  # noqa: E402


class FakeController:
    def __init__(self):
        self.hotkey_callbacks = []

    def on_hotkey(self, callback):
        self.hotkey_callbacks.append(callback)

    def press_hotkey(self):
        for callback in self.hotkey_callbacks:
            callback()


class FakeDetector:
    def __init__(self):
        self.window_info = []

    def set_window_info(self, bounds, scale_factor):
        self.window_info.append((bounds, scale_factor))


class FakeEngine:
    def __init__(self):
        self.controller = FakeController()
        self.detector = FakeDetector()
        self.stats = MacroStats()
        self.stats.reset()
        self._running = False
        self.starts = 0
        self.stops = 0

    def on_state_change(self, callback):
        pass

    def on_log(self, callback):
        pass

    def is_running(self):
        return self._running

    def start(self):
        self._running = True
        self.starts += 1

    def stop(self, wait: float = 0.0):
        self._running = False
        self.stops += 1


class FakeTracker:
    def __init__(self, bounds=WindowBounds(0, 33, 800, 480), scale=1.0):
        self.bounds = bounds
        self.scale = scale
        self.invalidations = 0
        self.backend = type("B", (), {"name": "fake"})()

    def invalidate_cache(self):
        self.invalidations += 1

    def get_roblox_bounds(self):
        return self.bounds

    def get_scale_factor(self):
        return self.scale


class FakeSettings:
    killswitch_key = "f6"
    active_profile = "Daybreaker"


@pytest.fixture
def runner():
    engine = FakeEngine()
    tracker = FakeTracker()
    return HeadlessRunner(engine, config_manager=None,
                          window_tracker=tracker, settings=FakeSettings())


# -- starting ----------------------------------------------------------


def test_starting_hands_the_window_to_the_detector(runner):
    assert runner.start_macro()

    assert runner.engine.starts == 1
    bounds, scale = runner.engine.detector.window_info[-1]
    assert (bounds.width, bounds.height) == (800, 480)
    assert scale == 1.0


def test_a_missing_window_does_not_start_a_blind_run(runner):
    runner.window_tracker.bounds = None

    assert runner.start_macro() is False
    assert runner.engine.starts == 0, "running without ROIs would click at random"


def test_every_start_re_resolves_the_window(runner):
    """The window can be moved or resized between runs, and ROIs are relative."""
    runner.start_macro()
    runner.stop_macro()
    runner.window_tracker.bounds = WindowBounds(120, 40, 1280, 720)
    runner.start_macro()

    assert runner.window_tracker.invalidations == 2
    bounds, _ = runner.engine.detector.window_info[-1]
    assert (bounds.width, bounds.height) == (1280, 720)


def test_starting_twice_does_not_stack_runs(runner):
    runner.start_macro()
    runner.start_macro()

    assert runner.engine.starts == 1


# -- hotkey ------------------------------------------------------------


def test_the_hotkey_toggles_the_run(runner):
    runner._connect_callbacks()

    runner.engine.controller.press_hotkey()
    assert runner.engine.is_running()

    runner._last_hotkey_toggle = 0.0  # step past the debounce
    runner.engine.controller.press_hotkey()
    assert not runner.engine.is_running()
    assert runner.engine.stops == 1


def test_a_double_fire_does_not_start_and_immediately_stop(runner):
    """The hotkey fires on key release; two in a row would cancel the start."""
    runner._connect_callbacks()

    runner.engine.controller.press_hotkey()
    runner.engine.controller.press_hotkey()

    assert runner.engine.is_running()
    assert runner.engine.stops == 0


def test_the_hotkey_can_retry_after_the_window_was_missing(runner):
    runner._connect_callbacks()
    runner.window_tracker.bounds = None

    runner.engine.controller.press_hotkey()
    assert not runner.engine.is_running()

    runner.window_tracker.bounds = WindowBounds(0, 33, 800, 480)
    runner._last_hotkey_toggle = 0.0
    runner.engine.controller.press_hotkey()
    assert runner.engine.is_running()


# -- shutdown ----------------------------------------------------------


def test_shutdown_stops_a_running_macro(runner):
    threading.Timer(0.05, runner.shutdown).start()

    assert runner.run(autostart=True) == 0
    assert runner.engine.stops == 1
    assert not runner.engine.is_running()


def test_waiting_mode_does_not_start_on_its_own(runner):
    threading.Timer(0.05, runner.shutdown).start()

    runner.run(autostart=False)
    assert runner.engine.starts == 0


# -- output ------------------------------------------------------------


@pytest.mark.parametrize(
    "seconds, expected",
    [(0, "0s"), (45, "45s"), (90, "1m30s"), (3600, "1h00m"), (7830, "2h10m")],
)
def test_uptime_reads_as_a_duration(seconds, expected):
    assert HeadlessRunner._format_duration(seconds) == expected


def test_stats_line_reports_the_session(runner, capsys):
    runner.engine.stats.record_cast()
    runner.engine.stats.record_catch()
    runner.engine.stats.record_fail()

    runner._print_stats()

    out = capsys.readouterr().out
    assert "caught=1" in out and "failed=1" in out and "rate=50%" in out
