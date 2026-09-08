"""Stopping must be felt immediately, not after the current tick unwinds.

``stop()`` is called from the Tk main thread — the Stop button, the hotkey and
window close all land there — so anything it waits for is time the whole UI is
frozen. It used to join the worker for up to three seconds, and the loop's
pauses were plain ``time.sleep``, so a stop could sit through a full interval
before it was even noticed.
"""

import pathlib
import sys
import threading
import time

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.config import Settings  # noqa: E402
from src.macro import MacroEngine, MacroState  # noqa: E402


class _SlowDetector:
    """A detector whose every call costs a realistic tick."""

    def __init__(self, cost=0.02):
        self.cost = cost
        self.calls = 0

    def detect_all(self, scan_shake: bool = True):
        self.calls += 1
        time.sleep(self.cost)
        return _Result()

    def reset_session(self):
        pass


class _Result:
    bar_active = False
    bite_confirmed = False
    fish_x = None
    bar_left = None
    bar_right = None
    on_target = False
    progress = 0.0
    shake_pos = None
    shake_confidence = 0.0
    debug_frame = None
    debug_source = None
    reading = None
    track_box = None


class _Controller:
    def __init__(self):
        self.killed = False
        self.released = 0

    def start(self):
        pass

    def stop(self):
        pass

    def is_killed(self):
        return self.killed

    def mouse_hold(self):
        pass

    def mouse_release(self, force: bool = False):
        self.released += 1

    def rapid_click(self, count: int = 1, interval: float = 0.0):
        pass

    def mouse_click(self, x=None, y=None):
        pass


class _Config:
    def __init__(self, **overrides):
        self._settings = Settings(**overrides)

    def load_settings(self):
        return self._settings


@pytest.fixture
def engine():
    # A long recast delay is the worst case: it is the state that waits longest
    # between detector calls.
    config = _Config(recast_delay=5.0, scan_interval_ms=100, auto_recast=True)
    eng = MacroEngine(_SlowDetector(), _Controller(), config)
    yield eng
    eng.stop()
    if eng._thread is not None:
        eng._thread.join(timeout=2.0)


def test_stop_returns_without_waiting_for_the_worker(engine):
    engine.start()
    time.sleep(0.05)

    started = time.perf_counter()
    engine.stop()
    elapsed = time.perf_counter() - started

    # Whatever the loop is in the middle of, the caller is not made to wait.
    assert elapsed < 0.15, f"stop() blocked the caller for {elapsed:.3f}s"
    assert not engine.is_running()


def test_worker_exits_within_one_tick_of_a_stop(engine):
    engine.state = MacroState.COMPLETE
    engine.start()
    time.sleep(0.05)

    engine.stop()
    engine._thread.join(timeout=1.0)

    assert not engine._thread.is_alive(), (
        "the loop was still running a second after the stop request"
    )


def test_a_pending_wait_wakes_on_a_stop_request():
    engine = MacroEngine(_SlowDetector(), _Controller(), _Config())
    engine._stop_event.clear()

    started = time.perf_counter()
    threading.Timer(0.05, engine._stop_event.set).start()
    should_stop = engine._wait(5.0)
    elapsed = time.perf_counter() - started

    assert should_stop
    assert elapsed < 0.5, f"_wait sat out the full interval ({elapsed:.3f}s)"


def test_restart_after_an_unwaited_stop(engine):
    engine.start()
    time.sleep(0.05)
    engine.stop()

    engine.start()
    assert engine.is_running()
