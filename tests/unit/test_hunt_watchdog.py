"""A cast that never took must not strand the macro.

The reported symptom: "sometimes a rod gets stuck — it thinks it has cast when
it hasn't, and ends up stuck, which is bad if you're going to AFK for a long
period". Nothing in either hunt state timed out, so the macro would wait for a
bite that could not come — the line was never in the water — for as long as the
session ran.
"""

import pathlib
import sys
import time

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.config import Settings  # noqa: E402
from src.macro import MacroEngine, MacroState  # noqa: E402


@pytest.fixture
def engine():
    """An engine with only the hunt bookkeeping wired up."""
    eng = MacroEngine.__new__(MacroEngine)
    eng._hunt_started_at = None
    eng._emit_log = lambda *_a, **_k: None
    return eng


def test_a_silent_hunt_eventually_recasts(engine):
    settings = Settings(hunt_timeout_seconds=30.0)
    engine._begin_hunt()

    assert not engine._hunt_has_stalled(settings), "a fresh hunt is not stalled"

    engine._hunt_started_at = time.time() - 31.0
    assert engine._hunt_has_stalled(settings)


def test_a_hunt_that_is_going_somewhere_is_left_alone(engine):
    """Clicking a shake prompt proves the line is in the water."""
    settings = Settings(hunt_timeout_seconds=30.0)
    engine._hunt_started_at = time.time() - 31.0

    # What _try_click_shake does on a successful click.
    engine._hunt_started_at = time.time()

    assert not engine._hunt_has_stalled(settings)


def test_the_watchdog_fires_once_per_hunt(engine):
    """It must not keep firing every tick once the deadline has passed."""
    settings = Settings(hunt_timeout_seconds=30.0)
    engine._hunt_started_at = time.time() - 31.0

    assert engine._hunt_has_stalled(settings)
    assert not engine._hunt_has_stalled(settings)


def test_the_watchdog_can_be_turned_off(engine):
    settings = Settings(hunt_timeout_seconds=0.0)
    engine._hunt_started_at = time.time() - 10_000.0

    assert not engine._hunt_has_stalled(settings)


def test_a_hunt_that_never_started_does_not_fire(engine):
    settings = Settings(hunt_timeout_seconds=30.0)

    assert engine._hunt_started_at is None
    assert not engine._hunt_has_stalled(settings)
