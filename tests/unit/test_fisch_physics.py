"""The published formulas, and the estimator's ability to recover them.

These are regression tests on the game model rather than on the code: if the
game changes and scripts/measure_game_model.py disagrees with the wiki, these
are what should be updated, deliberately.
"""

import pathlib
import random
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src import fisch_physics as phys  # noqa: E402
from src.fight_estimator import FightEstimator  # noqa: E402


# --- published formulas -----------------------------------------------------

@pytest.mark.parametrize("p, expected", [
    (0, 8.0),          # the quoted "6.8 s of reeling, 8 s including the lock"
    (-40, 12.5333),    # the median fish
    (-80, 35.2),       # a p25 fish: a fixed timeout well under this kills it
    (300, 2.9),        # Tryhard Rod
])
def test_catch_time(p, expected):
    assert phys.catch_seconds(p) == pytest.approx(expected, abs=1e-3)


@pytest.mark.parametrize("p, expected", [(0, 0.5), (-40, 0.625), (-80, 1 / 1.2)])
def test_required_on_target(p, expected):
    assert phys.required_on_target(p) == pytest.approx(expected, abs=1e-4)


@pytest.mark.parametrize("control, width", [
    (0.0, 0.30),       # the zero-Control width, not a floor
    (0.70, 1.00),      # the widest rod
    (-0.10, 0.20),     # what tests/clips/active-*.mov actually measure
])
def test_bar_width(control, width):
    assert phys.bar_width(control) == pytest.approx(width)


def test_loss_does_not_scale_with_progress_speed():
    """Measured: off-target slope held at ~0.118 while gain varied 0.084-0.178."""
    assert phys.loss_rate() == phys.PROGRESS_RATE
    assert phys.gain_rate(-80) < phys.gain_rate(0) < phys.gain_rate(300)


def test_mean_travel_is_half_the_bound():
    """"Between +-40r%" is a range: the mean distance is half of it.

    Read as a fixed step this would be 0.40, and the 47 movements pooled from
    tests/clips average 0.200.
    """
    assert phys.mean_travel(1.0) == pytest.approx(0.20)


def test_a_fight_below_its_required_fraction_never_finishes():
    assert phys.fight_seconds(0.60, -40) == float("inf")
    assert phys.fight_seconds(0.90, -40) < 20.0


# --- the estimator ----------------------------------------------------------

def _replay(progress_speed, coverage, seed=0, dt=0.02, quantum=1 / 664):
    """Drive an estimator with a synthetic fight at a known coverage."""
    est = FightEstimator()
    rng = random.Random(seed)
    p, t = phys.START_PROGRESS, 0.0
    gain, loss = phys.gain_rate(progress_speed), phys.loss_rate()
    state = None
    while 0.0 < p < 1.0 and t < 120.0:
        on = rng.random() < coverage
        p += (gain if on else -loss) * dt
        t += dt
        state = est.update(t, round(p / quantum) * quantum, on)
    return est, state


def test_recovers_progress_speed_and_required_fraction():
    est, st = _replay(-40, 0.90)
    assert st.gain_rate == pytest.approx(phys.gain_rate(-40), rel=0.15)
    assert st.loss_rate == pytest.approx(phys.loss_rate(), rel=0.15)
    assert st.required_on_target == pytest.approx(0.625, abs=0.05)
    assert est.verdict() == "winning"


def test_calls_a_hopeless_fight_lost():
    """Coverage under what the fight needs: no amount of time helps."""
    est, st = _replay(-80, 0.60)
    assert est.verdict() == "lost"
    assert st.projected_seconds() == float("inf")


def test_rejects_drops_faster_than_the_game_allows():
    """A progress collapse steeper than the loss rate is the detector, not the game."""
    est = FightEstimator()
    est.update(0.00, 0.50, True)
    est.update(0.02, 0.50, True)
    st = est.update(0.04, 0.05, True)     # 45% gone in one 20 ms frame
    assert st.rejected_frames == 1


def test_one_bad_frame_does_not_lose_a_fight_that_is_being_won():
    """The false abandon seen in the field, from the frame that caused it.

    A single spurious reading at the head of the recent window used to set the
    sign of the whole trend, and the trend alone ended the fight: a clean climb
    at 3%/s with the bar on the fish every single frame reported -9%/s and a
    verdict of "lost" four seconds in. The estimator had already flagged that
    frame as impossible -- the drop back out of it is faster than the game can
    take progress away -- which is what makes the verdict indefensible.
    """
    est = FightEstimator()
    est.update(0.00, 0.55, True)              # one spurious reading
    t = p = 0.0
    for _ in range(89):                       # then a clean, on-target climb
        t += 0.05
        p += 0.03 * 0.05
        st = est.update(t, p, True)

    assert st.rejected_frames == 1            # the estimator saw it was bogus
    assert st.on_target_fraction == pytest.approx(1.0)
    assert est.recent_net_rate() > 0         # the trend survives the outlier
    assert est.verdict() != "lost"


def test_progress_falling_at_full_coverage_is_not_a_lost_fight():
    """On-target and losing progress at once is a broken instrument.

    The game gains progress precisely when the bar is on the fish, so the two
    cannot both be true, and the field log shows the pair together: "on-target
    100% against 62% needed". Reading that as an unwinnable fish throws away a
    fish that by every other measure is being caught.

    The contradiction is set on the state directly, because coverage is derived
    from progress runs and so cannot be driven apart from progress by any input
    -- and because with the trend now fitted across the window rather than its
    two endpoints, no single bad frame reaches this state either. It is the
    second line of defence, tested as one.
    """
    est = FightEstimator()
    t, p = 0.0, 0.60
    for _ in range(120):                      # a fight genuinely being lost
        t += 0.05
        p -= 0.02 * 0.05
        est.update(t, p, True)
    assert est.recent_net_rate() < 0
    assert est.verdict() == "lost"            # still called, on its own merits

    # Same estimator, same falling readings, with the coverage from the log
    # put against them: the answer becomes "I cannot tell" rather than "give
    # up", which is the only honest reading of a self-contradicting instrument.
    est.state.on_target_fraction = 1.0
    assert est.verdict() == "measuring"


def test_a_short_burst_is_not_enough_to_call_a_trend():
    """Two frames a tenth of a second apart are noise with a sign."""
    est = FightEstimator()
    est.update(0.00, 0.50, True)
    est.update(0.05, 0.49, True)
    assert est.recent_net_rate() is None


def test_counts_a_slash_as_a_boost_not_as_the_rate():
    est = FightEstimator()
    est.update(0.00, 0.50, True)
    st = est.update(0.02, 0.62, True)     # +12% in one frame: a slash
    assert st.boosts == 1


def test_sub_quantum_ticks_still_count_toward_time():
    """Dropping frames that round to no change would inflate every rate.

    At a slow fish the true step is under two quantisation steps, so roughly
    half the ticks read zero; if their time is discarded the rate doubles.
    """
    est, st = _replay(-60, 0.95)
    assert st.gain_rate == pytest.approx(phys.gain_rate(-60), rel=0.25)
