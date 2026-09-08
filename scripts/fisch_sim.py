#!/usr/bin/env python3
"""The reeling minigame as the game actually implements it.

Every constant here is either quoted from the Fisch wiki or measured off the
recordings in tests/clips by scripts/measure_game_model.py. Where the two
disagree the measurement wins and the difference is noted.

This exists because the controller was being judged against a model that did not
resemble the game: a Gaussian random walk for the fish and a bar 13% of the track
wide. The real fish glides between destinations, and the narrowest bar in the
recordings is 20% with others at 44%, 61% and 92%. A controller tuned against
the wrong plant is tuned to the wrong problem.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

# --- bar physics ------------------------------------------------------------
# Measured by scripts/measure_physics.py from tests/clips/physics-*.mov. The
# wiki documents only the shape -- hold accelerates right, release accelerates
# left, and it bounces off the ends -- so these numbers are ours.
ACCEL_HOLD = 0.90      # track widths / s^2, rightward while held
ACCEL_FREE = 1.05      # leftward while released
DAMPING = 0.8          # velocity decay

# --- progress ---------------------------------------------------------------
# Wiki: gained or lost at 12%/s. Measured on active-fail and doing-nothing, the
# off-target slope is 0.1186 and 0.1179 -- 12% within 2%, and the same in two
# independent fights, while the on-target slope varied 0.084-0.178 with the
# fish. So loss is the fixed one and Progress Speed scales only the gain.
PROGRESS_LOSS_RATE = 0.12
PROGRESS_GAIN_RATE = 0.12

# Wiki: inputs are locked for 1.2 s or until progress reaches 20%. Starting at
# 20% with the clock already 1.2 s in reproduces T(p) = 1.2 + 6.8/(1+p/100)
# exactly, which is the published catch time.
INPUT_LOCK_SECONDS = 1.2
START_PROGRESS = 0.20

# --- fish movement (wiki: Resilience#Gameplay Effects) ----------------------
FISH_MIN, FISH_MAX = 0.03, 0.90     # the fish cannot leave this band
LEFT_ZONE = 0.05                    # inside this it always moves right
INTERVAL_PER_R = (2.0, 5.1)         # seconds; redrawn every tick
AMPLITUDE_PER_R = 0.40              # destination uniform in +-this
AMPLITUDE_R_CLAMP = (0.8, 1.2)
LEFT_KICK_PER_R = (0.09, 0.32)      # rightward escape from the left zone
LEFT_KICK_R_CLAMP = (0.8, 1.4)
TWEEN_PER_R = (1.3, 3.5)            # seconds a movement takes
TWEEN_R_CLAMP = (0.1, 1.5)

GAME_TICK = 1.0 / 60.0              # Roblox heartbeat; the reroll rate


def _clamp(v, lo, hi):
    return lo if v < lo else (hi if v > hi else v)


@dataclass
class Bar:
    """Damped double integrator, clamped so the bar stays fully on the track."""

    width: float = 0.40
    pos: float = 0.5
    vel: float = 0.0

    def step(self, hold: bool, dt: float) -> None:
        self.vel += (ACCEL_HOLD if hold else -ACCEL_FREE) * dt
        self.vel -= self.vel * DAMPING * dt
        self.pos += self.vel * dt
        lo, hi = self.width / 2, 1.0 - self.width / 2
        if lo >= hi:                      # a full-width bar cannot move
            self.pos, self.vel = 0.5, 0.0
        elif self.pos <= lo:
            self.pos, self.vel = lo, max(0.0, self.vel)
        elif self.pos >= hi:
            self.pos, self.vel = hi, min(0.0, self.vel)

    def covers(self, x: float) -> bool:
        return abs(x - self.pos) <= self.width / 2


class Fish:
    """The documented fish: a sequence of interruptible glides.

    Each tick the game draws a fresh number from uniform(2r, 5.1r) seconds and
    starts a new movement once more time than that has passed since the last one
    began. Because it is redrawn every tick rather than committed to, movements
    arrive sooner than the interval's mean would suggest; the wiki puts the
    effective average at about 2.15r.

    The destination is uniform within +-40r% of where the fish is -- *within*,
    not *at*, which matters: read as a fixed step it would give a mean travel of
    0.40 track widths, but read as a range it gives 0.20, and 0.20 is what the
    47 movements pooled from tests/clips actually measure.

    Movements are not protected once begun. A new one can start mid-glide, so
    the fish often turns around before arriving.
    """

    def __init__(self, resilience: float, rng: random.Random, start: float = 0.5):
        self.r = max(0.2, resilience)          # wiki: clamped above 0.2
        self.rng = rng
        self.pos = start
        self._since_move = 0.0
        self._from = start
        self._to = start
        self._elapsed = 0.0
        self._duration = 0.0

    def _begin(self) -> None:
        rng = self.rng
        if self.pos < LEFT_ZONE:
            rc = _clamp(self.r, *LEFT_KICK_R_CLAMP)
            lo, hi = (k * rc for k in LEFT_KICK_PER_R)
            dest = self.pos + rng.uniform(lo, hi)
        else:
            rc = _clamp(self.r, *AMPLITUDE_R_CLAMP)
            reach = AMPLITUDE_PER_R * rc
            dest = self.pos + rng.uniform(-reach, reach)

        rd = _clamp(self.r, *TWEEN_R_CLAMP)
        self._from = self.pos
        self._to = _clamp(dest, FISH_MIN, FISH_MAX)
        self._duration = rng.uniform(*(k * rd for k in TWEEN_PER_R))
        self._elapsed = 0.0
        self._since_move = 0.0

    def step(self, dt: float) -> None:
        self._since_move += dt
        if self._since_move > self.rng.uniform(*(k * self.r for k in INTERVAL_PER_R)):
            self._begin()

        if self._duration > 0.0:
            self._elapsed += dt
            frac = min(1.0, self._elapsed / self._duration)
            self.pos = self._from + (self._to - self._from) * frac
        self.pos = _clamp(self.pos, FISH_MIN, FISH_MAX)


@dataclass
class FightParams:
    """One fight's plant. All three are unknown to the macro when it starts.

    bar_width      30% of the track at zero Control, +1% per +0.01 of it.
                   Measured in tests/clips: 0.20, 0.44, 0.61, 0.92.
    resilience     (fish base + rod + bait + enchantments)/100, clamped >= 0.2.
                   Across 1512 fish and 260 rods the effective value is median
                   0.64, p25 0.35, p95 1.20.
    progress_speed percent. 836 of 1512 fish carry their own, median -40%.
    """

    bar_width: float = 0.40
    resilience: float = 0.64
    progress_speed: float = 0.0

    @property
    def gain_rate(self) -> float:
        return PROGRESS_GAIN_RATE * max(0.0, 1.0 + self.progress_speed / 100.0)

    @property
    def loss_rate(self) -> float:
        return PROGRESS_LOSS_RATE

    @property
    def required_on_target(self) -> float:
        """On-target fraction below which progress trends to zero.

        gain*f = loss*(1-f)  ->  f = loss/(gain+loss). At a neutral fish that is
        0.5; at the median fish's -40% it is 0.625; at -80% it is 0.833. This is
        the number that decides whether a fight is winnable, and nothing on
        screen announces it.
        """
        g, l = self.gain_rate, self.loss_rate
        return l / (g + l) if (g + l) > 0 else 1.0

    def catch_seconds(self) -> float:
        """Wiki's T(p) = 1.2 + 6.8/(1+p/100): a flawless catch, no boosts."""
        denom = 1.0 + self.progress_speed / 100.0
        return INPUT_LOCK_SECONDS + 6.8 / denom if denom > 0 else float("inf")


@dataclass
class FightResult:
    won: bool
    seconds: float
    on_target: float
    progress: float
    reversals: int = 0
    timed_out: bool = False


def default_grid():
    """The parameter space to judge a controller over.

    Widths are the four measured in tests/clips. Resiliences are the quantiles
    of the effective distribution, including the 0.2 floor that 117 fish sit on.
    Progress speeds are 0, the median fish and the p25 fish -- which is where
    the required on-target fraction climbs to 0.83 and fights stop being
    forgiving.
    """
    widths = (0.20, 0.40, 0.61, 0.92)
    resiliences = (0.2, 0.35, 0.64, 0.90, 1.20)
    speeds = (0, -40, -80)
    return [FightParams(w, r, p)
            for w in widths for r in resiliences for p in speeds]


# ---------------------------------------------------------------------------
# the macro's view of the fight
# ---------------------------------------------------------------------------

DT = 0.02                  # 20 ms tick, matching scan_interval_ms
LATENCY_TICKS = 3          # ~60 ms of capture, OpenCV and input round trip

# The detector's error is not Gaussian. Measured across the recorded clips the
# bar centre's median frame-to-frame step is under 0.005 track widths while the
# 90th percentile reaches 0.10 -- a heavy tail of occasional bad edges. That
# tail is what a squared braking term turns into a lurch, so a simulation with
# only small Gaussian noise cannot reproduce the failure at all.
POSITION_NOISE = 0.006
OUTLIER_RATE = 0.10
OUTLIER_SCALE = 0.07
DROPOUT_RATE = 0.20
BLIND_TICKS = 6            # after this many misses the macro releases


def perceive(history, rng, tick):
    """What the controller can see this tick: a delayed, noisy, lossy sample."""
    idx = tick - LATENCY_TICKS
    if idx < 0:
        return None
    fish, bar = history[idx]
    if rng.random() < DROPOUT_RATE:
        return None

    def noisy(value):
        err = rng.gauss(0, POSITION_NOISE)
        if rng.random() < OUTLIER_RATE:
            err += rng.choice((-1.0, 1.0)) * abs(rng.gauss(OUTLIER_SCALE, OUTLIER_SCALE / 2))
        return value + err

    return noisy(fish), noisy(bar)


def simulate_fight(params, decide, seed, max_seconds=90.0):
    """Run one fight to a win, a loss, or a timeout.

    ``decide(fish_x, bar_center, now, params)`` returns True to hold. It sees
    only what the detector reports, never the true state, and it is called every
    tick a reading is available -- the command is a latching physical state, so
    any tick that declines to answer leaves the button where it was.

    Progress starts at 20% with the clock 1.2 s in, which is the input lock the
    wiki describes and which makes a flawless neutral fight take 8.0 s.
    """
    rng = random.Random(seed * 31 + 7)
    bar = Bar(width=params.bar_width)
    fish = Fish(params.resilience, random.Random(seed), start=0.5)

    progress = START_PROGRESS
    t = INPUT_LOCK_SECONDS
    on_ticks = 0
    ticks = 0
    hold = False
    history = []
    last_seen = -99
    positions = []

    max_ticks = int(max_seconds / DT)
    while ticks < max_ticks:
        history.append((fish.pos, bar.pos))
        seen = perceive(history, rng, ticks)
        if seen is not None:
            hold = decide(seen[0], seen[1], t, params)
            last_seen = ticks
        elif ticks - last_seen > BLIND_TICKS:
            hold = False

        bar.step(hold, DT)
        fish.step(DT)

        on = bar.covers(fish.pos)
        progress += (params.gain_rate if on else -params.loss_rate) * DT
        on_ticks += on
        positions.append(bar.pos)
        ticks += 1
        t += DT

        if progress >= 1.0:
            return FightResult(True, t, on_ticks / ticks, 1.0,
                               count_reversals(positions))
        if progress <= 0.0:
            return FightResult(False, t, on_ticks / ticks, 0.0,
                               count_reversals(positions))

    return FightResult(False, t, on_ticks / max(ticks, 1), progress,
                       count_reversals(positions), timed_out=True)


def count_reversals(positions, min_amplitude: float = 0.02) -> int:
    """Direction changes in the bar's travel with a real excursion behind them.

    Counting sign changes of velocity does not work: velocity is near zero
    exactly when it flips, so any magnitude threshold applied at the flip tick
    rejects every genuine reversal. This tracks the running extremum instead and
    counts a reversal once the trace retraces from it by ``min_amplitude``,
    which is what a person actually sees as the bar wobbling.
    """
    if not positions:
        return 0
    count, direction, extremum = 0, 0, positions[0]
    for value in positions:
        if direction > 0:
            if value > extremum:
                extremum = value
            elif extremum - value >= min_amplitude:
                count, direction, extremum = count + 1, -1, value
        elif direction < 0:
            if value < extremum:
                extremum = value
            elif value - extremum >= min_amplitude:
                count, direction, extremum = count + 1, 1, value
        else:
            if value - extremum >= min_amplitude:
                direction, extremum = 1, value
            elif extremum - value >= min_amplitude:
                direction, extremum = -1, value
    return count
