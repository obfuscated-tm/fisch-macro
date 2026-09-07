"""The reeling minigame's rules, as documented and as measured.

Sources, in order of authority:

1. Measurements from tests/clips via scripts/measure_game_model.py. These win
   where they disagree with the wiki, and the disagreements are noted.
2. The Fisch wiki -- Fishing#Reeling, Resilience#Gameplay Effects, Control.
   Its Resilience page is flagged stale since v1.67.0 and the game is past
   v1.91, which is why (1) exists.

This module holds no state and imports nothing from the macro. It is the place
to look up what the game does; deciding what to do about it belongs elsewhere.
"""

from __future__ import annotations

# --- progress ---------------------------------------------------------------

# Wiki: progress is "gained or lost by 12% every second". Measured off-target
# slopes were 0.1186 (active-fail) and 0.1179 (doing-nothing) -- two independent
# fights agreeing to within 1%, and both within 2% of the wiki's figure. The
# small shortfall is the progress bar's border: the fill never quite reaches the
# last pixel column of the ROI.
PROGRESS_RATE = 0.12

# Progress Speed scales the *gain* only. The wiki says it changes "the rate at
# which reeling minigame progress is gained" and never mentions loss, which
# leaves it ambiguous -- but the measurements settle it: across four clips the
# on-target slope varied 0.084 to 0.178 with the fish while the off-target slope
# stayed put. That asymmetry is the whole reason some fish are much harder than
# others with no visible cue.
#
# 836 of 1512 fish carry their own Progress Speed, median -40%.

# Wiki: inputs are locked for 1.2 s or until progress reaches 20%. Taking
# progress as 20% with the clock already 1.2 s in reproduces the published catch
# time exactly, so that is how the two are reconciled here.
INPUT_LOCK_SECONDS = 1.2
START_PROGRESS = 0.20


def gain_rate(progress_speed_pct: float = 0.0) -> float:
    """Progress gained per second while the fish is inside the bar."""
    return PROGRESS_RATE * max(0.0, 1.0 + progress_speed_pct / 100.0)


def loss_rate() -> float:
    """Progress lost per second while it is outside. Does not scale."""
    return PROGRESS_RATE


def catch_seconds(progress_speed_pct: float = 0.0) -> float:
    """Wiki's T(p) = 1.2 + 6.8/(1+p/100): a flawless catch with no boosts.

    T(0) = 8.0 s, but the median fish's -40% makes it 12.5 s and a p25 fish's
    -80% makes it 35 s. A fixed timeout that does not know this will call a
    healthy fight dead.
    """
    denom = 1.0 + progress_speed_pct / 100.0
    if denom <= 0:
        return float("inf")
    return INPUT_LOCK_SECONDS + 6.8 / denom


def required_on_target(progress_speed_pct: float = 0.0) -> float:
    """The on-target fraction below which progress trends to zero.

    gain*f = loss*(1-f)  ->  f = loss/(gain+loss) = 1/(2 + p/100).

    0.500 at a neutral fish, 0.625 at the median fish's -40%, 0.833 at a p25
    fish's -80%. Nothing on screen announces which one is being fought.
    """
    g, l = gain_rate(progress_speed_pct), loss_rate()
    return l / (g + l) if (g + l) > 0 else 1.0


def progress_speed_from_gain(observed_gain: float) -> float:
    """Invert gain_rate: what Progress Speed explains this measured slope."""
    return (observed_gain / PROGRESS_RATE - 1.0) * 100.0


def fight_seconds(on_target_fraction: float, progress_speed_pct: float = 0.0,
                  from_progress: float = START_PROGRESS) -> float:
    """How long a fight takes at a sustained on-target fraction.

    Infinite when the fraction is at or below what the fight requires, which is
    the useful part: it says the fight is not slow, it is lost.
    """
    net = (gain_rate(progress_speed_pct) * on_target_fraction
           - loss_rate() * (1.0 - on_target_fraction))
    if net <= 0:
        return float("inf")
    return (1.0 - from_progress) / net


# --- the control bar --------------------------------------------------------

# Wiki: 30% of the track at zero Control, +1% per +0.01. Control runs -0.295 to
# +0.7 across rods, so 30% is the middle of the range and not a floor -- widths
# measured in tests/clips are 0.20, 0.44, 0.61 and 0.92.
ZERO_CONTROL_WIDTH = 0.30
CONTROL_RANGE = (-0.295, 0.70)


def bar_width(control: float) -> float:
    return min(1.0, max(0.005, ZERO_CONTROL_WIDTH + control))


def control_from_width(width: float) -> float:
    return width - ZERO_CONTROL_WIDTH


# --- the fish ---------------------------------------------------------------

# Wiki, Resilience#Gameplay Effects. Effective resilience is
# (fish base + rod + bait + enchantments)/100, clamped above 0.2. Across 1512
# fish and 260 rods that lands at median 0.64, p25 0.35, p95 1.20, with 117 fish
# pinned on the 0.2 floor.
RESILIENCE_FLOOR = 0.2
RESILIENCE_QUANTILES = {"p25": 0.35, "median": 0.64, "p95": 1.20}

FISH_MIN, FISH_MAX = 0.03, 0.90     # the fish cannot leave this band
LEFT_ZONE = 0.05                    # inside it, the fish always moves right

# Each tick the game draws from uniform(2r, 5.1r) seconds and starts a new
# movement once that much time has passed since the last. Redrawn every tick, so
# movements arrive sooner than the interval's mean suggests; the wiki puts the
# effective average at 2.15r.
INTERVAL_PER_R = (2.0, 5.1)
MEAN_INTERVAL_PER_R = 2.15

# The destination is uniform *within* +-40r% of the fish's position, not at it.
# That distinction is measurable: as a range the mean distance travelled is half
# the bound, 0.20 track widths at r=1.0, and 47 movements pooled from the clips
# average 0.200. Read as a fixed step it would predict 0.40.
AMPLITUDE_PER_R = 0.40
AMPLITUDE_R_CLAMP = (0.8, 1.2)
LEFT_KICK_PER_R = (0.09, 0.32)
LEFT_KICK_R_CLAMP = (0.8, 1.4)

# A movement takes this long, and is not protected: a new one can start
# mid-glide, so the fish often turns before arriving.
TWEEN_PER_R = (1.3, 3.5)
TWEEN_R_CLAMP = (0.1, 1.5)


def mean_travel(resilience: float) -> float:
    """Expected distance covered by one movement."""
    rc = min(max(resilience, AMPLITUDE_R_CLAMP[0]), AMPLITUDE_R_CLAMP[1])
    return 0.5 * AMPLITUDE_PER_R * rc


def resilience_from_interval(seconds: float) -> float:
    return max(RESILIENCE_FLOOR, seconds / MEAN_INTERVAL_PER_R)


def resilience_from_tween(seconds: float) -> float:
    return max(RESILIENCE_FLOOR, seconds / (0.5 * sum(TWEEN_PER_R)))
