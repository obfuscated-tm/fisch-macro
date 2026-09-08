"""What kind of fight this is, worked out while fighting it.

The macro cannot know in advance what it is up against. The rod is knowable, but
the rod is the smaller half: across 1512 fish the fish contributes the larger
share of resilience (median 44% against a rod's median 20%) and essentially all
of the Progress Speed variance, and which fish is on the line is not revealed
until it is caught. Two fights with identical equipment can need 50% on-target
or 83% with nothing on screen to distinguish them.

So the parameters are measured from the fight itself. Progress is the instrument:
its slope while the fish is inside the bar gives the gain rate, its slope while
outside gives the loss rate, and the ratio of the two gives the on-target
fraction this fight requires. All three are available within a few seconds.

Coverage is taken from progress too, rather than from the detector's own
on-target flag. Progress *is* the game's verdict on whether the fish is inside
the bar, so it settles the question that the geometry only estimates -- and the
geometry is measurably optimistic. Cross-checked against progress on the
recordings, the detector calls on-target while progress is falling 62 times in
active-fail against a single frame of the reverse, which is the arrow-glyph
latch that src/reel_vision.py documents: when the fish is momentarily lost, the
next best candidate is a glyph printed *inside* the bar, so the fish reads as
on-target by construction. A macro scoring itself on that flag thinks it is
winning a fight it is losing.

The estimator's most useful output is not a tuning value but a verdict: whether
what is actually being achieved beats what this fight demands. When it does not,
the fight is lost no matter how long it runs, and the macro is better off
letting go and recasting than spending another half minute on it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Deque, List, Optional, Tuple
from collections import deque

from src import fisch_physics as phys


# Progress is read as a pixel column out of a few hundred, so a single frame's
# change sits near the quantisation step and is far too coarse to differentiate.
# Rates are accumulated over seconds instead, and only reported once there is
# enough of each state to mean anything.
MIN_STATE_SECONDS = 0.8

# Rates come from sustained runs, not from individual frames, and the two have
# to be kept apart. Deciding a frame's side from the sign of its own change and
# then averaging those frames inflates both rates: conditioning on dp > 0 keeps
# the noise that pushed it over. Within a run lasting this long the sign is not
# in doubt, so the run's total change over its own duration is unbiased. Short
# runs still count toward coverage -- they happened -- they just do not get a
# vote on the rate.
MIN_RUN_SECONDS = 0.4

# The rates are properties of the fish and hold for the whole fight, so they are
# measured over all of it. Coverage is not: it is how well the fight is being
# fought right now, and a fight that went well and then fell apart still
# averages out fine over its lifetime. tests/clips/errors.mov has exactly that
# shape -- progress climbs to 45% and then collapses to nothing -- and reads as
# comfortably winning right up to the end if coverage is taken cumulatively.
RECENT_WINDOW_SECONDS = 6.0

# A frame-to-frame change beyond what the game can do is not the game. Downward
# it is a hard bound: nothing reduces progress faster than the loss rate, so a
# larger drop is the detector losing the bar or the UI changing underneath it.
# Upward it is not a bound at all -- slashes and rod abilities add 1-20% in a
# single frame, legitimately -- so a large rise is recorded as a boost and kept
# out of the slope rather than thrown away.
DROP_TOLERANCE = 0.02       # absolute slack on top of loss_rate * dt
BOOST_THRESHOLD = 0.015     # a rise this far beyond the gain bound is a boost


@dataclass
class FightState:
    """Everything the estimator currently believes, and how sure it is."""

    bar_width: Optional[float] = None
    gain_rate: Optional[float] = None
    loss_rate: Optional[float] = None
    progress_speed: Optional[float] = None
    resilience: Optional[float] = None

    progress: float = 0.0
    peak_progress: float = 0.0
    on_target_fraction: float = 0.0
    """Coverage over the last few seconds -- how the fight is going now."""

    lifetime_on_target: float = 0.0
    """Coverage over the whole fight, for reporting rather than for deciding."""
    elapsed: float = 0.0
    boosts: int = 0
    rejected_frames: int = 0

    # How often the detector's geometric on-target flag matches what progress
    # actually did. Persistently low means the fish detection is unreliable --
    # worth surfacing, because the controller is steering on that same flag.
    detector_agreement: float = 1.0

    @property
    def required_on_target(self) -> Optional[float]:
        if self.gain_rate is None or self.loss_rate is None:
            return None
        total = self.gain_rate + self.loss_rate
        return self.loss_rate / total if total > 0 else 1.0

    @property
    def margin(self) -> Optional[float]:
        """How far the achieved on-target fraction clears what is needed."""
        need = self.required_on_target
        if need is None:
            return None
        return self.on_target_fraction - need

    net_rate: Optional[float] = None
    """Progress per second over the fight so far, however it was achieved."""

    def projected_seconds(self) -> Optional[float]:
        """Time to fill the bar at the rates and coverage seen so far.

        Infinite when progress is not trending upward, which is the useful
        answer: the fight is not slow, it is lost.
        """
        if self.gain_rate is not None and self.loss_rate is not None:
            f = self.on_target_fraction
            net = self.gain_rate * f - self.loss_rate * (1.0 - f)
        elif self.net_rate is not None:
            # No sustained one-sided run to decompose yet. The aggregate still
            # says which way the fight is going, which is what this is for.
            net = self.net_rate
        else:
            return None
        if net <= 0:
            return float("inf")
        return (1.0 - self.progress) / net


class FightEstimator:
    """Fold per-frame readings into a picture of the current fight."""

    def __init__(self, min_state_seconds: float = MIN_STATE_SECONDS):
        self.min_state_seconds = min_state_seconds
        self.reset()

    def reset(self) -> None:
        self.state = FightState()
        self._t0: Optional[float] = None
        self._last_t: Optional[float] = None
        self._last_progress: Optional[float] = None
        self._first_progress: Optional[float] = None
        self._on_time = 0.0
        self._off_time = 0.0
        self._on_gain = 0.0
        self._off_loss = 0.0
        self._on_ticks = 0
        self._ticks = 0
        self._flag_agreements = 0
        self._flag_samples = 0
        self._recent: Deque[Tuple[float, bool]] = deque()
        self._recent_progress: Deque[Tuple[float, float]] = deque()
        self._run_rising: Optional[bool] = None
        self._run_dp = 0.0
        self._run_dt = 0.0
        self._widths: Deque[float] = deque(maxlen=90)
        self._onsets: List[float] = []
        self._fish_prev: Optional[float] = None
        self._fish_dir = 0

    # -- input ----------------------------------------------------------

    def update(
        self,
        now: float,
        progress: float,
        on_target: bool,
        bar_left: Optional[float] = None,
        bar_right: Optional[float] = None,
        fish_x: Optional[float] = None,
    ) -> FightState:
        if self._t0 is None:
            self._t0 = now
            self._first_progress = progress
        st = self.state
        st.elapsed = now - self._t0

        if bar_left is not None and bar_right is not None and bar_right > bar_left:
            self._widths.append(bar_right - bar_left)
            # The ROI clips the bar when it reaches either wall, so low readings
            # are normal there and the wide end of the distribution is the truth.
            ordered = sorted(self._widths)
            st.bar_width = ordered[int(0.9 * (len(ordered) - 1))]

        self._track_fish(now, fish_x)

        dt = 0.0 if self._last_t is None else now - self._last_t
        if 0.0 < dt < 1.0 and self._last_progress is not None:
            self._account(progress, dt, on_target)

        self._trim_recent(now)
        if self._recent:
            st.on_target_fraction = sum(r for _, r in self._recent) / len(self._recent)
        elif self._ticks:
            st.on_target_fraction = self._on_ticks / self._ticks
        else:
            st.on_target_fraction = float(bool(on_target))
        st.lifetime_on_target = (
            self._on_ticks / self._ticks if self._ticks else st.on_target_fraction
        )
        st.detector_agreement = (
            self._flag_agreements / self._flag_samples if self._flag_samples else 1.0
        )
        st.progress = progress
        st.peak_progress = max(st.peak_progress, progress)
        self._recent_progress.append((now, progress))
        if self._first_progress is not None and st.elapsed > 1e-6:
            st.net_rate = (progress - self._first_progress) / st.elapsed

        self._last_t = now
        self._last_progress = progress
        self._publish()
        return st

    def _account(self, progress: float, dt: float, on_target: bool) -> None:
        dp = progress - self._last_progress
        loss_bound = phys.loss_rate() * dt + DROP_TOLERANCE
        if dp < -loss_bound:
            # Faster than the game can take progress away: not the game.
            self.state.rejected_frames += 1
            return

        if dp == 0.0:
            # Progress is read as a pixel column, so a slow fight rounds to no
            # change on many ticks -- at a -29% fish the true step is under two
            # quantisation steps, so roughly half of them read zero.
            #
            # Those ticks still happened. Dropping them keeps the run's progress
            # but discards its time, which inflates every rate by the fraction
            # dropped. So a zero extends the current run rather than ending it,
            # and counts toward coverage as whatever the run is already doing.
            if self._run_rising is not None:
                self._run_dt += dt
                self._recent.append(
                    (self._last_t if self._last_t is not None else 0.0, self._run_rising)
                )
                self._ticks += 1
                self._on_ticks += self._run_rising
            return

        rising = dp > 0
        self._recent.append((self._last_t if self._last_t is not None else 0.0, rising))
        self._ticks += 1
        self._on_ticks += rising
        self._flag_samples += 1
        self._flag_agreements += (rising == bool(on_target))

        # An unusually large rise is a slash or a rod ability: real progress,
        # but it says nothing about the sustained rate, so it ends the run
        # rather than joining it.
        if rising and dp > phys.gain_rate(300.0) * dt + BOOST_THRESHOLD:
            self.state.boosts += 1
            self._close_run()
            return

        if self._run_rising is None or rising == self._run_rising:
            self._run_rising = rising
            self._run_dp += dp
            self._run_dt += dt
        else:
            self._close_run()
            self._run_rising = rising
            self._run_dp = dp
            self._run_dt = dt

    def _trim_recent(self, now: float) -> None:
        cutoff = now - RECENT_WINDOW_SECONDS
        while self._recent and self._recent[0][0] < cutoff:
            self._recent.popleft()
        while self._recent_progress and self._recent_progress[0][0] < cutoff:
            self._recent_progress.popleft()

    def _close_run(self) -> None:
        """Bank a finished run of one-sided motion, if it lasted long enough."""
        if self._run_rising is not None and self._run_dt >= MIN_RUN_SECONDS:
            if self._run_rising:
                self._on_time += self._run_dt
                self._on_gain += self._run_dp
            else:
                self._off_time += self._run_dt
                self._off_loss += self._run_dp
        self._run_rising = None
        self._run_dp = 0.0
        self._run_dt = 0.0

    def _track_fish(self, now: float, fish_x: Optional[float]) -> None:
        """Count movement onsets, which give the fish's resilience.

        A direction change is the cleanest marker available: the fish glides to
        a destination and the next movement usually reverses it. The estimate is
        coarse -- consecutive glides in the same direction read as one -- so it
        is used as a tempo hint and never as a control input.
        """
        if fish_x is None:
            return
        if self._fish_prev is None:
            self._fish_prev = fish_x
            return
        step = fish_x - self._fish_prev
        if abs(step) < 0.01:
            return
        direction = 1 if step > 0 else -1
        if self._fish_dir and direction != self._fish_dir:
            self._onsets.append(now)
            if len(self._onsets) >= 3:
                gaps = [b - a for a, b in zip(self._onsets, self._onsets[1:])]
                gaps.sort()
                median = gaps[len(gaps) // 2]
                self.state.resilience = phys.resilience_from_interval(median)
        self._fish_dir = direction
        self._fish_prev = fish_x

    def _publish(self) -> None:
        st = self.state

        # Include the run still in progress. A clean catch is one unbroken rise
        # that only ends when the fight does, so waiting for a direction change
        # to bank it means never measuring the gain rate on exactly the fights
        # that are going well.
        on_time, on_gain = self._on_time, self._on_gain
        off_time, off_loss = self._off_time, self._off_loss
        if self._run_rising is not None and self._run_dt >= MIN_RUN_SECONDS:
            if self._run_rising:
                on_time += self._run_dt
                on_gain += self._run_dp
            else:
                off_time += self._run_dt
                off_loss += self._run_dp

        if on_time >= self.min_state_seconds:
            st.gain_rate = max(0.0, on_gain / on_time)
            st.progress_speed = phys.progress_speed_from_gain(st.gain_rate)
        if off_time >= self.min_state_seconds:
            st.loss_rate = max(0.0, -off_loss / off_time)
        elif st.gain_rate is not None:
            # Loss does not scale with Progress Speed -- measured at 0.1186 and
            # 0.1179 in two independent fights while gain varied 0.084-0.178 --
            # so before any off-target time has been seen the documented rate is
            # a better estimate than nothing, and lets f_req be known from the
            # first seconds rather than only after the first mistake.
            st.loss_rate = phys.loss_rate()

    # -- verdict --------------------------------------------------------

    def verdict(self, grace_seconds: float = 4.0) -> str:
        """One of 'measuring', 'winning', 'struggling', 'lost'.

        'lost' is a claim about the fight, not the controller: at the coverage
        being achieved, progress trends to zero and more time cannot help. The
        grace period keeps the opening seconds -- where coverage is still being
        established -- from being read as a verdict.
        """
        st = self.state
        if st.elapsed < grace_seconds:
            return "measuring"

        # The decomposition into gain and loss needs sustained one-sided runs to
        # measure, and a fight being fought badly may not produce any -- which
        # is exactly when the verdict matters. The aggregate answers it without
        # them: progress going down over the whole fight means it is being lost,
        # whatever the component rates turn out to be.
        recent_net = self._recent_net_rate()
        if recent_net is not None and recent_net < 0:
            return "lost"
        if recent_net is None and st.net_rate is not None and st.net_rate < 0:
            return "lost"

        need = st.required_on_target
        if need is None:
            return "measuring"
        margin = st.on_target_fraction - need
        if margin > 0.05:
            return "winning"
        if margin > 0.0:
            return "struggling"
        return "lost"

    def _recent_net_rate(self) -> Optional[float]:
        """Progress per second across the recent window."""
        if len(self._recent_progress) < 2:
            return None
        (t0, p0), (t1, p1) = self._recent_progress[0], self._recent_progress[-1]
        return (p1 - p0) / (t1 - t0) if t1 - t0 > 1e-6 else None

    def stall_seconds(self, safety: float = 2.5, floor: float = 12.0) -> float:
        """How long to allow before treating a fight as stuck.

        Derived rather than fixed, because the honest range is wide: a neutral
        fish finishes in 8 s and a p25 fish needs 35 s, so one constant is either
        too tight for the slow fish or useless for the fast one.
        """
        projected = self.state.projected_seconds()
        if projected is None or projected == float("inf"):
            return max(floor, phys.catch_seconds(-80.0))
        return max(floor, self.state.elapsed + projected * safety)
