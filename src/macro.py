"""
Fisch Macro Engine — State Machine

Manages the full fishing loop:
IDLE → CASTING → WAITING → SHAKING → REELING → COMPLETE → CASTING...

Runs in a daemon thread so the GUI stays responsive.
Every state checks the killswitch on every iteration.
"""

import enum
import logging
import threading
from collections import deque
import time
from typing import Callable, Optional, List

from src.movement_tracker import MovementTracker
from src.vision import VisionSnapshot
from src.reel_controller import ControlParams, ReelController
from src.fight_estimator import FightEstimator

logger = logging.getLogger("macro")


class MacroState(enum.Enum):
    """States for the fishing macro state machine."""
    IDLE = "Idle"
    CASTING = "Casting"
    WAITING = "Waiting for Bite"
    SHAKING = "Shaking"
    REELING = "Reeling"
    COMPLETE = "Catch Complete"
    STOPPED = "Stopped"


class MacroStats:
    """Track session statistics."""

    def __init__(self):
        self.fish_caught: int = 0
        self.fish_failed: int = 0
        self.casts: int = 0
        self.perfect_catches: int = 0
        self.session_start: Optional[float] = None
        self._lock = threading.Lock()

    def reset(self):
        with self._lock:
            self.fish_caught = 0
            self.fish_failed = 0
            self.casts = 0
            self.perfect_catches = 0
            self.session_start = time.time()

    def record_catch(self, perfect: bool = False):
        with self._lock:
            self.fish_caught += 1
            if perfect:
                self.perfect_catches += 1

    def record_fail(self):
        with self._lock:
            self.fish_failed += 1

    def record_cast(self):
        with self._lock:
            self.casts += 1

    @property
    def session_duration(self) -> float:
        if self.session_start is None:
            return 0.0
        return time.time() - self.session_start

    @property
    def success_rate(self) -> float:
        total = self.fish_caught + self.fish_failed
        if total == 0:
            return 0.0
        return self.fish_caught / total


class MacroEngine:
    """
    Main macro engine that runs the fishing loop as a state machine.

    The engine runs in a daemon thread and communicates state changes
    to the GUI via callbacks.
    """

    # How much of the track's width the fish or the bar must cover across the
    # remembered ticks before a complete reading counts as a live minigame.
    # Small, because the point is only to tell movement from a still picture.
    STILL_READING_FRAMES = 5
    STILL_READING_EPSILON = 0.002

    def __init__(self, detector, controller, config_manager):
        """
        Args:
            detector: Detector instance for screen analysis
            controller: Controller instance for input simulation
            config_manager: ConfigManager for settings/profiles
        """
        self.detector = detector
        self.controller = controller
        self.config = config_manager

        self.state = MacroState.IDLE
        self.stats = MacroStats()

        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        # Callbacks for GUI updates
        self._state_callbacks: List[Callable[[MacroState], None]] = []
        self._stats_callbacks: List[Callable[[MacroStats], None]] = []
        self._log_callbacks: List[Callable[[str], None]] = []

        # Reeling state tracking
        self._reel_no_detection_count = 0
        self._reel_max_no_detection = 30  # ~1.5 seconds at 50ms intervals
        self._last_progress = 0.0
        self._progress_stuck_count = 0
        self.last_on_target = False
        self._last_fish_x = None
        self._fish_velocity = 0.0
        self._last_bar_center = None
        self._bar_velocity = 0.0
        self._last_catch_time = 0.0
        self._reeling_start_time = 0.0
        # Hysteresis counters for reeling exits
        self._fail_confirm_count = 0
        self._bar_gone_confirm_count = 0
        # Progress-based finish debounce (isolated from bar-gone to resist VFX flashes)
        self._progress_finish_count = 0
        self._progress_finish_low_count = 0
        self._peak_progress = 0.0
        # Action dwell / rate limiting for control stability
        self._last_action_time = 0.0
        self._last_action_type = None  # 'hold', 'release', 'rapid_click'
        self._last_shake_click_time = 0.0
        self._saw_midgame_progress = False
        self._prev_progress = 0.0
        self._fish_tracker = MovementTracker()
        self._reel_controller = ReelController()
        # Works out what kind of fight this is while fighting it: the gain and
        # loss rates, and so the on-target fraction this particular fish demands.
        self._estimator = FightEstimator()
        self._lost_fight_count = 0
        self._last_good_read_time = 0.0
        self._last_live_minigame_time = 0.0
        self._near_finish_streak = 0
        self._smoothed_progress = 0.0
        self._trusted_peak = 0.0
        self._raw_peak_progress = 0.0
        self._last_progress_gain_time = 0.0
        self._progress_collapse_count = 0
        self._reel_stall_count = 0
        self._post_catch_armed = True
        self._post_catch_clear_count = 0
        self._bite_confirm_count = 0
        self._expecting_bite_until = 0.0
        self._hunt_readings = deque(maxlen=self.STILL_READING_FRAMES)
        self.status_hint = "Idle"
        self._last_hunt_log_time = 0.0
        self._vision_lock = threading.Lock()
        self._vision_snapshot = VisionSnapshot()

    def get_vision_snapshot(self) -> VisionSnapshot:
        """Latest annotated frame + telemetry for the GUI live view."""
        with self._vision_lock:
            snap = self._vision_snapshot
            frame_copy = snap.frame_bgr.copy() if snap.frame_bgr is not None else None
            return VisionSnapshot(
                frame_bgr=frame_copy,
                fish_x=snap.fish_x,
                predicted_fish_x=snap.predicted_fish_x,
                bar_left=snap.bar_left,
                bar_right=snap.bar_right,
                bar_center=snap.bar_center,
                effective_bar=snap.effective_bar,
                progress_raw=snap.progress_raw,
                progress_smooth=snap.progress_smooth,
                peak_progress=snap.peak_progress,
                on_target=snap.on_target,
                pd_score=snap.pd_score,
                error=snap.error,
                fish_velocity=snap.fish_velocity,
                bar_velocity=snap.bar_velocity,
                last_action=snap.last_action,
                catch_allowed=snap.catch_allowed,
                macro_state=snap.macro_state,
                extras=dict(snap.extras),
            )

    def _publish_vision(self, frame_bgr, result, telemetry: dict) -> None:
        with self._vision_lock:
            self._vision_snapshot = VisionSnapshot(
                frame_bgr=frame_bgr,
                fish_x=result.fish_x,
                predicted_fish_x=telemetry.get("predicted_fish_x"),
                bar_left=result.bar_left,
                bar_right=result.bar_right,
                bar_center=telemetry.get("bar_center"),
                effective_bar=telemetry.get("effective_bar"),
                progress_raw=result.progress,
                progress_smooth=telemetry.get("progress_smooth", 0.0),
                peak_progress=telemetry.get("peak_progress", 0.0),
                on_target=result.on_target,
                pd_score=telemetry.get("pd_score", 0.0),
                error=telemetry.get("error", 0.0),
                fish_velocity=telemetry.get("fish_velocity", 0.0),
                bar_velocity=telemetry.get("bar_velocity", 0.0),
                last_action=telemetry.get("last_action", ""),
                catch_allowed=telemetry.get("catch_allowed", False),
                macro_state=self.state.value,
                extras=telemetry.get("extras", {}),
            )

    # ─── Callback Registration ────────────────────────────────────

    def on_state_change(self, callback: Callable[[MacroState], None]):
        """Register a callback for state changes."""
        self._state_callbacks.append(callback)

    def on_stats_update(self, callback: Callable[[MacroStats], None]):
        """Register a callback for stats updates."""
        self._stats_callbacks.append(callback)

    def on_log(self, callback: Callable[[str], None]):
        """Register a callback for log messages."""
        self._log_callbacks.append(callback)

    def _set_state(self, new_state: MacroState):
        """Update state and notify callbacks."""
        old_state = self.state
        self.state = new_state
        if old_state != new_state:
            logger.info(f"State: {old_state.value} → {new_state.value}")
            self._emit_log(f"State: {new_state.value}")
            for cb in self._state_callbacks:
                try:
                    cb(new_state)
                except Exception as e:
                    logger.error(f"State callback error: {e}")

    def _emit_stats(self):
        """Notify stats callbacks."""
        for cb in self._stats_callbacks:
            try:
                cb(self.stats)
            except Exception as e:
                logger.error(f"Stats callback error: {e}")

    def _emit_log(self, message: str):
        """Notify log callbacks."""
        for cb in self._log_callbacks:
            try:
                cb(message)
            except Exception as e:
                logger.error(f"Log callback error: {e}")

    # ─── Engine Control ───────────────────────────────────────────

    def start(self):
        """Start the macro engine in a daemon thread."""
        if self._thread is not None and self._thread.is_alive():
            logger.warning("Macro already running")
            return

        self._stop_event.clear()
        self.controller.start()
        self.stats.reset()

        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        logger.info("Macro engine started")
        self._emit_log("Macro started")

    def stop(self):
        """Stop the macro engine gracefully."""
        self._stop_event.set()
        self.controller.stop()
        self._set_state(MacroState.STOPPED)

        if self._thread is not None:
            self._thread.join(timeout=3.0)
            self._thread = None

        logger.info("Macro engine stopped")
        self._emit_log("Macro stopped")
        self._emit_stats()

    def is_running(self) -> bool:
        """Check if the macro is currently running."""
        return (
            self._thread is not None
            and self._thread.is_alive()
            and not self._stop_event.is_set()
        )

    def _should_stop(self) -> bool:
        """Check if we should stop (killswitch or manual stop)."""
        return self._stop_event.is_set() or self.controller.is_killed()

    def _stop_reason(self) -> str:
        """Return the current stop reason for logging."""
        if self._stop_event.is_set():
            return "stop requested"
        if self.controller.is_killed():
            return "emergency stop requested"
        return "none"

    # ─── Main Loop ────────────────────────────────────────────────

    def _run_loop(self):
        """Main macro loop — runs in daemon thread."""
        self._set_state(MacroState.CASTING)

        try:
            # Initial settings load
            settings = self.config.load_settings()
            last_settings_load = time.time()

            while not self._should_stop():
                # Refresh settings every 1 second instead of every iteration
                now = time.time()
                if now - last_settings_load > 1.0:
                    settings = self.config.load_settings()
                    last_settings_load = now

                interval = settings.scan_interval_ms / 1000.0

                if self.state == MacroState.CASTING:
                    self._do_casting(settings)
                elif self.state == MacroState.WAITING:
                    self._do_waiting(settings)
                elif self.state == MacroState.SHAKING:
                    self._do_shaking(settings)
                elif self.state == MacroState.REELING:
                    self._do_reeling(settings)
                elif self.state == MacroState.COMPLETE:
                    self._do_complete(settings)
                else:
                    time.sleep(interval)

                # If we are reeling, we want maximum responsiveness,
                # so we skip the extra sleep and rely on the scan_interval in the next loop.
                # For other states, a small sleep is fine.
                if self.state != MacroState.REELING:
                    time.sleep(interval)

        except Exception as e:
            logger.error(f"Macro loop error: {e}", exc_info=True)
            self._emit_log(f"Error: {e}")
        finally:
            self.controller.mouse_release(force=True)
            self._set_state(MacroState.STOPPED)
            reason = self._stop_reason()
            logger.info("Macro loop ended (%s)", reason)
            self._emit_log(f"Macro loop ended ({reason})")

    def _begin_reeling_session(self):
        """Reset per-catch tracking when entering the reeling minigame."""
        self._reel_no_detection_count = 0
        self._last_progress = 0.0
        self._peak_progress = 0.0
        self._progress_stuck_count = 0
        self._reeling_start_time = time.time()
        self._fail_confirm_count = 0
        self._bar_gone_confirm_count = 0
        self._progress_finish_count = 0
        self._progress_finish_low_count = 0
        self._last_fish_x = None
        self._last_bar_center = None
        self._fish_velocity = 0.0
        self._bar_velocity = 0.0
        self._last_action_time = 0.0
        self._last_action_type = None
        self._saw_midgame_progress = False
        self._prev_progress = 0.0
        self._fish_tracker.reset()
        self._last_good_read_time = 0.0

        # Rebuild the controller from settings each fight so tuning changes
        # take effect without a restart, and clear the vision's per-fight
        # memory (bar width prior, track extent) since the rod may have changed.
        settings = self.config.load_settings()
        self._reel_controller = ReelController(
            ControlParams(
                bar_accel=settings.control_bar_accel,
                latency_seconds=settings.control_latency_seconds,
                neutral_duty=settings.control_neutral_duty,
                duty_kp=settings.control_duty_kp,
                duty_ki=settings.control_duty_ki,
                max_brake_distance=settings.control_max_brake_distance,
                lead_seconds=settings.control_lead_seconds,
                stationary_speed=settings.control_stationary_speed,
            )
        )
        if hasattr(self.detector, "reset_session"):
            self.detector.reset_session()
        self._hunt_readings.clear()

        self._last_live_minigame_time = time.time()
        self._near_finish_streak = 0
        self._smoothed_progress = 0.0
        self._trusted_peak = 0.0
        self._raw_peak_progress = 0.0
        self._last_progress_gain_time = time.time()
        self._progress_collapse_count = 0
        self._reel_stall_count = 0
        self._estimator.reset()
        self._lost_fight_count = 0

    def _reset_post_catch_gate(self) -> None:
        """Require the minigame UI to clear before the next bite can start."""
        self._post_catch_armed = False
        self._post_catch_clear_count = 0
        self._bite_confirm_count = 0

    def _minigame_ui_clear(self, result) -> bool:
        """True when catch/reel UI is gone enough to hunt for a fresh bite."""
        if self._minigame_ready(result):
            return False
        if result.bite_confirmed:
            return False
        if result.progress >= 0.10:
            return False
        if result.fish_x is not None and result.bar_active:
            return False
        return True

    def _arm_for_cast_bite(self, settings) -> None:
        """After cast release, hunt immediately — do not wait for UI clear frames."""
        self._expecting_bite_until = time.time() + settings.post_cast_bite_window
        self._post_catch_armed = True
        self._post_catch_clear_count = settings.post_catch_clear_frames
        self._bite_confirm_count = 0

    def _expecting_cast_bite(self) -> bool:
        return time.time() < self._expecting_bite_until

    def _can_hunt_new_bite(self, result, settings) -> bool:
        if time.time() - self._last_catch_time < settings.post_catch_lockout_seconds:
            self._post_catch_clear_count = 0
            return False
        if self._expecting_cast_bite():
            return True
        if self._post_catch_armed:
            return True
        if self._minigame_ui_clear(result):
            self._post_catch_clear_count += 1
        else:
            self._post_catch_clear_count = 0
        if self._post_catch_clear_count >= settings.post_catch_clear_frames:
            self._post_catch_armed = True
        return self._post_catch_armed

    def _note_hunt_reading(self, result) -> None:
        """Remember what the reel ROI looked like this tick, while not fighting.

        Used only to tell a live minigame from a still picture of one; see
        :meth:`_reading_is_frozen`.
        """
        if result.fish_x is None or result.bar_left is None:
            self._hunt_readings.clear()
            return
        self._hunt_readings.append((result.fish_x, result.bar_left))

    def _reading_is_moving(self) -> bool:
        """True when the reel ROI has visibly changed over recent ticks.

        The reeling minigame is never still: the fish is driven continuously
        and the bar answers it, so consecutive frames always differ. Other
        screens are still, and some of them look enough like the minigame to
        pass for it -- the enchant panel draws a horizontal fill bar across the
        same rows the reel track occupies, and on tests/frames/STRUGGLE-ROD2 it
        yields a bar *and* a fish on 44% of the frames where no fight is
        running. Nothing in a single frame separates the two, but a fill bar
        that has not moved in a tenth of a second is not a fight.

        The cost on a real bite is at most one tick, since the fish is already
        moving when the minigame appears.
        """
        if len(self._hunt_readings) < 2:
            return False
        fish = [f for f, _ in self._hunt_readings]
        bar = [b for _, b in self._hunt_readings]
        return (
            max(fish) - min(fish) >= self.STILL_READING_EPSILON
            or max(bar) - min(bar) >= self.STILL_READING_EPSILON
        )

    def _reeling_trigger_strength(self, result, settings) -> int:
        """Higher = more confident the reeling minigame has started."""
        if self._minigame_ready(result):
            return 3
        if result.bite_confirmed and result.fish_x is not None:
            return 2
        if (
            result.fish_x is not None
            and result.bar_active
            and result.progress >= settings.bite_progress_threshold
        ):
            return 1
        return 0

    def _should_start_reeling(self, result, settings) -> bool:
        if (
            self._minigame_ready(result)
            and not result.bite_confirmed
            and not self._reading_is_moving()
        ):
            # A complete reading is otherwise enough on its own, which is what
            # makes a still picture of one dangerous. Positive evidence of
            # movement is required instead of absence of evidence of stillness,
            # because the fast path fires on the first frame and a stillness
            # test has nothing to look at yet. Costs one tick on a real bite;
            # bite_confirmed, which comes from elsewhere, still starts at once.
            self._bite_confirm_count = 0
            return False

        strength = self._reeling_trigger_strength(result, settings)
        if strength == 0:
            self._bite_confirm_count = 0
            return False
        if self._expecting_cast_bite() or strength >= 3:
            required = 1
        else:
            required = settings.bite_confirm_frames
        self._bite_confirm_count += 1
        return self._bite_confirm_count >= required

    @staticmethod
    def _fish_inside_bar(
        fish_x: Optional[float],
        bar_left: Optional[float],
        bar_right: Optional[float],
        margin: float = 0.02,
    ) -> Optional[bool]:
        if fish_x is None or bar_left is None or bar_right is None:
            return None
        return (bar_left + margin) <= fish_x <= (bar_right - margin)

    def _hunt_status_hint(self, result, settings) -> str:
        """Explain what the macro sees while waiting to bite."""
        if time.time() - self._last_catch_time < settings.post_catch_lockout_seconds:
            remain = settings.post_catch_lockout_seconds - (
                time.time() - self._last_catch_time
            )
            return f"Post-catch cooldown ({remain:.1f}s left)"

        strength = self._reeling_trigger_strength(result, settings)
        if strength > 0:
            if self._bite_confirm_count + 1 >= (
                1 if self._expecting_cast_bite() or strength >= 3 else settings.bite_confirm_frames
            ):
                return f"Bite detected (signal {strength}/3) — starting reel"
            return (
                f"Bite signal {strength}/3 — confirming "
                f"{self._bite_confirm_count + 1}/"
                f"{settings.bite_confirm_frames if strength < 3 and not self._expecting_cast_bite() else 1}"
            )

        if self._expecting_cast_bite():
            remain = self._expecting_bite_until - time.time()
            return f"Watching for instant bite after cast ({remain:.1f}s left)"

        if not self._post_catch_armed:
            return (
                f"Waiting for catch UI to clear "
                f"({self._post_catch_clear_count}/{settings.post_catch_clear_frames})"
            )

        parts = []
        if result.fish_x is not None:
            parts.append("fish")
        if result.bar_left is not None:
            parts.append("bar")
        if result.progress >= settings.bite_progress_threshold:
            parts.append(f"prog {result.progress:.0%}")
        if result.bite_confirmed:
            parts.append("bite_ok")
        if parts:
            return f"Hunting — sees {', '.join(parts)} (no full bite yet)"
        return "Hunting — no bite signal yet"

    def _publish_hunt_status(self, result, settings) -> None:
        self.status_hint = self._hunt_status_hint(result, settings)
        if not settings.show_live_vision:
            return
        geom = self._fish_inside_bar(result.fish_x, result.bar_left, result.bar_right)
        with self._vision_lock:
            self._vision_snapshot = VisionSnapshot(
                fish_x=result.fish_x,
                bar_left=result.bar_left,
                bar_right=result.bar_right,
                progress_raw=result.progress,
                on_target=result.on_target,
                macro_state=self.state.value,
                extras={
                    "hunt": self.status_hint,
                    "bite_confirmed": result.bite_confirmed,
                    "bar_active": result.bar_active,
                    "armed": self._post_catch_armed,
                    "expecting_cast_bite": self._expecting_cast_bite(),
                    "geom_on_target": geom,
                    "signal": self._reeling_trigger_strength(result, settings),
                },
            )

    def _maybe_log_hunt_status(self, settings) -> None:
        now = time.time()
        if now - self._last_hunt_log_time < 3.0:
            return
        self._last_hunt_log_time = now
        if self.state in (MacroState.WAITING, MacroState.CASTING, MacroState.COMPLETE):
            self._emit_log(f"📡 {self.status_hint}")

    def _try_start_reeling(self, result, settings, source: str = "") -> bool:
        self._note_hunt_reading(result)
        if not self._can_hunt_new_bite(result, settings):
            return False
        if not self._should_start_reeling(result, settings):
            return False
        label = f" ({source})" if source else ""
        logger.info("Minigame ready%s — fish on the line!", label)
        self._emit_log(f"🐟 Fish on the line!{label}")
        self._begin_reeling_session()
        self._set_state(MacroState.REELING)
        return True

    def _update_progress_tracking(self, raw_progress: float, settings) -> float:
        """Smooth progress and build a spike-resistant trusted peak."""
        alpha = settings.progress_smoothing
        if self._smoothed_progress <= 0.0 and raw_progress > 0.0:
            self._smoothed_progress = raw_progress
        else:
            self._smoothed_progress = (
                self._smoothed_progress * (1.0 - alpha)
            ) + (raw_progress * alpha)

        if raw_progress > self._raw_peak_progress + 0.008:
            self._last_progress_gain_time = time.time()
        self._raw_peak_progress = max(self._raw_peak_progress, raw_progress)

        cap = self._trusted_peak + settings.max_progress_tick
        if self._smoothed_progress <= cap + 0.001:
            self._trusted_peak = max(self._trusted_peak, self._smoothed_progress)

        # Finish gates use slow trusted peak; telemetry shows best-known fill level
        self._peak_progress = max(
            self._trusted_peak,
            self._smoothed_progress,
            self._raw_peak_progress,
        )
        return self._smoothed_progress

    def _minigame_ready(self, result) -> bool:
        """True when the full reeling UI is present (not cast VFX / slash effects)."""
        return (
            result.fish_x is not None
            and result.bar_left is not None
            and result.bar_right is not None
        )

    def _try_click_shake(self, settings, result) -> bool:
        """Click a visible SHAKE prompt if enabled and off cooldown."""
        if not settings.shake_enabled or result.shake_pos is None:
            return False
        confidence = getattr(result, "shake_confidence", 0.0)
        if confidence < settings.shake_min_confidence:
            logger.debug(
                "Shake ignored — confidence %.2f < %.2f",
                confidence,
                settings.shake_min_confidence,
            )
            return False
        now = time.time()
        if now - self._last_shake_click_time < settings.shake_click_cooldown_seconds:
            return False
        x, y = result.shake_pos
        self.controller.mouse_click(int(x), int(y))
        self._last_shake_click_time = now
        self._emit_log("Clicked SHAKE button")
        return True

    def _progress_finish_plausible(self, progress: float, max_progress_jump: float) -> bool:
        """Reject single-frame VFX spikes that jump progress unrealistically."""
        if self._last_progress < 0.75 and progress - self._last_progress > max_progress_jump:
            logger.debug(
                "Ignoring progress finish spike (%.2f -> %.2f)",
                self._last_progress,
                progress,
            )
            return False
        return True

    def _catch_allowed(self, settings, elapsed: float) -> bool:
        """Shared gate for any catch/fail completion during reeling."""
        if elapsed < settings.reeling_guard_seconds:
            return False
        if not self._saw_midgame_progress:
            return False
        if self._peak_progress < settings.min_midgame_progress:
            return False
        if elapsed >= settings.min_catch_seconds:
            return True
        return elapsed >= settings.fast_catch_min_seconds

    def _progress_finish_valid(
        self,
        settings,
        smooth_progress: float,
        max_progress_jump: float,
    ) -> bool:
        """True when smoothed progress looks like a real full catch, not mid-fight VFX."""
        if smooth_progress < settings.finish_progress_threshold:
            return False
        if not self._progress_finish_plausible(smooth_progress, max_progress_jump):
            return False
        if self._trusted_peak < settings.min_midgame_progress:
            return False
        if smooth_progress < self._trusted_peak - 0.15:
            return False
        if (
            self._trusted_peak < settings.finish_progress_threshold - 0.06
            and self._peak_progress < settings.finish_progress_threshold - 0.04
        ):
            return False
        return True

    def _progress_collapsed_likely_ended(self, result, settings) -> bool:
        """Fight ended: progress dropped after a real mid-fight fill (post-catch UI)."""
        if self._raw_peak_progress < settings.progress_collapse_min_peak:
            return False
        if result.progress > self._raw_peak_progress * 0.35:
            return False
        if (
            result.fish_x is not None
            and result.bar_left is not None
            and result.bar_right is not None
            and time.time() - self._last_live_minigame_time < 0.45
        ):
            return False
        return True

    def _reel_stall_likely_ended(self, result, settings, elapsed: float) -> bool:
        """Long fight with no progress gain — minigame likely finished but UI lingers.

        How long counts as long is measured, not fixed. A neutral fish finishes
        in 8 s, but the median fish's -40% Progress Speed makes it 12.5 s and a
        p25 fish's -80% makes it 35 s, so any single constant is either too
        tight for the slow fish or useless for the fast one. The estimator
        projects this fight's own finishing time from the rates it has observed,
        and the configured value becomes a floor rather than the rule.
        """
        if elapsed < settings.min_catch_seconds:
            return False
        if self._raw_peak_progress < settings.reel_stall_min_peak:
            return False
        stall_after = self._estimator.stall_seconds(floor=settings.reel_stall_seconds)
        if time.time() - self._last_progress_gain_time < stall_after:
            return False
        if result.progress > self._raw_peak_progress * 0.5:
            return False
        if (
            result.fish_x is not None
            and result.bar_left is not None
            and result.bar_right is not None
            and time.time() - self._last_live_minigame_time < 0.5
        ):
            return False
        return True

    def _bar_gone_likely_ended(self, result) -> bool:
        """Minigame UI is really gone — not a brief rod VFX overlay."""
        if result.bar_active:
            return False
        if result.fish_x is not None:
            return False
        if result.bite_confirmed:
            return False
        if time.time() - self._last_live_minigame_time < 0.35:
            return False
        return result.progress < max(0.10, self._peak_progress * 0.15)

    def _bar_gone_means_catch(self, settings) -> bool:
        """Bar disappeared after a near-complete reel, not mid-fight."""
        session_peak = max(self._peak_progress, self._raw_peak_progress)
        if session_peak < settings.peak_progress_catch_threshold:
            return False
        return self._prev_progress >= session_peak - settings.bar_gone_near_peak_delta

    def _chase_wants_hold(
        self,
        fish_x: float,
        bar_left: float,
        bar_right: float,
        bar_center: float,
        target_fish_x: float,
        off_state: bool,
        settings,
    ) -> bool:
        """Whether the bar should move right (hold) to chase the fish."""
        edge = settings.left_stall_bar_edge
        if off_state and bar_left < edge:
            return target_fish_x > bar_left + 0.05 or fish_x > bar_left + 0.04
        return target_fish_x >= bar_center

    def _should_bypass_action_cooldown(
        self,
        bar_left: float,
        fish_x: float,
        off_state: bool,
        settings,
    ) -> bool:
        """Recover from left-edge stalls even when dwell timer is active."""
        if not off_state or bar_left >= settings.left_stall_bar_edge:
            return False
        return fish_x > bar_left + 0.06

    def _update_velocities(
        self,
        fish_x: float,
        bar_center: float,
        fish_alpha: float,
        bar_alpha: float,
    ) -> None:
        if self._last_fish_x is not None:
            inst_fish_v = fish_x - self._last_fish_x
            self._fish_velocity = (self._fish_velocity * (1.0 - fish_alpha)) + (inst_fish_v * fish_alpha)
        self._last_fish_x = fish_x

        if self._last_bar_center is not None:
            inst_bar_v = bar_center - self._last_bar_center
            self._bar_velocity = (self._bar_velocity * (1.0 - bar_alpha)) + (inst_bar_v * bar_alpha)
        self._last_bar_center = bar_center

    def _fish_speed_per_second(self, settings) -> float:
        """Best estimate of how fast the fish is moving right now."""
        tick = max(settings.scan_interval_ms / 1000.0, 0.02)
        v_frame = abs(self._fish_velocity) / tick
        v_recent = self._fish_tracker.speed(settings.prediction_recent_window_seconds)
        return max(v_recent, v_frame)

    def _prediction_motion_scale(self, settings) -> float:
        """0 = fish treated as still; 1 = full prediction blend."""
        speed = self._fish_speed_per_second(settings)
        low = settings.prediction_stationary_speed
        high = low * 2.5
        if speed <= low:
            return 0.0
        if speed >= high:
            return 1.0
        return (speed - low) / (high - low)

    def _predicted_fish_x(self, fish_x: float, settings) -> float:
        motion = self._prediction_motion_scale(settings)
        if motion <= 0.0:
            return fish_x

        lookahead = settings.fish_prediction_ms / 1000.0
        use_accel = settings.prediction_use_acceleration and motion >= 0.35
        predicted = self._fish_tracker.predict(
            fish_x,
            lookahead,
            use_acceleration=use_accel,
            window_seconds=settings.prediction_recent_window_seconds,
        )
        weight = settings.prediction_weight * motion
        blended = (fish_x * (1.0 - weight)) + (predicted * weight)
        return max(0.0, min(1.0, blended))

    def _finish_catch(self, result, settings, reason: str) -> None:
        """Record a successful catch and transition to COMPLETE."""
        self.controller.mouse_release()
        perfect = (
            result.progress >= 0.995
            or self._peak_progress >= settings.finish_progress_threshold
        )
        self.stats.record_catch(perfect=perfect)
        self._emit_log(f"✅ Fish caught! ({reason})")
        self._emit_stats()
        self._last_catch_time = time.time()
        self._reset_post_catch_gate()
        self._set_state(MacroState.COMPLETE)

    def _arrival_lead_bias(
        self,
        fish_x: float,
        bar_center: float,
        settings,
    ) -> float:
        """digmacro-style lead: nudge control when fish is moving toward the bar."""
        if not settings.prediction_arrival_lead:
            return 0.0
        motion = self._prediction_motion_scale(settings)
        if motion < 0.25:
            return 0.0
        arrival = self._fish_tracker.arrival_seconds(bar_center, fish_x)
        if arrival is None:
            return 0.0
        lookahead = settings.fish_prediction_ms / 1000.0
        if arrival <= 0 or arrival > lookahead * 1.5:
            return 0.0
        v = self._fish_tracker.recent_velocity(settings.prediction_recent_window_seconds)
        direction = 1.0 if v > 0 else -1.0
        bias = direction * 0.04 * (1.0 - (arrival / max(lookahead, 1e-3)))
        return bias * motion

    def _effective_bar_center(self, bar_center: float, settings) -> float:
        """Bar center adjusted for coasting — won't instantly reverse after hold/release."""
        drift = self._bar_velocity * settings.bar_momentum_factor
        return max(0.0, min(1.0, bar_center + drift))

    # ─── State Handlers ───────────────────────────────────────────

    def _do_casting(self, settings):
        """
        Cast the fishing rod.

        Hold left-click for cast_hold_time seconds, then release.
        The power meter should land in the green zone with proper timing.
        """
        if self._should_stop():
            return

        self._set_state(MacroState.CASTING)
        self.stats.record_cast()
        self._emit_stats()
        self._emit_log("Casting rod...")

        # Hold mouse to charge cast
        self.controller.mouse_hold()

        # Wait for the cast hold time (checking killswitch periodically)
        elapsed = 0.0
        while elapsed < settings.cast_hold_time and not self._should_stop():
            result = self.detector.detect_all()
            if self._try_start_reeling(result, settings, "during cast"):
                self.controller.mouse_release()
                return
            time.sleep(0.05)
            elapsed += 0.05

        # Release to cast — instant-catch rods often bite in the same moment
        self.controller.mouse_release()
        self._arm_for_cast_bite(settings)

        if self._should_stop():
            return

        interval = max(0.02, settings.scan_interval_ms / 1000.0)
        post_cast = 0.0
        while post_cast < settings.post_cast_bite_window and not self._should_stop():
            result = self.detector.detect_all()
            self._publish_hunt_status(result, settings)
            if self._try_start_reeling(result, settings, "after cast"):
                return
            time.sleep(interval)
            post_cast += interval

        self._set_state(MacroState.WAITING)

    def _do_waiting(self, settings):
        """
        Wait for a fish to bite.

        Monitor for either:
        - SHAKE buttons appearing (transition to SHAKING)
        - Minigame bar appearing (transition to REELING)
        """
        if self._should_stop():
            return

        result = self.detector.detect_all()
        self._publish_hunt_status(result, settings)
        self._maybe_log_hunt_status(settings)

        if self._try_click_shake(settings, result):
            return

        self._try_start_reeling(result, settings)

    def _do_shaking(self, settings):
        """
        Handle the shake phase.

        Detect and click SHAKE buttons to speed up luring.
        Transition to REELING when the minigame bar appears.
        """
        if self._should_stop():
            return

        result = self.detector.detect_all()
        self._publish_hunt_status(result, settings)

        if self._try_click_shake(settings, result):
            return

        if self._try_start_reeling(result, settings):
            return

    def _do_reeling(self, settings):
        """
        Handle the reeling minigame.

        Core logic:
        - Detect fish indicator position
        - Detect control bar boundaries
        - Hold mouse to move bar RIGHT
        - Release mouse to move bar LEFT
        - Rapid click to hold position

        The fish indicator moves randomly; we chase it with the control bar.
        """
        if self._should_stop():
            return

        # Shake never appears mid-fight, and scanning for it costs more than
        # the rest of detection put together.
        result = self.detector.detect_all(scan_shake=False)
        prev_progress = self._prev_progress

        # Liveness, like bar_active, takes either element as proof the minigame
        # is still on screen. Demanding both made an ordinary fish dropout start
        # the clock ticking toward "the bar is gone".
        if (
            result.fish_x is not None
            or (result.bar_left is not None and result.bar_right is not None)
        ):
            self._last_live_minigame_time = time.time()

        success_confirm_frames = settings.success_confirm_frames
        fail_confirm_frames = settings.fail_confirm_frames
        bar_gone_confirm_frames = settings.bar_gone_confirm_frames
        max_progress_jump = settings.max_progress_jump
        progress_finish_reset_frames = settings.progress_finish_reset_frames
        elapsed = time.time() - self._reeling_start_time
        catch_allowed = False

        smooth_progress = self._update_progress_tracking(result.progress, settings)

        fight = self._estimator.update(
            time.time(),
            result.progress,
            bool(result.on_target),
            result.bar_left,
            result.bar_right,
            result.fish_x,
        )

        if elapsed >= settings.reeling_guard_seconds:
            if self._peak_progress >= settings.min_midgame_progress:
                self._saw_midgame_progress = True
            catch_allowed = self._catch_allowed(settings, elapsed)

        # ─── Progress-based success (debounced, VFX-resistant) ───────
        if catch_allowed:
            progress_finish_ok = self._progress_finish_valid(
                settings, smooth_progress, max_progress_jump
            )
            if progress_finish_ok:
                self._progress_finish_count += 1
                self._progress_finish_low_count = 0
                logger.debug(
                    "Progress finish confirmation: %d/%d consecutive",
                    self._progress_finish_count,
                    success_confirm_frames,
                )
                if self._progress_finish_count >= success_confirm_frames:
                    self._finish_catch(result, settings, "Progress Full")
                    return
            else:
                self._progress_finish_low_count += 1
                if self._progress_finish_low_count >= progress_finish_reset_frames:
                    self._progress_finish_count = 0

            if (
                smooth_progress >= settings.finish_progress_threshold - 0.02
                and self._peak_progress >= settings.finish_progress_threshold - 0.06
            ):
                self._near_finish_streak += 1
            else:
                self._near_finish_streak = 0

            if self._near_finish_streak >= settings.near_finish_confirm_frames:
                self._finish_catch(result, settings, "Progress Near Full")
                return

            if self._progress_collapsed_likely_ended(result, settings):
                self._progress_collapse_count += 1
                if (
                    self._progress_collapse_count
                    >= settings.progress_collapse_confirm_frames
                ):
                    self._finish_catch(result, settings, "Progress Collapsed")
                    return
            else:
                self._progress_collapse_count = 0

            if self._reel_stall_likely_ended(result, settings, elapsed):
                self._reel_stall_count += 1
                if self._reel_stall_count >= 8:
                    self._finish_catch(result, settings, "Reel Stall")
                    return
            else:
                self._reel_stall_count = 0

            # A fight can be unwinnable rather than merely slow. Progress is
            # gained at 12%/s scaled by the fish's Progress Speed but lost at a
            # flat 12%/s, so a fish at -80% needs the bar over it 83% of the
            # time; with a narrow rod against a fast fish that is not reachable,
            # and no extra minutes will change it. Once coverage has stayed
            # under what the fight demands, letting go and recasting beats
            # spending another half minute confirming it.
            if (
                elapsed >= settings.min_catch_seconds
                and self._estimator.verdict() == "lost"
                # Never let go while progress is still at its high-water mark.
                # The verdict looks at the last few seconds, so a fight that is
                # genuinely being won but has just given a little ground can
                # read as lost for a moment; requiring the fight to have
                # actually slipped costs nothing on a hopeless one, which is
                # well below its peak long before this is consulted.
                and fight.progress < fight.peak_progress - 0.05
            ):
                self._lost_fight_count += 1
                if self._lost_fight_count >= settings.lost_fight_confirm_frames:
                    logger.info(
                        "Abandoning: on-target %.0f%% against %.0f%% needed "
                        "(progress speed ~%+.0f%%)",
                        fight.on_target_fraction * 100.0,
                        (fight.required_on_target or 0.0) * 100.0,
                        fight.progress_speed or 0.0,
                    )
                    self._finish_catch(result, settings, "Fight Lost")
                    return
            else:
                self._lost_fight_count = 0
        else:
            self._progress_finish_count = 0
            self._progress_finish_low_count = 0
            self._near_finish_streak = 0
            self._progress_collapse_count = 0
            self._reel_stall_count = 0

        # ─── Failure: fish escaped (never started or fully drained) ─────
        fight_never_started = self._peak_progress < settings.min_midgame_progress
        if (
            not result.bite_confirmed
            and result.progress < 0.01
            and (fight_never_started or self._prev_progress < 0.05)
        ):
            if elapsed > 2.0:
                self._fail_confirm_count += 1
                logger.debug("Fail confirmation: %d/%d consecutive", self._fail_confirm_count, fail_confirm_frames)
                if self._fail_confirm_count >= fail_confirm_frames:
                    if catch_allowed or fight_never_started:
                        self.controller.mouse_release()
                        self.stats.record_fail()
                        self._emit_log("❌ Fish got away! (Progress Empty)")
                        self._emit_stats()
                        self._last_catch_time = time.time()
                        self._reset_post_catch_gate()
                        self._set_state(MacroState.COMPLETE)
                        return
                    logger.debug("Guard blocking failure exit (%.1fs)", elapsed)
                else:
                    self.controller.rapid_click(count=1, interval=0.0)
                    return
            else:
                self.controller.rapid_click(count=1, interval=0.0)
                return
        else:
            self._fail_confirm_count = 0

        # ─── Bar gone (minigame actually ended — not mid-fight VFX) ─────
        if catch_allowed and self._bar_gone_likely_ended(result):
            self._bar_gone_confirm_count += 1
            required_gone = bar_gone_confirm_frames
            if self._peak_progress < 0.85:
                required_gone = int(bar_gone_confirm_frames * 1.5)
            elif self._peak_progress >= 0.92:
                required_gone = max(8, bar_gone_confirm_frames // 2)
            logger.debug(
                "Bar-gone confirmation: %d/%d consecutive (peak=%.2f prev=%.2f)",
                self._bar_gone_confirm_count,
                required_gone,
                self._peak_progress,
                prev_progress,
            )
            if self._bar_gone_confirm_count >= required_gone:
                if self._bar_gone_means_catch(settings):
                    self._finish_catch(result, settings, "Bar Gone")
                else:
                    self.controller.mouse_release()
                    self.stats.record_fail()
                    self._emit_log("❌ Fish got away! (Bar Gone)")
                    self._emit_stats()
                    self._last_catch_time = time.time()
                    self._reset_post_catch_gate()
                    self._set_state(MacroState.COMPLETE)
                return
            self.controller.rapid_click(count=1, interval=0.0)
            return
        elif self._bar_gone_confirm_count > 0 and not self._bar_gone_likely_ended(result):
            self._bar_gone_confirm_count = max(0, self._bar_gone_confirm_count - 2)
        elif not catch_allowed:
            self._bar_gone_confirm_count = 0

        # Reset no-detection counter since something is visible
        self._reel_no_detection_count = 0

        # Track progress and target status
        self._prev_progress = result.progress
        self._last_progress = result.progress
        self.last_on_target = result.on_target

        # === Core reeling algorithm ===
        #
        # One switching law, applied unconditionally every tick. There is
        # deliberately no dwell timer, no cooldown and no early return here:
        # holding the mouse is a latching physical state, so any branch that
        # declines to act leaves the button *down* and the bar accelerating
        # right. That asymmetry is what pinned the bar against the wall.
        fish_x = result.fish_x
        bar_left = result.bar_left
        bar_right = result.bar_right

        if fish_x is None or bar_left is None or bar_right is None:
            # A dropout of a frame or two is common — VFX crossing the track, the
            # fish overlapping a bar glyph — and the fish cannot have moved far in
            # that time, so repeating the last command rides it out. Releasing
            # instead is not neutral: it accelerates the bar left, which turned
            # every brief dropout into a visible leftward twitch.
            #
            # The budget is wall-clock since the last *good* reading, not a count
            # of consecutive blind frames. Counting consecutively is wrong when
            # dropouts are interleaved with successes rather than contiguous: the
            # counter resets on every good frame, so it never reaches its limit
            # and the button ends up held almost continuously, driving the bar
            # into the right wall and parking it there.
            now = time.time()
            if self._last_good_read_time <= 0.0:
                self._last_good_read_time = now
            blind_for = now - self._last_good_read_time
            budget = max(0.0, settings.blind_coast_seconds)

            if blind_for <= budget and self._last_action_type == "hold":
                self.controller.mouse_hold()
            else:
                self.controller.mouse_release()
                self._last_action_type = "release"
            return

        self._last_good_read_time = time.time()

        bar_center = (bar_left + bar_right) / 2.0
        self._fish_tracker.add_sample(fish_x)

        decision = self._reel_controller.decide(fish_x, bar_center)

        if decision.hold:
            self.controller.mouse_hold()
        else:
            self.controller.mouse_release()

        action = decision.reason
        self._last_action_type = action
        self._last_action_time = time.time()

        # Kept for the GUI telemetry and the older exit heuristics below.
        self._fish_velocity = decision.fish_velocity
        self._bar_velocity = decision.bar_velocity
        self._last_fish_x = fish_x
        self._last_bar_center = bar_center
        target_fish_x = decision.fish_projected
        effective_bar = decision.bar_projected
        error = decision.error
        pd_score = error
        on_target = result.on_target
        off_state = not on_target

        if settings.show_live_vision:
            # detect_all already rendered the overlay from the frame it actually
            # analysed. Re-capturing settings.bar_roi here would show a stale
            # hand-calibrated rectangle rather than the auto-located track, so
            # the preview and the decisions could disagree.
            vis = result.debug_frame
            if vis is None and hasattr(self.detector, "capture_roi"):
                fallback = self.detector.capture_roi(settings.bar_roi)
                if fallback is not None:
                    vis = self.detector.get_debug_frame(
                        fallback,
                        result,
                        extras={
                            "predicted_fish_x": target_fish_x,
                            "effective_bar": effective_bar,
                            "progress_smooth": smooth_progress,
                            "macro_state": self.state.value,
                        },
                    )

            if vis is not None:
                self._publish_vision(
                    vis,
                    result,
                    {
                        "predicted_fish_x": target_fish_x,
                        "bar_center": bar_center,
                        "effective_bar": effective_bar,
                        "progress_smooth": smooth_progress,
                        "peak_progress": self._peak_progress,
                        "pd_score": pd_score,
                        "error": error,
                        "fish_velocity": self._fish_velocity,
                        "bar_velocity": self._bar_velocity,
                        "last_action": action,
                        "catch_allowed": catch_allowed,
                        "extras": {
                            "off_state": off_state,
                            "fish_outside": not on_target,
                            "geom_on_target": on_target,
                            "hold": decision.hold,
                            "vision_confidence": (
                                result.reading.confidence if result.reading else 0.0
                            ),
                            "fish_inside_bar": (
                                result.reading.fish_inside_bar if result.reading else False
                            ),
                        },
                    },
                )

    def _do_complete(self, settings):
        """
        Handle catch completion.

        Wait for the catch animation, then auto-recast if enabled.
        """
        if self._should_stop():
            return

        self._emit_log(f"Waiting {settings.recast_delay}s before recast...")
        self.status_hint = "Catch complete — waiting to recast"

        # Wait for recast delay; poll so post-catch gate can arm before cast
        elapsed = 0.0
        interval = max(0.05, settings.scan_interval_ms / 1000.0)
        while elapsed < settings.recast_delay and not self._should_stop():
            result = self.detector.detect_all()
            self._can_hunt_new_bite(result, settings)
            self._publish_hunt_status(result, settings)
            time.sleep(interval)
            elapsed += interval

        if self._should_stop():
            return

        if settings.auto_recast:
            self._set_state(MacroState.CASTING)
        else:
            self._set_state(MacroState.IDLE)
            self._emit_log("Auto-recast disabled — waiting in idle")
