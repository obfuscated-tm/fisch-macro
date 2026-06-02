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
import time
from typing import Callable, Optional, List

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
        self._success_confirm_count = 0
        self._fail_confirm_count = 0
        self._bar_gone_confirm_count = 0
        # Action dwell / rate limiting for control stability
        self._last_action_time = 0.0
        self._last_action_type = None  # 'hold', 'release', 'rapid_click'
        self._min_action_dwell = 0.08  # 80ms minimum between action changes

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
            self.controller.mouse_release()
            self._set_state(MacroState.STOPPED)
            reason = self._stop_reason()
            logger.info("Macro loop ended (%s)", reason)
            self._emit_log(f"Macro loop ended ({reason})")

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
            time.sleep(0.05)
            elapsed += 0.05

        # Release to cast
        self.controller.mouse_release()

        if self._should_stop():
            return

        # Brief wait for the cast animation
        time.sleep(0.8)

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
        
        # Post-catch lockout (ignore bites for 4s after a catch to clear animations/text)
        if time.time() - self._last_catch_time < 4.0:
            return

        if result.bite_confirmed:
            # Minigame bar detected — fish has bitten!
            logger.info("Bite confirmed — fish on the line!")
            self._emit_log("🐟 Fish on the line!")
            self._reel_no_detection_count = 0
            self._last_progress = 0.0
            self._progress_stuck_count = 0
            self._reeling_start_time = time.time()
            # Reset hysteresis counters when starting reeling
            self._success_confirm_count = 0
            self._fail_confirm_count = 0
            self._bar_gone_confirm_count = 0
            self._last_action_time = 0.0
            self._last_action_type = None
            self._set_state(MacroState.REELING)

        elif result.shake_pos is not None and settings.shake_enabled:
            # Shake button detected
            self._set_state(MacroState.SHAKING)

    def _do_shaking(self, settings):
        """
        Handle the shake phase.

        Detect and click SHAKE buttons to speed up luring.
        Transition to REELING when the minigame bar appears.
        """
        if self._should_stop():
            return

        result = self.detector.detect_all()
        
        # Post-catch lockout
        if time.time() - self._last_catch_time < 4.0:
            return

        if result.bite_confirmed:
            # Bar appeared — transition to reeling
            self._emit_log("🐟 Fish on the line!")
            self._reel_no_detection_count = 0
            self._last_progress = 0.0
            self._progress_stuck_count = 0
            # Reset hysteresis counters when starting reeling
            self._success_confirm_count = 0
            self._fail_confirm_count = 0
            self._bar_gone_confirm_count = 0
            self._last_action_time = 0.0
            self._last_action_type = None
            self._set_state(MacroState.REELING)
            return

        if result.shake_pos is not None:
            # Click the shake button
            x, y = result.shake_pos
            self.controller.mouse_click(int(x), int(y))
            self._emit_log("Clicked SHAKE button")
            time.sleep(0.15)  # Brief cooldown after clicking

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

        result = self.detector.detect_all()

        # ─── Reeling Guard: Block premature completion/failure ─────────
        REELING_GUARD_SECONDS = 1.5
        elapsed = time.time() - self._reeling_start_time
        if elapsed < REELING_GUARD_SECONDS:
            # During the guard window, still run the control loop but block exits
            # We'll skip the exit checks below by using a flag or early return pattern
            # For now, we'll let it fall through but modify exit logic to respect guard
            pass  # Will handle in exit logic below

        # Check if bar has disappeared (minigame over)
        # 1. Immediate End-Game Detection
        
        # Success: Progress bar reaches the end
        if result.progress > 0.98:
            self._success_confirm_count += 1
            logger.debug("Success confirmation: %d/3 consecutive", self._success_confirm_count)
            if self._success_confirm_count >= 3:
                # Guard check: only allow success if past guard window OR we have strong confirmation
                elapsed = time.time() - self._reeling_start_time
                if elapsed >= REELING_GUARD_SECONDS or self._success_confirm_count >= 5:  # Extra strict if in guard
                    self.controller.mouse_release()
                    self.stats.record_catch(perfect=(result.progress > 0.995))
                    self._emit_log("✅ Fish caught! (Progress Full)")
                    self._emit_stats()
                    self._last_catch_time = time.time()
                    self._set_state(MacroState.COMPLETE)
                    self._success_confirm_count = 0
                    return
                else:
                    logger.debug("Guard blocking success exit (%.1fs < %.1fs)", elapsed, REELING_GUARD_SECONDS)
            else:
                # Still play the control loop while confirming
                pass  # fall through to control logic
        else:
            self._success_confirm_count = 0  # reset on non-success readings

        # Failure: Progress reached zero and the bar disappeared
        # GRACE PERIOD: Ignore zero-progress failures in the first 2 seconds of reeling
        # to allow for intro animations and progress bar appearing.
        if not result.bite_confirmed and result.progress < 0.01:
            if time.time() - self._reeling_start_time > 2.0:
                self._fail_confirm_count += 1
                logger.debug("Fail confirmation: %d/5 consecutive", self._fail_confirm_count)
                if self._fail_confirm_count >= 5:
                    # Guard check: only allow failure if past guard window
                    elapsed = time.time() - self._reeling_start_time
                    if elapsed >= REELING_GUARD_SECONDS:
                        self.controller.mouse_release()
                        self.stats.record_fail()
                        self._emit_log("❌ Fish got away! (Progress Empty)")
                        self._emit_stats()
                        self._last_catch_time = time.time()
                        self._set_state(MacroState.COMPLETE)
                        self._fail_confirm_count = 0
                        return
                    else:
                        logger.debug("Guard blocking failure exit (%.1fs < %.1fs)", elapsed, REELING_GUARD_SECONDS)
                else:
                    # Hover during flicker/start while confirming failure
                    self.controller.rapid_click(count=1, interval=0.0)
                    return
            else:
                # Hover during flicker/start
                self.controller.rapid_click(count=1, interval=0.0)
                return
        else:
            self._fail_confirm_count = 0

        # Handle general bar disappearance (might be success text blocking progress bar)
        if not result.bar_active:
            self._bar_gone_confirm_count += 1
            logger.debug("Bar-gone confirmation: %d/15 consecutive", self._bar_gone_confirm_count)
            if self._bar_gone_confirm_count >= 15:
                # Guard check: only allow bar-gone exit if past guard window
                elapsed = time.time() - self._reeling_start_time
                if elapsed >= REELING_GUARD_SECONDS:
                    self.controller.mouse_release()
                    
                    # Success criteria fallback: 
                    # Progress WAS very high just before it disappeared
                    if self._last_progress > 0.90:
                        self.stats.record_catch(perfect=(self._last_progress > 0.98))
                        self._emit_log("✅ Fish caught! (Bar Gone)")
                    else:
                        self.stats.record_fail()
                        self._emit_log("❌ Fish got away! (Bar Gone)")
                    
                    self._last_catch_time = time.time()
                    self._emit_stats()
                    self._set_state(MacroState.COMPLETE)
                    self._bar_gone_confirm_count = 0
                    return
                else:
                    logger.debug("Guard blocking bar-gone exit (%.1fs < %.1fs)", elapsed, REELING_GUARD_SECONDS)
            else:
                # Hover during flickering while confirming bar gone
                self.controller.rapid_click(count=1, interval=0.0)
                return
        else:
            self._bar_gone_confirm_count = 0

        # Reset no-detection counter since something is visible
        self._reel_no_detection_count = 0

        # Track progress and target status
        self._last_progress = result.progress
        self.last_on_target = result.on_target

        # === Core reeling algorithm ===
        fish_x = result.fish_x
        bar_left = result.bar_left
        bar_right = result.bar_right
        on_target = result.on_target

        if fish_x is None or bar_left is None or bar_right is None:
            self.controller.rapid_click(count=1, interval=0.0)
            return

        # 1. Tracking and Velocities
        # Reduced smoothing (higher alpha) to reduce phase lag/delay
        alpha = 0.60
        
        # Fish velocity
        if self._last_fish_x is not None:
            inst_fish_v = (fish_x - self._last_fish_x)
            self._fish_velocity = (self._fish_velocity * (1.0 - alpha)) + (inst_fish_v * alpha)
        self._last_fish_x = fish_x
        
        # Bar velocity
        bar_center = (bar_left + bar_right) / 2.0
        if self._last_bar_center is not None:
            inst_bar_v = (bar_center - self._last_bar_center)
            self._bar_velocity = (self._bar_velocity * (1.0 - alpha)) + (inst_bar_v * alpha)
        self._last_bar_center = bar_center

        # 2. PD-Control Decision Logic (Proportional-Derivative)
        # error: how far the fish is from the bar center
        # velocity_diff: how fast the fish is moving relative to the bar
        error = fish_x - bar_center
        velocity_diff = self._fish_velocity - self._bar_velocity
        
        # LOWER GAINS to stop the swaying positive feedback loop
        kp = 0.5 
        kd = 3.5 
        
        # Boundary Awareness: Dampen derivative damping near walls to prevent 'bounce-panic'
        near_right_wall = bar_right > 0.96
        near_left_wall = bar_left < 0.04
        if (near_right_wall and self._bar_velocity > 0) or (near_left_wall and self._bar_velocity < 0):
            kd = 1.0 
        
        pd_score = (kp * error) + (kd * velocity_diff)
        
        # PD-score deadband: prevent tiny oscillations from flipping direction
        pd_deadband = 0.02
        
        # Deadzone based on target status
        deadzone = 0.05 if on_target else 0.01
        
        # Hysteresis: expand effective stable zone once settled
        if self._last_action_type == 'rapid_click':
            hysteresis_deadzone = deadzone * 1.5
        else:
            hysteresis_deadzone = deadzone
        
        # ─── Rate-limiting: Prevent action thrash ──────────────────────
        now = time.time()
        action_cooldown = (now - self._last_action_time < self._min_action_dwell)
        
        # 3. Decision Logic
        if near_right_wall and fish_x > bar_center:
            # At right limit and fish is right - stay pinned
            action = 'hold'
            if not (action_cooldown and action != self._last_action_type):
                self.controller.mouse_hold()
                self._last_action_time = time.time()
                self._last_action_type = action
        elif near_left_wall and fish_x < bar_center:
            # At left limit and fish is left - stay pinned
            action = 'release'
            if not (action_cooldown and action != self._last_action_type):
                self.controller.mouse_release()
                self._last_action_time = time.time()
                self._last_action_type = action
        elif fish_x > bar_right - 0.01:
            # Panic: Fish is escaping right
            action = 'hold'
            if not (action_cooldown and action != self._last_action_type):
                self.controller.mouse_hold()
                self._last_action_time = time.time()
                self._last_action_type = action
        elif fish_x < bar_left + 0.01:
            # Panic: Fish is escaping left
            action = 'release'
            if not (action_cooldown and action != self._last_action_type):
                self.controller.mouse_release()
                self._last_action_time = time.time()
                self._last_action_type = action
        elif abs(error) < hysteresis_deadzone and abs(self._bar_velocity) < 0.01:
            # Stable
            action = 'rapid_click'
            if not (action_cooldown and action != self._last_action_type):
                self.controller.rapid_click(count=1, interval=0.0)
                self._last_action_time = time.time()
                self._last_action_type = action
        elif pd_score > pd_deadband:
            # Need more upward/rightward force
            action = 'hold'
            if not (action_cooldown and action != self._last_action_type):
                if pd_score < 0.15:
                    self.controller.mouse_hold()
                    time.sleep(0.01) # Micro-pulse
                    self.controller.mouse_release()
                else:
                    self.controller.mouse_hold()
                self._last_action_time = time.time()
                self._last_action_type = action
        elif pd_score < -pd_deadband:
            # Need less force / let it fall
            action = 'release'
            if not (action_cooldown and action != self._last_action_type):
                if pd_score > -0.15:
                    self.controller.mouse_release()
                    time.sleep(0.01) # Micro-drift
                else:
                    self.controller.mouse_release()
                self._last_action_time = time.time()
                self._last_action_type = action
        else:
            # Hover
            action = 'rapid_click'
            if not (action_cooldown and action != self._last_action_type):
                self.controller.rapid_click(count=1, interval=0.0)
                self._last_action_time = time.time()
                self._last_action_type = action

    def _do_complete(self, settings):
        """
        Handle catch completion.

        Wait for the catch animation, then auto-recast if enabled.
        """
        if self._should_stop():
            return

        self._emit_log(f"Waiting {settings.recast_delay}s before recast...")

        # Wait for recast delay (check killswitch periodically)
        elapsed = 0.0
        while elapsed < settings.recast_delay and not self._should_stop():
            time.sleep(0.1)
            elapsed += 0.1

        if self._should_stop():
            return

        if settings.auto_recast:
            self._set_state(MacroState.CASTING)
        else:
            self._set_state(MacroState.IDLE)
            self._emit_log("Auto-recast disabled — waiting in idle")
