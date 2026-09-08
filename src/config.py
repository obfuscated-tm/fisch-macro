"""
Configuration management for the Fisch macro.

Handles loading/saving settings, color profiles, and ROI bounds.
All configuration is stored as JSON files in the project directory.
"""

import copy
import json
import logging
import os
from dataclasses import dataclass, field, asdict, fields
from typing import List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ROIBounds:
    """Region of Interest bounds as fractions of the Roblox window dimensions.

    Values are in [0.0, 1.0] representing percentages of window width/height.
    For example, x_start=0.28 means the ROI starts at 28% of the window width.
    """

    x_start: float = 0.28
    x_end: float = 0.72
    y_start: float = 0.81
    y_end: float = 0.87


@dataclass
class ColorProfile:
    """Rod profile defining HSV ranges and optional calibrated ROI bounds.

    HSV values follow OpenCV convention: H=[0,179], S=[0,255], V=[0,255].
    ROI fields are optional for backwards compatibility with older profiles.
    """

    name: str = "default"
    fish_hsv_low: List[int] = field(default_factory=lambda: [140, 80, 100])
    fish_hsv_high: List[int] = field(default_factory=lambda: [170, 255, 255])
    bar_hsv_low: List[int] = field(default_factory=lambda: [40, 80, 100])
    bar_hsv_high: List[int] = field(default_factory=lambda: [90, 255, 255])

    # New: Targeted color feedback for smarter behavior
    on_target_hsv_low: List[int] = field(default_factory=lambda: [40, 80, 100])   # Usually green
    on_target_hsv_high: List[int] = field(default_factory=lambda: [90, 255, 255])
    off_target_hsv_low: List[int] = field(default_factory=lambda: [15, 80, 100])  # Usually orange/white
    off_target_hsv_high: List[int] = field(default_factory=lambda: [35, 255, 255])

    bar_brightness_threshold: int = 80
    description: str = ""
    bar_roi: Optional[ROIBounds] = None
    progress_roi: Optional[ROIBounds] = None
    shake_roi: Optional[ROIBounds] = None


@dataclass
class Settings:
    """Global macro settings.

    Attributes:
        killswitch_key: Key to instantly stop the macro (default: F6).
        auto_recast: Whether to automatically re-cast after catching/losing a fish.
        cast_hold_time: How long to hold the mouse button when casting (seconds).
        recast_delay: Delay before re-casting after a catch (seconds).
        scan_interval_ms: Milliseconds between screen scans during fishing.
        active_profile: Name of the currently active color profile.
        bar_roi: ROI for the fishing bar slider area.
        progress_roi: ROI for the catch progress bar.
        shake_roi: ROI for shake/QTE detection area.
        shake_enabled: Whether shake detection is enabled.
        reeling_guard_seconds: Seconds to block premature completion/failure detection at start of reeling.
        success_confirm_frames: Consecutive frames needed to confirm progress-based success.
        fail_confirm_frames: Consecutive frames needed to confirm failure.
        bar_gone_confirm_frames: Consecutive frames needed to confirm bar disappearance.
        finish_progress_threshold: Progress fill ratio treated as a full catch.
        peak_progress_catch_threshold: Peak progress during reel to treat bar-gone as a catch.
        max_progress_jump: Ignore single-frame progress spikes larger than this (VFX).
        progress_finish_reset_frames: Consecutive sub-threshold frames before resetting progress finish debounce.
        min_catch_seconds: Minimum seconds in reeling before any catch can register.
        shake_click_cooldown_seconds: Minimum seconds between auto-shake clicks.
        shake_min_confidence: Minimum detection confidence (0–1) required to click SHAKE.
        shake_scan_every: Scan for SHAKE only every Nth tick; the ROI is huge and dominates tick cost.
        action_min_dwell_seconds: Minimum seconds between action changes for control stability.
        stable_hysteresis_multiplier: Multiplier for hysteresis deadzone when stable.
        pd_deadband: Deadband for PD controller score to prevent tiny oscillations.
        fish_prediction_ms: Milliseconds to predict fish position ahead (lookahead).
        fish_velocity_smoothing: EMA alpha for fish velocity (lower = more momentum lag).
        bar_velocity_smoothing: EMA alpha for bar velocity (lower = more drift after direction changes).
        bar_momentum_factor: How much bar drift offsets the target (0–1).
        prediction_weight: Blend of predicted vs current fish position (0–1).
        control_kp: Proportional gain for bar control.
        control_kd: Derivative gain for bar control.
        control_bar_drift_gain: Feed-forward gain compensating bar coasting.
        min_midgame_progress: Progress fill required before a catch can register.
        bar_gone_near_peak_delta: Bar-gone catch requires last progress within this of peak.
        roi_shift_x: Fine-tune all ROIs horizontally (normalized, positive = right).
        roi_shift_y: Fine-tune all ROIs vertically (normalized, positive = down).
        window_inset_top: Crop fraction from top of Roblox window for capture alignment.
        window_inset_left: Crop fraction from left of Roblox window for capture alignment.
        prediction_use_acceleration: Use acceleration term in fish lookahead (digmacro-style).
        prediction_arrival_lead: Bias control toward predicted arrival when fish is moving.
        near_finish_confirm_frames: Frames at ~full progress to catch even if bar still visible.
        show_live_vision: Show live detection preview on the Control tab.
        progress_smoothing: EMA alpha for progress (lower = smoother, resists gradient spikes).
        max_progress_tick: Max progress increase per frame for trusted peak tracking.
        off_target_chase_gain: Control gain multiplier when bar is off-target / fish outside bar.
        progress_collapse_min_peak: Raw peak progress required to treat a collapse as fight end.
        progress_collapse_confirm_frames: Consecutive collapse frames before completing catch.
        reel_stall_seconds: Seconds without progress gain before allowing stall-based completion.
            A floor: the actual bound is projected per fight from the measured progress rates,
            because catch time runs from 8s at a neutral fish to 35s at a slow one.
        lost_fight_confirm_frames: Frames of "this fight cannot be won" before abandoning it.
        reel_stall_min_peak: Raw peak required for stall-based completion.
        left_stall_bar_edge: Bar left edge below this triggers left-side recovery hold logic.
        prediction_stationary_speed: Below this speed (norm/s), prediction blend fades to zero.
        prediction_recent_window_seconds: Seconds of history used for recent velocity / prediction.
        post_catch_lockout_seconds: Ignore new bites briefly after a catch (lets UI clear).
        post_catch_clear_frames: Consecutive clear frames required before hunting a new bite.
        bite_confirm_frames: Frames to confirm a partial bite (fish without full bar yet).
        bite_progress_threshold: Progress fill that indicates a bite with fish visible.
        fast_catch_min_seconds: Minimum reeling time for fast/instant-catch rods (lower than min_catch_seconds).
        post_cast_bite_window: Seconds after cast release to aggressively scan for an instant bite.
        auto_locate_track: Search for the track instead of trusting bar_roi. Off by default — see src/detector.py.
        track_search_top: Fraction of window height below which to search for the track.
        track_relocate_every: Ticks between full track re-locations during a fight.
        track_search_expand: Vertical search half-height, as a multiple of the calibrated ROI height.
        control_bar_accel: Bar acceleration in track widths/s^2, measured by scripts/measure_physics.py.
        control_latency_seconds: Capture-plus-input latency, applied as a linear lead on the bar.
        blind_coast_seconds: Wall-clock budget for repeating the last command through a detection dropout.
        fish_relative_strength: Reject fish candidates weaker than this fraction of the tracked fish.
        fish_max_speed: Largest believable fish speed (track widths/s); bigger jumps are rejected.
        control_neutral_duty: Hold fraction that keeps the bar still; follows from the measured accelerations.
        control_duty_kp: Duty change per track width of error. Higher tracks harder but oscillates more.
        control_duty_ki: Integral gain trimming a wrong neutral duty for a given rod.
        control_max_brake_distance: Bound on the v^2 braking projection, so one bad edge cannot lurch the bar.
        control_lead_seconds: Fish velocity lookahead for the switching decision.
        control_stationary_speed: Below this fish speed, no lead is applied.
"""

    killswitch_key: str = "f6"
    auto_recast: bool = True
    cast_hold_time: float = 2.0
    recast_delay: float = 2.0
    scan_interval_ms: int = 50
    active_profile: str = "default"
    bar_roi: ROIBounds = field(default_factory=ROIBounds)
    progress_roi: ROIBounds = field(default_factory=lambda: ROIBounds(0.28, 0.72, 0.87, 0.90))
    shake_roi: ROIBounds = field(default_factory=lambda: ROIBounds(0.10, 0.90, 0.20, 0.70))
    shake_enabled: bool = True
    reeling_guard_seconds: float = 2.5
    success_confirm_frames: int = 6
    fail_confirm_frames: int = 5
    bar_gone_confirm_frames: int = 12
    finish_progress_threshold: float = 0.96
    peak_progress_catch_threshold: float = 0.88
    max_progress_jump: float = 0.35
    progress_finish_reset_frames: int = 2
    min_catch_seconds: float = 5.0
    shake_click_cooldown_seconds: float = 0.12
    shake_min_confidence: float = 0.42
    shake_scan_every: int = 4
    action_min_dwell_seconds: float = 0.08
    stable_hysteresis_multiplier: float = 1.5
    pd_deadband: float = 0.02
    fish_prediction_ms: float = 80.0
    fish_velocity_smoothing: float = 0.35
    bar_velocity_smoothing: float = 0.25
    bar_momentum_factor: float = 0.55
    prediction_weight: float = 0.65
    control_kp: float = 0.28
    control_kd: float = 1.2
    control_bar_drift_gain: float = 0.06
    min_midgame_progress: float = 0.35
    bar_gone_near_peak_delta: float = 0.12
    roi_shift_x: float = 0.0
    roi_shift_y: float = 0.0
    window_inset_top: float = 0.0
    window_inset_left: float = 0.0
    prediction_use_acceleration: bool = True
    prediction_arrival_lead: bool = True
    near_finish_confirm_frames: int = 10
    show_live_vision: bool = True
    progress_smoothing: float = 0.35
    max_progress_tick: float = 0.06
    off_target_chase_gain: float = 1.45
    progress_collapse_min_peak: float = 0.55
    progress_collapse_confirm_frames: int = 5
    reel_stall_seconds: float = 18.0

    # Consecutive frames the estimator must call a fight lost before letting go.
    # High, because abandoning a catchable fish costs more than sitting on an
    # uncatchable one: the verdict is only consulted after min_catch_seconds and
    # it takes several seconds of coverage below what the fight needs to reach.
    lost_fight_confirm_frames: int = 30
    reel_stall_min_peak: float = 0.60
    left_stall_bar_edge: float = 0.12
    prediction_stationary_speed: float = 0.04
    prediction_recent_window_seconds: float = 0.2
    post_catch_lockout_seconds: float = 1.5
    post_catch_clear_frames: int = 5
    bite_confirm_frames: int = 2
    bite_progress_threshold: float = 0.08
    fast_catch_min_seconds: float = 1.0
    post_cast_bite_window: float = 2.5

    # --- structure-based reel vision (see src/reel_vision.py) ---
    auto_locate_track: bool = False
    track_search_top: float = 0.60
    track_relocate_every: int = 12
    track_search_expand: float = 3.0
    # --- switching-law control (see src/reel_controller.py) ---
    control_bar_accel: float = 1.0
    control_latency_seconds: float = 0.06
    blind_coast_seconds: float = 0.12
    fish_relative_strength: float = 0.45
    fish_max_speed: float = 4.0
    control_neutral_duty: float = 0.538
    control_duty_kp: float = 10.0
    control_duty_ki: float = 0.6
    control_max_brake_distance: float = 0.35
    control_lead_seconds: float = 0.07
    control_stationary_speed: float = 0.05


class ConfigManager:
    """Manages loading and saving of settings and color profiles.

    All data is stored as JSON files:
      - settings.json in the base directory
      - profiles/*.json for color profiles

    Args:
        base_dir: Project root directory containing settings.json and profiles/.
    """

    def __init__(self, base_dir: str):
        self.base_dir = base_dir
        self.settings_path = os.path.join(base_dir, "settings.json")
        self.profiles_dir = os.path.join(base_dir, "profiles")

        # load_settings() is called several times per tick — by the macro loop,
        # by detect_all, and again by every ROI computation. Re-reading and
        # re-parsing JSON from disk that often is pure overhead on a loop that
        # is already tight, so the parse is cached and invalidated by mtime.
        # The cache entry itself is never handed out; see load_settings.
        self._settings_cache: Optional[Settings] = None
        self._settings_stamp: Optional[tuple] = None
        self._warned_unknown: Optional[tuple] = None

        # Ensure the profiles directory exists
        os.makedirs(self.profiles_dir, exist_ok=True)

    def load_settings(self) -> Settings:
        """Load settings from settings.json.

        Unknown keys are dropped with a warning rather than being allowed to
        fail the load. Previously any key that no longer matched a field — left
        behind by a rename, or written by an older build — raised TypeError from
        the constructor, and the handler quietly returned a default Settings.

        The result was silent and total: one stale key reverted the ROI to a
        rectangle that pointed nowhere near the reel bar, re-enabled an
        expensive scan that quadrupled the tick time, and reset every timing
        threshold, with nothing logged to say so. Losing one setting is a much
        smaller problem than losing all of them, so partial recovery is the
        right behaviour here — and it says what it dropped.

        Every caller gets its own copy, because several of them keep the
        object and edit it: the GUI holds one for the lifetime of the window,
        and the calibration overlay holds another while it is open. Handing
        out the cached instance made those the *same* object, so edits leaked
        between them — cancelling the calibrator still left its half-drawn ROI
        in the GUI's copy, to be written to disk by the next autosave, and a
        stale holder could write its whole snapshot back over a save someone
        else had just made. Copying is ~13us against a tick budget of 20ms.

        Returns:
            Settings object, falling back to defaults only if the file is
            missing or genuinely unreadable.
        """
        if not os.path.exists(self.settings_path):
            return Settings()

        try:
            stat = os.stat(self.settings_path)
            stamp = (stat.st_mtime_ns, stat.st_size)
        except OSError:
            stamp = None
        if stamp is not None and stamp == self._settings_stamp:
            return copy.deepcopy(self._settings_cache)

        try:
            with open(self.settings_path, "r") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError) as exc:
            logger.error(
                "settings.json could not be read (%s) — using defaults. "
                "Calibration and tuning will not be applied.", exc,
            )
            return Settings()

        if not isinstance(data, dict):
            logger.error("settings.json is not an object — using defaults.")
            return Settings()

        for key in ("bar_roi", "progress_roi", "shake_roi"):
            if isinstance(data.get(key), dict):
                try:
                    data[key] = ROIBounds(**data[key])
                except TypeError as exc:
                    logger.warning("Ignoring malformed %s in settings.json: %s", key, exc)
                    data.pop(key)

        known = {f.name for f in fields(Settings)}
        unknown = tuple(sorted(set(data) - known))
        if unknown and unknown != self._warned_unknown:
            # Said once per change, not once per tick.
            logger.warning(
                "Ignoring %d unrecognised setting(s) in settings.json: %s",
                len(unknown), ", ".join(unknown),
            )
        self._warned_unknown = unknown
        accepted = {k: v for k, v in data.items() if k in known}

        try:
            settings = Settings(**accepted)
        except TypeError as exc:
            logger.error(
                "settings.json could not be applied (%s) — using defaults.", exc
            )
            return Settings()

        self._settings_cache = settings
        self._settings_stamp = stamp
        return copy.deepcopy(settings)

    def save_settings(self, settings: Settings) -> None:
        """Save settings to settings.json.

        The cache is primed with what was written rather than left to be
        invalidated by the next stat, so a reader that arrives before the
        filesystem timestamp has moved on still sees the new values.

        Args:
            settings: The Settings object to persist.
        """
        with open(self.settings_path, "w") as f:
            f.write(json.dumps(asdict(settings), indent=2))

        self._settings_cache = copy.deepcopy(settings)
        try:
            stat = os.stat(self.settings_path)
            self._settings_stamp = (stat.st_mtime_ns, stat.st_size)
        except OSError:
            self._settings_stamp = None

    def load_profile(self, name: str) -> ColorProfile:
        """Load a color profile by name.

        Args:
            name: Profile name (without .json extension).

        Returns:
            ColorProfile loaded from disk, or a default profile if the file
            doesn't exist.
        """
        profile_path = os.path.join(self.profiles_dir, f"{name}.json")

        if not os.path.exists(profile_path):
            return ColorProfile(name=name)

        try:
            with open(profile_path, "r") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError) as exc:
            logger.error("Profile %r could not be read (%s) — using defaults.", name, exc)
            return ColorProfile(name=name)

        if not isinstance(data, dict):
            logger.error("Profile %r is not an object — using defaults.", name)
            return ColorProfile(name=name)

        for key in ("bar_roi", "progress_roi", "shake_roi"):
            if isinstance(data.get(key), dict):
                try:
                    data[key] = ROIBounds(**data[key])
                except TypeError as exc:
                    logger.warning("Ignoring malformed %s in profile %r: %s", key, name, exc)
                    data.pop(key)

        # Same reasoning as load_settings: drop unknown keys rather than let one
        # of them discard the whole profile.
        known = {f.name for f in fields(ColorProfile)}
        unknown = sorted(set(data) - known)
        if unknown:
            logger.warning(
                "Ignoring %d unrecognised key(s) in profile %r: %s",
                len(unknown), name, ", ".join(unknown),
            )

        try:
            return ColorProfile(**{k: v for k, v in data.items() if k in known})
        except TypeError as exc:
            logger.error("Profile %r could not be applied (%s) — using defaults.", name, exc)
            return ColorProfile(name=name)

    def save_profile(self, profile: ColorProfile) -> None:
        """Save a color profile to the profiles directory.

        Args:
            profile: The ColorProfile to persist.
        """
        profile_path = os.path.join(self.profiles_dir, f"{profile.name}.json")
        with open(profile_path, "w") as f:
            f.write(json.dumps(asdict(profile), indent=2))

    def delete_profile(self, name: str) -> bool:
        """Delete a color profile by name.

        Args:
            name: Profile name (without .json extension).

        Returns:
            True if deleted, False otherwise.
        """
        if name == "default":
            return False  # Prevent deleting the default profile

        profile_path = os.path.join(self.profiles_dir, f"{name}.json")
        if os.path.exists(profile_path):
            try:
                os.remove(profile_path)
                return True
            except OSError:
                return False
        return False

    def list_profiles(self) -> List[str]:
        """List all available profile names.

        Returns:
            List of profile names (without .json extension), sorted alphabetically.
        """
        profiles = []
        if os.path.isdir(self.profiles_dir):
            for fname in os.listdir(self.profiles_dir):
                if fname.endswith(".json"):
                    profiles.append(fname[:-5])
        return sorted(profiles)

    def get_active_profile(self) -> ColorProfile:
        """Load the currently active color profile.

        Reads the active profile name from settings, then loads that profile.

        Returns:
            The active ColorProfile.
        """
        settings = self.load_settings()
        return self.load_profile(settings.active_profile)
