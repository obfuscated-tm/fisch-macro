"""
Configuration management for the Fisch macro.

Handles loading/saving settings, color profiles, and ROI bounds.
All configuration is stored as JSON files in the project directory.
"""

import json
import os
from dataclasses import dataclass, field, asdict
from typing import List, Tuple, Optional


@dataclass
class ColorProfile:
    """Color profile defining HSV ranges for detecting fishing bar elements.

    Each profile can be tuned for different rods or visual conditions.
    HSV values follow OpenCV convention: H=[0,179], S=[0,255], V=[0,255].
    """

    name: str = "default"
    fish_hsv_low: List[int] = field(default_factory=lambda: [140, 80, 100])
    fish_hsv_high: List[int] = field(default_factory=lambda: [170, 255, 255])
    bar_hsv_low: List[int] = field(default_factory=lambda: [40, 80, 100])
    bar_hsv_high: List[int] = field(default_factory=lambda: [90, 255, 255])
    bar_brightness_threshold: int = 80
    description: str = ""


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

        # Ensure the profiles directory exists
        os.makedirs(self.profiles_dir, exist_ok=True)

    def load_settings(self) -> Settings:
        """Load settings from settings.json.

        Returns:
            Settings object. If the file doesn't exist or is corrupt,
            returns default Settings.
        """
        if not os.path.exists(self.settings_path):
            return Settings()

        try:
            with open(self.settings_path, "r") as f:
                data = json.load(f)

            # Reconstruct nested ROIBounds from dicts
            if "bar_roi" in data and isinstance(data["bar_roi"], dict):
                data["bar_roi"] = ROIBounds(**data["bar_roi"])
            if "progress_roi" in data and isinstance(data["progress_roi"], dict):
                data["progress_roi"] = ROIBounds(**data["progress_roi"])
            if "shake_roi" in data and isinstance(data["shake_roi"], dict):
                data["shake_roi"] = ROIBounds(**data["shake_roi"])

            return Settings(**data)
        except (json.JSONDecodeError, TypeError, KeyError):
            return Settings()

    def save_settings(self, settings: Settings) -> None:
        """Save settings to settings.json.

        Args:
            settings: The Settings object to persist.
        """
        with open(self.settings_path, "w") as f:
            f.write(json.dumps(asdict(settings), indent=2))

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
            return ColorProfile(**data)
        except (json.JSONDecodeError, TypeError, KeyError):
            return ColorProfile(name=name)

    def save_profile(self, profile: ColorProfile) -> None:
        """Save a color profile to the profiles directory.

        Args:
            profile: The ColorProfile to persist.
        """
        profile_path = os.path.join(self.profiles_dir, f"{profile.name}.json")
        with open(profile_path, "w") as f:
            f.write(json.dumps(asdict(profile), indent=2))

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
