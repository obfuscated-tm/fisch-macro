"""
Window tracking for locating the game window across platforms.

macOS uses Quartz, Windows uses the Win32 window list, and Linux uses X11 —
each behind the same `WindowTracker` interface, so callers never branch on
platform.
"""

from src.window_tracker.base import DEFAULT_APP_NAMES, WindowBackend, WindowBounds
from src.window_tracker.tracker import WindowTracker, create_backend, report_backend

__all__ = [
    "DEFAULT_APP_NAMES",
    "WindowBackend",
    "WindowBounds",
    "WindowTracker",
    "create_backend",
    "report_backend",
]
