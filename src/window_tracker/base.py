"""
Shared types and helpers for the platform window-tracking backends.

Each backend locates the game window and reports the display scale factor
for its own OS. The rest of the codebase only ever sees `WindowBounds` and
`WindowTracker`, so adding a platform means adding a backend here — not
touching the detector, the GUI or the calibrator.
"""

import logging
from dataclasses import dataclass
from typing import Optional, Sequence

logger = logging.getLogger(__name__)


# Window owner/title fragments that identify the game, matched
# case-insensitively as substrings. "Sober" is the Linux Roblox client;
# "Vinegar" is its predecessor.
DEFAULT_APP_NAMES = ("Roblox", "Sober", "Vinegar")


@dataclass
class WindowBounds:
    """Bounds of a window on screen in pixel coordinates.

    Attributes:
        x: Left edge of the window.
        y: Top edge of the window.
        width: Width of the window in pixels.
        height: Height of the window in pixels.
    """

    x: int
    y: int
    width: int
    height: int


def matches_app_name(candidate: Optional[str], app_names: Sequence[str]) -> bool:
    """Return True if `candidate` contains any of `app_names`, ignoring case."""
    if not candidate:
        return False
    lowered = candidate.lower()
    return any(name.lower() in lowered for name in app_names)


def scale_from_capture_vs_logical() -> Optional[float]:
    """Derive the scale factor by comparing capture pixels to logical pixels.

    `mss` reports the primary monitor in physical pixels; `pyautogui` reports
    it in the logical coordinates that window positions and clicks use. Their
    ratio is exactly the factor the detector needs to turn a window rect into
    a capture rect, which makes this correct on every platform by
    construction — 2.0 on a Retina Mac, 1.0 on a DPI-aware Windows box.

    Returns:
        The scale factor, or None if it could not be determined or the result
        failed the sanity check.
    """
    try:
        import mss
        import pyautogui

        with mss.mss() as sct:
            # monitors[0] is the union of all displays; monitors[1] is primary.
            monitors = sct.monitors
            if len(monitors) < 2:
                return None
            physical_width = monitors[1]["width"]

        logical_width = pyautogui.size()[0]
        if logical_width <= 0 or physical_width <= 0:
            return None

        scale = physical_width / logical_width
        # Real-world scale factors are 1.0, 1.25, 1.5 or 2.0. Anything outside
        # this band means we compared two unrelated numbers.
        if 0.5 <= scale <= 4.0:
            return scale
    except ImportError:
        logger.debug("mss/pyautogui unavailable for scale factor detection")
    except Exception as e:
        logger.debug("Capture-vs-logical scale detection failed: %s", e)

    return None


class WindowBackend:
    """Locates the game window on one platform.

    Subclasses override `find_window`, and may override `detect_scale_factor`
    when the OS exposes something better than the generic ratio.
    """

    #: Human-readable backend name, used in log messages.
    name = "unknown"

    #: Scale factor assumed when detection fails outright.
    default_scale = 1.0

    def find_window(self, app_names: Sequence[str]) -> Optional[WindowBounds]:
        """Return the bounds of the first window matching `app_names`."""
        raise NotImplementedError

    def detect_scale_factor(self) -> Optional[float]:
        """Return the display scale factor, or None if it is unknown."""
        return scale_from_capture_vs_logical()

    def diagnostics(self) -> Optional[str]:
        """Return a hint about why lookups may be failing, if one applies."""
        return None
