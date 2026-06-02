"""
Window tracker for locating the Roblox window on macOS.

Uses macOS Quartz APIs to find the Roblox window position and size,
and to determine the display scale factor (Retina vs non-Retina).
Falls back gracefully when Quartz is unavailable.
"""

import logging
import time
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


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


class WindowTracker:
    """Tracks the Roblox window position and display scale factor.

    Caches the window bounds to avoid querying the window list on every frame.
    Automatically refreshes the cache after `cache_ttl` seconds.
    """

    def __init__(self):
        self._cached_bounds: Optional[WindowBounds] = None
        self._cache_time: float = 0.0
        self.cache_ttl: float = 2.0
        self._scale_factor: Optional[float] = None

    def get_roblox_bounds(self) -> Optional[WindowBounds]:
        """Find the Roblox window bounds on screen.

        Uses macOS Quartz CGWindowListCopyWindowInfo to enumerate all on-screen
        windows and find the one owned by "Roblox".

        Returns:
            WindowBounds if the Roblox window is found, None otherwise.
            Returns cached bounds if the cache is still fresh.
        """
        # Return cached bounds if still valid
        now = time.time()
        if self._cached_bounds is not None and (now - self._cache_time) < self.cache_ttl:
            return self._cached_bounds

        try:
            from Quartz import (
                CGWindowListCopyWindowInfo,
                kCGWindowListOptionOnScreenOnly,
                kCGNullWindowID,
            )
        except ImportError:
            logger.warning(
                "Quartz framework not available. Cannot locate Roblox window. "
                "Install pyobjc-framework-Quartz: pip install pyobjc-framework-Quartz"
            )
            return None

        window_list = CGWindowListCopyWindowInfo(
            kCGWindowListOptionOnScreenOnly, kCGNullWindowID
        )

        if window_list is None:
            logger.debug("CGWindowListCopyWindowInfo returned None")
            return None

        for window in window_list:
            owner_name = window.get("kCGWindowOwnerName", "")
            if owner_name and "Roblox" in owner_name:
                bounds_dict = window.get("kCGWindowBounds")
                if bounds_dict is None:
                    continue

                bounds = WindowBounds(
                    x=int(bounds_dict.get("X", 0)),
                    y=int(bounds_dict.get("Y", 0)),
                    width=int(bounds_dict.get("Width", 0)),
                    height=int(bounds_dict.get("Height", 0)),
                )

                # Skip windows with zero area (menus, helper windows, etc.)
                if bounds.width <= 0 or bounds.height <= 0:
                    continue

                self._cached_bounds = bounds
                self._cache_time = now
                logger.debug(
                    "Found Roblox window: x=%d, y=%d, w=%d, h=%d",
                    bounds.x, bounds.y, bounds.width, bounds.height,
                )
                return bounds

        logger.debug("Roblox window not found among on-screen windows")
        return None

    def get_scale_factor(self) -> float:
        """Determine the display scale factor (1 for standard, 2 for Retina).

        Compares the physical pixel width of the main display to the logical
        width reported by pyautogui. Falls back to parsing system_profiler
        output, or defaults to 2 (safest for Retina Macs).

        Returns:
            Display scale factor as a float (typically 1.0 or 2.0).
        """
        if self._scale_factor is not None:
            return self._scale_factor

        # Method 1: Quartz API
        try:
            from Quartz import CGMainDisplayID, CGDisplayPixelsWide
            import pyautogui

            physical_width = CGDisplayPixelsWide(CGMainDisplayID())
            logical_width = pyautogui.size()[0]

            if logical_width > 0:
                scale = physical_width / logical_width
                # Sanity check: scale should be 1 or 2 (or occasionally 1.5)
                if 0.5 <= scale <= 4.0:
                    self._scale_factor = scale
                    logger.info("Display scale factor (Quartz): %.2f", scale)
                    return scale
        except ImportError:
            logger.debug("Quartz not available for scale factor detection")
        except Exception as e:
            logger.debug("Quartz scale factor detection failed: %s", e)

        # Method 2: system_profiler fallback
        try:
            import subprocess

            result = subprocess.run(
                ["system_profiler", "SPDisplaysDataType"],
                capture_output=True,
                text=True,
                timeout=5,
            )

            if result.returncode == 0:
                output = result.stdout
                # Look for "Retina" in the output as a strong indicator
                if "Retina" in output:
                    self._scale_factor = 2.0
                    logger.info("Display scale factor (system_profiler): 2.0 (Retina detected)")
                    return 2.0

                # Try to parse resolution lines for ratio
                # Typical format: "Resolution: 2560 x 1600 Retina"
                # or "Resolution: 1440 x 900"
                import re

                resolutions = re.findall(r"Resolution:\s*(\d+)\s*x\s*(\d+)", output)
                if len(resolutions) >= 2:
                    # If two resolution lines, the ratio might indicate scaling
                    w1 = int(resolutions[0][0])
                    w2 = int(resolutions[1][0])
                    if w1 > 0 and w2 > 0:
                        ratio = max(w1, w2) / min(w1, w2)
                        if 1.5 <= ratio <= 2.5:
                            self._scale_factor = 2.0
                            logger.info("Display scale factor (resolution ratio): 2.0")
                            return 2.0
        except Exception as e:
            logger.debug("system_profiler scale factor detection failed: %s", e)

        # Default: assume Retina (safer — avoids undersized captures)
        self._scale_factor = 2.0
        logger.warning(
            "Could not determine display scale factor. Defaulting to 2.0 (Retina)."
        )
        return 2.0

    def invalidate_cache(self) -> None:
        """Clear the cached window bounds, forcing a fresh lookup on next call."""
        self._cached_bounds = None
        self._cache_time = 0.0
        logger.debug("Window bounds cache invalidated")
