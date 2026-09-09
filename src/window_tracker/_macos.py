"""
macOS window-tracking backend.

Uses Quartz CGWindowListCopyWindowInfo to enumerate on-screen windows, and
CGDisplayPixelsWide vs pyautogui's logical width to spot a Retina display.
"""

import logging
import re
import subprocess
from typing import Optional, Sequence

from src.window_tracker.base import WindowBackend, WindowBounds, matches_app_name

logger = logging.getLogger(__name__)


class MacOSBackend(WindowBackend):
    """Locates the game window through the Quartz window server."""

    name = "macOS/Quartz"

    # Retina is the safer assumption on a Mac: guessing 1.0 on a 2x display
    # produces captures a quarter of the intended size, which fails silently.
    default_scale = 2.0

    def find_window(self, app_names: Sequence[str]) -> Optional[WindowBounds]:
        try:
            from Quartz import (
                CGWindowListCopyWindowInfo,
                kCGWindowListOptionOnScreenOnly,
                kCGNullWindowID,
            )
        except ImportError:
            logger.warning(
                "Quartz framework not available. Cannot locate the game window. "
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
            if not matches_app_name(owner_name, app_names):
                continue

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

            return bounds

        return None

    def detect_scale_factor(self) -> Optional[float]:
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
                    logger.info("Display scale factor (Quartz): %.2f", scale)
                    return scale
        except ImportError:
            logger.debug("Quartz not available for scale factor detection")
        except Exception as e:
            logger.debug("Quartz scale factor detection failed: %s", e)

        # Method 2: system_profiler fallback
        try:
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
                    logger.info(
                        "Display scale factor (system_profiler): 2.0 (Retina detected)"
                    )
                    return 2.0

                # Try to parse resolution lines for ratio
                # Typical format: "Resolution: 2560 x 1600 Retina"
                # or "Resolution: 1440 x 900"
                resolutions = re.findall(r"Resolution:\s*(\d+)\s*x\s*(\d+)", output)
                if len(resolutions) >= 2:
                    # If two resolution lines, the ratio might indicate scaling
                    w1 = int(resolutions[0][0])
                    w2 = int(resolutions[1][0])
                    if w1 > 0 and w2 > 0:
                        ratio = max(w1, w2) / min(w1, w2)
                        if 1.5 <= ratio <= 2.5:
                            logger.info("Display scale factor (resolution ratio): 2.0")
                            return 2.0
        except Exception as e:
            logger.debug("system_profiler scale factor detection failed: %s", e)

        return None
