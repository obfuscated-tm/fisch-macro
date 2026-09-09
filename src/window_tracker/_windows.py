"""
Windows window-tracking backend.

Enumerates top-level windows with the Win32 API and prefers the DWM extended
frame bounds over GetWindowRect: since Windows 10, GetWindowRect includes the
invisible resize border (roughly 7px per side), and every ROI in this project
is stored as a fraction of the window, so that padding would shift them.

Win32 imports are deliberately lazy so this module stays importable — and
testable — on any platform.
"""

import logging
from typing import Optional, Sequence

from src.window_tracker.base import WindowBackend, WindowBounds, matches_app_name

logger = logging.getLogger(__name__)

DWMWA_EXTENDED_FRAME_BOUNDS = 9


class WindowsBackend(WindowBackend):
    """Locates the game window through the Win32 window list."""

    name = "Windows/Win32"
    default_scale = 1.0

    def find_window(self, app_names: Sequence[str]) -> Optional[WindowBounds]:
        bounds = self._find_via_win32(app_names)
        if bounds is not None:
            return bounds
        return self._find_via_pygetwindow(app_names)

    # -- Win32 ---------------------------------------------------------

    def _find_via_win32(self, app_names: Sequence[str]) -> Optional[WindowBounds]:
        try:
            import ctypes
            from ctypes import wintypes

            user32 = ctypes.WinDLL("user32", use_last_error=True)
        except (ImportError, OSError, AttributeError) as e:
            logger.debug("Win32 window enumeration unavailable: %s", e)
            return None

        matches = []

        WNDENUMPROC = ctypes.WINFUNCTYPE(
            wintypes.BOOL, wintypes.HWND, wintypes.LPARAM
        )

        def _on_window(hwnd, _lparam):
            # Minimised windows report a bogus off-screen rect.
            if not user32.IsWindowVisible(hwnd) or user32.IsIconic(hwnd):
                return True

            length = user32.GetWindowTextLengthW(hwnd)
            if length <= 0:
                return True

            buffer = ctypes.create_unicode_buffer(length + 1)
            user32.GetWindowTextW(hwnd, buffer, length + 1)
            if matches_app_name(buffer.value, app_names):
                matches.append(hwnd)
            return True

        user32.EnumWindows(WNDENUMPROC(_on_window), 0)

        for hwnd in matches:
            rect = self._window_rect(hwnd)
            if rect is None:
                continue
            left, top, right, bottom = rect
            bounds = WindowBounds(
                x=int(left),
                y=int(top),
                width=int(right - left),
                height=int(bottom - top),
            )
            if bounds.width <= 0 or bounds.height <= 0:
                continue
            return bounds

        return None

    @staticmethod
    def _window_rect(hwnd):
        """Return (left, top, right, bottom), preferring DWM frame bounds."""
        import ctypes
        from ctypes import wintypes

        rect = wintypes.RECT()

        # Preferred: the frame the user actually sees, without the invisible
        # resize border that GetWindowRect bakes in.
        try:
            dwmapi = ctypes.WinDLL("dwmapi")
            dwmapi.DwmGetWindowAttribute.argtypes = [
                wintypes.HWND,
                wintypes.DWORD,
                ctypes.c_void_p,
                wintypes.DWORD,
            ]
            dwmapi.DwmGetWindowAttribute.restype = ctypes.c_long
            result = dwmapi.DwmGetWindowAttribute(
                hwnd,
                DWMWA_EXTENDED_FRAME_BOUNDS,
                ctypes.byref(rect),
                ctypes.sizeof(rect),
            )
            if result == 0:
                return rect.left, rect.top, rect.right, rect.bottom
        except (OSError, AttributeError) as e:
            logger.debug("DwmGetWindowAttribute unavailable: %s", e)

        try:
            user32 = ctypes.WinDLL("user32", use_last_error=True)
            user32.GetWindowRect.argtypes = [
                wintypes.HWND,
                ctypes.POINTER(wintypes.RECT),
            ]
            user32.GetWindowRect.restype = wintypes.BOOL
            if user32.GetWindowRect(hwnd, ctypes.byref(rect)):
                return rect.left, rect.top, rect.right, rect.bottom
        except (OSError, AttributeError) as e:
            logger.debug("GetWindowRect failed: %s", e)

        return None

    # -- PyGetWindow fallback ------------------------------------------

    def _find_via_pygetwindow(
        self, app_names: Sequence[str]
    ) -> Optional[WindowBounds]:
        try:
            import pygetwindow as gw
        except ImportError:
            logger.debug("pygetwindow not installed; no fallback available")
            return None

        try:
            windows = gw.getAllWindows()
        except Exception as e:
            logger.debug("pygetwindow enumeration failed: %s", e)
            return None

        for window in windows:
            if not matches_app_name(getattr(window, "title", ""), app_names):
                continue
            if getattr(window, "isMinimized", False):
                continue
            width = int(getattr(window, "width", 0))
            height = int(getattr(window, "height", 0))
            if width <= 0 or height <= 0:
                continue
            return WindowBounds(
                x=int(window.left),
                y=int(window.top),
                width=width,
                height=height,
            )

        return None
