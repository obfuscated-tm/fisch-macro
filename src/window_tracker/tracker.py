"""
Platform-agnostic window tracker.

Picks a backend for the current OS and caches its answers so the detector can
ask for the window bounds every frame without hammering the window server.
"""

import logging
import sys
import time
from typing import Optional, Sequence

from src.window_tracker.base import (
    DEFAULT_APP_NAMES,
    WindowBackend,
    WindowBounds,
)

logger = logging.getLogger(__name__)


def create_backend(platform: Optional[str] = None) -> Optional[WindowBackend]:
    """Instantiate the window backend for `platform` (defaults to this OS).

    Args:
        platform: A `sys.platform` string. Defaults to the running platform.

    Returns:
        A WindowBackend, or None when the platform has no implementation.
    """
    platform = platform or sys.platform

    if platform == "darwin":
        from src.window_tracker._macos import MacOSBackend

        return MacOSBackend()

    if platform.startswith("win"):
        from src.window_tracker._windows import WindowsBackend

        return WindowsBackend()

    if platform.startswith("linux"):
        from src.window_tracker._linux import LinuxBackend

        return LinuxBackend()

    return None


class WindowTracker:
    """Tracks the game window position and display scale factor.

    Caches the window bounds to avoid querying the window list on every frame.
    Automatically refreshes the cache after `cache_ttl` seconds.
    """

    def __init__(
        self,
        app_names: Sequence[str] = DEFAULT_APP_NAMES,
        backend: Optional[WindowBackend] = None,
    ):
        """
        Args:
            app_names: Window owner/title fragments identifying the game.
                Matched case-insensitively as substrings.
            backend: Explicit backend, mainly for tests. Defaults to the
                backend for the running platform.
        """
        self.app_names = tuple(app_names)
        self.backend = backend if backend is not None else create_backend()

        self._cached_bounds: Optional[WindowBounds] = None
        self._cache_time: float = 0.0
        self.cache_ttl: float = 0.4
        self._scale_factor: Optional[float] = None
        self._warned_unsupported = False
        self._warned_diagnostics = False

        if self.backend is None:
            logger.warning(
                "No window tracking backend for platform %r. The macro cannot "
                "locate the game window on this system.",
                sys.platform,
            )
        else:
            logger.info("Window tracking backend: %s", self.backend.name)

    def get_roblox_bounds(self) -> Optional[WindowBounds]:
        """Find the game window bounds on screen.

        Returns:
            WindowBounds if the window is found, None otherwise.
            Returns cached bounds if the cache is still fresh.
        """
        if self.backend is None:
            if not self._warned_unsupported:
                self._warned_unsupported = True
                logger.warning("Cannot locate the game window: no backend available.")
            return None

        # Return cached bounds if still valid
        now = time.time()
        if self._cached_bounds is not None and (now - self._cache_time) < self.cache_ttl:
            return self._cached_bounds

        try:
            bounds = self.backend.find_window(self.app_names)
        except Exception as e:
            logger.debug("Window lookup failed: %s", e)
            bounds = None

        if bounds is None:
            self._log_diagnostics_once()
            logger.debug("Game window not found among on-screen windows")
            return None

        self._cached_bounds = bounds
        self._cache_time = now
        logger.debug(
            "Found game window: x=%d, y=%d, w=%d, h=%d",
            bounds.x, bounds.y, bounds.width, bounds.height,
        )
        return bounds

    def get_scale_factor(self) -> float:
        """Determine the display scale factor (1 for standard, 2 for Retina).

        Returns:
            Display scale factor as a float (typically 1.0 or 2.0).
        """
        if self._scale_factor is not None:
            return self._scale_factor

        if self.backend is None:
            self._scale_factor = 1.0
            return self._scale_factor

        try:
            scale = self.backend.detect_scale_factor()
        except Exception as e:
            logger.debug("Scale factor detection failed: %s", e)
            scale = None

        if scale is not None:
            self._scale_factor = scale
            return scale

        self._scale_factor = self.backend.default_scale
        logger.warning(
            "Could not determine display scale factor. Defaulting to %.1f.",
            self._scale_factor,
        )
        return self._scale_factor

    def invalidate_cache(self) -> None:
        """Clear the cached window bounds, forcing a fresh lookup on next call."""
        self._cached_bounds = None
        self._cache_time = 0.0
        logger.debug("Window bounds cache invalidated")

    def _log_diagnostics_once(self) -> None:
        """Emit the backend's environment hint the first time a lookup fails."""
        if self._warned_diagnostics or self.backend is None:
            return
        hint = self.backend.diagnostics()
        if hint:
            self._warned_diagnostics = True
            logger.warning("%s", hint)


def report_backend(window_tracker: "WindowTracker") -> int:
    """Print what the window-tracking backend can see.

    Exercises the platform lookup for real — the Win32 window walk, the X11
    `_NET_CLIENT_LIST` read — rather than merely importing it, which is the
    difference between a build that packages and a build that works.

    Args:
        window_tracker: A constructed WindowTracker.

    Returns:
        0 if the platform has a backend and it answered without raising.
        Not finding the game is not a failure; nothing may be running.
    """
    backend = window_tracker.backend
    if backend is None:
        print(
            f"No window tracking backend for platform {sys.platform!r}. "
            "The macro cannot locate the game window on this system.",
            file=sys.stderr,
        )
        return 1

    print(f"backend:  {backend.name}")

    hint = backend.diagnostics()
    if hint:
        print(f"warning:  {hint}")

    bounds = window_tracker.get_roblox_bounds()
    if bounds is None:
        print("window:   not found — is the game running, and windowed?")
    else:
        print(
            f"window:   {bounds.width}x{bounds.height} "
            f"at ({bounds.x}, {bounds.y})"
        )

    scale = window_tracker.get_scale_factor()
    print(f"scale:    {scale}x")

    if not scale > 0:
        print(f"Implausible display scale factor: {scale}", file=sys.stderr)
        return 1

    return 0
