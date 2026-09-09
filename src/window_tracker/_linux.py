"""
Linux/X11 window-tracking backend.

Reads the window list straight from X11 via python-xlib (already a PyAutoGUI
dependency on Linux), falling back to shelling out to xdotool.

Wayland is not supported and cannot be: neither mss nor pyautogui can capture
or inject into another application's surface under a Wayland compositor, which
many distributions now default to. `diagnostics()` surfaces that clearly
rather than letting the macro fail with empty captures.
"""

import logging
import os
import shutil
import subprocess
from typing import Optional, Sequence

from src.window_tracker.base import WindowBackend, WindowBounds, matches_app_name

logger = logging.getLogger(__name__)


class LinuxBackend(WindowBackend):
    """Locates the game window through X11."""

    name = "Linux/X11"
    default_scale = 1.0

    def find_window(self, app_names: Sequence[str]) -> Optional[WindowBounds]:
        bounds = self._find_via_xlib(app_names)
        if bounds is not None:
            return bounds
        return self._find_via_xdotool(app_names)

    # -- python-xlib ---------------------------------------------------

    def _find_via_xlib(self, app_names: Sequence[str]) -> Optional[WindowBounds]:
        try:
            from Xlib import X
            from Xlib import display as xdisplay
        except ImportError:
            logger.debug(
                "python-xlib not installed. Install it with: pip install python-xlib"
            )
            return None

        display = None
        try:
            display = xdisplay.Display()
            root = display.screen().root

            net_client_list = display.intern_atom("_NET_CLIENT_LIST")
            net_wm_name = display.intern_atom("_NET_WM_NAME")
            utf8_string = display.intern_atom("UTF8_STRING")

            client_list = root.get_full_property(net_client_list, X.AnyPropertyType)
            if client_list is None:
                logger.debug("_NET_CLIENT_LIST unavailable (non-EWMH window manager?)")
                return None

            for window_id in client_list.value:
                window = display.create_resource_object("window", window_id)
                if not self._window_matches(window, net_wm_name, utf8_string, app_names):
                    continue

                geometry = window.get_geometry()
                if geometry.width <= 0 or geometry.height <= 0:
                    continue

                # get_geometry() reports coordinates relative to the parent,
                # which under a reparenting WM is the decoration frame, not the
                # root. Translate to get true screen coordinates. The receiver
                # of translate_coords is the destination window.
                origin = root.translate_coords(window, 0, 0)

                return WindowBounds(
                    x=int(origin.x),
                    y=int(origin.y),
                    width=int(geometry.width),
                    height=int(geometry.height),
                )

            return None
        except Exception as e:
            logger.debug("Xlib window lookup failed: %s", e)
            return None
        finally:
            if display is not None:
                try:
                    display.close()
                except Exception:
                    pass

    @staticmethod
    def _window_matches(window, net_wm_name, utf8_string, app_names) -> bool:
        """Check a window's EWMH name, legacy name and WM_CLASS for a match."""
        try:
            prop = window.get_full_property(net_wm_name, utf8_string)
            if prop is not None:
                value = prop.value
                if isinstance(value, bytes):
                    value = value.decode("utf-8", errors="replace")
                if matches_app_name(value, app_names):
                    return True
        except Exception:
            pass

        try:
            if matches_app_name(window.get_wm_name(), app_names):
                return True
        except Exception:
            pass

        try:
            wm_class = window.get_wm_class()
            if wm_class and any(matches_app_name(part, app_names) for part in wm_class):
                return True
        except Exception:
            pass

        return False

    # -- xdotool fallback ----------------------------------------------

    def _find_via_xdotool(self, app_names: Sequence[str]) -> Optional[WindowBounds]:
        if shutil.which("xdotool") is None:
            logger.debug("xdotool not installed; no fallback available")
            return None

        for app_name in app_names:
            try:
                search = subprocess.run(
                    ["xdotool", "search", "--name", app_name],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                if search.returncode != 0 or not search.stdout.strip():
                    continue

                for window_id in search.stdout.split():
                    geometry = subprocess.run(
                        ["xdotool", "getwindowgeometry", "--shell", window_id],
                        capture_output=True,
                        text=True,
                        timeout=5,
                    )
                    if geometry.returncode != 0:
                        continue

                    values = {}
                    for line in geometry.stdout.splitlines():
                        if "=" in line:
                            key, _, value = line.partition("=")
                            values[key.strip()] = value.strip()

                    try:
                        bounds = WindowBounds(
                            x=int(values["X"]),
                            y=int(values["Y"]),
                            width=int(values["WIDTH"]),
                            height=int(values["HEIGHT"]),
                        )
                    except (KeyError, ValueError):
                        continue

                    if bounds.width > 0 and bounds.height > 0:
                        return bounds
            except (subprocess.SubprocessError, OSError) as e:
                logger.debug("xdotool lookup for %r failed: %s", app_name, e)

        return None

    # -- diagnostics ---------------------------------------------------

    def diagnostics(self) -> Optional[str]:
        if os.environ.get("WAYLAND_DISPLAY") or (
            os.environ.get("XDG_SESSION_TYPE", "").lower() == "wayland"
        ):
            return (
                "This looks like a Wayland session. Screen capture and input "
                "injection into other applications do not work under Wayland — "
                "the macro cannot see or control the game. Log in to an X11 "
                "session instead — most display managers offer it as a choice "
                "on the login screen."
            )

        if not os.environ.get("DISPLAY"):
            return (
                "DISPLAY is not set, so there is no X11 session to query. Run "
                "the macro from a desktop session rather than a bare SSH shell, "
                "or export DISPLAY=:0 first."
            )

        return None
