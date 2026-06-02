"""
Input controller for the Fisch macro.

Handles mouse input (hold, release, click, rapid-click) and the killswitch
keyboard listener. All mouse actions check the killswitch before executing,
ensuring the macro can be stopped instantly at any time.
"""

import logging
import threading
import time
from typing import Optional, Callable, List

import pyautogui

# -- Module-level pyautogui safety configuration --
pyautogui.FAILSAFE = True  # Move mouse to upper-left corner = emergency stop
pyautogui.PAUSE = 0.01     # Minimal delay between pyautogui calls

logger = logging.getLogger(__name__)


class Controller:
    """Manages mouse input and killswitch for the fishing macro.

    The killswitch is a keyboard listener that, when triggered, immediately
    stops all macro activity and releases any held mouse buttons. Multiple
    callbacks can be registered to run on kill.

    Usage:
        controller = Controller()
        controller.setup_killswitch('f6')
        controller.start()
        # ... macro loop ...
        controller.mouse_hold()
        time.sleep(1)
        controller.mouse_release()
        controller.cleanup()
    """

    def __init__(self):
        self._killed = threading.Event()
        self._running = threading.Event()
        self._mouse_held: bool = False
        self._listener = None
        self._kill_callbacks: List[Callable] = []
        self._killswitch_key = None
        self.logger = logging.getLogger("controller")

    def setup_killswitch(self, key_name: str = "f6") -> None:
        """Set up the keyboard killswitch listener.

        Args:
            key_name: Name of the key to use as the killswitch (e.g. 'f6',
                      'esc', 'f12'). Mapped to the appropriate pynput key.
        """
        try:
            from pynput import keyboard
        except ImportError:
            self.logger.error(
                "pynput is not installed. Killswitch will not be available. "
                "Install it with: pip install pynput"
            )
            return

        # Map the key name string to a pynput Key or KeyCode
        self._killswitch_key = self._resolve_key(key_name, keyboard)

        # Stop any existing listener
        if self._listener is not None:
            self._listener.stop()

        self._listener = keyboard.Listener(on_press=self._on_key_press)
        self._listener.daemon = True
        self._listener.start()

        self.logger.info(
            "Killswitch active: press '%s' to stop the macro at any time.",
            key_name,
        )

    @staticmethod
    def _resolve_key(key_name: str, keyboard_module):
        """Resolve a key name string to a pynput Key or KeyCode.

        Args:
            key_name: Human-readable key name (e.g. 'f6', 'esc', 'a').
            keyboard_module: The pynput.keyboard module.

        Returns:
            A pynput Key enum member or a KeyCode.
        """
        # Try function keys and special keys first
        key_name_lower = key_name.lower().strip()

        # Map of common names to pynput Key attributes
        special_keys = {
            "esc": "esc",
            "escape": "esc",
            "space": "space",
            "enter": "enter",
            "return": "enter",
            "tab": "tab",
            "backspace": "backspace",
            "delete": "delete",
            "shift": "shift",
            "ctrl": "ctrl_l",
            "alt": "alt_l",
            "cmd": "cmd",
            "command": "cmd",
        }

        # Check special keys
        if key_name_lower in special_keys:
            return getattr(keyboard_module.Key, special_keys[key_name_lower])

        # Check function keys (f1–f20)
        if key_name_lower.startswith("f") and key_name_lower[1:].isdigit():
            fkey_attr = key_name_lower
            if hasattr(keyboard_module.Key, fkey_attr):
                return getattr(keyboard_module.Key, fkey_attr)

        # Fall back to character key
        if len(key_name_lower) == 1:
            return keyboard_module.KeyCode.from_char(key_name_lower)

        # Last resort: try as a Key attribute directly
        if hasattr(keyboard_module.Key, key_name_lower):
            return getattr(keyboard_module.Key, key_name_lower)

        logger.warning(
            "Could not resolve key '%s'. Using KeyCode.from_char fallback.",
            key_name,
        )
        return keyboard_module.KeyCode.from_char(key_name_lower[0])

    def _on_key_press(self, key) -> Optional[bool]:
        """Callback for keyboard listener. Triggers kill on matching key.

        Args:
            key: The key that was pressed (pynput Key or KeyCode).

        Returns:
            False to stop the listener if killswitch triggered, None otherwise.
        """
        if self._killswitch_key is None:
            return None

        try:
            # Compare the pressed key against the killswitch key
            if key == self._killswitch_key:
                self.kill()
                return False  # Stop the listener
        except AttributeError:
            pass

        return None

    def kill(self) -> None:
        """Activate the killswitch: stop all macro activity immediately.

        Sets the killed flag, clears running, releases held mouse buttons,
        and invokes all registered kill callbacks.
        """
        self._killed.set()
        self._running.clear()

        # Ensure the mouse is released
        if self._mouse_held:
            try:
                pyautogui.mouseUp()
            except Exception:
                pass
            self._mouse_held = False

        # Invoke all registered kill callbacks
        for callback in self._kill_callbacks:
            try:
                callback()
            except Exception as e:
                self.logger.error("Kill callback raised an exception: %s", e)

        self.logger.critical("KILLSWITCH ACTIVATED — macro stopped.")

    def is_killed(self) -> bool:
        """Check whether the killswitch has been triggered.

        Returns:
            True if the macro has been killed, False otherwise.
        """
        return self._killed.is_set()

    def start(self) -> None:
        """Start (or resume) the macro.

        Clears the killed flag and sets the running flag.
        """
        self._killed.clear()
        self._running.set()
        self.logger.info("Controller started.")

    def stop(self) -> None:
        """Stop the macro gracefully (without triggering kill callbacks).

        Clears the running flag and releases held mouse buttons.
        """
        self._running.clear()

        if self._mouse_held:
            try:
                pyautogui.mouseUp()
            except Exception:
                pass
            self._mouse_held = False

        self.logger.info("Controller stopped.")

    def _check_safety(self) -> bool:
        """Check if it's safe to perform an action.

        Returns:
            True if the macro is alive and safe to act, False if killed.
        """
        if self._killed.is_set():
            return False
        return True

    def mouse_hold(self) -> None:
        """Press and hold the left mouse button.

        Does nothing if the killswitch has been triggered or the mouse
        is already held.
        """
        if not self._check_safety():
            return

        if not self._mouse_held:
            pyautogui.mouseDown()
            self._mouse_held = True
            self.logger.debug("Mouse held down.")

    def mouse_release(self) -> None:
        """Release the left mouse button.

        Always attempts to release, regardless of killswitch state,
        to ensure the mouse is never stuck held.
        """
        try:
            pyautogui.mouseUp()
        except Exception:
            pass
        self._mouse_held = False
        self.logger.debug("Mouse released.")

    def mouse_click(self, x: Optional[int] = None, y: Optional[int] = None) -> None:
        """Perform a single left-click.

        Args:
            x: Optional x-coordinate. If None, clicks at current position.
            y: Optional y-coordinate. If None, clicks at current position.
        """
        if not self._check_safety():
            return

        if x is not None and y is not None:
            pyautogui.click(x, y)
        else:
            pyautogui.click()

        self.logger.debug("Mouse clicked at (%s, %s).", x, y)

    def rapid_click(self, count: int = 3, interval: float = 0.05) -> None:
        """Perform multiple rapid left-clicks.

        Used for shake/QTE events that require fast clicking.

        Args:
            count: Number of clicks to perform.
            interval: Seconds between each click.
        """
        for i in range(count):
            if not self._check_safety():
                self.logger.debug("Rapid click interrupted by killswitch at click %d.", i)
                return
            pyautogui.click()
            if i < count - 1:
                time.sleep(interval)

        self.logger.debug("Rapid clicked %d times.", count)

    def on_kill(self, callback: Callable) -> None:
        """Register a callback to be invoked when the killswitch triggers.

        Args:
            callback: A no-argument callable to run on kill.
        """
        self._kill_callbacks.append(callback)

    def cleanup(self) -> None:
        """Clean up resources: stop the keyboard listener and release the mouse.

        Should be called when the macro is shutting down.
        """
        if self._listener is not None:
            try:
                self._listener.stop()
            except Exception:
                pass
            self._listener = None

        if self._mouse_held:
            try:
                pyautogui.mouseUp()
            except Exception:
                pass
            self._mouse_held = False

        self.logger.info("Controller cleaned up.")
