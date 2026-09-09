"""
Headless runner — the macro without the tkinter panel.

For displays the control panel (372x572, minimum 340x500) does not fit, and for
setups where it would cover half the game anyway. Calibrate with the GUI, then
run here: the ROIs are stored as fractions of the game window, so they carry
over unchanged.

Everything the GUI drives is still driven — the same engine, the same global
hotkey — with state changes and stats going to the console instead of widgets.
"""

import logging
import signal
import threading
import time
from typing import Optional

logger = logging.getLogger("headless")

#: Seconds between periodic status lines while the macro is running.
STATUS_INTERVAL = 30.0


class HeadlessRunner:
    """Runs the macro engine with console output and hotkey control."""

    def __init__(self, engine, config_manager, window_tracker, settings):
        """
        Args:
            engine: MacroEngine instance.
            config_manager: ConfigManager instance.
            window_tracker: WindowTracker instance.
            settings: Loaded Settings.
        """
        self.engine = engine
        self.config_manager = config_manager
        self.window_tracker = window_tracker
        self.settings = settings

        self._shutdown = threading.Event()
        self._last_hotkey_toggle = 0.0
        self._last_status = 0.0

    # -- wiring --------------------------------------------------------

    def _connect_callbacks(self) -> None:
        """Route engine state, log and stats events to the console."""
        self.engine.on_state_change(
            lambda state: print(f"[state] {state.value}", flush=True)
        )
        self.engine.on_log(lambda message: print(f"[log]   {message}", flush=True))
        self.engine.controller.on_hotkey(self._toggle)

    def _resolve_window(self) -> bool:
        """Locate the game window and hand it to the detector.

        Returns:
            True if the detector now has valid window info.
        """
        self.window_tracker.invalidate_cache()
        bounds = self.window_tracker.get_roblox_bounds()

        if bounds is None:
            print(
                "\n  Could not find the game window.\n"
                "  Make sure Roblox (or Sober) is open and in WINDOWED mode,\n"
                "  then press "
                f"{self.settings.killswitch_key.upper()} to try again.\n",
                flush=True,
            )
            return False

        scale = self.window_tracker.get_scale_factor()
        self.engine.detector.set_window_info(bounds, scale)
        print(
            f"[window] {bounds.width}x{bounds.height} "
            f"at ({bounds.x}, {bounds.y}), scale={scale}x",
            flush=True,
        )
        return True

    # -- control -------------------------------------------------------

    def _toggle(self) -> None:
        """Start or stop the macro. Bound to the global hotkey.

        Debounced to match the GUI: the hotkey fires on key release and a
        double-trigger would start and immediately stop the run.
        """
        now = time.monotonic()
        if now - self._last_hotkey_toggle < 0.2:
            return
        self._last_hotkey_toggle = now

        if self.engine.is_running():
            self.stop_macro()
        else:
            self.start_macro()

    def start_macro(self) -> bool:
        """Re-resolve the window and start the engine.

        Returns:
            True if the engine was started.
        """
        if self.engine.is_running():
            return True

        # The window may have moved or been resized since the last run, and
        # every ROI is relative to it, so never reuse stale bounds.
        if not self._resolve_window():
            return False

        self.engine.start()
        self._last_status = time.time()
        print("[macro] started", flush=True)
        return True

    def stop_macro(self) -> None:
        """Stop the engine and print the session summary."""
        if not self.engine.is_running():
            return
        self.engine.stop()
        print("[macro] stopped", flush=True)
        self._print_stats()

    def shutdown(self) -> None:
        """Ask the run loop to exit."""
        self._shutdown.set()

    # -- output --------------------------------------------------------

    def _print_stats(self) -> None:
        stats = self.engine.stats
        duration = stats.session_duration
        hours = duration / 3600.0
        per_hour = (stats.fish_caught / hours) if hours > 0.01 else 0.0

        print(
            f"[stats] caught={stats.fish_caught} failed={stats.fish_failed} "
            f"casts={stats.casts} rate={stats.success_rate * 100:.0f}% "
            f"uptime={self._format_duration(duration)} "
            f"({per_hour:.0f}/hr)",
            flush=True,
        )

    @staticmethod
    def _format_duration(seconds: float) -> str:
        seconds = int(seconds)
        hours, remainder = divmod(seconds, 3600)
        minutes, secs = divmod(remainder, 60)
        if hours:
            return f"{hours}h{minutes:02d}m"
        if minutes:
            return f"{minutes}m{secs:02d}s"
        return f"{secs}s"

    def _print_banner(self, autostart: bool) -> None:
        profile = self.settings.active_profile
        key = self.settings.killswitch_key.upper()

        print("\n" + "─" * 52, flush=True)
        print("  Fisch Macro — headless", flush=True)
        print("─" * 52, flush=True)
        print(f"  profile   {profile}", flush=True)
        print(f"  hotkey    {key} to start/stop", flush=True)
        print(f"  backend   {getattr(self.window_tracker.backend, 'name', 'none')}", flush=True)
        print("  quit      Ctrl+C", flush=True)
        if not autostart:
            print(f"\n  Waiting for {key}...", flush=True)
        print("─" * 52 + "\n", flush=True)

    # -- main loop -----------------------------------------------------

    def run(self, autostart: bool = True) -> int:
        """Run until interrupted.

        Args:
            autostart: Start the macro immediately rather than waiting for
                the hotkey.

        Returns:
            A process exit code.
        """
        self._connect_callbacks()
        self._print_banner(autostart)

        # Ctrl+C has to unblock the wait below rather than raise inside
        # whatever the engine thread happens to be doing.
        def _on_signal(_signum, _frame):
            print("\n[macro] interrupt received, shutting down...", flush=True)
            self.shutdown()

        try:
            signal.signal(signal.SIGINT, _on_signal)
            signal.signal(signal.SIGTERM, _on_signal)
        except ValueError:
            # Not on the main thread (tests); the caller handles interrupts.
            logger.debug("Could not install signal handlers off the main thread")

        if autostart:
            self.start_macro()

        try:
            while not self._shutdown.wait(1.0):
                if self.engine.is_running():
                    now = time.time()
                    if now - self._last_status >= STATUS_INTERVAL:
                        self._last_status = now
                        self._print_stats()
        except KeyboardInterrupt:
            self.shutdown()

        if self.engine.is_running():
            self.stop_macro()

        return 0


def run_headless(
    engine,
    config_manager,
    window_tracker,
    settings,
    autostart: bool = True,
) -> int:
    """Construct a HeadlessRunner and run it. Returns a process exit code."""
    runner = HeadlessRunner(engine, config_manager, window_tracker, settings)
    return runner.run(autostart=autostart)
