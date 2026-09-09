"""
Fisch Macro Entry Point
"""

import argparse
import logging
import os
import sys

from src.config import ConfigManager
from src.controller import Controller
from src.detector import Detector
from src.macro import MacroEngine
from src.paths import log_dir, resolve_base_dir
from src.version import __version__
from src.window_tracker import WindowTracker, report_backend


def setup_logging(base_dir: str, console_level: int = logging.INFO):
    """Setup logging to console and file.

    Args:
        base_dir: Root to write `logs/` under. A bare relative path resolves
            against the working directory, which is "/" when a .app is opened
            from Finder.
        console_level: Level for the console handler. Headless runs raise this
            so engine chatter does not interleave with the status output; the
            log file always keeps everything at INFO.
    """
    file_handler = logging.FileHandler(os.path.join(log_dir(base_dir), "macro.log"))
    file_handler.setLevel(logging.INFO)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(console_level)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[file_handler, console_handler],
    )


def check_permissions() -> bool:
    """Basic check for required macOS permissions (Accessibility)."""
    if sys.platform != "darwin":
        return True

    try:
        from ApplicationServices import AXIsProcessTrusted
        is_trusted = AXIsProcessTrusted()
        if not is_trusted:
            logging.getLogger("main").warning(
                "Accessibility permissions not granted! The macro will not be able to "
                "simulate mouse clicks or listen for hotkeys."
            )
        return is_trusted
    except ImportError:
        # ApplicationServices might not be available, skip check
        return True


def parse_args(argv=None):
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        prog="fisch-macro",
        description="Automated fishing macro for Fisch.",
    )
    # A downloaded build is a directory of binaries with no other way to say
    # which release it came from; this is what a bug report should quote.
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help=(
            "Run without the GUI, controlled by the global hotkey. For small "
            "displays where the panel does not fit, or to keep it off the game."
        ),
    )
    parser.add_argument(
        "--no-autostart",
        action="store_true",
        help="Headless only: wait for the hotkey instead of starting at once.",
    )
    parser.add_argument(
        "--profile",
        metavar="NAME",
        help=(
            "Switch to this color profile before starting. The engine reads "
            "settings from disk, so this is saved like the GUI's profile picker."
        ),
    )
    parser.add_argument(
        "--list-profiles",
        action="store_true",
        help="Print the available color profiles and exit.",
    )
    parser.add_argument(
        "--check-backend",
        action="store_true",
        help=(
            "Report what the window tracker can see on this system and exit. "
            "Use this first when the macro cannot find the game."
        ),
    )
    return parser.parse_args(argv)


def main(argv=None):
    """Initialize components and start the GUI or the headless runner."""
    args = parse_args(argv)

    # From a checkout this is the project root; from a frozen build it is the
    # per-user data directory, seeded with the bundled profiles on first run.
    base_dir = resolve_base_dir()

    # In headless mode the console carries the status output, and --check-backend
    # is meant to be readable at a glance, so keep engine logging to the file and
    # let only warnings through.
    quiet = args.headless or args.check_backend
    setup_logging(base_dir, logging.WARNING if quiet else logging.INFO)
    logger = logging.getLogger("main")
    logger.info("Starting Fisch Macro...")
    logger.info("Data directory: %s", base_dir)

    # Check macOS permissions
    check_permissions()

    controller = None
    try:
        # Initialize core components
        logger.info("Initializing configuration...")
        config_manager = ConfigManager(base_dir)

        if args.list_profiles:
            for name in config_manager.list_profiles():
                print(name)
            return 0

        settings = config_manager.load_settings()

        if args.profile:
            available = config_manager.list_profiles()
            if args.profile not in available:
                print(
                    f"Unknown profile {args.profile!r}. Available: "
                    f"{', '.join(available) or '(none)'}",
                    file=sys.stderr,
                )
                return 2
            settings.active_profile = args.profile
            config_manager.save_settings(settings)
            logger.info("Active profile set to %s", args.profile)

        logger.info("Initializing window tracker...")
        window_tracker = WindowTracker()

        if args.check_backend:
            return report_backend(window_tracker)

        logger.info("Initializing detector...")
        detector = Detector(config_manager)

        logger.info("Initializing controller...")
        controller = Controller()

        # Set up the global start/stop hotkey
        logger.info(f"Setting up start/stop hotkey on key: {settings.killswitch_key}")
        controller.setup_killswitch(settings.killswitch_key)

        logger.info("Initializing macro engine...")
        macro_engine = MacroEngine(detector, controller, config_manager)

        if args.headless:
            logger.info("Starting headless runner...")
            from src.headless import run_headless

            return run_headless(
                macro_engine,
                config_manager,
                window_tracker,
                settings,
                autostart=not args.no_autostart,
            )

        # Start the GUI
        logger.info("Starting GUI...")
        from src.gui import MacroGUI

        app = MacroGUI(macro_engine, config_manager, window_tracker)

        # This blocks until the window is closed
        app.run()
        return 0

    except Exception as e:
        logger.error(f"Failed to start macro: {e}", exc_info=True)
        print(f"\nCRITICAL ERROR: {e}\nCheck logs/macro.log for details.")
        return 1
    finally:
        logger.info("Cleaning up...")
        if controller is not None:
            controller.cleanup()
        logger.info("Exiting.")


if __name__ == "__main__":
    sys.exit(main())
