"""
Fisch Macro Entry Point
"""

import logging
import os
import sys

from src.config import ConfigManager
from src.controller import Controller
from src.detector import Detector
from src.gui import MacroGUI
from src.macro import MacroEngine
from src.window_tracker import WindowTracker


def setup_logging():
    """Setup logging to console and file."""
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(log_dir, "macro.log")),
            logging.StreamHandler(sys.stdout),
        ],
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


def main():
    """Initialize components and start the GUI."""
    setup_logging()
    logger = logging.getLogger("main")
    logger.info("Starting Fisch Macro...")

    # Check macOS permissions
    check_permissions()

    # Get project root directory
    base_dir = os.path.dirname(os.path.abspath(__file__))

    try:
        # Initialize core components
        logger.info("Initializing configuration...")
        config_manager = ConfigManager(base_dir)
        settings = config_manager.load_settings()

        logger.info("Initializing window tracker...")
        window_tracker = WindowTracker()

        logger.info("Initializing detector...")
        detector = Detector(config_manager)

        logger.info("Initializing controller...")
        controller = Controller()

        # Set up the global start/stop hotkey
        logger.info(f"Setting up start/stop hotkey on key: {settings.killswitch_key}")
        controller.setup_killswitch(settings.killswitch_key)

        logger.info("Initializing macro engine...")
        macro_engine = MacroEngine(detector, controller, config_manager)

        # Start the GUI
        logger.info("Starting GUI...")
        app = MacroGUI(macro_engine, config_manager, window_tracker)

        # This blocks until the window is closed
        app.run()

    except Exception as e:
        logger.error(f"Failed to start macro: {e}", exc_info=True)
        print(f"\nCRITICAL ERROR: {e}\nCheck logs/macro.log for details.")
    finally:
        logger.info("Cleaning up...")
        if 'controller' in locals():
            controller.cleanup()
        logger.info("Exiting.")


if __name__ == "__main__":
    main()
