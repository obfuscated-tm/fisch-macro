"""
Filesystem locations for settings, profiles and logs.

Running from a checkout, everything lives next to main.py — that is what the
project has always done and nothing changes there.

Inside a PyInstaller bundle it cannot: a one-file build unpacks to a temporary
directory that is deleted on exit, and a macOS .app bundle is read-only in
practice (writing into it invalidates the code signature). Either way the
saved settings and every calibrated profile would be silently thrown away.

So a frozen build reads its bundled defaults from the bundle and writes user
data to the platform's own location, seeding it on first run.
"""

import logging
import os
import shutil
import sys
from typing import Optional

logger = logging.getLogger(__name__)

APP_NAME = "FischMacro"

#: Files and directories seeded from the bundle into the user data directory.
SEEDED_PROFILES_DIR = "profiles"
SEEDED_SETTINGS_FILE = "settings.json"


def is_frozen() -> bool:
    """True when running from a PyInstaller bundle."""
    return getattr(sys, "frozen", False)


def bundle_dir() -> str:
    """Directory holding read-only resources shipped with the app.

    For a one-file build this is the temporary extraction directory; for a
    one-directory or .app build it is the directory holding the executable.
    From a checkout it is the project root.
    """
    if is_frozen():
        # _MEIPASS is set for one-file builds; onedir falls back to the
        # executable's own directory.
        return getattr(sys, "_MEIPASS", os.path.dirname(sys.executable))
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def user_data_dir() -> str:
    """Writable per-user directory for settings, profiles and logs."""
    if sys.platform == "darwin":
        return os.path.join(
            os.path.expanduser("~/Library/Application Support"), APP_NAME
        )

    if sys.platform.startswith("win"):
        base = os.environ.get("APPDATA") or os.path.expanduser("~")
        return os.path.join(base, APP_NAME)

    base = os.environ.get("XDG_CONFIG_HOME") or os.path.expanduser("~/.config")
    return os.path.join(base, "fisch-macro")


def resolve_base_dir() -> str:
    """Return the directory ConfigManager should treat as its root.

    From a checkout this is the project root, so a developer's settings.json
    and profiles/ stay under version control where they expect them. Frozen,
    it is the user data directory, seeded from the bundle on first run.
    """
    if not is_frozen():
        return bundle_dir()

    target = user_data_dir()
    os.makedirs(target, exist_ok=True)
    _seed_user_data(bundle_dir(), target)
    return target


def log_dir(base_dir: Optional[str] = None) -> str:
    """Return the directory for log files, creating it if needed.

    Args:
        base_dir: Root to place `logs/` under. Defaults to `resolve_base_dir()`.

    The default used to be a bare relative "logs", which resolves against the
    working directory — that is the project root when launched from a shell,
    but "/" when a .app is launched from Finder.
    """
    target = os.path.join(base_dir or resolve_base_dir(), "logs")
    os.makedirs(target, exist_ok=True)
    return target


def _seed_user_data(source: str, target: str) -> None:
    """Copy bundled defaults into `target` without overwriting user files."""
    bundled_profiles = os.path.join(source, SEEDED_PROFILES_DIR)
    user_profiles = os.path.join(target, SEEDED_PROFILES_DIR)
    os.makedirs(user_profiles, exist_ok=True)

    if os.path.isdir(bundled_profiles):
        for name in os.listdir(bundled_profiles):
            if not name.endswith(".json"):
                continue
            destination = os.path.join(user_profiles, name)
            if os.path.exists(destination):
                continue  # never clobber a profile the user calibrated
            try:
                shutil.copy2(os.path.join(bundled_profiles, name), destination)
                logger.info("Seeded bundled profile %s", name)
            except OSError as e:
                logger.warning("Could not seed profile %s: %s", name, e)

    bundled_settings = os.path.join(source, SEEDED_SETTINGS_FILE)
    user_settings = os.path.join(target, SEEDED_SETTINGS_FILE)
    if os.path.isfile(bundled_settings) and not os.path.exists(user_settings):
        try:
            shutil.copy2(bundled_settings, user_settings)
            logger.info("Seeded default settings.json")
        except OSError as e:
            logger.warning("Could not seed settings.json: %s", e)
