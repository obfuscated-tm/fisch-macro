"""A control the user can move must change what the macro does.

Half the Settings tab was, at one point, wired to nothing: the sliders belonged
to a PD controller that had been replaced, and their settings were either read
nowhere at all or only inside helpers with no callers. Tuning them did nothing,
silently, which is worse than not offering them — time goes into chasing a
behaviour change that was never possible.

These tests are cheap and they close most of that door: they read the source
rather than run the UI, so they need no display.

The limit is worth knowing. Run against the state that prompted them, this
catches five of the eight dead controls — the ones read nowhere at all. The
other three were read inside helpers that themselves had no callers, and
spotting that from the text alone would need a call graph. So a pass here means
"something mentions this setting", not "the macro acts on it".
"""

import pathlib
import re
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.config import Settings  # noqa: E402

GUI = (ROOT / "src" / "gui.py").read_text()

# Where a setting counts as doing something: the loop, the detector, and the
# controller. gui.py and config.py are excluded because writing a value back to
# itself is not a use of it.
CONSUMERS = [
    p for p in (ROOT / "src").glob("*.py")
    if p.name not in {"gui.py", "config.py", "__init__.py"}
]
CONSUMER_SRC = "\n".join(p.read_text() for p in CONSUMERS)


def gui_written_settings():
    """Settings the GUI assigns to — i.e. every control it offers."""
    return set(re.findall(r"self\.settings\.(\w+)\s*=\s*self\.\w+_var", GUI))


def test_every_setting_the_gui_offers_is_read_by_something():
    dead = sorted(
        name for name in gui_written_settings()
        if f"settings.{name}" not in CONSUMER_SRC
        and f'"{name}"' not in CONSUMER_SRC
    )

    assert not dead, (
        "these settings have a control in the GUI but nothing reads them, so "
        f"moving them does nothing: {dead}"
    )


def test_every_setting_the_gui_offers_exists():
    settings = Settings()
    missing = sorted(
        name for name in gui_written_settings() if not hasattr(settings, name)
    )

    assert not missing, f"the GUI writes settings that do not exist: {missing}"


def test_every_gui_variable_is_both_traced_and_saved():
    """A control that is not traced never autosaves; one never saved is lost."""
    created = set(re.findall(r"self\.(\w+_var)\s*=\s*tk\.\w+Var", GUI))
    traced = set(re.findall(r"^\s+self\.(\w+_var),\s*$", GUI, re.M))
    saved = set(re.findall(r"=\s*self\.(\w+_var)\.get\(\)", GUI))

    assert not sorted(created - (traced | saved)), (
        f"controls that are never read back: {sorted(created - (traced | saved))}"
    )
    assert not sorted((traced | saved) - created), (
        f"controls read back but never created: {sorted((traced | saved) - created)}"
    )
