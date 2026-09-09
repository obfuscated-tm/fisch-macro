"""Every font in the panel is sized in pixels, so the layout survives Tk.

A positive Tk font size is a size in *points*, and the two Tk versions this
project ships against disagree about what a point is on macOS: 8.6 on Aqua
renders one as a pixel, while 9.0 converts at roughly 96dpi. The same
``("Helvetica Neue", 10)`` came out 12px of linespace under one and 16px under
the other, so the downloaded .app (which bundles Tk 8.6) and a checkout run on
a Homebrew Python (Tk 9.0) did not look like the same program -- buttons and
text were a third larger on one of them. Setting ``tk scaling`` does not fix
it; measured, the size-10 font is 16px under Tk 9 at either scaling factor.

A negative size is a size in pixels, which both versions honour exactly. Across
the 17 distinct specs the panel uses, measured under Tk 8.5.9 and Tk 9.0.3, the
pixel form differs by 0px of linespace and at most 1px of text width.

These read the source rather than run the UI, like the other GUI tests here,
so they need no display.
"""

import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

#: Every module that builds Tk widgets of its own.
UI_SOURCES = ["src/gui.py", "src/interactive_calibrator.py"]

FONT_SPEC = re.compile(r'font=\("([^"]+)",\s*(-?\d+)')


def _specs(relative_path):
    text = (ROOT / relative_path).read_text()
    return [
        (relative_path, family, int(size))
        for family, size in FONT_SPEC.findall(text)
    ]


def test_no_font_is_sized_in_points():
    offenders = [
        f"{path}: (\"{family}\", {size})"
        for path in UI_SOURCES
        for path, family, size in _specs(path)
        if size >= 0
    ]
    assert not offenders, (
        "These fonts are sized in points, which Tk 8.6 and Tk 9.0 render at "
        "different sizes. Use a negative size (pixels):\n  "
        + "\n  ".join(offenders)
    )


def test_the_panel_actually_specifies_fonts():
    """Guard the guard: a regex that matched nothing would pass silently."""
    found = [spec for path in UI_SOURCES for spec in _specs(path)]
    assert len(found) > 30, f"only found {len(found)} font specs; regex likely stale"
