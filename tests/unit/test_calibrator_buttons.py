"""The calibration toolbar's buttons have to be visible and readable.

Two separate ways they were not. macOS draws a tk.Button itself and ignores the
background it is given, so buttons asking for the panel's accent colours came
out as plain white rectangles -- with white text on them. Captured under Tk
9.0.3, a tk.Button asking for #e94560 renders white with or without a flat
relief and no border, while the same colour on a ttk button under the "clam"
theme draws correctly, because clam paints the button rather than asking the
system to.

And the colours themselves were too pale for white text once drawn: the
box-drawing steps sat at 2.0:1 against white, and the selected one at 1.5:1,
where WCAG AA asks for 4.5:1 on text this size.
"""

import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.interactive_calibrator import InteractiveCalibrator  # noqa: E402

SOURCE = (ROOT / "src" / "interactive_calibrator.py").read_text()

#: WCAG 2.1 AA for text below 18pt.
AA_CONTRAST = 4.5


def _relative_luminance(colour):
    channels = [int(colour[i:i + 2], 16) / 255 for i in (1, 3, 5)]
    linear = [
        c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4
        for c in channels
    ]
    return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]


def _contrast(foreground, background):
    lighter, darker = sorted(
        (_relative_luminance(foreground), _relative_luminance(background)),
        reverse=True,
    )
    return (lighter + 0.05) / (darker + 0.05)


def test_every_button_colour_is_readable_with_white_text():
    failures = [
        f"{name} {background} is {_contrast('#ffffff', background):.2f}:1"
        for name, background, _active in InteractiveCalibrator.BUTTON_STYLES
        if _contrast("#ffffff", background) < AA_CONTRAST
    ]
    assert not failures, (
        f"white text needs {AA_CONTRAST}:1 to stay readable:\n  "
        + "\n  ".join(failures)
    )


def test_the_hover_colours_are_readable_too():
    failures = [
        f"{name} active {active} is {_contrast('#ffffff', active):.2f}:1"
        for name, _background, active in InteractiveCalibrator.BUTTON_STYLES
        if _contrast("#ffffff", active) < AA_CONTRAST
    ]
    assert not failures, "\n  ".join(failures)


def base_has_selected(name):
    return any(n == name + "On" for n, _b, _a in InteractiveCalibrator.BUTTON_STYLES)


def test_a_selected_step_is_darker_than_an_unselected_one():
    """It used to be lighter, which is where the 1.5:1 came from."""
    styles = dict(
        (name, background)
        for name, background, _active in InteractiveCalibrator.BUTTON_STYLES
    )
    for base in [n for n, _b, _a in InteractiveCalibrator.BUTTON_STYLES
                 if not n.endswith("On") and base_has_selected(n)]:
        assert _relative_luminance(styles[base + "On"]) < _relative_luminance(
            styles[base]
        ), f"{base}On should be the darker shade"


def test_the_toolbar_uses_ttk_buttons():
    """A tk.Button here is a white rectangle on macOS, whatever colour it asks for."""
    # Anchored so it does not match the "tk.Button(" inside "ttk.Button(".
    plain = re.findall(r"(?<!t)tk\.Button\(", SOURCE)
    assert not plain, (
        "macOS ignores a tk.Button's background; use a clam-themed ttk.Button"
    )
    assert SOURCE.count("ttk.Button(") >= 4
