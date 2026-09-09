"""The Settings tab has to stay small without quietly losing controls.

Making a panel compact has an easy cheat — delete things — and a real fix:
put what gets touched every session at the top and fold the tuning knobs
away underneath. These tests hold both halves. They read the source rather
than run the UI, like the other GUI tests here, so they need no display.
"""

import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

GUI = (ROOT / "src" / "gui.py").read_text()

SETTINGS_TAB = GUI[
    GUI.index("def _build_settings_tab") : GUI.index("def _bind_mousewheel")
]

ADVANCED_AT = SETTINGS_TAB.index('_add_section_header(content, "Advanced")')

#: Everything the panel offered before it was compacted. Shortened wording is
#: fine; a control disappearing is not.
EVERYDAY = [
    "Cast Hold Time",
    "Recast Delay",
    "Scan Interval",
    "Auto Recast",
    "Auto Shake",
    "Live Vision Preview",
    "Start/Stop Key",
]

ADVANCED = [
    "Steering Strength",
    "Steering Trim",
    "Neutral Hold",
    "Fish Lookahead",
    "Stationary Speed",
    "Bar Acceleration",
    "Input Latency",
    "Max Brake Distance",
    "Reel Start Guard",
    "Min Catch Time",
    "Fast Rod Min Reel",
    "Post-Catch Lockout",
    "Min Catch Progress",
    "Finish Progress",
    "ROI Shift X",
    "ROI Shift Y",
    "Window Inset Top",
    "Window Inset Left",
]


def test_no_control_was_dropped_on_the_way_to_a_smaller_panel():
    missing = [name for name in EVERYDAY + ADVANCED if f'"{name}"' not in SETTINGS_TAB]
    assert not missing, f"controls that vanished from the Settings tab: {missing}"


def test_everyday_controls_come_before_the_advanced_sections():
    late = [
        name
        for name in EVERYDAY
        if SETTINGS_TAB.index(f'"{name}"') > ADVANCED_AT
    ]
    assert not late, f"these are used every session and must stay on top: {late}"


def test_tuning_controls_live_inside_fold_out_sections():
    """Nothing after the Advanced header may be packed straight onto the page.

    That is what keeps the tab a screen tall: the eighteen tuning sliders take
    up three header rows until someone opens one.
    """
    bodies = set(re.findall(r"(\w+) = self\._add_collapsible\(", SETTINGS_TAB))
    assert bodies, "the advanced sections are no longer collapsible"

    loose = [
        parent
        for parent in re.findall(
            r"self\._add_(?:slider|toggle)\(\s*(\w+),", SETTINGS_TAB[ADVANCED_AT:]
        )
        if parent not in bodies
    ]
    assert not loose, f"advanced controls packed outside a fold-out: {loose}"


def test_every_advanced_control_is_reachable_from_some_section():
    for name in ADVANCED:
        head = SETTINGS_TAB.rindex("self._add_collapsible(", 0, SETTINGS_TAB.index(f'"{name}"'))
        assert head > ADVANCED_AT, f'"{name}" is not under a fold-out section'


def test_fold_out_sections_start_closed():
    assert "def _add_collapsible(self, parent, title, expanded=False)" in GUI
    assert "_add_collapsible(content, " in SETTINGS_TAB
    assert "expanded=True" not in SETTINGS_TAB


def test_sliders_show_as_many_decimals_as_they_can_step():
    """A 0.005-step slider read as one decimal and looked stuck while dragging."""
    body = GUI[GUI.index("def _add_slider") : GUI.index("def _add_toggle")]
    assert "decimals" in body, "_add_slider ignores its resolution argument again"
    assert re.search(r"\{val:\.\{decimals\}f\}", body)
