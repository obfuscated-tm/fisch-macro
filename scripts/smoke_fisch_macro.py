#!/usr/bin/env python3
"""Smoke verification for the Fisch macro stabilization fixes.

Runs three small assertions against MacroEngine with fake detector/controller
objects and writes concise evidence files under .sisyphus/evidence/.
"""

from __future__ import annotations

import time
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE_DIR = ROOT / ".sisyphus" / "evidence"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import Settings
from src.macro import MacroEngine, MacroState


@dataclass
class DetectionResult:
    bar_active: bool = False
    bite_confirmed: bool = False
    fish_x: float | None = None
    bar_left: float | None = None
    bar_right: float | None = None
    on_target: bool = False
    progress: float = 0.0
    shake_pos: tuple[int, int] | None = None
    shake_confidence: float = 0.0
    # Mirrors src.detector.DetectionResult; keep these in step with it.
    debug_frame: object | None = None
    reading: object | None = None
    track_box: tuple[int, int, int, int] | None = None


class FakeDetector:
    def __init__(self, results: list[DetectionResult]):
        self._results = list(results)
        self._fallback = self._results[-1] if self._results else DetectionResult()

    def detect_all(self, scan_shake: bool = True) -> DetectionResult:
        if self._results:
            return self._results.pop(0)
        return self._fallback


class FakeController:
    def __init__(self):
        self.actions: list[str] = []
        self._killed = False

    def start(self) -> None:
        return None

    def stop(self) -> None:
        return None

    def is_killed(self) -> bool:
        return self._killed

    def mouse_hold(self) -> None:
        self.actions.append("hold")

    def mouse_release(self) -> None:
        self.actions.append("release")

    def rapid_click(self, count: int = 1, interval: float = 0.0) -> None:
        self.actions.append("rapid_click")

    def mouse_click(self, x=None, y=None) -> None:
        self.actions.append("click")

    def mouseUp(self) -> None:
        self.actions.append("mouseUp")


class FakeConfig:
    def load_settings(self) -> Settings:
        return Settings()


def make_engine(results: list[DetectionResult]) -> tuple[MacroEngine, FakeController, Settings]:
    detector = FakeDetector(results)
    controller = FakeController()
    engine = MacroEngine(detector, controller, FakeConfig())
    settings = Settings()
    engine.state = MacroState.REELING
    engine._reeling_start_time = time.time()
    engine._last_catch_time = 0.0
    engine._fail_confirm_count = 0
    engine._bar_gone_confirm_count = 0
    engine._progress_finish_count = 0
    engine._progress_finish_low_count = 0
    engine._peak_progress = 0.0
    engine._last_action_time = 0.0
    engine._last_action_type = None
    engine._last_fish_x = None
    engine._last_bar_center = None
    engine._fish_velocity = 0.0
    engine._bar_velocity = 0.0
    engine._reel_no_detection_count = 0
    engine._smoothed_progress = 0.0
    engine._trusted_peak = 0.0
    engine._raw_peak_progress = 0.0
    engine._last_progress_gain_time = time.time()
    engine._progress_collapse_count = 0
    engine._reel_stall_count = 0
    engine._post_catch_armed = True
    engine._post_catch_clear_count = 0
    engine._bite_confirm_count = 0
    return engine, controller, settings


def write_evidence(name: str, lines: list[str]) -> None:
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    (EVIDENCE_DIR / name).write_text("\n".join(lines) + "\n", encoding="utf-8")


def scenario_synthetic_shake_detected() -> None:
    """Synthetic dark-circle + white-ring pattern should score as SHAKE."""
    try:
        import cv2
        import numpy as np
    except ImportError:
        write_evidence(
            "task-5-smoke-synthetic-shake.txt",
            ["SKIP: opencv not installed in smoke environment."],
        )
        return
    from src.config import ConfigManager
    from src.detector import Detector

    # The button Detector._shake_button_score models: a dark disc inside a
    # thick white annulus, with white text across the middle. This fixture used
    # to draw a four-pixel outline instead, which the detector accepted back
    # when it took any bright contour of roughly the right size and has
    # rejected ever since it started checking the ring -- so the scenario has
    # been failing, and everything after it in this file never ran.
    #
    # NOTE: the proportions here are the detector's model of the Fisch UI, not
    # a measurement of it; no capture of a real SHAKE prompt exists in
    # tests/clips to check either against. If the macro ever misses real shake
    # prompts, this pair is the thing to re-derive from a screenshot.
    #
    # One shape is known to be outside the model: a *thin* (3px) pure-white
    # ring around a smaller dark fill scores nothing at all here. Whether that
    # is a gap in the detector or simply not what the prompt looks like cannot
    # be settled without a capture, so it is written down rather than asserted.
    radius = 52
    frame = np.zeros((400, 600, 3), dtype=np.uint8)
    cv2.circle(frame, (320, 200), radius, (230, 230, 230), -1)
    cv2.circle(frame, (320, 200), int(radius * 0.55), (25, 25, 28), -1)
    cv2.putText(
        frame, "SHAKE", (285, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (240, 240, 240), 2, cv2.LINE_AA
    )

    class _Bounds:
        x = 0
        y = 0
        width = 600
        height = 400

    detector = Detector(ConfigManager(str(ROOT)))
    detector.set_window_info(_Bounds(), scale_factor=1)
    pos, confidence = detector.detect_shake_button(frame)

    assert pos is not None, "synthetic SHAKE button should be detected"
    assert confidence >= 0.42, f"confidence too low: {confidence}"

    # Bright UI that is not a ringed button must not register. The brightness
    # gate is the part of the scorer most likely to be loosened when a real
    # SHAKE prompt is eventually captured and the model above is re-derived,
    # and these are the shapes that would start slipping through if it were
    # loosened too far.
    negatives = {}
    blob = np.zeros((400, 600, 3), dtype=np.uint8)
    cv2.circle(blob, (300, 200), 40, (245, 245, 245), -1)
    negatives["solid white blob"] = blob
    stripe = np.zeros((400, 600, 3), dtype=np.uint8)
    cv2.rectangle(stripe, (0, 180), (600, 220), (240, 240, 240), -1)
    negatives["bright stripe"] = stripe
    plates = np.zeros((400, 600, 3), dtype=np.uint8)
    for px, py in ((80, 90), (250, 300), (480, 120)):
        cv2.rectangle(plates, (px, py), (px + 70, py + 18), (235, 235, 235), -1)
    negatives["white nameplates"] = plates

    for label, neg_frame in negatives.items():
        neg_pos, neg_conf = detector.detect_shake_button(neg_frame)
        assert neg_pos is None, f"{label} must not register as SHAKE (conf={neg_conf})"

    write_evidence(
        "task-5-smoke-synthetic-shake.txt",
        [
            f"PASS: synthetic SHAKE detected at {pos} confidence={confidence:.2f}",
            f"PASS: rejected non-button bright UI: {', '.join(negatives)}",
        ],
    )


def scenario_low_confidence_shake_ignored() -> None:
    """Low-confidence detections must not click random bright UI."""
    weak = DetectionResult(shake_pos=(200, 200), shake_confidence=0.15)
    engine, controller, settings = make_engine([weak, weak])
    engine.state = MacroState.WAITING
    engine._last_shake_click_time = 0.0

    engine._do_waiting(settings)
    engine._do_waiting(settings)

    assert controller.actions.count("click") == 0, "low-confidence shake must not click"
    write_evidence(
        "task-5-smoke-shake-confidence.txt",
        ["PASS: low-confidence shake detection ignored.", f"actions={controller.actions}"],
    )


def scenario_intro_vfx_no_instant_catch() -> None:
    """Rod slash / intro VFX must not register a catch in the first seconds."""
    vfx = DetectionResult(
        bar_active=False,
        bite_confirmed=False,
        progress=0.99,
    )
    engine, controller, settings = make_engine([vfx] * 8)
    engine._reeling_start_time = time.time()
    engine._peak_progress = 0.99

    for _ in range(8):
        engine._do_reeling(settings)

    assert engine.state == MacroState.REELING, "intro VFX should not instantly complete"
    # The invariant is that no catch is registered — not that the mouse is
    # never released. With nothing detectable on the track the controller now
    # releases deliberately: an unheld bar drifts left and recovers, whereas a
    # held one accelerates into the right wall and stays there.
    assert engine.stats.fish_caught == 0, "intro VFX must not register a catch"
    assert engine.stats.fish_failed == 0, "intro VFX must not register a failure"
    write_evidence(
        "task-5-smoke-intro-vfx-no-catch.txt",
        ["PASS: intro VFX does not register instant catch.", f"actions={controller.actions}"],
    )


def scenario_premature_finish_blocked() -> None:
    result = DetectionResult(
        bar_active=True,
        bite_confirmed=True,
        fish_x=0.50,
        bar_left=0.40,
        bar_right=0.60,
        on_target=True,
        progress=0.99,
    )
    engine, controller, settings = make_engine([result, result, result])

    for _ in range(3):
        engine._do_reeling(settings)

    assert engine.state == MacroState.REELING, "guard should block early success"
    # A release no longer implies a catch: the controller releases whenever
    # it cannot see the bar, so completion is asserted on state and stats.
    assert engine.stats.fish_caught == 0, "no catch completion should fire early"
    write_evidence(
        "task-5-smoke-premature-finish.txt",
        ["PASS: premature finish is blocked during the reeling guard window.", f"actions={controller.actions}"],
    )


def scenario_flicker_hysteresis() -> None:
    flicker = DetectionResult(
        bar_active=False,
        bite_confirmed=False,
        fish_x=0.50,
        bar_left=0.40,
        bar_right=0.60,
        on_target=False,
        progress=0.0,
    )
    engine, controller, settings = make_engine([flicker, flicker, flicker, flicker])
    engine._reeling_start_time = time.time() - 3.0

    for _ in range(4):
        engine._do_reeling(settings)

    assert engine.state == MacroState.REELING, "flicker should not end the reel early"
    assert controller.actions.count("release") == 0, "failure should not trigger early"
    write_evidence(
        "task-5-smoke-flicker-hysteresis.txt",
        ["PASS: flicker hysteresis prevents early stop.", f"actions={controller.actions}"],
    )


def scenario_stable_target_no_thrash() -> None:
    stable = DetectionResult(
        bar_active=True,
        bite_confirmed=True,
        fish_x=0.50,
        bar_left=0.40,
        bar_right=0.60,
        on_target=True,
        progress=0.50,
    )
    engine, controller, settings = make_engine([stable, stable, stable, stable, stable])
    engine._reeling_start_time = time.time() - 3.0

    for _ in range(5):
        engine._do_reeling(settings)

    assert engine.state == MacroState.REELING, "stable input should keep reeling active"
    # The old macro answered a centred fish with rapid_click, treating it as a
    # "hold position" input. It is not one: a click is a brief mouse-down, so it
    # nudges the bar right every tick. This minigame has no neutral input at
    # all, so the correct steady state is alternating hold/release, and the
    # invariant worth protecting is that the bar does not run away — i.e. the
    # controller never emits an unbroken sequence of holds.
    assert "rapid_click" not in controller.actions, "clicking is not a hold-position input"
    assert set(controller.actions) <= {"hold", "release"}, controller.actions
    assert "release" in controller.actions, "a centred bar must not hold indefinitely"
    write_evidence(
        "task-5-smoke-stable-target.txt",
        ["PASS: centred fish produces bounded hold/release, never a runaway hold.",
         f"actions={controller.actions}"],
    )


def scenario_vfx_progress_spike() -> None:
    """Single-frame VFX progress spike should not count toward finish."""
    normal = DetectionResult(
        bar_active=True,
        bite_confirmed=True,
        fish_x=0.50,
        bar_left=0.40,
        bar_right=0.60,
        on_target=True,
        progress=0.50,
    )
    spike = DetectionResult(
        bar_active=True,
        bite_confirmed=True,
        fish_x=0.50,
        bar_left=0.40,
        bar_right=0.60,
        on_target=True,
        progress=0.99,
    )
    engine, controller, settings = make_engine([normal, spike, spike, normal, normal])
    engine._reeling_start_time = time.time() - 5.0
    engine._saw_midgame_progress = True

    for _ in range(5):
        engine._do_reeling(settings)

    assert engine.state == MacroState.REELING, "VFX progress spike should not finish the reel"
    assert engine._progress_finish_count == 0, "implausible spike should not advance finish debounce"
    write_evidence(
        "task-5-smoke-vfx-progress-spike.txt",
        ["PASS: implausible progress spike ignored.", f"finish_count={engine._progress_finish_count}"],
    )


def scenario_vfx_false_finish() -> None:
    """Sustained-looking VFX progress briefly but should not finish the reel."""
    # Normal progress
    normal = DetectionResult(
        bar_active=True,
        bite_confirmed=True,
        fish_x=0.50,
        bar_left=0.40,
        bar_right=0.60,
        on_target=True,
        progress=0.50,
    )
    # VFX flash - high progress but should be filtered by debounce
    vfx_flash = DetectionResult(
        bar_active=True,
        bite_confirmed=True,
        fish_x=0.50,
        bar_left=0.40,
        bar_right=0.60,
        on_target=True,
        progress=0.99,  # VFX makes it look complete
    )
    # Back to normal
    engine, controller, settings = make_engine([
        normal, normal, vfx_flash, normal, normal, normal, normal
    ])
    engine._reeling_start_time = time.time() - 5.0
    engine._saw_midgame_progress = True

    for _ in range(7):
        engine._do_reeling(settings)

    assert engine.state == MacroState.REELING, "VFX flash should not trigger premature finish"
    # A release no longer implies a catch: the controller releases whenever
    # it cannot see the bar, so completion is asserted on state and stats.
    assert engine.stats.fish_caught == 0, "no catch should fire from a VFX flash"
    write_evidence(
        "task-5-smoke-vfx-false-finish.txt",
        ["PASS: VFX false-finish scenario does not trigger premature completion.", f"actions={controller.actions}"],
    )


def scenario_auto_shake_clicks() -> None:
    """Shake prompts are clicked while waiting, before the minigame starts."""
    shake = DetectionResult(shake_pos=(400, 300), shake_confidence=0.85)
    engine, controller, settings = make_engine([shake, shake])
    engine.state = MacroState.WAITING
    engine._last_catch_time = 0.0
    engine._last_shake_click_time = 0.0

    engine._do_waiting(settings)
    engine._do_waiting(settings)

    assert controller.actions.count("click") >= 1, "shake button should be clicked"
    assert engine.state == MacroState.WAITING, "shake clicks should not start reeling"
    write_evidence(
        "task-5-smoke-auto-shake.txt",
        ["PASS: auto shake clicks while waiting.", f"actions={controller.actions}"],
    )


def scenario_midfight_bar_flicker_no_catch() -> None:
    """Brief bar hide mid-fight must not register as caught."""
    fighting = DetectionResult(
        bar_active=True,
        bite_confirmed=True,
        fish_x=0.50,
        bar_left=0.40,
        bar_right=0.60,
        progress=0.55,
    )
    flicker = DetectionResult(
        bar_active=False,
        bite_confirmed=False,
        fish_x=None,
        progress=0.0,
    )
    engine, controller, settings = make_engine([fighting, flicker, flicker, flicker, fighting])
    engine._reeling_start_time = time.time() - 5.0
    engine._peak_progress = 0.55
    engine._saw_midgame_progress = True

    for _ in range(5):
        engine._do_reeling(settings)

    assert engine.state == MacroState.REELING, "mid-fight flicker should not complete catch"
    write_evidence(
        "task-5-smoke-midfight-flicker.txt",
        ["PASS: mid-fight bar flicker does not register catch.", f"state={engine.state}"],
    )


def scenario_bar_gone_catch_with_peak() -> None:
    """Bar disappearance after high peak progress should complete the catch."""
    active = DetectionResult(
        bar_active=True,
        bite_confirmed=True,
        fish_x=0.50,
        bar_left=0.40,
        bar_right=0.60,
        on_target=True,
        progress=0.93,
    )
    gone = DetectionResult(
        bar_active=False,
        bite_confirmed=False,
        fish_x=None,
        progress=0.0,
    )
    settings = Settings()
    required = max(8, settings.bar_gone_confirm_frames // 2)
    engine, controller, _ = make_engine([active] + [gone] * required)
    engine._reeling_start_time = time.time() - 5.0
    engine._peak_progress = 0.93
    engine._prev_progress = 0.93
    engine._saw_midgame_progress = True
    engine._last_live_minigame_time = time.time() - 1.0

    for _ in range(required + 1):
        engine._do_reeling(settings)

    assert engine.state == MacroState.COMPLETE, "high peak + bar gone should finish catch"
    # The loop here keeps calling _do_reeling past completion, which the real
    # engine never does, so count at least one rather than exactly one. The
    # point is that it completed as a *catch* and not as a failure — the old
    # assertion ("release" in actions) could not tell those apart.
    assert engine.stats.fish_caught >= 1, "bar gone at high peak should count as a catch"
    assert engine.stats.fish_failed == 0, "bar gone at high peak must not count as a failure"
    write_evidence(
        "task-5-smoke-bar-gone-catch.txt",
        ["PASS: bar-gone with high peak progress completes catch.", f"state={engine.state}"],
    )


def scenario_bar_velocity_compensation() -> None:
    """Test that bar velocity creates appropriate compensation bias."""

    # Test case 1: Bar moving left (negative velocity) should bias toward hold
    # We simulate this by having changing bar positions over time
    moving_left_results = []
    bar_left_pos = 0.40
    bar_right_pos = 0.60
    for i in range(5):
        # Bar moving left: decreasing positions
        bar_left_pos -= 0.01
        bar_right_pos -= 0.01
        moving_left_results.append(DetectionResult(
            bar_active=True,
            bite_confirmed=True,
            fish_x=0.50,
            bar_left=max(bar_left_pos, 0.0),
            bar_right=max(bar_right_pos, 0.0),
            on_target=True,
            progress=0.50,
        ))

    engine, controller, settings = make_engine(moving_left_results)
    engine._reeling_start_time = time.time() - 3.0  # Past guard window

    # Execute the reeling logic multiple times
    for _ in range(len(moving_left_results)):
        engine._do_reeling(settings)

    # With bar moving left, we expect a bias toward holding (positive pd_score)
    # This should result in more hold actions than release actions
    hold_count = controller.actions.count("hold")
    release_count = controller.actions.count("release")

    # At minimum, we should not see more releases than holds due to leftward bias
    assert hold_count >= release_count, f"Expected hold bias for left-moving bar, got holds={hold_count}, releases={release_count}"

    write_evidence(
        "task-5-smoke-bar-velocity-compensation.txt",
        ["PASS: Bar velocity compensation creates appropriate bias for left-moving bar.", f"actions={controller.actions}, holds={hold_count}, releases={release_count}"],
    )

    # Test case 2: Bar moving right (positive velocity) should bias toward release
    controller.actions.clear()  # Reset actions
    moving_right_results = []
    bar_left_pos = 0.40
    bar_right_pos = 0.60
    for i in range(5):
        # Bar moving right: increasing positions
        bar_left_pos += 0.01
        bar_right_pos += 0.01
        moving_right_results.append(DetectionResult(
            bar_active=True,
            bite_confirmed=True,
            fish_x=0.50,
            bar_left=min(bar_left_pos, 1.0),
            bar_right=min(bar_right_pos, 1.0),
            on_target=True,
            progress=0.50,
        ))

    engine, controller, settings = make_engine(moving_right_results)
    engine._reeling_start_time = time.time() - 3.0  # Past guard window

    # Execute the reeling logic multiple times
    for _ in range(len(moving_right_results)):
        engine._do_reeling(settings)

    # With bar moving right, we expect a bias toward releasing (negative pd_score)
    # This should result in more release actions than hold actions
    hold_count = controller.actions.count("hold")
    release_count = controller.actions.count("release")

    # At minimum, we should not see more holds than releases due to rightward bias
    assert release_count >= hold_count, f"Expected release bias for right-moving bar, got holds={hold_count}, releases={release_count}"

    write_evidence(
        "task-5-smoke-bar-velocity-compensation.txt",
        ["PASS: Bar velocity compensation creates appropriate bias for left-moving bar.", f"actions={controller.actions}, holds={hold_count}, releases={release_count}",
         "PASS: Bar velocity compensation creates appropriate bias for right-moving bar.", f"actions={controller.actions}, holds={hold_count}, releases={release_count}"],
    )


def scenario_midgame_unlocks_from_display_peak() -> None:
    """High raw/smooth progress must unlock catch even when trusted peak lags."""
    mid = DetectionResult(
        bar_active=True,
        bite_confirmed=True,
        fish_x=0.50,
        bar_left=0.40,
        bar_right=0.60,
        on_target=True,
        progress=0.56,
    )
    engine, controller, settings = make_engine([mid, mid])
    engine._reeling_start_time = time.time() - 5.0
    engine._trusted_peak = 0.06
    engine._smoothed_progress = 0.42
    engine._raw_peak_progress = 0.56
    engine._peak_progress = 0.56

    engine._do_reeling(settings)

    assert engine._saw_midgame_progress, "display peak should unlock midgame"
    assert engine._catch_allowed(settings, 5.0), "catch should be allowed mid-fight"
    write_evidence(
        "task-5-smoke-midgame-display-peak.txt",
        ["PASS: midgame unlock uses display peak, not only trusted peak."],
    )


def scenario_progress_collapse_catch() -> None:
    """Progress collapse after a real fight should complete the catch."""
    fighting = DetectionResult(
        bar_active=True,
        bite_confirmed=True,
        fish_x=0.50,
        bar_left=0.40,
        bar_right=0.60,
        progress=0.82,
    )
    faded = DetectionResult(
        bar_active=False,
        bite_confirmed=False,
        fish_x=None,
        progress=0.05,
    )
    settings = Settings()
    required = settings.progress_collapse_confirm_frames
    engine, controller, _ = make_engine([fighting] + [faded] * required)
    engine._reeling_start_time = time.time() - 5.0
    engine._raw_peak_progress = 0.82
    engine._peak_progress = 0.82
    engine._saw_midgame_progress = True
    engine._last_live_minigame_time = time.time() - 1.0

    for _ in range(required + 1):
        engine._do_reeling(settings)

    assert engine.state == MacroState.COMPLETE, "progress collapse should finish catch"
    write_evidence(
        "task-5-smoke-progress-collapse.txt",
        ["PASS: progress collapse completes catch.", f"state={engine.state}"],
    )


def scenario_post_catch_gate_then_new_bite() -> None:
    """After a catch, require UI clear before accepting the next bite."""
    clear = DetectionResult(bar_active=False, bite_confirmed=False, progress=0.0)
    bite = DetectionResult(
        bar_active=True,
        bite_confirmed=True,
        fish_x=0.50,
        bar_left=0.40,
        bar_right=0.60,
        progress=0.20,
    )
    engine, controller, settings = make_engine([clear] * 6 + [bite, bite])
    engine.state = MacroState.WAITING
    engine._last_catch_time = time.time() - 10.0
    engine._reset_post_catch_gate()

    for _ in range(3):
        engine._do_waiting(settings)
    assert not engine._post_catch_armed, "should not arm until UI clears"

    for _ in range(settings.post_catch_clear_frames + 1):
        engine._do_waiting(settings)
    assert engine._post_catch_armed, "clear frames should arm hunting"

    engine._do_waiting(settings)
    engine._do_waiting(settings)
    assert engine.state == MacroState.REELING, "new bite should start reeling"
    write_evidence(
        "task-5-smoke-post-catch-gate.txt",
        ["PASS: post-catch clear gate then new bite starts reeling."],
    )


def scenario_instant_bite_after_cast_release() -> None:
    """Bite right after cast release should not be blocked by post-catch clear gate."""
    bite = DetectionResult(
        bar_active=True,
        bite_confirmed=True,
        fish_x=0.50,
        bar_left=0.42,
        bar_right=0.58,
        progress=0.20,
    )
    engine, controller, settings = make_engine([bite])
    engine.state = MacroState.CASTING
    engine._last_catch_time = time.time() - 10.0
    engine._reset_post_catch_gate()
    engine._arm_for_cast_bite(settings)

    assert engine._try_start_reeling(bite, settings, "after cast")
    assert engine.state == MacroState.REELING
    write_evidence(
        "task-5-smoke-instant-cast-bite.txt",
        ["PASS: instant bite after cast release starts reeling."],
    )


def scenario_still_reading_is_not_a_bite() -> None:
    """A still picture of a bar and a fish must not start a fight.

    The enchant panel draws a horizontal fill bar across the same rows as the
    reel track, and the vision reads a bar and a fish out of it on 44% of the
    non-fight frames in tests/frames/STRUGGLE-ROD2. A single frame cannot tell
    the two apart; a reading that has not changed for several ticks can, since
    the minigame is never still.
    """
    still = DetectionResult(
        bar_active=True,
        fish_x=0.50,
        bar_left=0.35,
        bar_right=0.65,
        progress=0.20,
    )
    engine, controller, settings = make_engine([still] * 40)
    engine.state = MacroState.WAITING
    engine._post_catch_armed = True
    engine._last_catch_time = 0.0

    for _ in range(20):
        engine._do_waiting(settings)
    assert engine.state != MacroState.REELING, "a frozen reading must not start a fight"

    # The same reading, once it starts moving, is a fight.
    moving = [
        DetectionResult(
            bar_active=True,
            fish_x=0.50 + 0.02 * i,
            bar_left=0.35,
            bar_right=0.65,
            progress=0.20,
        )
        for i in range(6)
    ]
    engine, controller, settings = make_engine(moving)
    engine.state = MacroState.WAITING
    engine._post_catch_armed = True
    engine._last_catch_time = 0.0
    engine._do_waiting(settings)
    assert engine.state != MacroState.REELING, "one frame cannot yet show movement"
    engine._do_waiting(settings)
    assert engine.state == MacroState.REELING, "a moving reading starts a fight on the second tick"

    write_evidence(
        "task-5-smoke-still-reading.txt",
        [
            "PASS: still bar+fish refused indefinitely; moving bar+fish starts",
            "reeling on the second tick (~20ms later at the default interval).",
        ],
    )


def scenario_bite_confirmed_without_bar_bounds() -> None:
    """Fish + bite_confirmed should start reeling before bar bounds lock in."""
    partial = DetectionResult(
        bar_active=True,
        bite_confirmed=True,
        fish_x=0.52,
        progress=0.15,
    )
    engine, controller, settings = make_engine([partial, partial])
    engine.state = MacroState.WAITING
    engine._post_catch_armed = True
    engine._last_catch_time = 0.0

    engine._do_waiting(settings)
    engine._do_waiting(settings)

    assert engine.state == MacroState.REELING
    write_evidence(
        "task-5-smoke-bite-confirmed-start.txt",
        ["PASS: bite_confirmed path starts reeling without bar bounds."],
    )


def scenario_calibration_rejects_black_slide_frame() -> None:
    """Fullscreen slide artifacts should score worse than a normal frame."""
    try:
        import numpy as np

        from src.calibrator import Calibrator
    except ImportError:
        write_evidence(
            "task-5-smoke-calibration-artifact.txt",
            ["SKIP: opencv not installed in test environment."],
        )
        return

    good = np.full((200, 400, 3), 120, dtype=np.uint8)
    bad = good.copy()
    bad[:, :120] = 0

    good_score = Calibrator.transition_artifact_score(good)
    bad_score = Calibrator.transition_artifact_score(bad)
    assert bad_score > good_score + 0.15, (
        f"black strip should score worse (good={good_score}, bad={bad_score})"
    )
    write_evidence(
        "task-5-smoke-calibration-artifact.txt",
        ["PASS: calibration artifact scorer rejects black slide frames."],
    )


def scenario_stationary_no_overpredict() -> None:
    """Still fish should use current position, not lookahead blend."""
    settings = Settings()
    engine, _, _ = make_engine([])
    engine._fish_velocity = 0.0

    for _ in range(12):
        engine._fish_tracker.add_sample(0.50)
        time.sleep(0.015)

    predicted = engine._predicted_fish_x(0.50, settings)
    assert abs(predicted - 0.50) < 0.008, (
        f"stationary fish should not over-predict, got {predicted}"
    )

    engine._fish_tracker.reset()
    for i in range(6):
        engine._fish_tracker.add_sample(0.30 + i * 0.02)
        time.sleep(0.02)
    for _ in range(6):
        engine._fish_tracker.add_sample(0.46)
        time.sleep(0.02)
    engine._fish_velocity = 0.0
    after_stop = engine._predicted_fish_x(0.46, settings)
    assert abs(after_stop - 0.46) < 0.02, (
        f"fish that stopped moving should not keep old momentum, got {after_stop}"
    )
    write_evidence(
        "task-5-smoke-stationary-predict.txt",
        ["PASS: stationary / stopped fish avoids over-prediction."],
    )


def scenario_left_stall_recovery_holds() -> None:
    """Off-target bar pinned left should hold to chase fish on the right."""
    stalled = DetectionResult(
        bar_active=True,
        bite_confirmed=True,
        fish_x=0.35,
        bar_left=0.05,
        bar_right=0.22,
        on_target=False,
        progress=0.50,
    )
    engine, controller, settings = make_engine([stalled, stalled, stalled])
    engine._reeling_start_time = time.time() - 5.0
    engine._last_action_time = 0.0
    engine._last_action_type = "release"

    for _ in range(3):
        engine._do_reeling(settings)

    assert controller.actions.count("hold") >= 1, "left stall should issue hold to recover"
    write_evidence(
        "task-5-smoke-left-stall-recovery.txt",
        ["PASS: left stall recovery issues hold.", f"actions={controller.actions}"],
    )


def main() -> int:
    scenario_intro_vfx_no_instant_catch()
    scenario_premature_finish_blocked()
    scenario_flicker_hysteresis()
    scenario_stable_target_no_thrash()
    scenario_synthetic_shake_detected()
    scenario_auto_shake_clicks()
    scenario_low_confidence_shake_ignored()
    scenario_vfx_progress_spike()
    scenario_vfx_false_finish()
    scenario_midfight_bar_flicker_no_catch()
    scenario_bar_gone_catch_with_peak()
    scenario_midgame_unlocks_from_display_peak()
    scenario_progress_collapse_catch()
    scenario_post_catch_gate_then_new_bite()
    scenario_instant_bite_after_cast_release()
    scenario_still_reading_is_not_a_bite()
    scenario_bite_confirmed_without_bar_bounds()
    scenario_calibration_rejects_black_slide_frame()
    scenario_stationary_no_overpredict()
    scenario_left_stall_recovery_holds()
    scenario_bar_velocity_compensation()
    write_evidence(
        "task-5-smoke-overall.txt",
        [
            "PASS: all smoke scenarios passed.",
            "- intro VFX no instant catch",
            "- premature finish blocked",
            "- flicker hysteresis held",
            "- stable target did not thrash",
            "- auto shake clicks while waiting",
            "- low-confidence shake ignored",
            "- VFX progress spike ignored",
            "- VFX false-finish blocked",
            "- mid-fight flicker does not false-catch",
            "- bar-gone catch with peak progress",
            "- midgame unlock from display peak",
            "- progress collapse catch",
            "- stationary fish no over-predict",
            "- post-catch gate arms new bite",
            "- bite_confirmed starts reeling",
            "- instant bite after cast release",
            "- left stall recovery hold",
            "- bar velocity compensation working",
        ],
    )
    print("All Fisch macro smoke checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
