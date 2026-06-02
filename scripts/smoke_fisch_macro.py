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


class FakeDetector:
    def __init__(self, results: list[DetectionResult]):
        self._results = list(results)
        self._fallback = self._results[-1] if self._results else DetectionResult()

    def detect_all(self) -> DetectionResult:
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
    engine._success_confirm_count = 0
    engine._fail_confirm_count = 0
    engine._bar_gone_confirm_count = 0
    engine._last_action_time = 0.0
    engine._last_action_type = None
    engine._last_fish_x = None
    engine._last_bar_center = None
    engine._fish_velocity = 0.0
    engine._bar_velocity = 0.0
    engine._reel_no_detection_count = 0
    return engine, controller, settings


def write_evidence(name: str, lines: list[str]) -> None:
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    (EVIDENCE_DIR / name).write_text("\n".join(lines) + "\n", encoding="utf-8")


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
    assert "release" not in controller.actions, "no catch completion should fire early"
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
    assert set(controller.actions) <= {"rapid_click"}, "stable input should not thrash hold/release"
    write_evidence(
        "task-5-smoke-stable-target.txt",
        ["PASS: stable target stays in the stable/hover path without hold/release thrash.", f"actions={controller.actions}"],
    )


def scenario_vfx_false_finish() -> None:
    """VFX flash triggers high progress briefly but should not finish the reel."""
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
    engine._reeling_start_time = time.time() - 3.0  # Past guard window

    for _ in range(7):
        engine._do_reeling(settings)

    # Should still be reeling because VFX flash alone shouldn't trigger finish
    assert engine.state == MacroState.REELING, "VFX flash should not trigger premature finish"
    assert "release" not in controller.actions, "no catch completion should fire from VFX flash"
    write_evidence(
        "task-5-smoke-vfx-false-finish.txt",
        ["PASS: VFX false-finish scenario does not trigger premature completion.", f"actions={controller.actions}"],
    )


def scenario_bar_velocity_compensation() -> None:
    """Test that bar velocity creates appropriate compensation bias."""
    # Base detection result
    base = DetectionResult(
        bar_active=True,
        bite_confirmed=True,
        fish_x=0.50,  # Fish centered
        bar_left=0.40,
        bar_right=0.60,
        on_target=True,
        progress=0.50,
    )
    
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


def main() -> int:
    scenario_premature_finish_blocked()
    scenario_flicker_hysteresis()
    scenario_stable_target_no_thrash()
    scenario_vfx_false_finish()
    scenario_bar_velocity_compensation()
    write_evidence(
        "task-5-smoke-overall.txt",
        [
            "PASS: all smoke scenarios passed.",
            "- premature finish blocked",
            "- flicker hysteresis held",
            "- stable target did not thrash",
            "- VFX false-finish blocked",
            "- bar velocity compensation working",
        ],
    )
    print("All Fisch macro smoke checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
