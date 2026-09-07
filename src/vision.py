"""Live vision snapshot for GUI debug overlay (digmacro-style preview)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np


@dataclass
class VisionSnapshot:
    """Thread-safe copy of what the macro sees during the last detection tick."""

    frame_bgr: Optional[np.ndarray] = None
    fish_x: Optional[float] = None
    predicted_fish_x: Optional[float] = None
    bar_left: Optional[float] = None
    bar_right: Optional[float] = None
    bar_center: Optional[float] = None
    effective_bar: Optional[float] = None
    progress_raw: float = 0.0
    progress_smooth: float = 0.0
    peak_progress: float = 0.0
    on_target: bool = False
    pd_score: float = 0.0
    error: float = 0.0
    fish_velocity: float = 0.0
    bar_velocity: float = 0.0
    last_action: str = ""
    catch_allowed: bool = False
    macro_state: str = ""
    extras: Dict[str, Any] = field(default_factory=dict)

    def summary_lines(self) -> list[str]:
        """Human-readable telemetry for the GUI."""
        lines = [
            f"State: {self.macro_state}",
            f"Fish: {self._fmt(self.fish_x)}  →  Predicted: {self._fmt(self.predicted_fish_x)}",
            f"Bar: {self._fmt(self.bar_left)}–{self._fmt(self.bar_right)}  "
            f"center {self._fmt(self.bar_center)}  eff {self._fmt(self.effective_bar)}",
            f"Target: {'ON' if self.on_target else 'OFF'}  "
            f"err {self.error:+.3f}  pd {self.pd_score:+.3f}  act {self.last_action or '—'}",
            f"Prog raw {self.progress_raw:.0%}  smooth {self.progress_smooth:.0%}  "
            f"peak {self.peak_progress:.0%}  catch_ok {self.catch_allowed}",
            f"v fish {self.fish_velocity:+.3f}  v bar {self.bar_velocity:+.3f}",
        ]
        hunt = self.extras.get("hunt")
        if hunt:
            lines.append(f"Hunt: {hunt}")
        geom = self.extras.get("geom_on_target")
        if geom is not None:
            color = "ON" if self.on_target else "OFF"
            lines.append(
                f"Align: geom {'ON' if geom else 'OFF'}  color {color}  "
                f"signal {self.extras.get('signal', '—')}"
            )
        for key, value in self.extras.items():
            if key in ("hunt", "geom_on_target", "signal", "color_on_target"):
                continue
            lines.append(f"{key}: {value}")
        return lines

    @staticmethod
    def _fmt(value: Optional[float]) -> str:
        return f"{value:.2f}" if value is not None else "—"
