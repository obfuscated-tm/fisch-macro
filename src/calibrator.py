"""
calibrator.py — Interactive calibration tools for the Fisch fishing macro.

Provides utilities for:
  • Full-screen capture for calibration overlays
  • Automatic colour-profile detection via K-means clustering
  • Converting user-drawn pixel rectangles to normalised ROI bounds
  • Generating mask-preview images for HSV threshold tuning
"""

import logging
import time
from typing import Optional, Tuple

import cv2
import numpy as np


class Calibrator:
    """Calibration helper that works alongside a :class:`Detector` instance.

    Parameters
    ----------
    detector : Detector
        The active detector (used for screen-capture utilities).
    config_manager : ConfigManager
        The configuration manager (used to read / write settings).
    """

    def __init__(self, detector, config_manager):
        self._detector = detector
        self._config = config_manager
        self.logger = logging.getLogger("calibrator")

    # ------------------------------------------------------------------
    # Screen capture
    # ------------------------------------------------------------------

    def capture_full_screenshot(self) -> Optional[np.ndarray]:
        """Capture the entire primary monitor and return a BGR image.

        Returns
        -------
        np.ndarray or None
            Full-screen BGR image, or *None* on failure.
        """
        try:
            frame = self._grab_primary_monitor_bgr()
            if frame is not None:
                self.logger.info(
                    "Full screenshot captured — %dx%d", frame.shape[1], frame.shape[0]
                )
            return frame
        except Exception as exc:
            self.logger.error("Failed to capture full screenshot: %s", exc)
            return None

    @staticmethod
    def transition_artifact_score(frame: np.ndarray) -> float:
        """Score how likely a frame is a macOS fullscreen slide (black bars).

        Higher = worse. Values above ~0.35 usually mean the capture is unusable.
        """
        if frame is None or frame.size == 0:
            return 1.0
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape[:2]
        if w < 4 or h < 4:
            return 1.0

        dark = gray < 14
        overall_dark = float(np.mean(dark))

        # Space-change animation often blacks out a large vertical strip on one side
        quarter = max(1, w // 4)
        left_dark = float(np.mean(dark[:, :quarter]))
        right_dark = float(np.mean(dark[:, -quarter:]))
        side_dark = max(left_dark, right_dark)

        if side_dark > 0.45 and side_dark > overall_dark + 0.12:
            return side_dark
        return overall_dark * 0.65

    def capture_stable_screenshot(
        self,
        wait_seconds: float = 0.2,
        max_attempts: int = 6,
        retry_delay: float = 0.18,
        max_artifact_score: float = 0.32,
    ) -> Optional[np.ndarray]:
        """Capture after UI settles; reject fullscreen slide black-bar frames."""
        if wait_seconds > 0:
            time.sleep(wait_seconds)

        best_frame = None
        best_score = 1.0
        for attempt in range(max_attempts):
            frame = self._grab_primary_monitor_bgr()
            if frame is None:
                time.sleep(retry_delay)
                continue
            score = self.transition_artifact_score(frame)
            self.logger.debug(
                "Calibration capture attempt %d/%d artifact=%.3f",
                attempt + 1,
                max_attempts,
                score,
            )
            if score < best_score:
                best_score = score
                best_frame = frame
            if score <= max_artifact_score:
                self.logger.info(
                    "Stable screenshot captured — %dx%d (artifact=%.3f)",
                    frame.shape[1],
                    frame.shape[0],
                    score,
                )
                return frame
            time.sleep(retry_delay)

        if best_frame is not None and best_score < 0.5:
            self.logger.warning(
                "Using best calibration screenshot (artifact=%.3f)", best_score
            )
            return best_frame
        self.logger.error(
            "Calibration screenshots looked like transition black frames (best=%.3f)",
            best_score,
        )
        return None

    def _grab_primary_monitor_bgr(self) -> Optional[np.ndarray]:
        import mss

        with mss.mss() as sct:
            monitor = sct.monitors[1]
            raw = sct.grab(monitor)
            frame = np.array(raw)
            return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

    # ------------------------------------------------------------------
    # Auto colour detection
    # ------------------------------------------------------------------

    def compute_roi_from_rect(
        self,
        rect: Tuple[int, int, int, int],
        window_bounds,
    ) -> dict:
        """Convert a pixel rectangle to normalised ROI bounds.

        Parameters
        ----------
        rect : tuple of int
            ``(x, y, w, h)`` in screen pixels, as drawn by the user.
        window_bounds : object
            Must expose ``.x``, ``.y``, ``.width``, ``.height``.

        Returns
        -------
        dict
            Keys: ``x_start``, ``x_end``, ``y_start``, ``y_end`` — each a
            float clamped to 0.0–1.0.
        """
        rx, ry, rw, rh = rect
        settings = self._config.load_settings()
        wx = window_bounds.x + window_bounds.width * settings.window_inset_left
        wy = window_bounds.y + window_bounds.height * settings.window_inset_top
        ww = max(1, window_bounds.width * (1.0 - settings.window_inset_left))
        wh = max(1, window_bounds.height * (1.0 - settings.window_inset_top))

        if ww <= 0 or wh <= 0:
            self.logger.error("Invalid window bounds: %s", window_bounds)
            return {"x_start": 0.0, "x_end": 1.0, "y_start": 0.0, "y_end": 1.0}

        x_start = float(np.clip((rx - wx) / ww, 0.0, 1.0))
        x_end = float(np.clip((rx + rw - wx) / ww, 0.0, 1.0))
        y_start = float(np.clip((ry - wy) / wh, 0.0, 1.0))
        y_end = float(np.clip((ry + rh - wy) / wh, 0.0, 1.0))

        self.logger.info(
            "ROI from rect (%d,%d,%d,%d) → x=[%.3f, %.3f]  y=[%.3f, %.3f]",
            rx, ry, rw, rh, x_start, x_end, y_start, y_end,
        )
        return {
            "x_start": x_start,
            "x_end": x_end,
            "y_start": y_start,
            "y_end": y_end,
        }

    # ------------------------------------------------------------------
    # Mask preview
    # ------------------------------------------------------------------

