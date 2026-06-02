"""
detector.py — Core detection engine for the Fisch fishing macro.

Captures the screen via mss and uses OpenCV to detect game elements:
  • Fishing minigame bar activation
  • Fish indicator position (normalized 0–1)
  • Control-bar left/right boundaries (normalized 0–1)
  • Progress-bar fill (normalized 0–1)
  • Shake-button screen coordinates

All colour detection uses HSV space with configurable thresholds loaded from
the ConfigManager, and a VFX filter strips bright flashes / particle effects
before analysis.
"""

import logging
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class DetectionResult:
    """Result bundle returned by :meth:`Detector.detect_all`."""

    bar_active: bool = False
    fish_x: Optional[float] = None       # 0.0–1.0 normalized position in bar
    bar_left: Optional[float] = None     # 0.0–1.0 control bar left edge
    bar_right: Optional[float] = None    # 0.0–1.0 control bar right edge
    progress: float = 0.0                # 0.0–1.0 progress bar fill
    shake_pos: Optional[Tuple[int, int]] = None  # (x, y) screen coords of shake button
    debug_frame: Optional[np.ndarray] = None     # Annotated frame for GUI


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------

class Detector:
    """Screen-capture + OpenCV detection engine.

    Parameters
    ----------
    config_manager : ConfigManager
        Provides ``load_settings()`` → Settings and ``get_active_profile()``
        → ColorProfile.
    """

    def __init__(self, config_manager):
        self._config = config_manager
        self._mss = None           # lazy-initialised mss instance
        self._window_bounds = None  # set via set_window_info()
        self._scale_factor = 1     # Retina = 2, standard = 1
        self.debug_mode = False
        self.logger = logging.getLogger("detector")

    # ------------------------------------------------------------------
    # Window / monitor helpers
    # ------------------------------------------------------------------

    def set_window_info(self, bounds, scale_factor: int):
        """Set the Roblox window bounds and display scale factor.

        Parameters
        ----------
        bounds : object
            Must have ``.x``, ``.y``, ``.width``, ``.height`` attributes
            (screen points, *not* pixels).
        scale_factor : int
            1 for standard displays, 2 for Retina.
        """
        self._window_bounds = bounds
        self._scale_factor = scale_factor
        self.logger.info(
            "Window info set — origin=(%s, %s)  size=%sx%s  scale=%s",
            bounds.x, bounds.y, bounds.width, bounds.height, scale_factor,
        )

    # ------------------------------------------------------------------
    # mss helpers
    # ------------------------------------------------------------------

    def _get_mss(self):
        """Lazy-init and return the ``mss`` screenshot instance."""
        if self._mss is None:
            import mss
            self._mss = mss.mss()
        return self._mss

    # ------------------------------------------------------------------
    # ROI helpers
    # ------------------------------------------------------------------

    def _compute_roi_pixels(self, roi_bounds) -> Optional[dict]:
        """Convert normalised ROI bounds to absolute pixel coordinates.

        Parameters
        ----------
        roi_bounds : object
            Must expose ``.x_start``, ``.x_end``, ``.y_start``, ``.y_end``
            (each 0.0–1.0 relative to the window).

        Returns
        -------
        dict or None
            Keys ``left``, ``top``, ``width``, ``height`` in pixels, or
            *None* if window bounds have not been set yet.
        """
        if self._window_bounds is None:
            return None

        wb = self._window_bounds
        scale = self._scale_factor
        return {
            "left": int((wb.x + wb.width * roi_bounds.x_start) * scale),
            "top": int((wb.y + wb.height * roi_bounds.y_start) * scale),
            "width": int(wb.width * (roi_bounds.x_end - roi_bounds.x_start) * scale),
            "height": int(wb.height * (roi_bounds.y_end - roi_bounds.y_start) * scale),
        }

    def capture_roi(self, roi_bounds) -> Optional[np.ndarray]:
        """Capture a region of the screen and return it as a BGR frame.

        Parameters
        ----------
        roi_bounds : object
            Normalised ROI bounds (see :meth:`_compute_roi_pixels`).

        Returns
        -------
        np.ndarray or None
            BGR image, or *None* if the ROI is invalid / empty.
        """
        roi_pixels = self._compute_roi_pixels(roi_bounds)
        if roi_pixels is None:
            self.logger.debug("Cannot capture ROI — window bounds not set")
            return None

        if roi_pixels["width"] <= 0 or roi_pixels["height"] <= 0:
            self.logger.debug("Invalid ROI dimensions: %s", roi_pixels)
            return None

        try:
            raw = self._get_mss().grab(roi_pixels)
            frame = np.array(raw)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            return frame
        except Exception as exc:
            self.logger.warning("Screen grab failed: %s", exc)
            return None

    # ------------------------------------------------------------------
    # VFX filtering
    # ------------------------------------------------------------------

    def _filter_vfx(self, hsv_frame: np.ndarray) -> np.ndarray:
        """Return a uint8 mask that *removes* extremely bright / flash pixels.

        Pixels marked 0 are considered VFX artefacts; pixels marked 255 are
        safe to analyse.
        """
        v_channel = hsv_frame[:, :, 2]
        s_channel = hsv_frame[:, :, 1]

        # Very bright pixels (particle effects, glowing edges)
        bright_mask = v_channel > 240

        # Low-saturation bright pixels (white flashes / bloom)
        flash_mask = (v_channel > 200) & (s_channel < 30)

        vfx_mask = bright_mask | flash_mask
        return (~vfx_mask).astype(np.uint8) * 255

    # ------------------------------------------------------------------
    # Individual detectors
    # ------------------------------------------------------------------

    def detect_bar_active(self, frame: np.ndarray) -> bool:
        """Determine whether the fishing minigame bar is currently on-screen.

        Uses a combination of mean brightness (the bar overlay darkens the
        region) and Canny edge detection (strong horizontal lines from the
        bar's borders).
        """
        if frame is None or frame.size == 0:
            return False

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mean_brightness = float(np.mean(gray))

        profile = self._config.get_active_profile()
        threshold = profile.bar_brightness_threshold

        if mean_brightness >= threshold:
            return False

        # Look for strong horizontal edges (bar borders)
        edges = cv2.Canny(gray, 50, 150)
        row_sums = np.sum(edges > 0, axis=1)
        # A "strong" row has edge pixels across ≥30 % of the width
        strong_rows = np.sum(row_sums > (frame.shape[1] * 0.3))

        if strong_rows >= 2:
            self.logger.debug(
                "Bar active — brightness=%.1f (<%.0f), strong_rows=%d",
                mean_brightness, threshold, strong_rows,
            )
            return True

        return False

    # ---- fish indicator ------------------------------------------------

    def detect_fish_x(self, frame: np.ndarray) -> Optional[float]:
        """Return the normalised X position of the fish indicator (0–1).

        The fish indicator is a distinctly coloured element (pink / magenta)
        whose HSV range is defined in the active colour profile.
        """
        if frame is None or frame.size == 0:
            return None

        profile = self._config.get_active_profile()
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        vfx_ok = self._filter_vfx(hsv)

        color_mask = cv2.inRange(
            hsv,
            np.array(profile.fish_hsv_low, dtype=np.uint8),
            np.array(profile.fish_hsv_high, dtype=np.uint8),
        )
        mask = cv2.bitwise_and(color_mask, vfx_ok)

        # Morphological cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        # Largest contour that passes noise filter
        valid = [c for c in contours if cv2.contourArea(c) > 20]
        if not valid:
            return None

        largest = max(valid, key=cv2.contourArea)
        M = cv2.moments(largest)
        if M["m00"] == 0:
            return None

        cx = int(M["m10"] / M["m00"])
        fish_x = cx / frame.shape[1]
        return float(np.clip(fish_x, 0.0, 1.0))

    # ---- control bar (arrows / zone) -----------------------------------

    def detect_control_bar(self, frame: np.ndarray) -> Optional[Tuple[float, float]]:
        """Return ``(left, right)`` normalised boundaries of the control zone.

        The control zone is bounded by green / cyan arrow markers whose HSV
        range is defined in the active colour profile.  Falls back to a
        brightness-based heuristic if colour matching fails.
        """
        if frame is None or frame.size == 0:
            return None

        profile = self._config.get_active_profile()
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        vfx_ok = self._filter_vfx(hsv)

        color_mask = cv2.inRange(
            hsv,
            np.array(profile.bar_hsv_low, dtype=np.uint8),
            np.array(profile.bar_hsv_high, dtype=np.uint8),
        )
        mask = cv2.bitwise_and(color_mask, vfx_ok)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        w = frame.shape[1]

        if len(contours) >= 2:
            # Two largest contours → left / right arrow markers
            sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)[:2]
            centroids = []
            for c in sorted_contours:
                M = cv2.moments(c)
                if M["m00"] == 0:
                    continue
                centroids.append(int(M["m10"] / M["m00"]))
            if len(centroids) == 2:
                left = min(centroids) / w
                right = max(centroids) / w
                return (float(np.clip(left, 0.0, 1.0)),
                        float(np.clip(right, 0.0, 1.0)))

        if len(contours) == 1:
            x, _, cw, _ = cv2.boundingRect(contours[0])
            left = x / w
            right = (x + cw) / w
            return (float(np.clip(left, 0.0, 1.0)),
                    float(np.clip(right, 0.0, 1.0)))

        # ---- Brightness fallback ----
        return self._brightness_fallback(frame)

    def _brightness_fallback(self, frame: np.ndarray) -> Optional[Tuple[float, float]]:
        """Attempt to locate the control zone via a sliding-window brightness scan."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        if w < 10:
            return None

        col_brightness = np.mean(gray, axis=0)  # mean brightness per column
        overall_mean = np.mean(col_brightness)

        # Find contiguous bright band (≥1.5× average)
        bright_cols = col_brightness > (overall_mean * 1.5)
        if not np.any(bright_cols):
            return None

        # Find runs of bright columns
        diffs = np.diff(bright_cols.astype(int))
        starts = np.where(diffs == 1)[0] + 1
        ends = np.where(diffs == -1)[0] + 1

        # Handle edge cases where the run starts at 0 or ends at w
        if bright_cols[0]:
            starts = np.concatenate(([0], starts))
        if bright_cols[-1]:
            ends = np.concatenate((ends, [w]))

        if len(starts) == 0 or len(ends) == 0:
            return None

        # Pick the longest bright run
        run_lengths = ends[:len(starts)] - starts[:len(ends)]
        if len(run_lengths) == 0:
            return None
        best_idx = int(np.argmax(run_lengths))

        left = starts[best_idx] / w
        right = ends[best_idx] / w
        if right - left < 0.05:
            return None  # too narrow to be meaningful

        self.logger.debug("Brightness fallback bar: %.2f–%.2f", left, right)
        return (float(np.clip(left, 0.0, 1.0)),
                float(np.clip(right, 0.0, 1.0)))

    # ---- progress bar --------------------------------------------------

    def detect_progress(self, progress_frame: np.ndarray) -> float:
        """Return progress-bar fill as a float 0.0–1.0.

        Scans columns left-to-right and finds the rightmost column that still
        contains a significant number of coloured (non-dark) pixels.
        """
        if progress_frame is None or progress_frame.size == 0:
            return 0.0

        hsv = cv2.cvtColor(progress_frame, cv2.COLOR_BGR2HSV)
        h, w = hsv.shape[:2]

        # Require both brightness and some saturation
        v_ok = hsv[:, :, 2] > 50
        s_ok = hsv[:, :, 1] > 30
        mask = v_ok & s_ok

        # Minimum pixel count per column to be considered "filled"
        min_pixels = max(1, int(h * 0.15))

        rightmost = -1
        col_sums = np.sum(mask, axis=0)  # vectorised column counts
        for col in range(w):
            if col_sums[col] >= min_pixels:
                rightmost = col

        if rightmost < 0:
            return 0.0

        progress = (rightmost + 1) / w
        return float(np.clip(progress, 0.0, 1.0))

    # ---- shake button --------------------------------------------------

    def detect_shake_button(self, shake_frame: np.ndarray) -> Optional[Tuple[int, int]]:
        """Detect the SHAKE button and return its centre in screen coordinates.

        Returns ``(screen_x, screen_y)`` or *None* if the button is not
        visible.
        """
        if shake_frame is None or shake_frame.size == 0:
            return None

        hsv = cv2.cvtColor(shake_frame, cv2.COLOR_BGR2HSV)
        v_channel = hsv[:, :, 2]
        s_channel = hsv[:, :, 1]

        mask = ((v_channel > 180) & (s_channel > 50)).astype(np.uint8) * 255

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        settings = self._config.load_settings()
        shake_roi = settings.shake_roi
        roi_pixels = self._compute_roi_pixels(shake_roi)
        if roi_pixels is None:
            return None

        scale = self._scale_factor

        for c in sorted(contours, key=cv2.contourArea, reverse=True):
            area = cv2.contourArea(c)
            if area < 500 or area > 20000:
                continue

            x, y, w, h = cv2.boundingRect(c)
            aspect = w / h if h > 0 else 0
            if aspect < 0.3 or aspect > 3.0:
                continue

            M = cv2.moments(c)
            if M["m00"] == 0:
                continue

            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])

            screen_x = int((roi_pixels["left"] + cx) / scale)
            screen_y = int((roi_pixels["top"] + cy) / scale)
            self.logger.debug("Shake button found at screen (%d, %d)", screen_x, screen_y)
            return (screen_x, screen_y)

        return None

    # ------------------------------------------------------------------
    # Orchestrator
    # ------------------------------------------------------------------

    def detect_all(self) -> DetectionResult:
        """Run the full detection pipeline and return a :class:`DetectionResult`.

        This is the main entry-point called each tick by the macro engine.
        """
        settings = self._config.load_settings()

        # --- Capture the bar ROI ---
        bar_frame = self.capture_roi(settings.bar_roi)
        if bar_frame is None:
            return DetectionResult(bar_active=False)

        shape_active = self.detect_bar_active(bar_frame)
        fish_x = self.detect_fish_x(bar_frame)
        active = shape_active or fish_x is not None

        if not active:
            # Check for a shake button instead
            shake_frame = self.capture_roi(settings.shake_roi)
            shake_pos = self.detect_shake_button(shake_frame) if shake_frame is not None else None
            return DetectionResult(bar_active=False, shake_pos=shake_pos)

        # --- Bar is active — run full detection ---
        bar_bounds = self.detect_control_bar(bar_frame)

        bar_left = bar_bounds[0] if bar_bounds else None
        bar_right = bar_bounds[1] if bar_bounds else None

        # Progress
        progress_frame = self.capture_roi(settings.progress_roi)
        progress = self.detect_progress(progress_frame) if progress_frame is not None else 0.0

        result = DetectionResult(
            bar_active=True,
            fish_x=fish_x,
            bar_left=bar_left,
            bar_right=bar_right,
            progress=progress,
        )

        if self.debug_mode:
            result.debug_frame = self.get_debug_frame(bar_frame, result)

        return result

    # ------------------------------------------------------------------
    # Debug visualisation
    # ------------------------------------------------------------------

    def get_debug_frame(self, frame: np.ndarray, result: DetectionResult) -> np.ndarray:
        """Annotate *frame* with detection results for GUI overlay.

        Parameters
        ----------
        frame : np.ndarray
            The raw bar-ROI capture (BGR).
        result : DetectionResult
            The detection result for this frame.

        Returns
        -------
        np.ndarray
            A copy of *frame* with visual annotations drawn on top.
        """
        vis = frame.copy()
        h, w = vis.shape[:2]

        # Fish indicator — red vertical line
        if result.fish_x is not None:
            fx = int(result.fish_x * w)
            cv2.line(vis, (fx, 0), (fx, h), (0, 0, 255), 2)
            cv2.putText(
                vis, f"Fish: {result.fish_x:.2f}",
                (fx + 4, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1,
            )

        # Control bar zone — green boundary lines + semi-transparent fill
        if result.bar_left is not None and result.bar_right is not None:
            lx = int(result.bar_left * w)
            rx = int(result.bar_right * w)
            overlay = vis.copy()
            cv2.rectangle(overlay, (lx, 0), (rx, h), (0, 255, 0), -1)
            cv2.addWeighted(overlay, 0.25, vis, 0.75, 0, vis)
            cv2.line(vis, (lx, 0), (lx, h), (0, 255, 0), 2)
            cv2.line(vis, (rx, 0), (rx, h), (0, 255, 0), 2)
            cv2.putText(
                vis, f"Bar: {result.bar_left:.2f}-{result.bar_right:.2f}",
                (lx + 4, h - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1,
            )

        # Progress bar — coloured strip at the bottom
        prog_h = max(4, int(h * 0.08))
        prog_w = int(result.progress * w)
        # Gradient green → yellow → red as progress increases
        r = int(min(255, (1.0 - result.progress) * 2 * 255))
        g = int(min(255, result.progress * 2 * 255))
        cv2.rectangle(vis, (0, h - prog_h), (prog_w, h), (0, g, r), -1)
        cv2.putText(
            vis, f"Prog: {result.progress:.0%}",
            (4, h - prog_h - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1,
        )

        return vis
