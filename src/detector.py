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
    bite_confirmed: bool = False         # True only if bar shape or fish icon is seen
    fish_x: Optional[float] = None       # 0.0–1.0 normalized position in bar
    bar_left: Optional[float] = None     # 0.0–1.0 control bar left edge
    bar_right: Optional[float] = None    # 0.0–1.0 control bar right edge
    on_target: bool = False              # Whether the bar is currently over the fish (color-based)
    progress: float = 0.0                # 0.0–1.0 progress bar fill
    shake_pos: Optional[Tuple[int, int]] = None  # (x, y) screen coords of shake button
    shake_confidence: float = 0.0  # 0–1 match quality for the SHAKE UI pattern
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

        settings = self._config.load_settings()
        wb = self._window_bounds
        scale = self._scale_factor

        inset_left = wb.width * settings.window_inset_left
        inset_top = wb.height * settings.window_inset_top
        eff_x = wb.x + inset_left
        eff_y = wb.y + inset_top
        eff_w = max(1, wb.width - inset_left)
        eff_h = max(1, wb.height - inset_top)

        xs = float(np.clip(roi_bounds.x_start + settings.roi_shift_x, 0.0, 1.0))
        xe = float(np.clip(roi_bounds.x_end + settings.roi_shift_x, 0.0, 1.0))
        ys = float(np.clip(roi_bounds.y_start + settings.roi_shift_y, 0.0, 1.0))
        ye = float(np.clip(roi_bounds.y_end + settings.roi_shift_y, 0.0, 1.0))
        if xe <= xs:
            xe = min(1.0, xs + 0.01)
        if ye <= ys:
            ye = min(1.0, ys + 0.01)

        return {
            "left": int((eff_x + eff_w * xs) * scale),
            "top": int((eff_y + eff_h * ys) * scale),
            "width": max(1, int(eff_w * (xe - xs) * scale)),
            "height": max(1, int(eff_h * (ye - ys) * scale)),
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
        # Only strip if they are ALSO low saturation (to avoid stripping the fish icon or bright bars)
        bright_mask = (v_channel > 250) & (s_channel < 60)

        # Low-saturation bright pixels (white flashes / bloom)
        flash_mask = (v_channel > 230) & (s_channel < 20)

        vfx_mask = bright_mask | flash_mask
        return (~vfx_mask).astype(np.uint8) * 255

    # ------------------------------------------------------------------
    # Individual detectors
    # ------------------------------------------------------------------

    def detect_bar_active(self, frame: np.ndarray) -> bool:
        """Determine whether the fishing minigame bar is currently on-screen."""
        if frame is None or frame.size == 0:
            return False

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Look for the characteristic horizontal lines of the bar track
        edges = cv2.Canny(gray, 50, 150)
        row_sums = np.sum(edges > 0, axis=1)
        # The bar track usually has two very long horizontal lines
        strong_rows = np.sum(row_sums > (frame.shape[1] * 0.5))

        if strong_rows >= 2:
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
        
        # Use a more targeted VFX filter for the fish to avoid losing it
        # The fish icon is usually saturated, so we avoid stripping high-saturation pixels.
        vfx_ok = self._filter_vfx(hsv)

        color_mask = cv2.inRange(
            hsv,
            np.array(profile.fish_hsv_low, dtype=np.uint8),
            np.array(profile.fish_hsv_high, dtype=np.uint8),
        )
        
        # Combine with VFX mask, but be lenient
        mask = cv2.bitwise_and(color_mask, vfx_ok)
        
        # If the combined mask is too empty, fall back to just the color mask 
        # (in case VFX filter is too aggressive for the fish icon)
        if cv2.countNonZero(mask) < 10:
            mask = color_mask

        # Smaller kernel for morphological cleanup to preserve small fish icon features
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        # Largest contour that passes noise filter
        valid = [c for c in contours if cv2.contourArea(c) > 10]
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

        The control zone can be On-Target (Green) or Off-Target (Orange/White).
        We combine both masks to find the full extent of the bar, while being
        careful to ignore the gray background track.
        """
        if frame is None or frame.size == 0:
            return None

        profile = self._config.get_active_profile()
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        vfx_ok = self._filter_vfx(hsv)

        # 1. Generate Masks
        on_mask = cv2.inRange(
            hsv,
            np.array(profile.on_target_hsv_low, dtype=np.uint8),
            np.array(profile.on_target_hsv_high, dtype=np.uint8),
        )
        off_mask = cv2.inRange(
            hsv,
            np.array(profile.off_target_hsv_low, dtype=np.uint8),
            np.array(profile.off_target_hsv_high, dtype=np.uint8),
        )
        
        # 2. Smart Background Exclusion
        # If the off_target color is low-saturation (grayish), we need to ensure 
        # we aren't just picking up the background track.
        s_channel = hsv[:, :, 1]
        v_channel = hsv[:, :, 2]
        
        # Typical background track is dull. The bar (even if gray) usually has 
        # higher 'V' (brightness) or a slight saturation pop.
        track_mask = ((s_channel < 30) & (v_channel < 100)).astype(np.uint8) * 255
        off_mask = cv2.bitwise_and(off_mask, cv2.bitwise_not(track_mask))
        
        combined_mask = cv2.bitwise_or(on_mask, off_mask)
        mask = cv2.bitwise_and(combined_mask, vfx_ok)

        # 3. Clean up and scan for horizontal extent
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        w = frame.shape[1]
        col_counts = np.sum(mask > 0, axis=0)
        # Requirement: at least 15% vertical fill to be considered part of the bar
        filled_cols = np.where(col_counts > (frame.shape[0] * 0.15))[0]
        
        if len(filled_cols) > 10:
            # Find the longest contiguous block of filled columns
            diffs = np.diff(filled_cols)
            breaks = np.where(diffs > 3)[0] 
            
            starts = np.insert(filled_cols[breaks + 1], 0, filled_cols[0])
            ends = np.append(filled_cols[breaks], filled_cols[-1])
            
            lengths = ends - starts
            best_idx = int(np.argmax(lengths))
            
            left = starts[best_idx] / w
            right = ends[best_idx] / w
            
            # Sanity check: the bar should have a reasonable width (e.g., > 5% of track)
            if right - left < 0.02:
                return self._brightness_fallback(frame)
                
            return (float(np.clip(left, 0.0, 1.0)),
                    float(np.clip(right, 0.0, 1.0)))

        gradient_bounds = self._gradient_bar_bounds(frame)
        if gradient_bounds is not None:
            return gradient_bounds

        return self._brightness_fallback(frame)

    def _gradient_bar_bounds(self, frame: np.ndarray) -> Optional[Tuple[float, float]]:
        """Column-gradient peak finder (digmacro-style fallback when colour masks fail)."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
        if gray.shape[1] < 12:
            return None

        grad = np.abs(np.gradient(gray, axis=1))
        col_strength = np.mean(grad, axis=0)
        if float(col_strength.max()) < 2.0:
            return None

        threshold = float(np.mean(col_strength) + np.std(col_strength) * 0.5)
        active = col_strength >= threshold
        if not np.any(active):
            return None

        indices = np.where(active)[0]
        splits = np.where(np.diff(indices) > 3)[0]
        starts = np.insert(indices[splits + 1], 0, indices[0])
        ends = np.append(indices[splits], indices[-1])
        lengths = ends - starts
        if len(lengths) == 0:
            return None

        best = int(np.argmax(lengths))
        left = starts[best] / frame.shape[1]
        right = ends[best] / frame.shape[1]
        if right - left < 0.03:
            return None
        return (float(np.clip(left, 0.0, 1.0)), float(np.clip(right, 0.0, 1.0)))

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

    def detect_on_target(self, frame: np.ndarray, left: float, right: float) -> bool:
        """Determine if the bar is 'on target' (active) based on color profile.

        Analyzes the region between left and right boundaries for the active color.
        """
        if frame is None or frame.size == 0 or left is None or right is None:
            return False

        profile = self._config.get_active_profile()
        h, w = frame.shape[:2]
        lx, rx = int(left * w), int(right * w)
        if rx <= lx:
            return False

        # Extract the bar interior
        bar_roi = frame[:, lx:rx]
        hsv = cv2.cvtColor(bar_roi, cv2.COLOR_BGR2HSV)

        # 1. Check for 'On Target' color (usually green)
        on_mask = cv2.inRange(
            hsv,
            np.array(profile.on_target_hsv_low, dtype=np.uint8),
            np.array(profile.on_target_hsv_high, dtype=np.uint8),
        )
        
        # 2. Check for 'Off Target' color (usually orange/white)
        off_mask = cv2.inRange(
            hsv,
            np.array(profile.off_target_hsv_low, dtype=np.uint8),
            np.array(profile.off_target_hsv_high, dtype=np.uint8),
        )

        on_pixels = cv2.countNonZero(on_mask)
        off_pixels = cv2.countNonZero(off_mask)
        total_pixels = bar_roi.shape[0] * bar_roi.shape[1]
        
        # Robust Gradient Logic:
        # 1. If we see a decent amount of On-Target color (>8% of the bar), we're likely on.
        # 2. If we see both, we check if On-Target is at least half as common as Off-Target.
        on_pct = on_pixels / total_pixels
        off_pct = off_pixels / total_pixels
        
        on_target = (on_pct > 0.08) or (on_pct > 0.02 and on_pct > off_pct * 0.5)
        
        if on_target:
            self.logger.debug("Bar ON TARGET (on: %.1f%%, off: %.1f%%)", on_pct*100, off_pct*100)
        return on_target

    # ---- progress bar --------------------------------------------------

    def detect_progress(self, progress_frame: np.ndarray) -> float:
        """Return progress-bar fill as a float 0.0–1.0.

        Uses row-consensus across the ROI so gradient bleed from the control bar
        does not read as sudden 90%+ completion.
        """
        if progress_frame is None or progress_frame.size == 0:
            return 0.0

        hsv = cv2.cvtColor(progress_frame, cv2.COLOR_BGR2HSV)
        h, w = hsv.shape[:2]
        if w < 8:
            return 0.0

        vfx_keep = self._filter_vfx(hsv) > 0
        h_ch = hsv[:, :, 0]
        s_ch = hsv[:, :, 1]
        v_ch = hsv[:, :, 2]

        # Progress fill: bright white/cyan strip (reject saturated red/blue VFX)
        fill_mask = (
            (v_ch > 75)
            & (s_ch > 20)
            & (s_ch < 200)
            & vfx_keep
            & ~((s_ch > 90) & ((h_ch < 25) | (h_ch > 115)))
        )

        row_need = max(0.18, min(0.40, 8.0 / max(h, 1)))
        col_ratio = np.mean(fill_mask, axis=0)
        filled = col_ratio >= row_need

        rightmost = -1
        for col in range(w):
            if filled[col]:
                rightmost = col
            elif col > 4 and rightmost >= 0 and (col - rightmost) > max(8, w // 12):
                break

        if rightmost < 0:
            return 0.0

        return float(np.clip((rightmost + 1) / w, 0.0, 1.0))

    # ---- shake button --------------------------------------------------

    def _shake_button_score(
        self,
        v_channel: np.ndarray,
        s_channel: np.ndarray,
        cx: int,
        cy: int,
        radius: int,
    ) -> float:
        """Score how closely a circle matches the Fisch SHAKE UI (dark fill + white ring + text)."""
        h, w = v_channel.shape[:2]
        if radius < 14 or radius > int(min(h, w) * 0.14):
            return 0.0

        yy, xx = np.ogrid[:h, :w]
        dist2 = (xx - cx) ** 2 + (yy - cy) ** 2
        inner = dist2 < (0.42 * radius) ** 2
        ring = (dist2 >= (0.58 * radius) ** 2) & (dist2 <= (0.98 * radius) ** 2)
        text_zone = dist2 < (0.38 * radius) ** 2

        if int(inner.sum()) < 25 or int(ring.sum()) < 40:
            return 0.0

        inner_v = float(v_channel[inner].mean())
        ring_v = float(v_channel[ring].mean())
        ring_sat_frac = float((s_channel[ring] > 85).sum()) / float(ring.sum())
        text_white_frac = float((v_channel[text_zone] > 175).sum()) / float(text_zone.sum())

        # Reject saturated red/blue water stripes and colored nameplates
        if ring_sat_frac > 0.22:
            return 0.0
        if inner_v > 125 or ring_v < 155:
            return 0.0
        if text_white_frac < 0.06:
            return 0.0

        contrast = (ring_v - inner_v) / 255.0
        ring_score = min((ring_v - 155) / 70.0, 1.0)
        dark_score = min((115 - inner_v) / 90.0, 1.0)
        text_score = min(text_white_frac * 5.0, 1.0)
        return float(
            np.clip(
                contrast * 0.40 + ring_score * 0.30 + dark_score * 0.20 + text_score * 0.10,
                0.0,
                1.0,
            )
        )

    def detect_shake_button(self, shake_frame: np.ndarray) -> Tuple[Optional[Tuple[int, int]], float]:
        """Detect the SHAKE button; return screen coords and a confidence score."""
        if shake_frame is None or shake_frame.size == 0:
            return None, 0.0

        h, w = shake_frame.shape[:2]
        hsv = cv2.cvtColor(shake_frame, cv2.COLOR_BGR2HSV)
        v_channel = hsv[:, :, 2]
        s_channel = hsv[:, :, 1]

        # White outline + SHAKE label only (ignore saturated environment colours)
        white_ui = (v_channel > 185) & (s_channel < 75)
        white_mask = white_ui.astype(np.uint8) * 255
        white_mask = cv2.bitwise_and(white_mask, self._filter_vfx(hsv))

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel)

        candidates: list[tuple[float, int, int]] = []

        # Circle search on desaturated bright edges
        gray = cv2.cvtColor(shake_frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(white_mask, 60, 160)
        min_r = max(16, int(min(h, w) * 0.022))
        max_r = max(min_r + 10, int(min(h, w) * 0.11))
        circles = cv2.HoughCircles(
            edges,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=max(50, min(h, w) // 6),
            param1=90,
            param2=22,
            minRadius=min_r,
            maxRadius=max_r,
        )
        if circles is not None:
            for cx, cy, r in np.round(circles[0]).astype(int):
                score = self._shake_button_score(v_channel, s_channel, cx, cy, r)
                if score > 0.0:
                    candidates.append((score, cx, cy))

        # Fallback: white ring contours with dark interior
        contours, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            area = cv2.contourArea(c)
            if area < 180 or area > h * w * 0.04:
                continue
            (cx, cy), r = cv2.minEnclosingCircle(c)
            score = self._shake_button_score(v_channel, s_channel, int(cx), int(cy), int(r))
            if score > 0.0:
                candidates.append((score, int(cx), int(cy)))

        if not candidates:
            return None, 0.0

        settings = self._config.load_settings()
        roi_pixels = self._compute_roi_pixels(settings.shake_roi)
        if roi_pixels is None:
            return None, 0.0

        best_score, best_cx, best_cy = max(candidates, key=lambda item: item[0])
        scale = self._scale_factor
        screen_pos = (
            int((roi_pixels["left"] + best_cx) / scale),
            int((roi_pixels["top"] + best_cy) / scale),
        )
        self.logger.debug(
            "Shake candidate at screen (%d, %d) confidence=%.2f",
            screen_pos[0],
            screen_pos[1],
            best_score,
        )
        return screen_pos, best_score

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

        progress_frame = self.capture_roi(settings.progress_roi)
        progress = self.detect_progress(progress_frame) if progress_frame is not None else 0.0

        # Always scan for shake — it must work even when bar/VFX look "active"
        shake_pos = None
        shake_confidence = 0.0
        if settings.shake_enabled:
            shake_frame = self.capture_roi(settings.shake_roi)
            if shake_frame is not None:
                shake_pos, shake_confidence = self.detect_shake_button(shake_frame)

        # Require fish + control bar for a real bite (blocks rod VFX / edge false positives)
        bar_bounds = None
        if fish_x is not None or shape_active:
            bar_bounds = self.detect_control_bar(bar_frame)

        bar_left = bar_bounds[0] if bar_bounds else None
        bar_right = bar_bounds[1] if bar_bounds else None

        has_minigame_ui = (
            fish_x is not None and bar_left is not None and bar_right is not None
        )
        bite_confirmed = has_minigame_ui or (
            shape_active and fish_x is not None
        ) or progress >= 0.12

        active = bite_confirmed or progress >= 0.08

        if not active:
            return DetectionResult(
                bar_active=False,
                bite_confirmed=False,
                shake_pos=shake_pos,
                shake_confidence=shake_confidence,
            )

        on_target = False
        if bar_left is not None and bar_right is not None:
            on_target = self.detect_on_target(bar_frame, bar_left, bar_right)

        result = DetectionResult(
            bar_active=True,
            bite_confirmed=bite_confirmed,
            fish_x=fish_x,
            bar_left=bar_left,
            bar_right=bar_right,
            on_target=on_target,
            progress=progress,
            shake_pos=shake_pos,
            shake_confidence=shake_confidence,
        )

        if self.debug_mode or settings.show_live_vision:
            result.debug_frame = self.get_debug_frame(bar_frame, result)

        return result

    # ------------------------------------------------------------------
    # Debug visualisation
    # ------------------------------------------------------------------

    def get_debug_frame(
        self,
        frame: np.ndarray,
        result: DetectionResult,
        extras: Optional[dict] = None,
    ) -> np.ndarray:
        """Build a clean schematic for the GUI (not a noisy overlay on raw pixels).

        The bar ROI is only a few pixels tall — drawing fills and text on the
        capture looks messy when scaled. This renders a fixed-size diagram plus
        a small dimmed camera inset for calibration checks.
        """
        extras = extras or {}
        schematic_w, schematic_h = 400, 64
        pad_x, pad_y = 12, 10
        track_w = schematic_w - pad_x * 2
        track_h = schematic_h - pad_y * 2

        def _x_norm(value: Optional[float]) -> Optional[int]:
            if value is None:
                return None
            return int(pad_x + float(np.clip(value, 0.0, 1.0)) * track_w)

        # Dark schematic background
        vis = np.full((schematic_h, schematic_w, 3), 32, dtype=np.uint8)
        mid_y = schematic_h // 2
        cv2.line(vis, (pad_x, mid_y), (schematic_w - pad_x, mid_y), (70, 70, 70), 1)

        # Control bar bracket (outline only — no solid fill)
        if result.bar_left is not None and result.bar_right is not None:
            lx = _x_norm(result.bar_left)
            rx = _x_norm(result.bar_right)
            if lx is not None and rx is not None and rx > lx:
                bar_color = (80, 220, 80) if result.on_target else (80, 220, 220)
                cv2.rectangle(vis, (lx, pad_y), (rx, schematic_h - pad_y), bar_color, 2)

        effective = extras.get("effective_bar")
        ex = _x_norm(effective) if effective is not None else None
        if ex is not None:
            cv2.line(vis, (ex, pad_y), (ex, schematic_h - pad_y), (220, 120, 255), 1)

        predicted = extras.get("predicted_fish_x")
        px = _x_norm(predicted) if predicted is not None else None
        if px is not None:
            for y in range(pad_y, schematic_h - pad_y, 4):
                cv2.line(vis, (px, y), (px, min(y + 2, schematic_h - pad_y)), (220, 220, 0), 1)

        if result.fish_x is not None:
            fx = _x_norm(result.fish_x)
            if fx is not None:
                cv2.line(vis, (fx, pad_y), (fx, schematic_h - pad_y), (60, 60, 255), 2)
                cv2.circle(vis, (fx, mid_y), 5, (0, 0, 255), -1)

        # Small dimmed inset of the real ROI (right side) — verify calibration
        if frame is not None and frame.size > 0:
            inset_w = 88
            src_h, src_w = frame.shape[:2]
            scale = inset_w / max(src_w, 1)
            inset_h = max(12, min(track_h, int(src_h * scale)))
            thumb = cv2.resize(frame, (inset_w, inset_h), interpolation=cv2.INTER_AREA)
            thumb = (thumb * 0.45).astype(np.uint8)
            x0 = schematic_w - inset_w - 6
            y0 = (schematic_h - inset_h) // 2
            y1 = y0 + inset_h
            if x0 > pad_x + 40 and y1 <= schematic_h:
                vis[y0:y1, x0 : x0 + inset_w] = thumb
                cv2.rectangle(
                    vis,
                    (x0 - 1, y0 - 1),
                    (x0 + inset_w, y1),
                    (90, 90, 90),
                    1,
                )

        return vis
