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

from src.reel_vision import ReelReading, ReelVision
from src.track_locator import TrackLocator


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
    on_target: bool = False              # Fish is between the bar edges (geometric)
    progress: float = 0.0                # 0.0–1.0 progress bar fill
    shake_pos: Optional[Tuple[int, int]] = None  # (x, y) screen coords of shake button
    shake_confidence: float = 0.0  # 0–1 match quality for the SHAKE UI pattern
    debug_frame: Optional[np.ndarray] = None     # Annotated frame for GUI
    reading: Optional["ReelReading"] = None      # Raw structure-based read of the track
    track_box: Optional[Tuple[int, int, int, int]] = None  # Track ROI within the window band


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

        # Structure-based reel vision. Replaces per-rod HSV matching for the
        # control bar and the fish; see src/reel_vision.py for why.
        self._vision = ReelVision()
        self._track_locator = TrackLocator()
        self._relocate_countdown = 0
        self._shake_countdown = 0

    def reset_session(self) -> None:
        """Forget per-fight vision state. Call when a new minigame starts."""
        settings = self._config.load_settings()
        self._vision.p.fish_relative_strength = settings.fish_relative_strength
        self._vision.p.fish_max_speed = settings.fish_max_speed
        self._vision.reset()
        self._track_locator.reset()
        self._relocate_countdown = 0

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
        changed = (
            self._window_bounds is None
            or (bounds.x, bounds.y, bounds.width, bounds.height)
            != (
                self._window_bounds.x,
                self._window_bounds.y,
                self._window_bounds.width,
                self._window_bounds.height,
            )
            or scale_factor != self._scale_factor
        )
        self._window_bounds = bounds
        self._scale_factor = scale_factor
        if not changed:
            # Called every tick; only say something when it actually moves.
            return
        self._track_locator.reset()
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
    # Structure-based reel reading
    # ------------------------------------------------------------------

    def capture_window_band(self, settings) -> Optional[np.ndarray]:
        """Capture the lower band of the window, where the reel UI lives.

        Auto-location needs more context than the calibrated bar ROI provides —
        it has to see the track's surroundings to find its edges — but grabbing
        the whole window every tick is wasteful. The lower band is the
        compromise.
        """
        class _Band:
            x_start = 0.0
            x_end = 1.0
            y_start = float(np.clip(settings.track_search_top, 0.0, 0.95))
            y_end = 1.0

        return self.capture_roi(_Band())

    def read_reel(self, settings) -> Tuple[Optional[ReelReading], Optional[np.ndarray], Optional[Tuple[int, int, int, int]]]:
        """Read the reel track. Returns (reading, band_frame, track_box).

        ``track_box`` is in the coordinate space of the captured band, not the
        screen — it is only used for the debug overlay.
        """
        if not settings.auto_locate_track:
            # Default path. The calibrated ROI is used as the track strip
            # directly, which measured better than searching for the track on
            # every one of the seven sample clips: 74-100% mid-fight detection,
            # against an auto-locate path that ranged from 7% to 99% depending
            # on the biome.
            #
            # The reason is that no single appearance cue survives the variety.
            # A flat black track has no vertical gradient in its interior, so an
            # edge-based band collapses onto its top border; a track with a
            # gradient fill does not. A calibrated rectangle sidesteps all of
            # that, and is stable frame to frame by construction — which matters
            # more here than being exactly right, since positions are normalised
            # against this span and a span that moves invents velocity.
            frame = self.capture_roi(settings.bar_roi)
            if frame is None:
                return None, None, None
            return self._vision.read(frame, pre_located=True), frame, (
                0, 0, frame.shape[1], frame.shape[0]
            )

        # Auto-location needs the surroundings, so it pays for the wider grab.
        # The default path above must not: this capture costs ~9ms of a 20ms
        # tick and was previously taken on every tick regardless, burning a
        # third of the detection budget on a frame nothing looked at.
        band = self.capture_window_band(settings)
        if band is None:
            return None, None, None

        # Search box: horizontal span straight from the calibrated ROI, which is
        # stable by construction; vertical range deliberately generous, because
        # a hand-dragged rectangle is routinely off by a dozen pixels and the
        # exact track rows are what the reading depends on.
        band_h, band_w = band.shape[:2]
        top = float(np.clip(settings.track_search_top, 0.0, 0.95))
        denom = max(1e-6, 1.0 - top)

        def _band_y(window_frac: float) -> int:
            return int(round((window_frac - top) / denom * band_h))

        roi = settings.bar_roi
        x0 = int(round(roi.x_start * band_w))
        x1 = int(round(roi.x_end * band_w))
        centre = (_band_y(roi.y_start) + _band_y(roi.y_end)) / 2.0
        half = max(24.0, (_band_y(roi.y_end) - _band_y(roi.y_start)) * settings.track_search_expand)
        search = (
            max(0, x0),
            max(0, int(centre - half)),
            min(band_w, x1),
            min(band_h, int(centre + half)),
        )

        box = self._track_locator.box
        if box is None or self._relocate_countdown <= 0:
            box = self._track_locator.update(band, search)
            self._relocate_countdown = max(1, settings.track_relocate_every)
        else:
            self._relocate_countdown -= 1

        if box is None:
            return None, band, None

        bx0, by0, bx1, by1 = box
        track_frame = band[by0:by1, bx0:bx1]
        if track_frame.size == 0:
            return None, band, box

        # The box already is the track, and its horizontal span is fixed by
        # calibration, so it defines a coordinate frame that is stable tick to
        # tick rather than being re-derived from each frame.
        return self._vision.read(track_frame, pre_located=True), band, box

    def capture_pair(self, roi_a, roi_b):
        """Capture two ROIs in one screen grab where that is cheaper.

        Grab cost here is dominated by per-call overhead rather than by area —
        a 534k-pixel region measured at 9ms while a 15k-pixel one measured at
        16ms — so two small grabs cost roughly twice one larger one. The bar and
        progress ROIs sit a few pixels apart vertically, so their union is
        barely bigger than either and a single grab serves both.

        Falls back to two separate grabs if the union is disproportionate, which
        would happen if the ROIs were ever calibrated far apart.
        """
        a = self._compute_roi_pixels(roi_a)
        b = self._compute_roi_pixels(roi_b)
        if a is None or b is None:
            return self.capture_roi(roi_a), self.capture_roi(roi_b)

        left = min(a["left"], b["left"])
        top = min(a["top"], b["top"])
        right = max(a["left"] + a["width"], b["left"] + b["width"])
        bottom = max(a["top"] + a["height"], b["top"] + b["height"])
        union_area = max(1, (right - left) * (bottom - top))
        parts_area = a["width"] * a["height"] + b["width"] * b["height"]

        if union_area > parts_area * 3:
            return self.capture_roi(roi_a), self.capture_roi(roi_b)

        try:
            raw = self._get_mss().grab(
                {"left": left, "top": top, "width": right - left, "height": bottom - top}
            )
            merged = cv2.cvtColor(np.array(raw), cv2.COLOR_BGRA2BGR)
        except Exception as exc:
            self.logger.warning("Combined grab failed: %s", exc)
            return self.capture_roi(roi_a), self.capture_roi(roi_b)

        def _slice(box):
            y0 = box["top"] - top
            x0 = box["left"] - left
            out = merged[y0:y0 + box["height"], x0:x0 + box["width"]]
            return out if out.size else None

        return _slice(a), _slice(b)

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
        # Use stricter thresholds (80, 200) to reject soft edges from transparent backgrounds
        edges = cv2.Canny(gray, 80, 200)
        row_sums = np.sum(edges > 0, axis=1)
        # The bar track usually has two very long horizontal lines
        strong_rows = np.sum(row_sums > (frame.shape[1] * 0.45))

        # Check color saturation/value to reject pure dark backgrounds
        # The actual bar track has some brightness, whereas transparent overlays are darker
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        v_channel = hsv[:, :, 2]
        bright_pixels = np.sum(v_channel > 120)
        has_brightness = bright_pixels > (frame.size // 3) * 0.1  # At least 10% bright pixels

        if strong_rows >= 2 and has_brightness:
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

        h_ch = hsv[:, :, 0]
        s_ch = hsv[:, :, 1]
        v_ch = hsv[:, :, 2]

        # Progress fill: a bright strip growing from the left. It is defined by
        # brightness, not by colour.
        #
        # It must NOT be gated on a saturation floor, and _filter_vfx must not be
        # applied to it. Measured across the sample clips the fill sits at
        # S=3..54 depending on the rod — near-white on four of the seven — while
        # the floor was s_ch > 15 and _filter_vfx discards exactly (V > 250,
        # S < 60). A white progress bar failed both tests, so this returned ~0
        # for the whole fight and every downstream catch/fail/stall heuristic was
        # reading a signal that was never there.
        #
        # What still has to be rejected is the world behind a translucent UI: it
        # is either dark (excluded by the brightness floor) or strongly
        # saturated at a hue the bar never takes.
        fill_mask = (
            (v_ch > 140)
            & (s_ch < 200)
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

    def detect_all(self, scan_shake: bool = True) -> DetectionResult:
        """Run the full detection pipeline and return a :class:`DetectionResult`.

        This is the main entry-point called each tick by the macro engine.

        ``scan_shake`` exists purely for speed. The shake ROI is most of the
        window, and grabbing plus Hough-searching it costs more than every other
        step combined — enough to push a tick past its budget on its own. It is
        only meaningful while waiting for a bite, so the reeling loop turns it
        off and gets its ticks back.

        Bar and fish come from the structure-based reader (:mod:`reel_vision`),
        so they need no per-rod colour calibration, and ``on_target`` is decided
        geometrically from those bounds rather than by sampling the bar's
        colour. Progress and shake still use colour, since they are stable
        single-purpose UI elements.
        """
        settings = self._config.load_settings()

        if settings.auto_locate_track:
            reading, band_frame, track_box = self.read_reel(settings)
            progress_frame = self.capture_roi(settings.progress_roi)
        else:
            # One grab for both, rather than one each: see capture_pair.
            band_frame, progress_frame = self.capture_pair(
                settings.bar_roi, settings.progress_roi
            )
            track_box = None
            reading = None
            if band_frame is not None:
                reading = self._vision.read(band_frame, pre_located=True)
                track_box = (0, 0, band_frame.shape[1], band_frame.shape[0])

        progress = self.detect_progress(progress_frame) if progress_frame is not None else 0.0

        # Always scan for shake — it must work even when bar/VFX look "active"
        shake_pos = None
        shake_confidence = 0.0
        if settings.shake_enabled and scan_shake:
            # Decimated on purpose. The shake ROI is most of the window and the
            # grab alone was measured at 17-100ms, which by itself pushed a tick
            # from 32ms to 126ms and dropped the whole loop to roughly 8Hz — slow
            # enough that bites took seconds to notice. A shake prompt stays up
            # for seconds, so checking a few times a second loses nothing.
            if self._shake_countdown <= 0:
                self._shake_countdown = max(1, settings.shake_scan_every)
                shake_frame = self.capture_roi(settings.shake_roi)
                if shake_frame is not None:
                    shake_pos, shake_confidence = self.detect_shake_button(shake_frame)
            else:
                self._shake_countdown -= 1

        if reading is None:
            return DetectionResult(
                bar_active=False,
                bite_confirmed=False,
                progress=progress,
                shake_pos=shake_pos,
                shake_confidence=shake_confidence,
                track_box=track_box,
            )

        fish_x = reading.fish_x
        bar_left = reading.bar_left
        bar_right = reading.bar_right

        # Two different questions, deliberately answered with different
        # strictness.
        #
        # "Has a bite started?" must be strict — both a control bar and a fish
        # — because a false start casts away a fish. That is a structural fact
        # about the minigame UI, so unlike the old colour heuristics it cannot
        # be faked by rod VFX or lava.
        #
        # "Is the minigame still running?" must be tolerant, because the exit
        # conditions read it: requiring both here meant a single-frame fish
        # dropout looked exactly like the bar having vanished, and the fight
        # could be declared over mid-fight. Either element alone is ample
        # evidence that the UI is still on screen.
        has_bar = bar_left is not None and bar_right is not None
        bite_confirmed = has_bar and fish_x is not None
        active = (
            has_bar
            or fish_x is not None
            or progress >= settings.bite_progress_threshold
        )

        if not active:
            return DetectionResult(
                bar_active=False,
                bite_confirmed=False,
                progress=progress,
                shake_pos=shake_pos,
                shake_confidence=shake_confidence,
                reading=reading,
                track_box=track_box,
            )

        result = DetectionResult(
            bar_active=True,
            bite_confirmed=bite_confirmed,
            fish_x=fish_x,
            bar_left=bar_left,
            bar_right=bar_right,
            on_target=reading.on_target,
            progress=progress,
            shake_pos=shake_pos,
            shake_confidence=shake_confidence,
            reading=reading,
            track_box=track_box,
        )

        if self.debug_mode or settings.show_live_vision:
            frame = band_frame
            if frame is not None and track_box is not None:
                x0, y0, x1, y1 = track_box
                frame = frame[y0:y1, x0:x1]
            result.debug_frame = self.get_debug_frame(frame, result)

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
        """Build an overlaid visualization of the actual captured bar ROI."""
        extras = extras or {}

        if frame is None or frame.size == 0:
            vis = np.full((64, 400, 3), 32, dtype=np.uint8)
            cv2.putText(vis, "No Input", (150, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 150, 150), 1)
            return vis

        # Scale up the raw frame for visibility
        scale_factor = max(1, min(4, 800 // max(1, frame.shape[1])))
        h, w = frame.shape[:2]
        vis = cv2.resize(frame, (w * scale_factor, h * scale_factor), interpolation=cv2.INTER_NEAREST)
        
        # Dim the image slightly so overlays pop
        vis = (vis * 0.6).astype(np.uint8)

        # Add synthetic progress bar underneath
        prog_h = 12
        vis = np.pad(vis, ((0, prog_h + 4), (0, 0), (0, 0)), mode='constant', constant_values=20)
        new_h = vis.shape[0]
        
        def _x_px(value: Optional[float]) -> Optional[int]:
            if value is None: return None
            return int(float(np.clip(value, 0.0, 1.0)) * (w * scale_factor))

        # Draw Progress bar
        cv2.rectangle(vis, (0, new_h - prog_h), (w * scale_factor, new_h), (40, 40, 40), -1)
        if result.progress > 0:
            px = _x_px(result.progress)
            cv2.rectangle(vis, (0, new_h - prog_h), (px, new_h), (255, 100, 255), -1)
            
            # Smooth progress if provided
            smooth_prog = extras.get("progress_smooth", 0.0)
            if smooth_prog > 0:
                spx = _x_px(smooth_prog)
                cv2.line(vis, (spx, new_h - prog_h), (spx, new_h), (255, 255, 255), 2)

        # Overlays
        pad_y = 2
        track_h = h * scale_factor
        mid_y = track_h // 2

        # Control bar bracket
        if result.bar_left is not None and result.bar_right is not None:
            lx = _x_px(result.bar_left)
            rx = _x_px(result.bar_right)
            if lx is not None and rx is not None and rx > lx:
                bar_color = (80, 255, 80) if result.on_target else (80, 220, 255)
                # Thick bracket
                cv2.line(vis, (lx, pad_y), (rx, pad_y), bar_color, 3)
                cv2.line(vis, (lx, track_h - pad_y), (rx, track_h - pad_y), bar_color, 3)
                cv2.line(vis, (lx, pad_y), (lx, track_h - pad_y), bar_color, 2)
                cv2.line(vis, (rx, pad_y), (rx, track_h - pad_y), bar_color, 2)

        # Effective bar center
        effective = extras.get("effective_bar")
        if effective is not None:
            ex = _x_px(effective)
            if ex is not None:
                cv2.line(vis, (ex, pad_y + 4), (ex, track_h - pad_y - 4), (220, 120, 255), 2)

        # Predicted fish
        predicted = extras.get("predicted_fish_x")
        if predicted is not None:
            px = _x_px(predicted)
            if px is not None:
                # Dashed yellow line
                for y in range(pad_y, track_h - pad_y, 8):
                    cv2.line(vis, (px, y), (px, min(y + 4, track_h - pad_y)), (0, 220, 220), 2)

        # Actual fish
        if result.fish_x is not None:
            fx = _x_px(result.fish_x)
            if fx is not None:
                cv2.line(vis, (fx, pad_y), (fx, track_h - pad_y), (0, 0, 255), 3)
                cv2.circle(vis, (fx, mid_y), 6, (0, 0, 255), -1)
                cv2.circle(vis, (fx, mid_y), 3, (255, 255, 255), -1)

        # State text
        state = extras.get("macro_state")
        if state:
            # Add a slight dark background for text readability
            cv2.rectangle(vis, (4, 4), (160, 28), (0, 0, 0), -1)
            cv2.putText(vis, state, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        return vis
