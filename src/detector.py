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
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

from src.config import ROIBounds
from src.reel_vision import ReelReading, ReelVision
from src.track_locator import TrackLocator


# ---------------------------------------------------------------------------
# Live-view overlay palette
# ---------------------------------------------------------------------------

# One table, used both by get_debug_frame to draw and by the GUI to render the
# key beside the picture. They used to be written out independently and had
# drifted: the key called the off-target bracket blue when it is drawn amber,
# and said nothing at all about the magenta aim marker or the progress strip.
#
# Values are BGR, because that is what OpenCV draws in. ``OVERLAY_LEGEND`` is
# ordered the way the eye meets them: the two things being aligned first, then
# the aids, then the progress readout.
OVERLAY_COLORS = {
    "fish": (0, 0, 255),            # red
    "fish_predicted": (0, 220, 220),  # yellow, dashed
    "bar_on": (80, 255, 80),        # green
    "bar_off": (80, 220, 255),      # amber
    "bar_aim": (220, 120, 255),     # magenta
    "progress": (255, 100, 255),    # pink strip
    "progress_smooth": (255, 255, 255),  # white tick
}

OVERLAY_LEGEND = (
    ("fish", "Fish now"),
    ("fish_predicted", "Fish predicted (dashed)"),
    ("bar_on", "Bar — on target"),
    ("bar_off", "Bar — off target"),
    ("bar_aim", "Where the bar is aimed"),
    ("progress", "Catch progress"),
    ("progress_smooth", "Progress (smoothed)"),
)


def legend_hex(key: str) -> str:
    """An ``#rrggbb`` string for a palette entry, for Tk swatches."""
    b, g, r = OVERLAY_COLORS[key]
    return f"#{r:02x}{g:02x}{b:02x}"


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
    # The exact crop that was analysed, kept so the caller can re-render the
    # overlay once it knows the control telemetry. detect_all cannot: the
    # predicted-fish line, the aim marker and the smoothed progress are all
    # decided a layer up, so the frame it renders on its own is missing them.
    debug_source: Optional[np.ndarray] = None
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

    # Structural test for "is there a progress bar here at all". Measured over
    # 158 labelled frames of tests/clips/errors.mov: a real bar separates by
    # ~220 grey levels at a separation-to-spread ratio of ~16, an empty ROI by
    # ~47 at ~3.
    # A real fill boundary is a *step*: the colour changes over a column or
    # two. These are Lab distances across that step, and how far it must stand
    # above the profile's ordinary column-to-column variation.
    # Radii the SHAKE button may have, as a fraction of the frame's short side.
    # The circle search and the scorer share them so that the two agree on what
    # is even a candidate.
    SHAKE_MIN_RADIUS_FRAC = 0.022
    SHAKE_MAX_RADIUS_FRAC = 0.14

    PROGRESS_MIN_STEP = 9.0
    # How far the boundary must stand above the profile's other steps. The 90th
    # percentile rather than the median, so a profile that is smooth apart from
    # a few texture edges is judged against those edges and not against its
    # smooth majority.
    PROGRESS_MIN_STEP_RATIO = 3.0
    # And the two sides must be flat: the Lab distance across the boundary over
    # the scatter within each side, which rejects a wall whose shading happens
    # to change partway along.
    PROGRESS_MIN_FLATNESS = 1.0
    # How much more vivid the fill is than the remainder, as saturation plus
    # value. This is the colour path's own separation requirement, and it is
    # placed past the tail of what world texture produces rather than at the
    # crossover: a wrong progress number is believed by everything downstream,
    # while a missing one is covered, because the caller holds the last reading
    # through a gap mid-fight.
    PROGRESS_MIN_VIVIDNESS = 100.0
    # The brightness path's requirements, unchanged from when this reader only
    # had that one. They are kept as their own test rather than folded into the
    # colour path because saturation and value trade off against each other:
    # the rods that draw a white bar draw its remainder in a saturated dark
    # red, so saturation-plus-value collapses a separation of 210 grey levels
    # down to 70 and threw away the case the reader was already good at.
    PROGRESS_MIN_SEPARATION = 80.0
    PROGRESS_MIN_SEP_RATIO = 4.0
    PROGRESS_EDGE_MARGIN = 0.015    # ignore this much at each end: the outline
    PROGRESS_MIN_BAND = 4           # rows; thinner than this is not the bar
    PROGRESS_MAX_BAND = 16
    PROGRESS_ROW_TOLERANCE = 2      # columns the bar's rows may disagree by

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
        self._last_progress = 0.0

    def reset_session(self) -> None:
        """Forget per-fight vision state. Call when a new minigame starts."""
        settings = self._config.load_settings()
        self._vision.p.fish_relative_strength = settings.fish_relative_strength
        self._vision.p.fish_max_speed = settings.fish_max_speed
        self._vision.reset()
        self._track_locator.reset()
        self._relocate_countdown = 0
        self._last_progress = 0.0

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

    @staticmethod
    def padded_progress_roi(roi_bounds):
        """The progress ROI grown vertically so the bar's outline is in frame.

        :meth:`_progress_rows` finds the bar by that outline, and a rectangle
        dragged by hand often sits a few pixels off it -- on the sample clips it
        misses the bar altogether more often than not. Growing the capture by
        the ROI's own height at each end costs a negligible amount of area and
        lets the reader correct the calibration from the image instead of
        depending on it.
        """
        height = max(0.004, roi_bounds.y_end - roi_bounds.y_start)
        return ROIBounds(
            x_start=roi_bounds.x_start,
            x_end=roi_bounds.x_end,
            y_start=max(0.0, roi_bounds.y_start - height),
            y_end=min(1.0, roi_bounds.y_end + height),
        )

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

    def _progress_step(self, profile: np.ndarray) -> Optional[Tuple[int, float, float]]:
        """Sharpest colour step along one Lab column profile.

        Returns ``(column, magnitude, rival)`` -- where the profile changes
        most abruptly, by how much, and how big the profile's other steps get,
        for comparison. The ends are trimmed first: the bar is drawn with an
        outline, and its two corners are steps too.

        A light blur precedes the difference because the boundary is
        antialiased over a column or two, and without it the step is split
        between them and can lose to noise.
        """
        width = len(profile)
        margin = max(3, int(width * self.PROGRESS_EDGE_MARGIN))
        core = profile[margin:width - margin]
        if len(core) < 12:
            return None
        smoothed = cv2.GaussianBlur(core.reshape(-1, 1, 3), (1, 5), 0).reshape(-1, 3)
        steps = np.linalg.norm(np.diff(smoothed, axis=0), axis=1)
        if steps.size == 0:
            return None
        peak = int(np.argmax(steps))
        return margin + peak + 1, float(steps[peak]), float(np.percentile(steps, 90))

    def _progress_bands(self, frame: np.ndarray) -> list:
        """Candidate row bands for the progress bar, most promising first.

        The ROI is calibrated by dragging a rectangle once, and a few pixels of
        error there is normal -- on the sample clips the calibrated rectangle
        misses the bar outright on five of seven, which is most of why progress
        went unread. So the bar is found from the image instead.

        What identifies it is that every one of its rows breaks at the *same*
        column: the fill boundary is vertical. World texture also produces a
        sharpest step in each row, but at a column that wanders from row to
        row, and a gradient in the scenery produces one that drifts steadily.
        Neither survives the requirement that a run of rows agree on it to
        within a couple of columns.

        Several runs can pass that test, so this returns all of them and lets
        :meth:`detect_progress` settle it by reading each: the bar is the band
        that reads as a bar. Ranking alone cannot decide it -- measured on the
        sample clips, the true band is sometimes the tighter run and sometimes
        the looser one, and sometimes has the weaker step of the two.

        Brightness deliberately plays no part. An earlier version looked for
        the bar's outline as two rows brighter than the fill between them,
        which is true of the rods that draw a dark bar and false of the ones
        that draw a white one -- there the fill is the brightest thing in the
        strip, and the test found the scenery instead.
        """
        if frame is None or frame.shape[0] < self.PROGRESS_MIN_BAND:
            return []
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB).astype(np.float32)
        columns = []
        for y in range(lab.shape[0]):
            found = self._progress_step(lab[y])
            columns.append(None if found is None else found[0])

        bands = []
        start = 0
        while start < len(columns):
            if columns[start] is None:
                start += 1
                continue
            lo = hi = columns[start]
            end = start
            while end + 1 < len(columns) and columns[end + 1] is not None:
                nlo = min(lo, columns[end + 1])
                nhi = max(hi, columns[end + 1])
                if nhi - nlo > self.PROGRESS_ROW_TOLERANCE:
                    break
                lo, hi = nlo, nhi
                end += 1
            height = end - start + 1
            if self.PROGRESS_MIN_BAND <= height <= self.PROGRESS_MAX_BAND:
                bands.append((height, hi - lo, start, end + 1))
            # Only maximal runs: a run that starts inside one already found
            # describes the same band, and trying its prefixes wastes work.
            start = end + 1

        bands.sort(key=lambda b: (-b[0], b[1]))
        return [(top, bottom) for _, _, top, bottom in bands]

    def detect_progress(self, progress_frame: np.ndarray) -> Optional[float]:
        """Progress-bar fill as 0.0-1.0, or None when there is no bar to read.

        Returning a number unconditionally is what made this dangerous. The ROI
        is a small strip of screen, and when no minigame is up the game draws
        the world there instead; on tests/clips/errors.mov that world is bright
        sand, and a reader that only looks for "where does bright stop" reported
        99% progress against it. A confident wrong number is worse than no
        number, because everything downstream believes it.

        So the bar is located first, from its outline (:meth:`_progress_rows`),
        and the fill boundary is then read as a *step* in the colour profile
        along it. A step, rather than the split that best separates the two
        halves on brightness, because neither assumption in that phrasing
        survives contact with the rods:

        * Brightness need not change. Several rods draw the bar as a saturated
          gradient whose unfilled remainder is the same gradient washed out --
          the step is in colour, not in grey, and read in grey the bar was
          simply invisible. Measured over the sample clips the old reader got a
          number on 68-83% of frames for the two white-bar rods and 5% for
          every other rod.
        * The two sides need not be flat. A rainbow fill sweeps yellow to blue
          across its own length, so the best-separating split lands in the
          middle of the fill rather than at its end. The boundary is the only
          place the profile changes *abruptly*, which is what picking the
          sharpest step finds.

        Presence is then decided from the same measurement: a real boundary is
        a step that stands well clear of the profile's ordinary variation, with
        the fill the more vivid side. World texture has no outline to find and
        no such step.

        A completely full or completely empty bar has no boundary to find and so
        reads as absent. That is left to the caller, which knows from the reel
        track whether a fight is in progress and can hold the last reading
        rather than inventing one.
        """
        if progress_frame is None or progress_frame.size == 0:
            return None
        if progress_frame.shape[1] < 16:
            return None

        # Every candidate band is read and the most bar-like wins -- not the
        # first that passes, because the true band is sometimes the marginal
        # one and a stricter test would then hand the frame to a patch of
        # scenery that passed more easily. The whole strip is included as a
        # last resort, for a caller that has already cropped tight to the fill.
        candidates = self._progress_bands(progress_frame)
        candidates.append((0, progress_frame.shape[0]))

        best = None
        best_confidence = 0.0
        for top, bottom in candidates:
            read = self._read_progress_band(progress_frame[top:bottom])
            if read is None:
                continue
            fill, confidence = read
            if confidence > best_confidence:
                best_confidence, best = confidence, fill
        return best

    def _read_progress_band(self, strip: np.ndarray) -> Optional[Tuple[float, float]]:
        """``(fill, confidence)`` for one candidate band, or None if it is not a bar.

        Confidence is how far clear of the presence thresholds the band reads,
        so that bands can be compared against each other rather than merely
        accepted or rejected.
        """
        if strip is None or strip.size == 0:
            return None

        lab = cv2.cvtColor(strip, cv2.COLOR_BGR2LAB).astype(np.float32)
        # Median down each column: the fill is uniform vertically, so this
        # rejects the odd overlaid sprite without blurring the boundary.
        profile = np.median(lab, axis=0)
        found = self._progress_step(profile)
        if found is None:
            return None
        cut, step, rival = found

        if step < self.PROGRESS_MIN_STEP:
            return None
        sharpness = step / max(rival, 1e-6)
        if sharpness < self.PROGRESS_MIN_STEP_RATIO:
            return None

        # Two flat regions with one boundary between them. Medians on each
        # side, not means, so a fill that carries a gradient along its own
        # length is still compared at its typical colour.
        width = len(profile)
        margin = max(3, int(width * self.PROGRESS_EDGE_MARGIN))
        fill, rest = profile[margin:cut], profile[cut:width - margin]
        if len(fill) < 2 or len(rest) < 2:
            return None
        jump = float(np.linalg.norm(np.median(fill, axis=0) - np.median(rest, axis=0)))
        spread = float(np.sqrt(
            (fill.var(axis=0).sum() * len(fill) + rest.var(axis=0).sum() * len(rest))
            / (len(fill) + len(rest))
        ))
        flatness = jump / max(spread, 1e-6)
        if flatness < self.PROGRESS_MIN_FLATNESS:
            return None

        # Fill grows from the left and is the more vivid side, whether that
        # means brighter (a white bar on dark) or more saturated (a coloured
        # bar against its own washed-out remainder). A dull left against a
        # vivid right is the world, not a progress bar.
        hsv = cv2.cvtColor(strip, cv2.COLOR_BGR2HSV).astype(np.float32)
        vivid = np.median(hsv, axis=0)[:, 1:].sum(axis=1)
        left, right = vivid[margin:cut], vivid[cut:width - margin]
        contrast = float(left.mean() - right.mean())

        # Two ways to be a bar, because the rods draw two kinds. Either the
        # fill is much brighter than the remainder -- the white-bar case this
        # reader has always handled -- or it is much more vivid, which is how a
        # saturated bar differs from its own washed-out remainder.
        grey = np.median(
            cv2.cvtColor(strip, cv2.COLOR_BGR2GRAY), axis=0
        ).astype(np.float32)
        bright_fill, bright_rest = grey[margin:cut], grey[cut:width - margin]
        separation = float(bright_fill.mean() - bright_rest.mean())
        bright_spread = float(np.sqrt(
            (bright_fill.var() * len(bright_fill) + bright_rest.var() * len(bright_rest))
            / (len(bright_fill) + len(bright_rest))
        ))
        by_brightness = (
            separation >= self.PROGRESS_MIN_SEPARATION
            and separation / max(bright_spread, 1e-6) >= self.PROGRESS_MIN_SEP_RATIO
        )
        by_colour = contrast >= self.PROGRESS_MIN_VIVIDNESS
        if not (by_brightness or by_colour):
            return None

        confidence = max(
            separation / self.PROGRESS_MIN_SEPARATION if by_brightness else 0.0,
            contrast / self.PROGRESS_MIN_VIVIDNESS if by_colour else 0.0,
        )
        return float(np.clip(cut / width, 0.0, 1.0)), confidence

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
        if radius < 14 or radius > int(min(h, w) * self.SHAKE_MAX_RADIUS_FRAC):
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

        # Circle search on the desaturated bright-edge mask
        edges = cv2.Canny(white_mask, 60, 160)
        # The search range has to cover everything _shake_button_score is
        # willing to accept, or a button inside the model's own bounds is never
        # offered to it. These were 0.022-0.11 against a scorer accepting up to
        # 0.14, so the largest buttons the scorer recognises could not be found.
        min_r = max(16, int(min(h, w) * self.SHAKE_MIN_RADIUS_FRAC))
        max_r = max(min_r + 10, int(min(h, w) * self.SHAKE_MAX_RADIUS_FRAC))
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
            progress_frame = self.capture_roi(
                self.padded_progress_roi(settings.progress_roi)
            )
        else:
            # One grab for both, rather than one each: see capture_pair.
            band_frame, progress_frame = self.capture_pair(
                settings.bar_roi, self.padded_progress_roi(settings.progress_roi)
            )
            track_box = None
            reading = None
            if band_frame is not None:
                reading = self._vision.read(band_frame, pre_located=True)
                track_box = (0, 0, band_frame.shape[1], band_frame.shape[0])

        # A missing progress reading means different things depending on
        # whether a fight is running. Mid-fight the bar is briefly occluded --
        # by the rod model, by a slash effect, or by being completely full --
        # and the last known value is the best available answer; a zero there
        # would look like the progress collapse that ends a fight. With no
        # minigame on screen there is nothing to hold and zero is correct.
        raw_progress = self.detect_progress(progress_frame)
        minigame_up = reading is not None and reading.bar_left is not None
        if raw_progress is None:
            progress = self._last_progress if minigame_up else 0.0
        else:
            progress = raw_progress
            self._last_progress = raw_progress

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
            result.debug_source = frame
            # Only rendered here for callers that have no telemetry of their
            # own (the offline scripts). The macro re-renders from
            # debug_source once it knows what the controller decided.
            if self.debug_mode:
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
        """Build an overlaid visualization of the actual captured bar ROI.

        Every colour comes from :data:`OVERLAY_COLORS`, which the GUI also
        reads to draw the key. Pass ``extras`` (predicted_fish_x, effective_bar,
        progress_smooth, macro_state) to get the control-loop overlays; without
        it only what the detector itself measured is drawn.
        """
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
            cv2.rectangle(
                vis, (0, new_h - prog_h), (px, new_h), OVERLAY_COLORS["progress"], -1
            )

        # Smoothed progress, drawn whether or not there is any raw fill: at the
        # moment raw collapses to zero the smoothed value is exactly what the
        # exit logic is still acting on, so that is when it most needs showing.
        smooth_prog = extras.get("progress_smooth", 0.0)
        if smooth_prog and smooth_prog > 0:
            spx = _x_px(smooth_prog)
            cv2.line(
                vis, (spx, new_h - prog_h), (spx, new_h),
                OVERLAY_COLORS["progress_smooth"], 2,
            )

        # Overlays
        pad_y = 2
        track_h = h * scale_factor
        mid_y = track_h // 2

        # Control bar bracket
        if result.bar_left is not None and result.bar_right is not None:
            lx = _x_px(result.bar_left)
            rx = _x_px(result.bar_right)
            if lx is not None and rx is not None and rx > lx:
                bar_color = (
                    OVERLAY_COLORS["bar_on"] if result.on_target
                    else OVERLAY_COLORS["bar_off"]
                )
                # Thick bracket
                cv2.line(vis, (lx, pad_y), (rx, pad_y), bar_color, 3)
                cv2.line(vis, (lx, track_h - pad_y), (rx, track_h - pad_y), bar_color, 3)
                cv2.line(vis, (lx, pad_y), (lx, track_h - pad_y), bar_color, 2)
                cv2.line(vis, (rx, pad_y), (rx, track_h - pad_y), bar_color, 2)

        # Effective bar center — where the controller believes the bar will be
        # by the time its command lands, which is what it actually aims with.
        effective = extras.get("effective_bar")
        if effective is not None:
            ex = _x_px(effective)
            if ex is not None:
                cv2.line(
                    vis, (ex, pad_y + 4), (ex, track_h - pad_y - 4),
                    OVERLAY_COLORS["bar_aim"], 2,
                )

        # Predicted fish
        predicted = extras.get("predicted_fish_x")
        if predicted is not None:
            px = _x_px(predicted)
            if px is not None:
                for y in range(pad_y, track_h - pad_y, 8):
                    cv2.line(
                        vis, (px, y), (px, min(y + 4, track_h - pad_y)),
                        OVERLAY_COLORS["fish_predicted"], 2,
                    )

        # Actual fish
        if result.fish_x is not None:
            fx = _x_px(result.fish_x)
            if fx is not None:
                cv2.line(
                    vis, (fx, pad_y), (fx, track_h - pad_y),
                    OVERLAY_COLORS["fish"], 3,
                )
                cv2.circle(vis, (fx, mid_y), 6, OVERLAY_COLORS["fish"], -1)
                cv2.circle(vis, (fx, mid_y), 3, (255, 255, 255), -1)

        # State text
        state = extras.get("macro_state")
        if state:
            # Add a slight dark background for text readability
            (tw, _th), _ = cv2.getTextSize(state, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(vis, (4, 4), (12 + tw, 28), (0, 0, 0), -1)
            cv2.putText(vis, state, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        return vis
