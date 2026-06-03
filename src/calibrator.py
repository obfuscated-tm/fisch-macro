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
from typing import Optional, Tuple, List

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

    def auto_detect_colors(self, bar_frame: np.ndarray) -> Optional[dict]:
        """Analyse an active-bar frame and auto-calibrate HSV colour ranges.

        The method:
        1. Converts to HSV and filters out very dark / very bright pixels.
        2. Runs K-means (k=3) on the remaining pixel HSV values.
        3. Identifies clusters matching the fish indicator (pink/magenta,
           H ≈ 140–170) and the control-bar arrows (green/cyan, H ≈ 40–90).
        4. Computes padded min/max HSV bounds for each cluster.

        Parameters
        ----------
        bar_frame : np.ndarray
            BGR image of the minigame bar region.

        Returns
        -------
        dict or None
            Keys: ``fish_hsv_low``, ``fish_hsv_high``,
            ``bar_hsv_low``, ``bar_hsv_high`` — each a list of three ints.
            Returns *None* if clustering fails or the expected hue clusters
            cannot be identified.
        """
        if bar_frame is None or bar_frame.size == 0:
            self.logger.warning("auto_detect_colors called with empty frame")
            return None

        hsv = cv2.cvtColor(bar_frame, cv2.COLOR_BGR2HSV)
        h, w = hsv.shape[:2]

        # Flatten to (N, 3) and filter extremes
        pixels = hsv.reshape(-1, 3).astype(np.float32)

        # Keep pixels that are not too dark and not VFX-bright
        keep = (pixels[:, 2] > 30) & (pixels[:, 2] < 240)
        pixels = pixels[keep]

        if len(pixels) < 50:
            self.logger.warning(
                "Not enough valid pixels for clustering (%d)", len(pixels)
            )
            return None

        # K-means clustering (k=3: background, fish, bar)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
        k = 3
        try:
            _, labels, centres = cv2.kmeans(
                pixels, k, None, criteria, 10, cv2.KMEANS_PP_CENTERS
            )
        except cv2.error as exc:
            self.logger.error("K-means failed: %s", exc)
            return None

        labels = labels.flatten()

        # Analyse each cluster
        fish_cluster = None
        bar_cluster = None

        for i in range(k):
            cluster_pixels = pixels[labels == i]
            if len(cluster_pixels) == 0:
                continue

            mean_h = float(np.mean(cluster_pixels[:, 0]))
            mean_s = float(np.mean(cluster_pixels[:, 1]))

            self.logger.debug(
                "Cluster %d: count=%d  mean_H=%.1f  mean_S=%.1f  mean_V=%.1f",
                i,
                len(cluster_pixels),
                mean_h,
                mean_s,
                float(np.mean(cluster_pixels[:, 2])),
            )

            # Fish indicator hue range: pink/magenta (H ≈ 140–170 in OpenCV 0-180)
            if 140 <= mean_h <= 170 and mean_s > 40:
                if fish_cluster is None or len(cluster_pixels) < len(
                    pixels[labels == fish_cluster]
                ):
                    fish_cluster = i

            # Control bar arrows hue range: green/cyan (H ≈ 40–90)
            if 40 <= mean_h <= 90 and mean_s > 40:
                if bar_cluster is None or len(cluster_pixels) < len(
                    pixels[labels == bar_cluster]
                ):
                    bar_cluster = i

        if fish_cluster is None and bar_cluster is None:
            self.logger.warning("Could not identify fish or bar clusters")
            return None

        result = {}

        if fish_cluster is not None:
            fp = pixels[labels == fish_cluster]
            result["fish_hsv_low"] = self._compute_hsv_bound(fp, lower=True)
            result["fish_hsv_high"] = self._compute_hsv_bound(fp, lower=False)
            self.logger.info(
                "Fish HSV: %s → %s", result["fish_hsv_low"], result["fish_hsv_high"]
            )
        else:
            self.logger.warning("Fish cluster not found — skipping")

        if bar_cluster is not None:
            bp = pixels[labels == bar_cluster]
            result["bar_hsv_low"] = self._compute_hsv_bound(bp, lower=True)
            result["bar_hsv_high"] = self._compute_hsv_bound(bp, lower=False)
            self.logger.info(
                "Bar HSV: %s → %s", result["bar_hsv_low"], result["bar_hsv_high"]
            )
        else:
            self.logger.warning("Bar cluster not found — skipping")

        # Only return if we got *both* ranges
        if "fish_hsv_low" in result and "bar_hsv_low" in result:
            return result

        self.logger.warning("Incomplete auto-detection (found only one cluster)")
        return None

    @staticmethod
    def _compute_hsv_bound(pixels: np.ndarray, lower: bool) -> list:
        """Compute a padded HSV lower or upper bound from a pixel array.

        Parameters
        ----------
        pixels : np.ndarray
            Shape (N, 3) float32, columns are H, S, V.
        lower : bool
            If *True*, compute the lower bound; otherwise the upper bound.

        Returns
        -------
        list of int
            Three-element list [H, S, V].
        """
        padding = [-15, -30, -30] if lower else [15, 30, 30]
        func = np.min if lower else np.max
        hsv_limits = [180, 255, 255]

        bound = []
        for ch in range(3):
            val = int(func(pixels[:, ch])) + padding[ch]
            val = max(0, min(val, hsv_limits[ch]))
            bound.append(val)
        return bound

    # ------------------------------------------------------------------
    # ROI computation
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

    def generate_mask_preview(
        self,
        frame: np.ndarray,
        hsv_low: list,
        hsv_high: list,
    ) -> np.ndarray:
        """Generate a visualisation of an HSV colour mask over a frame.

        Detected pixels are highlighted in bright green on top of the
        original image so the user can visually verify threshold accuracy.

        Parameters
        ----------
        frame : np.ndarray
            BGR image to analyse.
        hsv_low : list of int
            Lower HSV bound ``[H, S, V]``.
        hsv_high : list of int
            Upper HSV bound ``[H, S, V]``.

        Returns
        -------
        np.ndarray
            BGR visualisation frame.
        """
        if frame is None or frame.size == 0:
            self.logger.warning("generate_mask_preview called with empty frame")
            return np.zeros((100, 200, 3), dtype=np.uint8)

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(
            hsv,
            np.array(hsv_low, dtype=np.uint8),
            np.array(hsv_high, dtype=np.uint8),
        )

        # Build overlay: original dimmed + bright green where mask is active
        vis = frame.copy()
        vis = (vis * 0.5).astype(np.uint8)  # dim the background

        highlight = np.zeros_like(frame)
        highlight[:, :] = (0, 255, 0)  # bright green

        mask_3ch = cv2.merge([mask, mask, mask])
        vis = np.where(mask_3ch > 0, highlight, vis)

        # Add pixel count annotation
        pixel_count = int(np.sum(mask > 0))
        total_pixels = mask.shape[0] * mask.shape[1]
        pct = pixel_count / total_pixels * 100 if total_pixels > 0 else 0.0
        label = f"Detected: {pixel_count} px ({pct:.1f}%)"
        cv2.putText(
            vis, label, (8, 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
        )

        return vis
