"""
viewport.py — Locate the Roblox game viewport inside a full-screen capture.

Live, the macro captures the Roblox window directly and never needs this. It
exists for the offline harness, where the recordings are of the whole desktop
and therefore include the macOS menu bar, the window title bar and the Dock —
all of which are large, dark, high-contrast bands that otherwise compete with
the reel UI for attention.

Those chrome elements have one thing the game world never has: a horizontal edge
running the *entire* width of the screen. The game viewport is the tall region
between the lowest such edge near the top and the highest one near the bottom.
"""

from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np

FULL_WIDTH_FRAC = 0.97     # an edge must span this much of the frame to be chrome
EDGE_THRESHOLD = 20.0
TOP_SEARCH = 0.14          # look for the title bar within this fraction of height
BOTTOM_SEARCH = 0.80       # look for the Dock below this fraction


def find_viewport(frame: np.ndarray) -> Tuple[int, int, int, int]:
    """Return (x0, y0, x1, y1) of the game viewport.

    Falls back to the whole frame when no chrome is found, which is the correct
    answer for a capture that is already just the window.
    """
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
    dy = np.abs(cv2.Sobel(cv2.GaussianBlur(gray, (3, 3), 0), cv2.CV_32F, 0, 1, ksize=3))
    full_width = (dy > EDGE_THRESHOLD).mean(axis=1) >= FULL_WIDTH_FRAC

    top_rows = np.flatnonzero(full_width[: int(h * TOP_SEARCH)])
    y0 = int(top_rows.max()) + 1 if top_rows.size else 0

    bottom_start = int(h * BOTTOM_SEARCH)
    bottom_rows = np.flatnonzero(full_width[bottom_start:])
    y1 = bottom_start + int(bottom_rows.min()) if bottom_rows.size else h

    if y1 - y0 < h * 0.3:      # implausible; trust the whole frame instead
        return 0, 0, w, h
    return 0, y0, w, y1


def roi_to_pixels(
    viewport: Tuple[int, int, int, int], roi
) -> Tuple[int, int, int, int]:
    """Map a normalised ROI (fractions of the window) onto viewport pixels."""
    x0, y0, x1, y1 = viewport
    vw, vh = x1 - x0, y1 - y0
    return (
        x0 + int(roi["x_start"] * vw),
        y0 + int(roi["y_start"] * vh),
        x0 + int(roi["x_end"] * vw),
        y0 + int(roi["y_end"] * vh),
    )
