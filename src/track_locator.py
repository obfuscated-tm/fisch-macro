"""
track_locator.py — Find the reel minigame track inside a frame.

Locating the track matters for more than convenience. Bar and fish positions are
normalised against the track span, so if that span is re-derived differently on
each tick, a motionless bar appears to move and the controller brakes against
velocity that does not exist. The job here is therefore to produce a *stable*
box, not merely a plausible one.

Finding it by colour does not work. Across rods and biomes the track background
has been observed as near-black over lava and as muted purple over a stone wall,
and the control bar inside it renders white, dark red, or a saturated rainbow
gradient depending on the rod. Nor does any single-frame appearance test give a
stable horizontal extent: measured over consecutive frames of one clip, the best
such estimate wandered by ~70px, which is exactly the instability that makes a
motionless bar look like it is moving.

So the two axes are handled differently, according to which one has to be stable:

*Horizontally* the span comes from the caller — in practice the calibrated ROI.
A calibrated constant is perfectly stable by construction, which no per-frame
estimate can match, and the ROI only has to be dragged once per setup rather
than retuned per rod.

*Vertically* the band is refined from the frame, because the exact track rows
matter for reading the bar and the fish, and a calibrated rectangle is often off
by a dozen pixels. Rows are scored by the longest contiguous run of strong
vertical gradient they contain: world rows score low (tens of pixels), rows
spanned by the track's edges score high (hundreds). Bands are then filtered by
height, which is what separates the track from its neighbours — in a sample
frame the track band was 40px tall, the fish marker above it 10px, and the
progress bar below it 3px.
"""

from __future__ import annotations

from collections import deque
from typing import Deque, List, Optional, Tuple

import cv2
import numpy as np


# Search region as fractions of the frame, used when the caller does not supply
# one. Production passes an explicit box derived from the calibrated ROI.
SEARCH = (0.15, 0.60, 0.85, 0.99)   # x0, y0, x1, y1

EDGE_THRESHOLD = 12.0      # |dI/dy| above this counts as a UI edge
MIN_EDGE_RUN_FRAC = 0.045  # a row must span at least this fraction of the frame
EDGE_GAP_BRIDGE = 0.012    # bridge edge gaps up to this fraction (rounded corners)

MIN_BAND_HEIGHT = 10       # thinner than this is the progress bar or a marker
MAX_BAND_HEIGHT = 90
MIN_TRACK_WIDTH_FRAC = 0.12


def _runs(flags: np.ndarray) -> List[Tuple[int, int]]:
    """All contiguous True runs as (start, end_exclusive)."""
    if flags.size == 0 or not flags.any():
        return []
    padded = np.concatenate(([False], flags.astype(bool), [False]))
    edges = np.diff(padded.astype(np.int8))
    return list(
        zip(np.flatnonzero(edges == 1).tolist(), np.flatnonzero(edges == -1).tolist())
    )


def _bridged_runs(flags: np.ndarray, gap: int) -> List[Tuple[int, int]]:
    """Runs with short gaps closed — the track's edges are broken by the control
    bar's own outline and by rounded corners, so they are never one clean run."""
    runs = _runs(flags)
    if not runs:
        return []
    merged = [runs[0]]
    for start, end in runs[1:]:
        if start - merged[-1][1] <= gap:
            merged[-1] = (merged[-1][0], end)
        else:
            merged.append((start, end))
    return merged


def _longest_run_len(flags: np.ndarray, gap: int) -> Tuple[int, int, int]:
    """(length, start, end) of the longest gap-bridged run."""
    runs = _bridged_runs(flags, gap)
    if not runs:
        return 0, 0, 0
    start, end = max(runs, key=lambda r: r[1] - r[0])
    return end - start, start, end


def find_track_candidates(
    frame: np.ndarray, search: Optional[Tuple[int, int, int, int]] = None
) -> List[Tuple[int, int, int, int, float]]:
    """Candidate track bands as (x0, y0, x1, y1, score), best first.

    Coordinates are absolute within *frame*.
    """
    h, w = frame.shape[:2]
    if search is None:
        sx0, sy0, sx1, sy1 = (
            int(SEARCH[0] * w), int(SEARCH[1] * h),
            int(SEARCH[2] * w), int(SEARCH[3] * h),
        )
    else:
        sx0, sy0, sx1, sy1 = search
    sx0, sy0 = max(0, sx0), max(0, sy0)
    sx1, sy1 = min(w, sx1), min(h, sy1)
    if sx1 - sx0 < 32 or sy1 - sy0 < MIN_BAND_HEIGHT:
        return []

    region = frame[sy0:sy1, sx0:sx1]
    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY).astype(np.float32)
    dy = cv2.Sobel(cv2.GaussianBlur(gray, (3, 3), 0), cv2.CV_32F, 0, 1, ksize=3)


    gap = max(3, int(w * EDGE_GAP_BRIDGE))
    min_run = w * MIN_EDGE_RUN_FRAC

    # Score each row by its longest run of strong edge, in either direction:
    # the track's top edge is a rise and its bottom edge a fall, and rows in
    # between pick up the control bar's own outline.
    scores = np.zeros(dy.shape[0])
    spans: List[Tuple[int, int]] = []
    for y in range(dy.shape[0]):
        up_len, up_s, up_e = _longest_run_len(dy[y] > EDGE_THRESHOLD, gap)
        dn_len, dn_s, dn_e = _longest_run_len(dy[y] < -EDGE_THRESHOLD, gap)
        if up_len >= dn_len:
            scores[y], span = up_len, (up_s, up_e)
        else:
            scores[y], span = dn_len, (dn_s, dn_e)
        spans.append(span)

    candidates = []
    for band_start, band_end in _runs(scores >= min_run):
        height = band_end - band_start
        if not (MIN_BAND_HEIGHT <= height <= MAX_BAND_HEIGHT):
            continue
        # The horizontal extent is the caller's search span, deliberately: see
        # the module docstring on why it is not re-derived here.
        score = float(height) * float(np.median(scores[band_start:band_end]))
        candidates.append((sx0, sy0 + band_start, sx1, sy0 + band_end, score))

    candidates.sort(key=lambda c: c[4], reverse=True)
    return candidates


def locate_track(
    frame: np.ndarray,
    search: Optional[Tuple[int, int, int, int]] = None,
    pad_y: int = 4,
) -> Optional[Tuple[int, int, int, int]]:
    """Best-guess box (x0, y0, x1, y1) around the reel track.

    Horizontally the box is the track itself, with no padding: it defines the
    coordinate frame that positions are normalised against. Vertical padding is
    kept so the caller can still resolve the exact track rows.
    """
    candidates = find_track_candidates(frame, search)
    if not candidates:
        return None
    x0, y0, x1, y1, _ = candidates[0]
    h = frame.shape[0]
    return (x0, max(0, y0 - pad_y), x1, min(h, y1 + pad_y))


class TrackLocator:
    """Stabilises :func:`locate_track` across frames.

    The per-frame vertical estimate jitters by a row or two, so it is smoothed
    by taking the median of recent frames. It is deliberately *not* unioned:
    a union only ever grows, and over a long fight it swallows the world rows
    above and below the track until the reading is taken from mostly scenery.
    The horizontal span comes from the caller and is already constant, so it is
    carried through unchanged.

    Call :meth:`reset` when a new minigame starts, or when the window moves.
    """

    def __init__(self, memory_frames: int = 15, miss_tolerance: int = 90):
        self.memory_frames = memory_frames
        self.miss_tolerance = miss_tolerance
        self.reset()

    def reset(self) -> None:
        self._history: Deque[Tuple[int, int]] = deque(maxlen=self.memory_frames)
        self._box: Optional[Tuple[int, int, int, int]] = None
        self._age = 0

    @property
    def box(self) -> Optional[Tuple[int, int, int, int]]:
        return self._box

    def update(
        self, frame: np.ndarray, search: Optional[Tuple[int, int, int, int]] = None
    ) -> Optional[Tuple[int, int, int, int]]:
        """Fold this frame's estimate into the smoothed track box."""
        found = locate_track(frame, search)
        if found is None:
            self._age += 1
            if self._age > self.miss_tolerance:
                self.reset()
            return self._box

        self._age = 0
        fx0, fy0, fx1, fy1 = found

        # A jump larger than the band's own height means the UI actually moved
        # (window resize, resolution change) rather than the estimate wobbling.
        if self._history:
            median_y0 = int(np.median([y0 for y0, _ in self._history]))
            median_y1 = int(np.median([y1 for _, y1 in self._history]))
            if abs(fy0 - median_y0) > max(8, median_y1 - median_y0):
                self._history.clear()

        self._history.append((fy0, fy1))
        y0 = int(np.median([v for v, _ in self._history]))
        y1 = int(np.median([v for _, v in self._history]))
        self._box = (fx0, y0, fx1, y1)
        return self._box
