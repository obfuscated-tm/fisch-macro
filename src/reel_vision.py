"""
reel_vision.py — Structure-based detection for the Fisch reeling minigame.

Replaces per-rod HSV colour matching with a geometric read of the track strip.

The insight is that the minigame track has a fixed *structure* regardless of
which rod is equipped:

    ┌──────────────────────────────────────────────────────┐
    │ dark track background                                │
    │        ▓▓▓▓▓▓▓▓▓▓▓▓▓                 │               │  ← wide run  = control bar
    │        (control bar)                (fish)           │  ← narrow run = fish
    └──────────────────────────────────────────────────────┘

Both the control bar and the fish indicator are "not the background", and they
are separated by *width*, not by colour: the bar is a wide contiguous run of
columns, the fish is a narrow one. That holds whether the bar renders white
(on target) or red (off target), and whatever colour the rod tints the fish.

Consequently:
  • no per-rod HSV calibration is needed for the bar or the fish
  • "on target" is decided geometrically (is the fish between the bar edges),
    never by sampling the bar's colour
  • positions are normalised to the detected *track span*, not the ROI, so
    "bar is at the wall" means what it says

This module is deliberately free of screen-capture and window-geometry
concerns: it takes a BGR frame and returns a reading, so it can be run
offline against recorded frames (see scripts/test_detection.py).
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, List, Optional, Sequence, Tuple

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------

@dataclass
class VisionParams:
    """Geometry priors for the reel track. These are shape constraints, not
    colours, so they hold across rods."""

    # --- track band location within the ROI ---
    track_max_saturation: int = 130   # rows above this median S are not track
    track_max_value: int = 95         # rows above this median V are not track
    track_row_pad: float = 0.18       # trim this fraction off the band's top/bottom

    # --- lava / background rejection when finding the track's horizontal span ---
    lava_min_value: int = 200
    lava_min_saturation: int = 150
    min_track_span_frac: float = 0.5  # below this, distrust the span and use full ROI

    # --- segmentation ---
    # Boundaries are colour *steps* along the track, not colour matches, so a
    # bar with a smooth gradient stays one segment.
    segment_min_step: float = 10.0
    segment_percentile: float = 96.0
    segment_step_frac: float = 0.40
    group_tolerance: float = 18.0     # Lab distance within which colours are "the same"
    # Antialiasing at the track's rounded ends leaves 1-3px slivers. Left in,
    # they get mistaken for the background and invert the whole classification.
    min_segment_frac: float = 0.006
    run_merge_gap: int = 2

    # --- control bar shape prior (fraction of track width) ---
    # Bar width is set by the rod's Control stat and does not change during a
    # fight: per the Fisch wiki, 0 control gives a bar 30% of the track and each
    # +0.01 adds 1%, up to 100%. Widths seen here are fractions of the
    # calibrated ROI rather than of the true track, so the bounds stay loose —
    # an ROI wider than the track scales every measurement down. The real
    # filtering is done by bar_width_tolerance below.
    bar_min_width_frac: float = 0.03
    bar_max_width_frac: float = 1.0

    # The width is fixed by the rod, so once it is known a candidate far from it
    # cannot be the bar — and a bogus bar centre feeds the controller phantom
    # velocity, which the braking term then squares.
    #
    # Both sides are gated, but not symmetrically, because the two errors are
    # not symmetric. Near a wall the ROI clips the bar, so a *measured* width
    # below the established one is normal and the narrow gate stays loose.
    # Nothing clips a bar wider, though: clipping can only remove columns. A
    # reading wider than the rod's own bar is always the segmentation having
    # merged the bar with something beside it — an off-target red tint, a slash
    # effect — and on tests/clips/errors.mov that is exactly what happens a few
    # seconds into each fight, with widths of 0.53 and 0.69 against a rod whose
    # bar is 0.20 and readings that pin bar_right to the ROI edge.
    bar_min_width_ratio: float = 0.45
    bar_max_width_ratio: float = 1.25
    bar_width_samples: int = 12

    # --- fish shape prior (fraction of track width) ---
    fish_max_width_frac: float = 0.07
    fish_min_distance: float = 12.0   # Lab distance from both background and bar

    # --- fish motion gate ---
    # The control bar has left/right arrow glyphs printed inside it. They are
    # narrow and differ in colour from the bar fill, so on shape and colour
    # alone they look exactly like the fish, and the detector would occasionally
    # jump to one and back. Measured on the sample clips, those false latches
    # landed at bar-relative 0.11-0.12 and 0.87-0.88 — the arrow positions.
    #
    # What separates them from the fish is motion: the fish moves continuously,
    # so a candidate implying a jump the fish could not physically make is not
    # the fish. The gate is generous enough to pass a genuine dart (4 track
    # widths/s is well above anything observed) while rejecting a jump across
    # the bar, which at a 20ms tick is an order of magnitude larger.
    fish_max_speed: float = 4.0        # track widths per second
    fish_jump_slack: float = 0.03      # tolerance for detection noise

    # The fish cannot leave the middle of the track. Per the wiki's Resilience
    # page it is confined to 3%-90%, and every movement is clamped into that
    # band, so a candidate outside it is something else that happens to be
    # narrow -- on tests/clips/errors.mov the detector reports 0.951 and 0.969
    # while the bar sits against the right wall, which is the bar's own edge
    # being read as the fish.
    #
    # The bounds are generous because they are expressed against the calibrated
    # ROI rather than the true track, and the two differ by a couple of percent
    # at each end; only clearly impossible readings are rejected.
    fish_track_bounds: Tuple[float, float] = (0.0, 0.94)
    fish_reacquire_frames: int = 6     # give up and re-acquire after this many misses

    # The arrow glyphs are not merely narrow — they are *weak*: measured against
    # the bar fill they sit around 40 units away in Lab, while the fish sits at
    # 104-233 depending on the rod. So the fish normally wins on strength, and
    # the false latches happen only when the fish is momentarily missed and an
    # arrow becomes the best of what is left.
    #
    # Rejecting anything much weaker than the fish we have been tracking fixes
    # that without a fixed threshold, which cannot work: raising the absolute
    # cut-off high enough to exclude the arrows on one rod lost the fish 31% of
    # the time on another. The reference adapts per rod, so nothing needs tuning
    # when the tackle changes.
    fish_relative_strength: float = 0.45
    fish_strength_memory: float = 0.2

    # --- temporal priors ---
    bar_width_memory: float = 0.15    # EMA alpha for the learned bar width
    continuity_weight: float = 0.35   # how much to favour candidates near last frame

    on_target_tolerance: float = 0.005


@dataclass
class ReelReading:
    """One frame's read of the minigame track."""

    ok: bool = False
    bar_left: Optional[float] = None    # normalised to the track span
    bar_right: Optional[float] = None
    fish_x: Optional[float] = None
    on_target: bool = False
    confidence: float = 0.0

    # Diagnostics — consumed by the debug overlay and the offline harness.
    track_x0: int = 0
    track_x1: int = 0
    track_y0: int = 0
    track_y1: int = 0
    bg_lab: Optional[np.ndarray] = None
    bar_lab: Optional[np.ndarray] = None
    distance: Optional[np.ndarray] = None   # per-column Lab distance to background
    threshold: float = 0.0
    runs: List[Tuple[int, int]] = field(default_factory=list)
    fish_inside_bar: bool = False
    notes: str = ""

    @property
    def bar_center(self) -> Optional[float]:
        if self.bar_left is None or self.bar_right is None:
            return None
        return (self.bar_left + self.bar_right) / 2.0

    @property
    def bar_width(self) -> Optional[float]:
        if self.bar_left is None or self.bar_right is None:
            return None
        return self.bar_right - self.bar_left


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _longest_true_run(flags: np.ndarray) -> Optional[Tuple[int, int]]:
    """Return (start, end_exclusive) of the longest contiguous True run."""
    if flags.size == 0 or not flags.any():
        return None
    padded = np.concatenate(([False], flags.astype(bool), [False]))
    edges = np.diff(padded.astype(np.int8))
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1)
    best = int(np.argmax(ends - starts))
    return int(starts[best]), int(ends[best])


def _runs_from_mask(flags: np.ndarray, merge_gap: int) -> List[Tuple[int, int]]:
    """All contiguous True runs as (start, end_exclusive), gaps <= merge_gap bridged."""
    if flags.size == 0 or not flags.any():
        return []
    padded = np.concatenate(([False], flags.astype(bool), [False]))
    edges = np.diff(padded.astype(np.int8))
    starts = list(np.flatnonzero(edges == 1))
    ends = list(np.flatnonzero(edges == -1))

    merged: List[Tuple[int, int]] = []
    for s, e in zip(starts, ends):
        if merged and s - merged[-1][1] <= merge_gap:
            merged[-1] = (merged[-1][0], int(e))
        else:
            merged.append((int(s), int(e)))
    return merged


# ---------------------------------------------------------------------------
# Vision
# ---------------------------------------------------------------------------

class ReelVision:
    """Stateful reader for the reel track.

    State is only used for *continuity* — preferring candidates near where the
    bar and fish were last frame, and remembering how wide this rod's control
    bar is. Every frame is still segmented from scratch, so a bad frame cannot
    poison later ones beyond one tick.
    """

    def __init__(self, params: Optional[VisionParams] = None):
        self.p = params or VisionParams()
        self.reset()

    def reset(self) -> None:
        self._bg_lab: Optional[np.ndarray] = None
        self._width_samples: Deque[float] = deque(maxlen=48)
        self._established_width: Optional[float] = None
        self._last_fish_time: Optional[float] = None
        self._fish_reject_streak: int = 0
        self._fish_strength: Optional[float] = None
        self._bar_width_prior: Optional[float] = None
        self._last_bar_center: Optional[float] = None
        self._last_fish_x: Optional[float] = None

    # -- track band -----------------------------------------------------

    def _find_track_rows(self, hsv: np.ndarray) -> Tuple[int, int, bool]:
        """Locate the dark horizontal track band inside the ROI.

        Returns (y0, y1_exclusive, found). Falls back to the middle of the ROI
        when the band cannot be isolated.
        """
        h = hsv.shape[0]
        row_v = np.median(hsv[:, :, 2], axis=1)
        row_s = np.median(hsv[:, :, 1], axis=1)

        is_track = (row_v <= self.p.track_max_value) & (row_s <= self.p.track_max_saturation)
        run = _longest_true_run(is_track)

        if run is None or (run[1] - run[0]) < 3:
            # No clear band — assume the ROI is roughly the track already.
            lo = int(h * 0.2)
            hi = max(lo + 1, int(h * 0.8))
            return lo, hi, False

        y0, y1 = run
        pad = int((y1 - y0) * self.p.track_row_pad)
        y0c, y1c = y0 + pad, y1 - pad
        if y1c - y0c < 2:
            y0c, y1c = y0, y1
        return y0c, y1c, True

    def _find_track_span(self, hsv: np.ndarray, y0: int, y1: int) -> Tuple[int, int]:
        """Horizontal extent of the track.

        The ROI usually overhangs the track ends, so it can spill onto whatever
        the world is rendering behind the UI (lava, water, sky). Those columns
        are trimmed *inward from each edge only* — never from the middle, since
        a bright saturated fish indicator sitting mid-track would otherwise be
        carved straight out of the frame.
        """
        w = hsv.shape[1]
        band = hsv[y0:y1]
        lava = (band[:, :, 2] >= self.p.lava_min_value) & (
            band[:, :, 1] >= self.p.lava_min_saturation
        )
        col_lava = np.mean(lava, axis=0) > 0.5

        x0 = 0
        while x0 < w and col_lava[x0]:
            x0 += 1
        x1 = w
        while x1 > x0 and col_lava[x1 - 1]:
            x1 -= 1

        if (x1 - x0) < w * self.p.min_track_span_frac:
            return 0, w
        return x0, x1

    # -- column signature -----------------------------------------------

    @staticmethod
    def _column_lab(frame: np.ndarray, y0: int, y1: int, x0: int, x1: int) -> np.ndarray:
        """Per-column median colour of the track strip, in CIE Lab."""
        strip = frame[y0:y1, x0:x1]
        lab = cv2.cvtColor(strip, cv2.COLOR_BGR2LAB).astype(np.float32)
        return np.median(lab, axis=0)  # (width, 3)

    # -- candidate selection ---------------------------------------------

    # -- segmentation ------------------------------------------------------

    def _segment(self, col_lab: np.ndarray) -> List[Tuple[int, int]]:
        """Split the track into runs of uniform colour.

        Boundaries are steps in the column colour profile. Using steps rather
        than absolute colour matters because the control bar can carry a smooth
        left-to-right gradient — a rainbow, on some rods — which a colour test
        would split into pieces, while a step test ignores it entirely and fires
        only at the bar's actual edges.
        """
        if col_lab.shape[0] < 4:
            return [(0, col_lab.shape[0])]

        grad = np.linalg.norm(np.diff(col_lab, axis=0), axis=1)
        threshold = max(
            self.p.segment_min_step,
            float(np.percentile(grad, self.p.segment_percentile)) * self.p.segment_step_frac,
        )

        # Non-maximum suppression: a single edge spans a few pixels of
        # antialiasing and must not become several boundaries.
        cuts: List[int] = []
        for start_idx, end_idx in _runs_from_mask(grad > threshold, 2):
            local = grad[start_idx:end_idx]
            cuts.append(start_idx + int(np.argmax(local)) + 1)

        bounds = [0] + [c for c in cuts if 0 < c < col_lab.shape[0]] + [col_lab.shape[0]]
        bounds = sorted(set(bounds))
        segments = [(a, b) for a, b in zip(bounds, bounds[1:]) if b > a]
        return self._merge_slivers(segments, col_lab)

    def _merge_slivers(
        self, segments: List[Tuple[int, int]], col_lab: np.ndarray
    ) -> List[Tuple[int, int]]:
        """Absorb sub-pixel-ish segments into whichever neighbour they resemble.

        The track's rounded ends are antialiased, leaving a 1-3px sliver at each
        extremity. Those slivers sit at exactly the positions used to identify
        the background, so leaving them in makes an edge artefact the reference
        colour and inverts the classification of everything else.
        """
        width = col_lab.shape[0]
        floor = max(2, int(width * self.p.min_segment_frac))
        if len(segments) < 2:
            return segments

        result = [list(seg) for seg in segments]
        changed = True
        while changed and len(result) > 1:
            changed = False
            for i, (a, b) in enumerate(result):
                if b - a >= floor:
                    continue
                colour = np.median(col_lab[a:b], axis=0)
                left = i - 1 if i > 0 else None
                right = i + 1 if i < len(result) - 1 else None
                if left is None:
                    target = right
                elif right is None:
                    target = left
                else:
                    dl = np.linalg.norm(
                        np.median(col_lab[result[left][0]:result[left][1]], axis=0) - colour
                    )
                    dr = np.linalg.norm(
                        np.median(col_lab[result[right][0]:result[right][1]], axis=0) - colour
                    )
                    target = left if dl <= dr else right
                result[target][0] = min(result[target][0], a)
                result[target][1] = max(result[target][1], b)
                result.pop(i)
                changed = True
                break
        return [tuple(seg) for seg in result]

    def _extract_bar(self, segments, start_idx: int, end_idx: int, track_w: int):
        """Recover the bar from a block that merged it with a neighbour.

        The block is a run of consecutive segments, and the bar is a contiguous
        sub-run of them. Take the sub-run whose width is closest to the width
        this rod is known to produce, breaking ties toward the bar's last known
        position, which is where it has to be if it did not teleport.
        """
        target = self._established_width * track_w
        inner = [(a, b) for a, b in segments if a >= start_idx and b <= end_idx]
        if len(inner) < 2:
            return None

        best = None
        best_cost = None
        for i in range(len(inner)):
            for j in range(i, len(inner)):
                a, b = inner[i][0], inner[j][1]
                width = b - a
                if width <= 0:
                    continue
                cost = abs(width - target) / max(target, 1e-6)
                if cost > 0.25:
                    continue
                if self._last_bar_center is not None:
                    centre = ((a + b) / 2.0) / track_w
                    cost += 0.5 * abs(centre - self._last_bar_center)
                if best_cost is None or cost < best_cost:
                    best_cost, best = cost, (a, b)
        return best

    def _identify_background(
        self, segments: Sequence[Tuple[int, int]], colours: Sequence[np.ndarray]
    ) -> Optional[int]:
        """Index a representative background segment.

        The control bar sits *inside* the track, so it splits the background
        into a piece on either side. The background is therefore the colour that
        appears at both ends of the track — which holds whether the bar occupies
        5% of the track or 67% of it. Estimating the background as the average
        or median column, as an earlier version did, silently inverts as soon as
        the bar covers more than half the track.
        """
        if not segments:
            return None
        if len(segments) == 1:
            return 0

        first, last = 0, len(segments) - 1
        if np.linalg.norm(colours[first] - colours[last]) <= self.p.group_tolerance:
            return first

        # The bar is parked against one end, so only one end is background and
        # the frame alone cannot say which. Prefer whichever end matches the
        # background learned while the bar was elsewhere.
        if self._bg_lab is not None:
            d_first = np.linalg.norm(colours[first] - self._bg_lab)
            d_last = np.linalg.norm(colours[last] - self._bg_lab)
            if min(d_first, d_last) <= self.p.group_tolerance * 2:
                return first if d_first <= d_last else last

        # No memory to fall back on: take the narrower end, since the bar is the
        # element that moves and is usually the larger of the two.
        w_first = segments[first][1] - segments[first][0]
        w_last = segments[last][1] - segments[last][0]
        return first if w_first <= w_last else last

    def _pick_bar(
        self,
        segments: Sequence[Tuple[int, int]],
        colours: Sequence[np.ndarray],
        bg_lab: np.ndarray,
        track_w: int,
    ) -> Optional[Tuple[int, int]]:
        """The bar is the widest contiguous block of non-background segments."""
        is_bg = [
            bool(np.linalg.norm(c - bg_lab) <= self.p.group_tolerance) for c in colours
        ]

        blocks: List[Tuple[int, int]] = []
        current: Optional[List[int]] = None
        for (seg_start, seg_end), bg in zip(segments, is_bg):
            if bg:
                if current is not None:
                    blocks.append((current[0], current[1]))
                    current = None
            elif current is None:
                current = [seg_start, seg_end]
            else:
                current[1] = seg_end
        if current is not None:
            blocks.append((current[0], current[1]))

        best = None
        best_score = -1.0
        for start_idx, end_idx in blocks:
            width_frac = (end_idx - start_idx) / track_w
            if not (self.p.bar_min_width_frac <= width_frac <= self.p.bar_max_width_frac):
                continue

            # Once the rod's bar width is known, anything far from it is not
            # the bar. See bar_min_width_ratio for why the two sides differ.
            if self._established_width is not None:
                if width_frac < self._established_width * self.p.bar_min_width_ratio:
                    continue
                if width_frac > self._established_width * self.p.bar_max_width_ratio:
                    # Too wide means the bar has been merged with whatever sits
                    # beside it, so the bar is inside this block rather than
                    # absent. Dropping the frame outright costs ~11% of
                    # detections on tests/clips/longer.mov; the block's own
                    # segment boundaries are enough to cut the bar back out,
                    # since the thing it merged with differs in colour and so
                    # has a boundary between them.
                    narrowed = self._extract_bar(segments, start_idx, end_idx, track_w)
                    if narrowed is None:
                        continue
                    start_idx, end_idx = narrowed
                    width_frac = (end_idx - start_idx) / track_w

            # Before the width is known, prefer the widest block; after, prefer
            # the one closest to the width this rod actually produces. Scoring
            # on raw width throughout rewards exactly the merged candidate this
            # is trying to reject, since a bar fused with its neighbour is by
            # definition wider than the bar.
            if self._established_width is None:
                score = width_frac
            else:
                score = 1.0 - min(
                    1.0, abs(width_frac - self._established_width)
                    / max(self._established_width, 1e-3)
                )
            if self._bar_width_prior is not None:
                deviation = abs(width_frac - self._bar_width_prior) / max(
                    self._bar_width_prior, 1e-3
                )
                score *= max(0.15, 1.0 - deviation)
            if self._last_bar_center is not None:
                centre = ((start_idx + end_idx) / 2.0) / track_w
                score *= max(
                    0.25,
                    1.0 - self.p.continuity_weight * abs(centre - self._last_bar_center) * 4.0,
                )
            if score > best_score:
                best_score = score
                best = (start_idx, end_idx)
        return best

    def _pick_fish(
        self,
        segments: Sequence[Tuple[int, int]],
        colours: Sequence[np.ndarray],
        bg_lab: np.ndarray,
        bar: Optional[Tuple[int, int]],
        track_w: int,
        now: float,
    ) -> Tuple[Optional[float], bool]:
        """The fish is a narrow segment unlike both the background and the bar.

        It is found the same way whether it sits on bare track or on top of the
        control bar, and whether it renders lighter than its surroundings (over
        lava) or darker (the purple marker seen over stone).
        """
        bar_lab = None
        if bar is not None:
            inside = [
                c for (a, b), c in zip(segments, colours) if a >= bar[0] and b <= bar[1]
            ]
            if inside:
                widths = [
                    b - a for (a, b) in segments if a >= bar[0] and b <= bar[1]
                ]
                bar_lab = inside[int(np.argmax(widths))]

        # How far the fish could have travelled since it was last seen.
        gate: Optional[Tuple[float, float]] = None
        if (
            self._last_fish_x is not None
            and self._last_fish_time is not None
            and self._fish_reject_streak < self.p.fish_reacquire_frames
        ):
            dt = max(0.0, now - self._last_fish_time)
            reach = self.p.fish_max_speed * dt + self.p.fish_jump_slack
            gate = (self._last_fish_x - reach, self._last_fish_x + reach)

        # Anything far weaker than the fish we have been following is not the
        # fish. Lifted once the fish has been missed for a while, so a genuine
        # change in appearance can still be picked up.
        floor = self.p.fish_min_distance
        if (
            self._fish_strength is not None
            and self._fish_reject_streak < self.p.fish_reacquire_frames
        ):
            floor = max(floor, self._fish_strength * self.p.fish_relative_strength)

        best = None
        best_distance = 0.0
        best_strength = 0.0
        for (seg_start, seg_end), colour in zip(segments, colours):
            width_frac = (seg_end - seg_start) / track_w
            if width_frac > self.p.fish_max_width_frac:
                continue
            distance = float(np.linalg.norm(colour - bg_lab))
            if bar_lab is not None:
                distance = min(distance, float(np.linalg.norm(colour - bar_lab)))
            if distance < floor:
                continue

            centre = ((seg_start + seg_end) / 2.0) / track_w
            lo, hi = self.p.fish_track_bounds
            if not (lo <= centre <= hi):
                continue
            if gate is not None and not (gate[0] <= centre <= gate[1]):
                continue

            score = distance
            if self._last_fish_x is not None:
                score *= max(0.2, 1.0 - abs(centre - self._last_fish_x) * 2.0)
            if score > best_distance:
                best_distance = score
                best_strength = distance
                best = centre

        if best is None:
            # Nothing survived the gate. Report no fish rather than accepting a
            # candidate known to be implausible; after a few such frames the
            # gate lifts so a fish that genuinely teleported can be re-acquired.
            self._fish_reject_streak += 1
            return None, False

        self._fish_reject_streak = 0
        self._last_fish_time = now
        a = self.p.fish_strength_memory
        self._fish_strength = (
            best_strength
            if self._fish_strength is None
            else self._fish_strength * (1.0 - a) + best_strength * a
        )
        inside_bar = bar is not None and (bar[0] / track_w) <= best <= (bar[1] / track_w)
        return best, inside_bar

    # -- main entry point -------------------------------------------------

    def read(
        self,
        frame: np.ndarray,
        pre_located: bool = False,
        now: Optional[float] = None,
    ) -> ReelReading:
        """Read one BGR ROI frame containing the reel track.

        Pass ``pre_located=True`` when the frame already *is* the track, as
        produced by :func:`track_locator.locate_track`. Two things then change:

        * The horizontal span is taken as the full frame width rather than
          re-derived. Normalised positions are relative to that span, so a span
          that shifts between frames makes a motionless bar appear to move and
          the controller brakes against velocity that is not there.
        * The track rows are taken as the middle of the frame instead of being
          found by darkness. The track is only dark against some backdrops —
          over a stone wall it reads brighter than its surroundings — so the
          darkness test is not something to depend on when the band is already
          known.
        """
        if frame is None or frame.size == 0 or frame.shape[1] < 16:
            return ReelReading(notes="empty frame")

        now = time.time() if now is None else now

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        if pre_located:
            h = frame.shape[0]
            margin = max(1, int(h * 0.18))
            y0, y1 = margin, max(margin + 1, h - margin)
            x0, x1 = 0, frame.shape[1]
            band_found = True
        else:
            y0, y1, band_found = self._find_track_rows(hsv)
            x0, x1 = self._find_track_span(hsv, y0, y1)
        track_w = x1 - x0
        if track_w < 16:
            return ReelReading(notes="track span too narrow")

        col_lab = self._column_lab(frame, y0, y1, x0, x1)
        segments = self._segment(col_lab)
        colours = [np.median(col_lab[a:b], axis=0) for a, b in segments]

        reading = ReelReading(
            track_x0=x0, track_x1=x1, track_y0=y0, track_y1=y1,
            runs=list(segments),
            notes="" if band_found else "track band not isolated; using ROI middle",
        )

        bg_index = self._identify_background(segments, colours)
        if bg_index is None:
            reading.notes = (reading.notes + "; no segments").lstrip("; ")
            return reading

        bg_lab = colours[bg_index]
        reading.bg_lab = bg_lab

        bar = self._pick_bar(segments, colours, bg_lab, track_w)
        fish_x, inside = self._pick_fish(segments, colours, bg_lab, bar, track_w, now)

        if bar is not None:
            reading.bar_left = bar[0] / track_w
            reading.bar_right = bar[1] / track_w
        reading.fish_x = fish_x
        reading.fish_inside_bar = inside

        # Geometry decides on-target. The bar's colour never enters into it.
        if (
            reading.bar_left is not None
            and reading.bar_right is not None
            and fish_x is not None
        ):
            tol = self.p.on_target_tolerance
            reading.on_target = reading.bar_left - tol <= fish_x <= reading.bar_right + tol

        reading.ok = reading.bar_left is not None and reading.fish_x is not None
        reading.confidence = self._confidence(reading, len(segments))

        if reading.ok:
            self._remember(reading, bg_lab)
        return reading

    # -- bookkeeping -------------------------------------------------------

    def _confidence(self, reading: ReelReading, n_segments: int) -> float:
        if reading.bar_left is None:
            return 0.0
        # A clean read is a handful of segments: background, bar, fish. Many
        # segments means the track is being cut up by texture or VFX.
        score = 0.5 if n_segments <= 6 else max(0.1, 0.5 - 0.05 * (n_segments - 6))
        if reading.fish_x is not None:
            score += 0.35 if not reading.fish_inside_bar else 0.25
        if self._bar_width_prior is not None and reading.bar_width is not None:
            deviation = abs(reading.bar_width - self._bar_width_prior)
            score += 0.15 * max(0.0, 1.0 - deviation / max(self._bar_width_prior, 1e-3))
        return float(np.clip(score, 0.0, 1.0))

    def _establish_width(self, width: float) -> None:
        """Settle on this rod's bar width from a run of consistent readings.

        The median is used rather than a running average: an average is dragged
        by the occasional grossly wrong reading, which is exactly what the
        constraint exists to exclude.
        """
        self._width_samples.append(width)
        if self._established_width is None and len(self._width_samples) >= self.p.bar_width_samples:
            candidate = float(np.median(self._width_samples))
            spread = float(np.percentile(self._width_samples, 75) - np.percentile(self._width_samples, 25))
            # Only commit if the samples actually agree; a fight seen through
            # heavy VFX may never settle, and a wrong commitment is worse than
            # none.
            if candidate > 0 and spread / candidate <= 0.35:
                self._established_width = candidate

    def _remember(self, reading: ReelReading, bg_lab: np.ndarray) -> None:
        self._bg_lab = (
            bg_lab if self._bg_lab is None else self._bg_lab * 0.85 + bg_lab * 0.15
        )
        if reading.bar_width is not None:
            self._establish_width(reading.bar_width)
            # Feed the running prior only from readings the established width
            # vouches for. An unfiltered average is a ratchet: a few merged,
            # over-wide readings drag the prior up, which makes the next
            # over-wide reading look reasonable, and the estimate never comes
            # back. That is the "works for a few seconds, then breaks and stays
            # broken" failure, and it is why the prior is not simply an EMA over
            # whatever was seen.
            trusted = (
                self._established_width is None
                or reading.bar_width
                <= self._established_width * self.p.bar_max_width_ratio
            )
            if trusted:
                if self._bar_width_prior is None:
                    self._bar_width_prior = reading.bar_width
                else:
                    a = self.p.bar_width_memory
                    self._bar_width_prior = (
                        self._bar_width_prior * (1.0 - a) + reading.bar_width * a
                    )
        self._last_bar_center = reading.bar_center
        self._last_fish_x = reading.fish_x
