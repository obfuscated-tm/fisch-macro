#!/usr/bin/env python3
"""Run the reel vision over recorded frames and report what it saw.

    python3 scripts/test_detection.py                 # all of tests/frames/
    python3 scripts/test_detection.py reel_basic      # one clip subdirectory
    python3 scripts/test_detection.py --roi-from-settings

By default the track strip is located automatically in each frame, which is
also how calibration is meant to work now — see --roi-from-settings to instead
use the hand-calibrated bar_roi out of settings.json.

Writes annotated PNGs to tests/out/<clip>/ and a readings CSV alongside them.

A caveat when measuring against these clips: the recordings are of the whole
desktop, so every ROI here is placed by find_viewport, which locates the game
by its window chrome. That estimate moves between frames -- by a couple of
pixels on most clips and by 28 on tests/frames/STRUGGLE-ROD, where the
recording spans a window resize -- so a crop that is on target in one frame can
be off it in the next. Production does not go through this path at all; it
captures the Roblox window directly. So a detection rate measured here is a
lower bound, and a rate that differs between two clips may be saying something
about the recording rather than about the rod.
"""

import argparse
import csv
import json
import pathlib
import sys

import cv2

ROOT = pathlib.Path(__file__).resolve().parent.parent
# Both roots: this script imports modules bare (``reel_vision``), while
# src/detector.py imports them package-qualified (``src.reel_vision``).
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from config import ConfigManager  # noqa: E402
from detector import Detector  # noqa: E402
from reel_vision import ReelVision  # noqa: E402
from track_locator import TrackLocator  # noqa: E402
from viewport import find_viewport, roi_to_pixels  # noqa: E402


_SETTINGS = json.loads((ROOT / "settings.json").read_text())
BAR_ROI = _SETTINGS["bar_roi"]
PROGRESS_ROI = _SETTINGS["progress_roi"]

# The progress bar is what turns a clip into a measurement of the game's rules:
# its slope while on-target is the gain rate, and its slope while off-target is
# the loss rate. Reuse production's reader rather than a second implementation,
# so what the harness measures is what the macro will act on.
_PROGRESS_READER = Detector(ConfigManager(str(ROOT)))


AUTO = False


def roi_box(frame):
    """The calibrated ROI, used directly as the track strip.

    Measured against two clips with very different track appearances, the
    calibrated rectangle's interior fell inside the real track in both cases
    (track 755-800 and 768-803 against an ROI of 781-804). Refining the rows
    from the image was tried and made things worse, because the cues differ per
    biome: a flat black track has no vertical gradient in its interior at all,
    so an edge-based band collapses onto the top border.
    """
    vp = find_viewport(frame)
    return roi_to_pixels(vp, BAR_ROI)


def progress_box(frame):
    """Pixel rectangle around the progress bar, from the same calibration.

    Padded vertically exactly as production pads it, so the reader gets to
    locate the bar's outline rather than trusting the dragged rectangle.
    """
    roi = dict(PROGRESS_ROI)
    height = max(0.004, roi["y_end"] - roi["y_start"])
    roi["y_start"] = max(0.0, roi["y_start"] - height)
    roi["y_end"] = min(1.0, roi["y_end"] + height)
    return roi_to_pixels(find_viewport(frame), roi)


def read_progress(frame):
    """Progress fill, or None when there is no progress bar on screen."""
    px0, py0, px1, py1 = progress_box(frame)
    crop = frame[py0:py1, px0:px1]
    if crop.size == 0:
        return None
    return _PROGRESS_READER.detect_progress(crop)


def search_box(frame, expand: float = 3.0):
    """Where to look for the track — the same box production builds.

    Horizontal span comes straight from the calibrated ROI; vertical range is
    deliberately generous, because a hand-dragged rectangle is routinely off by
    a dozen pixels (measured at 14px on the sample clips).
    """
    vp = find_viewport(frame)
    x0, y0, x1, y1 = roi_to_pixels(vp, BAR_ROI)
    centre = (y0 + y1) / 2.0
    half = max(24.0, (y1 - y0) * expand)
    h = frame.shape[0]
    return int(x0), max(0, int(centre - half)), int(x1), min(h, int(centre + half))


def annotate(frame, box, reading):
    """Draw the detected track, bar and fish onto a copy of the full frame."""
    vis = frame.copy()
    x0, y0, x1, y1 = box
    cv2.rectangle(vis, (x0, y0), (x1, y1), (90, 90, 90), 1)

    if reading.track_x1 > reading.track_x0:
        tx0 = x0 + reading.track_x0
        tx1 = x0 + reading.track_x1
        ty0 = y0 + reading.track_y0
        ty1 = y0 + reading.track_y1
        cv2.rectangle(vis, (tx0, ty0), (tx1, ty1), (200, 200, 0), 1)
        span = tx1 - tx0

        if reading.bar_left is not None:
            bl = int(tx0 + reading.bar_left * span)
            br = int(tx0 + reading.bar_right * span)
            colour = (80, 255, 80) if reading.on_target else (80, 200, 255)
            cv2.rectangle(vis, (bl, ty0 - 6), (br, ty1 + 6), colour, 2)

        if reading.fish_x is not None:
            fx = int(tx0 + reading.fish_x * span)
            cv2.line(vis, (fx, ty0 - 14), (fx, ty1 + 14), (0, 0, 255), 2)

    label = (
        f"bar={reading.bar_left:.3f}-{reading.bar_right:.3f}"
        if reading.bar_left is not None else "bar=MISS"
    )
    label += f"  fish={reading.fish_x:.3f}" if reading.fish_x is not None else "  fish=MISS"
    label += f"  on={int(reading.on_target)}  conf={reading.confidence:.2f}"
    if reading.fish_inside_bar:
        label += "  [fish-in-bar]"
    if reading.notes:
        label += f"  ({reading.notes})"

    cv2.rectangle(vis, (x0, max(0, y0 - 26)), (x0 + 9 * len(label), max(0, y0 - 6)), (0, 0, 0), -1)
    cv2.putText(vis, label, (x0 + 4, max(12, y0 - 11)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return vis


def run_dir(frames_dir: pathlib.Path, out_dir: pathlib.Path, use_settings: bool,
            fps: float = 8.0) -> dict:
    images = sorted(
        p for p in frames_dir.iterdir()
        if p.suffix.lower() in {".png", ".jpg", ".jpeg"}
    )
    if not images:
        return {}

    out_dir.mkdir(parents=True, exist_ok=True)
    vision = ReelVision()
    # Mirror production: the track box accumulates across frames and then
    # defines the coordinate frame, rather than being re-derived per frame.
    locator = TrackLocator()
    rows = []
    prev_fish = None
    jumps = 0

    for frame_index, path in enumerate(images):
        frame = cv2.imread(str(path))
        if frame is None:
            continue

        # Progress is read from its own ROI and is independent of whether the
        # track was located, so it stays valid across a frame the bar detector
        # gives up on — which is exactly when the rate model has to carry the
        # state forward.
        t = frame_index / fps
        progress = read_progress(frame)

        box = locator.update(frame, search_box(frame)) if AUTO else roi_box(frame)
        if box is None:
            rows.append({"frame": path.name, "t": round(t, 4), "bar_left": "",
                         "bar_right": "", "fish_x": "", "on_target": "",
                         "progress": "" if progress is None else round(progress, 4), "confidence": 0.0,
                         "notes": "track not located"})
            continue

        x0, y0, x1, y1 = box
        # Frames are sampled at a fixed rate, so game time advances by 1/fps
        # between them. Wall-clock would be milliseconds apart and would
        # make the fish motion gate far tighter than it is in a live run.
        reading = vision.read(frame[y0:y1, x0:x1], pre_located=True, now=t)

        # Continuity check: between adjacent sampled frames the fish cannot
        # teleport. Big jumps mean the detector latched onto the wrong thing.
        if reading.fish_x is not None:
            if prev_fish is not None and abs(reading.fish_x - prev_fish) > 0.30:
                jumps += 1
            prev_fish = reading.fish_x

        rows.append({
            "frame": path.name,
            "t": round(t, 4),
            "bar_left": "" if reading.bar_left is None else round(reading.bar_left, 4),
            "bar_right": "" if reading.bar_right is None else round(reading.bar_right, 4),
            "fish_x": "" if reading.fish_x is None else round(reading.fish_x, 4),
            "on_target": int(reading.on_target),
            "progress": "" if progress is None else round(progress, 4),
            "confidence": round(reading.confidence, 3),
            "notes": reading.notes,
        })
        cv2.imwrite(str(out_dir / path.name), annotate(frame, box, reading))

    with (out_dir / "readings.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    found_bar = sum(1 for r in rows if r["bar_left"] != "")
    found_fish = sum(1 for r in rows if r["fish_x"] != "")
    return {
        "frames": len(rows),
        "bar": found_bar,
        "fish": found_fish,
        "on_target": sum(1 for r in rows if r["on_target"] == 1),
        "jumps": jumps,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("clip", nargs="?", default=None,
                    help="subdirectory of tests/frames to run (default: all)")
    ap.add_argument("--fps", type=float, default=8.0,
                    help="frame rate the clip was extracted at (default 8)")
    ap.add_argument("--roi-from-settings", action="store_true",
                    help="use bar_roi from settings.json instead of auto-locating")
    args = ap.parse_args()

    frames_root = ROOT / "tests" / "frames"
    out_root = ROOT / "tests" / "out"

    if args.clip:
        targets = [frames_root / args.clip]
    else:
        targets = [d for d in sorted(frames_root.iterdir()) if d.is_dir()]
        if any(p.suffix.lower() in {".png", ".jpg", ".jpeg"} for p in frames_root.iterdir()):
            targets.append(frames_root)

    if not targets:
        print("No frames found. See tests/README.md for how to record a clip.")
        return 1

    for target in targets:
        if not target.exists():
            print(f"missing: {target}")
            continue
        name = target.name if target != frames_root else "_loose"
        stats = run_dir(target, out_root / name, args.roi_from_settings, args.fps)
        if not stats:
            continue
        n = stats["frames"]
        print(
            f"{name:20} {n:4d} frames | "
            f"bar {stats['bar']:4d} ({stats['bar']/n:5.1%}) | "
            f"fish {stats['fish']:4d} ({stats['fish']/n:5.1%}) | "
            f"on-target {stats['on_target']/n:5.1%} | "
            f"discontinuities {stats['jumps']}"
        )
    print(f"\nAnnotated frames → {out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
