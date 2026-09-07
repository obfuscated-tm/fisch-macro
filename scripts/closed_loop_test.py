#!/usr/bin/env python3
"""End-to-end check: render → see → decide → act, with nothing stubbed out.

The bar physics are simulated, but everything between them is the real code:
the frame is *rendered as pixels*, ReelVision reads it back with no knowledge
of the true state, and ReelController drives from that reading alone. If the
vision mislocates the bar or the controller latches, the loop diverges and the
on-target number collapses — which is exactly the failure being fixed.

    python3 scripts/closed_loop_test.py [--save]
"""

import argparse
import pathlib
import random
import sys

import cv2
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from reel_controller import ControlParams, ReelController  # noqa: E402
from reel_vision import ReelVision  # noqa: E402
from simulate_control import DT, Bar, Fish, BAR_WIDTH, count_oscillations  # noqa: E402

W, H = 820, 30          # track size in pixels
FRAME_W, FRAME_H = 2000, 1131
TRACK_X, TRACK_Y = 600, 925


def render(
    bar_center: float,
    fish_x: float,
    on_target: bool,
    lava: bool,
    arrows: bool = True,
) -> np.ndarray:
    """Draw a frame that mimics the real UI.

    The left/right arrow glyphs printed inside the control bar are included
    because they are the detector's main source of confusion: they are narrow
    and differ in colour from the bar fill, so on shape alone they look like the
    fish. Leaving them out of the synthetic frames hid a real failure mode that
    only showed up on recorded footage.
    """
    frame = np.zeros((FRAME_H, FRAME_W, 3), np.uint8)
    frame[:, :] = (30, 90, 240) if lava else (150, 90, 40)   # lava or water
    frame[TRACK_Y:TRACK_Y + H, TRACK_X:TRACK_X + W] = (22, 20, 20)

    bw = int(BAR_WIDTH * W)
    bx = int(bar_center * W) - bw // 2
    bx = max(0, min(W - bw, bx))
    colour = (240, 240, 240) if on_target else (38, 28, 105)   # white / dark red
    frame[TRACK_Y + 2:TRACK_Y + H - 2, TRACK_X + bx:TRACK_X + bx + bw] = colour

    if arrows:
        # Shaded glyphs at fixed positions inside the bar, matching where the
        # false latches were measured on the sample clips (bar-relative
        # 0.11 and 0.88).
        shade = tuple(int(c * 0.72) for c in colour)
        for rel in (0.11, 0.88):
            ax = bx + int(rel * bw)
            frame[TRACK_Y + 8:TRACK_Y + H - 8,
                  TRACK_X + ax - 4:TRACK_X + ax + 4] = shade

    fx = int(fish_x * W)
    fx = max(2, min(W - 3, fx))
    frame[TRACK_Y:TRACK_Y + H, TRACK_X + fx - 2:TRACK_X + fx + 3] = (205, 150, 95)
    return frame


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--save", action="store_true", help="write frames to tests/out/closed_loop")
    ap.add_argument("--ticks", type=int, default=1200)
    args = ap.parse_args()

    out_dir = ROOT / "tests" / "out" / "closed_loop"
    if args.save:
        out_dir.mkdir(parents=True, exist_ok=True)

    totals = []
    for seed in range(6):
        bar, fish = Bar(), Fish(seed)
        vision = ReelVision()
        ctrl = ReelController(ControlParams())

        on = 0
        seen_bar = seen_fish = 0
        bar_trace, fish_trace = [], []
        bar_err = []
        fish_err = []
        rng = random.Random(seed)
        t = 0.0

        for i in range(args.ticks):
            true_on = abs(fish.pos - bar.pos) <= BAR_WIDTH / 2
            frame = render(bar.pos, fish.pos, true_on, lava=(seed % 2 == 0))

            # Mirrors the production default: the track strip is a known
            # rectangle (calibrated there, synthetic here), used directly.
            box = (TRACK_X, TRACK_Y, TRACK_X + W, TRACK_Y + H)
            reading = None
            if box is not None:
                x0, y0, x1, y1 = box
                reading = vision.read(frame[y0:y1, x0:x1], pre_located=True, now=t)

            if reading is not None and reading.bar_center is not None:
                seen_bar += 1
                # Vision reports positions relative to the detected track span,
                # which is padded; compare against truth in the same frame of
                # reference by mapping back to pixels.
                # Positions are normalised to the track span the vision
                # found *inside* the box, not to the box itself.
                span = reading.track_x1 - reading.track_x0
                bar_px = box[0] + reading.track_x0 + reading.bar_center * span
                bar_err.append(abs((bar_px - TRACK_X) / W - bar.pos))
            if reading is not None and reading.fish_x is not None:
                seen_fish += 1
                span = reading.track_x1 - reading.track_x0
                fish_px = box[0] + reading.track_x0 + reading.fish_x * span
                fish_err.append(abs((fish_px - TRACK_X) / W - fish.pos))

            if reading is not None and reading.ok:
                d = ctrl.decide(reading.fish_x, reading.bar_center, now=t)
                hold = d.hold
            else:
                hold = False   # blind: release, never coast

            if args.save and seed == 0 and i % 20 == 0:
                cv2.imwrite(str(out_dir / f"f{i:05d}.png"), frame)

            bar.step(hold)
            fish.step()
            if true_on:
                on += 1
            bar_trace.append(bar.pos)
            fish_trace.append(fish.pos)
            t += DT

        # Counting command transitions would be meaningless under duty-cycle
        # control, which alternates hold and release by design. What matters is
        # whether the bar physically wobbles beyond what the fish itself does.
        twitches = max(
            0, count_oscillations(bar_trace) - count_oscillations(fish_trace)
        )

        totals.append((
            on / args.ticks,
            seen_bar / args.ticks,
            seen_fish / args.ticks,
            float(np.mean(bar_err)) if bar_err else float("nan"),
            float(np.mean(fish_err)) if fish_err else float("nan"),
            bar.pos,
            twitches,
        ))

    on = np.mean([t[0] for t in totals])
    sb = np.mean([t[1] for t in totals])
    sf = np.mean([t[2] for t in totals])
    be = np.nanmean([t[3] for t in totals])
    fe = np.nanmean([t[4] for t in totals])
    pinned = sum(1 for t in totals if t[5] > 0.85 or t[5] < 0.15)

    print(f"{len(totals)} closed-loop runs x {args.ticks * DT:.0f}s\n")
    print(f"  bar detected      {sb:6.1%}   mean position error {be:.4f} track widths")
    print(f"  fish detected     {sf:6.1%}   mean position error {fe:.4f} track widths")
    print(f"  time on target    {on:6.1%}")
    print(f"  excess wobble     {np.mean([t[6] for t in totals]):6.1f} reversals beyond the fish's own")
    print(f"  pinned at a wall  {pinned}/{len(totals)}")
    if args.save:
        print(f"\n  sample frames → {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
