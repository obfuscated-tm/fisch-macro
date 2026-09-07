#!/usr/bin/env python3
"""Measure the game's documented reel constants against recorded clips.

The Fisch wiki documents the reeling minigame numerically, but its Resilience
page is flagged stale since v1.67.0 and the game is past v1.91. Everything the
macro derives from those numbers is only as good as they still are, so they get
checked against real footage before anything depends on them.

What is being tested, and why each one matters:

  progress gain rate   0.12*(1+p/100) per second while the fish is inside the
                       bar. Sets how long a fight should take.
  progress loss rate   documented as a flat 12%/s. The wiki says Progress Speed
                       changes the rate progress is *gained* and never says loss
                       scales, but never states it plainly either. If loss is
                       flat, the on-target fraction a fight needs is
                       f > 1/(2 + p/100) -- 0.5 for a neutral fish but 0.83 for
                       a slow one, which would explain why the macro wins some
                       fights and loses others with no visible difference.
                       Comparing the two slopes settles it.
  movement tempo       new movement every uniform(2r, 5.1r) s, each tween
                       lasting uniform(1.3r, 3.5r) s. Both estimate resilience.
  movement amplitude   +-40r% with r clamped to [0.8, 1.2], so it should land
                       in +-32-48% for every clip regardless of rod or fish.
  bar width            30% of the track at zero Control, +1% per +0.01 of it.
                       Not a floor: Control runs -0.295..+0.7 across rods, so
                       real widths span 0.5%..100%. A reading only says which
                       rod was equipped.
  fish bounds          the fish cannot leave [3%, 90%] of the track.

Frames are read straight from the .mov at its native rate rather than from
pre-extracted PNGs: tests/frames was extracted at 8 fps (one clip at 10), which
is too coarse to resolve a fast fish's 0.26-0.70 s tween, and a per-clip frame
rate mismatch would silently rescale every rate reported here.

    python3 scripts/measure_game_model.py                  # every clip
    python3 scripts/measure_game_model.py active-success   # one clip
    python3 scripts/measure_game_model.py --csv            # also dump per-frame
"""

import argparse
import csv
import pathlib
import sys

import cv2
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from test_detection import read_progress, roi_box  # noqa: E402
from reel_vision import ReelVision  # noqa: E402

# Wiki values under test. Named here so a failed check reads as a comparison
# against a specific documented claim rather than a bare number.
WIKI_PROGRESS_RATE = 0.12       # per second, gained inside the bar and lost outside
WIKI_AMPLITUDE = (0.32, 0.48)   # +-40r%, r clamped [0.8, 1.2]
WIKI_FISH_BOUNDS = (0.03, 0.90)
WIKI_ZERO_CONTROL_WIDTH = 0.30  # bar width at Control 0; NOT a lower bound
WIKI_TWEEN_PER_R = (1.3, 3.5)   # seconds
WIKI_INTERVAL_PER_R = 2.15      # mean seconds between movement onsets


# ---------------------------------------------------------------------------
# reading a clip
# ---------------------------------------------------------------------------

def read_clip(path: pathlib.Path) -> tuple[list[dict], float]:
    """Run the production detector over every frame of a clip."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise SystemExit(f"cannot open {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    vision = ReelVision()
    rows = []
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = idx / fps
        box = roi_box(frame)
        x0, y0, x1, y1 = box
        reading = vision.read(frame[y0:y1, x0:x1], pre_located=True, now=t)
        rows.append({
            "t": t,
            "bar_left": reading.bar_left,
            "bar_right": reading.bar_right,
            "fish_x": reading.fish_x,
            "on_target": bool(reading.on_target),
            "progress": read_progress(frame),
        })
        idx += 1
    cap.release()
    return rows, fps


# ---------------------------------------------------------------------------
# analysis
# ---------------------------------------------------------------------------

def _live(rows):
    """Frames where the reeling minigame is actually on screen.

    A clip starts with the cast and the lure, where there is no bar and no fish
    and the progress ROI shows whatever the UI happens to be doing. Measuring
    rates across that would fold UI noise into the game's constants.
    """
    return [r for r in rows if r["bar_left"] is not None and r["fish_x"] is not None]


def progress_rates(rows, min_run=12):
    """Slope of progress over each sustained on-target / off-target stretch.

    Not frame-to-frame differences. Progress is read as a pixel column out of
    ~664, and dt is one frame at 30 fps, so a single difference can only ever be
    an integer multiple of (1/664)/(1/30) = 0.045 per second. Taking a median of
    those lands on that lattice and reports 0.090 or 0.135 for everything,
    regardless of the true rate. Fitting a line across a whole stretch averages
    the quantisation out.

    Each maximal run of constant on/off state long enough to fit is regressed
    separately, and the runs are combined by their median slope so one boosted
    or mis-read stretch cannot set the answer.
    """
    runs, cur = [], []
    for r in rows:
        if cur and r["on_target"] != cur[-1]["on_target"]:
            runs.append(cur)
            cur = []
        cur.append(r)
    if cur:
        runs.append(cur)

    on, off = [], []
    for run in runs:
        if len(run) < min_run:
            continue
        t = np.array([r["t"] for r in run])
        p = np.array([r["progress"] for r in run])
        if t[-1] - t[0] < 0.3:
            continue
        slope = float(np.polyfit(t, p, 1)[0])
        (on if run[0]["on_target"] else off).append((slope, len(run)))

    def combine(xs):
        if not xs:
            return None, 0
        return float(np.median([v for v, _ in xs])), sum(n for _, n in xs)

    return {"gain": combine(on), "loss": combine(off)}


def movements(rows, enter_speed=0.10, exit_speed=0.035, min_travel=0.05,
              bridge_frames=3):
    """Split the fish track into tweens.

    The documented fish is not a random walk: it picks a destination and glides
    there over 1.3r-3.5r seconds, then holds until the next movement is rolled.
    So a movement is a run of frames travelling the same way, and the gaps
    between runs are the fish sitting still.

    Three things keep detector noise from being counted as movement, all of
    which matter at 30 fps where a single frame spans only 33 ms:

    * the position is median-filtered first, so one mislocated frame cannot
      manufacture a direction change;
    * runs separated by only a few frames are bridged, because a real tween
      briefly crossing below the speed threshold should not be split into two;
    * a run is kept only if the fish actually got somewhere, filtering jitter.

    Detection is hysteretic: a movement is recognised once the fish clearly
    exceeds ``enter_speed`` but is followed until it drops below the much lower
    ``exit_speed``. A tween accelerates and decelerates at its ends, so a single
    threshold clips both and under-reports the duration -- which matters here,
    because duration is one of the two independent estimates of resilience and
    a clipped one made it disagree with the other by a factor of three.
    """
    pts = [(r["t"], r["fish_x"]) for r in rows if r["fish_x"] is not None]
    if len(pts) < 8:
        return []
    t = np.array([p[0] for p in pts])
    x = np.array([p[1] for p in pts])

    k = 5
    if len(x) >= k:
        pad = np.pad(x, k // 2, mode="edge")
        x = np.array([np.median(pad[i:i + k]) for i in range(len(x))])

    v = np.gradient(x, t)
    speed = np.abs(v)

    runs, start = [], None
    for i, sp in enumerate(speed):
        if start is None:
            if sp >= enter_speed:
                # Walk back to where this movement actually began.
                j = i
                while j > 0 and speed[j - 1] >= exit_speed:
                    j -= 1
                start = j
        elif sp < exit_speed:
            runs.append([start, i - 1])
            start = None
    if start is not None:
        runs.append([start, len(speed) - 1])

    merged = []
    for run in runs:
        if merged and run[0] - merged[-1][1] <= bridge_frames:
            merged[-1][1] = run[1]
        else:
            merged.append(run)

    out = []
    for i0, i1 in merged:
        if abs(x[i1] - x[i0]) < min_travel:
            continue
        out.append({
            "t0": float(t[i0]), "t1": float(t[i1]),
            "duration": float(t[i1] - t[i0]),
            "amplitude": float(abs(x[i1] - x[i0])),
            "from": float(x[i0]), "to": float(x[i1]),
        })
    return out


def summarise(name, rows, fps):
    live = _live(rows)
    print(f"\n=== {name}  ({len(rows)} frames @ {fps:.1f} fps, "
          f"{len(live)} with a live minigame) ===")
    if len(live) < 10:
        print("  too few live frames to measure")
        return

    # --- progress rates -------------------------------------------------
    rates = progress_rates(live)
    gain, n_gain = rates["gain"]
    loss, n_loss = rates["loss"]
    print(f"  progress gain   {_fmt(gain)}/s   fitted over {n_gain:4d} on-target frames")
    print(f"  progress loss   {_fmt(loss)}/s   fitted over {n_loss:4d} off-target frames")
    if gain is not None and loss is not None and abs(loss) > 1e-4:
        ratio = abs(gain / loss)
        print(f"    gain/|loss| = {ratio:.2f}   "
              f"({'symmetric -> loss scales with Progress Speed' if 0.8 < ratio < 1.25 else 'asymmetric -> loss looks flat, f_req = 1/(2+p/100)'})")
        # Progress Speed implied by the gain rate, if loss is the flat baseline.
        p = (abs(gain) / WIKI_PROGRESS_RATE - 1.0) * 100.0
        print(f"    implied Progress Speed p = {p:+.0f}%  "
              f"(T = {1.2 + 6.8 / max(1 + p / 100, 1e-3):.1f}s)")

    # --- bar width ------------------------------------------------------
    widths = np.array([r["bar_right"] - r["bar_left"] for r in live])
    # The ROI clips the bar when it reaches either wall, so a low reading is
    # normal and the wide end of the distribution is the true width.
    w = float(np.percentile(widths, 90))
    print(f"  bar width       p90 {w:.3f}  (median {np.median(widths):.3f})")
    print(f"    implied rod Control = {w - WIKI_ZERO_CONTROL_WIDTH:+.2f}"
          f"   (width = 30% + Control)")

    # --- fish bounds ----------------------------------------------------
    fx = np.array([r["fish_x"] for r in live])
    lo, hi = float(fx.min()), float(fx.max())
    ok = lo >= WIKI_FISH_BOUNDS[0] - 0.03 and hi <= WIKI_FISH_BOUNDS[1] + 0.03
    print(f"  fish range      {lo:.3f} .. {hi:.3f}   "
          f"{'consistent with [0.03, 0.90]' if ok else '<-- OUTSIDE [0.03, 0.90]'}")

    # --- movement structure --------------------------------------------
    mv = movements(live)
    if len(mv) < 2:
        print("  movements       too few to characterise")
        return
    dur = np.array([m["duration"] for m in mv])
    amp = np.array([m["amplitude"] for m in mv])
    onsets = np.array([m["t0"] for m in mv])
    gaps = np.diff(onsets)

    print(f"  movements       {len(mv)} tweens")
    print(f"    duration      median {np.median(dur):.2f}s   "
          f"[{dur.min():.2f} .. {dur.max():.2f}]")
    print(f"    amplitude     median {np.median(amp):.3f}   "
          f"[{amp.min():.3f} .. {amp.max():.3f}]   "
          f"wiki says {WIKI_AMPLITUDE[0]}-{WIKI_AMPLITUDE[1]} for every fish")
    print(f"    fish speed    median {np.median(amp / np.maximum(dur, 1e-3)):.2f} track widths/s")
    if len(gaps):
        print(f"    onset gap     median {np.median(gaps):.2f}s")

    # Resilience, estimated two independent ways. Both depend on where one
    # movement is judged to end and the next to begin, and consecutive tweens
    # with little pause between them can be read either as one movement or two
    # -- which halves or doubles both numbers together. So these bound r rather
    # than pinning it; the wiki's own spread (p25 0.35, median 0.64, p95 1.20)
    # is the thing to compare against, and resolving r for a particular fish is
    # a job for the live estimator, not for offline clips at 30 fps.
    r_dur = float(np.median(dur)) / float(np.mean(WIKI_TWEEN_PER_R))
    print(f"    resilience    r ~ {r_dur:.2f} (from tween duration)", end="")
    if len(gaps):
        r_gap = float(np.median(gaps)) / WIKI_INTERVAL_PER_R
        lo, hi = min(r_dur, r_gap), max(r_dur, r_gap)
        inside = "within" if hi >= 0.35 and lo <= 1.20 else "OUTSIDE"
        print(f" .. {r_gap:.2f} (from onset gap)   {inside} the wiki's 0.35-1.20 spread")
    else:
        print()


def amplitude_model(all_moves):
    """Is the fish's step size a fraction of the track, or of where it already is?

    The wiki says the fish moves "+-40 * resilience% of the current location",
    which reads either way: a fixed 32-48% of the track, or a proportion of the
    fish's own position that shrinks toward the left wall. The two imply very
    different controllers, so it is worth settling rather than assuming.

    Within a single clip the fish spends most of its time mid-track, so
    amplitude/position is nearly a rescaling of amplitude and looks convincingly
    steady under either model. Only pooling clips spreads the positions out far
    enough to separate them.
    """
    amp = np.array([m["amplitude"] for m in all_moves])
    x0 = np.array([m["from"] for m in all_moves])
    rel = amp / np.maximum(x0, 1e-3)
    corr = float(np.corrcoef(x0, amp)[0, 1])

    print(f"\n=== amplitude model  ({len(all_moves)} movements pooled) ===")
    print(f"  as a fraction of the track     mean {amp.mean():.3f}  "
          f"CV {amp.std() / max(amp.mean(), 1e-9):.2f}")
    print(f"  as a fraction of the position  mean {rel.mean():.3f}  "
          f"CV {rel.std() / max(rel.mean(), 1e-9):.2f}")
    print(f"  correlation(position, amplitude) = {corr:+.3f}   "
          f"(~0 => track-relative; strongly positive => position-relative)")
    print("  median amplitude by starting position:")
    for lo, hi in [(0.0, 0.25), (0.25, 0.45), (0.45, 0.65), (0.65, 1.01)]:
        m = (x0 >= lo) & (x0 < hi)
        if m.sum():
            print(f"    {lo:.2f}-{hi:.2f}  n={int(m.sum()):3d}  {np.median(amp[m]):.3f}")
    print(f"\n  Observed amplitudes run about half the {WIKI_AMPLITUDE[0]}-"
          f"{WIKI_AMPLITUDE[1]} the wiki quotes, which is what the wiki itself\n"
          "  predicts: a movement 'can be cut short by the start of another', and\n"
          "  the reroll interval (2.15r) is shorter than the tween (1.3r-3.5r), so\n"
          "  most movements are interrupted. The drawn distance is 32-48%; the\n"
          "  distance the fish actually covers is what a controller has to chase.")


def _fmt(v):
    return "   n/a" if v is None else f"{v:+.4f}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("clip", nargs="?", help="clip stem in tests/clips (default: all)")
    ap.add_argument("--csv", action="store_true", help="also write per-frame CSV")
    args = ap.parse_args()

    clips_dir = ROOT / "tests" / "clips"
    if args.clip:
        clips = [clips_dir / f"{args.clip}.mov"]
    else:
        # physics-* are bar-acceleration recordings with no fish; they are
        # measured by scripts/measure_physics.py instead.
        clips = sorted(c for c in clips_dir.glob("*.mov")
                       if not c.name.startswith("physics-"))

    pooled = []
    for clip in clips:
        if not clip.exists():
            print(f"missing: {clip}")
            continue
        rows, fps = read_clip(clip)
        summarise(clip.stem, rows, fps)
        pooled += movements(_live(rows))
        if args.csv:
            out = ROOT / "tests" / "out" / clip.stem
            out.mkdir(parents=True, exist_ok=True)
            with (out / "model.csv").open("w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
                w.writeheader()
                w.writerows(rows)

    if len(pooled) >= 10:
        amplitude_model(pooled)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
