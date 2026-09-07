#!/usr/bin/env python3
"""Measure the reel bar's acceleration from a hold/release recording.

The controller's braking term is the time the bar needs to shed its velocity,
which depends on the game's acceleration and damping. Those were guesses until
now. This fits them from a clip where the bar was driven from one wall to the
other under a constant input.

    python3 scripts/measure_physics.py physics-hold physics-release
"""

import csv
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent


def load(clip: str):
    path = ROOT / "tests" / "out" / clip / "readings.csv"
    rows = list(csv.DictReader(path.open()))
    out = []
    for i, r in enumerate(rows):
        if r["bar_left"] and r["bar_right"]:
            out.append((i, (float(r["bar_left"]) + float(r["bar_right"])) / 2.0))
    return out


def sweeps(t, x, min_speed=0.15, min_len=8):
    """Contiguous stretches of consistent, non-trivial motion.

    The recordings begin with the bar parked at centre during the "Click & Hold
    Anywhere" prompt, before the physics start. A static stretch is trivially
    monotonic, so sweeps are found by speed and direction rather than by
    monotonicity alone.
    """
    v = np.gradient(x, t)
    out, cur = [], []
    sign = 0
    for i, vel in enumerate(v):
        s = np.sign(vel) if abs(vel) >= min_speed else 0
        if s != 0 and (sign == 0 or s == sign):
            cur.append(i)
            sign = s
        else:
            if len(cur) >= min_len:
                out.append((sign, np.array(cur)))
            cur, sign = ([i], s) if s != 0 else ([], 0)
    if len(cur) >= min_len:
        out.append((sign, np.array(cur)))
    return out


def analyse(clip: str, fps: float = 30.0):
    samples = load(clip)
    if len(samples) < 20:
        print(f"{clip}: too few readings ({len(samples)})")
        return []

    t = np.array([s[0] for s in samples], dtype=float) / fps
    x = np.array([s[1] for s in samples], dtype=float)

    print(f"{clip}:")
    results = []
    for sign, idx in sweeps(t, x):
        ts, xs = t[idx] - t[idx][0], x[idx]
        if len(ts) < 8:
            continue
        accel = 2.0 * np.polyfit(ts, xs, 2)[0]
        v = np.gradient(xs, ts)
        peak = float(np.max(np.abs(v)))
        stop = peak / abs(accel) if abs(accel) > 1e-6 else float("nan")
        label = "hold  (rightward)" if sign > 0 else "release (leftward)"
        print(f"  {label}  span {xs[0]:.2f}->{xs[-1]:.2f} in {ts[-1]:.2f}s | "
              f"accel {accel:+.2f} w/s^2 | peak {peak:.2f} w/s | stop {stop:.2f}s")
        if np.isfinite(stop):
            results.append((sign, abs(accel), peak, stop))
    return results


def main() -> int:
    clips = sys.argv[1:] or ["physics-hold", "physics-release"]
    all_results = []
    for c in clips:
        all_results.extend(analyse(c))
    if not all_results:
        print("\nno sweeps measured")
        return 1

    right = [r for r in all_results if r[0] > 0]
    left = [r for r in all_results if r[0] < 0]
    print()
    for name, group in (("hold / rightward", right), ("release / leftward", left)):
        if group:
            print(f"  {name:20} accel {np.mean([g[1] for g in group]):.2f} w/s^2 "
                  f"over {len(group)} sweep(s)")
    print(f"\nmeasured control_bar_accel = {np.mean([g[1] for g in all_results]):.2f} w/s^2 "
          f"(mean of {len(all_results)} sweeps)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
