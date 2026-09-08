#!/usr/bin/env python3
"""Judge a reel control strategy across the parameter space the game spans.

The point of a grid rather than a single number: the plant is not one system.
Bar width runs 0.20 to 0.92 across the rods in tests/clips, effective resilience
runs 0.2 to 1.2 across fish, and Progress Speed decides how long the fight lasts
and therefore how much has to go right. A strategy that wins the median case can
still lose every fast fish, and one average would hide it.

    python3 scripts/simulate_control.py              # the grid
    python3 scripts/simulate_control.py --seeds 40   # tighter estimates
    python3 scripts/simulate_control.py --detail     # per-cell breakdown
"""

import argparse
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import fisch_sim as sim  # noqa: E402
from reel_controller import ControlParams, ReelController  # noqa: E402


def switching_law(**overrides):
    """The controller in src/reel_controller.py, as production runs it."""
    params = ControlParams()
    for k, v in overrides.items():
        setattr(params, k, v)

    def factory():
        ctrl = ReelController(params)

        def decide(fish_x, bar_center, now, fight):
            return ctrl.decide(fish_x, bar_center, now=now).hold
        return decide
    return factory


def position_bangbang(dwell=0.08):
    """The rule the macro used before the switching law, kept as a floor.

    Hold whenever the fish is to the right, with a dwell timer that *skips the
    call* rather than deferring it -- so a blocked change leaves the button
    latched in its previous state. Any strategy worth shipping must beat this.
    """
    def factory():
        state = {"hold": False, "action": None, "t": -99.0}

        def decide(fish_x, bar_center, now, fight):
            want = "hold" if fish_x >= bar_center else "release"
            if not ((now - state["t"]) < dwell and want != state["action"]):
                state["hold"] = want == "hold"
                state["action"] = want
                state["t"] = now
            return state["hold"]
        return decide
    return factory


def evaluate(factory, grid, seeds):
    rows = []
    for fight in grid:
        results = [sim.simulate_fight(fight, factory(), s) for s in range(seeds)]
        rows.append((fight, results))
    return rows


def _agg(results):
    won = sum(r.won for r in results)
    n = len(results)
    times = [r.seconds for r in results if r.won]
    return {
        "win": won / n,
        "on": float(np.mean([r.on_target for r in results])),
        "t": float(np.mean(times)) if times else float("nan"),
        "rev": float(np.mean([r.reversals for r in results])),
        "timeout": sum(r.timed_out for r in results) / n,
    }


def report(name, rows, detail=False):
    overall = _agg([r for _, rs in rows for r in rs])
    print(f"\n{name}")
    print(f"  overall   win {overall['win']:6.1%}   on-target {overall['on']:6.1%}   "
          f"mean catch {overall['t']:5.1f}s   reversals {overall['rev']:5.0f}")

    print("\n  win rate by resilience (columns) x bar width (rows)")
    widths = sorted({f.bar_width for f, _ in rows})
    rs = sorted({f.resilience for f, _ in rows})
    print("        " + "".join(f"  r={r:<5.2f}" for r in rs))
    for w in widths:
        cells = []
        for r in rs:
            sel = [x for f, xs in rows if f.bar_width == w and f.resilience == r for x in xs]
            cells.append(f"  {_agg(sel)['win']:6.0%}")
        print(f"  w={w:.2f}" + "".join(cells))

    print("\n  win rate by Progress Speed")
    for p in sorted({f.progress_speed for f, _ in rows}, reverse=True):
        sel = [x for f, xs in rows if f.progress_speed == p for x in xs]
        a = _agg(sel)
        freq = next(f for f, _ in rows if f.progress_speed == p).required_on_target
        print(f"    p={p:+4.0f}%  needs on-target {freq:5.1%}   "
              f"win {a['win']:6.1%}   achieved {a['on']:6.1%}   timeouts {a['timeout']:5.1%}")

    # Most of the grid is settled: with a bar of 0.40 or wider almost anything
    # wins, and reporting one average over all of it hides every change in the
    # cells that decide whether the macro is usable. These are the contested
    # ones -- and note some are contested only in the sense that nobody can win
    # them; scripts/simulate_control.py --oracle separates those out.
    hard = [(f, rs) for f, rs in rows if _agg(rs)["win"] < 0.95]
    if hard:
        print(f"\n  contested cells ({len(hard)} of {len(rows)})")
        for fight, results in sorted(hard, key=lambda x: _agg(x[1])["win"]):
            a = _agg(results)
            print(f"    w={fight.bar_width:.2f} r={fight.resilience:.2f} "
                  f"p={fight.progress_speed:+4.0f}%   win {a['win']:5.0%}   "
                  f"on-target {a['on']:5.1%} vs {fight.required_on_target:5.1%} needed")

    if detail:
        print("\n  per-cell")
        for fight, results in rows:
            a = _agg(results)
            print(f"    w={fight.bar_width:.2f} r={fight.resilience:.2f} "
                  f"p={fight.progress_speed:+4.0f}%  win {a['win']:5.0%}  "
                  f"on {a['on']:5.1%}  rev {a['rev']:4.0f}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--detail", action="store_true")
    ap.add_argument("--oracle", action="store_true",
                    help="also run with a perfect sensor, to separate cells the "
                         "controller loses from cells nobody could win")
    args = ap.parse_args()

    grid = sim.default_grid()
    print(f"{len(grid)} fights x {args.seeds} seeds = "
          f"{len(grid) * args.seeds} runs per strategy")

    report("position bang-bang (pre-switching-law floor)",
           evaluate(position_bangbang(), grid, args.seeds), args.detail)
    report("switching law (src/reel_controller.py, current)",
           evaluate(switching_law(), grid, args.seeds), args.detail)

    if args.oracle:
        original = sim.perceive
        sim.perceive = lambda history, rng, tick: history[tick]
        try:
            report("same law with a perfect sensor (upper bound)",
                   evaluate(switching_law(), grid, args.seeds), args.detail)
        finally:
            sim.perceive = original
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
