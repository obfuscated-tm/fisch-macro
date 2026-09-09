"""
humanize.py — Break up the timings and positions the macro would otherwise
repeat exactly.

Every timing the macro uses comes out of a config value, so without help it
repeats to the microsecond: with the current profile each cast is charged for
0.6848591549295775 s, each recast waits 2.5412371134020617 s, and each shake
click lands on the pixel the detector named. A person cannot reproduce a
duration that precisely even once, and a distribution with *zero* variance is
the cheapest thing in the world to spot — in a log, in a histogram, or in a
server-side heuristic that only has to ask whether the gap between two casts
was ever different.

Two rules about where this belongs.

*   Jitter the **decisions**: how long to charge a cast, how long to wait before
    the next one, where inside the shake button to click. A person picks these
    by feel and the game does not care about a few tens of milliseconds either
    way, so scatter here costs nothing.

*   Never jitter the **reel loop**. `reel_controller` holds and releases at tick
    resolution to realise a duty cycle; noise added there is not disguise, it is
    error, and it comes straight off the win rate. That loop already emits an
    irregular, never-repeating pattern of holds, because it is chasing a fish
    that moves unpredictably — it is the one part of the macro that never needed
    help looking undecided.

None of this is a guarantee against detection. It removes an obvious signal; it
does not make the macro look like a person.
"""

from __future__ import annotations

import math
import random

_rng = random.Random()


def jitter_seconds(value: float, frac: float, floor: float = 0.0) -> float:
    """Scatter a duration by ``frac`` of itself, as one standard deviation.

    The draw is truncated at two sigma, which matters more than the shape of the
    distribution: an unbounded Gaussian will eventually return a cast held for
    nearly zero seconds or a recast that waits ten, and a macro that
    occasionally does something absurd is more conspicuous than one that is
    merely regular.

    Truncation is by resampling rather than by clamping. Clamping looks
    equivalent and is not: it piles every out-of-range draw onto the two
    boundary values, so about one cast in twenty is held for *exactly*
    ``value * (1 +/- 2 * frac)`` — a pair of durations that then recur, to the
    microsecond, forever. Reintroducing the repetition this function exists to
    remove, on 4.5% of casts, is not a rounding detail.

    Args:
        value: The configured duration, in seconds.
        frac: One sigma, as a fraction of ``value``. Zero disables the scatter.
        floor: Lower bound on the result.

    Returns:
        The scattered duration, never below ``floor``.
    """
    if frac <= 0.0 or value <= 0.0:
        return max(floor, value)
    sigma = value * frac
    limit = 2.0 * sigma
    for _ in range(16):
        delta = _rng.gauss(0.0, sigma)
        if -limit <= delta <= limit:
            break
    else:
        # Unreachable in practice (p < 1e-8), but a loop that can fall through
        # must still produce a scattered value rather than the bare config one.
        delta = _rng.uniform(-limit, limit)
    return max(floor, value + delta)


def jitter_point(x: float, y: float, radius_px: float) -> tuple[int, int]:
    """Scatter a click target around (x, y), staying within ``radius_px``.

    Gaussian rather than uniform-over-a-disc, because a person aiming at a
    button clusters near the middle of it and only occasionally clips the edge.
    Sigma is half the radius so the truncation at ``radius_px`` bites rarely.

    Args:
        x: Target x in screen pixels.
        y: Target y in screen pixels.
        radius_px: Largest offset permitted. Zero returns the point unchanged.

    Returns:
        The scattered point, rounded to whole pixels.
    """
    if radius_px <= 0.0:
        return int(x), int(y)
    sigma = radius_px / 2.0
    dx = _rng.gauss(0.0, sigma)
    dy = _rng.gauss(0.0, sigma)
    distance = math.hypot(dx, dy)
    if distance > radius_px:
        scale = radius_px / distance
        dx *= scale
        dy *= scale
    return int(round(x + dx)), int(round(y + dy))
